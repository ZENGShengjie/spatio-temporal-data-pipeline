"""VAE V3 — 重构得分升级为逐序列 top-k max z-score"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOCAL_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PATH.parent))

from config import REC_CFG, TRAIN_END, VAL_END, cache_path, DATA_DIR
from data_loader import get_flow_1d


class MLPVAE(nn.Module):
    """MLP Variational Autoencoder"""
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64,
                 latent_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.enc1 = nn.Linear(seq_len, 128)
        self.enc_bn = nn.BatchNorm1d(128)
        self.enc2 = nn.Linear(128, 64)
        self.enc3_bn = nn.BatchNorm1d(64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_lv = nn.Linear(64, latent_dim)
        self.drop = nn.Dropout(dropout)
        self.dec1 = nn.Linear(latent_dim, 64)
        self.dec_bn = nn.BatchNorm1d(64)
        self.dec2 = nn.Linear(64, 128)
        self.dec3_bn = nn.BatchNorm1d(128)
        self.dec3 = nn.Linear(128, seq_len)

    def encode(self, x):
        h = F.relu(self.enc_bn(self.enc1(x)))
        h = self.drop(h)
        h = F.relu(self.enc3_bn(self.enc2(h)))
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = F.relu(self.dec_bn(self.dec1(z)))
        h = self.drop(h)
        h = F.relu(self.dec3_bn(self.dec2(h)))
        return self.dec3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)


def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def compute_vae_errors(flow_split: np.ndarray, model: nn.Module,
                       device: torch.device,
                       batch_size: int = 256,
                       SEQ: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    """计算 VAE 重构误差。

    Returns:
        err: (T, N) float，重构误差
        seq_errs: (T-SEQ, N, SEQ) float，每条序列的重构误差（用于 top-k 计算）
    """
    model.eval()
    T, N = flow_split.shape
    T_out = max(0, T - SEQ)
    err = np.zeros((T, N), dtype=np.float32)
    seq_errs = np.zeros((T_out, N, SEQ), dtype=np.float32)

    with torch.no_grad():
        for t in range(T_out):
            seqs = torch.from_numpy(flow_split[t:t + SEQ].T).float()
            for b in range(0, N, batch_size):
                cell_idx = np.arange(b, min(b + batch_size, N))
                x = seqs[cell_idx].to(device)
                recon = model.reconstruct(x)
                mse = F.mse_loss(recon, x, reduction="none").mean(dim=1)
                err[t + SEQ, cell_idx] = mse.cpu().numpy()
                # 保存逐序列误差
                seq_err = F.mse_loss(recon, x, reduction="none")  # (BS, SEQ)
                seq_errs[t, cell_idx, :] = seq_err.cpu().numpy()

    return err, seq_errs


def vae_topk_scores(seq_errs: np.ndarray, k: int = 3) -> np.ndarray:
    """逐序列取 top-k 最大误差均值作为异常得分。

    Args:
        seq_errs: (T_out, N, SEQ) 每条序列每个时间步的误差
        k: 取最大误差的时间步数量

    Returns:
        topk: (T_out, N) float，每条序列的 top-k 异常得分
    """
    T_out, N, SEQ = seq_errs.shape
    if k >= SEQ:
        topk = np.max(seq_errs, axis=2)  # (T_out, N)
    else:
        topk = np.zeros((T_out, N), dtype=np.float32)
        for n in range(N):
            for t in range(T_out):
                vals = seq_errs[t, n, :]
                topk[t, n] = float(np.mean(np.partition(vals, -k)[-k:]))
    return topk


class VAEV3Trainer:
    def __init__(self, target: str = "taxi_flow_total"):
        self.target = target
        self.device = torch.device(
            REC_CFG.device if torch.cuda.is_available() else "cpu"
        )
        self.model = None

    def _preload(self):
        flow = get_flow_1d(self.target).astype(np.float32)
        SEQ = REC_CFG.seq_len
        N = flow.shape[1]
        T_train = TRAIN_END - SEQ
        T_val = VAL_END - TRAIN_END - SEQ

        self.train_data = np.stack(
            [flow[t:t + SEQ] for t in range(T_train)], axis=0
        ).transpose(0, 2, 1).astype(np.float32)
        self.val_data = np.stack(
            [flow[TRAIN_END + t:TRAIN_END + t + SEQ]
             for t in range(T_val)], axis=0
        ).transpose(0, 2, 1).astype(np.float32)

        self.N = N
        self.T_train = T_train
        self.T_val = T_val
        print(f"[VAE V3] preloaded: train={self.train_data.shape}, val={self.val_data.shape}")

    def train(self):
        self._preload()

        self.model = MLPVAE(
            seq_len=REC_CFG.seq_len,
            hidden_dim=REC_CFG.hidden_dim,
            latent_dim=REC_CFG.latent_dim,
            dropout=REC_CFG.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[VAE V3] params={n_params:,}")

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=REC_CFG.lr,
            weight_decay=REC_CFG.weight_decay,
        )

        best_val = float("inf")
        patience_counter = 0
        BS = REC_CFG.batch_size

        for epoch in range(1, REC_CFG.epochs + 1):
            self.model.train()
            cell_perm = np.random.permutation(self.N)
            t_losses = []

            for b in range(0, self.N, BS):
                cell_idx = cell_perm[b:b + BS]
                if len(cell_idx) == 0:
                    continue
                t = np.random.randint(0, self.T_train)
                seqs = torch.from_numpy(
                    self.train_data[t, cell_idx, :]
                ).float().to(self.device)

                opt.zero_grad()
                recon, mu, logvar = self.model(seqs)
                loss, _, _ = vae_loss(recon, seqs, mu, logvar, REC_CFG.kl_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                t_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for t in range(self.T_val):
                    cell_idx = np.arange(min(BS, self.N))
                    seqs = torch.from_numpy(
                        self.val_data[t, cell_idx, :]
                    ).float().to(self.device)
                    recon, _, _ = self.model(seqs)
                    val_losses.append(F.mse_loss(recon, seqs).item())

            avg_train = np.mean(t_losses) if t_losses else 0.0
            avg_val = np.mean(val_losses) if val_losses else 0.0
            if epoch % 5 == 0 or epoch == 1:
                print(f"[VAE V3] ep {epoch:2d} | loss={avg_train:.4f} | val={avg_val:.4f}")

            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                self._best = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= REC_CFG.patience:
                    print(f"[VAE V3] early stop @ epoch {epoch}")
                    break

        self.model.load_state_dict(self._best)
        self.model.to(self.device)

        path = cache_path("vae_weights_v3.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": dict(
                seq_len=REC_CFG.seq_len,
                hidden_dim=REC_CFG.hidden_dim,
                latent_dim=REC_CFG.latent_dim,
                dropout=REC_CFG.dropout,
            )
        }, path)
        print(f"[VAE V3] weights -> {path}")

    def predict(self, split: Literal["train", "val", "test"] = "test",
                use_topk: bool = True, k: int = 3
                ) -> Tuple[np.ndarray, np.ndarray]:
        """用 V3 得分方式预测异常"""
        if self.model is None:
            raise RuntimeError("Must train first")

        flow = get_flow_1d(self.target)
        SEQ = REC_CFG.seq_len

        if split == "train":
            f = flow[:TRAIN_END]
        elif split == "val":
            val_inj = os.path.join(DATA_DIR, "flow_val_injected.npy")
            f = np.load(val_inj) if os.path.exists(val_inj) else flow[TRAIN_END:VAL_END]
        elif split == "test":
            test_inj = os.path.join(DATA_DIR, "flow_test_injected.npy")
            f = np.load(test_inj) if os.path.exists(test_inj) else flow[VAL_END:]
        else:
            raise ValueError(split)

        print(f"[VAE V3] predict {split}: flow={f.shape}")

        # 计算所有序列的重构误差
        err, seq_errs = compute_vae_errors(f, self.model, self.device,
                                            batch_size=REC_CFG.batch_size, SEQ=SEQ)

        if use_topk:
            # V3: top-k max z-score
            # 用验证集正常期计算 baseline sigma
            val_inj_path = os.path.join(DATA_DIR, "flow_val_injected.npy")
            val_flow = np.load(val_inj_path) if os.path.exists(val_inj_path) else flow[TRAIN_END:VAL_END]
            _, val_seq_errs = compute_vae_errors(
                val_flow, self.model, self.device,
                batch_size=REC_CFG.batch_size, SEQ=SEQ)

            # 用验证集正常期序列误差的均值作 baseline
            baseline_sigma = float(np.mean(val_seq_errs) + 1e-6)

            # 计算 z-score
            z = err / baseline_sigma  # (T, N)

            # 直接用 z 作为得分（z-score already captures deviation）
            topk = z

        # 重新实现：逐序列滑动窗口 top-k
        # seq_errs shape: (T_out, N, SEQ)
        # 对每个 (t, n) 位置，取该位置所在 SEQ 窗口内的 top-k
        T_full, N_full = f.shape
        T_out = max(0, T_full - SEQ)
        topk_map = np.zeros((T_full, N_full), dtype=np.float32)

        for n in range(N_full):
            for t_out in range(T_out):
                vals = seq_errs[t_out, n, :]  # (SEQ,)
                if k >= len(vals):
                    topk_map[t_out + SEQ, n] = float(np.max(vals))
                else:
                    topk = np.partition(vals, -k)[-k:]
                    topk_map[t_out + SEQ, n] = float(np.mean(topk))

        # 归一化
        q99 = float(np.percentile(topk_map, 99))
        q00 = float(topk_map.min())
        r = q99 - q00
        if r < 1e-9:
            scores = np.zeros((T_full, N_full), dtype=np.float32)
        else:
            scores = np.clip((topk_map - q00) / r, 0, 1).astype(np.float32)

        # 用验证集找阈值
        val_inj = os.path.join(DATA_DIR, "flow_val_injected.npy")
        val_flow = np.load(val_inj) if os.path.exists(val_inj) else flow[TRAIN_END:VAL_END]
        _, val_seq_errs = compute_vae_errors(
            val_flow, self.model, self.device,
            batch_size=REC_CFG.batch_size, SEQ=SEQ)

        T_val_full = val_flow.shape[0]
        T_val_out = max(0, T_val_full - SEQ)
        val_topk = np.zeros((T_val_full, N_full), dtype=np.float32)
        for n in range(N_full):
            for t_out in range(T_val_out):
                vals = val_seq_errs[t_out, n, :]
                if k >= len(vals):
                    val_topk[t_out + SEQ, n] = float(np.max(vals))
                else:
                    topk = np.partition(vals, -k)[-k:]
                    val_topk[t_out + SEQ, n] = float(np.mean(topk))

        q99_v = float(np.percentile(val_topk, 99))
        q00_v = float(val_topk.min())
        r_v = q99_v - q00_v
        if r_v < 1e-9:
            val_scores = np.zeros((T_val_full, N_full), dtype=np.float32)
        else:
            val_scores = np.clip((val_topk - q00_v) / r_v, 0, 1).astype(np.float32)

        # F1 最大化搜索阈值
        val_labels_path = os.path.join(DATA_DIR, "anomaly_labels_val.npy")
        best_thresh = 0.5
        best_f1 = 0.0
        if os.path.exists(val_labels_path):
            true_labels = np.load(val_labels_path)
            for thresh in np.arange(0.01, 1.0, 0.01):
                pred = (val_scores >= thresh)
                tp = int(((pred == 1) & (true_labels == 1)).sum())
                fp = int(((pred == 1) & (true_labels == 0)).sum())
                fn = int(((pred == 0) & (true_labels == 1)).sum())
                p = tp / (tp + fp + 1e-9)
                r_val = tp / (tp + fn + 1e-9)
                f1 = 2 * p * r_val / (p + r_val + 1e-9)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            print(f"[VAE V3] threshold search: best={best_thresh:.4f}, val_f1={best_f1:.4f}")

        mask = (scores >= best_thresh).astype(bool)
        print(f"[VAE V3] thresh={best_thresh:.4f}, detected={mask.sum()} / {mask.size}")

        # 缓存
        np.save(cache_path(f"vae_scores_{split}_v3"), scores)
        np.save(cache_path(f"vae_mask_{split}_v3"), mask)
        return mask, scores


def run(target: str = "taxi_flow_total") -> Tuple[np.ndarray, np.ndarray]:
    trainer = VAEV3Trainer(target=target)
    trainer.train()

    # 验证集（用于阈值校准）
    mask_val, scores_val = trainer.predict(split="val", use_topk=True, k=3)
    np.save(cache_path("vae_scores_val_v3"), scores_val)
    np.save(cache_path("vae_mask_val_v3"), mask_val)

    # 测试集
    mask_test, scores_test = trainer.predict(split="test", use_topk=True, k=3)
    np.save(cache_path("vae_scores_test_v3"), scores_test)
    np.save(cache_path("vae_mask_test_v3"), mask_test)

    print(f"[VAE V3] done: val={mask_val.sum()}, test={mask_test.sum()}")
    return mask_test, scores_test


if __name__ == "__main__":
    run()
