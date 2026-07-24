"""单格点 VAE 异常检测 — MLP 自编码器，批次按网格划分

数据泄露红线：仅用正常训练集 | 验证集仅用于阈值 | 测试集仅用于评估

训练策略：每批处理 BS 个网格单元，每个单元取一个随机时间步的 SEQ 维向量。
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Tuple, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOCAL_PATH = Path(__file__).resolve().parent
from week5.config import REC_CFG, TRAIN_END, VAL_END, cache_path
from week5.data_loader import get_flow_1d


class MLPVAE(nn.Module):
    """MLP Variational Autoencoder for 1D sequence reconstruction."""

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


class VAETrainer:
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
        ).astype(np.float32)  # (T_train, T, N) -> need (T_train, N, SEQ)
        self.train_data = self.train_data.transpose(0, 2, 1)  # -> (T_train, N, SEQ)
        self.val_data = np.stack(
            [flow[TRAIN_END + t:TRAIN_END + t + SEQ]
             for t in range(T_val)], axis=0
        ).astype(np.float32)
        self.val_data = self.val_data.transpose(0, 2, 1)  # -> (T_val, N, SEQ)

        self.N = N
        self.T_train = T_train
        self.T_val = T_val
        print(f"[VAE] preloaded: train={self.train_data.shape}, val={self.val_data.shape}")

    def train(self):
        self._preload()

        self.model = MLPVAE(
            seq_len=REC_CFG.seq_len,
            hidden_dim=REC_CFG.hidden_dim,
            latent_dim=REC_CFG.latent_dim,
            dropout=REC_CFG.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[VAE] params={n_params:,}")

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=REC_CFG.lr,
            weight_decay=REC_CFG.weight_decay,
        )

        best_val = float("inf")
        patience = 0
        BS = REC_CFG.batch_size

        for epoch in range(1, REC_CFG.epochs + 1):
            self.model.train()

            cell_perm = np.random.permutation(self.N)
            t_losses, t_recon, t_kl = [], [], []

            n_batches = 0
            for b in range(0, self.N, BS):
                cell_idx = cell_perm[b:b + BS]
                if len(cell_idx) == 0:
                    continue
                t = np.random.randint(0, self.T_train)
                n_batches += 1

                seqs = torch.from_numpy(
                    self.train_data[t, cell_idx, :]
                ).float().to(self.device)

                opt.zero_grad()
                recon, mu, logvar = self.model(seqs)
                loss, recon_l, kl_l = vae_loss(recon, seqs, mu, logvar, REC_CFG.kl_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                t_losses.append(loss.item())
                t_recon.append(recon_l.item())
                t_kl.append(kl_l.item())

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
            print(f"[VAE] ep {epoch:2d} | loss={avg_train:.4f} "
                  f"| recon={np.mean(t_recon):.4f} | kl={np.mean(t_kl):.4f} "
                  f"| val={avg_val:.4f}")

            if avg_val < best_val:
                best_val = avg_val
                patience = 0
                self._best = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= REC_CFG.patience:
                    print(f"[VAE] early stop @ epoch {epoch}")
                    break

        self.model.load_state_dict(self._best)
        self.model.to(self.device)
        self._save_weights()

    def _save_weights(self):
        path = cache_path("vae_weights.pt")
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
        print(f"[VAE] weights -> {path}")

    def _errors(self, flow_split: np.ndarray) -> np.ndarray:
        self.model.eval()
        T, N = flow_split.shape
        SEQ = REC_CFG.seq_len
        T_out = max(0, T - SEQ)
        err = np.zeros((T, N), dtype=np.float32)

        BS = REC_CFG.batch_size
        with torch.no_grad():
            for t in range(T_out):
                seqs = torch.from_numpy(flow_split[t:t + SEQ].T).float()
                for b in range(0, N, BS):
                    cell_idx = np.arange(b, min(b + BS, N))
                    x = seqs[cell_idx].to(self.device)
                    recon = self.model.reconstruct(x)
                    mse = F.mse_loss(recon, x, reduction="none").mean(dim=1)
                    err[t + SEQ, cell_idx] = mse.cpu().numpy()

        return err

    def predict(self, split: Literal["train", "val", "test"] = "test",
                return_scores: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Must call train() first")

        flow = get_flow_1d(self.target)
        val_injected = os.path.join(
            os.path.dirname(__file__).replace("anomaly", ""),
            "data", "flow_val_injected.npy"
        )

        if split == "train":
            f = flow[:TRAIN_END]
        elif split == "val":
            f = np.load(val_injected) if os.path.exists(val_injected) else flow[TRAIN_END:VAL_END]
        elif split == "test":
            injected_path = os.path.join(
                os.path.dirname(__file__).replace("anomaly", ""),
                "data", "flow_test_injected.npy"
            )
            f = np.load(injected_path) if os.path.exists(injected_path) else flow[VAL_END:]
        else:
            raise ValueError(split)

        print(f"[VAE] predict {split}: flow={f.shape}")
        err = self._errors(f)

        # 用注入后验证集算阈值（F1 最大化）
        val_flow = np.load(val_injected) if os.path.exists(val_injected) else flow[TRAIN_END:VAL_END]
        val_err = self._errors(val_flow)
        np.save(cache_path("vae_scores_val"), val_err.astype(np.float32))

        val_labels_path = os.path.join(
            os.path.dirname(__file__).replace("anomaly", ""),
            "data", "anomaly_labels_val.npy"
        )
        if os.path.exists(val_labels_path):
            true_labels = np.load(val_labels_path)
            best_thresh = 0.5
            best_f1 = 0.0
            for thresh in np.arange(0.01, 1.0, 0.01):
                pred = (val_err >= thresh)
                tp = int(((pred == 1) & (true_labels == 1)).sum())
                fp = int(((pred == 1) & (true_labels == 0)).sum())
                fn = int(((pred == 0) & (true_labels == 1)).sum())
                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)
                f1 = 2 * p * r / (p + r + 1e-9)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            val_thresh = best_thresh
            print(f"[VAE] threshold search: best={val_thresh:.4f}, val_f1={best_f1:.4f}")
        else:
            val_thresh = float(np.percentile(val_err, REC_CFG.threshold_quantile * 100))

        mask = (err >= val_thresh).astype(bool)
        scores = np.clip(err / (val_thresh + 1e-9), 0, 1).astype(np.float32)
        print(f"[VAE] val_thresh={val_thresh:.6f}, detected={mask.sum()} / {mask.size}")
        return mask, scores


def run(target: str = "taxi_flow_total") -> Tuple[np.ndarray, np.ndarray]:
    trainer = VAETrainer(target=target)
    trainer.train()
    mask_val, scores_val = trainer.predict(split="val")
    mask, scores = trainer.predict(split="test")
    np.save(cache_path("vae_scores_val_v2"), scores_val)
    np.save(cache_path("vae_mask_val_v2"), mask_val)
    np.save(cache_path("vae_scores_test_v2"), scores)
    np.save(cache_path("vae_mask_test_v2"), mask)
    # V1 兼容
    np.save(cache_path("vae_scores_val"), scores_val)
    np.save(cache_path("vae_scores_test"), scores)
    print(f"[VAE] done: val={mask_val.sum()}, test={mask.sum()} anomalous")
    return mask, scores


if __name__ == "__main__":
    run()
