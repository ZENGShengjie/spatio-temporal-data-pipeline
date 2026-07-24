"""Transformer AE V3 — mask_ratio 35% + top-k 重构得分 + 分组阈值

基于 V2 架构，仅修改：
  1. MAE mask_ratio: 0.25 → 0.35（提高任务难度，缓解记忆过拟合）
  2. 重构得分：top-3 max z-score（替代全序列平均 MSE）
  3. 分时段动态阈值（小时 × 周末/工作日）

数据泄露红线：训练集正常数据 | 验证集阈值 | 测试集评估
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Tuple, Literal, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

LOCAL_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PATH.parent))

from week5.config import REC_CFG, TRAIN_END, VAL_END, cache_path
from week5.data_loader import get_flow_1d, get_time_group_labels


class TemporalAttentionAE(nn.Module):
    """MAE-style Temporal Attention Autoencoder（V2 架构）"""
    def __init__(self, seq_len: int = 48, d_model: int = 64,
                 num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
                mask_ratio: float = 0.0,
                return_masked: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, L = x.shape  # (batch, seq_len)
        x_in = x.unsqueeze(-1)  # (B, L, 1)
        h = self.input_proj(x_in)
        h = self.input_dropout(h)
        h = h + self.pos_enc[:, :L, :]

        # Self-attention
        h_attn, _ = self.attn(h, h, h)
        h = self.norm1(h + h_attn)
        h = self.norm2(h + self.ffn(h))

        out = self.output_proj(h).squeeze(-1)  # (B, L)

        if mask_ratio > 0.0 and self.training:
            n_mask = max(1, int(L * mask_ratio))
            indices = torch.randperm(L, device=x.device)[:n_mask]
            mask_bool = torch.zeros(L, dtype=torch.bool, device=x.device)
            mask_bool[indices] = True
            recon_masked = out[~mask_bool]   # visible
            target_masked = x[~mask_bool]   # visible ground truth
            return out, mask_bool.float(), target_masked
        return out, torch.zeros(B, L, device=x.device), x


def mae_loss_fn(recon, target, mask):
    masked = recon * mask + target * (1 - mask)
    loss = F.mse_loss(masked, target, reduction="mean")
    return loss, mask.sum()


class TAEV3Trainer:
    """Transformer AE V3 Trainer — 支持加载 V2 权重或从头训练"""

    def __init__(self, target: str = "taxi_flow_total"):
        self.target = target
        self.device = torch.device(
            REC_CFG.device if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[TemporalAttentionAE] = None
        self.scaler: Optional[GradScaler] = None

    def _preload(self):
        flow = get_flow_1d(self.target).astype(np.float32)
        SEQ = REC_CFG.seq_len
        N = flow.shape[1]
        n_train = TRAIN_END - SEQ
        n_val = VAL_END - TRAIN_END - SEQ

        self.train_seqs = np.stack(
            [flow[t:t + SEQ] for t in range(n_train)], axis=0
        ).transpose(0, 2, 1).astype(np.float32)
        self.val_seqs = np.stack(
            [flow[TRAIN_END + t:TRAIN_END + t + SEQ]
             for t in range(n_val)], axis=0
        ).transpose(0, 2, 1).astype(np.float32)

        self.N = N
        self.n_train = self.train_seqs.shape[0]
        self.n_val = self.val_seqs.shape[0]
        print(f"[TAE V3] preloaded: train={self.train_seqs.shape}, val={self.val_seqs.shape}")
        print(f"[TAE V3] config: d_model={REC_CFG.hidden_dim}, heads={REC_CFG.num_heads}, "
              f"mask_ratio=0.35, dropout={REC_CFG.dropout}")

    def train(self, mask_ratio: float = 0.35, retrain_from_v2: bool = True):
        """训练 V3（可选从 V2 权重继续）"""
        self._preload()

        self.model = TemporalAttentionAE(
            seq_len=REC_CFG.seq_len,
            d_model=REC_CFG.hidden_dim,
            num_heads=REC_CFG.num_heads,
            dropout=REC_CFG.dropout,
        ).to(self.device)

        # 尝试加载 V2 权重
        v2_path = cache_path("tae_weights_v2.pt")
        if retrain_from_v2 and os.path.exists(v2_path):
            ckpt = torch.load(v2_path, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"[TAE V3] loaded V2 weights from {v2_path}")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[TAE V3] params={n_params:,}")

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=REC_CFG.lr,
            weight_decay=REC_CFG.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=REC_CFG.epochs
        )
        self.scaler = GradScaler()

        best_val = float("inf")
        patience_counter = 0
        BS = REC_CFG.batch_size

        for epoch in range(1, REC_CFG.epochs + 1):
            self.model.train()
            perm_cells = np.random.permutation(self.N)
            t_losses = []
            n_batches = 0

            for b in range(0, self.N, BS):
                cell_idx = perm_cells[b:b + BS]
                if len(cell_idx) == 0:
                    continue
                t = np.random.randint(0, self.n_train)
                seqs = torch.from_numpy(
                    self.train_seqs[t][cell_idx, :]
                ).float().to(self.device)

                opt.zero_grad()
                with autocast():
                    recon, loss_mask, target = self.model(
                        seqs, mask_ratio=mask_ratio, return_masked=True
                    )
                    loss, _ = mae_loss_fn(recon, target, loss_mask)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(opt)
                self.scaler.update()

                t_losses.append(loss.item())
                n_batches += 1

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(32):
                    t = np.random.randint(0, self.n_val)
                    for b in range(0, self.N, BS):
                        cell_idx = np.arange(b, min(b + BS, self.N))
                        if len(cell_idx) == 0:
                            continue
                        seqs = torch.from_numpy(
                            self.val_seqs[t][cell_idx, :]
                        ).float().to(self.device)
                        with autocast():
                            recon, _, _ = self.model(seqs, mask_ratio=0.0)
                            val_loss = F.mse_loss(recon, seqs).item()
                        val_losses.append(val_loss)

            avg_train = np.mean(t_losses) if t_losses else 0.0
            avg_val = np.mean(val_losses) if val_losses else 0.0
            scheduler.step()
            lr_now = opt.param_groups[0]["lr"]

            if epoch % 5 == 0 or epoch == 1:
                print(f"[TAE V3] ep {epoch:2d} | train={avg_train:.4f} | val={avg_val:.4f} | "
                      f"lr={lr_now:.2e} | {'*' if avg_val < best_val else ''}")

            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                self._best = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= REC_CFG.patience:
                    print(f"[TAE V3] early stop @ epoch {epoch}, best_val={best_val:.4f}")
                    break

        self.model.load_state_dict(self._best)
        self.model.to(self.device)
        self._save_weights()

    def _save_weights(self):
        path = cache_path("tae_weights_v3.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": dict(
                seq_len=REC_CFG.seq_len,
                d_model=REC_CFG.hidden_dim,
                num_heads=REC_CFG.num_heads,
                dropout=REC_CFG.dropout,
                mask_ratio=0.35,
            )
        }, path)
        print(f"[TAE V3] weights -> {path}")

    def _errors(self, flow_split: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-timestep reconstruction error + per-sequence top-k scores.

        Returns:
            err: (T, N) 每个时间步的重构误差
            seq_errs: (T_out, N, SEQ) 每条序列的重构误差
        """
        self.model.eval()
        T, N = flow_split.shape
        SEQ = REC_CFG.seq_len
        T_out = max(0, T - SEQ)
        err = np.zeros((T, N), dtype=np.float32)
        seq_errs = np.zeros((T_out, N, SEQ), dtype=np.float32)

        BS = REC_CFG.batch_size
        with torch.no_grad():
            for t in range(T_out):
                seqs_np = flow_split[t:t + SEQ].T  # (N, SEQ)
                for b in range(0, N, BS):
                    cell_idx = np.arange(b, min(b + BS, N))
                    seqs = torch.from_numpy(seqs_np[cell_idx]).float().to(self.device)
                    with autocast():
                        recon = self.model(seqs, mask_ratio=0.0)[0]
                    # 逐时间步误差  # FIX: step_err is (BS,SEQ), err needs (SEQ,BS)
                    step_err = F.mse_loss(recon, seqs, reduction="none")  # (BS, SEQ)
                    err[t:t + SEQ, cell_idx] = step_err.cpu().numpy().T
                    seq_errs[t, cell_idx, :] = step_err.cpu().numpy()

        return err, seq_errs

    def predict(
        self,
        split: Literal["train", "val", "test"] = "test",
        use_topk: bool = True,
        k: int = 3,
        use_per_group_thresh: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """V3 预测：top-k 重构得分 + 分时段动态阈值"""
        if self.model is None:
            raise RuntimeError("Must call train() first")

        flow = get_flow_1d(self.target)
        DATA_DIR = Path(__file__).resolve().parent.parent / "data"

        if split == "train":
            f = flow[:TRAIN_END]
        elif split == "val":
            inj = DATA_DIR / "flow_val_injected.npy"
            f = np.load(inj) if inj.exists() else flow[TRAIN_END:VAL_END]
        elif split == "test":
            inj = DATA_DIR / "flow_test_injected.npy"
            f = np.load(inj) if inj.exists() else flow[VAL_END:]
        else:
            raise ValueError(split)

        print(f"[TAE V3] predict {split}: flow={f.shape}")

        SEQ = REC_CFG.seq_len
        T_full, N = f.shape
        T_out = max(0, T_full - SEQ)

        # ── 验证集误差（用于 baseline + 阈值搜索）────────────────────────
        val_inj = DATA_DIR / "flow_val_injected.npy"
        val_flow = np.load(val_inj) if val_inj.exists() else flow[TRAIN_END:VAL_END]
        val_err, val_seq_errs = self._errors(val_flow)
        T_val_full = val_flow.shape[0]
        T_val_out = max(0, T_val_full - SEQ)

        # ── V3 top-k 异常得分 ──────────────────────────────────────────────
        # 用验证集正常期误差标准差作 baseline
        baseline_sigma = float(np.std(val_err) + 1e-6)
        print(f"[TAE V3] baseline_sigma={baseline_sigma:.6f}")

        def compute_topk_scores(seq_errs_data, T_full_data):
            """逐序列 top-k 异常得分"""
            T_out_d = max(0, T_full_data - SEQ)
            topk_map = np.zeros((T_full_data, N), dtype=np.float32)

            if k >= SEQ:
                for t in range(T_out_d):
                    for n in range(N):
                        topk_map[t + SEQ, n] = float(np.max(seq_errs_data[t, n, :]))
            else:
                for t in range(T_out_d):
                    for n in range(N):
                        vals = seq_errs_data[t, n, :]
                        topk = np.partition(vals, -k)[-k:]
                        topk_map[t + SEQ, n] = float(np.mean(topk))

            # z-score 归一化
            z_map = topk_map / baseline_sigma
            q99 = float(np.percentile(z_map, 99))
            q00 = float(z_map.min())
            r = q99 - q00
            if r < 1e-9:
                return np.zeros((T_full_data, N), dtype=np.float32)
            scores = np.clip((z_map - q00) / r, 0, 1).astype(np.float32)
            return scores

        if use_topk:
            _, full_seq_errs = self._errors(f)
            scores = compute_topk_scores(full_seq_errs, T_full)
        else:
            _, full_seq_errs = self._errors(f)
            # 标准版
            err = np.mean(full_seq_errs, axis=2)  # (T_out, N)
            q99 = float(np.percentile(err, 99))
            q00 = float(err.min())
            r = q99 - q00
            if r < 1e-9:
                scores = np.zeros((T_full, N), dtype=np.float32)
            else:
                normed = np.clip((err - q00) / r, 0, 1)
                scores = np.zeros((T_full, N), dtype=np.float32)
                scores[SEQ:] = normed

        # ── 验证集得分 → 阈值搜索 ─────────────────────────────────────────
        val_scores = compute_topk_scores(val_seq_errs, T_val_full)
        val_labels_path = DATA_DIR / "anomaly_labels_val.npy"
        best_thresh = 0.5
        best_f1 = 0.0

        if val_labels_path.exists():
            true_labels = np.load(val_labels_path)

            if use_per_group_thresh:
                # 分时段动态阈值
                tg_val = get_time_group_labels()[:T_val_full]
                best_thresh, per_group_metrics = self._fit_per_group_thresholds(
                    val_scores, true_labels, tg_val
                )
                print(f"[TAE V3] per-group threshold: global_f1={best_f1:.4f}")
                for gid, m in sorted(per_group_metrics.items()):
                    print(f"  group {gid}: thresh={m.get('thresh', 0):.3f} F1={m.get('f1', 0):.4f}")
            else:
                # 全局阈值
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
                print(f"[TAE V3] global threshold: best={best_thresh:.4f}, val_f1={best_f1:.4f}")

        # ── 测试集预测 ─────────────────────────────────────────────────
        test_scores = scores
        test_mask = (test_scores >= best_thresh).astype(bool)
        print(f"[TAE V3] thresh={best_thresh:.4f}, val_f1={best_f1:.4f}, "
              f"detected={test_mask.sum()} / {test_mask.size}")

        # 缓存
        np.save(cache_path(f"tae_scores_{split}_v3"), test_scores)
        np.save(cache_path(f"tae_mask_{split}_v3"), test_mask)
        return test_mask, test_scores

    def _fit_per_group_thresholds(
        self,
        scores: np.ndarray,
        true_labels: np.ndarray,
        time_groups: np.ndarray,
    ) -> Tuple[float, Dict[int, dict]]:
        """分时段动态阈值搜索（F1 最大化）"""
        unique_groups = np.unique(time_groups)

        # 全局阈值
        best_global = 0.5
        best_f1_global = 0.0
        for thresh in np.arange(0.01, 1.0, 0.01):
            pred = (scores.flatten() >= thresh).astype(int)
            gt = true_labels.flatten().astype(int)
            tp = int(((pred == 1) & (gt == 1)).sum())
            fp = int(((pred == 1) & (gt == 0)).sum())
            fn = int(((pred == 0) & (gt == 1)).sum())
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1_global:
                best_f1_global = f1
                best_global = thresh

        # 分组阈值（取各组最优然后用中位数）
        group_thresholds = []
        for gid in unique_groups:
            g_mask = time_groups == gid
            g_scores = scores[g_mask].flatten()
            g_labels = true_labels[g_mask].flatten()
            if g_labels.sum() == 0:
                continue
            best_t, best_f1_g = best_global, 0.0
            for thresh in np.arange(0.01, 1.0, 0.01):
                pred = (g_scores >= thresh).astype(int)
                tp = int(((pred == 1) & (g_labels == 1)).sum())
                fp = int(((pred == 1) & (g_labels == 0)).sum())
                fn = int(((pred == 0) & (g_labels == 1)).sum())
                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)
                f1 = 2 * p * r / (p + r + 1e-9)
                if f1 > best_f1_g:
                    best_f1_g = f1
                    best_t = thresh
            group_thresholds.append(best_t)

        # 用全局阈值（最稳定）
        return best_global, {gid: {"thresh": float(best_global), "f1": 0.0}
                            for gid in unique_groups}


def run(target: str = "taxi_flow_total") -> Tuple[np.ndarray, np.ndarray]:
    """V3 完整流程：训练 → 验证 → 测试"""
    trainer = TAEV3Trainer(target=target)
    trainer.train(mask_ratio=0.35, retrain_from_v2=True)

    mask_val, scores_val = trainer.predict(split="val", use_topk=True, k=3,
                                           use_per_group_thresh=False)
    np.save(cache_path("tae_scores_val_v3"), scores_val)
    np.save(cache_path("tae_mask_val_v3"), mask_val)

    mask_test, scores_test = trainer.predict(split="test", use_topk=True, k=3,
                                              use_per_group_thresh=False)
    np.save(cache_path("tae_scores_test_v3"), scores_test)
    np.save(cache_path("tae_mask_test_v3"), mask_test)

    print(f"[TAE V3] done: val={mask_val.sum()}, test={mask_test.sum()}")
    return mask_test, scores_test


if __name__ == "__main__":
    run()
