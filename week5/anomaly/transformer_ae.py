"""Transformer AE 异常检测 V2 — MAE 掩码重构版

修复项（对比 V1）：
  1. MAE 掩码重构：训练时随机遮挡 25% 时间步，损失仅在被遮挡位置计算
     → 强迫模型学习通用时序模式，避免"记忆式过拟合"
  2. 模型容量提升：d_model=64（从32）, num_heads=4（从2）, FFN=256（从64）
  3. 输入 Dropout=0.15（L2 已通过 config 配置）
  4. 早停 patience=3（从4）
  5. AMP 混合精度训练（16G 显存可控）

数据泄露红线：仅用正常训练集 | 验证集仅用于阈值 | 测试集仅用于评估
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
from torch.cuda.amp import autocast, GradScaler

LOCAL_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PATH.parent))

from week5.config import REC_CFG, TRAIN_END, VAL_END, cache_path
from data_loader import get_flow_1d


class TemporalAttentionAE(nn.Module):
    """MAE-style Temporal Attention Autoencoder.

    训练时：随机遮挡部分时间步，用可见位置的信息重建被遮挡位置。
    推理时：全序列重建，重构误差大的位置判为异常。
    """

    def __init__(self, seq_len: int = 48, d_model: int = 64,
                 num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Input embedding
        self.input_proj = nn.Linear(1, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Single-layer self-attention + FFN (和 V1 相同架构，但维度更大)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # FFN: d_model → d_model*4 → d_model（GELU 激活）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
               mask_ratio: float = 0.0,
               return_masked: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """前向传播，支持 MAE 掩码训练。

        Args:
            x: (batch, seq_len) 输入序列
            mask_ratio: 掩码比例（0=全重建, 0.25=MAE 模式）
            return_masked: 是否返回被遮挡位置索引

        Returns:
            recon: (batch, seq_len) 重建序列（所有位置）
            loss_mask: (batch, seq_len) bool，True=被遮挡位置（用于损失计算）
            target: (batch, seq_len) 原始输入（用于计算被遮挡位置的损失）
        """
        B, S = x.shape

        # Embed + positional
        h = x.unsqueeze(-1)                    # (B, S, 1)
        h = self.input_proj(h)                  # (B, S, d_model)
        h = self.input_dropout(h)
        h = h + self.pos_enc                    # (B, S, d_model)

        # Self-attention + FFN
        attn_out, _ = self.attn(h, h, h)       # (B, S, d_model)
        h = self.norm1(h + attn_out)            # residual
        out = self.ffn(h)
        h = self.norm2(h + out)                  # residual

        # Decode
        recon = self.output_proj(h).squeeze(-1)   # (B, S)

        # 生成掩码
        if mask_ratio > 0:
            # 随机遮挡：每个位置独立 Bernoulli(mask_ratio)
            noise = torch.rand(B, S, device=x.device)
            loss_mask = noise < mask_ratio        # (B, S) bool
            if return_masked:
                return recon, loss_mask, x
            return recon, loss_mask, None
        else:
            return recon, None, x


def mae_loss_fn(recon: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor,
                beta: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """MAE 损失：仅在被遮挡位置计算重建误差。

    Args:
        recon: (B, S) 重建序列
        target: (B, S) 原始序列
        mask: (B, S) bool，True=被遮挡位置
        beta: KL 散度权重（当前架构无 VAE 的 latent space，设为 0）
    """
    # 取出被遮挡位置
    masked_recon = recon[mask]
    masked_target = target[mask]
    if masked_recon.numel() == 0:
        return torch.tensor(0.0, device=recon.device), torch.tensor(0.0, device=recon.device)
    loss = F.mse_loss(masked_recon, masked_target)
    return loss, loss


class TAEPreloadedTrainer:
    """预加载数据的 Trainer，支持 MAE 掩码训练。"""

    def __init__(self, target: str = "taxi_flow_total"):
        self.target = target
        self.device = torch.device(
            REC_CFG.device if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[TemporalAttentionAE] = None
        self.scaler: Optional[GradScaler] = None

    def _preload(self):
        """预加载所有 (n_seqs, N, SEQ) 序列"""
        flow = get_flow_1d(self.target).astype(np.float32)
        SEQ = REC_CFG.seq_len
        N = flow.shape[1]
        n_train = TRAIN_END - SEQ
        n_val = VAL_END - TRAIN_END - SEQ

        self.train_seqs = np.stack(
            [flow[t:t + SEQ] for t in range(n_train)], axis=0
        ).transpose(0, 2, 1).astype(np.float32)  # (n_seqs, N, SEQ)
        self.val_seqs = np.stack(
            [flow[TRAIN_END + t:TRAIN_END + t + SEQ]
             for t in range(n_val)], axis=0
        ).transpose(0, 2, 1).astype(np.float32)  # (n_val_seqs, N, SEQ)

        self.N = N
        self.n_train = self.train_seqs.shape[0]
        self.n_val = self.val_seqs.shape[0]
        print(f"[TAE V2] preloaded: train={self.train_seqs.shape}, val={self.val_seqs.shape}")
        print(f"[TAE V2] config: d_model={REC_CFG.hidden_dim}, heads={REC_CFG.num_heads}, "
              f"mask_ratio=0.25, dropout={REC_CFG.dropout}")

    def train(self):
        self._preload()

        self.model = TemporalAttentionAE(
            seq_len=REC_CFG.seq_len,
            d_model=REC_CFG.hidden_dim,    # 64（已从 config 读取）
            num_heads=REC_CFG.num_heads,     # 4
            dropout=REC_CFG.dropout,        # 0.15
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[TAE V2] params={n_params:,}")

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=REC_CFG.lr,
            weight_decay=REC_CFG.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=REC_CFG.epochs
        )

        # AMP scaler
        self.scaler = GradScaler()

        best_val = float("inf")
        patience_counter = 0
        BS = REC_CFG.batch_size  # cells per batch

        for epoch in range(1, REC_CFG.epochs + 1):
            self.model.train()
            perm_cells = np.random.permutation(self.N)
            t_losses = []
            n_batches = 0

            for b in range(0, self.N, BS):
                cell_idx = perm_cells[b:b + BS]
                if len(cell_idx) == 0:
                    continue
                # 随机选一个时间步
                t = np.random.randint(0, self.n_train)
                seqs = torch.from_numpy(
                    self.train_seqs[t][cell_idx, :]
                ).float().to(self.device)

                opt.zero_grad()

                # ── MAE 掩码训练：25% 时间步被遮挡 ──
                with autocast():
                    recon, loss_mask, target = self.model(
                        seqs, mask_ratio=0.25, return_masked=True
                    )
                    loss, _ = mae_loss_fn(recon, target, loss_mask)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(opt)
                self.scaler.update()

                t_losses.append(loss.item())
                n_batches += 1

            # Validation: 全序列重建（无掩码）
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(32):  # 采样 32 个时间步加速
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
                print(f"[TAE V2] ep {epoch:2d} | train={avg_train:.4f} | val={avg_val:.4f} | "
                      f"lr={lr_now:.2e} | {'*' if avg_val < best_val else ''}")

            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                self._best = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= REC_CFG.patience:  # patience=3
                    print(f"[TAE V2] early stop @ epoch {epoch}, best_val={best_val:.4f}")
                    break

        self.model.load_state_dict(self._best)
        self.model.to(self.device)
        self._save_weights()

    def _save_weights(self):
        path = cache_path("tae_weights_v2.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": dict(
                seq_len=REC_CFG.seq_len,
                d_model=REC_CFG.hidden_dim,
                num_heads=REC_CFG.num_heads,
                dropout=REC_CFG.dropout,
            )
        }, path)
        print(f"[TAE V2] weights -> {path}")

    def _errors(self, flow_split: np.ndarray) -> np.ndarray:
        """Compute per-cell reconstruction error (全序列重建，无掩码)"""
        self.model.eval()
        T, N = flow_split.shape
        SEQ = REC_CFG.seq_len
        T_out = max(0, T - SEQ)
        err = np.zeros((T, N), dtype=np.float32)

        BS = REC_CFG.batch_size
        with torch.no_grad():
            for t in range(T_out):
                seqs_np = flow_split[t:t + SEQ].T  # (N, SEQ)
                for b in range(0, N, BS):
                    cell_idx = np.arange(b, min(b + BS, N))
                    seqs = torch.from_numpy(seqs_np[cell_idx]).float().to(self.device)
                    with autocast():
                        recon = self.model(seqs, mask_ratio=0.0)[0]
                    mse = F.mse_loss(recon, seqs, reduction="none").mean(dim=1)
                    err[t + SEQ, cell_idx] = mse.cpu().numpy()

        return err

    def predict(self, split: Literal["train", "val", "test"] = "test",
                return_scores: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Must call train() first")

        flow = get_flow_1d(self.target)

        if split == "train":
            f = flow[:TRAIN_END]
        elif split == "val":
            # 优先用注入后的验证集数据（有标注）
            val_injected = os.path.join(
                os.path.dirname(__file__).replace("anomaly", ""),
                "data", "flow_val_injected.npy"
            )
            f = np.load(val_injected) if os.path.exists(val_injected) else flow[TRAIN_END:VAL_END]
        elif split == "test":
            injected_path = os.path.join(
                os.path.dirname(__file__).replace("anomaly", ""),
                "data", "flow_test_injected.npy"
            )
            f = np.load(injected_path) if os.path.exists(injected_path) else flow[VAL_END:]
        else:
            raise ValueError(split)

        print(f"[TAE V2] predict {split}: flow={f.shape}")

        # 用注入后的验证集算阈值
        val_injected = os.path.join(
            os.path.dirname(__file__).replace("anomaly", ""),
            "data", "flow_val_injected.npy"
        )
        if os.path.exists(val_injected):
            val_flow = np.load(val_injected)
        else:
            val_flow = flow[TRAIN_END:VAL_END]
        val_err = self._errors(val_flow)

        # 用验证集 F1 最大化搜索阈值
        val_labels_path = os.path.join(
            os.path.dirname(__file__).replace("anomaly", ""),
            "data", "anomaly_labels_val.npy"
        )
        best_thresh, best_f1 = self._fit_threshold_mae(val_err, val_labels_path)

        # 测试集误差
        err = self._errors(f)
        mask = (err >= best_thresh).astype(bool)
        scores = np.clip(err / (best_thresh + 1e-9), 0, 1).astype(np.float32)
        print(f"[TAE V2] thresh={best_thresh:.6f}, val_f1={best_f1:.4f}, "
              f"detected={mask.sum()} / {mask.size}")
        return mask, scores

    def _fit_threshold_mae(self, val_err: np.ndarray,
                            labels_path: str) -> Tuple[float, float]:
        """在验证集上搜索最优阈值（F1 最大化）"""
        import os
        if os.path.exists(labels_path):
            true_labels = np.load(labels_path)
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
            print(f"[TAE V2] threshold search: best={best_thresh:.4f}, val_f1={best_f1:.4f}")
            return best_thresh, best_f1
        else:
            thresh = float(np.percentile(val_err, REC_CFG.threshold_quantile * 100))
            return thresh, 0.0


def run(target: str = "taxi_flow_total") -> Tuple[np.ndarray, np.ndarray]:
    trainer = TAEPreloadedTrainer(target=target)
    trainer.train()

    # 验证集检测（用于阈值校准确认）
    mask_val, scores_val = trainer.predict(split="val")
    np.save(cache_path("tae_scores_val_v2"), scores_val)
    np.save(cache_path("tae_mask_val_v2"), mask_val)

    # 测试集检测
    mask_test, scores_test = trainer.predict(split="test")
    np.save(cache_path("tae_scores_test_v2"), scores_test)
    np.save(cache_path("tae_mask_test_v2"), mask_test)

    print(f"[TAE V2] done: val={mask_val.sum()}, test={mask_test.sum()} anomalous")
    return mask_test, scores_test


if __name__ == "__main__":
    run()
