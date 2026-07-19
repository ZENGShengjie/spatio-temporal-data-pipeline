"""rerun_tae_structural.py — 对 TAE 单独计算 structural scores。

策略：
  1) 构建 TAEV3Trainer
  2) 加载 V3 权重（不重训）
  3) 复制 predict() 的核心逻辑（top-k + 归一化），但用 structural 注入数据
"""
import os, sys, shutil
from pathlib import Path

DATA_DIR = Path("/home/ubuntu/amazon_repo/week5/data")
CACHE_DIR = Path("/home/ubuntu/amazon_repo/week5/cache")

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/amazon_repo")
sys.path.insert(0, "/home/ubuntu/amazon_repo/week5")
os.chdir("/home/ubuntu/amazon_repo/week5")

from config import cache_path, TRAIN_END, VAL_END, N_CELLS, REC_CFG
from anomaly.transformer_ae_v3 import TAEV3Trainer, TemporalAttentionAE


def compute_topk(seq_errs, T_full, k, baseline_sigma, SEQ):
    T_out = max(0, T_full - SEQ)
    topk_map = np.zeros((T_full, N_CELLS), dtype=np.float32)
    if k >= SEQ:
        for t in range(T_out):
            topk_map[t + SEQ, :] = seq_errs[t].max(axis=1)
    else:
        for t in range(T_out):
            vals = seq_errs[t]  # (N, SEQ)
            partition_idx = np.argpartition(vals, -k, axis=1)[:, -k:]
            topk_vals = np.take_along_axis(vals, partition_idx, axis=1)
            topk_map[t + SEQ, :] = topk_vals.mean(axis=1)
    z_map = topk_map / baseline_sigma
    q99 = float(np.percentile(z_map, 99))
    q00 = float(z_map.min())
    r = q99 - q00
    if r < 1e-9:
        return np.zeros_like(z_map)
    out = np.clip((z_map - q00) / r, 0, 1)
    return out


def main():
    print("=" * 60)
    print("TAE: Re-score on structural injected data (V3 weights, no retrain)")
    print("=" * 60)

    # ── 加载 V3 weights ────────────────────────────────────────────
    v3_w = cache_path("tae_weights_v3.pt")
    print(f"[load] V3 weights from {v3_w}")

    trainer = TAEV3Trainer("taxi_flow_total")
    trainer._preload()
    trainer.model = TemporalAttentionAE(
        seq_len=REC_CFG.seq_len,
        d_model=REC_CFG.hidden_dim,
        num_heads=REC_CFG.num_heads,
        dropout=REC_CFG.dropout,
    ).to(trainer.device)
    ckpt = torch.load(v3_w, map_location=trainer.device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    trainer.model.load_state_dict(sd, strict=False)
    trainer.model.eval()
    print("[load] V3 weights loaded into model")

    val_flow  = np.load(str(DATA_DIR / "flow_val_injected_structural.npy"))
    test_flow = np.load(str(DATA_DIR / "flow_test_injected_structural.npy"))
    print(f"[load] val_flow={val_flow.shape}, test_flow={test_flow.shape}")

    SEQ = REC_CFG.seq_len
    val_err, val_seq_errs   = trainer._errors(val_flow)
    test_err, test_seq_errs = trainer._errors(test_flow)
    baseline_sigma = float(np.std(val_err) + 1e-6)
    print(f"[base] baseline_sigma={baseline_sigma:.6f}")

    val_scores  = compute_topk(val_seq_errs,  val_flow.shape[0],  3, baseline_sigma, SEQ)
    test_scores = compute_topk(test_seq_errs, test_flow.shape[0], 3, baseline_sigma, SEQ)

    np.save(str(CACHE_DIR / "tae_scores_val_structural.npy"),  val_scores)
    np.save(str(CACHE_DIR / "tae_scores_test_structural.npy"), test_scores)
    print(f"[save] val_scores shape={val_scores.shape},  mean={val_scores.mean():.4f}")
    print(f"[save] test_scores shape={test_scores.shape}, mean={test_scores.mean():.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
