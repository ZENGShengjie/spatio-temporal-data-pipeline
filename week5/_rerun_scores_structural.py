"""rerun_scores_structural.py — 临时用结构性注入数据重算四种方法的 scores

策略：
  - monkey-patch 各 detector 的文件名查找，使其读取 _structural 后缀的文件
  - 保持 V3 训练好的权重不变，只重算前向推断
  - 输出到 cache/ 下 _structural 后缀的新 cache 文件，不覆盖 V3 cache
"""
import os, sys
from pathlib import Path

# EC2 path
_DATA = "/home/ubuntu/amazon_repo/week5/data"
_CACHE = "/home/ubuntu/amazon_repo/week5/cache"

import numpy as np
from config import DATA_DIR, CACHE_DIR, VAL_END, TEST_END, TRAIN_END, VAL_HOURS, TEST_HOURS, N_CELLS


def main():
    print("=" * 60)
    print("重算四种方法在结构性注入数据上的 scores（不重训练）")
    print("=" * 60)

    # ── 1. Statistical ────────────────────────────────────────────────────
    print("\n[1/4] Statistical method (3-sigma + IQR)")
    from anomaly.statistical import StatisticalAnomalyDetector

    # Patch the detector to use _structural injected data
    det = StatisticalAnomalyDetector("taxi_flow_total")
    det._load_data = lambda: _patched_load(det, suffix="_structural")

    # Fit on original training data (same as V3)
    from data_loader import get_flow_1d, get_time_group_labels, get_hour_labels

    flow = get_flow_1d("taxi_flow_total")
    det.flow_train = flow[:TRAIN_END]
    time_groups = get_time_group_labels()
    hour_labels = get_hour_labels()
    det.time_groups = time_groups
    det.hour_labels = hour_labels
    det.tg_train = time_groups[:TRAIN_END]

    # Use structural injected data for val/test
    det.flow_val   = np.load(os.path.join(_DATA, "flow_val_injected_structural.npy"))
    det.flow_test  = np.load(os.path.join(_DATA, "flow_test_injected_structural.npy"))
    det.tg_val     = time_groups[TRAIN_END:VAL_END]
    det.tg_test    = time_groups[VAL_END:]

    det._compute_group_stats()
    det._fitted = True

    # Score
    stat_val = det.score(det.flow_val)
    stat_test = det.score(det.flow_test)
    np.save(os.path.join(_CACHE, "stat_scores_val_structural.npy"), stat_val)
    np.save(os.path.join(_CACHE, "stat_scores_test_structural.npy"), stat_test)
    print(f"  stat_val  shape={stat_val.shape} mean={stat_val.mean():.4f}")
    print(f"  stat_test shape={stat_test.shape} mean={stat_test.mean():.4f}")

    # ── 2. Prediction (STF) ───────────────────────────────────────────────
    print("\n[2/4] Prediction method (STF)")
    from anomaly.prediction import PredictionAnomalyDetector

    pred_det = PredictionAnomalyDetector("taxi_flow_total")
    pred_det._load_data = lambda: _patched_pred_load(pred_det, suffix="_structural")

    pred_det.flow_train = flow[:TRAIN_END]
    pred_det.flow_val   = np.load(os.path.join(_DATA, "flow_val_injected_structural.npy"))
    pred_det.flow_test  = np.load(os.path.join(_DATA, "flow_test_injected_structural.npy"))
    pred_det.time_groups = time_groups
    pred_det.hour_labels = hour_labels
    pred_det.tg_train = time_groups[:TRAIN_END]
    pred_det.tg_val   = time_groups[TRAIN_END:VAL_END]
    pred_det.tg_test  = time_groups[VAL_END:]

    # Use existing V3 STF model weights if available
    stf_weight = os.path.join("/home/ubuntu/amazon_repo/week4/weights", "stf_loc_only_taxi_flow_total_v4fix.pth")
    if os.path.exists(stf_weight):
        # Try to load
        try:
            import torch
            pred_det.model.load_state_dict(torch.load(stf_weight, map_location=pred_det.device, weights_only=False))
            print(f"  Loaded STF weights: {stf_weight}")
        except Exception as e:
            print(f"  STF weight load skipped: {e}")

    pred_scores = pred_det.score()
    if isinstance(pred_scores, dict):
        # score() returns dict with val/test
        np.save(os.path.join(_CACHE, "pred_scores_val_structural.npy"), pred_scores["val"])
        np.save(os.path.join(_CACHE, "pred_scores_test_structural.npy"), pred_scores["test"])
        print(f"  pred_val mean={pred_scores['val'].mean():.4f}")
        print(f"  pred_test mean={pred_scores['test'].mean():.4f}")
    else:
        print("  WARNING: prediction.score() did not return expected dict")

    # ── 3. VAE ────────────────────────────────────────────────────────────
    print("\n[3/4] VAE")
    from anomaly.vae_v3 import VAEAnomalyDetector

    vae_det = VAEAnomalyDetector("taxi_flow_total")
    vae_det.flow_val   = np.load(os.path.join(_DATA, "flow_val_injected_structural.npy"))
    vae_det.flow_test  = np.load(os.path.join(_DATA, "flow_test_injected_structural.npy"))

    vae_weight = os.path.join(_CACHE, "vae_weights_v3.pt.npy")
    if os.path.exists(vae_weight):
        try:
            vae_det.model_weights = np.load(vae_weight, allow_pickle=True).item()
            print(f"  Loaded VAE weights")
        except Exception as e:
            print(f"  VAE weight load skipped: {e}")

    vae_scores = vae_det.score()
    if isinstance(vae_scores, dict):
        np.save(os.path.join(_CACHE, "vae_scores_val_structural.npy"), vae_scores["val"])
        np.save(os.path.join(_CACHE, "vae_scores_test_structural.npy"), vae_scores["test"])
        print(f"  vae_val mean={vae_scores['val'].mean():.4f}")
        print(f"  vae_test mean={vae_scores['test'].mean():.4f}")

    # ── 4. Transformer AE ─────────────────────────────────────────────────
    print("\n[4/4] Transformer AE")
    from anomaly.transformer_ae_v3 import TAEDetector

    tae_det = TAEDetector("taxi_flow_total")
    tae_det.flow_val   = np.load(os.path.join(_DATA, "flow_val_injected_structural.npy"))
    tae_det.flow_test  = np.load(os.path.join(_DATA, "flow_test_injected_structural.npy"))

    tae_weight = os.path.join(_CACHE, "tae_weights_v3.pt.npy")
    if os.path.exists(tae_weight):
        try:
            tae_det.model_weights = np.load(tae_weight, allow_pickle=True).item()
            print(f"  Loaded TAE weights")
        except Exception as e:
            print(f"  TAE weight load skipped: {e}")

    tae_scores = tae_det.score()
    if isinstance(tae_scores, dict):
        np.save(os.path.join(_CACHE, "tae_scores_val_structural.npy"), tae_scores["val"])
        np.save(os.path.join(_CACHE, "tae_scores_test_structural.npy"), tae_scores["test"])
        print(f"  tae_val mean={tae_scores['val'].mean():.4f}")
        print(f"  tae_test mean={tae_scores['test'].mean():.4f}")

    print("\n" + "=" * 60)
    print("Done. Scores saved with _structural suffix in cache/")
    print("=" * 60)


def _patched_load(self, suffix):
    pass  # placeholder, not used


def _patched_pred_load(self, suffix):
    pass  # placeholder


if __name__ == "__main__":
    main()
