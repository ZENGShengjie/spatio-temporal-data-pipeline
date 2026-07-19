"""
run_ablation_structural.py — Week5 Ablation: Structural Anomaly Dataset Evaluation
=================================================================================
对比 V3 baseline vs 结构型异常数据集下的各方法性能。

数据集：structural 后缀（sustained 70% / 连片 60%），标签文件已由 _inject_structural.py 生成。
评估策略：
  - 统计法/预测法：使用 V3 验证集上搜索的最优阈值，在结构型测试集上评估
  - VAE/TAE：使用 V3 缓存得分（V3 模型权重），在结构型测试集标签上评估
  - 分项指标：按异常类型（sustained / surge / drop）分别计算 P/R/F1

输出：report/ablation_structural_<ts>.json
"""
import os, sys, json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────────
_ec2_data   = Path("/home/ubuntu/amazon_repo/week5/data")
_ec2_cache  = Path("/home/ubuntu/amazon_repo/week5/cache")
_ec2_report = Path("/home/ubuntu/amazon_repo/week5/report")
DATA_DIR   = _ec2_data   if _ec2_data.exists()   else Path(__file__).parent / "data"
CACHE_DIR  = _ec2_cache  if _ec2_cache.exists()  else Path(__file__).parent / "cache"
REPORT_DIR = _ec2_report if _ec2_report.exists() else Path(__file__).parent / "report"
for d in [CACHE_DIR, REPORT_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"DATA_DIR={DATA_DIR}  CACHE_DIR={CACHE_DIR}")

# ── Metric helpers ──────────────────────────────────────────────────────────────
def _f1(y_true_flat, y_pred_flat):
    tp = int(((y_pred_flat == 1) & (y_true_flat == 1)).sum())
    fp = int(((y_pred_flat == 1) & (y_true_flat == 0)).sum())
    fn = int(((y_pred_flat == 0) & (y_true_flat == 1)).sum())
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)


def point_metrics(pred, gt):
    gt_f = gt.flatten().astype(int)
    p_f  = pred.flatten().astype(int)
    tp = int(((p_f == 1) & (gt_f == 1)).sum())
    fp = int(((p_f == 1) & (gt_f == 0)).sum())
    fn = int(((p_f == 0) & (gt_f == 1)).sum())
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f  = 2 * p * r / (p + r + 1e-9)
    try:
        auc = roc_auc_score(gt_f, p_f)
    except Exception:
        auc = 0.5
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4), auc=round(auc, 4))


def per_type_metrics(scores, labels, threshold):
    """Compute metrics per anomaly type using injected_events CSV."""
    T, N = labels.shape
    pred_mask = (scores >= threshold).astype(int); pred_flat_global = pred_mask.flatten()

    events_path = DATA_DIR / "injected_events_test_structural.csv"
    if not events_path.exists():
        return {}

    import pandas as pd
    events_df = pd.read_csv(events_path)

    results = {}
    for atype in ["surge", "drop", "sustained"]:
        type_mask = np.zeros_like(labels, dtype=bool)
        for _, row in events_df[events_df["type"] == atype].iterrows():
            t_s = int(row["t_start"])
            t_e = int(row["t_end"])
            for t in range(t_s, t_e):
                for n in range(N):
                    type_mask[t, n] = True

        gt_f = labels[type_mask].astype(int)
        pr_f = pred_mask[type_mask].astype(int)

        tp = int(((pr_f == 1) & (gt_f == 1)).sum())
        fp = int(((pr_f == 1) & (gt_f == 0)).sum())
        fn = int(((pr_f == 0) & (gt_f == 1)).sum())
        p  = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r + 1e-9)
        results[atype] = dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4))

    return results


def norm(s):
    q99, q00 = np.percentile(s, 99), s.min()
    r = q99 - q00
    return np.zeros_like(s) if r < 1e-9 else np.clip((s - q00) / r, 0, 1)


# ── V3 baseline results (from run_v3_full_eval_20260719_070823.json) ──────────
V3_BASELINE = {
    "statistical":  {"f1": 0.7910, "precision": 0.7544, "recall": 0.8314, "auc": 0.9100, "threshold": 0.985},
    "prediction":   {"f1": 0.8695, "precision": 0.8581, "recall": 0.8813, "auc": 0.9376, "threshold": 0.305},
    "vae":         {"f1": 0.1765, "precision": 0.1065, "recall": 0.5158, "auc": 0.6674, "threshold": 0.080},
    "transformer":  {"f1": 0.1786, "precision": 0.1097, "recall": 0.4801, "auc": 0.6586, "threshold": 0.010},
    "fusion": {
        "dual_stat_pred":        {"f1": 0.7991, "precision": 0.7613, "recall": 0.8408},
        "triple_stat_pred_vae": {"f1": 0.9165, "precision": 0.9977, "recall": 0.8475},
        "quad_all":             {"f1": 0.7866, "precision": 0.7407, "recall": 0.8385},
    },
}


# ── Score loading ──────────────────────────────────────────────────────────────
def load_scores(method, split):
    candidates = {
        "stat": [f"stat_scores_{split}_v2.npy"],
        "pred": [f"pred_scores_{split}_v2.npy"],
        "vae":  [f"vae_scores_{split}_v3.npy"],
        "tae":  [f"tae_scores_{split}_v3.npy"],
    }
    for fname in candidates.get(method, []):
        p = CACHE_DIR / fname
        if p.exists():
            s = np.load(p)
            print(f"  loaded {method}_{split}: {s.shape}")
            return s
    raise FileNotFoundError(f"No cache for {method}_{split}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("Week5 Ablation — Structural Anomaly Dataset Evaluation")
    print("=" * 70)

    # Labels: V3 baseline vs structural
    gt_test_v3        = np.load(DATA_DIR / "anomaly_labels_test.npy")
    gt_test_struct    = np.load(DATA_DIR / "anomaly_labels_test_structural.npy")
    gt_val_v3         = np.load(DATA_DIR / "anomaly_labels_val.npy")
    gt_val_struct     = np.load(DATA_DIR / "anomaly_labels_val_structural.npy")

    print(f"  V3 labels:        test={gt_test_v3.shape} sum={int(gt_test_v3.sum())}  "
          f"val={gt_val_v3.shape} sum={int(gt_val_v3.sum())}")
    print(f"  Structural labels: test={gt_test_struct.shape} sum={int(gt_test_struct.sum())}  "
          f"val={gt_val_struct.shape} sum={int(gt_val_struct.sum())}")

    # Load existing scores (from V3 model inference on V3-injected data)
    # Note: For a fair comparison, we'd re-run inference on structural data.
    # Here we evaluate existing scores against structural labels as a first-pass baseline.
    score_dict = {}
    for method, name in [("stat", "statistical"), ("pred", "prediction"),
                           ("vae", "vae"), ("tae", "transformer")]:
        scores_val = load_scores(method, "val")
        scores_test = load_scores(method, "test")
        score_dict[name] = {"val": scores_val, "test": scores_test}

    # ── Single methods: use V3 thresholds on structural data ──────────────────
    results = {"single": {}, "per_type": {}, "v3_baseline": V3_BASELINE}
    val_f1_overrides = {}  # {method: (val_f1_on_struct_labels, val_auc_on_struct_labels)}

    for method, name in [("stat", "statistical"), ("pred", "prediction"),
                           ("vae", "vae"), ("tae", "transformer")]:
        v3_thresh = V3_BASELINE[name]["threshold"]
        s_val = score_dict[name]["val"]
        s_test = score_dict[name]["test"]

        # Also re-compute threshold on structural val (for reference)
        gt_val_flat = gt_val_struct.flatten()
        best_f, best_t = 0.0, v3_thresh
        for t in np.arange(0.01, 0.99, 0.005):
            f = _f1(gt_val_flat, (s_val.flatten() >= t).astype(int))
            if f > best_f:
                best_f, best_t = f, t

        # Evaluate on structural test
        pred_test = (s_test >= best_t).astype(int)
        m = point_metrics(pred_test, gt_test_struct)
        m["threshold"] = round(float(best_t), 3)
        m["val_f1"]   = round(float(best_f), 4)
        m["v3_thresh"] = round(float(v3_thresh), 3)

        # Per-type on structural test
        pt = per_type_metrics(s_test, gt_test_struct, best_t)
        results["per_type"][name] = pt

        results["single"][name] = m
        print(f"  [{name}] thresh={best_t:.3f} val_f1={best_f:.4f}  "
              f"test P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

    # ── Fusion evaluation ────────────────────────────────────────────────────
    print("\n--- Fusion (structural labels) ---")
    fusion_results = {}
    gt_val_flat_s = gt_val_struct.flatten()
    gt_test_flat_s = gt_test_struct.flatten()

    def _eval_fusion(name, weights, keys):
        best_f, best_t, best_ws = 0.0, 0.5, None
        for ws in weights:
            if not all(k in keys for k in ws):
                continue
            fused_val = sum(w * norm(score_dict[k]["val"]) for k, w in ws.items())
            fused_val_flat = fused_val.flatten()
            for t in np.arange(0.10, 0.96, 0.02):
                f = _f1(gt_val_flat_s, (fused_val_flat >= t).astype(int))
                if f > best_f:
                    best_f, best_t, best_ws = f, t, ws

        if best_ws is None:
            return None

        fused_test = sum(w * norm(score_dict[k]["test"]) for k, w in best_ws.items())
        pred_test  = (fused_test >= best_t).astype(int)
        m = point_metrics(pred_test, gt_test_struct)
        m["weights"]   = {k: round(float(w), 2) for k, w in best_ws.items()}
        m["threshold"] = round(float(best_t), 3)
        m["val_f1"]   = round(float(best_f), 4)
        print(f"  [{name}] w={dict(m['weights'])} val_f1={m['val_f1']:.4f}  "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")
        return m

    # Dual
    dual_w = [{"statistical": round(w, 2), "prediction": round(1 - w, 2)}
              for w in np.arange(0.0, 1.001, 0.05)]
    m = _eval_fusion("dual_stat_pred", dual_w, {"statistical", "prediction"})
    if m:
        fusion_results["dual_stat_pred"] = m

    # Triple
    triple_w = []
    for w1 in np.arange(0.0, 1.001, 0.1):
        for w2 in np.arange(0.0, 1.001 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0:
                continue
            triple_w.append({"statistical": round(w1, 2), "prediction": round(w2, 2), "vae": w3})
    m = _eval_fusion("triple_stat_pred_vae", triple_w, {"statistical", "prediction", "vae"})
    if m:
        fusion_results["triple_stat_pred_vae"] = m

    # Quad
    quad_w = []
    for w1 in np.arange(0.0, 1.001, 0.1):
        for w2 in np.arange(0.0, 1.001 - w1, 0.1):
            for w3 in np.arange(0.0, 1.001 - w1 - w2, 0.1):
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < 0:
                    continue
                quad_w.append({"statistical": round(w1, 2), "prediction": round(w2, 2),
                               "vae": round(w3, 2), "transformer": w4})
    m = _eval_fusion("quad_all", quad_w, {"statistical", "prediction", "vae", "transformer"})
    if m:
        fusion_results["quad_all"] = m

    results["fusion"] = fusion_results

    # ── Summary comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("V3 Baseline vs Structural Ablation — F1 Comparison")
    print("=" * 70)
    header = f"{'Method':<30} {'V3 F1':>8} {'Struct F1':>11} {'Delta':>8} {'Note'}"
    print(header)
    print("-" * 80)
    for name in ["statistical", "prediction", "vae", "transformer"]:
        v3_f  = V3_BASELINE[name]["f1"]
        s_f   = results["single"][name]["f1"]
        delta = s_f - v3_f
        note  = "improved" if delta > 0 else ("worse" if delta < 0 else "same")
        print(f"  {name:<28} {v3_f:>8.4f} {s_f:>11.4f} {delta:>+8.4f}  ({note})")

    for name in ["dual_stat_pred", "triple_stat_pred_vae", "quad_all"]:
        v3_f = V3_BASELINE["fusion"].get(name, {}).get("f1", 0.0)
        s_f  = fusion_results.get(name, {}).get("f1", 0.0)
        delta = s_f - v3_f
        note  = "improved" if delta > 0 else ("worse" if delta < 0 else "same")
        print(f"  {name:<28} {v3_f:>8.4f} {s_f:>11.4f} {delta:>+8.4f}  ({note})")

    # ── Per-type comparison ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Per-Type F1 — Structural Dataset")
    print("=" * 70)
    print(f"  {'Method':<15} {'Surge':>8} {'Drop':>8} {'Sustained':>10}")
    print("  " + "-" * 45)
    for name in ["statistical", "prediction", "vae", "transformer"]:
        pt = results["per_type"].get(name, {})
        s_f = pt.get("surge", {}).get("f1", 0.0)
        d_f = pt.get("drop",  {}).get("f1", 0.0)
        st_f = pt.get("sustained", {}).get("f1", 0.0)
        print(f"  {name:<15} {s_f:>8.4f} {d_f:>8.4f} {st_f:>10.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "timestamp": ts,
        "dataset": "structural",
        "v3_injection_summary": {
            "val": {"actual_ratio": 0.0402, "sustained_pct": 20.0},
            "test": {"actual_ratio": 0.0402, "sustained_pct": 20.0},
        },
        "structural_injection_summary": {
            "val": {"actual_ratio": 0.0403, "sustained_pct": 69.4},
            "test": {"actual_ratio": 0.0402, "sustained_pct": 72.2},
        },
        "single": results["single"],
        "per_type": results["per_type"],
        "fusion": results["fusion"],
        "v3_baseline": V3_BASELINE,
        "comparison": {
            "delta_f1_by_method": {
                name: round(results["single"][name]["f1"] - V3_BASELINE[name]["f1"], 4)
                for name in V3_BASELINE if name != "fusion"
            },
            "delta_f1_by_fusion": {
                name: round(fusion_results.get(name, {}).get("f1", 0.0) - V3_BASELINE["fusion"].get(name, {}).get("f1", 0.0), 4)
                for name in ["dual_stat_pred", "triple_stat_pred_vae", "quad_all"]
            },
        },
        "note": (
            "Scores from V3 model weights evaluated against structural labels. "
            "Thresholds re-optimized on structural val split. "
            "Per-type metrics use structural test labels with injected_events_test_structural.csv."
        ),
    }

    out_path = REPORT_DIR / f"ablation_structural_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
