"""
_run_ablation_quick.py — 轻量快速对照评估
==========================================================
用 V3 模型得分（已缓存）直接对结构型标签做评估，
不再重训、不再重新生成得分，只看：F1 变化 + 分项 F1 对比。
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


_ec2_data   = Path("/home/ubuntu/amazon_repo/week5/data")
_ec2_cache  = Path("/home/ubuntu/amazon_repo/week5/cache")
_ec2_report = Path("/home/ubuntu/amazon_repo/week5/report")
DATA_DIR   = _ec2_data   if _ec2_data.exists()   else Path(__file__).parent / "data"
CACHE_DIR  = _ec2_cache  if _ec2_cache.exists()  else Path(__file__).parent / "cache"
REPORT_DIR = _ec2_report if _ec2_report.exists() else Path(__file__).parent / "report"
for d in [CACHE_DIR, REPORT_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)


def _f1(yt, yp):
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    p  = tp / (tp + fp + 1e-9) if (tp + fp) else 0.0
    r  = tp / (tp + fn + 1e-9) if (tp + fn) else 0.0
    return 2 * p * r / (p + r + 1e-9), p, r


def point_metrics(pred, gt):
    gt_f = gt.flatten().astype(int)
    pr_f = pred.flatten().astype(int)
    f, p, r = _f1(gt_f, pr_f)
    try:
        auc = float(roc_auc_score(gt_f, pr_f))
    except Exception:
        auc = 0.5
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4), auc=round(auc, 4))


def norm(s):
    q99, q00 = np.percentile(s, 99), s.min()
    r = q99 - q00
    return np.zeros_like(s) if r < 1e-9 else np.clip((s - q00) / r, 0, 1)


def best_thresh(s, gt):
    gt_f = gt.flatten()
    best_f, best_t = 0.0, 0.5
    for t in np.arange(0.01, 0.99, 0.005):
        f, _, _ = _f1(gt_f, (s.flatten() >= t).astype(int))
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def per_type_f1(scores, labels, threshold):
    """按异常类型分别计算 F1（依赖 injected_events_test_structural.csv）"""
    import pandas as pd
    events = pd.read_csv(DATA_DIR / "injected_events_test_structural.csv")
    T, N = labels.shape
    pred_flat = (scores.flatten() >= threshold).astype(int)
    label_flat = labels.flatten().astype(int)

    results = {}
    for atype in ["surge", "drop", "sustained"]:
        type_mask = np.zeros_like(labels, dtype=bool)
        for _, row in events[events["type"] == atype].iterrows():
            t_s, t_e = int(row["t_start"]), int(row["t_end"])
            for t in range(t_s, t_e):
                if 0 <= t < T:
                    type_mask[t, :] = True

        idx = type_mask.flatten()
        # 在此类型槽位上的混淆矩阵
        gt_t = label_flat[idx]
        pr_t = pred_flat[idx]
        tp = int(((pr_t == 1) & (gt_t == 1)).sum())
        fp = int(((pr_t == 1) & (gt_t == 0)).sum())
        fn = int(((pr_t == 0) & (gt_t == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f, p, r = 0.0, 0.0, 0.0
        else:
            p  = tp / (tp + fp)
            r  = tp / (tp + fn)
            f  = 2 * p * r / (p + r + 1e-9)
        n_pos = int(gt_t.sum())
        results[atype] = dict(f1=round(f, 4), precision=round(p, 4), recall=round(r, 4),
                               n_pos=n_pos)
    return results


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("Week5 Ablation — Structural Anomaly (Quick)")
    print("=" * 70)

    # 加载标签
    gt_val_struct  = np.load(DATA_DIR / "anomaly_labels_val_structural.npy")
    gt_test_struct = np.load(DATA_DIR / "anomaly_labels_test_structural.npy")
    print(f"val_struct  = {gt_val_struct.shape}  sum={int(gt_val_struct.sum())}  ratio={gt_val_struct.sum()/gt_val_struct.size:.4f}")
    print(f"test_struct = {gt_test_struct.shape}  sum={int(gt_test_struct.sum())}  ratio={gt_test_struct.sum()/gt_test_struct.size:.4f}")

    # V3 基线（来自 report/v3_full_eval_20260719_070823.json）
    v3_baseline = {
        "statistical":  {"f1": 0.7910, "precision": 0.7544, "recall": 0.8314, "auc": 0.9100, "threshold": 0.985},
        "prediction":   {"f1": 0.8695, "precision": 0.8581, "recall": 0.8813, "auc": 0.9376, "threshold": 0.305},
        "vae":         {"f1": 0.1765, "precision": 0.1065, "recall": 0.5158, "auc": 0.6674, "threshold": 0.080},
        "transformer":  {"f1": 0.1786, "precision": 0.1097, "recall": 0.4801, "auc": 0.6586, "threshold": 0.010},
        "fusion": {
            "dual_stat_pred":        {"f1": 0.7991, "precision": 0.7613, "recall": 0.8408},
            "triple_stat_pred_vae":  {"f1": 0.9165, "precision": 0.9977, "recall": 0.8475},
            "quad_all":              {"f1": 0.7866, "precision": 0.7407, "recall": 0.8385},
        },
    }

    # 加载缓存得分
    score_dict = {}
    for method, name in [("stat", "statistical"), ("pred", "prediction"),
                           ("vae", "vae"), ("tae", "transformer")]:
        sv = np.load(CACHE_DIR / f"{method}_scores_val_v3.npy"  if method in ("vae", "tae") else CACHE_DIR / f"{method}_scores_val_v2.npy")
        st = np.load(CACHE_DIR / f"{method}_scores_test_v3.npy" if method in ("vae", "tae") else CACHE_DIR / f"{method}_scores_test_v2.npy")
        score_dict[name] = {"val": sv, "test": st}
        print(f"  loaded {name}: val {sv.shape}  test {st.shape}")

    # 单方法评估
    results = {"single": {}, "per_type": {}, "v3_baseline": v3_baseline}
    print("\n--- Single Methods (re-tuned threshold on structural val) ---")
    for name in ["statistical", "prediction", "vae", "transformer"]:
        s_val  = score_dict[name]["val"]
        s_test = score_dict[name]["test"]
        best_t, best_val_f = best_thresh(s_val, gt_val_struct)
        pred_test = (s_test >= best_t).astype(int)
        m = point_metrics(pred_test, gt_test_struct)
        m["threshold"] = round(best_t, 3)
        m["val_f1"]    = round(best_val_f, 4)
        results["single"][name] = m

        # 分项 F1
        pt = per_type_f1(s_test, gt_test_struct, best_t)
        results["per_type"][name] = pt

        print(f"  [{name:<12}] t={best_t:.3f} val_f1={best_val_f:.4f}  "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

    # 融合评估
    print("\n--- Fusion (structural labels) ---")
    fusion_results = {}
    gt_val_f  = gt_val_struct.flatten()
    gt_test_f = gt_test_struct.flatten()

    def _try(name, weights):
        best_f, best_t, best_ws = 0.0, 0.5, None
        for ws in weights:
            fv = sum(w * norm(score_dict[k]["val"]) for k, w in ws.items())
            for t in np.arange(0.10, 0.96, 0.02):
                f, _, _ = _f1(gt_val_f, (fv.flatten() >= t).astype(int))
                if f > best_f:
                    best_f, best_t, best_ws = f, t, ws
        if best_ws is None:
            return None
        ft = sum(w * norm(score_dict[k]["test"]) for k, w in best_ws.items())
        pt = (ft >= best_t).astype(int)
        m  = point_metrics(pt, gt_test_struct)
        m["weights"]   = {k: round(float(w), 2) for k, w in best_ws.items()}
        m["threshold"] = round(best_t, 3)
        m["val_f1"]   = round(best_f, 4)
        print(f"  [{name:<24}] w={dict(m['weights'])} val_f1={best_f:.4f}  "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")
        return m

    # Dual
    dw = [{"statistical": round(w, 2), "prediction": round(1 - w, 2)}
          for w in np.arange(0.0, 1.001, 0.05)]
    m = _try("dual_stat_pred", dw)
    if m: fusion_results["dual_stat_pred"] = m

    # Triple
    tw = []
    for w1 in np.arange(0.0, 1.001, 0.1):
        for w2 in np.arange(0.0, 1.001 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= 0:
                tw.append({"statistical": round(w1, 2), "prediction": round(w2, 2), "vae": w3})
    m = _try("triple_stat_pred_vae", tw)
    if m: fusion_results["triple_stat_pred_vae"] = m

    # Quad
    qw = []
    for w1 in np.arange(0.0, 1.001, 0.1):
        for w2 in np.arange(0.0, 1.001 - w1, 0.1):
            for w3 in np.arange(0.0, 1.001 - w1 - w2, 0.1):
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 >= 0:
                    qw.append({"statistical": round(w1, 2), "prediction": round(w2, 2),
                               "vae": round(w3, 2), "transformer": w4})
    m = _try("quad_all", qw)
    if m: fusion_results["quad_all"] = m

    results["fusion"] = fusion_results

    # V3 vs Structural 对比表
    print("\n" + "=" * 70)
    print("V3 vs Structural F1 Comparison")
    print("=" * 70)
    print(f"  {'Method':<28} {'V3':>8} {'Struct':>9} {'Delta':>8} {'Note'}")
    print("  " + "-" * 70)
    for name in ["statistical", "prediction", "vae", "transformer"]:
        v3_f  = v3_baseline[name]["f1"]
        s_f   = results["single"][name]["f1"]
        delta = s_f - v3_f
        note  = "↑ improved" if delta > 0.005 else ("↓ worse" if delta < -0.005 else "≈ same")
        print(f"  {name:<28} {v3_f:>8.4f} {s_f:>9.4f} {delta:>+8.4f}  {note}")

    for name in ["dual_stat_pred", "triple_stat_pred_vae", "quad_all"]:
        v3_f = v3_baseline["fusion"][name]["f1"]
        s_f  = fusion_results.get(name, {}).get("f1", 0.0)
        delta = s_f - v3_f
        note  = "↑ improved" if delta > 0.005 else ("↓ worse" if delta < -0.005 else "≈ same")
        print(f"  {name:<28} {v3_f:>8.4f} {s_f:>9.4f} {delta:>+8.4f}  {note}")

    print("\n" + "=" * 70)
    print("Per-Type F1 (Structural Test Set)")
    print("=" * 70)
    print(f"  {'Method':<15} {'Surge':>8} {'Drop':>8} {'Sustained':>10}")
    print("  " + "-" * 45)
    for name in ["statistical", "prediction", "vae", "transformer"]:
        pt = results["per_type"].get(name, {})
        s_f   = pt.get("surge", {}).get("f1", 0.0)
        d_f   = pt.get("drop",  {}).get("f1", 0.0)
        st_f  = pt.get("sustained", {}).get("f1", 0.0)
        print(f"  {name:<15} {s_f:>8.4f} {d_f:>8.4f} {st_f:>10.4f}")

    # 保存结果
    output = {
        "timestamp": ts,
        "dataset": "structural",
        "note": "V3 cached scores evaluated on structural labels; threshold re-tuned on structural val split.",
        "v3_injection": {"val_ratio": 0.0402, "test_ratio": 0.0402, "sustained_pct_v3": 20.0},
        "structural_injection": {"val_ratio": 0.0403, "test_ratio": 0.0402, "sustained_pct": 72.2},
        "single": results["single"],
        "per_type": results["per_type"],
        "fusion": results["fusion"],
        "v3_baseline": v3_baseline,
        "delta_f1_by_method": {
            name: round(results["single"][name]["f1"] - v3_baseline[name]["f1"], 4)
            for name in v3_baseline if name != "fusion"
        },
        "delta_f1_by_fusion": {
            name: round(fusion_results.get(name, {}).get("f1", 0.0) - v3_baseline["fusion"].get(name, {}).get("f1", 0.0), 4)
            for name in ["dual_stat_pred", "triple_stat_pred_vae", "quad_all"]
        },
    }
    out_path = REPORT_DIR / f"ablation_structural_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
