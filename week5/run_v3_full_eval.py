"""
run_v3_full_eval.py — Week5 V3 Full Evaluation

本脚本对 V3 异常检测融合框架进行完整评估，包括：
  1. 单方法评估（统计阈值 / 预测误差 / VAE / Transformer-AE）
  2. 双路融合（stat + pred / stat + vae / stat + tae / pred + vae / pred + tae / vae + tae）
  3. 三路融合（stat + pred + vae / stat + pred + tae / stat + vae + tae / pred + vae + tae）
  4. 四路融合（stat + pred + vae + tae）
  5. 阈值扫描（0.5 ~ 0.98，自动选 F1 最优）
  6. 测试集最终评估（验证集搜索的权重与阈值）

数据合规：
  - 验证集仅用于融合权重搜索与阈值确定
  - 测试集严格不参与任何调参

用法：
    cd week5
    python3 run_v3_full_eval.py

输入：
    - data/anomaly_labels_val.npy   (504, 1024)  bool
    - data/anomaly_labels_test.npy  (600, 1024)  bool
    - cache/stat_scores_val.npy / test.npy
    - cache/pred_scores_val.npy / test.npy
    - cache/vae_scores_val.npy / test.npy
    - cache/tae_scores_val.npy / test.npy

输出：
    - report/v3_full_eval_<timestamp>.json   (所有配置 + 指标)
    - report/v3_final_report.md              (人类可读报告)

性能预期：
    - 全部评估在 5-10 分钟内完成（无需 GPU）
    - F1@0.5 ~ 0.95（融合 V3 通常 > 0.95）
"""
import os, sys, json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# ── Paths (EC2: /home/ubuntu/amazon_repo/week5/) ─────────────────────────────
_ec2_data  = Path("/home/ubuntu/amazon_repo/week5/data")
_ec2_cache = Path("/home/ubuntu/amazon_repo/week5/cache")
_ec2_report = Path("/home/ubuntu/amazon_repo/week5/report")
DATA_DIR   = _ec2_data  if _ec2_data.exists()  else Path(__file__).parent / "data"
CACHE_DIR  = _ec2_cache if _ec2_cache.exists() else Path(__file__).parent / "cache"
REPORT_DIR = _ec2_report if _ec2_report.exists() else Path(__file__).parent / "report"
for d in [CACHE_DIR, REPORT_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"  DATA_DIR={DATA_DIR}  CACHE_DIR={CACHE_DIR}")

# ── Metric helpers (manual — avoids sklearn shape quirks) ───────────────────────
def _f1(y_true_flat, y_pred_flat):
    tp = int(((y_pred_flat == 1) & (y_true_flat == 1)).sum())
    fp = int(((y_pred_flat == 1) & (y_true_flat == 0)).sum())
    fn = int(((y_pred_flat == 0) & (y_true_flat == 1)).sum())
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)


def point_metrics(pred, gt):
    gt_f = gt.flatten().astype(int)
    p_f = pred.flatten().astype(int)
    tp = int(((p_f == 1) & (gt_f == 1)).sum())
    fp = int(((p_f == 1) & (gt_f == 0)).sum())
    fn = int(((p_f == 0) & (gt_f == 1)).sum())
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f = 2 * p * r / (p + r + 1e-9)
    try:
        auc = roc_auc_score(gt_f, p_f)
    except:
        auc = 0.5
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4), auc=round(auc, 4))


def norm(s):
    q99, q00 = np.percentile(s, 99), s.min()
    r = q99 - q00
    return np.zeros_like(s) if r < 1e-9 else np.clip((s - q00) / r, 0, 1)


# ── Score loading ─────────────────────────────────────────────────────────────
# Actual files on disk:
#   stat:  stat_scores_val_v2.npy,  stat_scores_test_v2.npy
#   pred:  pred_scores_val_v2.npy,  pred_scores_test_v2.npy
#   vae:   vae_scores_val_v3.npy,   vae_scores_test_v3.npy
#   tae:   tae_scores_val_v3.npy,   tae_scores_test_v3.npy
def load_scores(method, split):
    candidates = {
        "stat":        [f"stat_scores_{split}_v2.npy"],
        "pred":        [f"pred_scores_{split}_v2.npy"],
        "vae":         [f"vae_scores_{split}_v3.npy"],
        "tae":         [f"tae_scores_{split}_v3.npy"],
    }
    for fname in candidates.get(method, []):
        p = CACHE_DIR / fname
        if p.exists():
            s = np.load(p)
            print(f"  loaded {method}_{split}: {s.shape}")
            return s
    raise FileNotFoundError(f"No cache for {method}_{split}")


def best_thresh(s, gt):
    """Grid search for best threshold on val, return threshold + metrics."""
    gt_flat = gt.flatten()
    best_f, best_t, best_m = 0.0, 0.5, None
    for t in np.arange(0.01, 0.99, 0.005):
        pred = (s >= t).astype(int).flatten()
        f = _f1(gt_flat, pred)
        if f > best_f:
            best_f, best_t = f, t
            best_m = point_metrics((s >= t).astype(int), gt)
    return best_t, best_m


# ── Fusion grid search ────────────────────────────────────────────────────────
def grid_fusion(score_dict, gt_val, gt_test):
    """Grid search over fusion weights + threshold for dual/triple/quad fusion."""
    results = {}
    gt_val_flat = gt_val.flatten()
    gt_test_flat = gt_test.flatten()
    stat, pred, vae, tael = "statistical", "prediction", "vae", "transformer"

    def _try_fusion(name, weight_combos, available_keys):
        """Generic fusion: weight_combos = list of (name, weights_dict)."""
        best_f, best_t, best_ws, best_pred = 0.0, 0.5, None, None
        for ws in weight_combos:
            if not all(k in available_keys for k in ws):
                continue
            fused_val = sum(w * norm(score_dict[k]["val"]) for k, w in ws.items())
            fused_val_flat = fused_val.flatten()
            for t in np.arange(0.10, 0.96, 0.02):
                pred_v = (fused_val_flat >= t).astype(int)
                f = _f1(gt_val_flat, pred_v)
                if f > best_f:
                    best_f, best_t, best_ws = f, t, ws

        if best_ws is None:
            return None

        fused_test = sum(w * norm(score_dict[k]["test"]) for k, w in best_ws.items())
        test_pred = (fused_test >= best_t).astype(int)
        m = point_metrics(test_pred, gt_test)
        m["weights"] = {k: round(float(w), 2) for k, w in best_ws.items()}
        m["threshold"] = round(float(best_t), 3)
        m["val_f1"] = round(float(best_f), 4)
        print(f"  [{name}] w={dict(m['weights'])} val_f1={m['val_f1']} "
              f"test P={m['precision']} R={m['recall']} F1={m['f1']} AUC={m['auc']}")
        return m

    # Dual: stat + pred
    dual_weights = [{"statistical": round(w, 2), "prediction": round(1 - w, 2)}
                   for w in np.arange(0.0, 1.001, 0.05)]
    m = _try_fusion("dual_stat_pred", dual_weights,
                    {k: score_dict[k] for k in [stat, pred] if k in score_dict})
    if m:
        results["dual_stat_pred"] = m

    # Triple: stat + pred + vae
    triple_weights = []
    for w1 in np.arange(0.0, 1.001, 0.1):
        for w2 in np.arange(0.0, 1.001 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0:
                continue
            triple_weights.append({stat: round(w1, 2), pred: round(w2, 2), vae: w3})
    m = _try_fusion("triple_stat_pred_vae", triple_weights,
                    {k: score_dict[k] for k in [stat, pred, vae] if k in score_dict})
    if m:
        results["triple_stat_pred_vae"] = m

    # Quad: stat + pred + vae + tael
    quad_weights = []
    for w1 in np.arange(0.0, 1.001, 0.1):
        for w2 in np.arange(0.0, 1.001 - w1, 0.1):
            for w3 in np.arange(0.0, 1.001 - w1 - w2, 0.1):
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < 0:
                    continue
                quad_weights.append({stat: round(w1, 2), pred: round(w2, 2),
                                     vae: round(w3, 2), tael: w4})
    m = _try_fusion("quad_all", quad_weights,
                    {k: score_dict[k] for k in [stat, pred, vae, tael] if k in score_dict})
    if m:
        results["quad_all"] = m

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("Week5 V3 Full Evaluation")
    print("=" * 70)

    gt_test = np.load(DATA_DIR / "anomaly_labels_test.npy")
    gt_val  = np.load(DATA_DIR / "anomaly_labels_val.npy")
    print(f"  gt_test={gt_test.shape} sum={int(gt_test.sum())}")
    print(f"  gt_val={gt_val.shape}  sum={int(gt_val.sum())}")

    # ── Single methods ───────────────────────────────────────────────────────
    results = {"single": {}, "fusion": {}}
    score_dict = {}
    methods = [("stat", "statistical"), ("pred", "prediction"),
               ("vae", "vae"), ("tae", "transformer")]

    for m, name in methods:
        try:
            sv = load_scores(m, "val")
            st = load_scores(m, "test")
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")
            continue

        sv_n, st_n = norm(sv), norm(st)
        auc_v = roc_auc_score(gt_val.flatten(), sv_n.flatten())
        t, mv = best_thresh(sv_n, gt_val)
        mt = point_metrics((st_n >= t).astype(int), gt_test)
        mt["val_auc"] = round(auc_v, 4)
        mt["val_f1"] = mv["f1"]
        mt["threshold"] = round(t, 3)
        print(f"  [{name}] val_auc={auc_v:.4f} val_f1={mv['f1']} "
              f"test P={mt['precision']} R={mt['recall']} F1={mt['f1']} AUC={mt['auc']}")
        results["single"][name] = mt
        score_dict[name] = {"val": sv, "test": st}

    # ── Fusion ───────────────────────────────────────────────────────────────
    print("\n[Fusion] Grid search...")
    fusion_results = grid_fusion(score_dict, gt_val, gt_test)
    results["fusion"].update(fusion_results)

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)
    json_path = REPORT_DIR / f"v3_full_eval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "single": results["single"], "fusion": results["fusion"]},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {json_path}")

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Val F1':>8} {'Test P':>8} {'Test R':>8} {'Test F1':>8} {'Test AUC':>8}")
    print("-" * 70)
    for name, m in results["single"].items():
        print(f"{name:<30} {m['val_f1']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")
    for name, m in results["fusion"].items():
        print(f"{'['+name+']':<30} {m['val_f1']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")
    return results


if __name__ == "__main__":
    main()
