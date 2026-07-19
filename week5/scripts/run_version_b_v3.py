"""run_version_b_v3.py — V3 evaluation via V2 caches"""
import os, sys, json
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(LOCAL_PATH))
from config import DATA_DIR, REPORT_DIR, VAL_HOURS, N_CELLS, cache_path


def load_labels():
    gt_test = np.load(os.path.join(DATA_DIR, "anomaly_labels_test.npy"))
    vp = os.path.join(DATA_DIR, "anomaly_labels_val_v3.npy")
    if os.path.exists(vp):
        gt_val = np.load(vp)
        print("  [V3] val labels: sum=" + str(int(gt_val.sum())))
    else:
        gt_val = np.zeros((VAL_HOURS, N_CELLS), dtype=bool)
        print("  [V3] WARNING: no val labels")
    return gt_test, gt_val


def load_scores(method, split):
    for suffix in ("_v2", ""):
        p = cache_path(f"{method}_scores_{split}{suffix}")
        if os.path.exists(p):
            s = np.load(p)
            print("  loaded " + method + "_" + split + ": " + str(s.shape))
            return s
    raise FileNotFoundError("No cache for " + method + "_" + split)


def norm(s):
    q99, q00 = np.percentile(s, 99), s.min()
    r = q99 - q00
    return np.zeros_like(s) if r < 1e-9 else np.clip((s - q00) / r, 0, 1)


def point_metrics(pred, gt):
    gt_f, p_f = gt.flatten().astype(int), pred.flatten().astype(int)
    tp = int(((p_f==1)&(gt_f==1)).sum())
    fp = int(((p_f==1)&(gt_f==0)).sum())
    fn = int(((p_f==0)&(gt_f==1)).sum())
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); f = 2*p*r/(p+r+1e-9)
    try: auc = roc_auc_score(gt_f, p_f)
    except: auc = 0.5
    return dict(precision=round(p,4), recall=round(r,4), f1=round(f,4), auc=round(auc,4))


def best_thresh(s, gt):
    best_f, best_t, best_m = 0, 0.5, None
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (s >= t).astype(int)
        gt_f = gt.flatten()
        _, _, f, _ = precision_recall_fscore_support(gt_f, pred, average="binary", zero_division=0)
        if f > best_f:
            best_f, best_t, best_m = f, t, point_metrics(pred, gt)
    return best_t, best_m


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("="*60)
    print("Week5 V3 Evaluation")
    print("="*60)

    gt_test, gt_val = load_labels()
    print("  gt_test=" + str(gt_test.shape) + " gt_val=" + str(gt_val.shape))

    results = {"single": {}, "fusion": {}}

    for m, name in [("stat","statistical"),("pred","prediction"),("vae","vae"),("tae","transformer")]:
        try:
            sv = load_scores(m, "val"); st = load_scores(m, "test")
        except FileNotFoundError as e:
            print("  SKIP " + name + ": " + str(e)); continue

        sv_n, st_n = norm(sv), norm(st)
        auc_v = roc_auc_score(gt_val.flatten(), sv_n.flatten())
        t, mv = best_thresh(sv_n, gt_val)
        mt = point_metrics((st_n >= t).astype(int), gt_test)
        mt["val_auc"] = round(auc_v, 4)
        mt["val_f1"] = mv["f1"]
        mt["threshold"] = round(t, 3)
        print("  [" + name + "] val_auc=" + str(round(auc_v,4)) + " val_f1=" + str(mv["f1"]) + " test P=" + str(mt["precision"]) + " R=" + str(mt["recall"]) + " F1=" + str(mt["f1"]) + " AUC=" + str(mt["auc"]))
        results["single"][name] = mt

    # dual fusion
    if "statistical" in results["single"] and "prediction" in results["single"]:
        sv_s = norm(load_scores("stat", "val")); st_s = norm(load_scores("stat", "test"))
        sv_p = norm(load_scores("pred", "val")); st_p = norm(load_scores("pred", "test"))
        best_f, best_t, best_w = 0, 0.5, 0.8
        for w in np.arange(0.0, 1.001, 0.05):
            fused = w*sv_s + (1-w)*sv_p
            for t in np.arange(0.10, 0.96, 0.02):
                pred = (fused >= t).astype(int)
                _, _, f, _ = precision_recall_fscore_support(gt_val.flatten(), pred, average="binary", zero_division=0)
                if f > best_f:
                    best_f, best_t, best_w = f, t, w
        m = point_metrics((best_w*st_s+(1-best_w)*st_p >= best_t).astype(int), gt_test)
        m["weights"] = {"statistical": round(float(best_w),2), "prediction": round(float(1-best_w),2)}
        m["threshold"] = round(float(best_t),3)
        m["val_f1"] = round(float(best_f),4)
        print("  [dual_fusion] w=" + str(round(best_w,2)) + " val_f1=" + str(round(best_f,4)) + " test P=" + str(m["precision"]) + " R=" + str(m["recall"]) + " F1=" + str(m["f1"]) + " AUC=" + str(m["auc"]))
        results["fusion"]["dual_stat_pred"] = m

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = {"timestamp": ts, "single": results["single"], "fusion": results["fusion"]}
    p = os.path.join(REPORT_DIR, "v3_eval_" + ts + ".json")
    with open(p, "w") as f: json.dump(out, f, indent=2)
    print("Saved: " + p)


if __name__ == "__main__": main()
