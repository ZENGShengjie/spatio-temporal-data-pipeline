"""Week4 主入口 — 高级时空模型训练与对比

用法:
    python run_week4.py --models stgcn agformer stf --target taxi_flow_total

    # 消融实验:
    python run_week4.py --models stgcn agformer agformer_static --target taxi_flow_total
    python run_week4.py --models stf stf_loc_only --target taxi_flow_total

输出:
    <WEEK4_DIR>/results/<model>_pred.npy
    <WEEK4_DIR>/results/<model>_gt.npy
    <WEEK4_DIR>/results/summary_<target>_<tag>.md
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from data_loader import load_raw_flow, load_time_features
from registry import get_trainer, list_models
from metrics import (evaluate_predictions, write_metrics_summary,
                     write_top_offender_table, save_npy)


def parse_args():
    p = argparse.ArgumentParser(description="Week4 Advanced Spatio-Temporal Models")
    p.add_argument("--models", nargs="+",
                   default=["stgcn", "agformer", "stf"],
                   choices=list_models())
    p.add_argument("--target", default="taxi_flow_total",
                   choices=["taxi_flow_total", "taxi_inflow", "taxi_outflow"])
    p.add_argument("--tag", default="v4fix",
                   help="result tag, e.g. v4fix → *_v4fix_pred.npy")
    p.add_argument("--ablation", action="store_true",
                   help="include ablation models (agformer_static, stf_loc_only)")
    p.add_argument("--skip_baseline", action="store_true",
                   help="skip week3 baseline models (gcn, gat) in comparison")
    return p.parse_args()


def main():
    args = parse_args()

    # 消融模式：自动补上消融模型
    models = list(args.models)
    if args.ablation:
        for m in ["agformer_static", "stf_loc_only"]:
            if m in list_models() and m not in models:
                models.append(m)

    # 输出目录
    WEEK4_DIR = os.environ.get("WEEK4_DIR",
                               os.path.join(os.path.dirname(__file__), ".."))
    PRED_DIR  = os.path.join(WEEK4_DIR, "results")
    LOG_DIR   = os.path.join(WEEK4_DIR, "logs")
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.chdir(WEEK4_DIR)

    print("=" * 70)
    print("Week4 — Advanced Spatio-Temporal Models")
    print("=" * 70)
    print(f"  models   : {models}")
    print(f"  target   : {args.target}")
    print(f"  tag      : {args.tag}")
    print(f"  data dir : {cfg.DATA_DIR}")
    print(f"  device   : cuda = {os.environ.get('CUDA_OFF') != '1'}")
    print("=" * 70)

    # ── 1. 加载数据 ──────────────────────────────────────────────────
    print("\n[1/3] loading data ...")
    flow_4d = load_raw_flow()
    time_feat = load_time_features()
    print(f"  flow_4d.shape = {flow_4d.shape}")
    print(f"  time_feat.shape = {time_feat.shape}")
    print(f"  cell_max: {flow_4d[:cfg.SPLIT.train_end].max():.1f}")
    print(f"  split: train={cfg.SPLIT.train_end} val={cfg.SPLIT.val_end} test={cfg.SPLIT.test_end}")
    print(f"  task: seq_len={cfg.cfg_train.seq_len} → horizon={cfg.cfg_train.horizon}")

    # ── 2. 训练 + 评估 ─────────────────────────────────────────────────
    print("\n[2/3] training & evaluating ...")
    rows = []
    preds = {}

    for name in models:
        print(f"\n{'='*60}")
        print(f"  [{name.upper()}]")
        print(f"{'='*60}")
        log_path = os.path.join(LOG_DIR, f"{name}_{args.target}_{args.tag}.log")
        t0 = time.time()

        try:
            trainer = get_trainer(name)
            pred, gt = trainer.fit_predict(flow_4d, time_features=time_feat,
                                           target=args.target)
            elapsed = time.time() - t0

            metrics = evaluate_predictions(pred, gt, target_cols=[args.target])
            np.save(os.path.join(PRED_DIR, f"{name}_{args.target}_{args.tag}_pred.npy"),
                    pred.astype(np.float32))
            np.save(os.path.join(PRED_DIR, f"{name}_{args.target}_{args.tag}_gt.npy"),
                    gt.astype(np.float32))

            row = {
                "model": name,
                "MAE": round(metrics["MAE"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "MAPE": round(metrics["MAPE"], 4),
                "Corr": round(metrics["Corr"], 4),
                "n_params": getattr(trainer, "n_params", None),
                "best_epoch": getattr(trainer, "best_epoch", None),
                "train_time_s": round(elapsed, 1),
                "test_time_s": 0.0,
                "pred_shape": str(pred.shape),
            }
            rows.append(row)
            preds[name] = pred

            print(f"  {name}: MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}"
                  f"  MAPE={metrics['MAPE']:.4f}  Corr={metrics['Corr']:.4f}"
                  f"  time={elapsed/60:.1f}min")

            # 写日志
            with open(log_path, "w") as f:
                f.write(f"model={name}\n")
                f.write(f"target={args.target}\n")
                f.write(f"elapsed={elapsed:.1f}s\n")
                json.dump(row, f, indent=2)

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print(f"  [FAIL] {name}: {e}")
            print(err)
            with open(log_path, "w") as f:
                f.write(err)
            rows.append({"model": name, "error": str(e)})

    # ── 3. 汇总 ───────────────────────────────────────────────────────
    print("\n[3/3] writing summary ...")
    rows = [r for r in rows if "error" not in r]
    summary_path = os.path.join(PRED_DIR, f"summary_{args.target}_{args.tag}.md")
    write_metrics_summary(summary_path, rows)

    if preds:
        gt_path = os.path.join(PRED_DIR,
                               f"{models[0]}_{args.target}_{args.tag}_gt.npy")
        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            offenders_path = os.path.join(PRED_DIR,
                                         f"top_offenders_{args.target}_{args.tag}.md")
            write_top_offender_table(offenders_path, preds, gt, top_k=20)

    # ── 4. 打印对比表 ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'模型':<22} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Corr':>8} {'耗时':>10}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model']:<22} {r['MAE']:8.4f} {r['RMSE']:8.4f} "
              f"{r['MAPE']:8.4f} {r['Corr']:8.4f} {r['train_time_s']:>8.1f}s")
    print("=" * 80)

    print(f"\nDONE. results → {PRED_DIR}/")
    for name in models:
        for fn in (f"{name}_{args.target}_{args.tag}_pred.npy",
                   f"{name}_{args.target}_{args.tag}_gt.npy"):
            p = os.path.join(PRED_DIR, fn)
            if os.path.exists(p):
                print(f"  ✓ {os.path.basename(p)}")


if __name__ == "__main__":
    main()
