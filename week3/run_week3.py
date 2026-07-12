"""Week3 主入口脚本

用法:
    python run_week3.py --models arima prophet lstm gru gcn gat --target taxi_flow_total

输出:
    /home/ubuntu/amazon/week3/results/<model>_pred.npy
    /home/ubuntu/amazon/week3/results/<model>_gt.npy
    /home/ubuntu/amazon/week3/results/summary.md
    /home/ubuntu/amazon/week3/logs/<model>.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

# 在容器/g4dn 上, 让 python -m / src 目录中的代码可找到
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from data_loader import load_raw_flow, load_time_features
from registry import get_trainer, list_models
from metrics import evaluate_predictions, write_metrics_summary, write_top_offender_table, save_npy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list_models(),
                   choices=list_models() + [[m for m in list_models()]])
    p.add_argument("--target", default="taxi_flow_total",
                   choices=["taxi_flow_total", "taxi_inflow", "taxi_outflow"])
    p.add_argument("--data_only", action="store_true",
                   help="just save data shape summary and exit")
    p.add_argument("--tag", default="v2",
                   help="result filename suffix (e.g. v2 → *_v2_pred.npy)")
    p.add_argument("--no_time_features", action="store_true",
                   help="V2 ablation: disable time features (SeqDataset will get None)")
    return p.parse_args()


def main():
    args = parse_args()
    if isinstance(args.models, list) and len(args.models) == 1 and isinstance(args.models[0], list):
        args.models = args.models[0]

    # 强制以这个根目录为工作目录
    WEEK3_DIR = cfg.WEEK3_DIR
    os.chdir(WEEK3_DIR)

    print("=" * 70)
    print("Week3 — 6 baseline models comparison")
    print("=" * 70)
    print(f"  models    : {args.models}")
    print(f"  target    : {args.target}")
    print(f"  data dir  : {cfg.DATA_DIR}")
    print(f"  output dir: {WEEK3_DIR}")
    print(f"  device    : cuda available = {os.environ.get('CUDA_OFF') != '1'}")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/3] loading data ...")
    flow_4d = load_raw_flow()
    if args.no_time_features:
        time_feat = None
        print("  [V2 ablation] time features DISABLED")
    else:
        time_feat = load_time_features()
    print(f"  flow_4d.shape = {flow_4d.shape}, dtype={flow_4d.dtype}")
    print(f"  time_feat.shape = {None if time_feat is None else time_feat.shape}")
    print(f"  cell_max: {flow_4d[:cfg.SPLIT.train_end].max():.1f}, "
          f"split: train={cfg.SPLIT.train_end}, val={cfg.SPLIT.val_end}, test={cfg.SPLIT.test_end}")
    print(f"  task: seq_len={cfg.cfg_train.seq_len} → horizon={cfg.cfg_train.horizon}")
    print(f"  output tag: {args.tag}")

    if args.data_only:
        print("data_only mode — exiting")
        return

    # 2. 跑模型
    print("\n[2/3] training & evaluating ...")
    rows = []
    preds = {}
    for name in args.models:
        print(f"\n--- [{name.upper()}] ---")
        log_path = os.path.join(cfg.LOG_DIR, f"{name}_{args.target}.log")
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        t0 = time.time()
        try:
            trainer = get_trainer(name)
            pred, gt = trainer.fit_predict(flow_4d, time_features=time_feat,
                                            target=args.target)
            elapsed = time.time() - t0
            # 评估
            metrics = evaluate_predictions(pred, gt, target_cols=[args.target])
            # 保存预测
            np.save(os.path.join(cfg.PRED_DIR, f"{name}_{args.target}_{args.tag}_pred.npy"), pred.astype(np.float32))
            np.save(os.path.join(cfg.PRED_DIR, f"{name}_{args.target}_{args.tag}_gt.npy"),   gt.astype(np.float32))
            row = {
                "model": name,
                "MAE": metrics["MAE"], "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"], "Corr": metrics["Corr"],
                "train_time_s": round(elapsed, 1),
                "test_time_s": 0.0,
                "pred_shape": str(pred.shape),
            }
            rows.append(row)
            preds[name] = pred
            print(f"  ✅ {name}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, "
                  f"Corr={metrics['Corr']:.4f}, elapsed={elapsed/60:.1f} min")
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print(f"  ❌ {name} FAILED: {e}")
            print(err)
            with open(log_path, "w") as f:
                f.write(err)
            rows.append({"model": name, "error": str(e)})

    # 3. 写入汇总
    print("\n[3/3] writing summary ...")
    rows = [r for r in rows if "error" not in r]
    summary_path = os.path.join(WEEK3_DIR, "results", f"summary_{args.target}_{args.tag}.md")
    write_metrics_summary(summary_path, rows)

    # worst-cell 表
    if preds:
        # gt 取第一个完整模型的 gt
        first_gt_path = os.path.join(cfg.PRED_DIR, f"{args.models[0]}_{args.target}_{args.tag}_gt.npy")
        if os.path.exists(first_gt_path):
            gt = np.load(first_gt_path)
            offenders_path = os.path.join(WEEK3_DIR, "results", f"top_offenders_{args.target}_{args.tag}.md")
            write_top_offender_table(offenders_path, preds, gt, top_k=20)

    print("\nDONE.")
    print(f"  results dir : {WEEK3_DIR}/results/")
    for name in args.models:
        for fn in (f"{name}_{args.target}_pred.npy", f"{name}_{args.target}_gt.npy"):
            p = os.path.join(cfg.PRED_DIR, fn)
            if os.path.exists(p):
                print(f"  ✓ {p}")


if __name__ == "__main__":
    main()
