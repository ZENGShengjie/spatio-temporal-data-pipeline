"""Week5 版本 A 主入口 — 统计法 + 预测误差法，快速出基线"""
from __future__ import annotations
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR


def step_inject():
    print("\n" + "=" * 60)
    print("Step 1/5: 异常注入 V2（按目标占比，强制校验）")
    print("=" * 60)
    from inject_anomalies import run as inject_run
    return inject_run()


def step_statistical():
    print("\n" + "=" * 60)
    print("Step 2/5: 统计法检测 (3σ + IQR)")
    print("=" * 60)
    from anomaly.statistical import run as stat_run
    return stat_run()


def step_prediction():
    print("\n" + "=" * 60)
    print("Step 3/5: 预测误差法检测")
    print("=" * 60)
    from anomaly.prediction import run as pred_run
    return pred_run()


def step_fusion():
    print("\n" + "=" * 60)
    print("Step 4/5: 融合框架")
    print("=" * 60)
    from anomaly.fusion import run as fusion_run
    return fusion_run()


def step_evaluate():
    print("\n" + "=" * 60)
    print("Step 5/5: 评估")
    print("=" * 60)
    from evaluation.metrics import run_evaluation
    return run_evaluation()


def run_version_a():
    print("Week5 Version A - 统计法 + 预测误差法 基线")
    print("数据泄露红线：注入仅测试集 | 阈值仅验证集 | 融合权重仅验证集")
    step_inject()
    step_statistical()
    step_prediction()
    step_fusion()
    df = step_evaluate()
    summary = df[df["method"].isin(["statistical", "prediction", "fusion"])]
    print("\n=== Version A Final Results ===")
    print(summary[["method", "precision", "recall", "f1", "auc_roc"]].to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+",
                        default=["inject", "statistical", "prediction", "fusion", "evaluate"])
    args = parser.parse_args()
    step_map = {
        "inject":       step_inject,
        "statistical":  step_statistical,
        "prediction":   step_prediction,
        "fusion":       step_fusion,
        "evaluate":     step_evaluate,
    }
    for step_name in args.steps:
        step_map[step_name]()
