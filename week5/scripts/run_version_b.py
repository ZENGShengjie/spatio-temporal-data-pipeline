"""Week5 版本 B 主入口 V2 — VAE + Transformer AE + 融合升级

V2 修复：
  - 注入步骤前置（确保验证集/测试集有标注）
  - 统计法 + 预测误差法 + VAE + TAE + 融合全链路
  - 缓存命名统一加 _v2 后缀，避免覆盖 V1 结果
"""
from __future__ import annotations
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def step_inject():
    print("\n" + "=" * 60)
    print("Step 1/6: 异常注入 V2（按目标占比，强制校验）")
    print("=" * 60)
    from inject_anomalies import run as inject_run
    return inject_run()


def step_statistical():
    print("\n" + "=" * 60)
    print("Step 2/6: 统计法检测 V2（注入后数据，验证集阈值）")
    print("=" * 60)
    from anomaly.statistical import run as stat_run
    return stat_run()


def step_prediction():
    print("\n" + "=" * 60)
    print("Step 3/6: 预测误差法检测 V2（pred分母 + F1阈值搜索）")
    print("=" * 60)
    from anomaly.prediction import run as pred_run
    return pred_run()


def step_vae():
    print("\n" + "=" * 60)
    print("Step 4/6: 单网格 VAE 异常检测")
    print("=" * 60)
    from anomaly.vae import run as vae_run
    return vae_run()


def step_transformer():
    print("\n" + "=" * 60)
    print("Step 5/6: Transformer AE V2（MAE掩码 + d_model=64 + 三层正则）")
    print("=" * 60)
    from anomaly.transformer_ae import run as tae_run
    return tae_run()


def step_fusion():
    print("\n" + "=" * 60)
    print("Step 6/6: 融合框架 V2（性能准入门槛 + 权重搜索）")
    print("=" * 60)
    from anomaly.fusion import run as fusion_run
    return fusion_run()


def step_evaluate():
    print("\n" + "=" * 60)
    print("评估：点级 + 事件级 + 分维度")
    print("=" * 60)
    from evaluation.metrics import run_evaluation
    return run_evaluation()


def run_version_b(skip_inject: bool = False):
    """完整运行 Version B。

    Args:
        skip_inject: 若注入已运行且数据存在，可跳过注入步骤
    """
    print("Week5 Version B V2 — 全链路异常检测")
    print("数据泄露红线：模型仅用正常训练集 | 阈值/权重仅验证集")
    print("-" * 60)

    if not skip_inject:
        step_inject()

    step_statistical()
    step_prediction()
    step_vae()
    step_transformer()
    step_fusion()
    df = step_evaluate()

    summary = df[df["method"].isin(["statistical", "prediction", "vae",
                                     "transformer", "fusion"])]
    print("\n" + "=" * 60)
    print("Version B V2 Final Results")
    print("=" * 60)
    print(summary[["method", "precision", "recall", "f1", "auc_roc", "auc_pr"]].to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-inject", action="store_true",
                        help="跳过注入步骤（注入已运行时使用）")
    args = parser.parse_args()
    run_version_b(skip_inject=args.skip_inject)
