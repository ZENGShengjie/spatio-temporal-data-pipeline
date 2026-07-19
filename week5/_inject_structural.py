#!/usr/bin/env python3
"""结构型异常对照数据集生成器 — Week5 Ablation 实验
===============================================================
保持总异常率 4.02% 不变，调整异常结构：
  - 60% 连片结构型异常（sustained 为主，破坏时序形态与空间联动）
  - 40% 单点数值突变异常（surge/drop，保持 V3 场景对照）
输出与 V3 完全隔离的后缀文件，不覆盖原版数据。
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── 本地 import ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    INJECT_CFG, DATA_DIR, CACHE_DIR,
    VAL_END, TEST_END, TEST_HOURS, N_CELLS, VAL_HOURS,
    TRAIN_END,
    cache_path, cache_json,
)
from inject_anomalies import inject_split, _build_spatial_bins, _build_temporal_bins


# ── 覆盖默认配置的参数组 ──────────────────────────────────────────────────────

# V3 原始参数（对照组）
V3_TYPE_WEIGHTS  = (0.40, 0.40, 0.20)   # surge, drop, sustained
V3_SPATIAL_MIX   = (0.50, 0.50)          # 50%单点, 50%连片
V3_DURATION      = (2, 8)
V3_AMPLITUDE     = (2.5, 5.0)

# 结构型异常参数（实验组）
# 目标：强化 sustained + 强制连片 → 破坏时序形态与空间联动
STRUCTURAL_TYPE_WEIGHTS = (0.15, 0.15, 0.70)  # 70% sustained
STRUCTURAL_SPATIAL_MIX = (0.40, 0.60)         # 60% 连片
STRUCTURAL_DURATION     = (3, 10)              # 更长持续时间（强化结构破坏）
STRUCTURAL_AMPLITUDE    = (1.8, 3.5)          # 适中幅度但持续更久


def inject_structural(
    flow: np.ndarray,
    target_ratio: float,
    split_name: str = "test",
    seed: int = 999,
    type_weights=STRUCTURAL_TYPE_WEIGHTS,
    spatial_mix=STRUCTURAL_SPATIAL_MIX,
    duration_range=STRUCTURAL_DURATION,
    amplitude_range=STRUCTURAL_AMPLITUDE,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """调用 inject_split，使用结构型异常参数组。"""
    return inject_split(
        flow=flow,
        target_ratio=target_ratio,
        split_name=split_name,
        seed=seed,
        duration_range=duration_range,
        amplitude_range=amplitude_range,
        type_weights=type_weights,
        spatial_mix=spatial_mix,
    )


def compute_type_breakdown(events_df: pd.DataFrame, labels: np.ndarray) -> dict:
    """计算各类型异常单元格数量（用于分项指标）。"""
    T, N = labels.shape
    result = {}
    for atype in ["surge", "drop", "sustained"]:
        mask = np.zeros_like(labels, dtype=bool)
        for _, row in events_df[events_df["type"] == atype].iterrows():
            t_s, t_e = int(row["t_start"]), int(row["t_end"])
            n_cells = eval(row["affected_cells"]) if isinstance(row["affected_cells"], str) else row["affected_cells"]
            for t in range(t_s, t_e):
                for n in n_cells:
                    if t < T and n < N:
                        mask[t, n] = True
        result[atype] = {"n_cells": int(mask.sum()), "ratio": float(mask.sum() / labels.size)}
    return result


# ── 主入口 ───────────────────────────────────────────────────────────────────

def run(target: str = "taxi_flow_total"):
    from data_loader import get_flow_1d

    print("=" * 60)
    print("结构型异常对照数据集生成 — Week5 Ablation")
    print("=" * 60)

    # 加载纯净全量数据（复用 V3 清洗后的同一来源）
    flow_all = get_flow_1d(target)  # (3888, N)

    target_ratio = INJECT_CFG.anomaly_ratio  # 0.04
    output_suffix = "_structural"

    # ── 验证集注入 ──────────────────────────────────────────────────────────
    val_start = TRAIN_END
    val_end   = VAL_END
    T_val     = VAL_HOURS   # = val_end - val_start

    flow_val = flow_all[val_start:val_end].copy()
    labels_val = np.zeros((T_val, N_CELLS), dtype=bool)

    print(f"\n[Val] 生成结构型异常对照数据集")
    print(f"  type_weights  = {STRUCTURAL_TYPE_WEIGHTS}  (surge/drop/sustained)")
    print(f"  spatial_mix   = {STRUCTURAL_SPATIAL_MIX}  (单点/连片)")
    print(f"  duration      = {STRUCTURAL_DURATION}")
    print(f"  amplitude     = {STRUCTURAL_AMPLITUDE}")

    labels_val, events_val, summary_val = inject_structural(
        flow_val, target_ratio, split_name="val_structural", seed=999
    )

    # ── 测试集注入 ─────────────────────────────────────────────────────────
    test_start = VAL_END
    test_end   = TEST_END
    T_test     = TEST_HOURS  # = test_end - test_start

    flow_test = flow_all[test_start:test_end].copy()
    labels_test = np.zeros((T_test, N_CELLS), dtype=bool)

    print(f"\n[Test] 生成结构型异常对照数据集")
    labels_test, events_test, summary_test = inject_structural(
        flow_test, target_ratio, split_name="test_structural", seed=1999
    )

    # ── 保存文件（后缀隔离，不覆盖 V3）────────────────────────────────────────
    labels_dir = DATA_DIR
    os.makedirs(labels_dir, exist_ok=True)

    # 标签
    np.save(os.path.join(labels_dir, f"anomaly_labels_val{output_suffix}.npy"), labels_val)
    np.save(os.path.join(labels_dir, f"anomaly_labels_test{output_suffix}.npy"), labels_test)

    # 流量（含注入）
    np.save(os.path.join(labels_dir, f"flow_val_injected{output_suffix}.npy"), flow_val)
    np.save(os.path.join(labels_dir, f"flow_test_injected{output_suffix}.npy"), flow_test)

    # 事件列表
    events_val.to_csv(os.path.join(labels_dir, f"injected_events_val{output_suffix}.csv"), index=False)
    events_test.to_csv(os.path.join(labels_dir, f"injected_events_test{output_suffix}.csv"), index=False)

    # 类型分布统计
    type_breakdown_val  = compute_type_breakdown(events_val,  labels_val)
    type_breakdown_test = compute_type_breakdown(events_test, labels_test)

    # 汇总 JSON
    injection_summary = {
        "version": f"structural_ablation{output_suffix}",
        "timestamp": datetime.now().isoformat(),
        "seed_val":  999,
        "seed_test": 1999,
        "val_summary": {
            **summary_val,
            "type_breakdown": type_breakdown_val,
            "params": {
                "type_weights":  list(STRUCTURAL_TYPE_WEIGHTS),
                "spatial_mix":  list(STRUCTURAL_SPATIAL_MIX),
                "duration":      list(STRUCTURAL_DURATION),
                "amplitude":     list(STRUCTURAL_AMPLITUDE),
            },
        },
        "test_summary": {
            **summary_test,
            "type_breakdown": type_breakdown_test,
            "params": {
                "type_weights":  list(STRUCTURAL_TYPE_WEIGHTS),
                "spatial_mix":  list(STRUCTURAL_SPATIAL_MIX),
                "duration":      list(STRUCTURAL_DURATION),
                "amplitude":     list(STRUCTURAL_AMPLITUDE),
            },
        },
        "comparison_note": (
            "与 V3 baseline 对照：保持 4.02% 总异常率不变，"
            "将 sustained 比例从 20% 提升至 70%，"
            "连片比例从 50% 提升至 60%，"
            "验证结构型异常场景下重构模型（VAE/TAE）的性能变化。"
        ),
        "data_leakage": "compliant — 仅验证集和测试集注入，训练集保持纯净",
    }

    summary_path = os.path.join(labels_dir, f"injection_summary{output_suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(injection_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"保存完成（后缀 = {output_suffix}，不覆盖 V3）：")
    print(f"  标签:   anomaly_labels_val{output_suffix}.npy / test{output_suffix}.npy")
    print(f"  流量:   flow_val_injected{output_suffix}.npy / test{output_suffix}.npy")
    print(f"  事件:   injected_events_val{output_suffix}.csv / test{output_suffix}.csv")
    print(f"  配置:   injection_summary{output_suffix}.json")
    print(f"\nVal  类型分布:  surge={type_breakdown_val['surge']['n_cells']}  "
          f"drop={type_breakdown_val['drop']['n_cells']}  "
          f"sustained={type_breakdown_val['sustained']['n_cells']}")
    print(f"Test 类型分布:  surge={type_breakdown_test['surge']['n_cells']}  "
          f"drop={type_breakdown_test['drop']['n_cells']}  "
          f"sustained={type_breakdown_test['sustained']['n_cells']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="结构型异常对照数据集生成器")
    parser.add_argument("--target", default="taxi_flow_total")
    args = parser.parse_args()
    run(args.target)
