"""合成异常注入脚本 V2 — 按目标单元格占比反向计算 + 时空均匀分布 + 强制校验

数据泄露红线：
  - 注入仅在 VAL_END:TEST_END（验证集）和 TST_START:TEST_END（测试集）进行
  - 训练集保持原始纯净数据
  - 输出: anomaly_labels_val.npy + anomaly_labels_test.npy + events_val.csv + events_test.csv

设计目标：
  - 验证集、测试集均注入 3%~5% 的异常单元格
  - 按类型（突增40%/突降40%/持续型20%）、空间（单点50%/连片50%）分层
  - 时空均匀分布，避免注入集中在某几小时或某几片区域
  - 注入完成后强制校验，不满足目标则 raise 阻断
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from datetime import datetime
import argparse

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


# ── 时空均匀分层抽样 ─────────────────────────────────────────────────────────

def _build_spatial_bins(n_bins: int = 8) -> dict:
    """将 32×32 网格划分为 n_bins×n_bins 的空间桶，返回每个桶内的格点索引列表"""
    H, W = 32, 32
    bin_h, bin_w = H // n_bins, W // n_bins
    bins = {}
    for bi in range(n_bins):
        for bj in range(n_bins):
            cells = []
            for r in range(bi * bin_h, (bi + 1) * bin_h):
                for c in range(bj * bin_w, (bj + 1) * bin_w):
                    cells.append(r * W + c)
            bins[(bi, bj)] = cells
    return bins


def _build_temporal_bins(T: int, n_bins: int = 10) -> list:
    """将给定的 T 小时均匀划分为 n_bins 个时间桶，返回每个桶的时间步范围"""
    bins = []
    bin_size = T // n_bins
    for i in range(n_bins):
        start = i * bin_size
        end = T if i == n_bins - 1 else (i + 1) * bin_size
        bins.append(list(range(start, end)))
    return bins


def _select_uniform_slots(
    n_slots: int,
    spatial_bins: dict,
    temporal_bins: list,
    rng: np.random.Generator,
) -> list:
    """均匀抽取时空注入槽位，每个槽位含 (t_bin_idx, spatial_bin_key, cell_idx)"""
    slots = []
    spatial_bin_keys = list(spatial_bins.keys())

    for _ in range(n_slots):
        # 时空分层：先选时间桶，再选空间桶，避免集中
        t_bin = rng.integers(0, len(temporal_bins))
        s_bin = rng.choice(spatial_bin_keys)
        cell_idx = rng.choice(spatial_bins[s_bin])
        t_in_bin = rng.choice(temporal_bins[t_bin])
        slots.append((t_in_bin, cell_idx, s_bin, t_bin))
    return slots


# ── 核心注入逻辑 ─────────────────────────────────────────────────────────────

def inject_split(
    flow: np.ndarray,
    target_ratio: float,
    split_name: str = "test",
    seed: int = 42,
    duration_range: tuple = (2, 8),
    amplitude_range: tuple = (2.5, 5.0),
    type_weights: tuple = (0.4, 0.4, 0.2),
    spatial_mix: tuple = (0.5, 0.5),
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """在给定数据切片上注入异常，保证时空均匀分布和目标占比。

    Args:
        flow: (T, N) 归一化后的流量
        target_ratio: 目标异常单元格占比（3%~5%）
        split_name: 用于日志输出
        seed: 随机种子
        ...

    Returns:
        labels: (T, N) bool
        events_df: DataFrame
        summary: dict 校验信息
    """
    rng = np.random.default_rng(seed)
    T, N = flow.shape

    # 计算基线（该切片自身的统计量）
    baseline = np.median(flow, axis=0)
    baseline = np.maximum(baseline, 1e-4)

    # 计算需要注入的异常单元格总数
    total_cells = T * N
    target_anomaly_cells = int(total_cells * target_ratio)

    # 构建时空分层桶
    spatial_bins = _build_spatial_bins(n_bins=8)
    temporal_bins = _build_temporal_bins(T, n_bins=10)

    labels = np.zeros((T, N), dtype=bool)

    # 计算各类型目标数量
    n_surge = int(target_anomaly_cells * type_weights[0])
    n_drop = int(target_anomaly_cells * type_weights[1])
    n_sustained = int(target_anomaly_cells * type_weights[2])

    events = []
    eid = 0

    # 按类型分别注入，确保类型配比
    for type_name, target_count, sign_or_zero in [
        ("surge", n_surge, +1),
        ("drop", n_drop, -1),
        ("sustained", n_sustained, 0),
    ]:
        # 用事件级迭代，每次注入覆盖 ~10-50 个单元格
        estimated_cells_per_event = 20  # 粗估
        n_events_needed = max(1, target_count // estimated_cells_per_event)

        cells_injected = 0
        attempts = 0
        max_attempts = n_events_needed * 10

        while cells_injected < target_count and attempts < max_attempts:
            attempts += 1

            # 随机选类型
            r = rng.random()
            if r < type_weights[0]:
                anom_type = "surge"
                sign = +1
            elif r < type_weights[0] + type_weights[1]:
                anom_type = "drop"
                sign = -1
            else:
                anom_type = "sustained"
                sign = 0

            # 随机选空间：单点 or 连片
            spatial_pick = rng.random()
            s_bin_keys = list(spatial_bins.keys())
            if spatial_pick < spatial_mix[0]:
                # 单点：从均匀分布的空间桶中选
                s_bin = s_bin_keys[rng.integers(0, len(s_bin_keys))]
                n_idx = int(rng.choice(spatial_bins[s_bin]))
                affected = [n_idx]
            else:
                # 连片 3×3
                s_bin = s_bin_keys[rng.integers(0, len(s_bin_keys))]
                bin_cells = spatial_bins[s_bin]
                if len(bin_cells) == 0:
                    continue
                # 选一个中心点，构造 3×3
                center = int(rng.choice(bin_cells))
                row, col = center // 32, center % 32
                if row < 1 or row > 30 or col < 1 or col > 30:
                    continue
                affected = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        r2, c2 = row + dr, col + dc
                        if 0 <= r2 < 32 and 0 <= c2 < 32:
                            affected.append(r2 * 32 + c2)
                n_idx = center

            # 随机选起始时间（从均匀时间桶中选）
            t_bin_idx = int(rng.integers(0, len(temporal_bins)))
            t_candidates = temporal_bins[t_bin_idx]
            if len(t_candidates) == 0:
                continue
            t_start = int(rng.choice(t_candidates))
            duration = int(rng.integers(duration_range[0], duration_range[1] + 1))
            t_end = min(int(t_start) + duration, T)

            # 计算注入幅度
            amp = rng.uniform(amplitude_range[0], amplitude_range[1])

            # 注入
            event_cells = 0
            for t in range(t_start, t_end):
                for n in affected:
                    if not labels[t, n]:  # 不重复注入已有异常
                        base = flow[t, n]
                        if anom_type == "sustained":
                            mid = (t_start + t_end) // 2
                            s = +1 if t < mid else -1
                        else:
                            s = sign
                        injected = base + s * amp * baseline[n]
                        injected = max(injected, 0.0)
                        flow[t, n] = injected
                        labels[t, n] = True
                        event_cells += 1

            # 记录事件
            event = {
                "event_id": eid,
                "split": split_name,
                "type": anom_type,
                "t_start": int(t_start),
                "t_end": int(t_end),
                "duration": int(t_end - t_start),
                "n_center": int(n_idx),
                "n_cells": len(affected),
                "event_cells_injected": event_cells,
                "affected_cells": str(affected[:5]) + "..." if len(affected) > 5 else str(affected),
                "amplitude": float(amp),
                "is_spatial": len(affected) > 1,
                "spatial_bin": str(s_bin),
                "temporal_bin": int(t_bin_idx),
            }
            events.append(event)
            eid += 1
            cells_injected += event_cells

    # 转为 DataFrame
    events_df = pd.DataFrame(events)

    # 校验
    actual_ratio = labels.sum() / labels.size
    summary = {
        "split": split_name,
        "seed": int(seed),
        "target_ratio": float(target_ratio),
        "actual_ratio": float(actual_ratio),
        "n_anomaly_cells": int(labels.sum()),
        "total_cells": int(total_cells),
        "n_events": int(len(events_df)),
        "type_counts": events_df["type"].value_counts().to_dict() if len(events_df) > 0 else {},
        "spatial_ratio": float(events_df["is_spatial"].mean()) if len(events_df) > 0 else 0.0,
        "temporal_bin_counts": events_df["temporal_bin"].value_counts().to_dict() if len(events_df) > 0 else {},
    }

    # 强制校验：实际占比必须在目标±1% 以内
    lower = target_ratio - 0.01
    upper = target_ratio + 0.01
    if actual_ratio < lower or actual_ratio > upper:
        raise ValueError(
            f"[inject V2] {split_name} 异常占比 {actual_ratio:.4f} 超出目标区间 "
            f"[{lower:.4f}, {upper:.4f}]，请检查注入逻辑！"
        )

    print(f"[inject V2] {split_name}: {labels.sum()}/{total_cells} = "
          f"{actual_ratio:.4f} (目标 {target_ratio:.4f}), "
          f"{len(events_df)} 事件, 类型={summary['type_counts']}")
    return labels, events_df, summary


# ── 主入口 ───────────────────────────────────────────────────────────────────

def run(target: str = "taxi_flow_total"):
    from data_loader import get_flow_1d

    # 加载纯净全量数据
    flow_all = get_flow_1d(target)  # (3888, N)
    target_ratio = INJECT_CFG.anomaly_ratio

    # ── 验证集注入 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[inject V2] 开始验证集注入（目标比例 {target_ratio:.2%}）")
    print(f"{'='*60}")

    flow_val = flow_all[TRAIN_END:VAL_END].copy()
    labels_val, events_val, summary_val = inject_split(
        flow=flow_val,
        target_ratio=target_ratio,
        split_name="val",
        seed=INJECT_CFG.seed,
        duration_range=INJECT_CFG.duration_range,
        amplitude_range=INJECT_CFG.amplitude_range,
        type_weights=INJECT_CFG.type_weights,
        spatial_mix=INJECT_CFG.spatial_mix,
    )

    # 保存验证集注入结果
    val_labels_path = os.path.join(DATA_DIR, "anomaly_labels_val.npy")
    np.save(val_labels_path, labels_val)
    print(f"[inject V2] 验证集标签 → {val_labels_path}")

    val_events_path = os.path.join(DATA_DIR, "injected_events_val.csv")
    events_val.to_csv(val_events_path, index=False)
    print(f"[inject V2] 验证集事件 → {val_events_path}")

    # 保存注入后的验证集流量（用于后续检测器阈值校准）
    val_injected_path = os.path.join(DATA_DIR, "flow_val_injected.npy")
    np.save(val_injected_path, flow_val)
    print(f"[inject V2] 验证集流量（已注入）→ {val_injected_path}")

    # ── 测试集注入 ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[inject V2] 开始测试集注入（目标比例 {target_ratio:.2%}）")
    print(f"{'='*60}")

    flow_test = flow_all[VAL_END:].copy()
    labels_test, events_test, summary_test = inject_split(
        flow=flow_test,
        target_ratio=target_ratio,
        split_name="test",
        seed=INJECT_CFG.seed + 100,  # 与验证集不同的 seed，避免重复
        duration_range=INJECT_CFG.duration_range,
        amplitude_range=INJECT_CFG.amplitude_range,
        type_weights=INJECT_CFG.type_weights,
        spatial_mix=INJECT_CFG.spatial_mix,
    )

    # 保存测试集注入结果
    test_labels_path = os.path.join(DATA_DIR, "anomaly_labels_test.npy")
    np.save(test_labels_path, labels_test)
    print(f"[inject V2] 测试集标签 → {test_labels_path}")

    test_events_path = os.path.join(DATA_DIR, "injected_events_test.csv")
    events_test.to_csv(test_events_path, index=False)
    print(f"[inject V2] 测试集事件 → {test_events_path}")

    # 保存注入后的测试集流量（用于后续检测）
    test_injected_path = os.path.join(DATA_DIR, "flow_test_injected.npy")
    np.save(test_injected_path, flow_test)
    print(f"[inject V2] 测试集流量（已注入）→ {test_injected_path}")

    # 保存原始纯净测试集（用于对比）
    clean_path = os.path.join(DATA_DIR, "flow_test_clean.npy")
    np.save(clean_path, flow_all[VAL_END:])
    print(f"[inject V2] 纯净测试集 → {clean_path}")

    # ── 统一标签（验证集+测试集拼接）─────────────────────────────────────────
    all_labels = np.concatenate([labels_val, labels_test], axis=0)  # (1104, N)
    all_labels_path = os.path.join(DATA_DIR, "anomaly_labels.npy")
    np.save(all_labels_path, all_labels)
    print(f"[inject V2] 统一标签（val+test）→ {all_labels_path}")

    # ── 全局摘要 ───────────────────────────────────────────────────────────
    global_summary = {
        "version": "v2_injection",
        "timestamp": datetime.now().isoformat(),
        "val_summary": summary_val,
        "test_summary": summary_test,
        "data_leakage": "compliant — 仅验证集和测试集注入，训练集保持纯净",
    }
    summary_path = os.path.join(DATA_DIR, "injection_summary_v2.json")
    with open(summary_path, "w") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    print(f"[inject V2] 全局摘要 → {summary_path}")

    print(f"\n{'='*60}")
    print("[inject V2] ✅ 注入完成！校验摘要：")
    print(f"  验证集: 占比={summary_val['actual_ratio']:.4f}, "
          f"事件={summary_val['n_events']}, "
          f"类型={summary_val['type_counts']}")
    print(f"  测试集: 占比={summary_test['actual_ratio']:.4f}, "
          f"事件={summary_test['n_events']}, "
          f"类型={summary_test['type_counts']}")
    print(f"{'='*60}\n")

    return all_labels, events_val, events_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="taxi_flow_total")
    args = parser.parse_args()
    run(args.target)
