"""Week6 评估指标计算 — 分层评估（时段 / 区域 / 连续性）

设计原则：
- 纯函数输入输出，无副作用，可在 task 1/2/3/4 复用
- 不依赖 PyTorch / Spark，符合「profiles 隔离拆分」原则
- 异常检测无 ground truth，单独提供合理性评估函数
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


# ── 分段定义 ──────────────────────────────────────────────────────────────────
# 网格坐标 → row × col
GRID_H = GRID_W = 32
N_CELLS = GRID_H * GRID_W

# 时段定义（与 week5/config.py EVAL_CFG.time_periods 一致）
# 夜间的 21-24 + 0-7 是一段，需要在外部按 24 小时 wrap
TIME_PERIODS = {
    "morning_peak": (7, 10),    # 早高峰
    "off_peak":    (10, 18),    # 平峰
    "evening_peak": (18, 21),   # 晚高峰
    "night":       (21, 32),    # 夜间（21-24 + 0-7，编码为 21~31 → 实际 21~23, 0~7）
}

# 区域定义（get_raw_flow 的核心/郊区划分）
# 核心区：8~24 行/列（避开边缘噪声）
# 郊区：其余
CORE_ROW = (8, 24)
CORE_COL = (8, 24)


def hour_of_t(t: int, val_end: int = 3288) -> int:
    """返回 t 在全局时间轴上的小时（0~23），假设 30 分钟步长无重叠，按整点取模"""
    # 注意：week5 中的步长是 30 分钟，但小时计算按 2 步 = 1 小时
    # t ∈ [VAL_END, TEST_END)，hour = (t - global_start) // 2 % 24
    # 起点 t=0 对应 2015-03-01 00:00，按这个推算
    # 实际更简单：t=0 是 hour 0，t=2 是 hour 1，依此类推
    return ((t - 0) // 2) % 24


def period_id_of_t(t: int) -> str:
    """返回 t 所属时段"""
    h = hour_of_t(t)
    if 7 <= h < 10:
        return "morning_peak"
    if 10 <= h < 18:
        return "off_peak"
    if 18 <= h < 21:
        return "evening_peak"
    return "night"


def is_core_cell(grid_id: int) -> bool:
    """网格是否属于核心区"""
    r, c = divmod(grid_id, GRID_W)
    return CORE_ROW[0] <= r < CORE_ROW[1] and CORE_COL[0] <= c < CORE_COL[1]


# ── 预测精度指标 ──────────────────────────────────────────────────────────────

def compute_predict_metrics(
    flow: np.ndarray,
    pred_scores: np.ndarray,
    timestamps: np.ndarray,
    val_end: int,
) -> Dict[str, float]:
    """计算预测误差评估指标（基于 STF 输出 + 实际流量）

    Args:
        flow: (T, N) 实际归一化人流量
        pred_scores: (T, N) STF 预测值（pred_scores_test_v2）
        timestamps: (T,) datetime64，对应 flow
        val_end: 测试集起点（用于计算全局时间 / 时段）

    Returns:
        指标字典：全局 + 时段分层 + 区域分层 + 连续性
    """
    if flow.shape != pred_scores.shape:
        raise ValueError(f"flow {flow.shape} vs pred {pred_scores.shape} 形状不一致")

    # 全局绝对误差
    abs_err = np.abs(flow - pred_scores)
    sq_err = (flow - pred_scores) ** 2

    # 分母防 0：MAPE 用 |flow|，与 0.05 阈值对比
    denom = np.maximum(np.abs(flow), 0.05)

    results = {
        # ── 全局 ──
        "overall_mae": float(abs_err.mean()),
        "overall_rmse": float(np.sqrt(sq_err.mean())),
        "overall_mape": float((abs_err / denom).mean() * 100),
        # ── 整体 ± 尺度 ──
        "flow_mean": float(flow.mean()),
        "pred_mean": float(pred_scores.mean()),
    }

    # ── 时段分层 ──
    hours = np.array([hour_of_t(val_end + t) for t in range(len(flow))])  # (T,)
    for period_name, (h_lo, h_hi) in TIME_PERIODS.items():
        # 时段 hour 跨日（night: 21~32 → 21~23, 0~7）
        if h_hi > 24:
            mask = (hours >= h_lo) | (hours < (h_hi - 24))
        else:
            mask = (hours >= h_lo) & (hours < h_hi)
        if not mask.any():
            results[f"{period_name}_mae"] = None
            continue
        results[f"{period_name}_mae"] = float(abs_err[mask].mean())
        results[f"{period_name}_rmse"] = float(np.sqrt(sq_err[mask].mean()))
        results[f"{period_name}_count"] = int(mask.sum())

    # ── 区域分层 ──
    cell_ids = np.arange(N_CELLS)
    core_mask = np.array([is_core_cell(gid) for gid in cell_ids])
    suburb_mask = ~core_mask
    if core_mask.any():
        results["core_area_mae"] = float(abs_err[:, core_mask].mean())
        results["core_area_rmse"] = float(np.sqrt(sq_err[:, core_mask].mean()))
    if suburb_mask.any():
        results["suburban_mae"] = float(abs_err[:, suburb_mask].mean())
        results["suburban_rmse"] = float(np.sqrt(sq_err[:, suburb_mask].mean()))

    # ── 时序连续性（t+1 方向准确率）────────────────────────────────────────
    # 预测方向 = pred[t+1] - pred[t]
    # 真实方向 = flow[t+1] - flow[t]
    if len(flow) >= 2:
        pred_dir = np.sign(pred_scores[1:] - pred_scores[:-1])
        true_dir = np.sign(flow[1:] - flow[:-1])
        # 完全平（=0）跳过
        nonzero = true_dir != 0
        if nonzero.any():
            agree = (pred_dir[nonzero] == true_dir[nonzero]).mean()
            results["next_step_direction_acc"] = float(agree)
        else:
            results["next_step_direction_acc"] = None

    return results


# ── 异常检测合理性评估（无 ground truth 时的替代）─────────────────────────────

def compute_anomaly_rationality(
    anomaly_mask: np.ndarray,
    scores: np.ndarray,
    timestamps: np.ndarray,
    val_end: int,
) -> Dict[str, float]:
    """异常检测合理性评估（无 ground truth，纯统计意义）

    验证：
    1. 时段合理性：夜间异常率显著低于白天
    2. 区域合理性：核心区异常密度高于郊区
    3. 连片性：异常格点在空间上有聚类倾向
    """
    assert anomaly_mask.shape == scores.shape

    # 1. 时段合理性
    hours = np.array([hour_of_t(val_end + t) for t in range(len(anomaly_mask))])
    night_mask = (hours >= 21) | (hours < 7)
    day_mask = ~night_mask

    n_anomaly = int(anomaly_mask.sum())
    n_total = int(anomaly_mask.size)
    overall_rate = n_anomaly / n_total

    day_rate = float(anomaly_mask[day_mask].mean()) if day_mask.any() else 0.0
    night_rate = float(anomaly_mask[night_mask].mean()) if night_mask.any() else 0.0
    ratio_day_night = (day_rate / night_rate) if night_rate > 0 else float("inf")

    # 2. 区域合理性
    cell_ids = np.arange(N_CELLS)
    core_mask = np.array([is_core_cell(gid) for gid in cell_ids])
    suburb_mask = ~core_mask
    core_anomaly_density = float(anomaly_mask[:, core_mask].mean()) if core_mask.any() else 0.0
    suburb_anomaly_density = float(anomaly_mask[:, suburb_mask].mean()) if suburb_mask.any() else 0.0

    # 3. 连片性：相邻异常共现率
    # 用最后一帧的 32×32 mask 算：异常格点的 4-邻接异常邻居占比
    from scipy import ndimage
    last_mask = anomaly_mask[-1].reshape(GRID_H, GRID_W) if len(anomaly_mask) > 0 else np.zeros((GRID_H, GRID_W))
    if last_mask.any():
        labeled, n_clusters = ndimage.label(last_mask)
        if n_clusters > 0:
            sizes = ndimage.sum(last_mask, labeled, range(1, n_clusters + 1))
            avg_cluster_size = float(sizes.mean())
            max_cluster_size = int(sizes.max())
        else:
            avg_cluster_size = 0.0
            max_cluster_size = 0
    else:
        avg_cluster_size = 0.0
        max_cluster_size = 0

    # 4. 分数分布合理性：高分区不多但合理
    high_score_ratio = float((scores >= 0.99).mean())

    return {
        "overall_anomaly_rate": overall_rate,
        "day_anomaly_rate": day_rate,
        "night_anomaly_rate": night_rate,
        "day_night_ratio": ratio_day_night,
        "core_anomaly_density": core_anomaly_density,
        "suburb_anomaly_density": suburb_anomaly_density,
        "core_suburb_ratio": (core_anomaly_density / suburb_anomaly_density) if suburb_anomaly_density > 0 else float("inf"),
        "avg_cluster_size": avg_cluster_size,
        "max_cluster_size": max_cluster_size,
        "high_score_ratio": high_score_ratio,
    }


# ── 事件质量分维度统计 ────────────────────────────────────────────────────────

def aggregate_events(
    events: List[dict],
    timestamps: np.ndarray,
    val_end: int,
) -> Dict[str, float]:
    """从 anomaly_events 列表汇总质量指标

    Args:
        events: list of dict, each with keys:
            t_start, t_end, duration, n_cells, avg_score, warning_level, event_type
    """
    if not events:
        return {"total_events": 0}

    n_cells = np.array([e["n_cells"] for e in events])
    duration = np.array([e["duration"] for e in events])
    avg_score = np.array([e["avg_score"] for e in events])

    # 时间分布
    t_starts = [e["t_start"] for e in events]
    hours = [hour_of_t(t) for t in t_starts]
    night_events = sum(1 for h in hours if h >= 21 or h < 7)

    # 事件类型分布
    event_types = [e.get("event_type", "unknown") for e in events]
    type_counts = {}
    for et in event_types:
        type_counts[et] = type_counts.get(et, 0) + 1

    # 等级分布（兼容缺字段的旧事件格式）
    level_counts = {1: 0, 2: 0, 3: 0}
    for e in events:
        wl = e.get("warning_level")
        if wl is None:
            # 从 n_cells + duration 推断
            if e.get("n_cells", 0) >= 20 and e.get("duration", 0) >= 3:
                wl = 3
            elif e.get("n_cells", 0) >= 20:
                wl = 2
            elif e.get("n_cells", 0) >= 16:
                wl = 1
            else:
                wl = 0
        if int(wl) in level_counts:
            level_counts[int(wl)] += 1

    # 事件平均等级用于返回
    if level_counts[1] + level_counts[2] + level_counts[3] > 0:
        avg_warning_level = (
            1 * level_counts[1] + 2 * level_counts[2] + 3 * level_counts[3]
        ) / (level_counts[1] + level_counts[2] + level_counts[3])
    else:
        avg_warning_level = 0

    return {
        "total_events": len(events),
        "events_per_day": len(events) / 25.0,  # TEST_END - VAL_END = 600 步 = 25 天
        "avg_n_cells": float(n_cells.mean()),
        "max_n_cells": int(n_cells.max()),
        "avg_duration": float(duration.mean()),
        "max_duration": int(duration.max()),
        "avg_score": float(avg_score.mean()),
        "night_event_ratio": night_events / len(events),
        "level_1_count": level_counts.get(1, 0),
        "level_2_count": level_counts.get(2, 0),
        "level_3_count": level_counts.get(3, 0),
        "type_distribution": type_counts,
    }


# ── 分类指标（当有 ground truth 时） ──────────────────────────────────────

def compute_classification_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    scores: np.ndarray = None,
) -> Dict[str, float]:
    """对异常检测的 ground truth 评估

    Args:
        pred_mask: (T, N) bool, 预测的异常掩码
        gt_mask: (T, N) bool, ground truth 异常掩码
        scores: (T, N) float, 可选——给出后算 AUC-ROC

    Returns:
        {
            "precision", "recall", "f1", "accuracy",
            "TP", "FP", "FN", "TN",
            "auc_roc" (可选),
        }
    """
    pred = pred_mask.astype(bool).ravel()
    gt = gt_mask.astype(bool).ravel()
    assert pred.shape == gt.shape, f"{pred.shape} vs {gt.shape}"

    TP = int((pred & gt).sum())
    FP = int((pred & ~gt).sum())
    FN = int((~pred & gt).sum())
    TN = int((~pred & ~gt).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
    }

    # AUC-ROC（如果有 scores）
    if scores is not None:
        try:
            from sklearn.metrics import roc_auc_score
            flat_scores = scores.ravel()
            # sklearn 要求 gt_label ∈ {0, 1}
            flat_gt = gt_mask.astype(int).ravel()
            # 过滤掉 scores 是 NaN
            valid = ~np.isnan(flat_scores)
            if valid.sum() > 0 and len(np.unique(flat_gt[valid])) > 1:
                result["auc_roc"] = float(roc_auc_score(flat_gt[valid], flat_scores[valid]))
        except Exception:
            pass

    return result


# ── 聚合入口 ──────────────────────────────────────────────────────────────────

def full_evaluation(
    flow: np.ndarray,
    pred_scores: np.ndarray,
    anomaly_mask: np.ndarray,
    fused_scores: np.ndarray,
    events: List[dict],
    timestamps: np.ndarray,
    val_end: int,
) -> Dict[str, any]:
    """一次性输出所有评估指标

    Returns:
        {
            "predict": {...},        # 预测精度
            "anomaly": {...},        # 异常检测合理性
            "events": {...},         # 事件质量
            "n_samples": int,        # 总样本数
            "val_end": int,
        }
    """
    return {
        "predict": compute_predict_metrics(flow, pred_scores, timestamps, val_end),
        "anomaly": compute_anomaly_rationality(anomaly_mask, fused_scores, timestamps, val_end),
        "events":  aggregate_events(events, timestamps, val_end),
        "n_samples": int(flow.size),
        "t_steps": int(len(flow)),
        "val_end": int(val_end),
    }


if __name__ == "__main__":
    # 调试：造 fake 数据验证流程
    print("=== metrics.py 自检 ===")
    np.random.seed(42)
    T, N = 600, 1024
    flow = np.random.rand(T, N).astype(np.float32) * 0.5
    pred = flow + np.random.randn(T, N).astype(np.float32) * 0.05
    pred = np.clip(pred, 0, 1)
    mask = np.random.rand(T, N) < 0.05
    scores = np.random.rand(T, N).astype(np.float32)
    ts = np.arange(T)

    result = full_evaluation(flow, pred, mask, scores, [], ts, val_end=3288)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
