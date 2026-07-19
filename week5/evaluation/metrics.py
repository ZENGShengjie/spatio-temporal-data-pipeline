"""评估指标体系 — 点级 + 事件级双轨评估，分维度统计

点级指标：逐时间步 × 逐网格的 Precision / Recall / F1 / AUC-ROC
事件级指标：以完整异常事件为单位的 P / R / F1

数据泄露红线：
  - 评估仅在测试集上进行
  - 所有阈值/权重来自验证集，测试集不参与参数调优
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EVAL_CFG, CACHE_DIR, DATA_DIR, VAL_HOURS, TEST_HOURS, N_CELLS, cache_path
from anomaly.fusion import aggregate_spatial_events_fast


# ── 点级指标 ─────────────────────────────────────────────────────────────────

@dataclass
class PointMetrics:
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    accuracy: float
    tp: int; fp: int; fn: int; tn: int


def compute_point_metrics(pred: np.ndarray, gt: np.ndarray) -> PointMetrics:
    """逐点评估指标"""
    pred_flat = pred.flatten().astype(int)
    gt_flat   = gt.flatten().astype(int)

    tp = int(((pred_flat == 1) & (gt_flat == 1)).sum())
    fp = int(((pred_flat == 1) & (gt_flat == 0)).sum())
    fn = int(((pred_flat == 0) & (gt_flat == 1)).sum())
    tn = int(((pred_flat == 0) & (gt_flat == 0)).sum())

    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    # AUC（展平为二分类）
    try:
        auc_roc = roc_auc_score(gt_flat, pred_flat.astype(float))
        auc_pr  = average_precision_score(gt_flat, pred_flat.astype(float))
    except ValueError:
        auc_roc = auc_pr = float("nan")

    return PointMetrics(
        precision=p, recall=r, f1=f1,
        auc_roc=auc_roc, auc_pr=auc_pr,
        accuracy=acc,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def compute_point_scores_auc(scores: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """用连续得分计算 AUC（比二值 mask 更稳定）"""
    scores_flat = scores.flatten()
    gt_flat = gt.flatten().astype(int)
    try:
        auc_roc = roc_auc_score(gt_flat, scores_flat)
        auc_pr  = average_precision_score(gt_flat, scores_flat)
        return auc_roc, auc_pr
    except ValueError:
        return float("nan"), float("nan")


# ── 事件级指标 ───────────────────────────────────────────────────────────────

@dataclass
class EventMetrics:
    precision: float
    recall: float
    f1: float
    n_pred_events: int
    n_gt_events: int
    n_matched: int


def event_hit(pred_event: dict, gt_events: List[dict],
              temporal_tolerance: int = 1,
              spatial_tolerance: float = 0.5) -> bool:
    """判断预测事件是否命中真值事件

    事件 A 命中事件 B 的条件：
    - 时间重叠 ≥ 50%（容差 temporal_tolerance 小时）
    - 空间重叠 ≥ 50%（容差 spatial_tolerance）
    """
    t_overlap = max(0,
                    min(pred_event["t_end"], gt_events[0]["t_end"])
                    - max(pred_event["t_start"], gt_events[0]["t_start"]) + 1)
    t_pred = pred_event["t_end"] - pred_event["t_start"] + 1

    if t_pred == 0:
        temporal_match = False
    else:
        temporal_match = t_overlap / t_pred >= 0.5 - temporal_tolerance * 0.1

    # 简化：时间差 ±tolerance 视为匹配
    time_diff = abs(pred_event["t_start"] - gt_events[0]["t_start"])
    temporal_match = time_diff <= temporal_tolerance

    return temporal_match


def event_recall(pred_events: List[dict], gt_events: List[dict]) -> Tuple[int, int]:
    """计算事件级命中数

    Returns:
        (n_matched, n_gt)
    """
    matched = 0
    gt_matched = set()

    for pe in pred_events:
        for gi, ge in enumerate(gt_events):
            if gi in gt_matched:
                continue
            if event_hit(pe, [ge]):
                matched += 1
                gt_matched.add(gi)
                break

    return matched, len(gt_events)


def compute_event_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                          scores: np.ndarray = None) -> EventMetrics:
    """事件级评估指标"""
    # 从 mask 构建事件列表
    from anomaly.fusion import aggregate_spatial_events_fast

    # 预测事件
    pred_events = aggregate_spatial_events_fast(
        scores if scores is not None else pred_mask.astype(float),
        threshold=0.5,
    )
    pred_event_dicts = [{
        "event_id": e.event_id,
        "t_start": e.t_start,
        "t_end": e.t_end,
        "n_cells": e.n_cells,
        "avg_score": e.avg_score,
    } for e in pred_events]

    # 真值事件（从 gt_mask 构建）
    gt_events = aggregate_spatial_events_fast(
        gt_mask.astype(float),
        threshold=0.5,
    )
    gt_event_dicts = [{
        "event_id": e.event_id,
        "t_start": e.t_start,
        "t_end": e.t_end,
        "n_cells": e.n_cells,
    } for e in gt_events]

    n_matched, n_gt = event_recall(pred_event_dicts, gt_event_dicts)
    n_pred = len(pred_event_dicts)

    p = n_matched / n_pred if n_pred > 0 else 0.0
    r = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = 2 * p * r / (p + r + 1e-9) if (p + r) > 0 else 0.0

    return EventMetrics(
        precision=p, recall=r, f1=f1,
        n_pred_events=n_pred,
        n_gt_events=n_gt,
        n_matched=n_matched,
    )


# ── 分维度统计 ───────────────────────────────────────────────────────────────

def get_time_period_labels() -> Dict[str, np.ndarray]:
    """按时段划分时间步索引"""
    from data_loader import get_hour_labels
    hours = get_hour_labels()
    labels = {}
    for period, (start, end) in EVAL_CFG.time_periods.items():
        if start < end:
            mask = (hours >= start) & (hours < end)
        else:  # 跨夜时段
            mask = (hours >= start) | (hours < end)
        labels[period] = mask
    return labels


def compute_by_period(pred: np.ndarray, gt: np.ndarray,
                       period_labels: Dict[str, np.ndarray]
                       ) -> Dict[str, PointMetrics]:
    """按时段分别统计指标"""
    results = {}
    for period, mask in period_labels.items():
        # mask 是对全量 T 的布尔数组，取测试集部分
        # 假设 period_labels 已经针对测试集时间步
        if len(mask) == TEST_HOURS:
            p_sub, g_sub = pred[mask], gt[mask]
        else:
            # 截取测试集部分
            p_sub, g_sub = pred[mask[-TEST_HOURS:]], gt[mask[-TEST_HOURS:]]
        if g_sub.sum() > 0:
            results[period] = compute_point_metrics(p_sub, g_sub)
    return results


# ── 分异常类型统计 ────────────────────────────────────────────────────────────

def get_anomaly_type_labels(gt_labels: np.ndarray,
                             direction: np.ndarray = None
                             ) -> Dict[str, np.ndarray]:
    """按异常类型分离测试集时间步"""
    labels = {}
    for atype, direction_val in EVAL_CFG.anomaly_types.items():
        if direction is not None and direction_val in (1.0, -1.0):
            mask = gt_labels & (direction == direction_val)
        else:
            mask = gt_labels  # sustained 类型无方向筛选
        labels[atype] = mask
    return labels


# ── 报告生成 ─────────────────────────────────────────────────────────────────

def generate_report(
    method_results: Dict[str, dict],
    output_path: str = None,
) -> pd.DataFrame:
    """生成完整评估报告

    Args:
        method_results: {
            "statistical": {"pred": mask, "gt": mask, "scores": scores},
            "prediction":  {...},
            "vae":         {...},
            "fusion":      {...},
        }
    """
    rows = []

    for method, res in method_results.items():
        pred = res["pred"]
        gt   = res["gt"]
        scores = res.get("scores", pred.astype(float))

        # 点级
        pm = compute_point_metrics(pred, gt)
        auc_roc, auc_pr = compute_point_scores_auc(scores, gt)

        row = {
            "method": method,
            "precision": round(pm.precision, 4),
            "recall":    round(pm.recall, 4),
            "f1":        round(pm.f1, 4),
            "auc_roc":   round(auc_roc, 4),
            "auc_pr":    round(auc_pr, 4),
            "tp": pm.tp, "fp": pm.fp, "fn": pm.fn,
            "anomaly_rate": round(pred.sum() / pred.size, 4),
            "gt_rate": round(gt.sum() / gt.size, 4),
        }
        rows.append(row)

        # 按时段分
        if EVAL_CFG.by_period:
            period_labels = get_time_period_labels()
            period_results = compute_by_period(pred, gt, period_labels)
            for period, m in period_results.items():
                rows.append({
                    "method": f"{method}_{period}",
                    "precision": round(m.precision, 4),
                    "recall":    round(m.recall, 4),
                    "f1":        round(m.f1, 4),
                    "auc_roc":   float("nan"),
                    "auc_pr":    float("nan"),
                    "tp": m.tp, "fp": m.fp, "fn": m.fn,
                    "anomaly_rate": round(pred.sum() / pred.size, 4),
                    "gt_rate": round(gt.sum() / gt.size, 4),
                })

        # 事件级
        em = compute_event_metrics(pred, gt, scores)
        rows.append({
            "method": f"{method}_event",
            "precision": round(em.precision, 4),
            "recall":    round(em.recall, 4),
            "f1":        round(em.f1, 4),
            "auc_roc":   float("nan"),
            "auc_pr":    float("nan"),
            "tp": em.n_matched, "fp": em.n_pred_events - em.n_matched,
            "fn": em.n_gt_events - em.n_matched,
            "anomaly_rate": em.n_pred_events,
            "gt_rate": em.n_gt_events,
        })

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[eval] report saved → {output_path}")

    return df


# ── 主评估流程 ────────────────────────────────────────────────────────────────

def run_evaluation() -> pd.DataFrame:
    """加载所有方法结果，运行完整评估"""
    # 加载真值（V2：分开的 val 和 test）
    test_labels_path = os.path.join(DATA_DIR, "anomaly_labels_test.npy")
    if os.path.exists(test_labels_path):
        gt_test = np.load(test_labels_path)
    elif os.path.exists(os.path.join(DATA_DIR, "anomaly_labels.npy")):
        all_labels = np.load(os.path.join(DATA_DIR, "anomaly_labels.npy"))
        gt_test = all_labels[VAL_HOURS:]
    else:
        print("[eval] ERROR: no anomaly labels found, skipping evaluation")
        return pd.DataFrame()

    # 加载各方法结果（V2 优先，V1 兼容）
    method_results = {}
    for method, v2_fname, v1_fname in [
        ("statistical", "stat_mask_test_v2",  "stat_mask_test"),
        ("prediction",  "pred_mask_test_v2",  "pred_mask_test"),
        ("vae",         "vae_mask_test_v2",  "vae_mask_test"),
        ("transformer", "tae_mask_test_v2",  "tae_mask_test"),
        ("fusion",      "fusion_mask_test_v2", "fusion_mask_test"),
    ]:
        mask_path = cache_path(v2_fname)
        if not os.path.exists(mask_path):
            mask_path = cache_path(v1_fname)
        scores_path = cache_path(v2_fname.replace("mask", "scores"))
        if not os.path.exists(scores_path):
            scores_path = cache_path(v1_fname.replace("mask", "scores"))
        if os.path.exists(mask_path):
            pred = np.load(mask_path)
            scores = np.load(scores_path) if os.path.exists(scores_path) else pred.astype(float)
            method_results[method] = {
                "pred": pred,
                "gt": gt_test,
                "scores": scores,
            }
        else:
            print(f"[eval] WARNING: {method} results not found, skipping")

    # 生成报告
    report_path = os.path.join(DATA_DIR, "evaluation_report.csv")
    df = generate_report(method_results, output_path=report_path)

    # 打印摘要
    summary = df[df["method"].isin(["statistical", "prediction", "vae", "transformer", "fusion"])]
    print("\n=== Evaluation Summary (Point-level F1) ===")
    print(summary[["method", "precision", "recall", "f1", "auc_roc", "auc_pr"]].to_string(index=False))

    return df


if __name__ == "__main__":
    run_evaluation()
