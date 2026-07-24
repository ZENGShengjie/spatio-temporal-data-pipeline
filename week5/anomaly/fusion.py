"""融合框架 V2 — 加权投票 + 性能准入过滤 + 权重搜索 + 时空事件聚合 + 异常分类

V2 修复：
  - 性能准入门槛：F1≥0.01 且 AUC-ROC≥0.55 才纳入融合
  - 缓存命名规范化：_v2 后缀区分版本
  - 验证集/测试集标签分开存储

数据泄露红线：
  - 权重仅在验证集上搜索（网格搜索，F1 最大化）
  - 测试集仅用于最终评估
  - 得分归一化用分位数截断法
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy import ndimage

import numpy as np
import pandas as pd
from scipy import ndimage

from week5.config import (
    FUSION_CFG, CACHE_DIR, DATA_DIR,
    VAL_END, VAL_HOURS, TEST_HOURS, N_CELLS,
    cache_path, cache_json,
)


# ── 异常分类 ─────────────────────────────────────────────────────────────────

def classify_anomaly_type(
    scores: np.ndarray,
    direction: np.ndarray,
    threshold: float = 0.5,
    spatial_mask: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """根据误差方向和时空特征分类异常类型。"""
    T, N = scores.shape
    is_anomaly = scores >= threshold

    labels = np.zeros((T, N), dtype=np.int32)
    labels[is_anomaly & (direction > 0)] = 1   # 突增
    labels[is_anomaly & (direction < 0)] = 2   # 突降
    if spatial_mask is not None:
        labels[is_anomaly & spatial_mask] = 3   # 区域持续型

    return {
        "surge":    labels == 1,
        "drop":     labels == 2,
        "regional": labels == 3 if spatial_mask is not None else np.zeros_like(labels, dtype=bool),
        "any":      is_anomaly,
    }


# ── 时空事件聚合 ─────────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    event_id: int
    t_start: int
    t_end: int
    duration: int
    n_cells: int
    n_center: int
    event_type: str
    avg_score: float
    is_spatial: bool


def build_spatial_mask(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """判断每个时间步是否属于空间连片事件（4-连通）"""
    T, N = scores.shape
    is_anomaly = scores >= threshold
    H, W = 32, 32
    spatial_mask = np.zeros_like(is_anomaly)

    struct = ndimage.generate_binary_structure(2, 1)
    for t in range(T):
        labeled, n = ndimage.label(is_anomaly[t].reshape(H, W), structure=struct)
        for label_id in range(1, n + 1):
            component = labeled == label_id
            if component.sum() > 1:
                spatial_mask[t, component.flatten()] = True
    return spatial_mask


def aggregate_spatial_events_fast(
    scores: np.ndarray,
    threshold: float = 0.5,
    time_gap_max: int = 1,
) -> List[AnomalyEvent]:
    """3D 连通域聚合：时空相连的异常点合并为事件"""
    T, N = scores.shape
    is_anomaly = scores >= threshold
    H, W = 32, 32

    anomaly_3d = is_anomaly.reshape(T, H, W)
    struct_3d = np.zeros((3, 3, 3), dtype=bool)
    struct_3d[1, :, :] = True  # 同帧全连通

    labeled_3d, n_events = ndimage.label(anomaly_3d, structure=struct_3d)

    events = []
    for eid in range(1, n_events + 1):
        mask_3d = labeled_3d == eid
        mask_2d = mask_3d.reshape(T, H * W)
        t_indices, r_indices, c_indices = np.where(mask_3d)
        t_start, t_end = int(t_indices.min()), int(t_indices.max())
        duration = t_end - t_start + 1
        n_cells = len(t_indices)
        n_center = r_indices[len(r_indices) // 2] * W + c_indices[len(c_indices) // 2]
        avg_score = float(scores[mask_2d].mean())
        is_spatial = (n_cells > 1) or (duration > 3)

        events.append(AnomalyEvent(
            event_id=eid - 1,
            t_start=t_start, t_end=t_end, duration=duration,
            n_cells=n_cells, n_center=int(n_center),
            event_type="spatial_sustained",
            avg_score=avg_score, is_spatial=is_spatial,
        ))
    print(f"[fusion] {n_events} events from {is_anomaly.sum()} anomaly points")
    return events


# ── 得分归一化 ────────────────────────────────────────────────────────────────

def normalize_scores(scores: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """分位数截断 + min-max 归一化到 [0,1]"""
    q99 = np.percentile(scores, quantile * 100)
    q00 = scores.min()
    r = q99 - q00
    if r < 1e-9:
        return np.zeros_like(scores)
    return np.clip((scores - q00) / r, 0, 1).astype(np.float32)


# ── 权重搜索 ──────────────────────────────────────────────────────────────────

def search_weights(
    score_dict: Dict[str, np.ndarray],
    true_labels: np.ndarray,
    step: float = 0.05,
) -> Tuple[Dict[str, float], float]:
    """暴力网格搜索最优融合权重（在验证集上 F1 最大化）"""
    methods = list(score_dict.keys())
    n = len(methods)

    steps = np.arange(0, 1.0 + step / 2, step)
    all_weights = [w for w in product(steps, repeat=n)
                   if abs(sum(w) - 1.0) < step / 2]
    print(f"[fusion] searching {len(all_weights)} weight combos × 17 thresholds")

    best_f1 = 0.0
    best_weights = None

    for w in all_weights:
        fused = np.zeros_like(true_labels, dtype=np.float32)
        for mi, method in enumerate(methods):
            fused += w[mi] * normalize_scores(score_dict[method])

        for thresh in np.arange(0.1, 0.95, 0.05):
            pred = (fused >= thresh).astype(int)
            tp = int(((pred == 1) & (true_labels == 1)).sum())
            fp = int(((pred == 1) & (true_labels == 0)).sum())
            fn = int(((pred == 0) & (true_labels == 1)).sum())
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = {m: float(w[mi]) for mi, m in enumerate(methods)}

    print(f"[fusion] best weights: {best_weights}, val F1={best_f1:.4f}")
    return best_weights, best_f1


# ── 融合主类 ────────────────────────────────────────────────────────────────

class AnomalyFusion:
    """统一融合检测接口 V2"""

    def __init__(self, target: str = "taxi_flow_total"):
        self.target = target
        self.weights: Optional[Dict[str, float]] = None
        self.threshold: float = FUSION_CFG.decision_threshold
        self.score_dict: Dict[str, np.ndarray] = {}
        self.events: List[AnomalyEvent] = []
        self.classification: Dict[str, np.ndarray] = {}
        self._fitted = False

    def load_all_scores(self, split: str = "val"):
        """加载所有方法的得分（V2 缓存命名）。

        V2 命名规范（cache_path 自动加 .npy 后缀）：
          stat_scores_val_v2.npy
          pred_scores_val_v2.npy
          vae_scores_val.npy
          tae_scores_val_v2.npy
        """
        score_names = {
            "statistical": f"stat_scores_{split}_v2",
            "prediction":  f"pred_scores_{split}_v2",
            "vae":         f"vae_scores_{split}",
            "transformer": f"tae_scores_{split}_v2",
        }
        score_names_v1 = {
            "statistical": f"stat_scores_{split}",
            "prediction":  f"pred_scores_{split}",
            "vae":         f"vae_scores_{split}",
            "transformer": f"tae_scores_{split}",
        }

        loaded = {}
        for method, v2_name in score_names.items():
            v1_name = score_names_v1[method]
            path = cache_path(v2_name)  # cache_path already adds .npy
            if not os.path.exists(path):
                path = cache_path(v1_name)
            if os.path.exists(path):
                self.score_dict[method] = np.load(path)
                loaded[method] = path
            else:
                print(f"[fusion] WARNING: no scores for {method} ({split})")
        print(f"[fusion] loaded: {list(loaded.keys())}")

    def load_test_scores(self):
        """加载测试集得分"""
        score_names = {
            "statistical": "stat_scores_test_v2",
            "prediction":  "pred_scores_test_v2",
            "vae":         "vae_scores_test",
            "transformer": "tae_scores_test_v2",
        }
        for method, name in score_names.items():
            path = cache_path(name)
            if os.path.exists(path):
                self.score_dict[method] = np.load(path)
                print(f"[fusion] test scores: {method}")
            else:
                print(f"[fusion] WARNING: missing {method}")

    def fit(self, true_labels_val: np.ndarray = None,
             perf_gate_f1: float = 0.01, perf_gate_auc: float = 0.55):
        """权重搜索 + 性能门槛过滤"""
        self.load_all_scores("val")

        if len(self.score_dict) == 0:
            print("[fusion] ERROR: no scores loaded")
            return

        if true_labels_val is None:
            self.weights = dict(FUSION_CFG.init_weights)
            self.threshold = 0.5
            print("[fusion] no val labels, using default weights")
            self._fitted = True
            self._save()
            return

        # ── 性能门槛过滤 ──────────────────────────────────────────────────────
        from sklearn.metrics import roc_auc_score, f1_score
        qualified = {}
        for method, scores in self.score_dict.items():
            scores_flat = scores.flatten()
            labels_flat = true_labels_val.flatten()
            try:
                auc = roc_auc_score(labels_flat, scores_flat)
            except ValueError:
                auc = 0.5
            # 用每个方法自己的决策阈值（如 percentile 95）来算 F1
            threshold_q = float(np.percentile(scores, 95))
            pred_binary = (scores > threshold_q).astype(int)
            try:
                f1 = f1_score(labels_flat, pred_binary)
            except ValueError:
                f1 = 0.0

            ok = auc >= perf_gate_auc and f1 >= perf_gate_f1
            print(f"[fusion] {method:15s} val_auc={auc:.4f} val_f1={f1:.4f} "
                  f"{'✓' if ok else '✗'}")
            if ok:
                qualified[method] = scores

        if not qualified:
            print("[fusion] WARNING: no methods passed gate, using all")
            qualified = self.score_dict

        self.score_dict = qualified
        self.weights, best_f1 = search_weights(
            self.score_dict, true_labels_val,
            step=FUSION_CFG.weight_search_step,
        )
        self._fitted = True
        self._save()

    def predict(self, scores_test: Dict[str, np.ndarray] = None,
                threshold_override: float = None
                ) -> Tuple[np.ndarray, np.ndarray, List[AnomalyEvent]]:
        """在测试集上融合预测"""
        if scores_test:
            self.score_dict = scores_test
        else:
            self.load_test_scores()

        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        # 加权融合（仅纳入训练时达标的方法）
        T, N = list(self.score_dict.values())[0].shape
        fused = np.zeros((T, N), dtype=np.float32)
        for method, w in self.weights.items():
            if method in self.score_dict:
                fused += w * normalize_scores(
                    self.score_dict[method], quantile=FUSION_CFG.score_quantile)

        thresh = threshold_override if threshold_override is not None else self.threshold
        mask = (fused >= thresh).astype(bool)

        events = aggregate_spatial_events_fast(fused, threshold=thresh)

        direction_path = cache_path("pred_direction_test_v2")
        direction = np.zeros((T, N), dtype=np.float32)
        if os.path.exists(direction_path):
            direction = np.load(direction_path)
        spatial_mask = build_spatial_mask(fused, threshold=thresh)
        self.classification = classify_anomaly_type(fused, direction, thresh, spatial_mask)
        self.events = events

        np.save(cache_path("fusion_scores_test_v2"), fused)
        np.save(cache_path("fusion_mask_test_v2"), mask)
        return mask, fused, events

    def detect_anomalies(self, input_sequence: np.ndarray) -> Dict:
        """对外统一接口"""
        from anomaly.prediction import PredictionAnomalyDetector
        from anomaly.statistical import StatisticalAnomalyDetector

        det_pred = PredictionAnomalyDetector(target=self.target)
        det_pred.fit()
        scores_pred, _ = det_pred.predict_scores(input_sequence)

        det_stat = StatisticalAnomalyDetector(target=self.target)
        det_stat.fit()
        scores_stat = det_stat.predict_scores(input_sequence)

        fused = (FUSION_CFG.init_weights.get("prediction", 0.5) * normalize_scores(scores_pred) +
                 FUSION_CFG.init_weights.get("statistical", 0.5) * normalize_scores(scores_stat))
        mask = fused >= self.threshold
        events = aggregate_spatial_events_fast(fused, threshold=self.threshold)
        classification = classify_anomaly_type(
            fused, np.zeros_like(fused), self.threshold,
            build_spatial_mask(fused, self.threshold))

        return {
            "anomaly_mask": mask,
            "fused_scores": fused,
            "events": events,
            "classification": classification,
        }

    def _save(self):
        params = {
            "weights": self.weights,
            "threshold": self.threshold,
            "target": self.target,
        }
        with open(cache_json("fusion_params_v2"), "w") as f:
            json.dump(params, f, indent=2)


# ── 便捷入口 ─────────────────────────────────────────────────────────────────

def run(target: str = "taxi_flow_total") -> Tuple[np.ndarray, np.ndarray, List[AnomalyEvent]]:
    """单步运行"""
    # 加载验证集真值（V2：分开存 val 和 test）
    val_labels_path = os.path.join(DATA_DIR, "anomaly_labels_val.npy")
    if os.path.exists(val_labels_path):
        true_val = np.load(val_labels_path)
    elif os.path.exists(os.path.join(DATA_DIR, "anomaly_labels.npy")):
        all_labels = np.load(os.path.join(DATA_DIR, "anomaly_labels.npy"))
        true_val = all_labels[:VAL_HOURS]
    else:
        true_val = None

    fusion = AnomalyFusion(target=target)
    fusion.fit(true_labels_val=true_val)
    mask_test, scores_test, events = fusion.predict()

    if events:
        rows = [{
            "event_id": e.event_id, "t_start": e.t_start, "t_end": e.t_end,
            "duration": e.duration, "n_cells": e.n_cells, "n_center": e.n_center,
            "event_type": e.event_type, "avg_score": round(e.avg_score, 4),
            "is_spatial": e.is_spatial,
        } for e in events]
        pd.DataFrame(rows).to_csv(
            os.path.join(DATA_DIR, "detected_events.csv"), index=False)

    print(f"[fusion] test: {mask_test.sum()} anomaly steps, {len(events)} events")
    return mask_test, scores_test, events


if __name__ == "__main__":
    run()
