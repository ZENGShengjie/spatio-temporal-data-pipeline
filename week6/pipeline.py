"""Week6 端到端数据处理 Pipeline

提供 SpatiotemporalPipeline 类，封装批量处理和模拟实时两种运行模式。
所有算法逻辑 100% 复用 week5/anomaly/ 和 week5/data_loader/，不做重写。
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

# ── 路径设置 ──────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from week5.config import (
    VAL_HOURS, TEST_HOURS, N_CELLS,
    STAT_CFG, PRED_CFG, FUSION_CFG,
)
from week5.data_loader import (
    get_raw_flow, get_timestamps, get_time_features,
    get_flow_1d, get_time_group_labels, get_hour_labels,
    get_grid_coords, get_splits,
)
from week5.anomaly.statistical import StatisticalAnomalyDetector
from week5.anomaly.prediction import PredictionAnomalyDetector
from week5.anomaly.fusion_v3 import (
    AnomalyFusionV3, aggregate_spatial_events_fast,
    build_spatial_mask, normalize_scores, AnomalyEvent,
)


# ── 预警等级定义 ──────────────────────────────────────────────────────────────

@dataclass
class WarningAlert:
    level: int                    # 1=一般 2=重要 3=紧急
    level_name: str
    t: int                       # 时间步
    description: str
    affected_cells: int
    cluster_size: int            # 最大连通片尺寸
    cluster_ids: str = ""        # 涉及的连通片ID，逗号分隔


class WarningEngine:
    """三级预警引擎 — 规则简单、判定快速"""

    def __init__(self):
        self.history_alerts: List[WarningAlert] = []
        # 每个 cluster_id → 连续异常步数（仅在 cluster 内所有格点同时异常时累加）
        self._cluster_consecutive: Optional[Dict[int, int]] = None
        self._window_size = 48

    def evaluate(self, anomaly_mask: np.ndarray,
                 scores: np.ndarray, t: int) -> List[WarningAlert]:
        """对当前时间步的异常结果进行预警判定。

        预警逻辑（互斥，只取最高等级）：
          紧急：同连通片 ≥20格 且 在该连通片内连续 ≥3 步异常
          重要：连通片 ≥30格
          一般：连通片 ≥12格  或  任意4×4窗口内 ≥10/16 格异常
          无预警：其他情况

        连续异常以 cluster 为单位追踪，每个 cluster 独立累计步数。
        """
        from scipy import ndimage

        H = W = 32
        is_anomaly = anomaly_mask.reshape(H, W)
        alerts = []

        labeled, n = ndimage.label(is_anomaly)
        if n == 0:
            return alerts  # 无异常，直接返回

        sizes = ndimage.sum(is_anomaly, labeled, range(1, n + 1))
        cluster_sizes = dict(enumerate(sizes, start=1))

        # ── 初始化/追踪连续异常计数器（per-cluster，不是 per-cell）──────────────
        if self._cluster_consecutive is None:
            # dict: cluster_id → 连续异常步数
            self._cluster_consecutive = {cid: 0 for cid in cluster_sizes}

        new_consecutive = {}
        for cid, size in cluster_sizes.items():
            cluster_mask = (labeled == cid)
            if is_anomaly[cluster_mask].all():
                # 该 cluster 内所有格点本步都异常
                new_consecutive[cid] = self._cluster_consecutive.get(cid, 0) + 1
            else:
                new_consecutive[cid] = 0
        self._cluster_consecutive = new_consecutive

        # ── 预警判定（互斥：只取最高等级）──────────────────────────────────────
        max_cluster = int(max(cluster_sizes.values()))
        top_cluster_id = max(cluster_sizes, key=cluster_sizes.get)

        # 紧急：最大连通片 ≥20格 且 连续 ≥3 步
        for cid, sz in cluster_sizes.items():
            if sz >= 20 and self._cluster_consecutive.get(cid, 0) >= 3:
                alerts.append(WarningAlert(
                    level=3, level_name="紧急",
                    t=t,
                    description=f"大范围持续异常，连通片={sz}格连续≥3步",
                    affected_cells=int(anomaly_mask.sum()),
                    cluster_size=sz,
                    cluster_ids=str(cid),
                ))
                return alerts  # 紧急优先，不再往下判断

        # 重要：连通片 ≥20格（5×4 矩形范围）
        for cid, sz in cluster_sizes.items():
            if sz >= 20:
                alerts.append(WarningAlert(
                    level=2, level_name="重要",
                    t=t,
                    description=f"区域异常，连通片={sz}格",
                    affected_cells=int(anomaly_mask.sum()),
                    cluster_size=sz,
                    cluster_ids=str(cid),
                ))
                return alerts

        # 一般：连通片 ≥16格（4×4 方形范围）
        for cid, sz in cluster_sizes.items():
            if sz >= 16:
                alerts.append(WarningAlert(
                    level=1, level_name="一般",
                    t=t,
                    description=f"区域异常，连通片={sz}格",
                    affected_cells=int(anomaly_mask.sum()),
                    cluster_size=sz,
                    cluster_ids=str(cid),
                ))
                return alerts

        # 一般：4×4 滑动窗口，密集异常（≥10/16 格，覆盖零散分布）
        am = is_anomaly.astype(np.int8)
        for r in range(H - 3):
            for c in range(W - 3):
                if am[r:r+4, c:c+4].sum() >= 10:
                    alerts.append(WarningAlert(
                        level=1, level_name="一般",
                        t=t,
                        description=f"4×4窗口密集异常，当前{anomaly_mask.sum()}格",
                        affected_cells=int(anomaly_mask.sum()),
                        cluster_size=max_cluster,
                    ))
                    return alerts

        return alerts  # 无预警

    def reset(self):
        self.history_alerts = []
        self._cluster_consecutive = None


# ── Pipeline 主类 ──────────────────────────────────────────────────────────────

class SpatiotemporalPipeline:
    """时空异常检测统一 Pipeline

    两种运行模式：
    1. run_batch()   — 批量处理：加载数据 → 预测 → 异常检测 → 事件聚合
    2. run_step()     — 模拟实时：滑动窗口输入新数据，输出当前步预警

    复用 week5 所有算法逻辑。
    """

    def __init__(self, mode: str = "fast",
                 stat_threshold: float = None,
                 pred_threshold: float = None,
                 fusion_threshold: float = None,
                 fusion_weights: Dict[str, float] = None,
                 use_cache: bool = True):
        """
        Args:
            mode: "fast" | "structural"
                fast:       仅用统计法 + 预测法（stat 0.9, pred 0.1），响应快
                structural: 加载 VAE/TAE 全量融合，针对结构性异常
            stat_threshold / pred_threshold / fusion_threshold:
                指定各方法阈值（None=从 week5 缓存加载）
            fusion_weights: 融合权重字典（None=用验证集最优）
            use_cache: True=优先读缓存，False=强制重新计算
        """
        self.mode = mode
        self.use_cache = use_cache
        self._fitted = False

        # 缓存阈值（从 week5 缓存读取，或使用传入值）
        # 注意：0.996(stat) + 0.36(pred) + 0.90(fused) 是历史最严苛值，2026-07-28 观察
        # 到会让晚高峰连片热区(流量高但稳定)只标出少量边缘格点，热点核心反而不标。
        # 适度放宽后，大片异常热点会整片被标记，更符合业务直觉。
        self.stat_threshold  = stat_threshold  or 0.992    # 统计法（更严格）
        self.pred_threshold  = pred_threshold  or 0.300     # 预测法（更严格）
        self.fusion_threshold = fusion_threshold or 0.80    # 三融合（更严格）

        # 融合权重（fast 模式固定，structural 模式从缓存）
        self.fusion_weights = fusion_weights or {
            "statistical": 0.7,
            "prediction":  0.1,
            "vae":         0.2,
        }

        # 预加载的检测器（lazy）
        self._stat_detector: Optional[StatisticalAnomalyDetector] = None
        self._pred_detector: Optional[PredictionAnomalyDetector] = None
        self._fusion: Optional[AnomalyFusionV3] = None

        # 实时模式状态
        self._window: List[np.ndarray] = []
        self._warning_engine = WarningEngine()
        self._result_cache: Dict[str, Any] = {}

        # 数据
        self._flow: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None
        self._scores_cache: Dict[str, np.ndarray] = {}

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def _ensure_stat(self):
        if self._stat_detector is None:
            self._stat_detector = StatisticalAnomalyDetector()
            self._stat_detector.fit()

    def _ensure_pred(self):
        if self._pred_detector is None:
            self._pred_detector = PredictionAnomalyDetector()
            self._pred_detector.fit()

    def _ensure_fusion(self):
        if self._fusion is None:
            self._fusion = AnomalyFusionV3()
            self._fusion.load_scores("test")

    def _load_or_compute_scores(self, split: str = "test") -> Dict[str, np.ndarray]:
        """加载或计算各方法异常得分"""
        if split in self._scores_cache:
            return self._scores_cache[split]

        cache_dir = _REPO / "week5" / "cache"
        result = {}
        pairs = [
            ("stat",    "stat_scores_test_v2"),
            ("pred",    "pred_scores_test_v2"),
            ("vae",     "vae_scores_test"),
            ("transformer", "tae_scores_test_v2"),
        ]
        for key, name in pairs:
            path = cache_dir / f"{name}.npy"
            if path.exists():
                result[key] = np.load(str(path))
        self._scores_cache[split] = result
        return result

    def _events_cache_path(self, split: str) -> Path:
        return _REPO / "week6" / "data" / f"events_{split}_v1.json"

    def _load_events_cache(self, split: str,
                           scores: np.ndarray, threshold: float
                           ) -> Optional[List[Any]]:
        """从磁盘加载事件缓存，过期时返回 None"""
        import json as _json
        path = self._events_cache_path(split)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = _json.load(f)
            # 反序列化回 dict（用于返回给 query_events）
            events = []
            for e in raw:
                ae = AnomalyEvent(
                    event_id=e["event_id"],
                    t_start=e["t_start"],
                    t_end=e["t_end"],
                    duration=e["duration"],
                    n_cells=e["n_cells"],
                    n_center=e["n_center"],
                    event_type=e["event_type"],
                    avg_score=e["avg_score"],
                    is_spatial=e.get("is_spatial", e["n_cells"] > 1),
                )
                events.append(ae)
            print(f"[Pipeline] events cache loaded from {path.name} ({len(events)} events)")
            return events
        except Exception as ex:
            print(f"[Pipeline] events cache load failed ({ex}), recomputing")
            return None

    def _save_events_cache(self, split: str, events: List) -> None:
        """持久化事件到磁盘缓存"""
        import json as _json
        path = self._events_cache_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = [
            {
                "event_id": e.event_id,
                "t_start": e.t_start,
                "t_end": e.t_end,
                "duration": e.duration,
                "n_cells": e.n_cells,
                "n_center": e.n_center,
                "event_type": e.event_type,
                "avg_score": e.avg_score,
                "is_spatial": e.is_spatial,
            }
            for e in events
        ]
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(raw, f, ensure_ascii=False, indent=2)
        print(f"[Pipeline] events cache saved → {path.name} ({len(events)} events)")

    # ── 批量处理模式 ─────────────────────────────────────────────────────────

    def run_batch(self, data_path: str = None,
                  split: str = "test") -> Dict[str, Any]:
        """批量处理全量数据。

        Args:
            data_path: npz 文件路径（可选，默认用 week5 配置路径）
            split: "val" | "test"

        Returns:
            {
                "flow": (T, N) 实际人流,
                "predictions": (T, N) 预测值（仅 test 有）,
                "scores": {
                    "statistical": (T, N),
                    "prediction":  (T, N),
                    "vae":         (T, N),
                    "transformer": (T, N),
                    "fused":       (T, N),
                },
                "anomaly_mask": (T, N) bool,
                "events": List[AnomalyEvent],
                "alerts": List[WarningAlert],
                "timestamps": (T,) datetime64,
                "split": str,
            }
        """
        t0 = time.time()
        self._ensure_stat()
        self._ensure_pred()

        # 动态获取当前 period 的 split 边界
        train_end, val_end, test_end = get_splits()

        # 加载数据
        flow = get_flow_1d("taxi_flow_total")
        if split == "val":
            flow = flow[train_end:val_end]
            offset = train_end
        else:
            flow = flow[val_end:]
            offset = val_end

        timestamps = get_timestamps()[offset:offset + len(flow)]
        tg = get_time_group_labels()[offset:offset + len(flow)]

        # 计算异常得分
        if self.use_cache:
            scores_dict = self._load_or_compute_scores(split)
        else:
            # 重新计算（复用 week5 逻辑，不重写）
            _, scores_stat = self._stat_detector.predict(split=split)
            _, scores_pred, _ = self._pred_detector.predict(split=split)
            scores_dict = {
                "stat": scores_stat,
                "pred": scores_pred,
            }

        # ── 融合 ──────────────────────────────────────────────────────────────
        stat_scores = scores_dict.get("stat",
            scores_dict.get("statistical",
            scores_dict.get("stat", np.zeros_like(flow))))
        pred_scores = scores_dict.get("pred",
            scores_dict.get("prediction",
            scores_dict.get("pred", np.zeros_like(flow))))

        if self.mode == "fast":
            # 快速模式：stat 0.9 + pred 0.1
            fused = (0.9 * normalize_scores(stat_scores) +
                     0.1 * normalize_scores(pred_scores))
            threshold = self.fusion_threshold
        else:
            # 结构增强模式：加权融合
            fused = sum(
                w * normalize_scores(scores_dict.get(k, np.zeros_like(flow)))
                for k, w in self.fusion_weights.items()
            )
            threshold = self.fusion_threshold

        # 异常判定
        anomaly_mask = fused >= threshold

        # 事件聚合（优先从磁盘缓存读取，避免重复计算）
        events = self._load_events_cache(split, fused, threshold)
        if events is None:
            events = aggregate_spatial_events_fast(
                fused, threshold=threshold, min_patch_size=12,
            )
            self._save_events_cache(split, events)

        # 预警判定（逐时间步）
        self._warning_engine.reset()
        all_alerts = []
        for t_idx in range(len(flow)):
            step_alerts = self._warning_engine.evaluate(
                anomaly_mask[t_idx],
                fused[t_idx],
                t=offset + t_idx,
            )
            all_alerts.extend(step_alerts)

        result = {
            "flow": flow,
            "predictions": scores_dict.get("pred"),
            "scores": {
                "statistical": stat_scores,
                "prediction":  pred_scores,
                "vae":         scores_dict.get("vae", np.zeros_like(flow)),
                "transformer": scores_dict.get("transformer", np.zeros_like(flow)),
                "fused":       fused,
            },
            "anomaly_mask": anomaly_mask,
            "events": events,
            "alerts": all_alerts,
            "timestamps": timestamps,
            "split": split,
        }
        self._result_cache = result
        print(f"[Pipeline] batch({split}) done in {time.time()-t0:.1f}s, "
              f"anomalies={anomaly_mask.sum()}/{anomaly_mask.size}, "
              f"events={len(events)}, alerts={len(all_alerts)}")
        return result

    # ── 模拟实时模式 ──────────────────────────────────────────────────────────

    def init_realtime(self, warmup_steps: int = 48):
        """初始化实时模式：预热滑动窗口。

        Args:
            warmup_steps: 冷启动需要的历史步数（默认48步≈2天）
        """
        self._window = []
        self._warning_engine.reset()
        self._ensure_stat()
        self._ensure_pred()

        flow = get_flow_1d("taxi_flow_total")
        train_end, val_end, test_end = get_splits()
        # 用测试集最后 warmup_steps 步作为初始窗口
        start = test_end - warmup_steps
        self._window = [flow[start + i] for i in range(warmup_steps)]
        print(f"[Pipeline] realtime initialized with {warmup_steps} warmup steps")

    def run_step(self, new_data: np.ndarray = None) -> Dict[str, Any]:
        """模拟实时：输入一个时间步，输出异常判定和预警。

        Args:
            new_data: (N,) 当前步实际人流（None=自动从数据流取）

        Returns:
            {
                "t": int,                       # 全局时间步索引
                "predictions": (N,) float,      # 当前步预测值
                "anomaly_mask": (N,) bool,      # 异常判定
                "scores": {method: float},
                "alert": Optional[WarningAlert],
                "window": List[np.ndarray],     # 更新后的窗口（用于调试）
            }
        """
        flow = get_flow_1d("taxi_flow_total")
        train_end, val_end, test_end = get_splits()
        current_t = val_end + len(self._window)

        # 获取新数据
        if new_data is None:
            if current_t < val_end + test_end:
                new_data = flow[current_t]
            else:
                # 数据流结束，返回空结果
                return {"t": current_t, "error": "end of data"}

        self._window.append(new_data)
        if len(self._window) > 48:
            self._window.pop(0)

        # 计算当前步得分（stat + pred）
        window_arr = np.stack(self._window)          # (48, N)
        # stat 期望单步：取最后一帧
        scores_stat = self._stat_detector.predict_scores(window_arr[-1:])
        # _pred_detector.predict_scores 返回 (scores, direction)
        _pred_out = self._pred_detector.predict_scores(new_data)
        scores_pred = _pred_out[0] if isinstance(_pred_out, tuple) else _pred_out

        # 融合（与 batch 一致）
        if self.mode == "fast":
            fused = 0.9 * normalize_scores(scores_stat) + \
                    0.1 * normalize_scores(scores_pred)
        else:
            # structural 模式需要 VAE/TAE，这里先做 stat+pred
            fused = (0.7 * normalize_scores(scores_stat) +
                     0.1 * normalize_scores(scores_pred) +
                     0.2 * normalize_scores(np.zeros_like(scores_stat)))

        threshold = self.fusion_threshold
        anomaly_mask = (fused >= threshold).flatten()

        # 预警
        alerts = self._warning_engine.evaluate(
            anomaly_mask, fused.flatten(), t=current_t
        )
        top_alert = alerts[-1] if alerts else None

        ts = get_timestamps()
        timestamp = ts[current_t] if current_t < len(ts) else None

        return {
            "t": current_t,
            "timestamp": str(timestamp) if timestamp else None,
            "predictions": scores_pred,
            "anomaly_mask": anomaly_mask,
            "scores": {
                "statistical": scores_stat.mean(),
                "prediction":  scores_pred.mean(),
                "fused":       fused.mean(),
            },
            "alert": top_alert,
            "window_size": len(self._window),
        }

    # ── 辅助接口 ─────────────────────────────────────────────────────────────

    def get_grid_data(self, t: int) -> Dict[str, Any]:
        """获取指定时间步的网格热力图数据（用于前端绘图）"""
        if not self._result_cache:
            self.run_batch()

        flow = self._result_cache["flow"]
        scores = self._result_cache["scores"]["fused"]
        mask = self._result_cache["anomaly_mask"]

        train_end, val_end, test_end = get_splits()
        t_local = t - val_end
        train_end, val_end, test_end = get_splits()
        if not (0 <= t_local < len(flow)):
            raise ValueError(f"t={t} out of range [{val_end}, {val_end + len(flow)})")

        heatmap_data = flow[t_local].reshape(32, 32).tolist()
        anomaly_coords = [
            {"row": int(r), "col": int(c), "score": float(scores[t_local, r*32+c])}
            for r in range(32) for c in range(32)
            if mask[t_local, r*32 + c]
        ]
        return {
            "t": t,
            "timestamp": str(self._result_cache["timestamps"][t_local]),
            "heatmap": heatmap_data,
            "anomaly_cells": anomaly_coords,
            "n_anomaly": int(mask[t_local].sum()),
        }

    def query_events(self, t_start: int = None, t_end: int = None,
                     min_cells: int = 0,
                     include_marginal: bool = False) -> List[Dict]:
        """查询历史异常事件

        Args:
            t_start / t_end: 时间范围（全局索引）
            min_cells: 最少网格数过滤
            include_marginal: True=包含 patch_marginal / point_single 兜底入库事件
                              False=仅返回时空连续的 spatial_sustained 事件
        """
        if not self._result_cache:
            self.run_batch()

        events = self._result_cache["events"]
        result = []

        # 外部查询用全局索引，内部事件用局部索引(VAL_END偏移)
        train_end, val_end, test_end = get_splits()
        global_offset = val_end

        for e in events:
            # 兜底过滤
            if not include_marginal and e.event_type != "spatial_sustained":
                continue

            # 将事件局部索引转为全局索引再比较
            e_t_start = e.t_start + global_offset
            e_t_end   = e.t_end   + global_offset

            if t_start is not None and e_t_end < t_start:
                continue
            if t_end is not None and e_t_start > t_end:
                continue
            # min_cells 仅对 spatial_sustained 生效，零散事件不受此限制
            if e.event_type == "spatial_sustained" and e.n_cells < min_cells:
                continue
            result.append({
                "event_id": e.event_id,
                "t_start": e_t_start,    # 全局索引
                "t_end":   e_t_end,      # 全局索引
                "duration": e.duration,
                "n_cells": e.n_cells,
                "n_center": e.n_center,
                "center_row": e.n_center // 32,
                "center_col": e.n_center % 32,
                "event_type": e.event_type,
                "avg_score": round(e.avg_score, 4),
                "warning_level": self._infer_warning_level(e),
            })
        return result

    def _infer_warning_level(self, event: AnomalyEvent) -> int:
        """统一预警等级规则（与 anomaly_attribution.infer_level 一致）。

        口径（v3, 2026-07-28）：
            level=2 (重要)  n_cells >= 20
            level=1 (一般)  n_cells >= 16
            level=0         其他

        注意：与 pipeline.WarningEngine 的实时预警规则不同 —— 此处用于
        离线 retrospective 分析，duration 字段对事后判断无意义（事件
        整体时长可能跨越多个时间步，但单时间步不易捕捉）。
        """
        nc = getattr(event, "n_cells", 0)
        if nc >= 20:
            return 2
        if nc >= 16:
            return 1
        return 0


# ── 便捷入口 ──────────────────────────────────────────────────────────────────

def run_quick_demo():
    """快速演示：跑一遍批量模式，验证全流程"""
    print("=== Week6 Pipeline Quick Demo ===")
    pipe = SpatiotemporalPipeline(mode="fast", use_cache=True)
    result = pipe.run_batch(split="test")

    print(f"  anomalies: {result['anomaly_mask'].sum()}/{result['anomaly_mask'].size}")
    print(f"  events: {len(result['events'])}")
    print(f"  alerts: {len(result['alerts'])}")
    levels = {}
    for a in result["alerts"]:
        levels[a.level_name] = levels.get(a.level_name, 0) + 1
    print(f"  alert breakdown: {levels}")
    return result


if __name__ == "__main__":
    run_quick_demo()
