"""融合框架 V3 — 定向精准融合 + 三路对比 + 重构模型得分升级

V3 修复（针对 V2 问题）：
  1. 收紧准入门槛：F1≥0.4（仅 stat + pred 达标），排除 VAE/TAE 稀释信号
  2. 双方法精准融合：statistical + prediction，暴力网格搜索最优权重
  3. 重构得分升级：逐序列 top-k 最大残差 z-score，替代全序列平均 MSE
  4. 缓存命名 _v3 后缀，不覆盖 V2 数据

数据泄露红线：
  - 权重搜索/阈值校准均在验证集上完成
  - 测试集仅用于最终一次性评估
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

import numpy as np
import pandas as pd
from scipy import ndimage

from week5.config import (
    FUSION_CFG, CACHE_DIR, DATA_DIR,
    VAL_END, VAL_HOURS, TEST_HOURS, N_CELLS, TRAIN_END,
    cache_path, cache_json,
)


# ── 异常分类 ─────────────────────────────────────────────────────────────────

def classify_anomaly_type(
    scores: np.ndarray,
    direction: np.ndarray,
    threshold: float = 0.5,
    spatial_mask: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    T, N = scores.shape
    is_anomaly = scores >= threshold
    labels = np.zeros((T, N), dtype=np.int32)
    labels[is_anomaly & (direction > 0)] = 1
    labels[is_anomaly & (direction < 0)] = 2
    if spatial_mask is not None:
        labels[is_anomaly & spatial_mask] = 3
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
    T, N = scores.shape
    is_anomaly = scores >= threshold
    H, W = 32, 32
    spatial_mask = np.zeros_like(is_anomaly)
    struct = ndimage.generate_binary_structure(2, 1)
    for t in range(T):
        labeled, n = ndimage.label(is_anomaly[t].reshape(H, W), structure=struct)
        for lid in range(1, n + 1):
            comp = (labeled == lid)
            if comp.sum() > 1:
                spatial_mask[t, comp.flatten()] = True
    return spatial_mask


def aggregate_spatial_events_fast(
    scores: np.ndarray,
    threshold: float = 0.5,
    single_point_score: float = 0.85,
    min_patch_size: int = 12,
) -> List[AnomalyEvent]:
    """3D 连通域聚合 + 两类兜底规则（与热力图实时异常对齐）。

    入库优先级（从高到低）：
      1. 3D 时空连通域聚合事件（主逻辑，任何规模的连通片）
      2. 连片兜底：单时间步空间连通片 ≥min_patch_size 格，标记为 patch_marginal
      3. 单点兜底：孤立格点得分 ≥single_point_score，标记为 point_single

    新增的两条兜底规则确保：
      - 所有在热力图上肉眼可见的异常（≥12格/单步 或 高分孤立点）
        在事件列表中都有对应记录，解决"有红无事件"的体感脱节
      - 真正的时空连续大事件优先收录，零散噪声不会淹没列表
    """
    T, N = scores.shape
    is_anomaly = scores >= threshold
    H, W = 32, 32
    anomaly_3d = is_anomaly.reshape(T, H, W)
    struct_3d = np.zeros((3, 3, 3), dtype=bool)
    struct_3d[1, :, :] = True
    labeled_3d, n_events = ndimage.label(anomaly_3d, structure=struct_3d)
    events = []
    covered = np.zeros_like(is_anomaly)        # 已被连通事件覆盖的格点
    next_eid = 0
    for eid in range(1, n_events + 1):
        mask_3d = labeled_3d == eid
        mask_2d = mask_3d.reshape(T, H * W)
        t_idx, r_idx, c_idx = np.where(mask_3d)
        t_start, t_end = int(t_idx.min()), int(t_idx.max())
        duration = t_end - t_start + 1
        n_cells = len(t_idx)

        # 单格单步的 3D 组件直接跳过，交给 fallback 处理
        # 这样 fallback 才能区分：真正的时空大事件 vs 零散孤立异常
        if n_cells == 1 and duration == 1:
            continue  # 不标记 covered，fallback 兜底会处理

        n_center = r_idx[len(r_idx) // 2] * W + c_idx[len(c_idx) // 2]
        avg_score = float(scores[mask_2d].mean())
        is_spatial = (n_cells > 1) or (duration > 1)
        events.append(AnomalyEvent(
            event_id=next_eid, t_start=t_start, t_end=t_end, duration=duration,
            n_cells=n_cells, n_center=int(n_center),
            event_type="spatial_sustained",
            avg_score=avg_score, is_spatial=is_spatial,
        ))
        covered |= mask_2d
        next_eid += 1

    # ── 兜底 1：单时间步空间连通片 ≥min_patch_size ─────────────────────────────
    # 对每个时间步单独做 2D 连通域分析，不满足主逻辑但足够大的连片也收录
    struct_2d = ndimage.generate_binary_structure(2, 1)
    patch_covered = np.zeros(T, dtype=bool)   # 标记哪些时间步已处理过
    for t in range(T):
        frame = is_anomaly[t].reshape(H, W)
        labeled_2d, n_patches = ndimage.label(frame, structure=struct_2d)
        for pid in range(1, n_patches + 1):
            comp = labeled_2d == pid
            sz = int(comp.sum())
            if sz < min_patch_size:
                continue
            # 该连通片有多少格点未被主逻辑覆盖（可能部分被3D事件覆盖）
            comp_flat = comp.flatten()
            uncovered = comp_flat & ~covered[t]
            n_uncovered = int(uncovered.sum())
            if n_uncovered < min_patch_size:
                continue
            t_idx_arr = np.full(sz, t, dtype=int)
            n_idx_arr = np.where(comp_flat)[0]
            n_center = n_idx_arr[len(n_idx_arr) // 2]
            events.append(AnomalyEvent(
                event_id=next_eid,
                t_start=t, t_end=t, duration=1,
                n_cells=sz, n_center=int(n_center),
                event_type="patch_marginal",   # 兜底入库，等级由 pipeline 统一判定
                avg_score=float(scores[t][comp_flat].mean()),
                is_spatial=True,
            ))
            next_eid += 1
            patch_covered[t] = True

    # ── 兜底 2：单点高得分孤立点 ───────────────────────────────────────────────
    single_mask = is_anomaly & ~covered
    if single_mask.any():
        t_idx, n_idx = np.where(single_mask)
        for t_i, n_i in zip(t_idx, n_idx):
            if scores[t_i, n_i] < single_point_score:
                continue
            r, c = divmod(int(n_i), W)
            events.append(AnomalyEvent(
                event_id=next_eid,
                t_start=int(t_i), t_end=int(t_i), duration=1,
                n_cells=1, n_center=int(n_i),
                event_type="point_single",
                avg_score=float(scores[t_i, n_i]),
                is_spatial=False,
            ))
            next_eid += 1

    return events


# ── 得分归一化 ────────────────────────────────────────────────────────────────

def normalize_scores(scores: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    q99 = np.percentile(scores, quantile * 100)
    q00 = scores.min()
    r = q99 - q00
    if r < 1e-9:
        return np.zeros_like(scores)
    return np.clip((scores - q00) / r, 0, 1).astype(np.float32)


# ── 重构模型得分升级（逐序列 top-k max z-score）──────────────────────────────

def compute_recon_topk_zscore(
    err: np.ndarray,
    baseline_sigma: float = None,
    k: int = 3,
) -> np.ndarray:
    """逐序列 top-k 最大残差 z-score。

    对每个格点的 SEQ 步时间序列，计算误差 z-score，取最大的 k 步均值。
    放大局部数值突变信号。
    """
    T, N = err.shape  # (T, N) 其中 T >= SEQ

    if baseline_sigma is None:
        baseline_sigma = float(np.std(err))

    sigma = max(baseline_sigma, 1e-6)
    z = err / sigma  # z-score

    # top-k max z-score（沿时间维度，取最大的 k 步）
    if k >= T:
        topk_scores = np.max(z, axis=0)
    else:
        topk_scores = np.zeros(N, dtype=np.float32)
        for n in range(N):
            vals = z[:, n]
            topk = np.partition(vals, -k)[-k:]
            topk_scores[n] = float(np.mean(topk))

    # 归一化到 [0,1]（全局 99th 分位截断）
    q99 = float(np.percentile(topk_scores, 99))
    q00 = float(topk_scores.min())
    r = q99 - q00
    if r < 1e-9:
        return np.zeros((T, N), dtype=np.float32)
    normed = np.clip((topk_scores - q00) / r, 0, 1).astype(np.float32)
    return np.broadcast_to(normed[None, :], (T, N))


def compute_recon_mean_mse(
    err: np.ndarray,
    baseline_sigma: float = None,
) -> np.ndarray:
    """标准版：全序列平均 MSE（对比基准）"""
    T, N = err.shape
    if baseline_sigma is None:
        baseline_sigma = float(np.std(err))
    sigma = max(baseline_sigma, 1e-6)
    z = err / sigma
    mean_z = np.mean(z, axis=0)  # (N,)
    q99 = float(np.percentile(mean_z, 99))
    q00 = float(mean_z.min())
    r = q99 - q00
    if r < 1e-9:
        return np.zeros((T, N), dtype=np.float32)
    normed = np.clip((mean_z - q00) / r, 0, 1).astype(np.float32)
    return np.broadcast_to(normed[None, :], (T, N))


# ── 分时段动态阈值 ────────────────────────────────────────────────────────────

def fit_per_group_thresholds(
    scores: np.ndarray,
    true_labels: np.ndarray,
    time_groups: np.ndarray,
    step: float = 0.01,
) -> Tuple[Dict[int, float], float]:
    """分时段独立搜索最优阈值（F1 最大化）。

    Returns:
        (per_group_thresh_dict, best_global_thresh)
    """
    unique_groups = np.unique(time_groups)
    per_group_thresh = {}
    group_metrics = {}

    # 全局搜索
    all_pred = scores.flatten()
    all_labels = true_labels.flatten()
    best_global = 0.5
    best_f1_global = 0.0
    for thresh in np.arange(0.05, 1.0, step):
        pred = (all_pred >= thresh).astype(int)
        tp = int(((pred == 1) & (all_labels == 1)).sum())
        fp = int(((pred == 1) & (all_labels == 0)).sum())
        fn = int(((pred == 0) & (all_labels == 1)).sum())
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1_global:
            best_f1_global = f1
            best_global = thresh

    print(f"[fusion V3] global best: thresh={best_global:.3f}, F1={best_f1_global:.4f}")

    # 分组搜索
    for gid in unique_groups:
        g_mask = time_groups == gid
        g_scores = scores[g_mask].flatten()
        g_labels = true_labels[g_mask].flatten()
        if g_labels.sum() == 0:
            per_group_thresh[int(gid)] = float(best_global)
            group_metrics[int(gid)] = {"f1": 0.0}
            continue
        best_t = best_global
        best_f1 = 0.0
        for thresh in np.arange(0.05, 1.0, step):
            pred = (g_scores >= thresh).astype(int)
            tp = int(((pred == 1) & (g_labels == 1)).sum())
            fp = int(((pred == 1) & (g_labels == 0)).sum())
            fn = int(((pred == 0) & (g_labels == 1)).sum())
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = thresh
        per_group_thresh[int(gid)] = float(best_t)
        group_metrics[int(gid)] = {"f1": float(best_f1), "thresh": float(best_t)}

    print(f"[fusion V3] per-group thresholds: {per_group_thresh}")
    return per_group_thresh, float(best_global)


def predict_with_per_group_thresh(
    scores: np.ndarray,
    time_groups: np.ndarray,
    per_group_thresh: Dict[int, float],
) -> np.ndarray:
    """用分时段阈值判定异常"""
    T, N = scores.shape
    mask = np.zeros((T, N), dtype=bool)
    for gid, thresh in per_group_thresh.items():
        g_mask = time_groups == gid
        mask[g_mask] = scores[g_mask] >= thresh
    return mask


# ── 融合主类 ────────────────────────────────────────────────────────────────

class AnomalyFusionV3:
    """融合框架 V3：支持三种融合模式对比"""

    def __init__(self, target: str = "taxi_flow_total"):
        self.target = target
        self.weights: Optional[Dict[str, float]] = None
        self.threshold: float = 0.5
        self.per_group_thresh: Optional[Dict[int, float]] = None
        self.score_dict: Dict[str, np.ndarray] = {}
        self.events: List[AnomalyEvent] = []
        self._fitted = False

    def load_scores(self, split: str = "val"):
        """加载各方法得分（V3 优先，V2 兜底）"""
        score_map = {
            "statistical": [f"stat_scores_{split}_v3", f"stat_scores_{split}_v2"],
            "prediction":  [f"pred_scores_{split}_v3", f"pred_scores_{split}_v2"],
            "vae":         [f"vae_scores_{split}_v3", f"vae_scores_{split}_v2"],
            "transformer":  [f"tae_scores_{split}_v3", f"tae_scores_{split}_v2"],
        }
        for method, names in score_map.items():
            for name in names:
                path = cache_path(name)
                if os.path.exists(path):
                    self.score_dict[method] = np.load(path)
                    print(f"[fusion V3] loaded {method} from {os.path.basename(path)}")
                    break

    def fit(
        self,
        true_labels_val: np.ndarray,
        time_groups_val: np.ndarray = None,
        mode: str = "dual",
        perf_gate_f1: float = 0.4,
    ):
        """融合权重搜索。

        Args:
            mode: "all" | "dual" | "stat_only"
            perf_gate_f1: F1 准入门槛
        """
        self.load_scores("val")

        if len(self.score_dict) == 0:
            print("[fusion V3] ERROR: no scores loaded")
            return

        # ── 方法筛选 ────────────────────────────────────────────────────────
        from sklearn.metrics import roc_auc_score, f1_score

        if mode == "stat_only":
            qualified = {"statistical": self.score_dict["statistical"]} \
                if "statistical" in self.score_dict else {}
        elif mode == "dual":
            qualified = {m: s for m, s in self.score_dict.items()
                       if m in ("statistical", "prediction")}
        else:  # "all"
            qualified = {}
            for method, scores in self.score_dict.items():
                s_flat = scores.flatten()
                l_flat = true_labels_val.flatten()
                try:
                    auc = roc_auc_score(l_flat, s_flat)
                except ValueError:
                    auc = 0.5
                pred_b = (scores > np.percentile(scores, 95)).astype(int)
                try:
                    f1 = f1_score(l_flat, pred_b)
                except ValueError:
                    f1 = 0.0
                ok = auc >= 0.55 and f1 >= perf_gate_f1
                print(f"[fusion V3] {method:15s} auc={auc:.4f} f1={f1:.4f} "
                      f"{'✓' if ok else '✗'}")
                if ok:
                    qualified[method] = scores
            if not qualified:
                qualified = {m: s for m, s in self.score_dict.items()
                           if m in ("statistical", "prediction")}

        self.score_dict = qualified
        if not self.score_dict:
            print("[fusion V3] ERROR: no qualified methods")
            return

        # ── 权重 + 阈值联合搜索 ──────────────────────────────────────────
        if len(self.score_dict) == 1 or mode == "stat_only":
            self.weights = {list(self.score_dict.keys())[0]: 1.0}
            # 分时段阈值
            if time_groups_val is not None:
                method = list(self.score_dict.keys())[0]
                self.per_group_thresh, self.threshold = fit_per_group_thresholds(
                    self.score_dict[method], true_labels_val, time_groups_val
                )
            self._fitted = True
            return

        # 多方法：网格搜索
        methods = list(self.score_dict.keys())
        n = len(methods)
        step = 0.05
        steps_vals = np.arange(0.0, 1.001, step)

        best_f1 = 0.0
        best_weights = None
        best_thresh = 0.5
        best_metrics = {}

        for combo in _weighted_combos(steps_vals, n):
            w_sum = sum(combo)
            if abs(w_sum - 1.0) > step * 2:
                continue

            fused = sum(combo[i] * normalize_scores(self.score_dict[m])
                       for i, m in enumerate(methods))

            for thresh in np.arange(0.10, 0.95, 0.02):
                pred_b = (fused >= thresh).astype(int)
                tp = int(((pred_b == 1) & (true_labels_val == 1)).sum())
                fp = int(((pred_b == 1) & (true_labels_val == 0)).sum())
                fn = int(((pred_b == 0) & (true_labels_val == 1)).sum())
                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)
                f1 = 2 * p * r / (p + r + 1e-9)
                # 约束：P≥0.7, R≥0.85
                if p >= 0.7 and r >= 0.85 and f1 > best_f1:
                    best_f1 = f1
                    best_weights = {m: float(combo[i]) for i, m in enumerate(methods)}
                    best_thresh = float(thresh)
                    best_metrics = {"precision": p, "recall": r, "f1": f1}

        self.weights = best_weights or {m: 1.0 / n for m in methods}
        self.threshold = best_thresh
        print(f"[fusion V3] search: weights={self.weights}, thresh={self.threshold:.3f}, "
              f"P={best_metrics.get('precision',0):.4f} "
              f"R={best_metrics.get('recall',0):.4f} F1={best_f1:.4f}")

        self._fitted = True

    def predict(self, split: str = "test") -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Must call fit() first")

        self.load_scores(split)
        T, N = list(self.score_dict.values())[0].shape

        fused = np.zeros((T, N), dtype=np.float32)
        for method, w in self.weights.items():
            if method in self.score_dict:
                fused += w * normalize_scores(self.score_dict[method])

        thresh = self.threshold
        mask = (fused >= thresh).astype(bool)

        direction_path = cache_path(f"pred_direction_{split}_v2")
        direction = np.zeros((T, N), dtype=np.float32)
        if os.path.exists(direction_path):
            direction = np.load(direction_path)
        spatial_mask = build_spatial_mask(fused, threshold=thresh)
        self.events = aggregate_spatial_events_fast(fused, threshold=thresh)

        tag = "_v3"
        np.save(cache_path(f"fusion_scores_{split}{tag}"), fused)
        np.save(cache_path(f"fusion_mask_{split}{tag}"), mask)

        return mask, fused

    def _save(self):
        params = {
            "weights": self.weights,
            "threshold": float(self.threshold),
            "target": self.target,
        }
        with open(cache_json("fusion_params_v3"), "w") as f:
            json.dump(params, f, indent=2)


def _weighted_combos(steps: np.ndarray, n: int):
    """生成 n 个权重和为 1 的组合"""
    from itertools import product
    for combo in product(steps, repeat=n):
        if abs(sum(combo) - 1.0) < 0.01:
            yield combo


# ── 三路融合对比 ─────────────────────────────────────────────────────────────

def run_fusion_comparison(target: str = "taxi_flow_total") -> Dict[str, dict]:
    """运行 V3 三路融合对比。

    Returns:
        {
            "all_methods":  {...metrics},
            "dual_methods":  {...metrics},
            "stat_only":     {...metrics},
        }
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    # ── 加载数据 ──────────────────────────────────────────────────────────
    test_labels_path = os.path.join(DATA_DIR, "anomaly_labels_test.npy")
    val_labels_path = os.path.join(DATA_DIR, "anomaly_labels_val.npy")
    if os.path.exists(test_labels_path):
        gt_test = np.load(test_labels_path)
        gt_val = np.load(val_labels_path) if os.path.exists(val_labels_path) else None
    else:
        all_labels = np.load(os.path.join(DATA_DIR, "anomaly_labels.npy"))
        gt_test = all_labels[VAL_HOURS:]
        gt_val = all_labels[:VAL_HOURS]

    # 时间分组
    from data_loader import get_time_features
    from data_loader import get_time_group_labels
    tg_val = get_time_group_labels()[:VAL_END - TRAIN_END]
    tg_test = get_time_group_labels()[VAL_END - TRAIN_END:]
    tg_full = get_time_group_labels()
    # dummy tg_full for compatibility
    tg_val = tg_full[TRAIN_END:VAL_END]
    tg_test = tg_full[VAL_END:]

    results = {}
    for mode in ["all", "dual", "stat_only"]:
        fusion = AnomalyFusionV3(target=target)
        if gt_val is not None:
            fusion.fit(gt_val, tg_val, mode=mode, perf_gate_f1=0.4)
        else:
            fusion.fit(np.zeros((504, 1024), dtype=bool), mode=mode)

        mask_test, scores_test = fusion.predict("test")

        pred_flat = mask_test.flatten().astype(int)
        gt_flat = gt_test.flatten().astype(int)
        tp = int(((pred_flat == 1) & (gt_flat == 1)).sum())
        fp = int(((pred_flat == 1) & (gt_flat == 0)).sum())
        fn = int(((pred_flat == 0) & (gt_flat == 1)).sum())
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        try:
            auc_roc = roc_auc_score(gt_flat, scores_test.flatten())
            auc_pr = average_precision_score(gt_flat, scores_test.flatten())
        except ValueError:
            auc_roc = auc_pr = float("nan")

        results[mode] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "auc_roc": round(auc_roc, 4), "auc_pr": round(auc_pr, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "weights": fusion.weights,
            "threshold": float(fusion.threshold),
        }
        print(f"[fusion V3] {mode:12s}: P={p:.4f} R={r:.4f} F1={f1:.4f} "
              f"AUC={auc_roc:.4f} | w={fusion.weights}")

    # 保存
    rows = []
    for mode, m in results.items():
        row = {k: v for k, v in m.items() if k not in ("weights", "threshold")}
        row["mode"] = mode
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "fusion_comparison_v3.csv")
    df.to_csv(out_path, index=False)
    print(f"[fusion V3] comparison saved → {out_path}")
    return results


# ── 便捷入口 ─────────────────────────────────────────────────────────────────

def run(target: str = "taxi_flow_total") -> Dict[str, dict]:
    return run_fusion_comparison(target=target)


if __name__ == "__main__":
    run()
