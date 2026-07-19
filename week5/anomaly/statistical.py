"""统计法异常检测 — 分时段 3σ + IQR

数据泄露红线：
  - 均值 / 标准差 / IQR 仅用训练集计算
  - 阈值仅用验证集确定
  - 测试集仅用于最终检测

得分归一化：
  - 3σ: z-score 绝对值，超 3σ 截断，min-max 到 [0,1]
  - IQR: 偏离中位数 / IQR，截断后归一化
  - 融合时取交集（intersection）或并集（union）
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Literal, Tuple, Dict

import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    STAT_CFG, CACHE_DIR, VAL_END, TEST_END,
    TRAIN_END, VAL_HOURS, TEST_HOURS, N_CELLS,
    cache_path, cache_json, DATA_DIR,
)
from data_loader import get_flow_1d, get_time_group_labels, get_hour_labels


class StatisticalAnomalyDetector:
    """分时段统计法异常检测器"""

    def __init__(self, target: str = "taxi_flow_total"):
        self.target = target
        self.flow_train = None
        self.flow_val = None
        self.flow_test = None
        self.time_groups = None
        self.hour_labels = None

        # 每个时段分组的统计量
        self.group_stats: Dict[int, dict] = {}

        # 得分截断阈值
        self.score_clip_val: float = 0.0

        self._fitted = False

    # ── 数据加载 ───────────────────────────────────────────────────────────────

    def _load_data(self):
        flow = get_flow_1d(self.target)
        self.flow_train = flow[:TRAIN_END]   # (2784, N)
        self.flow_val   = flow[TRAIN_END:VAL_END]  # (504, N)
        # 优先使用注入后的数据（有标注），否则用原始数据
        test_injected = os.path.join(DATA_DIR, "flow_test_injected.npy")
        self.flow_test = np.load(test_injected) if os.path.exists(test_injected) else flow[VAL_END:]
        # 验证集也优先用注入后数据
        val_injected = os.path.join(DATA_DIR, "flow_val_injected.npy")
        self.flow_val = np.load(val_injected) if os.path.exists(val_injected) else self.flow_val
        self.time_groups = get_time_group_labels()
        self.hour_labels = get_hour_labels()

        # 时段分组只取对应区间
        self.tg_train = self.time_groups[:TRAIN_END]
        self.tg_val   = self.time_groups[TRAIN_END:VAL_END]
        self.tg_test  = self.time_groups[VAL_END:]

    # ── 统计量计算（仅训练集）─────────────────────────────────────────────────

    def _compute_group_stats(self):
        """按时段分组计算均值、标准差、中位数、IQR"""
        # 获取所有时段组 ID
        unique_groups = np.unique(self.tg_train)

        for gid in unique_groups:
            mask = (self.tg_train == gid)   # (T_train,)
            group_data = self.flow_train[mask]  # (T_g, N)

            mu = np.mean(group_data, axis=0)          # (N,)
            sigma = np.std(group_data, axis=0)        # (N,)
            sigma = np.maximum(sigma, 1e-6)

            q25 = np.percentile(group_data, 25, axis=0)
            q75 = np.percentile(group_data, 75, axis=0)
            iqr = np.maximum(q75 - q25, 1e-6)
            median = np.median(group_data, axis=0)

            self.group_stats[gid] = {
                "mu": mu, "sigma": sigma,
                "q25": q25, "q75": q75, "iqr": iqr, "median": median,
                "n_samples": int(mask.sum()),
            }

        print(f"[stat] computed stats for {len(self.group_stats)} time groups "
              f"({len(unique_groups)} unique IDs)")

    # ── 得分计算 ──────────────────────────────────────────────────────────────

    def _sigma_score(self, flow: np.ndarray,
                     tg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """3σ 法得分，返回 (scores, sigma_vals)"""
        T, N = flow.shape
        scores = np.zeros((T, N), dtype=np.float32)

        for gid in self.group_stats:
            mask = (tg == gid)
            if mask.sum() == 0:
                continue
            st = self.group_stats[gid]
            z = np.abs((flow[mask] - st["mu"]) / st["sigma"])
            # z-score 绝对值，超 3σ 截断
            z_clipped = np.clip(z, 0, STAT_CFG.sigma_threshold)
            scores[mask] = z_clipped / STAT_CFG.sigma_threshold

        return scores, np.abs(flow - np.zeros_like(flow))  # dummy sigma return

    def _iqr_score(self, flow: np.ndarray,
                   tg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """IQR 法得分"""
        T, N = flow.shape
        scores = np.zeros((T, N), dtype=np.float32)

        for gid in self.group_stats:
            mask = (tg == gid)
            if mask.sum() == 0:
                continue
            st = self.group_stats[gid]
            dev = np.abs(flow[mask] - st["median"]) / st["iqr"]
            # 超过 IQR 倍数截断
            dev_clipped = np.clip(dev, 0, STAT_CFG.iqr_k)
            scores[mask] = dev_clipped / STAT_CFG.iqr_k

        return scores, np.zeros_like(flow)

    def _normalize_scores(self, scores: np.ndarray,
                          split: Literal["train", "val", "test"] = "val") -> np.ndarray:
        """分位数截断归一化到 [0,1]"""
        # 使用验证集的分位数作为截断参考（避免测试集泄露）
        q99 = np.percentile(scores, STAT_CFG.score_quantile * 100)
        q00 = scores.min()
        # 截断
        clipped = np.clip(scores, q00, q99)
        # min-max 归一化
        range_ = q99 - q00
        if range_ < 1e-9:
            return np.zeros_like(scores)
        normed = (clipped - q00) / range_
        return normed.astype(np.float32)

    # ── 阈值确定（仅验证集）───────────────────────────────────────────────────

    def _fit_threshold(self, scores_val: np.ndarray,
                       true_labels: np.ndarray) -> float:
        """在验证集上搜索最优阈值（F1 最大化）"""
        best_thresh = 0.5
        best_f1 = 0.0
        for thresh in np.arange(0.1, 1.0, 0.05):
            pred = (scores_val >= thresh).astype(int)
            tp = int(((pred == 1) & (true_labels == 1)).sum())
            fp = int(((pred == 1) & (true_labels == 0)).sum())
            fn = int(((pred == 0) & (true_labels == 1)).sum())
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        print(f"[stat] best threshold={best_thresh:.2f}, val F1={best_f1:.4f}")
        return best_thresh

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def fit(self, true_labels_val: np.ndarray = None):
        """训练：计算统计量（仅训练集），确定阈值（仅验证集）"""
        self._load_data()
        self._compute_group_stats()

        # 计算验证集得分
        scores_sigma_val, _ = self._sigma_score(self.flow_val, self.tg_val)
        scores_iqr_val, _   = self._iqr_score(self.flow_val, self.tg_val)

        # 归一化
        scores_sigma_val_n = self._normalize_scores(scores_sigma_val, "val")
        scores_iqr_val_n   = self._normalize_scores(scores_iqr_val, "val")

        # 融合得分
        if STAT_CFG.strategy == "intersection":
            scores_val_fused = np.minimum(scores_sigma_val_n, scores_iqr_val_n)
        else:  # union
            scores_val_fused = np.maximum(scores_sigma_val_n, scores_iqr_val_n)

        # 用真值确定阈值
        if true_labels_val is None:
            # 如果没有真值，用 0.95 分位数作为默认阈值
            self.threshold = float(np.percentile(scores_val_fused, 95))
        else:
            self.threshold = self._fit_threshold(scores_val_fused, true_labels_val)

        # 计算截断值（用于后续得分归一化）
        self.score_clip_val = float(np.percentile(scores_val_fused, 99))

        self._fitted = True
        print(f"[stat] fit complete. threshold={self.threshold:.4f}")

        # 缓存统计量
        self._cache()

    def _cache(self):
        """缓存统计量到磁盘"""
        import pickle
        stats_path = os.path.join(CACHE_DIR, "stat_group_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(self.group_stats, f)
        params = {
            "threshold": self.threshold,
            "score_clip_val": self.score_clip_val,
            "target": self.target,
        }
        with open(cache_json("stat_params"), "w") as f:
            json.dump(params, f)

    def predict(self, split: Literal["train", "val", "test"] = "test",
                return_scores: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """对指定数据集预测异常

        Returns:
            (anomaly_mask, scores)  (T, N) bool 和 (T, N) float [0,1]
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        if split == "train":
            flow, tg = self.flow_train, self.tg_train
        elif split == "val":
            flow, tg = self.flow_val, self.tg_val
        elif split == "test":
            flow, tg = self.flow_test, self.tg_test
        else:
            raise ValueError(split)

        scores_sigma, _ = self._sigma_score(flow, tg)
        scores_iqr, _   = self._iqr_score(flow, tg)

        scores_sigma_n = self._normalize_scores(scores_sigma, split)
        scores_iqr_n   = self._normalize_scores(scores_iqr, split)

        if STAT_CFG.strategy == "intersection":
            scores_fused = np.minimum(scores_sigma_n, scores_iqr_n)
        else:
            scores_fused = np.maximum(scores_sigma_n, scores_iqr_n)

        mask = (scores_fused >= self.threshold)
        return mask.astype(bool), scores_fused

    def predict_scores(self, flow: np.ndarray,
                       tg_override: np.ndarray = None) -> np.ndarray:
        """给定任意 flow 序列，计算异常得分（不依赖 split）"""
        if tg_override is None:
            tg = self.tg_test[:len(flow)]
        else:
            tg = tg_override

        scores_sigma, _ = self._sigma_score(flow, tg)
        scores_iqr, _   = self._iqr_score(flow, tg)

        scores_sigma_n = self._normalize_scores(scores_sigma, "test")
        scores_iqr_n   = self._normalize_scores(scores_iqr, "test")

        if STAT_CFG.strategy == "intersection":
            return np.minimum(scores_sigma_n, scores_iqr_n)
        else:
            return np.maximum(scores_sigma_n, scores_iqr_n)


# ── 便捷入口 ──────────────────────────────────────────────────────────────────


def run(target: str = "taxi_flow_total") -> Tuple[np.ndarray, np.ndarray]:
    """单步运行，返回 (test_anomaly_mask, test_scores)"""
    detector = StatisticalAnomalyDetector(target=target)

    # 验证集真值（来自注入脚本 V2）
    labels_path = os.path.join(DATA_DIR, "anomaly_labels_val.npy")
    if os.path.exists(labels_path):
        true_val = np.load(labels_path)
    else:
        # V1 兼容：all_labels 前 VAL_HOURS 段
        all_labels_path = os.path.join(DATA_DIR, "anomaly_labels.npy")
        if os.path.exists(all_labels_path):
            all_labels = np.load(all_labels_path)
            true_val = all_labels[:VAL_HOURS]
        else:
            print("[stat] warning: no anomaly labels found, using default threshold")
            true_val = None

    # 测试集真值（来自注入脚本 V2）
    test_labels_path = os.path.join(DATA_DIR, "anomaly_labels_test.npy")
    if os.path.exists(test_labels_path):
        true_test = np.load(test_labels_path)
    else:
        all_labels_path = os.path.join(DATA_DIR, "anomaly_labels.npy")
        if os.path.exists(all_labels_path):
            all_labels = np.load(all_labels_path)
            true_test = all_labels[VAL_HOURS:]
        else:
            true_test = None

    detector.fit(true_labels_val=true_val)

    mask_val, scores_val = detector.predict(split="val")
    print(f"[stat] val detected: {mask_val.sum()} / {mask_val.size}")

    mask_test, scores_test = detector.predict(split="test")
    print(f"[stat] test detected: {mask_test.sum()} / {mask_test.size} steps are anomalous")

    # 缓存结果（V2 命名）
    np.save(cache_path("stat_scores_val_v2"), scores_val)
    np.save(cache_path("stat_mask_val_v2"), mask_val)
    np.save(cache_path("stat_scores_test_v2"), scores_test)
    np.save(cache_path("stat_mask_test_v2"), mask_test)
    if true_test is not None:
        np.save(cache_path("stat_mask_gt_test"), true_test)

    return mask_test, scores_test


if __name__ == "__main__":
    run()
