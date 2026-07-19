"""Week5 数据加载器 — 完全自包含，不依赖 Week4 data_loader

数据泄露红线：
  1. 所有统计量（均值/标准差/IQR）仅用训练集计算
  2. 归一化参数（cell_max）仅用训练集计算
  3. 异常注入仅在测试集进行
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Optional, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── Week4 配置常量（通过 exec 读取，不依赖 data_loader）────────────────────────
_REPO = Path(__file__).resolve().parents[1]          # /home/ubuntu/spatio-temporal-pipeline
_W4_DIR = _REPO / "week4"
_W4_CONFIG_PATH = _W4_DIR / "config.py"

_ns: dict = {"__file__": str(_W4_CONFIG_PATH)}
exec(open(_W4_CONFIG_PATH).read(), _ns)
_SPLIT = _ns.get("SPLIT")
if _SPLIT is not None:
    TRAIN_END   = int(_SPLIT.train_end)   # 2784
    VAL_END     = int(_SPLIT.val_end)    # 3288
    TEST_END    = int(_SPLIT.test_end)    # 3888
else:
    TRAIN_END = int(_ns["TRAIN_HOURS"])
    VAL_END   = TRAIN_END + int(_ns.get("VAL_HOURS", 504))
    TEST_END  = int(_ns["N_HOURS"])

TRAIN_HOURS = TRAIN_END
VAL_HOURS   = VAL_END - TRAIN_END
TEST_HOURS  = TEST_END - VAL_END
N_HOURS     = TEST_END

# 数据路径 — 通过环境变量配置，默认 /home/ubuntu/data/
_NPZ_PATH = Path(os.environ.get(
    "BJ_FLOW_NPZ",
    "/home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz"
))


# ── 全局数据（惰性加载）────────────────────────────────────────────────────────
_flow: Optional[np.ndarray] = None
_timestamps: Optional[np.ndarray] = None
_time_features: Optional[np.ndarray] = None
_cell_max: Optional[np.ndarray] = None
_normed_flow: Optional[np.ndarray] = None


def _load_raw_npz() -> Tuple[np.ndarray, np.ndarray]:
    """直接读取 npz 文件"""
    data = np.load(_NPZ_PATH)
    flow = data["flow"].astype(np.float32)   # (3888, 2, 32, 32)
    ts   = data["timestamps"]                # (3888,) datetime64
    return flow, ts


def load_raw_flow() -> np.ndarray:
    return _load_raw_npz()[0]


def load_timestamps() -> np.ndarray:
    return _load_raw_npz()[1]


def _compute_time_features(ts: np.ndarray) -> np.ndarray:
    """时间特征 (T, 5): hour_sin, hour_cos, is_weekend, is_workday, is_holiday"""
    dt = pd.to_datetime(ts)
    hours = dt.hour.values.astype(np.float32)
    dayofweek = dt.dayofweek.values

    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    is_weekend  = (dayofweek >= 5).astype(np.float32)
    is_workday  = (dayofweek < 5).astype(np.float32)
    is_holiday  = is_weekend  # 简化：周末=节假日

    return np.stack([hour_sin, hour_cos, is_weekend, is_workday, is_holiday], axis=1).astype(np.float32)


def load_time_features() -> np.ndarray:
    global _time_features
    if _time_features is None:
        _, ts = _load_raw_npz()
        _time_features = _compute_time_features(ts)
    return _time_features


# ── Week5 特有：全局归一化流 ─────────────────────────────────────────────────

def get_raw_flow() -> np.ndarray:
    global _flow
    if _flow is None:
        _flow = load_raw_flow()
    return _flow


def get_timestamps() -> np.ndarray:
    global _timestamps
    if _timestamps is None:
        _timestamps = load_timestamps()
    return _timestamps


def get_time_features() -> np.ndarray:
    return load_time_features()


def get_cell_max() -> np.ndarray:
    """每个格点的最大值（仅用训练集，禁止泄露）"""
    global _cell_max
    if _cell_max is None:
        flow = get_raw_flow()                           # (T, 2, 32, 32)
        T, C, H, W = flow.shape
        flow_2d = flow.reshape(T, C, H * W)            # (T, 2, 1024)
        train = flow_2d[:TRAIN_END]                     # (2784, 2, 1024)
        _cell_max = np.maximum(train.max(axis=0), 1.0)  # (2, 1024)
    return _cell_max


def get_normalized_flow() -> np.ndarray:
    """归一化全量数据（仅用训练集参数）"""
    global _normed_flow
    if _normed_flow is None:
        flow = get_raw_flow()
        cell_max = get_cell_max()                         # (2, 1024)
        T, C, H, W = flow.shape
        flow_2d = flow.reshape(T, C, H * W)              # (T, 2, 1024)
        _normed_flow = (flow_2d / cell_max[None, :, :]).astype(np.float32)
    return _normed_flow


def get_flow_1d(target: str = "taxi_flow_total") -> np.ndarray:
    """(T, N) 归一化 1D 流量，N=1024"""
    normed = get_normalized_flow()   # (T, 2, 1024)
    if target == "taxi_flow_total":
        return normed[:, 0, :] + normed[:, 1, :]
    elif target == "taxi_inflow":
        return normed[:, 0, :]
    elif target == "taxi_outflow":
        return normed[:, 1, :]
    else:
        raise ValueError(f"Unknown target: {target}")


# ── 数据集类 ───────────────────────────────────────────────────────────────────

class AnomalyDataset(Dataset):
    """单格点时序数据集（用于 VAE）"""

    def __init__(self, flow_1d: np.ndarray, seq_len: int = 48,
                 split: Literal["train", "val", "test"] = "train"):
        self.seq_len = seq_len
        self.flow = flow_1d   # (T, N)
        self.N = flow_1d.shape[1]
        if split == "train":
            self.start, self.end = 0, TRAIN_END
        elif split == "val":
            self.start, self.end = TRAIN_END, VAL_END
        elif split == "test":
            self.start, self.end = VAL_END, TEST_END
        else:
            raise ValueError(split)
        self.valid_ts = list(range(self.start + seq_len, self.end))

    def __len__(self):
        return len(self.valid_ts) * self.N

    def __getitem__(self, idx):
        t_idx = idx // self.N
        n_idx = idx % self.N
        t = self.valid_ts[t_idx]
        seq = self.flow[t - self.seq_len: t, n_idx]
        return torch.from_numpy(seq).float(), n_idx, t


def collate_single_cell(batch):
    seqs, n_idxs, ts = zip(*batch)
    return torch.stack(seqs), torch.tensor(n_idxs), torch.tensor(ts)


def get_cell_dataloader(flow_1d: np.ndarray, seq_len: int = 48,
                         split: Literal["train", "val", "test"] = "train",
                         batch_size: int = 256, shuffle: bool = True,
                         num_workers: int = 2) -> DataLoader:
    ds = AnomalyDataset(flow_1d, seq_len, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_single_cell)


class MultiCellDataset(Dataset):
    """多格点同时输入（用于 Transformer AE）"""

    def __init__(self, flow_1d: np.ndarray, seq_len: int = 48,
                 split: Literal["train", "val", "test"] = "train"):
        self.seq_len = seq_len
        self.flow = flow_1d
        self.N = flow_1d.shape[1]
        if split == "train":
            self.start, self.end = 0, TRAIN_END
        elif split == "val":
            self.start, self.end = TRAIN_END, VAL_END
        elif split == "test":
            self.start, self.end = VAL_END, TEST_END
        self.valid_ts = list(range(self.start + seq_len, self.end))

    def __len__(self):
        return len(self.valid_ts)

    def __getitem__(self, idx):
        t = self.valid_ts[idx]
        seq = self.flow[t - self.seq_len: t].T   # (N, seq_len)
        return torch.from_numpy(seq).float(), t


# ── 时间分组 ──────────────────────────────────────────────────────────────────

def get_time_group_labels() -> np.ndarray:
    """时段分组: hour * 2 + is_weekend → 0~47"""
    tf = get_time_features()   # (T, 5)
    # 反算 hour（sin/cos → degree）
    angle = np.arctan2(tf[:, 0], tf[:, 1])       # -pi ~ pi
    hours = ((angle * 24 / (2 * np.pi)) % 24).astype(int)
    is_weekend = (tf[:, 2] > 0.5).astype(int)
    return (hours * 2 + is_weekend).astype(np.int32)


def get_hour_labels() -> np.ndarray:
    """每个时间步的小时 (0~23)"""
    tf = get_time_features()
    angle = np.arctan2(tf[:, 0], tf[:, 1])
    return ((angle * 24 / (2 * np.pi)) % 24).astype(int)


def get_grid_coords() -> np.ndarray:
    """(N, 2) 网格坐标"""
    H, W = 32, 32
    return np.array([[r, c] for r in range(H) for c in range(W)], dtype=np.int32)


# ── 快捷入口 ──────────────────────────────────────────────────────────────────

def load_all(target: str = "taxi_flow_total") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (train, val, test, time_groups)"""
    flow = get_flow_1d(target)
    return flow[:TRAIN_END], flow[TRAIN_END:VAL_END], flow[VAL_END:], get_time_group_labels()
