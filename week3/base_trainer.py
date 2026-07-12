"""模型基类 — 通用训练工具"""
from __future__ import annotations

import os
import time
import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from config import cfg_train, SPLIT
from data_loader import load_raw_flow, load_time_features, SeqDataset, GraphSeqDataset


def make_device():
    return torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CUDA_OFF") != "1" else "cpu")


def build_loaders(flow_4d: np.ndarray, time_feat: np.ndarray | None,
                  seq_len: int, horizon: int, batch_size: int = 32):
    train_ds = SeqDataset(flow_4d, seq_len, horizon,
                          SPLIT.train_start, SPLIT.train_end,
                          time_features=time_feat)
    val_ds   = SeqDataset(flow_4d, seq_len, horizon,
                          SPLIT.train_end, SPLIT.val_end,
                          time_features=time_feat)
    test_ds  = SeqDataset(flow_4d, seq_len, horizon,
                          SPLIT.val_end, SPLIT.test_end,
                          time_features=time_feat)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def standard_normalize(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """按 (2, H, W) 每个 cell 的训练集最大值归一化"""
    cell_max = compute_cell_max(flow)
    normed = flow / cell_max
    normed = np.clip(normed, 0, None)
    return normed.astype(np.float32), cell_max


def compute_cell_max(flow: np.ndarray) -> np.ndarray:
    return np.maximum(flow[:SPLIT.train_end].max(axis=0), 1.0)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
