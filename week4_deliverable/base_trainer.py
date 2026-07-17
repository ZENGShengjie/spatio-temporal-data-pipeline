"""模型基类 — 通用训练工具"""
from __future__ import annotations
import os, time, json
from typing import Dict, Tuple
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from config import cfg_train, SPLIT
from data_loader import load_raw_flow, load_time_features, SeqDataset, GraphSeqDataset

def make_device():
    return torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CUDA_OFF") != "1" else "cpu")

def build_loaders(flow_4d, time_feat, seq_len, horizon, batch_size=32):
    train_ds = SeqDataset(flow_4d, seq_len, horizon, SPLIT.train_start, SPLIT.train_end, time_features=time_feat)
    val_ds   = SeqDataset(flow_4d, seq_len, horizon, SPLIT.train_end,   SPLIT.val_end,   time_features=time_feat)
    test_ds  = SeqDataset(flow_4d, seq_len, horizon, SPLIT.val_end,     SPLIT.test_end,  time_features=time_feat)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=False),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
            train_ds, val_ds, test_ds)

def standard_normalize(flow):
    cell_max = compute_cell_max(flow)
    return (flow / cell_max).clip(min=0).astype(np.float32), cell_max

def compute_cell_max(flow):
    return np.maximum(flow[:SPLIT.train_end].max(axis=0), 1.0)

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
