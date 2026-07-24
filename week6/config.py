"""Week6 统一配置 — 复用 Week5 配置，按需覆盖 Week6 特有路径"""
from __future__ import annotations
import os
from pathlib import Path

# ── Week6 自身目录 ─────────────────────────────────────────────────────────────
WEEK6_DIR = Path(__file__).resolve().parent
API_DIR   = WEEK6_DIR / "api"
CACHE_DIR = WEEK6_DIR / "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── 复用 Week5 配置 ────────────────────────────────────────────────────────────
import sys as _sys
_parent = str(WEEK6_DIR.parent)
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)

from week5.config import (
    TRAIN_END, VAL_END, TEST_END,
    VAL_HOURS, TEST_HOURS, N_CELLS,
    N_HOURS, DEVICE,
    DATA_DIR, CACHE_DIR as W5_CACHE,
    TRAIN_END as _TE, VAL_END as _VE, TEST_END as _TEE,
)
from week5.data_loader import (
    get_raw_flow, get_timestamps, get_time_features,
    get_flow_1d, get_time_group_labels, get_hour_labels,
    get_grid_coords,
    TRAIN_END as _TRAIN_END, VAL_END as _VAL_END,
)
from week5.config import STAT_CFG, PRED_CFG, FUSION_CFG

# ── Week6 特有缓存路径 ─────────────────────────────────────────────────────────
def week6_cache(name: str) -> str:
    return str(CACHE_DIR / f"{name}.npy")

def week6_cache_json(name: str) -> str:
    return str(CACHE_DIR / f"{name}.json")

# ── 网格尺寸 ──────────────────────────────────────────────────────────────────
GRID_H = GRID_W = 32

# ── 滑动窗口参数 ──────────────────────────────────────────────────────────────
SLIDING_WINDOW_LEN = 48   # 实时模式窗口长度
