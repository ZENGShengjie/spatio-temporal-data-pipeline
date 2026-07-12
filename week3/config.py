"""Week3 — 6 个基线模型共享配置"""
import os
from dataclasses import dataclass, field
from typing import List
import numpy as np

# === 数据路径（g4dn 默认）/ t3 部署时会通过环境变量覆盖 ===
DATA_DIR  = os.environ.get("BJ_DATA_DIR", "/home/ubuntu/data")
GRAPH_DIR = os.path.join(DATA_DIR, "graph_bj")
FEAT_DIR  = os.path.join(DATA_DIR, "features_bj")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned_bj")
GRID_DIR  = os.path.join(DATA_DIR, "grid_bj")

# === Week3 输出 ===
WEEK3_DIR = os.environ.get("WEEK3_DIR", "/home/ubuntu/amazon/week3")
CACHE_DIR = os.path.join(WEEK3_DIR, "data")           # 中间训练缓存
PRED_DIR  = os.path.join(WEEK3_DIR, "results")        # 各模型预测 npy
LOG_DIR   = os.path.join(WEEK3_DIR, "logs")
for d in [WEEK3_DIR, CACHE_DIR, PRED_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# === 模型 ===
ALL_MODELS = ["arima", "prophet", "lstm", "gru", "gcn", "gat"]
DEFAULT_SEED = 20260710

# === 数据规模 ===
N_HOURS       = 3888      # 周日边界外推的不算
TRAIN_HOURS   = 2784      # ~70%
VAL_HOURS     =  504      # ~14%   (12 周 * 7 天 = 84 天; 整周分布)
TEST_HOURS    =  600      # ~16%
# actual breakdown: train=2015-11-01 → 2016-04-30, val=…→…, test=2016-04-04 → 2016-04-10
# NOTE: 切分时间索引不是绝对的——按"自然周"切，留出最后一整周做 test。
# 实际切分比例：train/val/test ≈ 72 / 14 / 14

@dataclass
class Split:
    train_start: int = 0
    train_end:   int = 2784              # 第 0 ~ 第 2783 小时（含）
    val_end:     int = 2784 + 504        # 2784 ~ 3287（共 504h, 21 天）
    test_end:    int = 3888              # 3288 ~ 3887（共 600h, 25 天)

    @property
    def train_indices(self): return (self.train_start, self.train_end)
    @property
    def val_indices(self):   return (self.train_end, self.val_end)
    @property
    def test_indices(self):  return (self.val_end, self.test_end)

SPLIT = Split()


@dataclass
class TrainCfg:
    """通用训练超参 — V2: seq_len=48(24h) → horizon=48(24h)"""
    seq_len:  int = 48                 # input time window (slots, 30 min each)
    horizon:  int = 48                  # 一次预测多少 slots
    batch:    int = 16                  # V2: 48-step 输出更大, batch 减半
    epochs:   int = 50
    lr:       float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 7                  # early stopping
    hidden:   int = 64
    layers:   int = 2
    dropout:  float = 0.1
    seed:     int = DEFAULT_SEED
    device:   str = "cuda" if os.environ.get("CUDA_OFF") != "1" else "cpu"


cfg_train = TrainCfg()


@dataclass
class EvalCfg:
    """统一评估指标"""
    target_keys: List[str] = field(default_factory=lambda: ["taxi_inflow", "taxi_outflow"])
    norm:       bool = True            # 数据按 cell 历史最大值归一


# === 评价指标 ===
EPS = 1e-6

def masked_mae(pred, gt):
    """Mean Absolute Error (normalized zeros 自动跳过)"""
    return float(np.abs(pred - gt).mean())

def masked_rmse(pred, gt):
    return float(np.sqrt(((pred - gt) ** 2).mean()))

def masked_mape(pred, gt):
    """绝对百分比误差；gt 极小时跳过"""
    mask = (np.abs(gt) > 1.0)
    if mask.sum() == 0:
        return float("nan")
    return float((np.abs(pred[mask] - gt[mask]) / np.abs(gt[mask])).mean())

def correlation(pred, gt):
    """Pearson correlation of vectors (per cell)"""
    p = pred.flatten(); g = gt.flatten()
    if p.std() < EPS or g.std() < EPS:
        return 0.0
    return float(np.corrcoef(p, g)[0, 1])


# === 归一化（按 cell 历史最大值）===
def compute_cell_max(flow_4d):
    """
    flow_4d: (T, 2, H, W)
    返回 (2, H, W) 每 cell 的流入/流出训练集最大值（用于反归一化）
    """
    train = flow_4d[:SPLIT.train_end]    # only train set
    cell_max = train.max(axis=0)         # (2, H, W)
    cell_max = np.maximum(cell_max, 1.0)  # 防止 0
    return cell_max


def normalize(flow_4d, cell_max):
    return flow_4d / cell_max


def denormalize(flow_4d_normed, cell_max):
    return flow_4d_normed * cell_max


def get_split_flow(flow_4d):
    train = flow_4d[:SPLIT.train_end]
    val   = flow_4d[SPLIT.train_end:SPLIT.val_end]
    test  = flow_4d[SPLIT.val_end:SPLIT.test_end]
    return train, val, test
