"""Week4 — 共享配置（与 Week3 完全一致）"""
import os
from dataclasses import dataclass, field
from typing import List
import numpy as np

DATA_DIR  = os.environ.get("BJ_DATA_DIR", "/home/ubuntu/data")
GRAPH_DIR = os.path.join(DATA_DIR, "graph_bj")
FEAT_DIR  = os.path.join(DATA_DIR, "features_bj")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned_bj")
GRID_DIR  = os.path.join(DATA_DIR, "grid_bj")

WEEK4_DIR = os.environ.get("WEEK4_DIR",
                           os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(WEEK4_DIR, "data")
PRED_DIR  = os.path.join(WEEK4_DIR, "results")
LOG_DIR   = os.path.join(WEEK4_DIR, "logs")
for d in [WEEK4_DIR, CACHE_DIR, PRED_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

ALL_MODELS = ["arima", "prophet", "lstm", "gru", "gcn", "gat",
              "stgcn", "agformer", "stf",
              "agformer_static", "stf_loc_only"]
DEFAULT_SEED = 20260710

N_HOURS = 3888; TRAIN_HOURS = 2784; VAL_HOURS = 504; TEST_HOURS = 600

@dataclass
class Split:
    train_start: int = 0
    train_end:   int = 2784
    val_end:     int = 2784 + 504
    test_end:    int = 3888
    @property
    def train_indices(self): return (self.train_start, self.train_end)
    @property
    def val_indices(self):   return (self.train_end, self.val_end)
    @property
    def test_indices(self):  return (self.val_end, self.test_end)

SPLIT = Split()

@dataclass
class TrainCfg:
    seq_len: int = 48; horizon: int = 48
    batch:   int = 4                      # V4: STGCN/AGFormer memory tight → batch 4
    epochs:  int = 20                     # V4: tighter epoch budget for advanced models
    lr:      float = 1e-3; weight_decay: float = 1e-5
    patience: int = 5
    hidden:  int = 64; layers: int = 2; dropout: float = 0.1
    seed:    int = DEFAULT_SEED
    device:  str = "cuda"

cfg_train = TrainCfg()

@dataclass
class EvalCfg:
    target_keys: List[str] = field(default_factory=lambda: ["taxi_inflow", "taxi_outflow"])
    norm: bool = True

EPS = 1e-6

def masked_mae(pred, gt): return float(np.abs(pred - gt).mean())
def masked_rmse(pred, gt): return float(np.sqrt(((pred - gt) ** 2).mean()))
def masked_mape(pred, gt):
    mask = (np.abs(gt) > 1.0)
    return float("nan") if mask.sum() == 0 else float((np.abs(pred[mask] - gt[mask]) / np.abs(gt[mask])).mean())
def correlation(pred, gt):
    p, g = pred.flatten(), gt.flatten()
    return 0.0 if p.std() < EPS or g.std() < EPS else float(np.corrcoef(p, g)[0, 1])

def compute_cell_max(flow):
    train = flow[:SPLIT.train_end]; cell_max = train.max(axis=0)
    return np.maximum(cell_max, 1.0)

def normalize(flow, cell_max): return flow / cell_max
def denormalize(flow_normed, cell_max): return flow_normed * cell_max

def get_split_flow(flow_4d):
    return (flow_4d[:SPLIT.train_end],
            flow_4d[SPLIT.train_end:SPLIT.val_end],
            flow_4d[SPLIT.val_end:SPLIT.test_end])
