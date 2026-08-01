"""Week5 — 异常检测全局配置（数据泄露红线严格遵守）"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

# ── 路径 ────────────────────────────────────────────────────────────────────────
WEEK4_DIR = os.environ.get(
    "WEEK4_DIR",
    os.path.join(os.path.dirname(__file__), "../week4"),
)
WEEK5_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(WEEK5_DIR, "cache")
DATA_DIR  = os.path.join(WEEK5_DIR, "data")
REPORT_DIR = os.path.join(WEEK5_DIR, "report")
for d in [CACHE_DIR, DATA_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 数据切分（与 Week4 完全一致，禁止修改）─────────────────────────────────────
N_HOURS  = 3888
TRAIN_END = VAL_START = 2784
VAL_END   = TST_START = 2784 + 504
TEST_END  = 3888
TRAIN_HOURS, VAL_HOURS, TEST_HOURS = 2784, 504, 600

# ── 网格配置 ───────────────────────────────────────────────────────────────────
N_CELLS = 1024   # 北京出租车数据 32×32 格

# ── 异常注入配置（仅测试集，禁止混入训练/验证集）───────────────────────────────
@dataclass
class InjectCfg:
    seed: int = 42                           # 固定种子保证可复现
    anomaly_ratio: float = 0.04               # 测试集异常时间步占比 4%
    type_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    # 异常类型权重: 突增40% / 突降40% / 持续型20%
    spatial_mix: Tuple[float, float] = (0.5, 0.5)
    # 空间分布: 单点50% / 连片50% (3×3)
    injection_only_on_test: bool = True        # 严格限制仅测试集注入
    amplitude_range: Tuple[float, float] = (2.5, 5.0)   # 异常幅度为基线2.5~5倍
    duration_range: Tuple[int, int] = (2, 8)   # 持续时间 2~8 小时

INJECT_CFG = InjectCfg()

# ── 统计法配置 ─────────────────────────────────────────────────────────────────
@dataclass
class StatisticalCfg:
    strategy: str = "intersection"           # "intersection" 或 "union"
    use_time_groups: bool = True             # 按时段分组（hour × weekend）
    sigma_threshold: float = 3.0
    iqr_k: float = 1.5
    # 得分截断分位数（用于归一化）
    score_quantile: float = 0.99

STAT_CFG = StatisticalCfg()

# ── 重构法（VAE / Transformer AE）配置 ───────────────────────────────────────
@dataclass
class ReconstructionCfg:
    seed: int = 42
    seq_len: int = 48
    hidden_dim: int = 64
    latent_dim: int = 16
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.15                    # TAE 输入 dropout（正则）
    batch_size: int = 256                     # 单网格批次
    lr: float = 1e-3
    weight_decay: float = 1e-4               # L2 权重衰减（增强正则）
    epochs: int = 20
    patience: int = 3                       # 早停（更严格，防止过拟合）
    device: str = "cuda"
    threshold_quantile: float = 0.95         # 重构误差阈值分位数
    kl_weight: float = 0.1                   # VAE KL 散度权重
    smooth_window: int = 2

REC_CFG = ReconstructionCfg()

# ── 预测误差法配置 ──────────────────────────────────────────────────────────────
@dataclass
class PredictionCfg:
    model_name: str = "stf"                  # Week4 最优模型
    target: str = "taxi_flow_total"
    tag: str = "v4fix"
    threshold_quantile: float = 0.95          # 验证集相对误差分位数
    smooth_window: int = 2                    # 误差平滑窗口（小时）
    abs_err_floor: float = 5.0                # 预测接近0时切换绝对误差

PRED_CFG = PredictionCfg()

# ── 融合配置 ───────────────────────────────────────────────────────────────────
@dataclass
class FusionCfg:
    # 融合策略: "weighted_vote" | "score_sum"
    strategy: str = "weighted_vote"
    # 初始权重（预测误差法通常最优）
    init_weights: Dict[str, float] = field(default_factory=lambda: {
        "statistical": 0.15,
        "vae":         0.20,
        "transformer": 0.30,
        "prediction":  0.35,
    })
    weight_search_step: float = 0.05          # 网格搜索步长
    # 得分归一化：分位数截断归一化
    score_quantile: float = 0.99
    # 最终判定阈值
    decision_threshold: float = 0.5
    # 时空事件聚合
    time_gap_max: int = 1                    # 时间连通最大间隔（小时）
    spatial_connectivity: str = "4-connected"  # "4-connected" | "8-connected"

FUSION_CFG = FusionCfg()

# ── 评估配置 ────────────────────────────────────────────────────────────────────
@dataclass
class EvalCfg:
    metrics: List[str] = field(default_factory=lambda: [
        "precision", "recall", "f1", "auc_roc", "mae", "mse"
    ])
    # 分维度统计
    by_type: bool = True                     # 按异常类型分
    by_period: bool = True                   # 按时段分（早高峰/平峰/夜间）
    by_region: bool = False                  # 按网格类型分（需外部区域标注）
    # 异常类型定义
    anomaly_types: Dict[str, float] = field(default_factory=lambda: {
        "surge":       1.0,   # 突增
        "drop":       -1.0,   # 突降
        "sustained":  2.0,   # 持续型（正负双向）
    })
    # 时段定义
    time_periods: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "morning_peak":  (7, 10),
        "off_peak":      (10, 18),
        "evening_peak":  (18, 21),
        "night":         (21, 7),
    })

EVAL_CFG = EvalCfg()

# ── 缓存路径 ───────────────────────────────────────────────────────────────────
def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.npy")

def cache_json(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")

# ── 设备 ───────────────────────────────────────────────────────────────────────
# 2026-07-28: Streamlit 跑在无 torch 环境（Python310 + streamlit-only 依赖），
# 但 api/main.py 仍需 import week6.pipeline → week5.config。如果 torch 缺失，
# 回退到 "cpu" 占位（pipeline 里 torch 调用在没 GPU 时本来就是 cpu）。
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
    _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = True
