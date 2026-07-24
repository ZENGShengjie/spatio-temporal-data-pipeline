from .statistical import StatisticalAnomalyDetector
from .prediction import PredictionAnomalyDetector
from .fusion import AnomalyFusion
from .fusion_v3 import AnomalyFusionV3

# 注：vae.py / transformer_ae.py 现版本只暴露 MLPVAE / TemporalAttentionAE + Trainer，
# 不再保留旧名 VAEAnomalyDetector / TransformerAEAnomalyDetector。
# 保留旧符号以避免破坏调用方：
VAEAnomalyDetector = None  # type: ignore
TransformerAEAnomalyDetector = None  # type: ignore

__all__ = [
    'StatisticalAnomalyDetector',
    'PredictionAnomalyDetector',
    'VAEAnomalyDetector',
    'TransformerAEAnomalyDetector',
    'AnomalyFusion',
    'AnomalyFusionV3',
]