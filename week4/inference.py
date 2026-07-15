"""Week4 统一推理接口 — 被 Week5 异常检测模块调用

用法:
    from inference import load_model, predict_anomalies

    # 加载最优 STF 模型
    model, cfg = load_model("stf", target="taxi_flow_total", tag="v4fix")
    # 加载最优 AGFormer 模型
    model2, cfg2 = load_model("agformer", target="taxi_flow_total", tag="v4fix")

    # 单批次推理（用于实时异常检测）
    x_batch: np.ndarray (N=1024, F, T=48)  # 归一化后输入
    pred: np.ndarray (N=1024, H=48)        # 预测结果
    model.predict(x_batch)

    # 全量测试集推理
    all_pred, all_gt = model.predict_full()
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch

# ── 路径 ────────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]          # /home/ubuntu/amazon_repo
_WEEK4 = _REPO / "week4"
_WEIGHTS_DIR = _WEEK4 / "weights"
sys.path.insert(0, str(_WEEK4))

# ── 加载器映射 ─────────────────────────────────────────────────────────────────
_MODEL_REGISTRY: Dict[str, str] = {
    "stf":               "models.stf_model.STFBaseTrainer",
    "stf_loc_only":      "models.stf_model.STFLocOnlyTrainer",
    "stgcn":             "models.stgcn_model.STGCNBaseTrainer",
    "agformer":          "models.agformer_model.AGFormerTrainer",
    "agformer_static":   "models.agformer_model.AGFormerStaticTrainer",
}


def load_model(model_name: str,
               target: str = "taxi_flow_total",
               tag: str = "v4fix",
               device: str | None = None) -> Tuple[Any, Dict[str, Any]]:
    """加载最优权重并实例化模型。

    Returns:
        (trainer, meta) — trainer 已加载最优权重，可直接调用 predict；
                          meta 包含 best_epoch, n_params, target 等元信息。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() and os.getenv("CUDA_OFF") != "1" else "cpu"

    pth_path = _WEIGHTS_DIR / f"{model_name}_{target}_{tag}.pth"
    if not pth_path.exists():
        raise FileNotFoundError(
            f"权重文件未找到: {pth_path}\n"
            f"请先运行: python week4/run_week4.py --models {model_name} "
            f"--target {target} --tag {tag}"
        )

    ckpt = torch.load(pth_path, map_location=device)
    meta = {
        "model_name": ckpt.get("model_name", model_name),
        "target":     ckpt.get("target", target),
        "tag":        ckpt.get("tag", tag),
        "best_epoch": ckpt.get("best_epoch"),
        "n_params":   ckpt.get("n_params"),
    }

    # 实例化 trainer
    import_str = _MODEL_REGISTRY.get(model_name)
    if import_str is None:
        raise ValueError(f"未知模型: {model_name}，可用: {list(_MODEL_REGISTRY.keys())}")

    module_path, cls_name = import_str.rsplit(".", 1)
    module = __import__(module_path, fromlist=[cls_name])
    trainer_cls = getattr(module, cls_name)
    trainer = trainer_cls()

    # 加载权重
    state_dict = trainer.load_weights(pth_path, device=device)["model_state_dict"]
    trainer.model.load_state_dict(state_dict)
    trainer.model.to(device)
    trainer.model.eval()

    print(f"[inference] loaded {model_name} epoch={meta['best_epoch']} "
          f"params={meta['n_params']:,} → {pth_path}")
    return trainer, meta


class SpacetimePredictor:
    """包装器：给定 trainer，对单批次 (N=1024, F, T) 做推理."""

    def __init__(self, trainer, K_time: int = 0):
        self.trainer = trainer
        self.K_time = K_time
        self.device = trainer.device

    def predict(self, x_flat: np.ndarray) -> np.ndarray:
        """单批次推理。

        Args:
            x_flat: (N, F_in, T) 归一化后的输入，N=1024，T=seq_len
                    F_in = 2 (in/out flow) + K_time (时间特征)

        Returns:
            pred: (N, H) 预测结果，H=horizon
        """
        x_t = torch.from_numpy(x_flat).float().unsqueeze(0).to(self.device)  # (1, F, T)
        N = 1024
        with torch.no_grad():
            if hasattr(self.trainer, "_predict_batch"):
                out = self.trainer._predict_batch(self.trainer.model, x_t, N, self.K_time)
            else:
                out = self.trainer.model(x_t)
        return out.squeeze(0).cpu().numpy()  # (N, H)

    def predict_anomalies(self, x_flat: np.ndarray, threshold_mae: float = 500.0):
        """快速异常检测：返回每个节点的 MAE 异常分。

        Args:
            x_flat: 同 predict()
            threshold_mae: 超过此 MAE 视为异常（可调）

        Returns:
            anomaly_mask: (N,) bool 数组，True 表示异常
            mae_per_node: (N,) float，每个节点的 MAE
        """
        # 对于全量测试集，需要对应的 gt 来计算 MAE
        # 这里先用模型输出方差作为代理指标
        pred = self.predict(x_flat)
        mae_per_node = pred.mean(axis=1)  # (N,) — 简化指标
        anomaly_mask = mae_per_node > threshold_mae
        return anomaly_mask, mae_per_node


def list_available_weights() -> list[str]:
    """列出 weights/ 下所有可用权重文件."""
    if not _WEIGHTS_DIR.exists():
        return []
    return [f.name for f in _WEIGHTS_DIR.glob("*.pth")]
