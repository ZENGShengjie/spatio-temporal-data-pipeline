"""Week3 所有模型的基类 + 评估/落盘 helper"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import config as cfg
from config import cfg_train, masked_mae, masked_rmse, masked_mape, correlation


# ============================================================
# 模型基类
# ============================================================
class BaseTrainer:
    """
    统一的模型接口 — 6 个模型都实现:
      .name : str
      .fit_predict(flow_4d, time_features) -> (pred, gt) npy 保存
    """
    name: str = "base"

    def fit_predict(self, flow_4d: np.ndarray, time_features: np.ndarray | None = None,
                    target: str = "taxi_flow_total") -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# ============================================================
# 评估模块：批量跑模型并产出统一格式的对比表
# ============================================================
def evaluate_predictions(pred: np.ndarray, gt: np.ndarray,
                         target_cols: List[str] = ["taxi_flow_total"]) -> Dict[str, float]:
    """pred, gt: (n_test, N=1024)"""
    metrics = {}
    metrics["MAE"]  = masked_mae(pred, gt)
    metrics["RMSE"] = masked_rmse(pred, gt)
    metrics["MAPE"] = masked_mape(pred, gt)
    metrics["Corr"] = correlation(pred, gt)
    return metrics


def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    return path


def model_signature(model_name: str,
                    seq_len: int = cfg_train.seq_len,
                    epochs: int = cfg_train.epochs,
                    **extra) -> Dict:
    return {
        "model": model_name,
        "seq_len": seq_len,
        "epochs": epochs,
        **extra,
    }


def write_metrics_summary(path: str, rows: List[Dict]):
    """
    rows: list of {model: str, MAE: float, RMSE: float, MAPE: float, Corr: float,
                   train_time_s: float, test_time_s: float, extras: dict}
    """
    keys = ["model", "MAE", "RMSE", "MAPE", "Corr",
            "train_time_s", "test_time_s"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # 写 markdown 表
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("| " + " | ".join(["---"] * len(keys)) + " |\n")
        for row in rows:
            vals = []
            for k in keys:
                v = row.get(k)
                if v is None:
                    vals.append("-")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")
        # 写 json
        f.write("\n\n```json\n")
        f.write(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        f.write("\n```\n")
    print(f"[summary] saved → {path}")


def write_top_offender_table(path: str, preds_dict: Dict[str, np.ndarray],
                             gt: np.ndarray, top_k: int = 20):
    """选 worst-K cells by per-model MAE，写成 markdown 表格"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for name, p in preds_dict.items():
        per_cell_mae = np.abs(p - gt).mean(axis=0)        # (N,)
        idx = np.argsort(-per_cell_mae)[:top_k]
        row = {"model": name,
               "top_k_indices": idx.tolist(),
               "top_k_mae": [float(per_cell_mae[i]) for i in idx]}
        rows.append(row)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Top-20 误差最高的网格（跨模型）\n\n")
        for row in rows:
            f.write(f"## {row['model']}\n\n")
            f.write("| Rank | grid_idx | MAE (mean over test) |\n")
            f.write("|----|---|---|\n")
            for r, (i, m) in enumerate(zip(row["top_k_indices"], row["top_k_mae"]), 1):
                f.write(f"| {r} | {i} | {m:.4f} |\n")
            f.write("\n")
    print(f"[offenders] saved → {path}")
