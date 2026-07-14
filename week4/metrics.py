"""Week4 所有模型的基类 + 评估/落盘 helper"""
from __future__ import annotations
import json, os
from typing import Dict, List, Tuple
import numpy as np, torch, torch.nn as nn
import config as cfg
from config import cfg_train, masked_mae, masked_rmse, masked_mape, correlation

class BaseTrainer:
    name: str = "base"
    def fit_predict(self, flow_4d, time_features=None, target="taxi_flow_total", **kw) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

def evaluate_predictions(pred, gt, target_cols=["taxi_flow_total"]) -> Dict[str, float]:
    return {"MAE": masked_mae(pred, gt), "RMSE": masked_rmse(pred, gt),
            "MAPE": masked_mape(pred, gt), "Corr": correlation(pred, gt)}

def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    return path

def write_metrics_summary(path: str, rows: List[Dict]):
    keys = ["model", "MAE", "RMSE", "MAPE", "Corr", "n_params", "best_epoch", "train_time_s", "test_time_s"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("| " + " | ".join(["---"] * len(keys)) + " |\n")
        for row in rows:
            vals = [str(row.get(k, "-")) if not isinstance(row.get(k), float)
                    else f"{row[k]:.4f}" for k in keys]
            f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n\n```json\n")
        f.write(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        f.write("\n```\n")
    print(f"[summary] saved → {path}")

def write_top_offender_table(path: str, preds_dict, gt, top_k=20):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for name, p in preds_dict.items():
        per_cell_mae = np.abs(p - gt).mean(axis=0)
        idx = np.argsort(-per_cell_mae)[:top_k]
        rows.append({"model": name, "top_k_indices": idx.tolist(),
                     "top_k_mae": [float(per_cell_mae[i]) for i in idx]})
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Top-20 Error Grid Cells (Cross-Model)\n\n")
        for row in rows:
            f.write(f"## {row['model']}\n\n")
            f.write("| Rank | grid_idx | MAE |\n|----|---|---|\n")
            for r, (i, m) in enumerate(zip(row["top_k_indices"], row["top_k_mae"]), 1):
                f.write(f"| {r} | {i} | {m:.4f} |\n")
            f.write("\n")
    print(f"[offenders] saved → {path}")
