"""Week6 任务2：用 Optuna 找出的最优参数重训 STF，写到 weights/ 目录

输入：results/optuna/study_result.json
输出：week4/weights/stf_optuna.pth + 完整训练日志

执行：
    python -m week6_evaluation.optimization.retrain_best \\
        --study-result week6_evaluation/results/optuna/study_result.json \\
        --epochs 30
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "week4"))

os.environ.setdefault("WEEK4_DIR", str(_REPO / "week4"))
os.environ.setdefault("BJ_DATA_DIR", "/home/ubuntu/data")

import week4.config as cfg
from week4.models.stf_model import SpacetimeformerLite
from week6_evaluation.optimization.optuna_stf import STFSearchData, evaluate_mae, x_flat_to_x_node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-result", required=True, help="Optuna study_result.json 路径")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", default="week4/weights/stf_optuna.pth")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.is_dir():
        out_path = out_path / "stf_optuna.pth"
        print(f"[Retrain] output is dir, using {out_path}")
    args.output = str(out_path)

    with open(args.study_result, "r", encoding="utf-8") as f:
        study = json.load(f)
    params = study["best_params"]
    print(f"[Retrain] 使用最优参数: {params}")

    # Ensure output directory exists
    out_path = Path(args.output)
    os.makedirs(out_path.parent, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Retrain] device={device}")

    # ── 加载数据 ──
    data = STFSearchData(seq_len=48, horizon=1, batch_size=params["batch_size"])
    train_loader, val_loader = data.make_loaders(params["batch_size"])

    # ── 构造模型 ──
    # Ensure output directory exists (guard for direct python -c invocation)
    _out_path = Path(args.output)
    os.makedirs(_out_path.parent, exist_ok=True)

    model = SpacetimeformerLite(
        in_dim=2 + data.n_time_features,
        hidden=params["hidden"],
        horizon=1,
        n_nodes=data.n_nodes,
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params.get("weight_decay", 1e-5),
    )

    # ── 训练 ──
    history = []
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, y, _ in train_loader:
            x_flat = x.permute(0, 2, 1)  # (B, F_total, T)
            x_node = x_flat_to_x_node(x_flat, data.n_nodes, data.n_time_features).to(device)
            y = y.to(device)

            opt.zero_grad()
            out = model(x_node)[:, :, 0]
            loss = (out - y).abs().mean()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1

        val_mae = evaluate_mae(model, val_loader, device, data.n_nodes, data.n_time_features)
        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
            "val_mae": val_mae,
            "elapsed_sec": epoch_time,
        })
        print(f"[Retrain] epoch {epoch:3d}  train_loss={train_loss/n_batches:.4f}  val_mae={val_mae:.4f}  elapsed={epoch_time:.0f}s")

        if val_mae < best_val:
            best_val = val_mae
            torch.save({
                "model_state_dict": model.state_dict(),
                "params": params,
                "val_mae": val_mae,
                "epoch": epoch,
            }, args.output)
            print(f"[Retrain] [SAVE] {args.output}  val_mae={val_mae:.4f}")

    # ── 输出训练历史 ──
    history_path = Path(args.output).with_suffix(".history.json")
    history_path.write_text(
        json.dumps({
            "params": params,
            "history": history,
            "best_val_mae": best_val,
            "total_epochs": args.epochs,
            "elapsed_sec": time.time() - t0,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[Retrain] 完成，最优 val_mae={best_val:.4f}")
    print(f"[Retrain] 训练历史: {history_path}")


if __name__ == "__main__":
    main()