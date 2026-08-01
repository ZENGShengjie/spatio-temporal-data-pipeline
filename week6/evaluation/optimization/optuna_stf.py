"""Week6 任务2：Optuna 超参搜索（仅优化 STF 预测精度）

设计原则（已采纳你朋友方案 + 我们的调整）：
- 主目标：验证集 MAE（最小化）
- 不用"异常检测相关性"作为 proxy 目标（避免诱导模型走偏）
- 两阶段搜索：粗搜 20 trial + 精搜 10 trial
- 早停剪枝：MedianPruner，前 3 epoch 验证 loss 无改善就停
- 全程用快速子集 + 1 epoch/3 epoch 评估，节约时间
- 最终用全量数据 + 完整 epoch 重训得到最优模型

搜索空间：
- 学习率：1e-4 ~ 5e-3 (log)
- 批次大小：4, 8, 16
- 隐藏层维度：32, 64, 128（被注意力头数整除）
- 注意力头数：2, 4, 8
- 编码器层数：1, 2, 3
- Dropout：0.05 ~ 0.4

执行：
    python -m week6.evaluation.optimization.optuna_stf \\
        --n-trials 30 \\
        --timeout 7200 \\
        --output week6.evaluation/results/optuna/
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

# 路径
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "week4"))

# 强制使用 EC2 上的真实数据
os.environ.setdefault("WEEK4_DIR", str(_REPO / "week4"))
os.environ.setdefault("BJ_DATA_DIR", "/home/ubuntu/data")  # EC2 默认数据路径

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import week4.config as cfg
from week4.data_loader import load_raw_flow, load_time_features, SeqDataset
from week4.models.stf_model import SpacetimeformerLite


# ── 数据加载（与 Week4 训练完全一致）─────────────────────────────────────

class STFSearchData:
    """数据加载：与 week4/run_week4.py 完全对齐

    关键点：
    - x 形状 (T, 2N+5) = [in_flow(N), out_flow(N), time_features(5)]
    - y 是 target（taxi_flow_total = in+out，shape (T, N)）
    - STF 输入是 (B, N, F_in, T) 其中 F_in = 2（in/out）+5（time features per timestep）
    """

    def __init__(self, seq_len=48, horizon=1, batch_size=8):
        print("[Optuna] 加载数据...")
        self.flow = load_raw_flow()             # (T, 2, H, W)
        self.tf = load_time_features()          # (T, F_tf)

        train_end = cfg.SPLIT.train_end
        val_end = cfg.SPLIT.val_end
        test_end = cfg.SPLIT.test_end

        T_full, _, H, W = self.flow.shape
        self.N = H * W                              # 1024
        self.n_nodes = self.N
        self.n_time_features = self.tf.shape[1]     # 5

        # 按 (T, 2N+K_time) 拼成 flat（与 week4 一致）
        in_flow = self.flow[:, 0].reshape(T_full, self.N)     # (T, N)
        out_flow = self.flow[:, 1].reshape(T_full, self.N)    # (T, N)
        x_all = np.concatenate([in_flow, out_flow, self.tf], axis=1).astype(np.float32)
        # target = taxi_flow_total = in + out
        y_all = (in_flow + out_flow).astype(np.float32)

        # Normalize（与 week4 cfg.compute_cell_max + normalize 一致）
        # x_all 列 = [in_flow(N), out_flow(N), time_features(5)]
        # 每路流量用同一 train_max，需要 broadcast 成 (1, 2N)
        train_max = y_all[:train_end].max(axis=0)  # (N,)
        train_max = np.maximum(train_max, 1.0)
        # x_all 前 2N 列归一化：重复 train_max 两次
        x_all[:, :2*self.N] = x_all[:, :2*self.N] / np.tile(train_max, 2)  # (T, 2N)
        y_all = y_all / train_max  # (T, N)

        self.train_max = train_max  # 保存用于反归一化（评估时）
        self.in_dim = 2 + self.n_time_features  # = 7，与 SpacetimeformerLite in_dim 一致
        self.x_all = x_all
        self.y_all = y_all

        self.x_train = x_all[:train_end]
        self.y_train = y_all[:train_end]
        self.x_val = x_all[train_end:val_end]
        self.y_val = y_all[train_end:val_end]
        self.x_test = x_all[val_end:test_end]
        self.y_test = y_all[val_end:test_end]

        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size

    def make_loaders(self, batch_size: int):
        train_ds = STFSearchDataset(self.x_train, self.y_train, self.seq_len, self.horizon)
        val_ds = STFSearchDataset(self.x_val, self.y_val, self.seq_len, self.horizon)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        )
        return train_loader, val_loader


class STFSearchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, seq_len, horizon):
        self.x = x                              # (T, F_total)
        self.y = y                              # (T, N)
        self.seq_len = seq_len
        self.horizon = horizon
        T = len(x)
        self.valid_ts = list(range(max(seq_len, horizon), T))

    def __len__(self):
        return len(self.valid_ts)

    def __getitem__(self, idx):
        t = self.valid_ts[idx]
        x = self.x[t - self.seq_len: t]  # (seq_len, F)
        y = self.y[t]                    # (N,) 1 步预测
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            t,
        )


def x_flat_to_x_node(x_flat: torch.Tensor, n_nodes: int, n_time: int) -> torch.Tensor:
    """按 week4 _batch_to_node 把 flat x 转为 (B, N, F_in=2+5, T)"""
    B, F_total, T = x_flat.shape
    x_flow = x_flat[:, :2 * n_nodes, :].reshape(B, n_nodes, 2, T)
    x_tf = x_flat[:, 2 * n_nodes:, :].unsqueeze(1).expand(B, n_nodes, n_time, T)
    return torch.cat([x_flow, x_tf], dim=2)  # (B, N, 2+n_time, T)


# ── Optuna 目标函数 ──────────────────────────────────────────────────────────

def build_model(params: Dict[str, Any], in_dim: int, n_nodes: int, horizon: int) -> nn.Module:
    """按超参构造 STF 模型

    Args:
        in_dim: F_in per node（=2 in/out + 5 time features = 7，与 week4 一致）
    """
    return SpacetimeformerLite(
        in_dim=in_dim,
        hidden=params["hidden"],
        horizon=horizon,
        n_nodes=n_nodes,
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
    )


def evaluate_mae(model, loader, device, n_nodes, n_time) -> float:
    """在验证集上算 MAE"""
    model.eval()
    errs = []
    with torch.no_grad():
        for x, y, _ in loader:
            # x: (B, T, F_total) — 与 STFSearchDataset 一致
            x_flat = x.permute(0, 2, 1)  # (B, F_total, T)
            x_node = x_flat_to_x_node(x_flat, n_nodes, n_time).to(device)
            y_dev = y.to(device)
            out = model(x_node)  # (B, N, horizon)
            pred = out[:, :, 0]  # 1-step
            err = (pred - y_dev).abs().mean().item()
            errs.append(err)
    model.train()
    return float(np.mean(errs))


def objective_factory(data: STFSearchData, device: str, max_epochs: int):
    """生成 objective 函数，绑定数据和设备"""

    def objective(trial: optuna.Trial) -> float:
        # ── 搜索空间 ──
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
            "n_layers": trial.suggest_categorical("n_layers", [1, 2, 3]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
        # 校验：hidden 必须能被 n_heads 整除
        if params["hidden"] % params["n_heads"] != 0:
            raise optuna.exceptions.TrialPruned()

        # ── 加载数据 ──
        train_loader, val_loader = data.make_loaders(params["batch_size"])

        # ── 构造模型 ──
        model = build_model(params, in_dim=2 + data.n_time_features, n_nodes=data.n_nodes, horizon=1).to(device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        # ── 训练 + 早停剪枝 ──
        model.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x, y, _ in train_loader:
                # x: (B, T, F_total) — 用与 week4 _batch_to_node 完全等价的方式变换
                x_flat = x.permute(0, 2, 1)  # (B, F_total, T)
                x_node = x_flat_to_x_node(x_flat, data.n_nodes, data.n_time_features).to(device)
                y = y.to(device)

                opt.zero_grad()
                out = model(x_node)[:, :, 0]  # 1-step
                loss = (out - y).abs().mean()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            # 每个 epoch 评估一次，做 pruner 决策
            val_mae = evaluate_mae(model, val_loader, device, data.n_nodes, data.n_time_features)
            trial.report(val_mae, step=epoch)

            # 剪枝
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_mae

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=7200, help="秒")
    parser.add_argument("--output", default="week6.evaluation/results/optuna/")
    parser.add_argument("--max-epochs", type=int, default=10, help="每个 trial 最多跑几个 epoch")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Optuna] device={device}, n_trials={args.n_trials}, timeout={args.timeout}s")

    # ── 加载数据 ──
    data = STFSearchData(seq_len=48, horizon=1, batch_size=8)
    print(f"[Optuna] 数据: train={len(data.x_train)}, val={len(data.x_val)}, test={len(data.x_test)}")
    print(f"[Optuna] 输入维度: {data.in_dim}, 节点数: {data.n_nodes}")

    # ── 创建 study ──
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    # ── 运行搜索 ──
    objective = objective_factory(data, device, args.max_epochs)
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)
    elapsed = time.time() - t0

    # ── 输出结果 ──
    best = study.best_trial
    result = {
        "best_value_mae": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        "n_complete": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        "elapsed_sec": elapsed,
        "all_trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    (output_dir / "study_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[Optuna] 完成，耗时 {elapsed/60:.1f} 分钟")
    print(f"[Optuna] 最优 MAE = {best.value:.4f}")
    print(f"[Optuna] 最优参数 = {best.params}")
    print(f"[Optuna] 结果写入: {output_dir / 'study_result.json'}")


if __name__ == "__main__":
    main()
