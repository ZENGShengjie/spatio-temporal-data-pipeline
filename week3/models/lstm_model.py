"""LSTM baseline — city-level sequence-to-sequence (one-step ahead)

输入:  x: (B, F_in, T_seq)     F_in = 2*N + K_global
输出:  y: (B, N)               预测 horizon 步后的 taxi_flow_total

训练模式 (loop inside):
  - 每个 batch: 全 1024 cell 同时训练
  - 评估: MAE / RMSE / MAPE / Corr
"""
from __future__ import annotations

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from data_loader import SeqDataset
from base_trainer import set_seed, make_device
from metrics import BaseTrainer


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float,
                 horizon: int, n_cells: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0)
        # V2: 多步输出 → 把所有 horizon 步一次性预测 (teaching forced)
        self.fc = nn.Linear(hidden, horizon * n_cells)
        self.horizon = horizon
        self.n_cells = n_cells

    def forward(self, x):
        # x: (B, F_in, T_seq)
        x = x.transpose(1, 2)            # (B, T_seq, F_in)
        out, _ = self.lstm(x)            # (B, T_seq, H)
        last = out[:, -1, :]
        pred = self.fc(last)             # (B, horizon * n_cells)
        return pred.view(-1, self.horizon, self.n_cells)   # (B, H, N)


class LSTMBaseTrainer(BaseTrainer):
    name = "lstm"

    def __init__(self):
        self.device = make_device()
        set_seed(cfg.cfg_train.seed)

    def fit_predict(self, flow_4d, time_features=None, target="taxi_flow_total",
                    **kwargs):
        # 1. 标准化（按 cell 历史最大值）
        cell_max = flow_4d[:cfg.SPLIT.train_end].max(axis=0)  # (2, H, W)
        cell_max = np.maximum(cell_max, 1.0)
        normed = (flow_4d / cell_max).clip(min=0).astype(np.float32)

        # 2. Datasets
        common = dict(seq_len=cfg.cfg_train.seq_len, horizon=cfg.cfg_train.horizon,
                      time_features=time_features, target=target)
        train_ds = SeqDataset(normed, start=cfg.SPLIT.train_start,
                              end=cfg.SPLIT.train_end, **common)
        val_ds   = SeqDataset(normed, start=cfg.SPLIT.train_end,
                              end=cfg.SPLIT.val_end, **common)
        test_ds  = SeqDataset(normed, start=cfg.SPLIT.val_end,
                              end=cfg.SPLIT.test_end, **common)
        print(f"[LSTM] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        # 只用全局时间特征最后一刻拼接在 fc head 之前的方案，我们用更直接的：不在 LSTM 的
        # 序列输入中加 tf（避免 input_size 过大），而是直接把 tf 拼到 fc head 的输出端。
        # 这里简化：把 time_features 直接 ignore，由 LSTM/GRU 隐式学时间周期性
        # V2: F_in = 2N + K (K 个时间特征已被 SeqDataset 拼到 x 上)
        F_in = normed.shape[1] * normed.shape[2] * normed.shape[3] + (time_features.shape[1] if time_features is not None else 0)  # 2N + K
        horizon = cfg.cfg_train.horizon
        n_cells = train_ds.target_dim    # 1024
        out_dim = horizon * n_cells      # V2 多步输出
        print(f"[LSTM] F_in={F_in} (2N + K), horizon={horizon}, n_cells={n_cells}, out_dim={out_dim}")

        train_loader = DataLoader(train_ds, batch_size=cfg.cfg_train.batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.cfg_train.batch)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.cfg_train.batch)

        model = LSTMForecaster(F_in, cfg.cfg_train.hidden, cfg.cfg_train.layers,
                               cfg.cfg_train.dropout, horizon, n_cells).to(self.device)
        opt = optim.Adam(model.parameters(), lr=cfg.cfg_train.lr,
                         weight_decay=cfg.cfg_train.weight_decay)
        loss_fn = nn.SmoothL1Loss()

        best_val = float("inf"); best_state = None; no_improve = 0
        start = time.time()
        for epoch in range(cfg.cfg_train.epochs):
            model.train()
            losses = []
            for x, y in train_loader:
                x = x.to(self.device); y = y.to(self.device)
                # y: (B, H, N)
                if y.dim() == 3:
                    B, H, N = y.shape
                    y_flat = y.reshape(B, H * N)
                else:  # 单步 fallback
                    y_flat = y
                pred = model(x)            # (B, H, N)
                pred_flat = pred.reshape(pred.size(0), -1)   # (B, H*N)
                loss = loss_fn(pred_flat, y_flat)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            tr_loss = float(np.mean(losses))

            model.eval()
            with torch.no_grad():
                vs = []
                for x, y in val_loader:
                    x = x.to(self.device); y = y.to(self.device)
                    if y.dim() == 3:
                        B, H, N = y.shape
                        y_flat = y.reshape(B, H * N)
                    else:
                        y_flat = y
                    pred = model(x)
                    pred_flat = pred.reshape(pred.size(0), -1)
                    vs.append(loss_fn(pred_flat, y_flat).item())
            val_loss = float(np.mean(vs))
            if epoch % 2 == 0 or no_improve == 0:
                print(f"  [epoch {epoch:2d}] tr={tr_loss:.5f}  val={val_loss:.5f}")
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.cfg_train.patience:
                    print(f"  [early stop @ epoch {epoch}]")
                    break
        train_t = time.time() - start
        model.load_state_dict(best_state)

        # test inference
        t0 = time.time()
        all_pred = []; all_gt = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                p = model(x).cpu().numpy()      # (B, H, N)
                if p.ndim == 3:
                    B, H, N = p.shape
                    p = p.reshape(B * H, N)
                if y.dim() == 3:
                    B, H, N = y.shape
                    y_np = y.numpy().reshape(B * H, N)
                else:
                    y_np = y.numpy()
                all_pred.append(p); all_gt.append(y_np)
        test_t = time.time() - t0
        pred = np.concatenate(all_pred, axis=0)
        gt   = np.concatenate(all_gt,   axis=0)

        # 反归一化
        if target == "taxi_inflow":
            scale = cell_max[0].flatten()
        elif target == "taxi_outflow":
            scale = cell_max[1].flatten()
        else:
            scale = (cell_max[0] + cell_max[1]).flatten()
        pred = pred * scale[None, :]
        gt   = gt   * scale[None, :]
        print(f"[LSTM] train {train_t:.1f}s test {test_t:.1f}s pred {pred.shape}")
        return pred, gt
