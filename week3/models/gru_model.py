"""GRU baseline — 与 LSTM 同架构，把 LSTM cell 换成 GRU."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from data_loader import SeqDataset
from base_trainer import set_seed, make_device
from metrics import BaseTrainer
from models.lstm_model import LSTMForecaster  # 仅复用基类骨架, 实际模型不同


class GRUForecaster(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, horizon, n_cells):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0)
        self.fc = nn.Linear(hidden, horizon * n_cells)
        self.horizon = horizon
        self.n_cells = n_cells

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        pred = self.fc(last)
        return pred.view(-1, self.horizon, self.n_cells)


class GRUBaseTrainer(BaseTrainer):
    name = "gru"

    def __init__(self):
        self.device = make_device()
        set_seed(cfg.cfg_train.seed)

    def fit_predict(self, flow_4d, time_features=None, target="taxi_flow_total",
                    **kwargs):
        cell_max = flow_4d[:cfg.SPLIT.train_end].max(axis=0)
        cell_max = np.maximum(cell_max, 1.0)
        normed = (flow_4d / cell_max).clip(min=0).astype(np.float32)

        common = dict(seq_len=cfg.cfg_train.seq_len, horizon=cfg.cfg_train.horizon,
                      time_features=time_features, target=target)
        train_ds = SeqDataset(normed, start=cfg.SPLIT.train_start,
                              end=cfg.SPLIT.train_end, **common)
        val_ds   = SeqDataset(normed, start=cfg.SPLIT.train_end,
                              end=cfg.SPLIT.val_end, **common)
        test_ds  = SeqDataset(normed, start=cfg.SPLIT.val_end,
                              end=cfg.SPLIT.test_end, **common)
        print(f"[GRU] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        F_in = normed.shape[1] * normed.shape[2] * normed.shape[3] + (time_features.shape[1] if time_features is not None else 0)  # 2N + K
        horizon = cfg.cfg_train.horizon
        n_cells = train_ds.target_dim   # 1024
        out_dim = horizon * n_cells
        print(f"[GRU] F_in={F_in} (2N + K), horizon={horizon}, n_cells={n_cells}")

        train_loader = DataLoader(train_ds, batch_size=cfg.cfg_train.batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.cfg_train.batch)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.cfg_train.batch)

        model = GRUForecaster(F_in, cfg.cfg_train.hidden, cfg.cfg_train.layers,
                              cfg.cfg_train.dropout, horizon, n_cells).to(self.device)
        opt = optim.Adam(model.parameters(), lr=cfg.cfg_train.lr,
                         weight_decay=cfg.cfg_train.weight_decay)
        loss_fn = nn.SmoothL1Loss()

        def reshape_for_loss(y, pred):
            if y.dim() == 3:
                B, H, N = y.shape
                y_flat = y.reshape(B, H * N)
            else:
                y_flat = y
            pred_flat = pred.reshape(pred.size(0), -1)
            return pred_flat, y_flat

        best_val = float("inf"); best_state = None; no_improve = 0
        start = time.time()
        for epoch in range(cfg.cfg_train.epochs):
            model.train()
            losses = []
            for x, y in train_loader:
                x = x.to(self.device); y = y.to(self.device)
                pred = model(x)
                pred_flat, y_flat = reshape_for_loss(y, pred)
                loss = loss_fn(pred_flat, y_flat)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            tr_loss = float(np.mean(losses))

            model.eval()
            with torch.no_grad():
                vs = []
                for x, y in val_loader:
                    x = x.to(self.device); y = y.to(self.device)
                    pred = model(x)
                    pred_flat, y_flat = reshape_for_loss(y, pred)
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

        t0 = time.time()
        all_pred = []; all_gt = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                p = model(x).cpu().numpy()        # (B, H, N)
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

        if target == "taxi_inflow":
            scale = cell_max[0].flatten()
        elif target == "taxi_outflow":
            scale = cell_max[1].flatten()
        else:
            scale = (cell_max[0] + cell_max[1]).flatten()
        pred = pred * scale[None, :]
        gt   = gt   * scale[None, :]
        print(f"[GRU] train {train_t:.1f}s test {test_t:.1f}s pred {pred.shape}")
        return pred, gt
