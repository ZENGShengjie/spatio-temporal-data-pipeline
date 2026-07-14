"""STGCN — Sandwich Block: GCNConv + causal GLU-TCN

Architecture:
  Input: (B, N=1024, F_in, T_seq=48)
  ├── input_proj
  ├── [STBlock × 4]  spatial GCNConv → causal GLU-TCN
  ├── per-node GRU over T
  └── decoder → (B, N, horizon=48)
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv

import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/..")
import config as cfg
from data_loader import SeqDataset
from base_trainer import set_seed, make_device
from metrics import BaseTrainer


class GCNBaseTrainer(BaseTrainer):
    """Shared trainer skeleton for all GNN models."""
    name = "gcn"

    def __init__(self):
        self.device = make_device()
        set_seed(cfg.cfg_train.seed)

    def fit_predict(self, flow_4d, time_features=None,
                    target="taxi_flow_total", graph=None, **kwargs):
        cell_max = flow_4d[:cfg.SPLIT.train_end].max(axis=0)
        cell_max = np.maximum(cell_max, 1.0)
        normed = (flow_4d / cell_max).clip(min=0).astype(np.float32)

        from data_loader import load_graph
        if graph is None:
            graph = load_graph()

        spatial_ei = graph["spatial","adjacent","spatial"].edge_index.to(self.device)
        similar_ei = graph["spatial","similar","spatial"].edge_index.to(self.device)
        ei_dict = {"spatial": spatial_ei, "similar": similar_ei}
        N = 1024

        common = dict(seq_len=cfg.cfg_train.seq_len, horizon=cfg.cfg_train.horizon,
                      time_features=time_features, target=target)
        train_ds = SeqDataset(normed, start=cfg.SPLIT.train_start, end=cfg.SPLIT.train_end, **common)
        val_ds   = SeqDataset(normed, start=cfg.SPLIT.train_end,   end=cfg.SPLIT.val_end,   **common)
        test_ds  = SeqDataset(normed, start=cfg.SPLIT.val_end,     end=cfg.SPLIT.test_end,  **common)
        print(f"  [data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

        horizon = cfg.cfg_train.horizon
        K_time = time_features.shape[1] if time_features is not None else 0
        F_in_per_node = 2 + K_time
        print(f"  [data] F_in={F_in_per_node}, horizon={horizon}")

        train_loader = DataLoader(train_ds, batch_size=cfg.cfg_train.batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.cfg_train.batch)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.cfg_train.batch)

        model = self._build_model(F_in_per_node, horizon, ei_dict).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [model] {model.__class__.__name__} params: {n_params:,}")

        opt = torch.optim.Adam(model.parameters(), lr=cfg.cfg_train.lr,
                               weight_decay=cfg.cfg_train.weight_decay)
        loss_fn = nn.SmoothL1Loss()

        run_batch_fn = self._make_run_batch(N, K_time, model, loss_fn)

        best_val = float("inf"); best_state = None; no_improve = 0; best_epoch = -1
        start = time.time()

        for epoch in range(cfg.cfg_train.epochs):
            model.train(); losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = run_batch_fn(x, y)
                if not torch.isfinite(loss): opt.zero_grad(); continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); losses.append(loss.item())
            tr_loss = float(np.mean(losses)) if losses else float("nan")

            model.eval()
            with torch.no_grad():
                vs = []
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    v = run_batch_fn(x, y).item()
                    if np.isfinite(v): vs.append(v)
            val_loss = float(np.mean(vs)) if vs else float("nan")

            if epoch % 5 == 0 or no_improve == 0:
                print(f"    epoch {epoch:3d}  tr={tr_loss:.5f}  val={val_loss:.5f}")
            if np.isfinite(val_loss) and val_loss < best_val - 1e-5:
                best_val = val_loss; best_epoch = epoch
                self.n_params = n_params
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.cfg_train.patience:
                    print(f"    early stop @ epoch {epoch}"); break

        train_t = time.time() - start
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.best_epoch = best_epoch
        model.load_state_dict(best_state)

        t0 = time.time(); all_pred, all_gt = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                out = self._predict_batch(model, x, N, K_time)
                p = out.cpu().numpy()
                B, Nn, Hh = p.shape
                p = p.transpose(0,2,1).reshape(B*Hh, Nn)
                B2, Hh2, Nn2 = y.shape
                y_np = y.numpy().reshape(B2*Hh2, Nn2)
                all_pred.append(p); all_gt.append(y_np)
        pred = np.concatenate(all_pred, axis=0); gt = np.concatenate(all_gt, axis=0)
        test_t = time.time() - t0

        if target == "taxi_inflow":   scale = cell_max[0].flatten()
        elif target == "taxi_outflow": scale = cell_max[1].flatten()
        else:                           scale = (cell_max[0]+cell_max[1]).flatten()
        pred = pred * scale[None,:]; gt = gt * scale[None,:]
        print(f"  [done] train={train_t:.1f}s test={test_t:.1f}s pred={pred.shape}")
        return pred, gt

    def _build_model(self, in_dim, horizon, ei_dict): raise NotImplementedError
    def _make_run_batch(self, N, K_time, model, loss_fn): raise NotImplementedError
    def _predict_batch(self, model, x, N, K_time): raise NotImplementedError


# ── GLU-TCN ──────────────────────────────────────────────────────────────

class GLUTCN(nn.Module):
    """Causal GLU-TCN: pad left only, no future leakage."""
    def __init__(self, c, kernel=3, dropout=0.1):
        super().__init__()
        self.pad_l = kernel - 1
        self.conv = nn.Conv1d(c, c * 2, kernel, padding=0)
        self.norm = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.pad(x, (self.pad_l, 0))
        h = self.conv(h)
        l, r = h.chunk(2, dim=1)
        return self.norm(self.drop(l * torch.sigmoid(r))) + x


# ── STGCN Block ───────────────────────────────────────────────────────────

class STGCNBlock(nn.Module):
    """Sandwich Block: spatial GCNConv → causal GLU-TCN + residual + LN"""
    def __init__(self, hidden, edge_types=("spatial","similar"), kernel=3, dropout=0.1):
        super().__init__()
        self.edge_types = list(edge_types)
        self.spatial = nn.ModuleDict({
            et: GCNConv(hidden, hidden, cached=False, normalize=True, add_self_loops=True)
            for et in self.edge_types
        })
        self.fuse = nn.Linear(hidden * len(self.edge_types), hidden)
        self.spatial_ln = nn.LayerNorm(hidden)
        self.temporal = GLUTCN(hidden, kernel=kernel, dropout=dropout)
        self.temporal_ln = nn.LayerNorm(hidden)

    def forward(self, x, edge_index_dict):
        # spatial
        outs = [self.spatial[et](x, edge_index_dict[et]) for et in self.edge_types]
        h = self.fuse(torch.cat(outs, dim=-1))
        h = F.gelu(h)
        h = self.spatial_ln(h)
        h = h + x
        # temporal
        BTN, N, C = h.shape
        h = h.permute(1,2,0).reshape(N, C, BTN)
        h = self.temporal(h)
        h = h.reshape(N, C, BTN).permute(2,0,1)
        h = self.temporal_ln(h)
        return h + x


# ── STGCN Model ───────────────────────────────────────────────────────────

class STGCN(nn.Module):
    def __init__(self, in_dim, hidden, horizon,
                 n_layers=4, kernel=3, dropout=0.1, edge_types=("spatial","similar")):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            STGCNBlock(hidden, edge_types, kernel=kernel, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.temporal = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, dropout=0.0)
        self.temporal_ln = nn.LayerNorm(hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, horizon), nn.ReLU())

    def forward(self, x_node, edge_index_dict):
        B, N, F_in, T = x_node.shape
        x = x_node.permute(0,1,3,2)
        x = F.gelu(self.input_proj(x))
        for block in self.blocks:
            x_flat = x.reshape(B*T, N, -1)
            h_flat = block(x_flat, edge_index_dict)
            x = h_flat.reshape(B, T, N, -1).permute(0,2,1,3)
        x = x.reshape(B*N, T, -1)
        _, h_last = self.temporal(x)
        last = h_last.squeeze(0).reshape(B, N, -1)
        return self.decoder(self.temporal_ln(last))


# ── STGCN Trainer ─────────────────────────────────────────────────────────

class STGCNBaseTrainer(GCNBaseTrainer):
    name = "stgcn"

    def _build_model(self, in_dim, horizon, ei_dict):
        return STGCN(in_dim, cfg.cfg_train.hidden, horizon,
                     n_layers=4, dropout=cfg.cfg_train.dropout)

    def _make_run_batch(self, N, K_time, model, loss_fn):
        def run_batch(x, y):
            x_node = self._batch_to_node(x, N, K_time)
            out = model(x_node, self._ei_dict)
            pred_loss = out.transpose(1,2).reshape(-1, N)
            y_loss = y.transpose(1,2).reshape(-1,N) if y.dim()==3 else y
            return loss_fn(pred_loss, y_loss)
        return run_batch

    def _predict_batch(self, model, x, N, K_time):
        return model(self._batch_to_node(x, N, K_time), self._ei_dict)

    def fit_predict(self, flow_4d, time_features=None,
                    target="taxi_flow_total", graph=None, **kwargs):
        from data_loader import load_graph
        if graph is None: graph = load_graph()
        self._ei_dict = {
            "spatial": graph["spatial","adjacent","spatial"].edge_index.to(self.device),
            "similar": graph["spatial","similar","spatial"].edge_index.to(self.device),
        }
        return super().fit_predict(flow_4d, time_features, target, graph, **kwargs)

    @staticmethod
    def _batch_to_node(x_flat, N, K_time):
        B, F_total, T = x_flat.shape
        x_flow = x_flat[:,:2*N,:].reshape(B, N, 2, T)
        x_tf   = x_flat[:,2*N:,:].unsqueeze(1).expand(B, N, K_time, T)
        return torch.cat([x_flow, x_tf], dim=2)
