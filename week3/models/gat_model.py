"""GAT baseline V2 → ST-GAT (P1 重构, 用 PyG GATConv)

设计: 同 ST-GCN, 但空间聚合用 PyG GATConv
  - 每个时间步都做一次 attention 加权空间聚合 (PyG GATConv, 共享权重)
  - 时序 backbone: per-node GRU
  - 2 类边 (spatial + similar), 每类独立 GATConv, 然后 concat → fuse
  - PyG GATConv 要求 x.dim() == 2, 所以我们在 T 维上循环 (共享权重, 类似 ST-GCN 标准做法)
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv

import config as cfg
from data_loader import SeqDataset
from base_trainer import set_seed, make_device
from metrics import BaseTrainer


class STHeteroGATLayer(nn.Module):
    """单层 ST-GAT: 对每个时间步独立过 PyG GATConv (共享权重)

    因为 PyG GATConv 要求 2D x (单 batch), 所以我们在 T 维上循环,
    但所有时间步共享权重 — 这是 ST-GCN 的标准做法.
    """
    def __init__(self, in_dim, out_dim, edge_types, dropout=0.1, heads=1):
        super().__init__()
        self.edge_types = list(edge_types)
        self.dropout = dropout
        # 每类边一个 PyG GATConv (单头, concat=False)
        self.convs = nn.ModuleDict({
            et: GATConv(in_dim, out_dim, heads=heads,
                        dropout=dropout, add_self_loops=True,
                        negative_slope=0.2, concat=False)
            for et in self.edge_types
        })
        self.fuse = nn.Linear(out_dim * len(self.edge_types), out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_bt, edge_index_dict):
        """x_bt: (B*T, N, in_dim) — 每个时间步独立过 PyG GATConv
        注意: x_bt[i] = (N, in_dim), 一个时间步的 N 个节点
        """
        BT, N, _ = x_bt.shape
        outs_et = {et: [] for et in self.edge_types}
        # 对每个时间步独立过 PyG GATConv
        for t in range(BT):
            h_t = x_bt[t]                            # (N, in_dim)
            for et in self.edge_types:
                ei = edge_index_dict[et]
                h_et = self.convs[et](h_t, ei)       # (N, out_dim)
                outs_et[et].append(h_et)
        # 拼接: 每个 et 的 list → (BT, N, out_dim)
        h_list = []
        for et in self.edge_types:
            h_et = torch.stack(outs_et[et], dim=0)   # (BT, N, out_dim)
            h_list.append(h_et)
        h = torch.cat(h_list, dim=-1)                # (BT, N, out_dim*n_edge_types)
        h = self.fuse(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.norm(h)
        return h


class STHeteroGAT(nn.Module):
    """ST-GAT 主干: 多层 ST-HeteroGAT + 时序 GRU + per-node decoder"""
    def __init__(self, in_dim, hidden, horizon, n_layers=2, dropout=0.1,
                 heads=1, edge_types=("spatial", "similar")):
        super().__init__()
        self.edge_types = list(edge_types)
        self.hidden = hidden
        self.horizon = horizon
        self.input_proj = nn.Linear(in_dim, hidden)
        self.spatial_layers = nn.ModuleList([
            STHeteroGATLayer(hidden, hidden, self.edge_types,
                             dropout=dropout, heads=heads)
            for _ in range(n_layers)
        ])
        self.temporal = nn.GRU(input_size=hidden, hidden_size=hidden,
                                num_layers=1, batch_first=True, dropout=0.0)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )
        self.temporal_norm = nn.LayerNorm(hidden)

    def forward(self, x_node, edge_index_dict):
        """x_node: (B, N, F_in, T_seq)"""
        B, N, F_in, T_seq = x_node.shape
        x = x_node.permute(0, 1, 3, 2)              # (B, N, T, F_in)
        x = self.input_proj(x)                      # (B, N, T, hidden)
        x = F.elu(x)
        # permute 让循环顺序为 (t, b, n) — 这样每时步只需过 N*hidden 一遍 PyG
        # x: (B, N, T, hidden) → (T, B*N, hidden) — 每时步一次 GATConv 在所有 batch 上并行
        x = x.permute(2, 0, 1, 3).reshape(T_seq, B * N, self.hidden)
        for layer in self.spatial_layers:
            x = layer(x, edge_index_dict)            # (T, B*N, hidden)
        # reshape 回 (B, N, T, hidden)
        x = x.reshape(T_seq, B, N, self.hidden).permute(1, 2, 0, 3)
        # per-node GRU over T_seq
        x = x.reshape(B * N, T_seq, self.hidden)
        h_seq, _ = self.temporal(x)
        last = h_seq[:, -1, :]                      # (B*N, hidden)
        last = last.reshape(B, N, self.hidden)
        last = self.temporal_norm(last)
        out = self.decoder(last)
        return out


class GATBaseTrainer(BaseTrainer):
    name = "gat"

    def __init__(self):
        self.device = make_device()
        set_seed(cfg.cfg_train.seed)

    def fit_predict(self, flow_4d, time_features=None, target="taxi_flow_total",
                    graph=None, **kwargs):
        cell_max = flow_4d[:cfg.SPLIT.train_end].max(axis=0)
        cell_max = np.maximum(cell_max, 1.0)
        normed = (flow_4d / cell_max).clip(min=0).astype(np.float32)

        from data_loader import load_graph
        if graph is None:
            graph = load_graph()

        spatial_ei = graph["spatial", "adjacent", "spatial"].edge_index.to(self.device)
        similar_ei = graph["spatial", "similar", "spatial"].edge_index.to(self.device)
        ei_dict = {"spatial": spatial_ei, "similar": similar_ei}
        print(f"[ST-GAT-v2] spatial edges={spatial_ei.shape[1]}, similar edges={similar_ei.shape[1]}")
        N = 1024

        common = dict(seq_len=cfg.cfg_train.seq_len, horizon=cfg.cfg_train.horizon,
                      time_features=time_features, target=target)
        train_ds = SeqDataset(normed, start=cfg.SPLIT.train_start,
                              end=cfg.SPLIT.train_end, **common)
        val_ds   = SeqDataset(normed, start=cfg.SPLIT.train_end,
                              end=cfg.SPLIT.val_end, **common)
        test_ds  = SeqDataset(normed, start=cfg.SPLIT.val_end,
                              end=cfg.SPLIT.test_end, **common)
        print(f"[ST-GAT-v2] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        horizon = cfg.cfg_train.horizon
        K_time = time_features.shape[1] if time_features is not None else 0
        F_in_per_node = 2 + K_time
        print(f"[ST-GAT-v2] F_in_per_node={F_in_per_node} horizon={horizon}")

        train_loader = DataLoader(train_ds, batch_size=cfg.cfg_train.batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.cfg_train.batch)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.cfg_train.batch)

        model = STHeteroGAT(F_in_per_node, cfg.cfg_train.hidden,
                            horizon=horizon, n_layers=cfg.cfg_train.layers,
                            dropout=cfg.cfg_train.dropout).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[ST-GAT-v2] params: {n_params:,}")
        opt = optim.Adam(model.parameters(), lr=cfg.cfg_train.lr,
                         weight_decay=cfg.cfg_train.weight_decay)
        loss_fn = nn.SmoothL1Loss()

        def batch_to_node(x_flat, N, K_time):
            B, F_total, T = x_flat.shape
            x_flow = x_flat[:, :2*N, :]
            x_tf   = x_flat[:, 2*N:, :]
            x_flow = x_flow.reshape(B, N, 2, T)
            x_tf = x_tf.unsqueeze(1).expand(B, N, K_time, T)
            return torch.cat([x_flow, x_tf], dim=2)

        def run_batch(x, y):
            x_node = batch_to_node(x, N, K_time)
            B = x_node.shape[0]
            out = model(x_node, ei_dict)
            pred_loss = out.transpose(1, 2).reshape(-1, N)
            if y.dim() == 3:
                y_loss = y.transpose(1, 2).reshape(-1, N)
            else:
                y_loss = y
            return out, pred_loss, y_loss

        best_val = float("inf")
        best_state = None
        no_improve = 0
        start = time.time()
        for epoch in range(cfg.cfg_train.epochs):
            model.train()
            losses = []
            for x, y in train_loader:
                x = x.to(self.device); y = y.to(self.device)
                _, pred_loss, y_loss = run_batch(x, y)
                loss = loss_fn(pred_loss, y_loss)
                if not torch.isfinite(loss):
                    opt.zero_grad()
                    continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(loss.item())
            torch.cuda.empty_cache()
            tr_loss = float(np.mean(losses)) if losses else float("nan")

            model.eval()
            with torch.no_grad():
                vs = []
                for x, y in val_loader:
                    x = x.to(self.device); y = y.to(self.device)
                    _, pred_loss, y_loss = run_batch(x, y)
                    v = loss_fn(pred_loss, y_loss).item()
                    if np.isfinite(v):
                        vs.append(v)
            val_loss = float(np.mean(vs)) if vs else float("nan")
            if epoch % 2 == 0 or no_improve == 0:
                print(f"  [epoch {epoch:2d}] tr={tr_loss:.5f}  val={val_loss:.5f}")
            if np.isfinite(val_loss) and val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.cfg_train.patience:
                    print(f"  [early stop @ epoch {epoch}]")
                    break
        train_t = time.time() - start
        if best_state is None:
            print("  using last-epoch weights as best")
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        t0 = time.time()
        all_pred = []; all_gt = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                out, _, _ = run_batch(x, y)
                p = out.cpu().numpy()
                B, Nn, Hh = p.shape
                p = p.transpose(0, 2, 1).reshape(B * Hh, Nn)
                if y.dim() == 3:
                    B, Hh, Nn = y.shape
                    y_np = y.numpy().reshape(B * Hh, Nn)
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
        print(f"[ST-GAT-v2] train {train_t:.1f}s test {test_t:.1f}s pred {pred.shape}")
        return pred, gt
