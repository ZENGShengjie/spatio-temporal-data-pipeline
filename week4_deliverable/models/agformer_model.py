"""AGFormer — Adaptive Graph Transformer

Architecture:
  ① Adaptive adj initialized from Beijing static graph (alpha=0.1 bias)
  ② Top-K spatial attention (K=16) — sparse, no full N²
  ③ 2 alternating SpatialAttn → TemporalAttn → FFN blocks
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys, os
_pkg_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _pkg_root)
sys.path.insert(0, os.path.dirname(_pkg_root))
sys.path.insert(0, os.path.dirname(os.path.dirname(_pkg_root)))

import config as cfg
from data_loader import SeqDataset
from base_trainer import set_seed, make_device
from metrics import BaseTrainer
try:
    from stgcn_model import GCNBaseTrainer
except ImportError:
    from models.stgcn_model import GCNBaseTrainer


def build_static_adj_bias(graph, n_nodes=1024, alpha=0.1):
    """Static graph → symmetric normalized adj → adaptive adj init bias."""
    try:
        import scipy.sparse as sp
    except ImportError:
        return torch.zeros(n_nodes, n_nodes)

    def ei_to_sparse(ei, n):
        v = torch.ones(ei.shape[1])
        return sp.csr_matrix((v.numpy(), (ei[0].numpy(), ei[1].numpy())), shape=(n,n))

    g = graph
    spatial = ei_to_sparse(g["spatial","adjacent","spatial"].edge_index, n_nodes)
    similar = ei_to_sparse(g["spatial","similar","spatial"].edge_index, n_nodes)
    A = spatial + similar; A.setdiag(1.0)
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(np.maximum(d, 1e-6), -0.5)
    A_norm = sp.diags(d_inv_sqrt) @ A @ sp.diags(d_inv_sqrt)
    return torch.tensor(A_norm.toarray(), dtype=torch.float32) * alpha


class TopKSpatialAttention(nn.Module):
    """Each node attends only to its Top-K most related neighbors."""
    def __init__(self, d_model, n_heads, n_nodes=1024, K=16, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.K = K; self.n_nodes = n_nodes
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.adaptive_adj = nn.Parameter(torch.zeros(n_nodes, n_nodes))

    def init_from_static(self, A_bias):
        with torch.no_grad(): self.adaptive_adj.copy_(A_bias)

    def forward(self, x):
        """Top-K sparse attention over the spatial dim.
        Memory-efficient: only compute K-slot attention per query.
        Input x: (BT, N, d_model)
        """
        BT, N, C = x.shape
        H = self.n_heads; d_k = self.d_k
        Q = self.Wq(x).view(BT, N, H, d_k).permute(0, 2, 1, 3)  # (BT, H, N, d_k)
        K_ = self.Wk(x).view(BT, N, H, d_k).permute(0, 2, 1, 3)
        V = self.Wv(x).view(BT, N, H, d_k).permute(0, 2, 1, 3)

        # Top-K neighbor indices from adaptive adj
        topk_idx = torch.topk(self.adaptive_adj, self.K, dim=-1).indices  # (N, K)
        # Gather K_ and V at the K neighbor positions for each query
        # index expanded: (1, 1, N, K) -> for gather on dim=-2 of (BT, H, N, d_k)
        idx = topk_idx.view(1, 1, N, self.K).expand(BT, H, N, self.K)  # (BT, H, N, K)
        # gather over dim=2 (node dim). Result: (BT, H, N, K, d_k)
        K_topk = K_.unsqueeze(3).expand(BT, H, N, self.K, d_k).gather(
            2, idx.unsqueeze(-1).expand(BT, H, N, self.K, d_k))
        V_topk = V.unsqueeze(3).expand(BT, H, N, self.K, d_k).gather(
            2, idx.unsqueeze(-1).expand(BT, H, N, self.K, d_k))

        # Q at query (BT, H, N, 1, d_k), K_topk at neighbors (BT, H, N, K, d_k)
        Qh = Q.unsqueeze(3)  # (BT, H, N, 1, d_k)
        scores = (Qh * K_topk).sum(-1) / (d_k ** 0.5)   # (BT, H, N, K)
        attn = torch.softmax(scores, dim=-1)              # (BT, H, N, K)
        attn = self.drop(attn)
        # weighted sum
        out = (attn.unsqueeze(-1) * V_topk).sum(dim=3)    # (BT, H, N, d_k)
        out = out.permute(0, 2, 1, 3).contiguous().view(BT, N, -1)
        return self.proj(out)


class AGFormerBlock(nn.Module):
    """1 block: spatial attention → temporal attention → FFN + residual LN"""
    def __init__(self, d_model, n_heads, n_nodes=1024, K=16, dropout=0.1):
        super().__init__()
        self.spatial_attn = TopKSpatialAttention(d_model, n_heads, n_nodes, K, dropout)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bt_n, B, T):
        # spatial attn
        h = self.norm1(x_bt_n)
        h = x_bt_n + self.drop(self.spatial_attn(h))
        # temporal attn
        h2 = h.reshape(B, T, h.shape[1], -1).permute(0,2,1,3).reshape(B*h.shape[1], T, -1)
        h2 = self.norm2(h2)
        attn_out, _ = self.temporal_attn(h2, h2, h2)
        h2 = h2 + self.drop(attn_out)
        h2 = h2.reshape(B, h.shape[1], T, -1).permute(0,2,1,3).reshape(B*T, h.shape[1], -1)
        # ffn
        h2 = self.norm3(h2)
        return h2 + self.drop(self.ffn(h2))


class AGFormer(nn.Module):
    def __init__(self, in_dim, hidden, horizon,
                 n_layers=2, n_heads=4, K=16, dropout=0.1, n_nodes=1024):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            AGFormerBlock(hidden, n_heads, n_nodes, K, dropout)
            for _ in range(n_layers)
        ])
        self.temporal = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, dropout=0.0)
        self.temporal_ln = nn.LayerNorm(hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x_node, B, T):
        x = x_node.permute(0,1,3,2)
        x = F.gelu(self.input_proj(x))
        for block in self.blocks:
            x_flat = x.reshape(B*T, x.shape[1], -1)
            h_flat = block(x_flat, B, T)
            x = h_flat.reshape(B, T, h_flat.shape[1], -1).permute(0,2,1,3)
        # x is (B, N, T, d). Reshape to (B*N, T, d) for per-node GRU over T
        N = x.shape[1]
        x = x.reshape(B * N, T, -1)
        _, h_last = self.temporal(x)
        last = h_last.squeeze(0).reshape(B, N, -1)
        return self.decoder(self.temporal_ln(last))


class AGFormerTrainer(GCNBaseTrainer):
    name = "agformer"

    def fit_predict(self, flow_4d, time_features=None,
                    target="taxi_flow_total", graph=None, **kwargs):
        from data_loader import load_graph
        if graph is None: graph = load_graph()

        cell_max = flow_4d[:cfg.SPLIT.train_end].max(axis=0)
        cell_max = np.maximum(cell_max, 1.0)
        normed = (flow_4d / cell_max).clip(min=0).astype(np.float32)

        spatial_ei = graph["spatial","adjacent","spatial"].edge_index.to(self.device)
        similar_ei = graph["spatial","similar","spatial"].edge_index.to(self.device)
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

        train_loader = DataLoader(train_ds, batch_size=cfg.cfg_train.batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.cfg_train.batch)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.cfg_train.batch)

        model = AGFormer(F_in_per_node, cfg.cfg_train.hidden, horizon,
                         n_layers=2, n_heads=4, K=16,
                         dropout=cfg.cfg_train.dropout).to(self.device)

        A_bias = build_static_adj_bias(graph, N, alpha=0.1)
        for block in model.blocks:
            block.spatial_attn.init_from_static(A_bias.to(self.device))
        print(f"  [AGFormer] adaptive adj initialized from static graph (alpha=0.1)")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [model] AGFormer params: {n_params:,}")

        # separate LR for adaptive adj
        opt = torch.optim.Adam([
            {"params": (p for n, p in model.named_parameters() if "adaptive_adj" not in n)},
            {"params": (p for n, p in model.named_parameters() if "adaptive_adj" in n),
             "lr": cfg.cfg_train.lr * 0.1},
        ], lr=cfg.cfg_train.lr, weight_decay=cfg.cfg_train.weight_decay)
        loss_fn = nn.SmoothL1Loss()

        def batch_to_node(x_flat, N, K_time):
            B, F_total, T = x_flat.shape
            x_flow = x_flat[:,:2*N,:].reshape(B, N, 2, T)
            x_tf   = x_flat[:,2*N:,:].unsqueeze(1).expand(B, N, K_time, T)
            return torch.cat([x_flow, x_tf], dim=2)

        best_val = float("inf"); best_state = None; no_improve = 0
        start = time.time()

        for epoch in range(cfg.cfg_train.epochs):
            model.train(); losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                B, F_total, T = x.shape
                x_node = batch_to_node(x, N, K_time)
                out = model(x_node, B, T)
                pred_loss = out.transpose(1,2).reshape(-1, N)
                y_loss = y.transpose(1,2).reshape(-1,N) if y.dim()==3 else y
                loss = loss_fn(pred_loss, y_loss)
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
                    B, F_total, T = x.shape
                    x_node = batch_to_node(x, N, K_time)
                    out = model(x_node, B, T)
                    pred_loss = out.transpose(1,2).reshape(-1,N)
                    y_loss = y.transpose(1,2).reshape(-1,N) if y.dim()==3 else y
                    v = loss_fn(pred_loss, y_loss).item()
                    if np.isfinite(v): vs.append(v)
            val_loss = float(np.mean(vs)) if vs else float("nan")

            if epoch % 5 == 0 or no_improve == 0:
                print(f"    epoch {epoch:3d}  tr={tr_loss:.5f}  val={val_loss:.5f}")
            if np.isfinite(val_loss) and val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.cfg_train.patience:
                    print(f"    early stop @ epoch {epoch}"); break

        train_t = time.time() - start
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        t0 = time.time(); all_pred, all_gt = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                B, F_total, T = x.shape
                x_node = batch_to_node(x, N, K_time)
                out = model(x_node, B, T)
                p = out.cpu().numpy()
                Bp, Nn, Hh = p.shape
                p = p.transpose(0,2,1).reshape(Bp*Hh, Nn)
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


class AGFormerStaticTrainer(AGFormerTrainer):
    """Ablation: freeze adaptive adj at static graph initialization."""
    name = "agformer_static"
