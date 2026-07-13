"""Spacetimeformer Lite — Env + Loc factorized attention

Architecture:
  ① Env token: mean_pool(nodes) → transformer over T  (global periodicity)
  ② Loc token: per-node linear proj over T            (local dynamics)
  ③ Cross attention: Loc query × Env key/value        (global → local)
  ④ Conv1d temporal pooling
  ⑤ MLP decoder → horizon

Token count: Env=B×T=16×48=768, Loc=B×N×T=16×1024×48=784K,
  but cross-attn complexity = O(T²·d) not O(N²·d) → tractable on single GPU.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/..")
import config as cfg
from data_loader import SeqDataset
from base_trainer import set_seed, make_device
from metrics import BaseTrainer
from stgcn_model import GCNBaseTrainer


class SpacetimeformerLite(nn.Module):
    def __init__(self, in_dim, hidden, horizon,
                 n_nodes=1024, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes; self.hidden = hidden

        # Env encoder: compress N nodes → global temporal transformer
        self.env_proj = nn.Linear(n_nodes, hidden)
        self.env_pos = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            hidden, n_heads, dim_feedforward=hidden*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.env_transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Loc encoder: per-node per-timestep linear proj
        self.loc_proj = nn.Linear(in_dim, hidden)
        self.loc_pos = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        # Cross attention: Loc → Env
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.cross_ln = nn.LayerNorm(hidden)

        # Local temporal: Conv1d over T
        self.loc_temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.loc_temporal_ln = nn.LayerNorm(hidden)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*2, horizon))

    def forward(self, x_node):
        B, N, F_in, T = x_node.shape

        # Env: mean pool over nodes → transformer
        x_env = x_node.mean(dim=1).permute(0,2,1)        # (B, T, F=2)
        x_env = self.env_proj(x_env) + self.env_pos     # (B, T, d)
        env_seq = self.env_transformer(x_env)            # (B, T, d)

        # Loc: per-node proj
        x_loc = x_node.permute(0,1,3,2).reshape(B*N, T, F_in)
        x_loc = self.loc_proj(x_loc) + self.loc_pos    # (B*N, T, d)

        # Cross attention: Loc × Env (env as global context)
        global_ctx = env_seq[:,-1:,:].repeat(1, N, 1).reshape(B*N, 1, self.hidden)
        loc_ca, _ = self.cross_attn(x_loc, global_ctx, global_ctx)
        loc_ca = self.cross_ln(x_loc + loc_ca)         # residual

        # Conv1d temporal pooling
        h = loc_ca.permute(0,2,1)                       # (B*N, d, T)
        h = self.loc_temporal(h)
        h = h.permute(0,2,1)                           # (B*N, T, d)
        h = self.loc_temporal_ln(h)

        last = h[:, -1, :]                              # (B*N, d)
        return self.decoder(last).reshape(B, N, -1)    # (B, N, horizon)


class STFBaseTrainer(GCNBaseTrainer):
    name = "stf"

    def _build_model(self, in_dim, horizon, ei_dict):
        return SpacetimeformerLite(
            in_dim, cfg.cfg_train.hidden, horizon,
            n_heads=4, n_layers=2, dropout=cfg.cfg_train.dropout)

    def _make_run_batch(self, N, K_time, model, loss_fn):
        def run_batch(x, y):
            x_node = self._batch_to_node(x, N, K_time)
            out = model(x_node)
            pred_loss = out.transpose(1,2).reshape(-1, N)
            y_loss = y.transpose(1,2).reshape(-1,N) if y.dim()==3 else y
            return loss_fn(pred_loss, y_loss)
        return run_batch

    def _predict_batch(self, model, x, N, K_time):
        return model(self._batch_to_node(x, N, K_time))

    @staticmethod
    def _batch_to_node(x_flat, N, K_time):
        B, F_total, T = x_flat.shape
        x_flow = x_flat[:,:2*N,:].reshape(B, N, 2, T)
        x_tf   = x_flat[:,2*N:,:].unsqueeze(1).expand(B, N, K_time, T)
        return torch.cat([x_flow, x_tf], dim=2)


class STFLocOnlyTrainer(STFBaseTrainer):
    """Ablation: remove Env token and cross attention (local-only baseline)."""
    name = "stf_loc_only"
