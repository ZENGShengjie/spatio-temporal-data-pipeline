"""P2 方案 A: GRU 主导 + ST-GCN 空间残差混合模型

训练流程
  1. 独立训练 GRU (同 baseline, 含 early stop)
  2. 在 train/val/test 上做前向传播 → 得到 gru_pred + y (均在归一化空间)
  3. 计算残差: residual = y - gru_pred  (split 级别对齐)
  4. 用 ST-GCN V2 架构以 residual 为新目标重新训练
  5. 最终预测 = GRU_pred_norm + ST-GCN_residual_pred_norm → 一次性反归一化
  6. 返回 (final_pred, gt) 与 run_week3.py 接口一致

注意
  - GRU 模型不保存权重, 因此每跑一次都完整重训 GRU (与 baseline 等价耗时)
  - 残差在归一化空间计算, 合成后再一次性反归一化, 保证与 baseline 同口径
  - ST-GCN 架构不改 (gcn_model.STHeteroGCN), 只替换训练目标
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import config as cfg
from data_loader import SeqDataset, load_graph
from base_trainer import set_seed, make_device
from metrics import BaseTrainer
from models.gru_model import GRUForecaster
from models.gcn_model import STHeteroGCN  # 直接复用 V2 架构, 不改


# ============================================================
# 数据包装：用预计算残差替换 SeqDataset 的 y
# ============================================================
class ResidualDataset(Dataset):
    """Wrap a SeqDataset but replace y with pre-computed residual array."""
    def __init__(self, base_ds: SeqDataset, residuals: np.ndarray):
        assert len(residuals) == len(base_ds), (
            f"ResidualDataset size mismatch: residuals={len(residuals)} "
            f"base_ds={len(base_ds)}"
        )
        self.base_ds = base_ds
        self.res = torch.from_numpy(residuals.astype(np.float32))

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, _ = self.base_ds[idx]
        return x, self.res[idx]


# ============================================================
# 通用 inference helper
# ============================================================
# ============================================================
# Generic inference helper — returns per-sample (num_samples, H, N)
# ============================================================
@torch.no_grad()
def _run_gru_inference(model, loader, device):
    """Run GRU in eval mode, return (num_samples, H, N) arrays."""
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).cpu().numpy()           # (B, H, N)
        if y.dim() == 3:
            y_np = y.numpy()                  # (B, H, N)
        else:
            y_np = y.numpy()
        preds.append(p); gts.append(y_np)
    return np.concatenate(preds, axis=0), np.concatenate(gts, axis=0)


@torch.no_grad()
def _run_stgcn_inference(model, loader, device, N, ei_dict):
    """Run ST-GCN in eval mode, return (num_samples, H, N) arrays."""
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(device)
        # ST-GCN expects node-format x; reshape from (B, F_total, T_seq)
        B, F_total, T = x.shape
        K_time = F_total - 2 * N
        x_flow = x[:, : 2 * N, :].reshape(B, N, 2, T)
        x_tf   = x[:, 2 * N :, :].unsqueeze(1).expand(B, N, K_time, T)
        x_node = torch.cat([x_flow, x_tf], dim=2)
        p = model(x_node, ei_dict).cpu().numpy()           # (B, N, H)
        p = p.transpose(0, 2, 1)                            # (B, H, N)
        preds.append(p)
    return np.concatenate(preds, axis=0)


# ============================================================
# P2 混合模型
# ============================================================
class GRU_STGCN_ResidualTrainer(BaseTrainer):
    name = "p2_res"

    def __init__(self):
        self.device = make_device()
        set_seed(cfg.cfg_train.seed)

    def fit_predict(self, flow_4d, time_features=None, target="taxi_flow_total", **kwargs):
        # ========== shared prep ==========
        cell_max = flow_4d[:cfg.SPLIT.train_end].max(axis=0)
        cell_max = np.maximum(cell_max, 1.0)
        normed = (flow_4d / cell_max).clip(min=0).astype(np.float32)

        common = dict(
            seq_len=cfg.cfg_train.seq_len,
            horizon=cfg.cfg_train.horizon,
            time_features=time_features,
            target=target,
        )
        train_ds = SeqDataset(normed, start=cfg.SPLIT.train_start, end=cfg.SPLIT.train_end, **common)
        val_ds   = SeqDataset(normed, start=cfg.SPLIT.train_end,   end=cfg.SPLIT.val_end,   **common)
        test_ds  = SeqDataset(normed, start=cfg.SPLIT.val_end,     end=cfg.SPLIT.test_end,  **common)
        print(f"[P2] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

        F_in = normed.shape[1] * normed.shape[2] * normed.shape[3] + (
            time_features.shape[1] if time_features is not None else 0
        )
        n_cells = train_ds.target_dim
        N = 1024
        K_time = time_features.shape[1] if time_features is not None else 0
        F_in_node = 2 + K_time

        train_loader = DataLoader(train_ds, batch_size=cfg.cfg_train.batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.cfg_train.batch)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.cfg_train.batch)

        loss_fn = nn.SmoothL1Loss()

        def batch_to_node(x_flat, N, K_time):
            B, F_total, T = x_flat.shape
            x_flow = x_flat[:, : 2 * N, :]
            x_tf   = x_flat[:, 2 * N :, :]
            x_flow = x_flow.reshape(B, N, 2, T)
            x_tf   = x_tf.unsqueeze(1).expand(B, N, K_time, T)
            return torch.cat([x_flow, x_tf], dim=2)

        # ==========================================================
        # Phase 1: Train GRU base
        # ==========================================================
        print("\n[Phase 1] GRU base ...")
        gru = GRUForecaster(
            F_in, cfg.cfg_train.hidden, cfg.cfg_train.layers,
            cfg.cfg_train.dropout, cfg.cfg_train.horizon, n_cells,
        ).to(self.device)
        opt_gru = optim.Adam(gru.parameters(), lr=cfg.cfg_train.lr,
                             weight_decay=cfg.cfg_train.weight_decay)

        best_val = float("inf")
        best_state = None
        no_improve = 0
        t0 = time.time()
        for epoch in range(cfg.cfg_train.epochs):
            gru.train()
            losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                p = gru(x)
                p_flat = p.reshape(p.size(0), -1)
                y_flat = y.reshape(y.size(0), -1) if y.dim() == 3 else y
                loss = loss_fn(p_flat, y_flat)
                opt_gru.zero_grad(); loss.backward(); opt_gru.step()
                losses.append(loss.item())
            tr = float(np.mean(losses))

            gru.eval()
            with torch.no_grad():
                vs = []
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    p = gru(x)
                    p_flat = p.reshape(p.size(0), -1)
                    y_flat = y.reshape(y.size(0), -1) if y.dim() == 3 else y
                    vs.append(loss_fn(p_flat, y_flat).item())
            val = float(np.mean(vs))
            if epoch % 2 == 0 or no_improve == 0:
                print(f"  [GRU {epoch:2d}] tr={tr:.5f}  val={val:.5f}")
            if val < best_val - 1e-5:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in gru.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.cfg_train.patience:
                    print(f"  [GRU early stop @ {epoch}]")
                    break
        print(f"  [GRU] trained {time.time() - t0:.1f}s  best_val={best_val:.5f}")
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in gru.state_dict().items()}
        gru.load_state_dict(best_state)

        # Inference on all splits — keep (num_samples, H, N) per-sample format
        print("[Phase 1] GRU inference on train/val/test ...")
        gru_pred_train, y_train = _run_gru_inference(gru, train_loader, self.device)
        gru_pred_val,   y_val   = _run_gru_inference(gru, val_loader,   self.device)
        gru_pred_test,  y_test  = _run_gru_inference(gru, test_loader,  self.device)
        print(f"  [GRU] train {gru_pred_train.shape}  val {gru_pred_val.shape}  test {gru_pred_test.shape}")

        # Compute residuals (still in normalized space)
        res_train = y_train - gru_pred_train
        res_val   = y_val   - gru_pred_val
        res_test  = y_test  - gru_pred_test

        # ==========================================================
        # Phase 2: Train ST-GCN on residuals
        # ==========================================================
        print("\n[Phase 2] ST-GCN residual learner ...")
        graph = load_graph()
        spatial_ei = graph["spatial", "adjacent", "spatial"].edge_index.to(self.device)
        similar_ei = graph["spatial", "similar", "spatial"].edge_index.to(self.device)
        ei_dict = {"spatial": spatial_ei, "similar": similar_ei}
        print(f"  [ST-Res] spatial={spatial_ei.shape[1]} similar={similar_ei.shape[1]}")

        res_train_ds = ResidualDataset(train_ds, res_train)
        res_val_ds   = ResidualDataset(val_ds,   res_val)
        res_test_ds  = ResidualDataset(test_ds,  res_test)

        res_train_loader = DataLoader(res_train_ds, batch_size=cfg.cfg_train.batch, shuffle=True)
        res_val_loader   = DataLoader(res_val_ds,   batch_size=cfg.cfg_train.batch)
        res_test_loader  = DataLoader(res_test_ds,  batch_size=cfg.cfg_train.batch)

        stgcn = STHeteroGCN(
            F_in_node, cfg.cfg_train.hidden,
            horizon=cfg.cfg_train.horizon,
            n_layers=cfg.cfg_train.layers,
            dropout=cfg.cfg_train.dropout,
        ).to(self.device)
        n_params = sum(p.numel() for p in stgcn.parameters())
        print(f"  [ST-Res] params={n_params:,}")

        opt_st = optim.Adam(stgcn.parameters(), lr=cfg.cfg_train.lr,
                            weight_decay=cfg.cfg_train.weight_decay)

        best_vres = float("inf")
        best_sres = None
        no_imp = 0
        t1 = time.time()
        for epoch in range(cfg.cfg_train.epochs):
            stgcn.train()
            losses = []
            for x, yr in res_train_loader:
                x, yr = x.to(self.device), yr.to(self.device)
                x_node = batch_to_node(x, N, K_time)
                out = stgcn(x_node, ei_dict)
                loss = loss_fn(
                    out.transpose(1, 2).reshape(-1, N),
                    yr.transpose(1, 2).reshape(-1, N),
                )
                if not torch.isfinite(loss):
                    opt_st.zero_grad()
                    continue
                opt_st.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(stgcn.parameters(), 5.0)
                opt_st.step()
                losses.append(loss.item())
            torch.cuda.empty_cache()
            tr = float(np.mean(losses)) if losses else float("nan")

            stgcn.eval()
            with torch.no_grad():
                vs = []
                for x, yr in res_val_loader:
                    x, yr = x.to(self.device), yr.to(self.device)
                    x_node = batch_to_node(x, N, K_time)
                    out = stgcn(x_node, ei_dict)
                    v = loss_fn(
                        out.transpose(1, 2).reshape(-1, N),
                        yr.transpose(1, 2).reshape(-1, N),
                    ).item()
                    if np.isfinite(v):
                        vs.append(v)
            val = float(np.mean(vs)) if vs else float("nan")
            if epoch % 2 == 0 or no_imp == 0:
                print(f"  [ST-Res {epoch:2d}] tr={tr:.5f}  val={val:.5f}")
            if np.isfinite(val) and val < best_vres - 1e-5:
                best_vres = val
                best_sres = {k: v.detach().cpu().clone() for k, v in stgcn.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= cfg.cfg_train.patience:
                    print(f"  [ST-Res early stop @ {epoch}]")
                    break
        print(f"  [ST-Res] trained {time.time()-t1:.1f}s  best_val={best_vres:.5f}")
        if best_sres is None:
            best_sres = {k: v.detach().cpu().clone() for k, v in stgcn.state_dict().items()}
        stgcn.load_state_dict(best_sres)

        # ==========================================================
        # Phase 3: Compose final pred
        # ==========================================================
        print("\n[Phase 3] Composing final prediction ...")
        stgcn_res_test = _run_stgcn_inference(stgcn, res_test_loader, self.device, N, ei_dict)  # (num_samples, H, N)

        # 在归一化空间合成, 最后一次性反归一化
        if target == "taxi_inflow":
            scale = cell_max[0].flatten()
        elif target == "taxi_outflow":
            scale = cell_max[1].flatten()
        else:
            scale = (cell_max[0] + cell_max[1]).flatten()
        # Match run_week3.py output convention: flatten to (num_samples*H, N)
        B_H = gru_pred_test.shape[0] * gru_pred_test.shape[1]
        final_pred = (gru_pred_test + stgcn_res_test).reshape(B_H, N) * scale[None, :]
        gt = y_test.reshape(B_H, N) * scale[None, :]
        print(f"[P2] final_pred={final_pred.shape}  gt={gt.shape}")
        return final_pred, gt
