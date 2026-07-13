"""Week4 数据加载器 — 与 Week3 完全一致"""
from __future__ import annotations
import os, json, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
import config as cfg

def load_raw_flow() -> np.ndarray:
    p = os.path.join(cfg.CLEAN_DIR, "taxi_p4_4d.npz")
    d = np.load(p); return d["flow"].astype(np.float32)

def load_timestamps() -> np.ndarray:
    d = np.load(os.path.join(cfg.CLEAN_DIR, "taxi_p4_4d.npz")); return d["timestamps"]

def load_graph():
    p = os.path.join(cfg.GRAPH_DIR, "bj_hetero_graph.pt")
    return torch.load(p, weights_only=False)

def load_feature_table() -> pd.DataFrame:
    p = os.path.join(cfg.FEAT_DIR, "bj_features.parquet"); return pd.read_parquet(p)

class SeqDataset(Dataset):
    def __init__(self, flow, seq_len, horizon, start, end,
                 time_features=None, target="taxi_flow_total"):
        self.seq_len = seq_len; self.horizon = horizon
        self.start = start; self.end = end
        if flow.ndim == 4:
            T, _, H, W = flow.shape
            self.flow = flow.transpose(0,1,2,3).reshape(T, -1).astype(np.float32)
        else:
            self.flow = flow.astype(np.float32); T = self.flow.shape[0]
        self.in_channels = self.flow.shape[1]
        self.time_features = time_features
        if target == "taxi_flow_total":
            self.flow_total = self.flow[:, :self.in_channels//2] + self.flow[:, self.in_channels//2:]
            self.target_arr = self.flow_total
        elif target == "taxi_inflow":
            self.target_arr = self.flow[:, :self.in_channels//2]
        elif target == "taxi_outflow":
            self.target_arr = self.flow[:, self.in_channels//2:]
        else:
            raise ValueError(target)
        self.target_dim = self.target_arr.shape[1]
        self.valid_ts = list(range(max(start, seq_len + horizon - 1), end))

    def __len__(self): return len(self.valid_ts)

    def __getitem__(self, i):
        t = self.valid_ts[i]
        t_end = t - self.horizon + 1
        x = self.flow[t_end - self.seq_len: t_end]
        if self.horizon == 1:
            y = self.target_arr[t]
        else:
            y = self.target_arr[t_end: t_end + self.horizon]
        x = torch.from_numpy(x.T).float()
        if self.time_features is not None:
            tf = self.time_features[t_end - self.seq_len: t_end]
            tf_t = torch.from_numpy(tf).float().T
            x = torch.cat([x, tf_t], dim=0)
        if self.horizon == 1:
            y = torch.from_numpy(y).float()
        else:
            y = torch.from_numpy(y).float()
        return x, y

class GraphSeqDataset(Dataset):
    def __init__(self, flow, node_features, seq_len, horizon, start, end,
                 time_features=None, target="taxi_flow_total"):
        self.seq_len = seq_len; self.horizon = horizon; self.end = end
        if flow.ndim == 4:
            T, _, H, W = flow.shape
            self.flow = flow.transpose(0,1,2,3).reshape(T, -1).astype(np.float32)
        else:
            self.flow = flow.astype(np.float32); T = self.flow.shape[0]
        self.N = self.flow.shape[1] // 2; self.half = self.flow.shape[1] // 2
        self.node_features = node_features
        self.time_features = time_features
        if target == "taxi_flow_total":
            self.target_arr = self.flow[:, :self.half] + self.flow[:, self.half:]
        elif target == "taxi_inflow":
            self.target_arr = self.flow[:, :self.half]
        elif target == "taxi_outflow":
            self.target_arr = self.flow[:, self.half:]
        else:
            raise ValueError(target)
        self.valid_ts = list(range(seq_len + horizon - 1, end))

    def __len__(self): return len(self.valid_ts)

    def __getitem__(self, i):
        t = self.valid_ts[i]; t_end = t - self.horizon + 1
        xs = []
        for s in range(t_end - self.seq_len, t_end):
            v = self.flow[s]
            v_in = v[:self.half]; v_out = v[self.half:]
            cell_vec = np.concatenate([v_in, v_out])
            cell_t = torch.from_numpy(cell_vec).float()
            tf = self.time_features[s] if self.time_features is not None else None
            if tf is not None:
                tf_t = torch.from_numpy(tf).float()
                full = torch.cat([cell_t, tf_t], dim=0)
            else:
                full = cell_t
            xs.append(full)
        x = torch.stack(xs, dim=0).permute(1, 0)
        F_in = x.shape[0] // self.N
        x = x.reshape(self.N, F_in, self.seq_len)
        y = torch.from_numpy(self.target_arr[t]).float()
        return x, y, t

def load_time_features() -> np.ndarray:
    p = os.path.join(cfg.FEAT_DIR, "bj_features.parquet")
    df = pd.read_parquet(p, columns=["timestamp","hour_sin","hour_cos",
                                      "is_weekend","is_holiday","weather_pressure_norm"])
    df = df.groupby("timestamp", as_index=False).first()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["hour_sin","hour_cos","is_weekend",
               "is_holiday","weather_pressure_norm"]].values.astype(np.float32)

def hourly_timestamp(start=0, end=None):
    p = os.path.join(cfg.FEAT_DIR, "bj_features.parquet")
    df = pd.read_parquet(p, columns=["timestamp"]).drop_duplicates().sort_values("timestamp")
    ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
    return ts.iloc[start:end].reset_index(drop=True) if end else ts.iloc[start:].reset_index(drop=True)
