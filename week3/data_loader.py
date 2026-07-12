"""Week3 数据加载器 — 统一的训练/验证/测试 Tensor

输入:
  /home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz →  shape (3888, 2, 32, 32), float
  /home/ubuntu/data/graph_bj/bj_hetero_graph.pt
  /home/ubuntu/data/features_bj/bj_features.parquet (用于评估 & 时间特征)
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


def load_raw_flow() -> np.ndarray:
    """load 4D flow (T, 2, H, W) — raw float"""
    p = os.path.join(cfg.CLEAN_DIR, "taxi_p4_4d.npz")
    d = np.load(p)
    return d["flow"].astype(np.float32)


def load_timestamps() -> np.ndarray:
    p = os.path.join(cfg.CLEAN_DIR, "taxi_p4_4d.npz")
    d = np.load(p)
    return d["timestamps"]


def load_graph() -> dict:
    """返回 HeteroData 对象"""
    p = os.path.join(cfg.GRAPH_DIR, "bj_hetero_graph.pt")
    return torch.load(p, weights_only=False)


def load_feature_table() -> pd.DataFrame:
    """返回长表数据 (N_grid * N_hour ≈ 4M 行)
    cols:
      grid_id, row, col, lon_center, lat_center, timestamp, hour, month,
      hour_sin, hour_cos, is_weekend, is_holiday,
      dist_to_tiananmen, dist_to_capital_airport, dist_to_beijing_station,
      dist_to_beijing_west, dist_to_beijing_north, dist_to_summer_palace,
      dist_to_city_center, dist_to_nearest_landmark,
      poi_total_count, poi_density_per_km2,
      taxi_inflow, taxi_outflow, taxi_flow_total,
      weather_pressure_norm,
      poi_count_food, poi_count_shopping, poi_count_transport, poi_count_work,
      poi_count_leisure, poi_count_residence, poi_count_education,
      poi_count_health, poi_count_tourism, poi_count_finance
    """
    p = os.path.join(cfg.FEAT_DIR, "bj_features.parquet")
    return pd.read_parquet(p)


# ============================================================
# 数据集
# ============================================================
class SeqDataset(Dataset):
    """PyTorch 序列回归数据集 — 通用，可喂 LSTM/GRU/GCN/GAT

    输入:
      flow: ndarray (T, 2, H, W) or (T, N)  —  *un-normalized*
      seq_len, horizon
      feature_table_cols: 额外附加的时间特征 → (T, K) — 例如 hour_sin, hour_cos, is_weekend, is_holiday
      start, end: 数据集有效的时间范围
    输出 (per item):
      x: (seq_len, F)   — 过去 seq_len 个时刻，每个 cell 的特征 + 时间特征
      y: (1, N)         — 下一个时刻的 N 个 cell 的值（flow_total 或 in/out 之一）

    注：为了支持 GCN/GAT 的批处理，下面默认每个 item 是"全 city 的一个时步预测"，
        即 y 是 (2, N=1024)
    """
    def __init__(self, flow: np.ndarray, seq_len: int, horizon: int,
                 start: int, end: int,
                 time_features: np.ndarray | None = None,
                 target: str = "taxi_flow_total"):
        """
        flow: (T, 2, H, W)            ←  即 npz 的 'flow'
        time_features: (T, K)         ←  每时刻的时间特征（已在配置里选定）
        target: 'taxi_inflow' / 'taxi_outflow' / 'taxi_flow_total'
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.start = start
        self.end   = end

        # 把 4D 转成 (T, 2, H, W) 然后转 (T, N) for tabular baseline
        # (T, 2, H, W) → (T, 2*N)
        if flow.ndim == 4:
            T, _, H, W = flow.shape
            self.flow = flow.transpose(0, 1, 2, 3).reshape(T, -1).astype(np.float32)
        else:
            self.flow = flow.astype(np.float32)
            T = self.flow.shape[0]
        self.in_channels = self.flow.shape[1]

        # 时间特征
        self.time_features = time_features  # may be None

        # 决定 target 索引
        if target == "taxi_flow_total":
            # 在 flow_total 上训练 = in + out
            self.flow_total = self.flow[:, :self.in_channels // 2] + self.flow[:, self.in_channels // 2:]
            self.target_arr = self.flow_total
        elif target == "taxi_inflow":
            self.target_arr = self.flow[:, :self.in_channels // 2]
        elif target == "taxi_outflow":
            self.target_arr = self.flow[:, self.in_channels // 2:]
        else:
            raise ValueError(target)

        self.target_dim = self.target_arr.shape[1]  # N=1024

        # 把有效 time indices 收集 — 注意 start: 输入序列开头不能早于 start
        self.valid_ts = list(range(max(start, seq_len + horizon - 1), end))

    def __len__(self):
        return len(self.valid_ts)

    def __getitem__(self, i):
        t = self.valid_ts[i]
        # V2: 多步预测 — y 是 [t - horizon + 1, t] 的连续 horizon 个时刻
        t_end = t - self.horizon + 1      # 输入序列结束 = y 第一步的前一时刻
        x = self.flow[t_end - self.seq_len: t_end]            # (T_seq, 2*N)
        if self.horizon == 1:
            y = self.target_arr[t]                           # (N,)
        else:
            y = self.target_arr[t_end: t_end + self.horizon]  # (H, N)
        # 转成 torch
        x = torch.from_numpy(x.T).float()       # (F=2*N, T_seq) — 适合 LSTM
        # V2: 把时间特征拼到 x 的特征维上 → (F + K, T_seq)
        if self.time_features is not None:
            tf = self.time_features[t_end - self.seq_len: t_end]  # (T_seq, K)
            tf_t = torch.from_numpy(tf).float().T               # (K, T_seq)
            x = torch.cat([x, tf_t], dim=0)
            # 时间特征也要拼到 y 的每一步 (在每个时刻取该时刻的时间特征)
            if self.horizon > 1:
                tf_y = self.time_features[t_end: t_end + self.horizon]  # (H, K)
                tf_y_t = torch.from_numpy(tf_y).float()                # (H, K)
                # y: (H, N) 与 tf_y_t: (H, K) 沿特征维拼 → (H, N+K)
                # 但下游预测只用 N 维前段; tf_y 仅作为未来时间信息,
                # 这里我们改返回方式: 仅返回 (x, y); y 仍是 (H, N)
        if self.horizon == 1:
            y = torch.from_numpy(y).float()         # (N,)
        else:
            y = torch.from_numpy(y).float()         # (H, N)
        return x, y


class GraphSeqDataset(Dataset):
    """GCN/GAT 输入：使用 spatial 图结构 + 全城级别的时序窗口
    x: (T_seq, N, F_node)   — 每个 node 的 feature_dim 维时序
    y: (N,)                — 下一时刻的目标
    extra_features (可选): (T, K) 时间特征
    """
    def __init__(self, flow: np.ndarray, node_features: torch.Tensor,
                 seq_len: int, horizon: int, start: int, end: int,
                 time_features: np.ndarray | None = None,
                 target: str = "taxi_flow_total"):
        self.seq_len = seq_len
        self.horizon = horizon
        self.end = end

        if flow.ndim == 4:
            T, _, H, W = flow.shape
            self.flow = flow.transpose(0, 1, 2, 3).reshape(T, -1).astype(np.float32)
        else:
            self.flow = flow.astype(np.float32)
            T = self.flow.shape[0]

        self.N = self.flow.shape[1] // 2
        self.half = self.flow.shape[1] // 2

        # 节点特征: (N, F_node)
        self.node_features = node_features  # torch tensor

        # 拼接时间特征到节点特征上
        if time_features is not None:
            self.time_features = time_features
        else:
            self.time_features = None

        if target == "taxi_flow_total":
            self.target_arr = self.flow[:, :self.half] + self.flow[:, self.half:]
        elif target == "taxi_inflow":
            self.target_arr = self.flow[:, :self.half]
        elif target == "taxi_outflow":
            self.target_arr = self.flow[:, self.half:]
        else:
            raise ValueError(target)

        # 预 compute node feature 拼接时间特征的 per-step feature
        self.valid_ts = list(range(seq_len + horizon - 1, end))

    def __len__(self):
        return len(self.valid_ts)

    def __getitem__(self, i):
        t = self.valid_ts[i]
        t_end = t - self.horizon + 1
        # 收集 [t_end - seq_len, t_end) 区间的节点动态特征
        xs = []
        for s in range(t_end - self.seq_len, t_end):
            v = self.flow[s]                                # (2*N,)
            # 取 inflow / outflow 或者拼接
            v_in  = v[:self.half]
            v_out = v[self.half:]
            cell_vec = np.concatenate([v_in, v_out])        # (2N,)
            cell_t  = torch.from_numpy(cell_vec).float()    # (2N,)
            tf = self.time_features[s] if self.time_features is not None else None
            if tf is not None:
                tf_t = torch.from_numpy(tf).float()
                # 拼接： (2N + K, )
                full = torch.cat([cell_t, tf_t], dim=0)
            else:
                full = cell_t
            xs.append(full)
        x = torch.stack(xs, dim=0)      # (seq_len, 2N+K)
        x = x.permute(1, 0)             # (2N+K, seq_len)  (NN 每一行是一个节点的 1D 信号)
        # GNN 输入： (N, F_in) per timestep；这里我们把 (2N+K) 拆成 (N, 2+K_per_node? )
        # 简化方案：每节点使用 its_inflow + its_outflow + K + K_global
        # 重塑：(2N + K) → (N, 2 + K/N近似)
        F_in = x.shape[0] // self.N
        # (N, F_in, seq_len)
        x = x.reshape(self.N, F_in, self.seq_len)

        y = torch.from_numpy(self.target_arr[t]).float()  # (N,)
        return x, y, t


# ============================================================
# 时间特征 (从 bj_features 表中提取)
# ============================================================
def load_time_features() -> np.ndarray:
    """从 bj_features.parquet 中提取全局时间特征
    返回 (T, K)
    """
    p = os.path.join(cfg.FEAT_DIR, "bj_features.parquet")
    df = pd.read_parquet(p, columns=["timestamp", "hour_sin", "hour_cos",
                                      "is_weekend", "is_holiday",
                                      "weather_pressure_norm"])
    df = df.groupby("timestamp", as_index=False).first()
    df = df.sort_values("timestamp").reset_index(drop=True)
    feat = df[["hour_sin", "hour_cos", "is_weekend",
               "is_holiday", "weather_pressure_norm"]].values.astype(np.float32)
    return feat


# ============================================================
# 时间戳 helper
# ============================================================
def hourly_timestamp(start=0, end=None):
    """生成从 bj_features 提取的时间戳 list"""
    p = os.path.join(cfg.FEAT_DIR, "bj_features.parquet")
    df = pd.read_parquet(p, columns=["timestamp"]).drop_duplicates().sort_values("timestamp")
    ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
    if end is not None:
        ts = ts.iloc[start:end].reset_index(drop=True)
    else:
        ts = ts.iloc[start:].reset_index(drop=True)
    return ts
