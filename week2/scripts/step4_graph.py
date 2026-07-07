"""
Week 2 — 步骤 4: 多模态异构图构建 (低内存版)

节点: 每个 500m 网格
边类型:
  - spatial:     地理 KNN 邻居 (haversine KNN=8)
  - semantic:    POI/距离特征 KNN (euclidean KNN=6)
  - correlated:  taxi_pickup 时序相关 >0.7

输入:  grid_nyc/*, features_nyc/*
输出:  /home/ubuntu/amazon/graph_nyc/nyc_hetero_graph.pt
"""
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch_geometric.data import HeteroData
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

FEATURES_DIR = os.getenv("FEATURES_DIR", "/home/ubuntu/amazon/features_nyc")
GRID_DIR     = os.getenv("GRID_DIR",     "/home/ubuntu/amazon/grid_nyc")
OUT_DIR      = os.getenv("GRAPH_DIR",    "/home/ubuntu/amazon/graph_nyc")
os.makedirs(OUT_DIR, exist_ok=True)

KNN_SPATIAL  = 8
KNN_SEMANTIC = 6
CORR_THRESH  = 0.7


def load_data():
    print("[加载数据] ...")
    # grid parquet 无几何元数据 → 改用 geojson
    grid_df = gpd.read_file(os.path.join(GRID_DIR, "nyc_grid_500m.geojson"))
    if "grid_id" not in grid_df.columns:
        grid_df = grid_df.reset_index().rename(columns={"index": "grid_id"})
    grid_df["lon_center"] = grid_df.geometry.centroid.x
    grid_df["lat_center"] = grid_df.geometry.centroid.y

    feat_df = pd.read_parquet(os.path.join(FEATURES_DIR, "nyc_features.parquet"))
    print(f"  网格: {len(grid_df)}")
    print(f"  特征: {len(feat_df)}")
    return grid_df, feat_df


def build_node_features(grid_df, feat_df):
    print("[构建节点特征] ...")

    # 按 grid_id 聚合时空特征 (mean)
    agg_cols = [
        "dist_to_times_square", "dist_to_central_park", "dist_to_wall_street",
        "dist_to_grand_central", "dist_to_jfk_airport", "dist_to_nearest_landmark",
        "poi_total_count", "poi_density_per_km2",
        "taxi_pickup_count", "taxi_dropoff_count",
        "ndvi_mean", "ndvi_max", "ndvi_min", "is_water",
    ]
    available = [c for c in agg_cols if c in feat_df.columns]
    node_feat = feat_df.groupby("grid_id")[available].mean()  # grid_id 已是一级索引，直接是列
    grid_feat = grid_df.merge(node_feat, on="grid_id", how="left")
    grid_feat[available] = grid_feat[available].fillna(0)

    # Min-Max 归一化
    for col in available:
        mn, mx = grid_feat[col].min(), grid_feat[col].max()
        if mx > mn:
            grid_feat[col + "_norm"] = ((grid_feat[col] - mn) / (mx - mn)).astype(np.float32)
        else:
            grid_feat[col + "_norm"] = 0.0

    norm_cols = [c + "_norm" for c in available]
    node_features = torch.from_numpy(grid_feat[norm_cols].values.astype(np.float32))
    print(f"  节点特征维度: {node_features.shape}")
    return node_features, grid_feat


def build_spatial_edges(grid_feat):
    print("[构建空间邻接边] ...")
    coords = grid_feat[["lon_center", "lat_center"]].values
    nbrs = NearestNeighbors(n_neighbors=KNN_SPATIAL + 1, metric="haversine").fit(np.radians(coords))
    _, indices = nbrs.kneighbors(np.radians(coords))
    edge_src, edge_dst = [], []
    for i in range(len(coords)):
        for j in indices[i][1:]:
            edge_src.append(i); edge_dst.append(j)
    spatial_edge_index = torch.LongTensor([edge_src, edge_dst])
    print(f"  空间边: {spatial_edge_index.size(1)} 条")
    return spatial_edge_index


def build_semantic_edges(grid_feat):
    print("[构建语义相似性边] ...")
    spatial_cols = [c for c in grid_feat.columns if c.endswith("_norm")]
    if not spatial_cols:
        return torch.empty(2, 0, dtype=torch.long)
    X = grid_feat[spatial_cols].fillna(0).values
    nbrs = NearestNeighbors(n_neighbors=KNN_SEMANTIC + 1, metric="euclidean").fit(X)
    _, indices = nbrs.kneighbors(X)
    edge_src, edge_dst = [], []
    for i in range(len(X)):
        for j in indices[i][1:]:
            edge_src.append(i); edge_dst.append(j)
    semantic_edge_index = torch.LongTensor([edge_src, edge_dst])
    print(f"  语义边: {semantic_edge_index.size(1)} 条")
    return semantic_edge_index


def build_flow_edges(feat_df):
    print("[构建交通流量相关边] ...")
    has_pickup  = "taxi_pickup_count"   in feat_df.columns
    has_dropoff = "taxi_dropoff_count" in feat_df.columns
    if not has_pickup:
        return torch.empty(2, 0, dtype=torch.long)

    try:
        # Build a combined flow signal = pickup + dropoff (captures both directions)
        if has_dropoff:
            feat_df = feat_df.copy()
            feat_df["taxi_flow"] = feat_df["taxi_pickup_count"] + feat_df["taxi_dropoff_count"]
            flow_col = "taxi_flow"
            print("  流量信号: taxi_pickup + taxi_dropoff (双向流量)")
        else:
            flow_col = "taxi_pickup_count"
            print("  流量信号: taxi_pickup_count")

        pivot = (
            feat_df.groupby(["grid_id", "weekday", "hour"])[flow_col]
            .mean().reset_index()
            .pivot_table(index=["weekday", "hour"], columns="grid_id",
                         values=flow_col, fill_value=0)
        )
        grid_ids = pivot.columns.tolist()
        n = len(grid_ids)
        if n < 2:
            return torch.empty(2, 0, dtype=torch.long)

        M = pivot.values.astype(np.float32)
        M = (M - M.mean(0)) / (M.std(0) + 1e-8)

        corr = np.corrcoef(M.T)
        np.fill_diagonal(corr, 0)
        upper = np.triu(corr, k=1)
        src, dst = np.where(upper > CORR_THRESH)

        edge_src = np.concatenate([src, dst])
        edge_dst = np.concatenate([dst, src])
        flow_edge_index = torch.LongTensor(np.stack([edge_src, edge_dst]))
        print(f"  流量边: {flow_edge_index.size(1)} 条 (corr > {CORR_THRESH}, n_grids={n})")
        return flow_edge_index
    except Exception as e:
        print(f"  流量边构建失败: {e}")
        import traceback
        traceback.print_exc()
        return torch.empty(2, 0, dtype=torch.long)


def save_graph(data, grid_feat):
    print("[保存图结构] ...")
    out_graph = os.path.join(OUT_DIR, "nyc_hetero_graph.pt")
    out_meta  = os.path.join(OUT_DIR, "graph_metadata.json")

    torch.save(data, out_graph)
    metadata = {
        "num_nodes": data["spatial"].x.size(0),
        "num_spatial_edges":  data["spatial", "adjacent",   "spatial"].edge_index.size(1),
        "num_semantic_edges": data["spatial", "similar",    "spatial"].edge_index.size(1),
        "num_flow_edges":     data["spatial", "correlated", "spatial"].edge_index.size(1),
        "node_feature_dim":   data["spatial"].x.size(1),
        "grid_ids": grid_feat["grid_id"].tolist(),
        "saved_at": datetime.now().isoformat(),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 图: {out_graph}")
    print(f"  ✅ 元数据: {out_meta}")
    print(f"    节点: {metadata['num_nodes']}")
    print(f"    空间边: {metadata['num_spatial_edges']}")
    print(f"    语义边: {metadata['num_semantic_edges']}")
    print(f"    流量边: {metadata['num_flow_edges']}")
    print(f"    节点特征维度: {metadata['node_feature_dim']}")
    return metadata


def main():
    print("=" * 60)
    print(f"Week 2 步骤 4 (低内存版) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  KNN_SPATIAL={KNN_SPATIAL}, KNN_SEMANTIC={KNN_SEMANTIC}, CORR_THRESH={CORR_THRESH}")
    print("=" * 60)

    grid_df, feat_df = load_data()
    node_features, grid_feat = build_node_features(grid_df, feat_df)
    spatial_edges  = build_spatial_edges(grid_feat)
    semantic_edges = build_semantic_edges(grid_feat)
    flow_edges     = build_flow_edges(feat_df)

    data = HeteroData()
    data["spatial"].x = node_features
    data["spatial", "adjacent",   "spatial"].edge_index = spatial_edges
    data["spatial", "similar",    "spatial"].edge_index = semantic_edges
    data["spatial", "correlated", "spatial"].edge_index = flow_edges

    save_graph(data, grid_feat)
    print("\n" + "=" * 60)
    print("异构图构建完成!")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()