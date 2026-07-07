"""
Week 2 — 步骤 3: 时空特征工程 (低内存版)
对 NYC 每个 500m 网格，生成代表性 7×24 时空特征

输入文件 (cleaned_nyc/):
  - taxi_nyc_clean.parquet     PULocationID (zone_id), pickup_datetime, ...
  - weather_clean.parquet      datetime (hourly), temp, humidity, ...
  - road_density_clean.parquet grid_id, road_length_m, road_density_km_per_km2
  - poi_clean.parquet          lon, lat, category, amenity, shop, office, building

输出:
  features_nyc/nyc_features.parquet
"""
import os
import json
import warnings
from datetime import datetime
import gc

import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import box, Point

warnings.filterwarnings("ignore")

CLEANED_DIR = os.getenv("CLEANED_DIR",   "/home/ubuntu/amazon/cleaned_nyc")
GRID_DIR    = os.getenv("GRID_DIR",      "/home/ubuntu/amazon/grid_nyc")
OUT_DIR     = os.getenv("FEATURES_DIR",  "/home/ubuntu/amazon/features_nyc")
RAW_TAXI_ZONES = "/home/ubuntu/amazon/raw_nyc/taxi_zones"
os.makedirs(OUT_DIR, exist_ok=True)

# NYC 地标 (用于空间特征)
NYC_LANDMARKS = {
    "times_square":  (-73.9855, 40.7580),
    "central_park":  (-73.9654, 40.7829),
    "wall_street":   (-74.0089, 40.7074),
    "grand_central": (-73.9772, 40.7527),
    "jfk_airport":   (-73.7781, 40.6413),
}
# 城市中心: Midtown Manhattan
NYC_CENTER = (-73.9857, 40.7484)

HOLIDAYS_2024 = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14",
    "2024-11-11", "2024-11-28", "2024-12-25",
]

# weather 归一化区间
WEATHER_NORM_STATS = {
    "temp":       {"min": -20.0, "max":  45.0},   # °C
    "humidity":   {"min":   0.0, "max": 100.0},   # %
    "pressure":   {"min": 980.0, "max": 1050.0},  # hPa
    "wind_speed": {"min":   0.0, "max":  30.0},   # m/s
    "clouds":     {"min":   0.0, "max": 100.0},   # %
}


# ============== 工具 ==============
def _minmax(v, lo, hi):
    if pd.isna(v):
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _haversine_km(lon1, lat1, lon2, lat2):
    """公里数"""
    R = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ============== 网格加载 ==============
def load_grid():
    print("[加载网格] ...")
    gdf = gpd.read_file(os.path.join(GRID_DIR, "nyc_grid_500m.geojson"))
    if "grid_id" not in gdf.columns:
        gdf = gdf.reset_index().rename(columns={"index": "grid_id"})
    gdf["grid_id"] = gdf["grid_id"].astype(str)  # road_density 也是 str, 保持一致
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    print(f"  网格数: {len(gdf):,} (grid_id 类型: {gdf['grid_id'].dtype})")
    return gdf


# ============== 时空索引 ==============
def build_space_time_index(grid_gdf):
    print("\n[生成时空序列] ...")
    rep_dates = pd.date_range("2024-06-02", periods=7, freq="D")
    hours = np.arange(24)
    records = []
    for d in rep_dates:
        for h in hours:
            records.append((d + pd.Timedelta(hours=h), d.weekday(), d.month, h))
    ts_df = pd.DataFrame(records, columns=["datetime", "weekday", "month", "hour"])

    grid_df = pd.DataFrame({
        "grid_id": grid_gdf["grid_id"].values,
        "centroid_lon": grid_gdf["centroid_lon"].values,
        "centroid_lat": gdf_to_lat_safe(grid_gdf),
    })
    grid_df["__k"] = 1
    ts_df["__k"] = 1
    df = grid_df.merge(ts_df, on="__k").drop(columns="__k")
    print(f"  序列形状: {df.shape}")
    return df


def gdf_to_lat_safe(grid_gdf):
    return grid_gdf["centroid_lat"].values


# ============== 时间特征 ==============
def build_time_features(df):
    print("[时间特征] ...")
    df["is_weekend"] = (df["weekday"] >= 5).astype(np.int8)
    df["is_holiday"] = df["datetime"].dt.strftime("%Y-%m-%d").isin(HOLIDAYS_2024).astype(np.int8)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]   / 24).astype(np.float32)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]   / 24).astype(np.float32)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] /  7).astype(np.float32)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] /  7).astype(np.float32)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"]   / 12).astype(np.float32)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"]   / 12).astype(np.float32)
    return df


# ============== 空间特征 ==============
def build_spatial_features(df):
    print("[空间特征] ...")
    for name, (lon, lat) in NYC_LANDMARKS.items():
        dist = _haversine_km(df["centroid_lon"], df["centroid_lat"], lon, lat)
        df[f"dist_to_{name}"] = dist.astype(np.float32)
    # 到城市中心 (Midtown) 距离
    df["dist_to_city_center"] = _haversine_km(
        df["centroid_lon"], df["centroid_lat"], NYC_CENTER[0], NYC_CENTER[1]
    ).astype(np.float32)
    dist_cols = [f"dist_to_{n}" for n in NYC_LANDMARKS] + ["dist_to_city_center"]
    df["dist_to_nearest_landmark"] = df[dist_cols].min(axis=1).astype(np.float32)
    return df


# ============== 气象特征 (weather_clean.parquet) ==============
def _load_weather_hourly():
    wx_path = os.path.join(CLEANED_DIR, "weather_clean.parquet")
    if not os.path.exists(wx_path):
        return None
    cols = ["datetime", "temp", "humidity", "pressure", "wind_speed", "clouds"]
    try:
        wx = pd.read_parquet(wx_path, columns=cols)
    except Exception as e:
        print(f"  ⚠️ 读气象文件失败: {e}")
        return None
    if wx.empty:
        return None
    ts = pd.to_datetime(wx["datetime"], utc=True, errors="coerce")
    wx["hour"]    = ts.dt.hour.astype(np.int8)
    wx["weekday"] = ts.dt.weekday.astype(np.int8)
    val_cols = ["temp", "humidity", "pressure", "wind_speed", "clouds"]
    wx = wx.groupby(["weekday", "hour"], as_index=False)[val_cols].mean()
    print(f"  气象表: {len(wx):,} 行 (weekday,hour) keys")
    return wx


def build_weather_features(df):
    print("[气象特征 — cleaned_nyc/weather_clean.parquet] ...")
    wx = _load_weather_hourly()
    if wx is None:
        print("  ⚠️ 未找到 weather_clean.parquet, 用 0 占位")
        for col in ["temp", "humidity", "pressure", "wind_speed", "clouds"]:
            df[f"weather_{col}_norm"] = 0.0
        return df
    df = df.merge(wx, on=["weekday", "hour"], how="left")
    for col, s in WEATHER_NORM_STATS.items():
        lo, hi = s["min"], s["max"]
        df[f"weather_{col}_norm"] = df[col].apply(
            lambda v: _minmax(v, lo, hi)
        ).astype(np.float32)
    df = df.drop(columns=list(WEATHER_NORM_STATS.keys()))
    return df


# ============== 路网密度特征 (road_density_clean.parquet) ==============
def _load_road_density():
    rd_path = os.path.join(CLEANED_DIR, "road_density_clean.parquet")
    if not os.path.exists(rd_path):
        return None
    rd = pd.read_parquet(rd_path)
    rd["grid_id"] = rd["grid_id"].astype(str)
    base_keep = ["grid_id", "road_length_m", "road_segment_count",
                 "road_density_km_per_km2", "area_m2"]
    tier_keep = [c for c in rd.columns if c.startswith("road_len_")]
    rd = rd[[c for c in base_keep + tier_keep if c in rd.columns]]
    print(f"  路网表: {len(rd):,} 行, tier列: {[c for c in rd.columns if c.startswith('road_len_')]}")
    return rd


def build_road_features(df):
    print("[路网密度特征 — cleaned_nyc/road_density_clean.parquet] ...")
    rd = _load_road_density()
    if rd is None:
        print("  ⚠️ 未找到 road_density_clean.parquet, 用 0 占位")
        df["road_length_m"] = 0.0
        df["road_segment_count"] = 0
        df["road_density_km_per_km2"] = 0.0
        for tier in ["tier_highway", "tier_major", "tier_minor", "tier_local", "tier_other"]:
            df[f"road_len_{tier}_m"] = 0.0
        return df
    df = df.merge(rd, on="grid_id", how="left")
    for col in ["road_length_m", "road_density_km_per_km2"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(np.float32)
    if "road_segment_count" in df.columns:
        df["road_segment_count"] = df["road_segment_count"].fillna(0).astype(np.int32)
    if "area_m2" in df.columns:
        df = df.drop(columns=["area_m2"])
    tier_cols = [c for c in rd.columns if c.startswith("road_len_")]
    for col in tier_cols:
        df[col] = df[col].fillna(0.0).astype(np.float32)
    return df


# ============== POI 特征 (poi_clean.parquet -> grid_id 聚合) ==============
# 1. POI -> grid_id (按 lon/lat 落格, 用 0.0045° ≈ 500m)
# 2. 按 grid_id groupby 算 poi_total_count + 各类计数
NYC_LON_MIN, NYC_LAT_MIN = -74.26, 40.49
GRID_SIZE_DEG = 0.0045  # ≈ 500m


def _poi_to_grid(poi_df):
    """把 POI 按 0.0045° 网格分桶, 生成 grid_id 字符串"""
    poi_df = poi_df.copy()
    lon = poi_df["lon"].values
    lat = poi_df["lat"].values
    col = np.floor((lon - NYC_LON_MIN) / GRID_SIZE_DEG).astype(np.int32)
    row = np.floor((lat - NYC_LAT_MIN) / GRID_SIZE_DEG).astype(np.int32)
    poi_df["grid_id"] = [f"nyc_g{int(c):04d}_{int(r):04d}" for c, r in zip(col, row)]
    return poi_df


def _load_poi_aggregated():
    """读 cleaned_nyc/poi_clean.parquet, 按 grid_id + category 聚合."""
    poi_path = os.path.join(CLEANED_DIR, "poi_clean.parquet")
    if not os.path.exists(poi_path):
        return None
    poi = pd.read_parquet(poi_path)
    if poi.empty:
        return None
    print(f"  POI 原始: {len(poi):,} 行, category 取值: {poi['category'].value_counts().head().to_dict()}")
    poi = _poi_to_grid(poi)

    # 各类 POI 计数 (wide 表: 1 行/grid, N 列/category)
    pivot = (
        poi.groupby(["grid_id", "category"]).size()
           .unstack(fill_value=0)
           .reset_index()
    )
    # 列名标准化: poi_count_<category>
    pivot.columns = ["grid_id"] + [f"poi_count_{c}" for c in pivot.columns[1:]]
    # 总数
    cnt_cols = [c for c in pivot.columns if c.startswith("poi_count_")]
    pivot["poi_total_count"] = pivot[cnt_cols].sum(axis=1).astype(np.int32)
    print(f"  网格级 POI 表: {len(pivot):,} 行, 类别列: {len(cnt_cols)}")
    return pivot


def build_ndvi_features(df):
    """NDVI 卫星植被指数 (MOD13A1, 500m, 16-day composite) -> 网格聚合"""
    print("[NDVI 卫星特征 — cleaned_nyc/ndvi_clean.parquet] ...")
    ndvi_path = os.path.join(CLEANED_DIR, "ndvi_clean.parquet")
    if not os.path.exists(ndvi_path):
        print("  ⚠️ 未找到 ndvi_clean.parquet, 用 0 占位")
        df["ndvi_mean"] = 0.0
        df["ndvi_max"] = 0.0
        df["ndvi_min"] = 0.0
        df["ndvi_count"] = 0
        df["is_water"] = 1
        return df
    ndvi = pd.read_parquet(ndvi_path)
    ndvi["grid_id"] = ndvi["grid_id"].astype(str)
    keep_cols = ["grid_id", "ndvi_mean", "ndvi_max", "ndvi_min", "ndvi_count", "is_water"]
    ndvi = ndvi[[c for c in keep_cols if c in ndvi.columns]]
    print(f"  NDVI 表: {len(ndvi):,} 行")
    print(f"  ndvi_mean 范围: {ndvi['ndvi_mean'].min():.3f} ~ {ndvi['ndvi_mean'].max():.3f}")
    df = df.merge(ndvi, on="grid_id", how="left")
    df["ndvi_mean"]  = df["ndvi_mean"].fillna(0.0).astype(np.float32)
    df["ndvi_max"]   = df["ndvi_max"].fillna(0.0).astype(np.float32)
    df["ndvi_min"]   = df["ndvi_min"].fillna(0.0).astype(np.float32)
    df["ndvi_count"] = df["ndvi_count"].fillna(0).astype(np.int32)
    df["is_water"]   = df["is_water"].fillna(1).astype(np.int32)
    return df


def build_poi_features(df):
    print("[POI 特征 — cleaned_nyc/poi_clean.parquet] ...")
    pivot = _load_poi_aggregated()
    if pivot is None:
        print("  ⚠️ 未找到 poi_clean.parquet, 用 0 占位")
        df["poi_total_count"] = 0
        df["poi_density_per_km2"] = 0.0
        return df
    df = df.merge(pivot, on="grid_id", how="left")
    cnt_cols = [c for c in df.columns if c.startswith("poi_count_")]
    for c in cnt_cols + ["poi_total_count"]:
        df[c] = df[c].fillna(0).astype(np.int32) if "count" in c else df[c]
    df["poi_total_count"] = df["poi_total_count"].fillna(0).astype(np.int32)
    df["poi_density_per_km2"] = (df["poi_total_count"] / 0.25).astype(np.float32)
    return df


# ============== taxi zone -> grid_id 映射 ==============
def _build_zone_to_grid(grid_gdf):
    """读 taxi_zone_lookup.csv + taxi_zones.shp, 把 zone_id 映射到最近 grid_id (str)."""
    zone_csv   = os.path.join(RAW_TAXI_ZONES, "taxi_zone_lookup.csv")
    shape_path = os.path.join(RAW_TAXI_ZONES, "taxi_zones", "taxi_zones.shp")

    if not os.path.exists(zone_csv):
        print(f"  ⚠️ 缺 {zone_csv}")
        return {}

    zones = pd.read_csv(zone_csv)
    if os.path.exists(shape_path):
        zgdf = gpd.read_file(shape_path).to_crs("EPSG:4326")
        zones = zones.merge(zgdf[["LocationID", "geometry"]], on="LocationID", how="left")
        zones = gpd.GeoDataFrame(zones, geometry="geometry", crs="EPSG:4326")
        zones["lon"] = zones.geometry.centroid.x
        zones["lat"] = zones.geometry.centroid.y
    else:
        zones["lon"] = -73.95
        zones["lat"] =  40.75
        print("  ⚠️ 无 shapefile, 用 NYC 中心点兜底")

    grid_pts = grid_gdf[["grid_id", "centroid_lon", "centroid_lat"]].copy()
    grid_pts["__k"] = 1
    zone_pts = zones[["LocationID", "lon", "lat"]].copy()
    zone_pts["__k"] = 1

    # 向量化 haversine
    zlon = zone_pts["lon"].values
    zlat = zone_pts["lat"].values
    glon = grid_pts["centroid_lon"].values
    glat = grid_pts["centroid_lat"].values
    n_z, n_g = len(zlon), len(glon)

    # 矩阵 (n_z, n_g) 内存太大时分批; NYC zone=263, grid≈1.6万, 一次约 16MB float64 可接受
    from numpy import radians, sin, cos, arcsin, sqrt
    zlon_r, zlat_r = radians(zlon), radians(zlat)
    glon_r, glat_r = radians(glon), radians(glat)
    # 用 chunks 分批算最近
    BATCH = 32
    best = np.full(n_z, -1, dtype=np.int64)
    best_dist = np.full(n_z, np.inf)
    for i in range(0, n_g, BATCH):
        glon_b = glon_r[i:i+BATCH][None, :]      # (1, b)
        glat_b = glat_r[i:i+BATCH][None, :]
        dlon = glon_b - zlon_r[:, None]          # (n_z, b)
        dlat = glat_b - zlat_r[:, None]
        a = sin(dlat/2)**2 + cos(zlat_r[:, None]) * cos(glat_b) * sin(dlon/2)**2
        dist = 2 * 6371.0088 * arcsin(sqrt(a))    # km
        idx = dist.argmin(axis=1)
        d   = dist[np.arange(n_z), idx]
        improve = d < best_dist
        best[improve] = i + idx[improve]
        best_dist[improve] = d[improve]
    grid_ids = grid_pts["grid_id"].values[best.clip(0, n_g-1)]
    return dict(zip(zones["LocationID"].values, grid_ids))


# ============== 出租车特征 (taxi_nyc_clean.parquet, PULocationID -> grid_id) ==============
def build_taxi_flow_features(df, grid_gdf):
    print("[Taxi 人流特征 — cleaned_nyc/taxi_nyc_clean.parquet] ...")
    taxi_path = os.path.join(CLEANED_DIR, "taxi_nyc_clean.parquet")
    if not os.path.exists(taxi_path):
        print("  ⚠️ 未找到 taxi_nyc_clean.parquet")
        df["taxi_pickup_count"] = 0
        df["taxi_dropoff_count"] = 0
        return df
    try:
        zone_to_grid = _build_zone_to_grid(grid_gdf)
        if not zone_to_grid:
            df["taxi_pickup_count"] = 0
            df["taxi_dropoff_count"] = 0
            return df
        print(f"  zone→grid 映射: {len(zone_to_grid)} zones")

        cols_needed = ["PULocationID", "DOLocationID", "pickup_datetime", "dropoff_datetime"]
        taxi_df = pd.read_parquet(taxi_path, columns=cols_needed)
        print(f"  Taxi 记录: {len(taxi_df):,} 行")

        # 截到小时
        pickup_grid  = taxi_df["PULocationID"].map(zone_to_grid)
        dropoff_grid = taxi_df["DOLocationID"].map(zone_to_grid)
        pu_dt = pd.to_datetime(taxi_df["pickup_datetime"], utc=True, errors="coerce")
        do_dt = pd.to_datetime(taxi_df["dropoff_datetime"], utc=True, errors="coerce")

        pickup_cnt = (
            pd.DataFrame({
                "grid_id":   pickup_grid.values,
                "weekday":   pu_dt.dt.weekday.values,
                "hour":      pu_dt.dt.hour.values,
            })
            .dropna(subset=["grid_id"])
            .groupby(["grid_id", "weekday", "hour"]).size()
            .reset_index(name="taxi_pickup_count")
        )
        dropoff_cnt = (
            pd.DataFrame({
                "grid_id":   dropoff_grid.values,
                "weekday":   do_dt.dt.weekday.values,
                "hour":      do_dt.dt.hour.values,
            })
            .dropna(subset=["grid_id"])
            .groupby(["grid_id", "weekday", "hour"]).size()
            .reset_index(name="taxi_dropoff_count")
        )

        df = df.merge(pickup_cnt,  on=["grid_id", "weekday", "hour"], how="left")
        df = df.merge(dropoff_cnt, on=["grid_id", "weekday", "hour"], how="left")
        df["taxi_pickup_count"]  = df["taxi_pickup_count"].fillna(0).astype(np.int32)
        df["taxi_dropoff_count"] = df["taxi_dropoff_count"].fillna(0).astype(np.int32)
        print(f"  pickup 总数: {df['taxi_pickup_count'].sum():,}, dropoff 总数: {df['taxi_dropoff_count'].sum():,}")

        del taxi_df, pickup_cnt, dropoff_cnt, pickup_grid, dropoff_grid
        gc.collect()
    except Exception as e:
        print(f"  Taxi 特征构建失败: {e}")
        import traceback
        traceback.print_exc()
        df["taxi_pickup_count"] = 0
        df["taxi_dropoff_count"] = 0
    return df


# ============== 主函数 ==============
def main():
    print("=" * 60)
    print(f"Week 2 步骤 3 (低内存版, 真实 schema) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    grid_gdf = load_grid()
    df = build_space_time_index(grid_gdf)
    df = build_time_features(df)
    df = build_spatial_features(df)
    df = build_poi_features(df)
    df = build_road_features(df)
    df = build_weather_features(df)
    df = build_ndvi_features(df)
    df = build_taxi_flow_features(df, grid_gdf)

    # 固定列 + 动态 POI 类别列
    # 固定列 + 动态 POI 类别列 + 动态道路等级列
    base_cols = [
        "grid_id", "datetime", "hour", "weekday", "month",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
        "is_weekend", "is_holiday",
        "dist_to_times_square", "dist_to_central_park", "dist_to_wall_street",
        "dist_to_grand_central", "dist_to_jfk_airport", "dist_to_city_center",
        "dist_to_nearest_landmark",
        "poi_total_count", "poi_density_per_km2",
        "road_length_m", "road_segment_count", "road_density_km_per_km2",
        "taxi_pickup_count", "taxi_dropoff_count",
        "weather_temp_norm", "weather_humidity_norm", "weather_pressure_norm",
        "weather_wind_speed_norm", "weather_clouds_norm",
        "ndvi_mean", "ndvi_max", "ndvi_min", "ndvi_count", "is_water",
    ]
    poi_count_cols = [c for c in df.columns if c.startswith("poi_count_")]
    road_tier_cols = [c for c in df.columns if c.startswith("road_len_")]
    feature_cols = base_cols + poi_count_cols + road_tier_cols
    df = df[feature_cols]
    print(f"\n[最终] 形状: {df.shape}, 内存: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  POI 类别列: {poi_count_cols}")
    print(f"  道路等级列: {road_tier_cols}")

    out_path = os.path.join(OUT_DIR, "nyc_features.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[OK] 已保存: {out_path}")

    log = {
        "started": datetime.now().isoformat(),
        "n_rows": int(len(df)),
        "n_grids": int(df["grid_id"].nunique()),
        "n_hours": int(df.groupby(["weekday", "hour"]).ngroups),
        "features": feature_cols,
    }
    with open(os.path.join(OUT_DIR, "step3_log.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print("[OK] 日志已保存")


if __name__ == "__main__":
    main()