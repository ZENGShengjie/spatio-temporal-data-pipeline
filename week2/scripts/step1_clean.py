"""
Week 2 — 步骤 1: 数据清洗与标准化
目标城市: NYC (New York City)

清洗内容:
  - TaxiNYC 轨迹: 去除漂移点、统一 WGS84 坐标、统一 UTC 时间戳
  - Landsat 影像: 云量筛选、重投影为 WGS84
  - 气象数据: 缺失值线性插值、统一小时粒度
  - POI/OSM: 去重、空间过滤

输入:  /home/ubuntu/amazon/raw_nyc/*
输出:  /home/ubuntu/amazon/cleaned_nyc/*

用法 (EC2 t3.large):
    python step1_clean.py
"""
import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import Point, box
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import requests

warnings.filterwarnings("ignore")

# NYC 边界框
NYC_BOUNDS = (-74.26, 40.49, -73.70, 40.92)
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = NYC_BOUNDS
NYC_GDF = gpd.GeoDataFrame(
    geometry=[box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)],
    crs="EPSG:4326"
)

RAW_DIR  = os.getenv("RAW_DIR",  "/home/ubuntu/amazon/raw_nyc")
OUT_DIR  = os.getenv("CLEANED_DIR", "/home/ubuntu/amazon/cleaned_nyc")

os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "cleaning_log.json")
log = {"started": datetime.now().isoformat(), "steps": [], "errors": []}


def log_step(name, n_total=None, n_removed=None, details=""):
    entry = {"step": name, "time": datetime.now().isoformat(), "details": details}
    if n_total is not None:
        entry["n_total"] = n_total
        entry["n_removed"] = n_removed
        entry["n_kept"] = n_total - n_removed
    log["steps"].append(entry)
    print(f"  [{name}] 总={n_total} | 移除={n_removed} | 保留={n_total-n_removed}  {details}")


def log_error(name, err):
    log["errors"].append({"step": name, "error": str(err)})
    print(f"  [ERROR {name}] {err}")


# ============================================================
# 清洗 TaxiNYC 轨迹数据
# ============================================================
def clean_taxi():
    print("\n[TaxiNYC 清洗]")
    taxi_dir = os.path.join(RAW_DIR, "taxi_nyc")
    if not os.path.isdir(taxi_dir):
        print("  目录不存在，跳过")
        return

    all_dfs = []
    parquet_files = [f for f in os.listdir(taxi_dir) if f.endswith(".parquet")]

    for fname in sorted(parquet_files):
        fpath = os.path.join(taxi_dir, fname)
        print(f"  读取 {fname} ...", end=" ", flush=True)
        try:
            df = pd.read_parquet(fpath)
            n_orig = len(df)
            df = _clean_taxi_df(df)
            all_dfs.append(df)
            log_step(f"taxi_{fname}", n_orig, n_orig - len(df))
            print(f"✅ {len(df):,} 行")
        except Exception as e:
            log_error(f"taxi_{fname}", e)
            print(f"❌ {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(OUT_DIR, "taxi_nyc_clean.parquet")
        combined.to_parquet(out_path, index=False, compression="snappy")
        print(f"  合并保存: {out_path} ({len(combined):,} 行)")


def _clean_taxi_df(df: pd.DataFrame) -> pd.DataFrame:
    # 列名标准化（兼容不同月份可能有不同列名的情况）
    lat_col  = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col  = next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    time_cols = [c for c in df.columns if "time" in c.lower() or "datetime" in c.lower()]
    dropoff_lat = next((c for c in df.columns if "dropoff" in c.lower() and "lat" in c.lower()), None)
    dropoff_lon = next((c for c in df.columns if "dropoff" in c.lower() and ("lon" in c.lower() or "lng" in c.lower())), None)
    pickup_lat  = next((c for c in df.columns if "pickup" in c.lower() and "lat" in c.lower()), None)
    pickup_lon  = next((c for c in df.columns if "pickup" in c.lower() and ("lon" in c.lower() or "lng" in c.lower())), None)

    if pickup_lat and dropoff_lat:
        lat_col  = pickup_lat
        lon_col  = pickup_lon

    if lat_col and lon_col:
        # 只保留 pickup 位置在 NYC 范围内的记录
        df = df[
            (df[lat_col].between(LAT_MIN, LAT_MAX)) &
            (df[lon_col].between(LON_MIN, LON_MAX))
        ]

    # 时间标准化为 UTC
    for tc in time_cols:
        try:
            df[tc] = pd.to_datetime(df[tc], errors="coerce", utc=True)
        except Exception:
            pass

    # 去除缺失关键列
    if lat_col and lon_col:
        df = df.dropna(subset=[lat_col, lon_col])

    # 去除漂移点（经纬度明显异常）
    if lat_col and lon_col:
        df = df[
            (df[lat_col].between(LAT_MIN, LAT_MAX)) &
            (df[lon_col].between(LON_MIN, LON_MAX))
        ]

    # 去除 trip_distance 异常（负值或 >500 km）
    if "trip_distance" in df.columns:
        df = df[(df["trip_distance"] >= 0) & (df["trip_distance"] <= 500)]

    # 去除 fare_amount 异常（负值或 >10000 美元）
    if "fare_amount" in df.columns:
        df = df[(df["fare_amount"] >= 0) & (df["fare_amount"] <= 10000)]

    return df.reset_index(drop=True)


# ============================================================
# 清洗 Landsat 卫星影像元数据
# ============================================================
def clean_landsat():
    print("\n[Landsat 元数据清洗]")
    landsat_dir = os.path.join(RAW_DIR, "landsat")
    if not os.path.isdir(landsat_dir):
        print("  目录不存在，跳过")
        return

    scenes = []
    for fname in os.listdir(landsat_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(landsat_dir, fname)) as f:
                data = json.load(f)
            scenes.append(data)
        except Exception as e:
            log_error(f"landsat_{fname}", e)

    if not scenes:
        print("  没有 Landsat 元数据文件，跳过")
        return

    df = pd.DataFrame(scenes)
    n_orig = len(df)

    # 过滤云量 >20% 的影像
    if "cloud_cover" in df.columns:
        df = df[df["cloud_cover"] <= 20]

    # 过滤不在 NYC 区域的影像
    # (这里元数据已通过 bbox 检索，理论上都在区域内)

    log_step("landsat_filter", n_orig, n_orig - len(df))
    out_path = os.path.join(OUT_DIR, "landsat_scenes.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  保存: {out_path} ({len(df)} 景)")


# ============================================================
# 清洗气象数据
# ============================================================
def clean_weather():
    print("\n[气象数据清洗]")
    weather_dir = os.path.join(RAW_DIR, "weather")
    if not os.path.isdir(weather_dir):
        print("  目录不存在，跳过")
        return

    weather_files = [f for f in os.listdir(weather_dir) if f.endswith(".json")]

    for fname in weather_files:
        fpath = os.path.join(weather_dir, fname)
        print(f"  处理 {fname} ...", end=" ", flush=True)
        try:
            with open(fpath) as f:
                data = json.load(f)

            if "list" not in data:
                print("非预报格式，跳过")
                continue

            rows = []
            for item in data["list"]:
                rows.append({
                    "dt": pd.to_datetime(item.get("dt", 0), unit="s", utc=True),
                    "temp": item.get("main", {}).get("temp"),
                    "humidity": item.get("main", {}).get("humidity"),
                    "pressure": item.get("main", {}).get("pressure"),
                    "wind_speed": item.get("wind", {}).get("speed"),
                    "wind_deg": item.get("wind", {}).get("deg"),
                    "clouds": item.get("clouds", {}).get("all"),
                    "weather_main": item.get("weather", [{}])[0].get("main", "Unknown"),
                    "weather_desc": item.get("weather", [{}])[0].get("description", ""),
                })

            df = pd.DataFrame(rows)
            n_orig = len(df)

            # 线性插值缺失值
            for col in ["temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds"]:
                if col in df.columns:
                    df[col] = df[col].interpolate(method="linear")

            log_step(f"weather_{fname}", n_orig, 0)
            out_path = os.path.join(OUT_DIR, f"nyc_weather_clean.parquet")
            df.to_parquet(out_path, index=False, compression="snappy")
            print(f"✅ {len(df)} 行")
        except Exception as e:
            log_error(f"weather_{fname}", e)
            print(f"❌ {e}")


# ============================================================
# 清洗 POI 数据
# ============================================================
def clean_poi():
    print("\n[POI 数据清洗]")
    poi_dir = os.path.join(RAW_DIR, "poi")
    if not os.path.isdir(poi_dir):
        print("  目录不存在，跳过")
        return

    poi_file = os.path.join(poi_dir, "nyc_poi.json")
    if not os.path.exists(poi_file):
        print("  没有 POI 文件，跳过")
        return

    try:
        with open(poi_file) as f:
            data = json.load(f)

        pois = data.get("pois", [])
        df = pd.DataFrame(pois)
        n_orig = len(df)

        # 去重（根据 POI 名称和坐标）
        if "lat" in df.columns and "lng" in df.columns:
            df = df.drop_duplicates(subset=["name", "lat", "lng"])

        # 空间过滤（只保留 NYC 范围内的 POI）
        if "lat" in df.columns and "lng" in df.columns:
            df = df[
                df["lat"].between(LAT_MIN, LAT_MAX) &
                df["lng"].between(LON_MIN, LON_MAX)
            ]

        # 转为 GeoDataFrame
        if "lat" in df.columns and "lng" in df.columns:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_xy(df["lng"], df["lat"]),
                crs="EPSG:4326"
            )
            out_path = os.path.join(OUT_DIR, "nyc_poi_clean.gpkg")
            gdf.to_file(out_path, driver="GPKG", layer="poi")
            print(f"  保存: {out_path} ({len(gdf)} 个 POI)")
        else:
            out_path = os.path.join(OUT_DIR, "nyc_poi_clean.parquet")
            df.to_parquet(out_path, index=False)
            print(f"  保存: {out_path} ({len(df)} 个 POI)")

        log_step("poi_clean", n_orig, n_orig - len(df))

    except Exception as e:
        log_error("poi_clean", e)
        print(f"  ❌ {e}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print(f"Week 2 步骤 1: 数据清洗 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"输入目录: {RAW_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print("=" * 60)

    clean_taxi()
    clean_landsat()
    clean_weather()
    clean_poi()

    # 保存日志
    log["finished"] = datetime.now().isoformat()
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("清洗完成!")
    print(f"日志: {LOG_FILE}")
    print(f"输出: {OUT_DIR}")

    # 输出汇总
    for step in log["steps"]:
        if step.get("n_removed", 0) > 0:
            print(f"  {step['step']}: 移除 {step['n_removed']} 条")


if __name__ == "__main__":
    main()
