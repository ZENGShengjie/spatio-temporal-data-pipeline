"""
Week 2 — 步骤 2: 纽约市 500m×500m 规则网格划分

- 将 NYC 划分为 500m×500m 的规则网格
- 计算每个网格的地理边界、中心点坐标
- 建立 grid_id ↔ 经纬度映射关系
- 输出 GeoJSON + CSV

输入: 无需原始数据（仅 NYC 边界）
输出: /home/ubuntu/amazon/grid_nyc/*

用法:
    python step2_grid.py
"""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import box, Point, Polygon
from shapely.ops import transform as shapely_transform
import pyproj
from functools import partial

warnings_filter_was_here = True  # placeholder
import warnings
warnings.filterwarnings("ignore")

# ========== NYC 范围（经纬度 WGS84）==========
NYC_LON_MIN, NYC_LAT_MIN = -74.26, 40.49
NYC_LON_MAX, NYC_LAT_MAX = -73.70, 40.92

GRID_SIZE_METERS = 500  # 500m × 500m

OUT_DIR = os.getenv("GRID_DIR", "/home/ubuntu/amazon/grid_nyc")
os.makedirs(OUT_DIR, exist_ok=True)


def lonlat_to_m_mercator(lon, lat):
    """经纬度 -> Web Mercator EPSG:3857 (米)"""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transformer.transform(lon, lat)


def m_mercator_to_lonlat(x, y):
    """Web Mercator (米) -> 经纬度 WGS84"""
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


def build_grid():
    print("[Step 2] 构建 NYC 500m×500m 网格 ...")
    print(f"  区域: ({NYC_LON_MIN}, {NYC_LAT_MIN}) → ({NYC_LON_MAX}, {NYC_LAT_MAX})")

    # 1. 转 Mercator 米坐标（保证 500m 是真正的 500m）
    x_min, y_min = lonlat_to_m_mercator(NYC_LON_MIN, NYC_LAT_MIN)
    x_max, y_max = lonlat_to_m_mercator(NYC_LON_MAX, NYC_LAT_MAX)

    print(f"  Mercator 范围: X=[{x_min:.0f}, {x_max:.0f}] Y=[{y_min:.0f}, {y_max:.0f}]")

    # 2. 按 GRID_SIZE_METERS 切分
    g = GRID_SIZE_METERS
    x_coords = np.arange(x_min, x_max + g, g)
    y_coords = np.arange(y_min, y_max + g, g)

    n_cols = len(x_coords) - 1
    n_rows = len(y_coords) - 1
    n_total = n_cols * n_rows

    print(f"  网格数: {n_rows} 行 × {n_cols} 列 = {n_total} 个网格")

    # 3. 构建每个网格
    records = []
    geometries = []

    for i in range(n_rows):
        for j in range(n_cols):
            x0, x1 = x_coords[j], x_coords[j + 1]
            y0, y1 = y_coords[i], y_coords[i + 1]

            # 网格多边形 (Mercator)
            poly_m = box(x0, y0, x1, y1)

            # 转回 WGS84
            poly_wgs = shapely_transform(
                partial(
                    lambda x, y: m_mercator_to_lonlat(x, y),
                ),
                poly_m
            )

            # 中心点 (Mercator → WGS84)
            cx_m = (x0 + x1) / 2
            cy_m = (y0 + y1) / 2
            cx_wgs, cy_wgs = m_mercator_to_lonlat(cx_m, cy_m)

            # 经纬度边界的 min/max
            bounds = poly_wgs.bounds  # (minx, miny, maxx, maxy)
            lon_min, lat_min, lon_max, lat_max = bounds

            grid_id = f"nyc_g{i:04d}_{j:04d}"

            records.append({
                "grid_id": grid_id,
                "row": i,
                "col": j,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "lon_center": round(cx_wgs, 6),
                "lat_center": round(cy_wgs, 6),
                "lon_min": round(lon_min, 6),
                "lat_min": round(lat_min, 6),
                "lon_max": round(lon_max, 6),
                "lat_max": round(lat_max, 6),
                "area_m2": poly_m.area,
                "width_m":  x1 - x0,
                "height_m": y1 - y0,
            })
            geometries.append(poly_wgs)

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

    # 4. 保存
    out_geojson = os.path.join(OUT_DIR, "nyc_grid_500m.geojson")
    out_csv     = os.path.join(OUT_DIR, "nyc_grid_id_mapping.csv")
    out_parquet = os.path.join(OUT_DIR, "nyc_grid_500m.parquet")

    gdf.to_file(out_geojson, driver="GeoJSON")
    df.drop(columns=["geometry"], errors="ignore").to_csv(out_csv, index=False)
    df.drop(columns=["geometry"], errors="ignore").to_parquet(out_parquet, index=False)

    print(f"\n  ✅ GeoJSON: {out_geojson}")
    print(f"  ✅ CSV:     {out_csv}")
    print(f"  ✅ Parquet: {out_parquet}")

    # 5. 统计
    print(f"\n  网格统计:")
    print(f"    总网格数: {len(gdf)}")
    print(f"    总面积: {gdf['area_m2'].sum() / 1e6:.2f} km²")
    print(f"    NYC 估算面积: ~783 km²")
    print(f"    覆盖率: {gdf['area_m2'].sum() / (783 * 1e6) * 100:.1f}%")

    return gdf, df


def main():
    print("=" * 60)
    print(f"Week 2 步骤 2: 网格划分 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"网格尺寸: {GRID_SIZE_METERS}m × {GRID_SIZE_METERS}m")
    print(f"输出目录: {OUT_DIR}")
    print("=" * 60)

    gdf, df = build_grid()

    print("\n" + "=" * 60)
    print(f"网格划分完成! 共 {len(gdf)} 个网格")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
