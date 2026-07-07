"""
独立脚本: 只跑 [网格提取 + 聚合 + 保存] 步骤
跳过搜索/下载/HDF转换 (假设上一步已下完)
"""
import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from step3b_ndvi import (
    CLEANED_DIR, GRID_DIR, TEMP_DIR, NYC_BBOX, hdf_path_to_ndvi_geotiff,
    extract_ndvi_per_grid,
)

if __name__ == "__main__":
    print("=" * 60)
    print("Step3b 重跑: 仅 网格提取 + 聚合 + 保存")
    print("=" * 60)
    tif_paths = sorted(glob_glob := __import__("glob").glob(
        os.path.join(TEMP_DIR, "*.ndvi.tif")
    ))
    print(f"找到 {len(tif_paths)} 个 GeoTIFF")
    if not tif_paths:
        print("没有 GeoTIFF, 退出")
        sys.exit(1)
    ndvi_df = extract_ndvi_per_grid(tif_paths)
    if ndvi_df.empty:
        sys.exit(2)

    agg = ndvi_df.groupby("grid_id", as_index=False).agg(
        ndvi_mean=("ndvi_mean", "mean"),
        ndvi_max=("ndvi_max", "max"),
        ndvi_min=("ndvi_min", "min"),
        ndvi_count=("ndvi_count", "sum"),
        sample_months=("sample_month", lambda s: sorted({int(x) for x in s})),
    )
    print(f"  网格聚合表: {len(agg):,} 行")
    print(f"  ndvi_mean 范围: {agg['ndvi_mean'].min():.3f} ~ {agg['ndvi_mean'].max():.3f}")

    out_path = os.path.join(CLEANED_DIR, "ndvi_clean.parquet")
    agg.to_parquet(out_path, index=False)
    print(f"\n[OK] 已保存: {out_path}")
    print(f"  head:\n{agg.head(3)}")