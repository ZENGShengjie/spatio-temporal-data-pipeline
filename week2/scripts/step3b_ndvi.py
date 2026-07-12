"""
Week 2 — 步骤 3b: NDVI 卫星特征接入
从 NASA Earthdata Cloud 拉 MOD13A1 (500m, 16天合成, NDVI),
按 500m 网格取样后保存为 cleaned_nyc/ndvi_clean.parquet.

要求:
  - 先注册 https://urs.earthdata.nasa.gov/ 并把凭证写入 ~/.netrc
  - pip install earthaccess h5py rasterio

输入:
  /home/ubuntu/amazon/grid_nyc/nyc_grid_500m.geojson
输出:
  /home/ubuntu/amazon/cleaned_nyc/ndvi_clean.parquet
    列: grid_id (str), ndvi_mean (float), ndvi_max, ndvi_min, ndvi_count, sample_month (int)

跑法 (需要你提前完成 Earthdata 注册 + .netrc):
  python3 scripts/step3b_ndvi.py
"""
import os
import sys
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask

CLEANED_DIR = os.getenv("CLEANED_DIR", "/home/ubuntu/amazon/cleaned_nyc")
GRID_DIR    = os.getenv("GRID_DIR",    "/home/ubuntu/amazon/grid_nyc")
TEMP_DIR    = "/home/ubuntu/amazon/tmp_ndvi"
os.makedirs(TEMP_DIR, exist_ok=True)

# NYC 经纬度 bbox (覆盖 Manhattan+Brooklyn+Queens+Bronx+Staten Island)
NYC_BBOX = (-74.30, 40.45, -73.65, 41.05)

# 选择 2024 全年 (代表性样本)
DATE_RANGE = ("2024-04-01", "2024-10-31")


def ensure_earthaccess():
    """检查 earthaccess 是否可用, 不在则报错"""
    try:
        import earthaccess
        return earthaccess
    except ImportError:
        print("❌ 缺 earthaccess 库. 请运行:")
        print("   pip install earthaccess h5py rasterio")
        sys.exit(1)


def login_earthaccess(ea):
    """登录 NASA Earthdata (优先读 .netrc)"""
    print("[登录] 正在使用 ~/.netrc / EARTHDATA_USERNAME/PASSWORD 登录 ...")
    try:
        auth = ea.login(strategy="netrc")
        if not auth:
            auth = ea.login(strategy="environment")
        if not auth:
            print("⚠️ 自动登录失败, 进入交互登录")
            auth = ea.login(strategy="interactive")
        return auth
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        print("\n👉 请确认 ~/.netrc:")
        print('   machine urs.earthdata.nasa.gov login <USERNAME> password <PASSWORD>')
        sys.exit(1)


def download_mod13a1(ea):
    """搜索 MOD13A1 (500m, V061) 2024-04 ~ 2024-10 整个 NYC 覆盖范围"""
    print(f"\n[搜索] MODIS/061/MOD13A1: {DATE_RANGE[0]} ~ {DATE_RANGE[1]} bbox={NYC_BBOX}")
    results = ea.search_data(
        short_name="MOD13A1",
        version="061",
        temporal=DATE_RANGE,
        bounding_box=NYC_BBOX,
    )
    print(f"  找到 {len(results)} 个 HDF 切片")
    if len(results) == 0:
        sys.exit(2)
    return results


def download_to_dir(ea, results, out_dir=TEMP_DIR):
    """下载所有 .hdf 切片到本地"""
    print(f"\n[下载] -> {out_dir} (约 {len(results)} 个文件) ...")
    paths = ea.download(results, out_dir)
    print(f"  已下完 {len(paths)} 个文件")
    return paths


def hdf_to_ndvi_array(hdf_path):
    """
    MOD13A1 .hdf 里有 12 个 subdataset:
      0: 250m 16 days NDVI  (500m Native)
      1: 250m 16 days EVI
      ...
    取 subdataset 0 (NDVI), scale=0.0001, fill=-3000
    """
    import h5py
    with h5py.File(hdf_path, "r") as f:
        # MOD13A1 顶层结构: HDF4 -> groups, 先 scan
        def find_ndvi(obj):
            for k in obj.keys():
                if k.upper() == "NDVI":
                    return obj[k]
                v = obj[k]
                if hasattr(v, "keys"):
                    r = find_ndvi(v)
                    if r is not None:
                        return r
            return None
        g = find_ndvi(f)
        if g is None:
            return None, None
        ds = None
        for k in g.keys():
            if k in ("Data_Fields", "MODIS_Grid_500m_16_Day_VI"):
                ds = g[k]
                break
        if ds is None:
            ds = list(g.values())[0]
        ndvi_data = ds["500m_16_days_NDVI"][:].astype(np.float32)
        ndvi_data[ndvi_data <= -2000] = np.nan
        ndvi_data = ndvi_data * 0.0001
        # HDF 缺 lat/lon, 后面用 rasterio reproject
    return ndvi_data, None


def hdf_path_to_ndvi_geotiff(hdf_path, out_tif):
    """
    MOD13A1 HDF -> GeoTIFF (WGS84 NYC bbox, native 500m 分辨率)
    一步到位: Sinusoidal -> WGS84, 跳过中间 Sinusoidal GeoTIFF.
    """
    from pyhdf.SD import SD, SDC
    from rasterio.warp import reproject as rio_reproject, Resampling as RioResampling
    # 1) 读 NDVI 数据 (2400x2400 int16)
    h = SD(hdf_path, SDC.READ)
    sds_obj = h.select("500m 16 days NDVI")
    ndvi = sds_obj[:, :].astype(np.float32)
    ndvi[ndvi <= -2000] = np.nan
    ndvi = ndvi * 0.0001
    h.end()

    # 2) MODIS Sinusoidal tile 经纬度边界 (从 NASA 官方 sn_gring_10deg.txt 读)
    #    NYC 主要在 h12v04, 也涉及 h11v04/h12v05
    #    格式: (iv, ih) -> (W, E, S, N) 经纬度
    TILE_BBOX = {
        (4, 12):  (-93.3822, -65.0781, 39.7858,  50.0754),   # h12v04 (NYC main)
        (3, 11):  (-108.9007, -78.3136, 49.5789,  59.7053),  # h11v04 (north of NYC)
        (4, 11):  (-91.3388, -78.1497, 39.7728,  49.9863),   # h11v04 different row
        (5, 12):  (-78.3244, -57.7254, 29.8506,  40.0000),   # h12v05 (south of NYC)
    }
    import re as _re
    m = _re.search(r"h(\d{2})v(\d{2})", os.path.basename(hdf_path))
    if not m:
        return None
    ih = int(m.group(1)); iv = int(m.group(2))
    if (iv, ih) not in TILE_BBOX:  # key = (vertical, horizontal)
        print(f"  ❌ tile h{ih}v{iv} 不在 NYC 范围")
        return None
    west_lon, east_lon, south_lat, north_lat = TILE_BBOX[(iv, ih)]

    # 3) 把 G-ring 经纬度 bbox 转 Sinusoidal 米制 (用 pyproj)
    #    h12v04 真实米制尺寸 ~1134km x 1144km
    from pyproj import Transformer
    sinu_crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +no_defs"
    tf = Transformer.from_crs("EPSG:4326", sinu_crs, always_xy=True)
    # 用 G-ring 4 个角点 (经 NASA 文档)
    llx, lly = tf.transform(west_lon, south_lat)   # 西南 (LL)
    urx, ury = tf.transform(east_lon, north_lat)   # 东北 (UR)
    width_m  = urx - llx
    height_m = ury - lly
    # 像素大小 (500m native, 但只到 WGS84 投影后才会变化)
    src_transform = rasterio.transform.from_bounds(llx, lly, urx, ury, 2400, 2400)
    src_crs = sinu_crs

    # 4) 重投影到 WGS84 NYC bbox
    dst_res = 0.001
    west, south, east, north = NYC_BBOX
    width  = int(np.ceil((east - west)  / dst_res))
    height = int(np.ceil((north - south) / dst_res))
    dst_transform = rasterio.transform.from_bounds(west, south,
                                                    west + width  * dst_res,
                                                    south + height * dst_res,
                                                    width, height)
    arr = np.full((height, width), np.nan, dtype=np.float32)
    rio_reproject(
        source=ndvi,
        destination=arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=RioResampling.bilinear,
    )

    # 5) 直接写 WGS84 GeoTIFF
    with rasterio.open(out_tif, "w", driver="GTiff", height=height, width=width, count=1,
                       dtype="float32", crs="EPSG:4326", transform=dst_transform) as dst:
        dst.write(arr, 1)
    if not hasattr(hdf_path_to_ndvi_geotiff, "_logged"):
        hdf_path_to_ndvi_geotiff._logged = True
        valid = arr[~np.isnan(arr)]
        print(f"  [debug] h{ih}v{iv} -> {width}x{height} WGS84, "
              f"valid min={valid.min():.3f} max={valid.max():.3f} mean={valid.mean():.3f}")
    return out_tif


def extract_ndvi_per_grid(tif_paths):
    """GeoTIFF 已经是 WGS84 NYC bbox -> 直接按网格 centroids 取 ndvi 均值"""
    import rasterio.mask
    import rasterio.warp

    print(f"\n[网格提取] {len(tif_paths)} 个 GeoTIFF (WGS84, 直接采样)")

    grid_gdf = gpd.read_file(os.path.join(GRID_DIR, "nyc_grid_500m.geojson"))
    grid_gdf["grid_id"] = grid_gdf["grid_id"].astype(str)
    if grid_gdf.crs is None:
        grid_gdf = grid_gdf.set_crs("EPSG:4326")
    else:
        grid_gdf = grid_gdf.to_crs("EPSG:4326")
    print(f"  网格: {len(grid_gdf):,}, CRS={grid_gdf.crs}")

    centroids = np.array([(g.x, g.y) for g in grid_gdf.geometry.centroid], dtype=np.float64)
    grid_ids = grid_gdf["grid_id"].values

    ndvi_records = []
    for tif in tif_paths:
        basename = os.path.basename(tif).replace(".hdf", ".ndvi.tif")
        raw = basename.replace(".ndvi.tif", "")
        date_str = raw.split(".A")[1].split(".h")[0]
        from datetime import datetime as _dt
        try:
            yy, doy = int(date_str[:4]), int(date_str[4:7])
            sample_month = (_dt(yy, 1, 1) + __import__("datetime").timedelta(days=doy-1)).month
        except Exception:
            sample_month = 6

        with rasterio.open(tif) as src:
            arr = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs

        if crs and str(crs) != "EPSG:4326":
            # 万一是 Sinusoidal, reproject 到 WGS84 NYC bbox
            dst_res = 0.001
            west, south, east, north = NYC_BBOX
            width  = int(np.ceil((east - west)  / dst_res))
            height = int(np.ceil((north - south) / dst_res))
            dst_transform = rasterio.transform.from_bounds(west, south,
                                                            west + width  * dst_res,
                                                            south + height * dst_res,
                                                            width, height)
            arr_wgs = np.full((height, width), np.nan, dtype=np.float32)
            from rasterio.warp import reproject, Resampling
            reproject(source=arr, destination=arr_wgs,
                      src_transform=transform, src_crs=crs,
                      dst_transform=dst_transform, dst_crs="EPSG:4326",
                      resampling=Resampling.bilinear)
            arr = arr_wgs
            transform = dst_transform

        inv = ~transform
        cols_full, rows_full = inv * (centroids[:, 0], centroids[:, 1])
        cols_full = np.floor(cols_full).astype(np.int32)
        rows_full = np.floor(rows_full).astype(np.int32)
        in_bounds = (rows_full >= 2) & (rows_full < arr.shape[0] - 2) & \
                    (cols_full >= 2) & (cols_full < arr.shape[1] - 2)
        if not in_bounds.any():
            continue
        print(f"  {basename}: 范围 ({arr.shape[1]}x{arr.shape[0]}) "
              f"in_bounds: {in_bounds.sum():,}/{len(in_bounds):,}")

        half = 2  # 5x5 ~550m 物理窗口
        for i, gid in enumerate(grid_ids):
            if not in_bounds[i]:
                continue
            r, c = rows_full[i], cols_full[i]
            block = arr[r-half:r+half+1, c-half:c+half+1]
            valid = block[~np.isnan(block)]
            ndvi_records.append({
                "grid_id": gid,
                "sample_month": sample_month,
                "year_doy": date_str,
                "ndvi_mean": float(valid.mean()) if valid.size else np.nan,
                "ndvi_max":  float(valid.max()) if valid.size else np.nan,
                "ndvi_min":  float(valid.min()) if valid.size else np.nan,
                "ndvi_count": int(valid.size),
            })

    df = pd.DataFrame(ndvi_records)
    print(f"  提取记录: {len(df):,}  (含 NaN)")
    return df


def main():
    print("=" * 60)
    print(f"Week 2 步骤 3b — NDVI 接入  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 60)

    ea = ensure_earthaccess()

    # 1) 登录
    auth = login_earthaccess(ea)
    print(f"  登录状态: {'OK' if auth else 'FAIL'}")
    if not auth:
        sys.exit(1)

    # 2) 搜索 HDF
    results = download_mod13a1(ea)
    if not results:
        sys.exit(2)

    # 3) 下载 HDF
    hdf_paths = download_to_dir(ea, results)
    if not hdf_paths:
        sys.exit(3)

    # 4) HDF -> NDVI GeoTIFF
    print(f"\n[转换 HDF -> GeoTIFF] ({len(hdf_paths)} 个)")
    tif_paths = []
    bad = 0
    for h in hdf_paths:
        out = os.path.join(TEMP_DIR, os.path.basename(h).replace(".hdf", ".ndvi.tif"))
        try:
            r = hdf_path_to_ndvi_geotiff(h, out)
            if r is not None and os.path.exists(out):
                tif_paths.append(out)
            else:
                bad += 1
        except Exception as e:
            print(f"  转换失败 {h}: {e}")
            bad += 1
    print(f"  成功: {len(tif_paths)}, 失败: {bad}")
    if not tif_paths:
        sys.exit(4)

    # 5) 网格提取
    ndvi_df = extract_ndvi_per_grid(tif_paths)
    if ndvi_df.empty:
        sys.exit(5)

    # 6) 聚合 (按 grid_id 聚合, 取多月均值; NaN = 无 NDVI 数据, 后填默认值)
    #    全 15,875 网格都进 parquet (含水体)
    grid_gdf_for_agg = gpd.read_file(os.path.join(GRID_DIR, "nyc_grid_500m.geojson"))
    grid_gdf_for_agg["grid_id"] = grid_gdf_for_agg["grid_id"].astype(str)
    all_grids = pd.DataFrame({"grid_id": grid_gdf_for_agg["grid_id"].values})
    agg = ndvi_df.groupby("grid_id", as_index=False).agg(
        ndvi_mean=("ndvi_mean", "mean"),
        ndvi_max=("ndvi_max", "max"),
        ndvi_min=("ndvi_min", "min"),
        ndvi_count=("ndvi_count", "sum"),
        sample_months=("sample_month", lambda s: sorted({int(x) for x in s})),
    )
    # left join: 没采到 NDVI 的网格 (水体 / 边界外) 保留 NaN
    agg = all_grids.merge(agg, on="grid_id", how="left")
    n_land = agg["ndvi_mean"].notna().sum()
    n_water = agg["ndvi_mean"].isna().sum()
    print(f"  网格聚合表: {len(agg):,} 行  (陆地 NDVI: {n_land:,}, 水体/空: {n_water:,})")
    print(f"  ndvi_mean 范围 (陆地): {agg['ndvi_mean'].min():.3f} ~ {agg['ndvi_mean'].max():.3f}")
    # 全局均值 fallback
    global_mean = float(agg["ndvi_mean"].mean()) if n_land > 0 else 0.5
    print(f"  全局均值: {global_mean:.3f}  (用于填充水体)")
    # 填充策略: 水体 = 0 (NDVI 真实定义水体 ~ 0)
    agg["ndvi_mean"] = agg["ndvi_mean"].fillna(0.0)
    agg["ndvi_max"]  = agg["ndvi_max"].fillna(0.0)
    agg["ndvi_min"]  = agg["ndvi_min"].fillna(0.0)
    agg["ndvi_count"] = agg["ndvi_count"].fillna(0).astype(int)
    agg["sample_months"] = agg["sample_months"].apply(lambda x: x if isinstance(x, list) else [])
    agg["is_water"] = (agg["ndvi_count"] == 0).astype(int)
    print(f"  已填 NaN, 最终 ndvi_mean 范围: {agg['ndvi_mean'].min():.3f} ~ {agg['ndvi_mean'].max():.3f}")

    out_path = os.path.join(CLEANED_DIR, "ndvi_clean.parquet")
    agg.to_parquet(out_path, index=False)
    print(f"\n[OK] 已保存: {out_path}")

    log = {
        "fetched_at": datetime.now().isoformat(),
        "n_hdf": len(hdf_paths),
        "n_tif_ok": len(tif_paths),
        "n_grids": len(agg),
        "ndvi_mean_range": [float(agg["ndvi_mean"].min()), float(agg["ndvi_mean"].max())],
        "date_range": list(DATE_RANGE),
        "bbox": list(NYC_BBOX),
    }
    with open(os.path.join(CLEANED_DIR, "ndvi_fetch_log.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print("[OK] 日志已保存")


if __name__ == "__main__":
    main()