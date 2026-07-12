"""验证: 读一个 GeoTIFF, 检查 reproject 前后值"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

tif = "/home/ubuntu/amazon/tmp_ndvi/MOD13A1.A2024081.h12v04.061.2024099104930.ndvi.tif"

# 1) 读 Sinusoidal 原图
with rasterio.open(tif) as src:
    arr0 = src.read(1)
    print("=== Sinusoidal GeoTIFF ===")
    print(f"shape={arr0.shape}, crs={src.crs}")
    print(f"min={np.nanmin(arr0):.4f}, max={np.nanmax(arr0):.4f}, mean={np.nanmean(arr0):.4f}")
    print(f"nan count: {np.isnan(arr0).sum()}, nonzero: {(arr0 != 0).sum()}")
    valid = arr0[~np.isnan(arr0)]
    print(f"valid count: {len(valid)}, valid min/max: {valid.min():.4f}/{valid.max():.4f}")
    print(f"sample: {valid[:10]}")
    src_crs = src.crs

# 2) 重投影到 WGS84 NYC bbox
NYC_BBOX = (-74.30, 40.45, -73.65, 41.05)
dst_res = 0.001
west, south, east, north = NYC_BBOX
width  = int(np.ceil((east - west)  / dst_res))
height = int(np.ceil((north - south) / dst_res))
transform = rasterio.transform.from_bounds(west, south,
                                            west + width  * dst_res,
                                            south + height * dst_res,
                                            width, height)
arr1 = np.full((height, width), np.nan, dtype=np.float32)
with rasterio.open(tif) as src:
    reproject(source=rasterio.band(src, 1), destination=arr1,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=transform, dst_crs="EPSG:4326",
              resampling=Resampling.bilinear)

print(f"\n=== Reprojected WGS84 ===")
print(f"shape={arr1.shape}")
print(f"min={np.nanmin(arr1):.4f}, max={np.nanmax(arr1):.4f}, mean={np.nanmean(arr1):.4f}")
print(f"nan count: {np.isnan(arr1).sum()}, nonzero: {(arr1 != 0).sum()}")
valid = arr1[~np.isnan(arr1)]
print(f"valid count: {len(valid)}")
print(f"valid min/max: {valid.min():.4f}/{valid.max():.4f}")
print(f"sample: {valid[:10]}")