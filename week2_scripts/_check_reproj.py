import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

tif = "/home/ubuntu/amazon/tmp_ndvi/MOD13A1.A2024081.h12v04.061.2024099104930.ndvi.tif"

# reproject 到 NYC bbox, 0.001° 分辨率
NYC_BBOX = (-74.30, 40.45, -73.65, 41.05)
dst_res = 0.001
west, south, east, north = NYC_BBOX
width  = int(np.ceil((east - west)  / dst_res))
height = int(np.ceil((north - south) / dst_res))
print(f"target shape: {width}x{height}")
transform = rasterio.transform.from_bounds(west, south,
                                            west + width  * dst_res,
                                            south + height * dst_res,
                                            width, height)
arr = np.full((height, width), np.nan, dtype=np.float32)
with rasterio.open(tif) as src:
    reproject(source=rasterio.band(src, 1), destination=arr,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=transform, dst_crs="EPSG:4326",
              resampling=Resampling.bilinear)
print(f"arr shape: {arr.shape}")
print(f"valid: min={np.nanmin(arr):.4f} max={np.nanmax(arr):.4f} mean={np.nanmean(arr):.4f}")
print(f"nonzero: {(arr!=0).sum()} / {arr.size}, nan: {np.isnan(arr).sum()}")
print(f"sample 5x5:\n{arr[300:305, 300:305]}")