import rasterio
from rasterio.crs import CRS

# 测试 rasterio 能不能识别 EPSG:6842
try:
    crs = CRS.from_epsg(6842)
    print(f"CRS: {crs}")
    print(f"linear_units: {crs.linear_units}")
except Exception as e:
    print(f"FAIL: {e}")

# 测试 wkt 是否 ok
try:
    from rasterio.warp import calculate_default_transform
    import numpy as np
    with rasterio.open("/home/ubuntu/amazon/tmp_ndvi/MOD13A1.A2024081.h12v04.061.2024099104930.ndvi.tif") as src:
        print(f"src.crs={src.crs}")
        print(f"src.transform={src.transform}")
        print(f"src.bounds={src.bounds}")
        # reproject src -> EPSG:4326 NYC bbox
        t, w, h = calculate_default_transform(src.crs, "EPSG:4326",
                                               src.width, src.height,
                                               *src.bounds,
                                               resolution=0.005)
        print(f"OK default transform: {w}x{h}, transform={t}")
except Exception as e:
    print(f"FAIL calculate_default_transform: {e}")