import rasterio
import numpy as np
tif = "/home/ubuntu/amazon/tmp_ndvi/MOD13A1.A2024081.h12v04.061.2024099104930.ndvi.tif"
with rasterio.open(tif) as src:
    a = src.read(1)
    print(f"shape={a.shape} crs={src.crs}")
    print(f"bounds={src.bounds}")
    print(f"transform={src.transform}")
    print(f"valid: min={np.nanmin(a):.4f} max={np.nanmax(a):.4f} mean={np.nanmean(a):.4f}")
    print(f"nonzero: {(a!=0).sum()} / {a.size}, nan: {np.isnan(a).sum()}")
    # 看中心 100x100 切片
    print("center 10x10 sample:")
    print(a[1000:1010, 1000:1010])