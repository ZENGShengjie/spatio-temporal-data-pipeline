import glob, os
from pyhdf.SD import SD, SDC

hd = sorted(glob.glob("/home/ubuntu/amazon/tmp_ndvi/*.hdf"))[0]
print("file:", os.path.basename(hd))
h = SD(hd, SDC.READ)
print("datasets:")
for name, info in h.datasets().items():
    print(f"  {name}: shape={info[0]} dtype={info[1]} dim={info[3]}")
sds = h.select("500m 16 days NDVI")
arr = sds[:, :]
import numpy as np
unique = np.unique(arr)
print(f"\nNDVI raw: shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}")
print(f"unique count: {len(unique)}, first 30: {unique[:30].tolist()}")
print(f"nonzero count: {(arr != 0).sum()}, negative count: {(arr < 0).sum()}")
h.end()