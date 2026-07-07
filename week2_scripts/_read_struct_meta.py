"""读 MOD13A1 HDF 里的 StructMetadata.0 看真实 tile 角点"""
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF, HC
import re

hd = "/home/ubuntu/amazon/tmp_ndvi/MOD13A1.A2024081.h12v04.061.2024099104930.hdf"
hdf = HDF(hd, HC.READ)
attrs = hdf.attrinfo()
struct_meta = None
for a in attrs:
    if a[0] == b"StructMetadata.0":
        struct_meta = a[1]
        break

if struct_meta is None:
    print("no StructMetadata.0")
else:
    text = struct_meta.decode("utf-8", errors="ignore")
    # 找 UpperLeftPointMtrs / LowerRightPointMtrs
    for tag in ("UpperLeftPointMtrs", "LowerRightPointMtrs"):
        m = re.search(rf"{tag}\s*=\s*\(([-\d.,\s]+)\)", text)
        if m:
            vals = [float(v) for v in re.findall(r"-?\d+\.?\d*", m.group(1))]
            print(f"{tag}: {vals}")

    # 找 Projection, GridOrigin
    for tag in ("Projection", "GridOrigin", "GridResolution"):
        m = re.search(rf"{tag}\s*=\s*([^\n]+)", text)
        if m:
            print(f"{tag}: {m.group(1).strip()[:80]}")

    # 找 sinu 中心 lon_0
    print("---")
    print("Projection block excerpt:")
    for ln in text.split("\n"):
        if "PROJECTION" in ln.upper() or "PARAMETER" in ln.upper() or "GRID" in ln.upper():
            print(ln.strip()[:200])
hdf.close()