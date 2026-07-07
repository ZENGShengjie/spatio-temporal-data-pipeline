"""Test pyhdf VS API for reading StructMetadata"""
from pyhdf.HDF import HDF, HC
from pyhdf.VS import VS
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/amazon/tmp_ndvi/MOD13A1.A2024081.h12v04.061.2024099104930.hdf"

f = HDF(path, HC.READ)
print("file attrs:")
try:
    print(f.attributes())
except Exception as e:
    print("no attrs:", e)

vs = VS(f)
print(f"VS nrefs: {vs.nrefs()}")
for r in range(vs.nrefs()):
    vs.setid(r)
    for tag in (b"GRID", b"Projection", b"SwathStructure", b"Orbit"):
        n = vs.nentries(tag)
        if n > 0:
            print(f"  ref={r} tag={tag} nentries={n}")
            buf, size = vs.read(tag, n - 1)  # 读最后一个 entry
            print(f"    last entry: {buf[:500]}")

f.close()