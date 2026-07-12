#!/usr/bin/env python3
"""检查 EC2 Python 依赖是否齐全"""
pkgs = ["numpy", "pandas", "geopandas", "pyarrow", "scikit-learn",
        "scipy", "shapely", "pyproj", "boto3", "torch"]
missing = []
for p in pkgs:
    try:
        __import__(p)
        print(f"OK  {p}")
    except ImportError:
        print(f"MISS  {p}")
        missing.append(p)
if missing:
    print(f"\n# Install missing:")
    print(f"pip install --user {' '.join(missing)}")
else:
    print("\nAll deps installed!")
