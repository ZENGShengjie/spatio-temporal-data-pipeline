"""
NASA Landsat 下载 - 最终版（生产脚本）
使用 earthaccess + HLSL30 数据源
注意：这是远程服务器运行版本，OUT_DIR 写死为 Linux 路径
"""
import requests
import earthaccess
import os
import time

# 凭证从环境变量读取
USERNAME = os.getenv("NASA_USERNAME")
PASSWORD = os.getenv("NASA_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("请先设置环境变量 NASA_USERNAME 和 NASA_PASSWORD")
os.environ["EARTHDATA_USERNAME"] = USERNAME
os.environ["EARTHDATA_PASSWORD"] = PASSWORD

# 服务器输出目录
OUT_DIR = os.getenv("LANDSAT_OUT_DIR", "/home/ubuntu/amazon/raw_landsat")
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 诊断函数 ============
def test_cmr_direct(url, params, label):
    print(f"  [{label}] ", end="", flush=True)
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        # ECHO JSON 格式
        if "feed" in data:
            hits = data.get("feed", {}).get("hits", {}).get("total", 0)
            entries = data.get("feed", {}).get("entry", [])
            print(f"hits={hits}, entries={len(entries)}")
            if entries:
                print(f"    第一个: {entries[0].get('id', 'N/A')[:80]}")
            return entries
        # UMM JSON 格式
        elif "items" in data:
            items = data.get("items", [])
            print(f"items={len(items)}")
            return items
        else:
            print(f"unknown format: {list(data.keys())}")
            return []
    except Exception as e:
        print(f"ERROR: {e}")
        return []

# ============ Step 1: 登录 ============
print("=" * 60)
print("Step 1: 登录 NASA Earthdata...")
auth = earthaccess.login(strategy="environment", persist=True)
print(f"  登录状态: {'成功' if auth.authenticated else '失败'}")
session = earthaccess.get_requests_https_session()
print(f"  Session Bearer token: {session.headers.get('Authorization', 'N/A')[:50]}...")

# ============ Step 2: 诊断 - 测试不同数据源 ============
print("\n" + "=" * 60)
print("Step 2: 诊断不同数据源...")

cmr_base = "https://cmr.earthdata.nasa.gov/search/granules.json"

# 测试 Landsat C2 L2（各种方式）
print("\n  --- Landsat C2 L2 ---")
test_cmr_direct(cmr_base, {
    "short_name": "LANDSAT_OT_C2_L2",
    "version": "02",
    "temporal": "2024-01-01,2024-12-31",
    "page_size": 3,
}, "LANDSAT_OT_C2_L2 v02 无过滤")

test_cmr_direct(cmr_base, {
    "short_name": "LANDSAT_OT_C2_L2",
    "version": "02",
    "temporal": "2024-01-01,2024-12-31",
    "bounding_box": "-125,25,-66,49",  # 美国本土
    "page_size": 3,
}, "LANDSAT_OT_C2_L2 美国区域")

# 测试 HLSL30 (Landsat from HLS project)
print("\n  --- HLSL30 (HLS Landsat) ---")
test_cmr_direct(cmr_base, {
    "short_name": "HLSL30",
    "version": "2.0",
    "temporal": "2024-06-01,2024-06-30",
    "bounding_box": "-125,25,-66,49",
    "page_size": 3,
}, "HLSL30 v2.0 美国")

test_cmr_direct(cmr_base, {
    "short_name": "HLSL30",
    "version": "2.0",
    "temporal": "2024-06-01,2024-06-30",
    "page_size": 3,
}, "HLSL30 v2.0 全球")

# 测试 MODIS（确认 CMR 可用）
print("\n  --- MODIS ---")
test_cmr_direct(cmr_base, {
    "short_name": "MOD09GQ",
    "version": "006",
    "temporal": "2024-06-01,2024-06-30",
    "page_size": 3,
}, "MOD09GQ v006")

# ============ Step 3: earthaccess.search_data 诊断 ============
print("\n" + "=" * 60)
print("Step 3: earthaccess.search_data 诊断...")

test_cases = [
    {"short_name": "HLSL30", "version": "2.0", "temporal": ("2024-06-01", "2024-06-30")},
    {"short_name": "MOD09GQ", "version": "006", "temporal": ("2024-06-01", "2024-06-30")},
    {"short_name": "LANDSAT_OT_C2_L2", "version": "02", "temporal": ("2024-06-01", "2024-06-30")},
]

for params in test_cases:
    try:
        results = earthaccess.search_data(count=5, **params)
        print(f"  earthaccess {params['short_name']}: {len(results)} 景")
        if results:
            print(f"    第一个: {results[0].data['Granule']['UR'][:80]}")
    except Exception as e:
        print(f"  earthaccess {params['short_name']}: ERROR {type(e).__name__}: {e}")

# ============ Step 4: 最终检索 - 找到数据的区域 ============
print("\n" + "=" * 60)
print("Step 4: 最终检索 - 北京区域...")

# 北京区域
bbox_beijing = (115.7, 39.4, 117.4, 41.0)

all_results = []

# 尝试 HLSL30
print("\n  检索 HLSL30...")
try:
    results = earthaccess.search_data(
        short_name="HLSL30",
        version="2.0",
        temporal=("2024-06-01", "2024-06-30"),
        bounding_box=bbox_beijing,
        count=10,
    )
    print(f"  HLSL30 结果: {len(results)} 景")
    all_results.extend(results)
except Exception as e:
    print(f"  HLSL30 出错: {e}")

# 尝试原始 Landsat C2 L2
print("\n  检索 LANDSAT_OT_C2_L2...")
try:
    results = earthaccess.search_data(
        short_name="LANDSAT_OT_C2_L2",
        version="02",
        temporal=("2024-01-01", "2024-12-31"),
        bounding_box=bbox_beijing,
        count=10,
    )
    print(f"  LANDSAT_OT_C2_L2 结果: {len(results)} 景")
    all_results.extend(results)
except Exception as e:
    print(f"  LANDSAT_OT_C2_L2 出错: {e}")

# 尝试 LOS_ANGELES 区域（美国区域更容易找到）
print("\n  检索 Los Angeles 区域（美国更容易找到）...")
bbox_la = (-118.6, 33.7, -117.7, 34.3)
try:
    results_la = earthaccess.search_data(
        short_name="HLSL30",
        version="2.0",
        temporal=("2024-06-01", "2024-06-30"),
        bounding_box=bbox_la,
        count=10,
    )
    print(f"  LA HLSL30 结果: {len(results_la)} 景")
    all_results.extend(results_la)
except Exception as e:
    print(f"  LA HLSL30 出错: {e}")

# ============ Step 5: 下载 ============
print("\n" + "=" * 60)
print(f"Step 5: 下载（共 {len(all_results)} 景）...")

if all_results:
    print(f"  准备下载 {len(all_results)} 个文件...")
    for i, r in enumerate(all_results):
        try:
            ur = r.data.get("Granule", {}).get("UR", "unknown")
            print(f"  [{i+1}] {ur[:80]}")
        except:
            print(f"  [{i+1}] (无法获取 UR)")

    try:
        downloaded = earthaccess.download(all_results, OUT_DIR, threads=4)
        success = sum(1 for f in downloaded if f is not None)
        print(f"  成功下载: {success}/{len(all_results)}")
    except Exception as e:
        print(f"  批量下载出错: {e}")
        print("  尝试逐个下载...")
        for i, r in enumerate(all_results):
            try:
                earthaccess.download([r], OUT_DIR)
                print(f"  [{i+1}/{len(all_results)}] OK")
            except Exception as ex:
                print(f"  [{i+1}/{len(all_results)}] FAIL: {ex}")
            time.sleep(2)
else:
    print("  没有找到任何影像！")

print("\n" + "=" * 60)
print("完成！")
