"""
NYC 数据下载脚本 — Week 1（纽约版）
目标城市：New York City
覆盖范围：经度 -74.26~-73.70, 纬度 40.49~40.92
运行方式（EC2 t3.large）：
    export NASA_USERNAME=tony060514
    export NASA_PASSWORD=SCzsj060514#
    python download_nyc.py
"""
import os
import sys
import subprocess
import json
import time
import requests
import earthaccess
from datetime import datetime

# ========== 输出目录 ==========
OUT_DIR = os.getenv("NYC_OUT_DIR", "/home/ubuntu/amazon/raw_nyc")
os.makedirs(OUT_DIR, exist_ok=True)

NYC_BBOX = (-74.26, 40.49, -73.70, 40.92)  # NYC 边界框

# ========== 凭证检查 ==========
NASA_USER = os.getenv("NASA_USERNAME")
NASA_PASS = os.getenv("NASA_PASSWORD")
if not NASA_USER or not NASA_PASS:
    sys.exit("ERROR: 设置 NASA_USERNAME 和 NASA_PASSWORD 环境变量")

os.environ["EARTHDATA_USERNAME"] = NASA_USER
os.environ["EARTHDATA_PASSWORD"] = NASA_PASS


# ============================================================
# 步骤 1: 下载 Landsat 影像 (HLSL30 - 每日重访，更适合城市)
# ============================================================
def download_landsat():
    print("=" * 60)
    print("[Step 1] 下载 Landsat 影像 (HLSL30) ...")
    print(f"  区域: NYC {NYC_BBOX}")

    auth = earthaccess.login(strategy="environment", persist=True)
    if not auth.authenticated:
        sys.exit("ERROR: NASA Earthdata 登录失败")

    out = os.path.join(OUT_DIR, "landsat")
    os.makedirs(out, exist_ok=True)

    # 下载 2024 年 6-8 月（夏季度高峰期，纽约人流最多）
    dates = [
        ("2024-06-01", "2024-06-30"),
        ("2024-07-01", "2024-07-31"),
        ("2024-08-01", "2024-08-31"),
    ]

    for start, end in dates:
        print(f"\n  检索 {start} ~ {end} ...")
        try:
            results = earthaccess.search_data(
                short_name="HLSL30",
                version="2.0",
                temporal=(start, end),
                bounding_box=NYC_BBOX,
                count=50,
            )
            print(f"    找到 {len(results)} 景")
            if results:
                downloaded = earthaccess.download(results, out, threads=4)
                success = sum(1 for f in downloaded if f is not None)
                print(f"    成功下载: {success}/{len(results)}")
        except Exception as e:
            print(f"    错误: {e}")
        time.sleep(3)

    print("  Landsat 下载完成!")


# ============================================================
# 步骤 2: 下载 OpenWeather 历史气象数据 (NYC)
# ============================================================
def download_weather():
    print("\n" + "=" * 60)
    print("[Step 2] 下载 NYC 气象数据 ...")

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        print("  WARNING: 未设置 OPENWEATHER_API_KEY，跳过气象数据")
        return

    out = os.path.join(OUT_DIR, "weather")
    os.makedirs(out, exist_ok=True)

    # 纽约市中心
    lat, lon = 40.7128, -74.0060
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall?"
        f"lat={lat}&lon={lon}&appid={api_key}&units=metric"
        f"&exclude=minutely,alerts&dt=1735689600"  # 2025-01-01
    )

    # OpenWeather 免费版不支持历史回溯，用预报数据代替
    # 历史数据需要付费订阅，这里用 5 天/小时预报 API 作为代理
    forecast_url = (
        f"https://api.openweathermap.org/data/2.5/forecast?"
        f"q=NewYork,US&appid={api_key}&units=metric"
    )

    try:
        resp = requests.get(forecast_url, timeout=30)
        data = resp.json()
        out_path = os.path.join(out, "nyc_weather_forecast.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {out_path} ({len(data.get('list', []))} 条)")
    except Exception as e:
        print(f"  气象数据下载失败: {e}")

    print("  气象数据完成!")


# ============================================================
# 步骤 3: 下载 GeoNames POI (NYC)
# ============================================================
def download_poi():
    print("\n" + "=" * 60)
    print("[Step 3] 下载 NYC POI 数据 ...")

    username = os.getenv("GEONAMES_USERNAME")
    if not username:
        print("  WARNING: 未设置 GEONAMES_USERNAME，跳过 POI")
        return

    out = os.path.join(OUT_DIR, "poi")
    os.makedirs(out, exist_ok=True)

    # NYC 五个行政区各取一个中心点
    boroughs = [
        {"name": "Manhattan", "lat": 40.7831, "lon": -73.9712},
        {"name": "Brooklyn",  "lat": 40.6782, "lon": -73.9442},
        {"name": "Queens",    "lat": 40.7282, "lon": -73.7949},
        {"name": "Bronx",     "lat": 40.8448, "lon": -73.8648},
        {"name": "Staten",    "lat": 40.5795, "lon": -74.1502},
    ]

    all_pois = []

    for b in boroughs:
        try:
            url = (
                f"http://api.geonames.org/findNearbyPOIsOSMJSON"
                f"?lat={b['lat']}&lng={b['lon']}&username={username}"
            )
            resp = requests.get(url, timeout=30)
            data = resp.json()
            pois = data.get("pois", {}).get("poi", [])
            all_pois.extend(pois)
            print(f"  {b['name']}: {len(pois)} POI")
            time.sleep(1)
        except Exception as e:
            print(f"  {b['name']} 错误: {e}")

    out_path = os.path.join(out, "nyc_poi.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"boroughs": boroughs, "pois": all_pois}, f, ensure_ascii=False, indent=2)
    print(f"  总 POI: {len(all_pois)}，已保存: {out_path}")


# ============================================================
# 步骤 4: 下载 OSM 道路数据 (NYC)
# ============================================================
def download_osm():
    print("\n" + "=" * 60)
    print("[Step 4] 下载 OSM NYC 道路数据 ...")

    out = os.path.join(OUT_DIR, "osm")
    os.makedirs(out, exist_ok=True)

    # NYC 从 Geofabrik 下载
    url = (
        "https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf"
    )
    out_path = os.path.join(out, "nyc.osm.pbf")

    try:
        print(f"  下载: {url}")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r  下载进度: {pct}%", end="", flush=True)
        print(f"\n  已保存: {out_path}")
    except Exception as e:
        print(f"  OSM 下载失败: {e}")


# ============================================================
# 步骤 5: 下载 TaxiNYC 轨迹数据
# ============================================================
def download_taxi_nyc():
    print("\n" + "=" * 60)
    print("[Step 5] 下载 TaxiNYC 轨迹数据 ...")

    out = os.path.join(OUT_DIR, "taxi_nyc")
    os.makedirs(out, exist_ok=True)

    # TaxiNYC TLC 官方下载地址（2024年数据）
    base_url = "https://d37ci6mwuryvrs.cloudfront.net/trip-data"
    months = [
        "2024-01", "2024-02", "2024-03",
        "2024-04", "2024-05", "2024-06",
    ]

    for m in months:
        fname = f"yellow_tripdata_{m}.parquet"
        url = f"{base_url}/{fname}"
        out_path = os.path.join(out, fname)
        if os.path.exists(out_path):
            size = os.path.getsize(out_path) / 1024 / 1024
            print(f"  [跳过] {fname} ({size:.1f} MB 已存在)")
            continue

        print(f"  下载 {fname} ...", end=" ", flush=True)
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(f"\r  下载 {fname}: {pct}%", end="", flush=True)
            size = os.path.getsize(out_path) / 1024 / 1024
            print(f"  ✅ {size:.1f} MB")
        except Exception as e:
            print(f"  ❌ {e}")
        time.sleep(2)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print(f"NYC 多源数据下载 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"输出目录: {OUT_DIR}")
    print(f"区域: NYC {NYC_BBOX}")
    print("=" * 60)

    # 1. Landsat 卫星影像
    download_landsat()

    # 2. 气象数据
    download_weather()

    # 3. POI
    download_poi()

    # 4. OSM 道路
    download_osm()

    # 5. TaxiNYC 轨迹
    download_taxi_nyc()

    print("\n" + "=" * 60)
    print("所有数据下载完成!")

    # 汇总
    total_size = 0
    for root, dirs, files in os.walk(OUT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)

    print(f"总大小: {total_size / 1024 / 1024:.1f} MB")
    print(f"目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
