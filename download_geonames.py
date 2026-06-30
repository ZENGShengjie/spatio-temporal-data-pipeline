import requests
import json
import os

# ========== GeoNames 用户名从环境变量读取 ==========
# 请设置环境变量 GEONAMES_USERNAME
username = os.getenv("GEONAMES_USERNAME")
if not username:
    raise ValueError("请先设置环境变量 GEONAMES_USERNAME")
# 北京、洛杉矶边界
cities = [
    {"name":"Beijing","lat":39.90,"lon":116.40},
    {"name":"Los Angeles","lat":34.05,"lon":-118.24}
]

for item in cities:
    url = f"http://api.geonames.org/findNearbyPOIsOSMJSON?lat={item['lat']}&lng={item['lon']}&username={username}"
    res = requests.get(url)
    with open(f"{item['name']}_poi_raw.json", "w", encoding="utf-8") as f:
        json.dump(res.json(), f, ensure_ascii=False, indent=2)
