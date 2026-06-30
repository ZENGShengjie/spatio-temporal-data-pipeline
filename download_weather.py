import requests
import json
import os

# ========== API Key 从环境变量读取 ==========
# 请设置环境变量 OPENWEATHER_API_KEY
API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("请先设置环境变量 OPENWEATHER_API_KEY")
city_list = ["Beijing", "Los Angeles"]

# 只保存原始返回数据，不清洗、不转表格
for city in city_list:
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    with open(f"{city}_raw_weather.json", "w", encoding="utf-8") as f:
        json.dump(res.json(), f, ensure_ascii=False, indent=2)