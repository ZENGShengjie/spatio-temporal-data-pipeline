# Week 1 ── 多源时空数据采集

> 数据采集脚本与配套文档。Week 2 在此基础上做清洗与特征化。

## 📂 本周目录

```
week1/
├── README.md            ← 你正在看
├── scripts/             (7 个下载脚本)
│   ├── download_final.py
│   ├── download_geonames.py
│   ├── download_landsat.py
│   ├── download_nyc.py
│   ├── download_weather.py
│   ├── nasa_download.py
│   └── test_nasa.py
└── docs/
    └── 原始资料/        (3 份 .docx，归档本地，git 忽略)
```

## 🛰️ 数据源

| 数据源 | 用途 | 脚本 |
|--------|------|------|
| NASA Earthdata / HLSL30 | Landsat 卫星影像 | `scripts/nasa_download.py`、`scripts/download_final.py` |
| NYC Taxi TLC | 出租车轨迹 | `scripts/download_nyc.py` |
| OpenWeather | 气象预报 | `scripts/download_weather.py` |
| GeoNames / OSM | POI 兴趣点 | `scripts/download_geonames.py` |

## 🚀 快速开始

```bash
# 1) 安装依赖
pip install requests pandas earthaccess geopandas

# 2) 复制 .env 模板
cp .env.example .env
# 填入 NASA_USERNAME / NASA_PASSWORD

# 3) 运行下载
python scripts/download_final.py     # 卫星影像
python scripts/download_nyc.py       # NYC Taxi
python scripts/download_weather.py   # 天气
python scripts/download_geonames.py  # POI
```

## ⚠️ 注意事项

1. **不要**把 `.env` 或真实凭证提交到 Git
2. 数据文件（`.tif`、`.zip`、`.parquet`）已加入 `.gitignore`
3. `.docx` 原始资料归档在 `docs/原始资料/`（本地参考，不入库）
4. 详细部署步骤见 `docs/原始资料/AWS EC2城市交通时空预测项目开发环境配置指南.docx`

## 📦 数据产物 → Week 2 输入

Week 1 下载完成后，Week 2 脚本（见 `../week2/scripts/`）会读取以下目录：

| 路径 | 内容 |
|------|------|
| `raw_nyc/` | NYC Taxi 原始 CSV（按月） |
| `raw_weather/` | OpenWeather 原始 JSON |
| `landsat_raw/` | Landsat HLSL30 .tif |
| `osm_extracts/` | OSM .pbf / .xml |

详细处理流程见 [`../week2/README.md`](../week2/README.md)。