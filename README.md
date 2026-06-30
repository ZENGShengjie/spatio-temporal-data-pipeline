# Spatio-Temporal Urban Computing Project

> 基于多源时空数据融合的城市计算异常检测与预测系统 —— 第一周：数据采集

## 📋 项目背景

本项目融合 **多源时空数据**（卫星影像、气象、POI、出租车轨迹等），构建城市计算异常检测与预测系统。本仓库对应 **Week 1：多源数据下载** 部分。

> 配套论文参见根目录 `机器学习算法在城市计算异常检测与预测系统(1).pdf`

## 📂 目录结构

```
week 1/
├── nasa_download.py                    # NASA Earthdata Landsat 影像检索与下载
├── download_landsat.py                 # Landsat 专用下载脚本
├── download_final.py                   # 最终整合脚本（HLSL30 / earthaccess）
├── download_weather.py                 # OpenWeather 气象预报数据下载
├── download_geonames.py                # GeoNames POI 兴趣点下载
│
├── AWS EC2城市交通时间预测项目搭建操作指南.docx          # 部署指南
├── 多源时空数据集说明文档.docx                          # 数据集说明
├── 城市信息时空行业研究前沿关键技术报告.docx            # 行业前沿报告
│
├── .env.example                        # 环境变量模板
└── README.md
```

## 🛰️ 数据源

| 数据源 | 用途 | 脚本 |
|---|---|---|
| NASA Earthdata / HLSL30 | Landsat 卫星影像 | `nasa_download.py`、`download_final.py` |
| OpenWeather | 气象预报数据 | `download_weather.py` |
| GeoNames / OSM | POI 兴趣点 | `download_geonames.py` |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install requests pandas earthaccess geopandas
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入真实凭证：

```bash
cp .env.example .env
```

或者直接在脚本里修改硬编码值（不推荐，勿提交到 Git）。

### 3. 运行下载

```bash
python download_final.py        # 下载卫星影像（推荐）
python download_weather.py     # 下载天气数据
python download_geonames.py    # 下载 POI 数据
```

## 📅 项目进度

- ✅ **Week 1**: 多源数据下载与准备（当前）
- ⏳ **Week 2**: 多源数据清洗与标准化、500m 网格划分、时空特征工程
- ⏳ **Week 3**: 多模态图结构构建（异构图）

## 🛠️ 运行环境

- Python 3.10+
- 推荐 Linux / WSL
- 远程部署：AWS EC2 (Ubuntu)

## ⚠️ 注意事项

1. **不要**把 `.env` 或含有真实凭证的脚本提交到 GitHub
2. 数据文件（`.tif` / `.zip`）已加入 `.gitignore`，避免仓库膨胀
3. 详细部署步骤请参阅 `AWS EC2城市交通时间预测项目搭建操作指南.docx`

## 📄 License

仅用于学习与研究用途。