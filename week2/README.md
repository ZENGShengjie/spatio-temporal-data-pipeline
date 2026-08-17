# Week 2: 数据预处理与特征工程

> **开发过程临时脚本**：`week2/scripts/` 内 `_*` 前缀脚本多为开发过程临时调试（POI 重建、插值修复等），**不在主清洗流程内**。答辩使用主入口 `week2/scripts/` 下 21 个处理脚本即可，临时脚本仅作过程证据保留。

> 基于多源时空数据融合的城市级人流异常检测与预警系统 — 第二周

| 字段 | 值 |
|------|----|
| 目标城市 | New York City (NYC) |
| 城市边界 | lon ∈ [-74.26, -73.70], lat ∈ [40.49, 40.92] |
| 网格 | 500m × 500m, Mercator → WGS84, 共 15,875 个 |
| 数据范围 | 2024-06-02 ~ 2024-10-31 (1 周代表序列) |
| 主要 EC2 | t3.large (2 vCPU / 8 GiB) |
| 一键部署 | `python deploy.py` |
| 📊 数据质量 | [`docs/数据质量分析报告.md`](docs/数据质量分析报告.md) |
| 🏗 代码架构 | [`docs/代码架构设计报告.md`](docs/代码架构设计报告.md) |
| 📑 接口定义 | [`docs/接口定义文档.md`](docs/接口定义文档.md) |
| 🧪 测试文档 | [`docs/测试文档.md`](docs/测试文档.md) |

---

## 目录结构

```
week 2/
├── scripts/                       # 所有 step 脚本
│   ├── step1_clean.py             # baseline 多源数据清洗
│   ├── step1_clean_new.py         # 增强版 (含时序插值)
│   ├── step2_grid.py              # NYC 500m×500m 网格
│   ├── extract_osm_v2.py          # OSM 抽取 (poi/roads)
│   ├── step3b_ndvi.py             # NDVI 卫星特征
│   ├── step3_features.py          # 时空特征工程
│   ├── step4_graph.py             # 异构图构建
│   └── _*.py                      # 调试/检查小工具
├── docs/                          # ⭐ 本周交付文档 (新增)
│   ├── 数据质量分析报告.md
│   ├── 代码架构设计报告.md
│   ├── 接口定义文档.md
│   └── 测试文档.md
├── outputs/                       # (本地临时)
├── data/                          # (本地临时)
├── deploy.py                      # 一键部署到 EC2 t3.large
└── README.md                      # 本文件
```

EC2 运行时目录结构详见 [`docs/代码架构设计报告.md`](docs/代码架构设计报告.md) §4。

---

## 本周交付物一览 (对应 leader 要求)

| # | 交付物 | 路径 |
|---|--------|------|
| 1 | **数据预处理 Python 脚本** | `scripts/step1_*.py`, `step2_*.py`, `step3_*.py`, `step4_*.py`, `extract_osm_v2.py` |
| 2 | **特征工程代码** | `scripts/step3_features.py` (主) + `step3b_ndvi.py` (卫星) + `extract_osm_v2.py` (OSM) |
| 3 | **预处理后的数据集 (≈20 GB)** | EC2 端：`/home/ubuntu/amazon/{cleaned_nyc, grid_nyc, features_nyc, graph_nyc}/`，已落盘 ~1.07 GB 核心 + Landsat 原始 (~5 GB) |
| 4 | **数据质量分析报告** | [`docs/数据质量分析报告.md`](docs/数据质量分析报告.md) |

leader 后续要求追加：

| 交付物 | 路径 |
|--------|------|
| **代码架构设计报告** | [`docs/代码架构设计报告.md`](docs/代码架构设计报告.md) |
| **接口定义文档**     | [`docs/接口定义文档.md`](docs/接口定义文档.md) |
| **测试文档**         | [`docs/测试文档.md`](docs/测试文档.md) |

---

## 环境要求

- Python 3.10+
- 核心库: `numpy`, `pandas`, `geopandas`, `pyarrow`, `scikit-learn`, `scipy`
- 图神经: `torch`, `torch-geometric`, `torch-scatter`, `torch-sparse`
- 地理: `shapely`, `pyproj`, `rasterio`
- OSM 工具: `osmium-tool`
- 部署: `boto3`, `awscli`

安装：

```bash
pip install -r requirements.txt  # 见 Week 2 仓库根目录
sudo apt-get install -y osmium-tool
```

---

## 执行顺序

```bash
# 0. 设置凭证
export NASA_USERNAME=tony060514
export NASA_PASSWORD='SCzsj060514#'

# 1. 数据清洗
python scripts/step1_clean.py        # baseline
python scripts/step1_clean_new.py    # 增强 (含 taxi 时序插值)

# 2. OSM 抽取
python scripts/extract_osm_v2.py     # poi + roads → processed/

# 3. NDVI 卫星特征
python scripts/step3b_ndvi.py        # MOD13A1 → cleaned_nyc/ndvi_clean.parquet

# 4. 网格划分
python scripts/step2_grid.py         # → grid_nyc/

# 5. 特征工程
python scripts/step3_features.py     # → features_nyc/nyc_features.parquet (2,667,000 × 36)

# 6. 异构图构建
python scripts/step4_graph.py        # → graph_nyc/nyc_hetero_graph.pt

# 或一键部署
python deploy.py
```

---

## 数据管道 (端到端)

```
raw_nyc/                          (Week 1 下载)
    ├── taxi_nyc/                 yellow_tripdata_2024-MM.parquet × 12 (~3 GB)
    ├── weather/                  nyc_weather_hourly.json (~5 MB)
    ├── osm/                      nyc.osm.pbf (~700 MB)
    ├── landsat/                  MOD13A1.*.hdf (~5 GB)
    └── poi/                      nyc_poi.json
            ↓ (step1_clean / step1_clean_new)
cleaned_nyc/                      (~1.1 GB)
    ├── taxi_nyc_clean.parquet               19.85M 行
    ├── taxi_interpolated_buckets.parquet     44,184 行 (含 is_interpolated 标记)
    ├── weather_clean.parquet                  8,808 行
    ├── poi_clean.parquet                      1,673 行
    ├── road_density_clean.parquet            15,875 行
    ├── ndvi_clean.parquet                    15,875 行
    └── cleaning_log.json
            ↓ (step2_grid)
grid_nyc/
    ├── nyc_grid_500m.geojson                 15,875 grid
    ├── nyc_grid_500m.parquet
    └── nyc_grid_id_mapping.csv
            ↓ (step3_features)
features_nyc/
    ├── nyc_features.parquet                  2,667,000 × 36 (~685 MB)
    └── step3_log.json
            ↓ (step4_graph)
graph_nyc/
    ├── nyc_hetero_graph.pt                   15,875 nodes (PyG HeteroData, ~4.3 MB)
    └── graph_metadata.json
```

### 当前已落盘大小

| 层 | 大小 |
|----|------|
| raw_nyc（不含 Landsat） | ~3.8 GB |
| cleaned_nyc | ~1.1 GB |
| grid_nyc | ~14 MB |
| features_nyc | ~685 MB |
| graph_nyc | ~4.3 MB |
| **总计核心** | **~1.8 GB** |

加上 `raw_nyc/landsat/*.hdf` (~5 GB) 后约 ~9 GB；含 S3 全量冗余后约 ~20 GB。

---

## 特征说明 (36 列)

| 组 | 列 | 说明 |
|----|----|-----|
| 主键 | `grid_id`, `datetime`, `weekday`, `hour`, `month` | 网格 × 时空 |
| 时间 | `hour_sin/cos`, `weekday_sin/cos`, `month_sin/cos`, `is_weekend`, `is_holiday` | 周期编码 |
| 空间 | `dist_to_times_square`, `dist_to_central_park`, `dist_to_wall_street`, `dist_to_grand_central`, `dist_to_jfk_airport`, `dist_to_city_center`, `dist_to_nearest_landmark` | Haversine 距离 (km) |
| POI | `poi_food_count`, `poi_shopping_count`, `poi_entertainment_count`, `poi_office_count`, `poi_residential_count`, `poi_total_count`, `poi_density_per_km2` | 当前只有 residential 有值 |
| 路网 | `road_length_m`, `road_segment_count`, `road_density_km_per_km2`, `road_len_tier_highway_m`, `road_len_tier_major_m`, `road_len_tier_minor_m`, `road_len_tier_local_m` | OSM |
| 轨迹 | `taxi_pickup_count`, `taxi_dropoff_count` | 时序桶 |
| 气象 | `weather_temp_norm`, `weather_humidity_norm`, `weather_pressure_norm`, `weather_wind_speed_norm`, `weather_clouds_norm` | 归一化 |
| 卫星 | `ndvi_mean`, `ndvi_max`, `ndvi_min`, `ndvi_count`, `is_water` | MODIS MOD13A1 |

完整 schema 见 [`docs/接口定义文档.md`](docs/接口定义文档.md)。

---

## 异构图说明

```python
HeteroData(
  spatial : NodeData(x = [15875, 10]),
  (spatial, adjacent,   spatial) : 127,000 edges,
  (spatial, similar,    spatial) : 95,250 edges,
  (spatial, correlated, spatial) : 8,482 edges,
)
```

- `spatial` 边：haversine KNN=8
- `semantic` 边：归一化特征上的 euclidean KNN=6
- `correlated` 边：taxi 流量时序 Pearson ≥ 0.7

---

## EC2 规格

| 步骤 | 实例 | 备注 |
|------|------|------|
| step1 (清洗) | t3.large | 已通过 |
| step2 (网格) | t3.large | 已通过 |
| step3 (特征) | t3.large | 已通过 (~3 GB peak) |
| step4 (图)   | t3.large | 已通过 (~4 GB peak) |

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [`docs/数据质量分析报告.md`](docs/数据质量分析报告.md) | 数据完整性、schema、保留率、分布、已知问题 |
| [`docs/代码架构设计报告.md`](docs/代码架构设计报告.md) | 分层架构、模块依赖、性能扩展路径 |
| [`docs/接口定义文档.md`](docs/接口定义文档.md) | 每一步输入/输出 schema、函数签名、环境变量 |
| [`docs/测试文档.md`](docs/测试文档.md) | 单元/合约/集成/性能/数据健康测试方案 |

---

## 项目进度

- ✅ **Week 1**: 多源数据下载与准备
- ✅ **Week 2**: 多源数据清洗、500m 网格划分、特征工程、异构图（当前）
- ⏳ **Week 3**: GNN 训练 (HGT / HAN)

---

## License

仅用于学习与研究用途。