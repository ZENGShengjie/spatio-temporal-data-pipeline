# ⭐ Week 2 交付物导读（Leader 优先读这里）

> **用途：** 本文件是 Week 2 当天交付物的索引，让你不必翻全仓库就能定位
> 所需的 4 项原始交付物 + 3 项后续追加交付物。

---

## 📦 leader 原始要求交付的 4 项

| # | 交付内容 | 路径 | 状态 |
|---|---------|------|------|
| 1 | **数据预处理 Python 脚本** | `week2_scripts/`（8 个 .py） | ✅ |
| 2 | **特征工程代码库** | `week2_scripts/step3_features.py`（主） + `week2_scripts/step3b_ndvi.py`（卫星） + `week2_scripts/extract_osm_v2.py`（OSM） | ✅ |
| 3 | **预处理后的数据集（约 20 GB）** | EC2 `/home/ubuntu/amazon/{cleaned_nyc,grid_nyc,features_nyc,graph_nyc}/` 共 ~9 GB（含 Landsat 5GB；纯净后 ~1.8 GB） | ✅（see `WEEK2_README.md`） |
| 4 | **数据质量分析报告** | `WEEK2_docs/数据质量分析报告.md` | ✅ |

## 📑 leader 后续追加的 3 项

| # | 交付内容 | 路径 |
|---|---------|------|
| 5 | **代码架构设计报告** | `WEEK2_docs/代码架构设计报告.md` |
| 6 | **接口定义文档**         | `WEEK2_docs/接口定义文档.md` |
| 7 | **测试文档**             | `WEEK2_docs/测试文档.md` |

---

## 🗂 仓库布局速览

```
week 1/                                       # = spatio-temporal-data-pipeline 仓库根
├── .gitignore            # 已扩展忽略大文件输出
├── README.md             # Week 1 原始 README
├── .env.example
├── AWS EC2城市交通时间预测项目搭建操作指南.docx
├── 多源时空数据集说明文档.docx
├── 城市信息时空行业研究前沿关键技术报告.docx
├── download_*.py         # Week 1 下载脚本（原始）
├── nasa_download.py
├── test_nyc.py
│
├── ⭐ WEEK2_DELIVERY.md  # ← 本文件 (Leader 优先读)
├── WEEK2_README.md       # = 原 e:\amazon\week 2\README.md
├── week2_deploy.py       # = 原 e:\amazon\week 2\deploy.py
├── week2_scripts/        # = 原 e:\amazon\week 2\scripts\
│   ├── step1_clean.py
│   ├── step1_clean_new.py
│   ├── step2_grid.py
│   ├── step3b_ndvi.py
│   ├── step3_features.py
│   ├── step4_graph.py
│   ├── extract_osm_v2.py
│   └── _*.py / check_*.py
│
└── WEEK2_docs/           # = 原 e:\amazon\week 2\docs\
    ├── 数据质量分析报告.md
    ├── 代码架构设计报告.md
    ├── 接口定义文档.md
    └── 测试文档.md
```

> **平铺原因：** 直接沿用现有 `spatio-temporal-data-pipeline` 仓库结构，
> 在 week1 根目录加 `WEEK2_*` 前缀避免命名冲突；Week 3 可按需整合成子目录。

---

## 🚀 Week 2 一键复现（已跑通）

```bash
# 环境变量
export NASA_USERNAME=xxx
export NASA_PASSWORD=xxx

# Step 1: 数据清洗
python step1_clean.py        # baseline
python step1_clean_new.py    # 增强版 + 时序插值

# Step 2: 网格划分
python step2_grid.py

# Step 3: 特征工程
python step3_features.py     # 主特征
python step3b_ndvi.py        # NDVI 卫星

# Step 4: 异构图
python step4_graph.py

# 一键部署
python week2_deploy.py
```

完整流程、性能数据、已知问题见 `WEEK2_README.md` 和 `WEEK2_docs/`。

---

## ✅ 验收清单（本周已通过）

| 项目 | 状态 |
|------|------|
| TaxiNYC 清洗（41M → 19.85M 行，48% 保留） | ✅ |
| 气象清洗（9,072 → 8,808 行，97.1%） | ✅ |
| 500m 网格 15,875 个 | ✅ |
| 特征矩阵 2,667,000 × 36 列 | ✅ |
| 异构图 15,875 节点 / 230,732 边 | ✅ |
| 端到端 `week2_deploy.py` < 5 min 在 EC2 t3.large 通过 | ✅ |

数据质量细节详见 `WEEK2_docs/数据质量分析报告.md`。
