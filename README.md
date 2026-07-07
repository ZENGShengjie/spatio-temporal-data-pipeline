# Spatio-Temporal Urban Computing Pipeline

> **多源时空数据融合 · 城市计算异常检测与预测系统**
>
> 配套论文：`多源时空数据集说明文档.pdf`（见 `week1/docs/原始资料/`）

---

## 🗂 仓库结构（按周组织）

```
spatio-temporal-data-pipeline/
├── README.md                  ← 你正在看
├── .gitignore
├── .env.example
│
├── week1/   ── Week 1：多源数据采集（已完成）
│   ├── README.md
│   ├── scripts/   (7 个下载脚本)
│   └── docs/
│       └── 原始资料/  (3 份 .docx，归档本地，git 忽略)
│
└── week2/   ── Week 2：数据清洗 + 网格 + 特征工程 + 异构图（已完成）
    ├── README.md            (操作手册 + 性能数据)
    ├── DELIVERY.md          (Leader 优先看：交付物索引)
    ├── deploy.py            (一键部署)
    ├── scripts/             (21 个 .py + EARTHDATA_RUNBOOK.md)
    └── docs/                (4 份设计文档：质量 / 架构 / 接口 / 测试)
```

> **Week 3+** 待添加：直接新建 `week3/` 子目录即可。

---

## 🚦 Leader 验收入口

| 优先级 | 路径 | 看点 |
|--------|------|------|
| ⭐⭐⭐ | [`week2/DELIVERY.md`](week2/DELIVERY.md) | Week 2 当天交付物的清单 + 验收状态 |
| ⭐⭐⭐ | [`week2/docs/数据质量分析报告.md`](week2/docs/数据质量分析报告.md) | 数据质量分析报告（原始交付要求 4） |
| ⭐⭐ | [`week2/README.md`](week2/README.md) | Week 2 完整流程与性能数据 |
| ⭐⭐ | [`week2/docs/代码架构设计报告.md`](week2/docs/代码架构设计报告.md) | 代码架构设计报告（追加交付 5） |
| ⭐⭐ | [`week2/docs/接口定义文档.md`](week2/docs/接口定义文档.md) | 接口定义文档（追加交付 6） |
| ⭐ | [`week2/docs/测试文档.md`](week2/docs/测试文档.md) | 测试文档（追加交付 7） |
| ⭐ | [`week1/README.md`](week1/README.md) | Week 1 数据采集说明 |

---

## 📅 项目进度

- ✅ **Week 1**：多源数据下载与准备（NASA Landsat、OpenWeather、GeoNames、NYC Taxi、OSM）
- ✅ **Week 2**：数据清洗（41M→19.85M Taxi）→ 500m 网格（15,875）→ 36 维时空特征（2,667,000 行）→ 异构图（230,732 边）
- ⏳ **Week 3+**：模型训练与异常检测

---

## 🛠️ 运行环境

- Python 3.10+
- 推荐 Linux / WSL / EC2 Ubuntu 22.04
- 大文件（`.tif`、`.parquet`、`.zip`、`.docx`、凭证 `.env`）已加入 `.gitignore`

## 📄 License

仅用于学习与研究用途。