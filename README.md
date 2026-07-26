# Spatio-Temporal Urban Computing Pipeline

> **多源时空数据融合 · 城市计算异常检测与预测系统**
>
> 配套论文：`多源时空数据集说明文档.pdf`（见 `week1/docs/原始资料/`）

---

## 仓库结构（按周组织）

```
spatio-temporal-data-pipeline/
├── README.md                          ← 你正在看
├── .gitignore
├── .env.example
│
├── week1/   ── Week 1：多源数据采集（已完成）
│   ├── README.md
│   ├── scripts/   (7 个下载脚本)
│   └── docs/
│
├── week2/   ── Week 2：数据清洗 + 网格 + 特征工程 + 异构图（已完成）
│   ├── README.md
│   ├── DELIVERY.md
│   ├── deploy.py
│   ├── scripts/   (21 个 .py)
│   └── docs/      (4 份设计文档)
│
├── week3/   ── Week 3：时序预测基线实验（已完成）
│   ├── README.md
│   ├── run_week3.py                  ← 主入口
│   ├── config.py / registry.py        ← 共享配置
│   ├── data_loader.py
│   ├── base_trainer.py / metrics.py
│   ├── models/                        ← 7 个模型
│   │   ├── arima_model.py / prophet_model.py
│   │   ├── lstm_model.py / gru_model.py
│   │   └── gcn_model.py / gat_model.py / gru_stgcn_residual.py
│   ├── report/                        ← 报告与日志
│   └── scripts/
│
├── week4/   ── Week 4：时空联合建模 + 消融实验（已完成）
│   ├── README.md
│   ├── run_week4.py                  ← 主入口（STF / AGFormer / STGCN + 消融）
│   ├── config.py / registry.py
│   ├── data_loader.py / metrics.py
│   ├── base_trainer.py
│   ├── models/                        ← 3 个时空模型
│   │   ├── stf_model.py             ← 时空解耦 Transformer（最优，222K 参数）
│   │   ├── agformer_model.py         ← 自适应图时空 Transformer（2.26M 参数）
│   │   └── stgcn_model.py           ← 时空图卷积网络（200K 参数）
│   ├── report/                        ← Week4 报告
│   ├── results/v4fix/                ← 完整训练日志 + 5 张可视化图
│   └── weights/                       ← 模型权重
│
├── docs/
│   └── 北京数据质量分析报告.md        ← 完整研究报告（P0 溯源 + Week3/4 实验）
│
└── [证据 / 数据文件 / 训练权重  均已加入 .gitignore]
```

---

## 核心结果总览

### Week3：时序基线（7 模型对比）

| 排名 | 模型 | MAE | MAPE | Corr | 训练时间 |
|------|------|-----|------|------|---------|
| 🥇 | **GRU** | **158.12** | **42.83** | **0.9452** | **<1 min** |
| 🥈 | LSTM | 162.35 | 44.65 | 0.9445 | 1.3 min |
| 🥉 | GRU+ST-GCN Res | 162.12 | — | 0.9450 | ~25 min |
| 4 | GCN | 332.71 | 125.97 | 0.7931 | 7.8 min |
| 5 | GAT | 334.21 | 122.15 | 0.7942 | 12.7 min |

> Prophet MAE=93.66（1-shot 口径不同），GRU 为多步滚动预测最优

### Week4：时空联合建模（滑动窗口多步专项）

| 排名 | 模型 | MAE | Corr | 参数量 | 训练耗时 |
|------|------|-----|------|--------|---------|
| 🥇 | **STF**（时空解耦 Transformer）| **327.19** | **0.8043** | 222K | 515s |
| 🥈 | AGFormer（自适应图 Transformer）| 386.91 | 0.6990 | 2,264K | 9,491s |
| 🥉 | STGCN（时空图卷积） | 429.18 | 0.6502 | 200K | 9,695s |

### 消融实验（量化组件贡献）

| 消融组 | 被剥离组件 | MAE 代价 | 贡献评级 |
|--------|-----------|---------|---------|
| STF → stf_loc_only | Env token + 跨节点注意力 | +2.2% | 中等 |
| AGFormer → agformer_static | **自适应邻接矩阵** | **+5.1%** | **关键** |

---

## 关键结论

1. **STF 是小样本最优**：222K 参数 + 1 epoch 收敛，时空解耦架构在有限数据下参数效率最高
2. **自适应邻接矩阵是最有价值的设计**（贡献 5.1% MAE），但需要更大数据量才能兑现
3. **Week3 局限性**：GRU 仅捕获时序，固定邻接 GCN 引入噪声；空间建模是突破瓶颈的关键
4. **GRU 城市级 Corr=0.9452**：时序建模天花板极高，级联架构（GRU 引导 + STF 空间分配）是值得追求的方向

---

## 快速复现

```bash
# 登录 EC2（实例 ID 可在 AWS 控制台查看；公网 IP 每次重启可能变）
ssh -i ~/.ssh/aws-spatio-key.pem ubuntu@<EC2_PUBLIC_IP>

# Week3 基线
cd ~/spatio-temporal-pipeline/week3
python3 run_week3.py --models gru --target taxi_flow_total --tag v2

# Week4 时空模型
cd ~/spatio-temporal-pipeline/week4
python3 run_week4.py --models stf agformer stgcn --target taxi_flow_total --tag v4fix

# Week5 异常检测
cd ~/spatio-temporal-pipeline/week5
python3 run_v3_full_eval.py  # 4 方法 + 多组融合 + 全量评估

# Week5 结构性异常消融
python3 _inject_structural.py                            # 注入 surge/drop/sustained
python3 _rerun_struct.py && python3 _rerun_tae_only.py   # 4 方法分数重算
python3 _run_ablation_structural.py                      # 消融评估入口
```

---

## 完整报告索引

| 优先级 | 路径 | 说明 |
|--------|------|------|
| ⭐⭐⭐ | [`docs/北京数据质量分析报告.md`](docs/北京数据质量分析报告.md) | 完整研究报告（P0 溯源 + Week3 基线 + Week4 实验 + 局限性分析 + 综合总结）|
| ⭐⭐⭐ | [`week4/report/WEEK4_REPORT.md`](week4/report/WEEK4_REPORT.md) | Week4 实验专项报告 |
| ⭐⭐⭐ | [`week5/docs/ablation_structural_REPORT.md`](week5/docs/ablation_structural_REPORT.md) | **Week5 结构性异常消融实验**（V3 vs surge/drop/sustained）|
| ⭐⭐⭐ | [`week5/README.md`](week5/README.md) | **Week5 异常检测模块**（4 方法 + 融合框架 + V3 闭环评估）|
| ⭐⭐⭐ | [`docs/演示视频脚本.md`](docs/演示视频脚本.md) | **演示脚本（API + Streamlit 完整版）** |
| ⭐⭐ | [`week3/report/WEEK3_REPORT.md`](week3/report/WEEK3_REPORT.md) | Week3 基线实验报告 |
| ⭐⭐ | [`week4/results/v4fix/WEEK4_FINAL_REPORT.md`](week4/results/v4fix/WEEK4_FINAL_REPORT.md) | Week4 完整训练日志 |

---

## 项目进度

- ✅ **Week 1**：多源数据下载（NASA Landsat、OpenWeather、GeoNames、NYC Taxi、OSM）
- ✅ **Week 2**：数据清洗 → 网格划分 → 时空特征工程 → 异构图构建
- ✅ **Week 3**：7 模型时序基线（ARIMA/Prophet/LSTM/GRU/GCN/GAT/GRU+GCN 残差）
- ✅ **Week 4**：时空联合建模（STF/AGFormer/STGCN）+ 消融实验 + 局限性分析
- ✅ **Week 5**：异常检测 4 范式（statistical / prediction / VAE / Transformer-AE）+ 多组融合 + V3 闭环评估
- ✅ **Week5 结构性消融**：V3 vs surge/drop/sustained 三种结构性异常对照实验（详见 `week5/docs/ablation_structural_REPORT.md`）
- ⏳ **Week 6+**：STF 规模化 / 自适应图预热 / 级联架构

---

## 运行环境

- Python 3.10+
- 推荐 Linux / WSL / EC2 Ubuntu 22.04（g4dn.xlarge 用于模型训练）
- 大文件（`.pth`、`.h5`、`.npz`、`.npy`、`.tif`、`.docx`、凭证 `.env`）已加入 `.gitignore`

---

## 访问演示界面（Week6）

部署完成后，在浏览器打开：

- **可视化界面**：`http://<EC2_PUBLIC_IP>:8501`（注意 `http://` 双斜杠，侧边栏显示「API 在线 (ok)」即正常）
- **API 文档 (Swagger)**：`http://<EC2_PUBLIC_IP>:8000/docs`

> ⚠️ **EC2 公网 IP 每次重启可能变化**。建议在 AWS 控制台分配 Elastic IP 固定，否则演示前需查询当前 IP：
> ```bash
> ssh ubuntu@<EC2_PUBLIC_IP> 'curl -s ifconfig.me'
> ```
> Streamlit 端 `week6/app.py` 第 38 行默认 `http://localhost:8000`，可通过环境变量覆盖：
> ```bash
> export API_BASE="http://<EC2_PUBLIC_IP>:8000"
> streamlit run week6/app.py --server.port 8501
> ```
> 详细演示流程见 [`docs/演示视频脚本.md`](docs/演示视频脚本.md)（含 5 个 Tab 完整操作 + curl 命令 + curl 失败/弹窗拦截等应急方案）。

---

## License

仅用于学习与研究用途。
