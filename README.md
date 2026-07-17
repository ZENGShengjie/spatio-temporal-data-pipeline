# 北京出租车流量时空预测系统

> 基于北京出租车 GPS 轨迹数据，构建高精度时空流量预测系统。Week3 建立时序基线，Week4 引入先进时空图神经网络，完成消融实验。

## 📂 目录结构

```
e:\amazon\
├── docs/
│   └── 北京数据质量分析报告.md        ← 完整研究报告（P0 溯源校验 + Week3/4 实验）
├── week3_deliverable/                ← Week3 交付包（时序基线实验）
│   ├── README.md
│   ├── run_week3.py                  ← 主入口
│   ├── config.py / registry.py        ← 共享配置
│   ├── data_loader.py
│   ├── base_trainer.py / metrics.py
│   ├── models/                        ← 7 个模型实现
│   │   ├── arima_model.py / prophet_model.py
│   │   ├── lstm_model.py / gru_model.py
│   │   └── gcn_model.py / gat_model.py / gru_stgcn_residual.py
│   ├── scripts/
│   └── report/                        ← Week3 报告与日志
├── week4_deliverable/                ← Week4 交付包（时空联合建模）
│   ├── README.md
│   ├── run_week4.py                  ← 主入口（支持 STF/AGFormer/STGCN + 消融）
│   ├── config.py / registry.py
│   ├── data_loader.py / metrics.py
│   ├── models/                        ← 3 个时空模型
│   │   ├── stf_model.py              ← 时空解耦 Transformer
│   │   ├── agformer_model.py         ← 自适应图时空 Transformer
│   │   └── stgcn_model.py            ← 时空图卷积网络
│   ├── base_trainer.py
│   └── report/
├── results/                           ← 实验结果（模型预测 / 权重 / 日志）
└── evidence/                          ← 可视化图表 / P0 溯源证据
```

## 📊 核心结果

### 滑动窗口多步预测模型排名（1024 网格，48 步滚动）

| 排名 | 模型 | MAE ↓ | Corr ↑ | 参数量 |
|------|------|-------|--------|--------|
| 🥇 | **STF**（时空解耦 Transformer）| **327.19** | **0.8043** | 222K |
| 🥈 | AGFormer（自适应图 Transformer）| 386.91 | 0.6990 | 2,264K |
| 🥉 | STGCN（时空图卷积） | 429.18 | 0.6502 | 200K |
| 4 | GRU（城市级，Week3 基线）| 158.12 | 0.9452 | — |

> GRU 为城市级序列预测（口径不同），Corr 仅供参考。STF 在逐网格空间预测任务中表现最优。

### 消融实验

| 消融组 | 被剥离组件 | MAE 代价 | 贡献评级 |
|--------|-----------|---------|---------|
| STF → stf_loc_only | Env token + 跨节点注意力 | +2.2% | 中等 |
| AGFormer → agformer_static | **自适应邻接矩阵** | **+5.1%** | **关键** |

## 🔑 核心结论

1. **STF 是小样本最优**：222K 参数，1 epoch 收敛，时空解耦架构在有限数据下效率最高。
2. **自适应邻接矩阵是最有价值的设计**（贡献 5.1% MAE），但需要更大数据量才能兑现。
3. **Week3 局限性**：GRU 仅捕获时间维度，固定邻接 GCN 引入噪声；空间建模是突破瓶颈的关键。
4. **推荐路线图**：STF 预训练扩展 → 自适应图预热 → 级联 GRU+STF。

## 🚀 快速复现

```bash
# 登录 EC2
ssh -i ~/.ssh/aws-spatio-key.pem ubuntu@<EC2_IP>

# Week3 基线
cd ~/amazon/week3
python3 run_week3.py --models gru --target taxi_flow_total --tag v2

# Week4 时空模型
cd ~/amazon/week4
python3 run_week4.py --models stf agformer stgcn --target taxi_flow_total --tag v4fix
```

## 📄 完整报告

- `docs/北京数据质量分析报告.md` — 包含数据溯源（P0）、Week3 基线、Week4 实验、消融分析、综合总结
- `week3_deliverable/report/WEEK3_REPORT.md`
- `week4_deliverable/report/WEEK4_REPORT.md`

---

*Created: 2026-07-17 | Data: BJ Taxi GPS, BJ16 H5, ohsome POI (2015-11)*
