# Week 3 — 北京出租车流量预测基线交付包

> 7 个模型（ARIMA / Prophet / LSTM / GRU / GCN / GAT / GRU+ST-GCN 残差）
> 统一框架、统一评估口径、完整可复现

---

## 📦 目录结构

```
week3_deliverable/
├── README.md                              ← 本文件
├── run_week3.py                           ← 主入口（通用 CLI）
├── config.py                              ← 共享配置（Split / TrainCfg / 指标）
├── registry.py                            ← name → trainer class 查找表
├── data_loader.py                         ← 数据加载 / Dataset 定义
├── base_trainer.py                         ← 共享工具（seed / device / normalize）
├── metrics.py                              ← BaseTrainer ABC + 评估函数
│
├── models/                                ← 7 个模型实现
│   ├── arima_model.py  + arima_runner.py
│   ├── prophet_model.py + prophet_runner.py
│   ├── lstm_model.py
│   ├── gru_model.py
│   ├── gcn_model.py                       ← STHeteroGCN（V2，P1.1 重构）
│   ├── gat_model.py                       ← STHeteroGAT（V2，P1.1 重构）
│   └── gru_stgcn_residual.py              ← P2 混合模型
│
├── scripts/                               ← 高效训练脚本
│   ├── run_all.sh                         ← 6 基线并行调度（g4dn 上执行）
│   ├── install_torch.sh                   ← PyTorch + PyG 安装
│   └── fix_crlf.sh                        ← Windows CRLF 修复
│
└── report/                                ← 报告与日志
    ├── WEEK3_REPORT.md                    ← 主报告（设计 / 结果 / 分析）
    ├── training_log.md                    ← 训练日志（各模型 epoch / 耗时）
    ├── comparison_table.md                ← 性能对比表（含 P2）
    ├── metrics_v2_all.json                ← 原始数值（JSON 可直接解析）
    ├── summary_p2_planA.md                ← P2 残差实验专项报告
    └── top_offenders.md                   ← Bad Cells 分析
```

---

## 🚀 快速复现（EC2 g4dn.xlarge）

```bash
# 1. 登录
ssh -i ~/.ssh/aws-spatio-key.pem ubuntu@3.235.182.223

# 2. 一键跑完 6 个基线（后台自动调度，日志写 logs/*.log）
cd /home/ubuntu/amazon/week3
bash scripts/run_all.sh

# 3. 单独跑某个模型
python3 run_week3.py --models gru --target taxi_flow_total --tag v2

# 4. 跑 P2 残差实验
python3 run_week3.py --models gru_stgcn_residual --target taxi_flow_total --tag p2

# 5. 拉回本地结果
scp -i ~/.ssh/aws-spatio-key.pem \
    ubuntu@3.235.182.223:/home/ubuntu/amazon/week3/results/*.npy \
    ./results/
```

---

## 📊 核心结果速览

| 排名 | 模型 | MAE | MAPE(%) | Corr | 训练时间 |
|------|------|-----|---------|------|----------|
| 1 | **Prophet** | **93.66** | 56.89 | 0.9283 | ~5 min |
| 2 | ARIMA | 154.76 | 112.32 | 0.8102 | 3.7 min |
| 3 | **GRU** | **158.12** | **42.83** | **0.9452** | **0.9 min** |
| 4 | LSTM | 162.35 | 44.65 | 0.9445 | 1.3 min |
| 5 | GRU+ST-GCN Res | 162.12 | — | 0.9450 | ~25 min |
| 6 | GCN | 332.71 | 125.97 | 0.7931 | 7.8 min |
| 7 | GAT | 334.21 | 122.15 | 0.7942 | 12.7 min |

> Prophet/ARIMA 为 1次预测（600 点），GRU/LSTM/GCN/GAT 为多步滚动预测（28800 点），口径不同请勿直接比较 MAE。

---

## 🔑 关键结论

1. **Prophet 意外领先**：周期季节性建模 > 深度时序 / 图模型
2. **GRU 是多步模型最优**：快（<1min）+ 准（MAPE 43%）
3. **GCN/GAT 图信号无效**：MAE 是 GRU 的 2.1 倍
4. **P2 残差实验负面**：ST-GCN 在 GRU 残差上完全学不到东西（val_loss 停滞在 0.02684），图卷积不适用这批数据
5. **Bad Cells 集中在高流量区**：格点 375、427、302 等所有模型一致预测差，需专项修正

---

## 📋 模型注册机制

所有模型通过 `registry.py` 统一注册：

```python
from registry import get_trainer, list_models
print(list_models())
# ['arima', 'prophet', 'lstm', 'gru', 'gcn', 'gat', 'gru_stgcn_residual']

trainer = get_trainer("gru")
pred, gt = trainer.fit_predict(flow_4d, time_features, target="taxi_flow_total")
```

---

*交付时间：2026-07-12 21:00 UTC+8*
