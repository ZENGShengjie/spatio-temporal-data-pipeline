# Week 4 — 时空联合建模实验交付包

> 3 个高级时空模型（STF / AGFormer / STGCN）+ 2 组消融实验 + 完整可复现框架

---

## 📦 目录结构

```
week4_deliverable/
├── README.md                              ← 本文件
├── run_week4.py                           ← 主入口（通用 CLI，支持所有模型 + 消融）
├── config.py                              ← 共享配置（Split / TrainCfg / 超参数）
├── registry.py                            ← name → trainer class 查找表
├── base_trainer.py                        ← 共享工具（seed / device / normalize）
├── metrics.py                             ← 评估函数（MAE / RMSE / MAPE / Corr）
├── data_loader.py                         ← 数据加载 / Dataset 定义
├── __init__.py
│
├── models/                                ← 3 个时空模型实现
│   ├── stf_model.py                      ← 时空解耦 Transformer（222K 参数）
│   ├── agformer_model.py                 ← 自适应图时空 Transformer（2.26M 参数）
│   └── stgcn_model.py                    ← 时空图卷积网络（200K 参数）
│
└── report/                               ← 报告与日志
    └── WEEK4_REPORT.md                   ← Week4 专项报告
```

---

## 🚀 快速复现（EC2 g4dn.xlarge）

```bash
# 1. 登录
ssh -i ~/.ssh/aws-spatio-key.pem ubuntu@<EC2_IP>

# 2. 一键跑完全部模型（含消融）
cd ~/amazon/week4
python3 run_week4.py --target taxi_flow_total --tag v4fix

# 3. 单独跑某个模型
python3 run_week4.py --models stf --target taxi_flow_total --tag v4fix
python3 run_week4.py --models agformer --target taxi_flow_total --tag v4fix
python3 run_week4.py --models stgcn --target taxi_flow_total --tag v4fix

# 4. 运行消融实验
python3 run_week4.py --models stf_loc_only --target taxi_flow_total --tag ablation1
python3 run_week4.py --models agformer_static --target taxi_flow_total --tag ablation1

# 5. 拉回本地结果
scp -i ~/.ssh/aws-spatio-key.pem \
    ubuntu@<EC2_IP>:/home/ubuntu/amazon/week4/weights/*.pth \
    ./week4_deliverable/models/
```

---

## 📊 核心结果速览

### 完整模型排名

| 排名 | 模型 | MAE | RMSE | MAPE | Corr | 参数 | 训练耗时 |
|------|------|-----|------|------|------|------|---------|
| 1 | **STF** | **327.19** | 540.17 | **1.23** | **0.8043** | 222K | 515s |
| 2 | AGFormer | 386.91 | 655.39 | 1.30 | 0.6990 | 2,264K | 9,491s |
| 3 | STGCN | 429.18 | 727.29 | 1.45 | 0.6502 | 200K | 9,695s |

### 消融实验

| 消融组 | 被剥离组件 | MAE | Δ MAE | Corr | Δ Corr |
|--------|-----------|------|-------|------|-------|
| stf → stf_loc_only | Env token + 跨节点注意力 | 334.39 | +7.20 (+2.2%) | 0.7942 | −0.0101 |
| agformer → agformer_static | 自适应邻接矩阵 | 406.50 | +19.59 (+5.1%) | 0.6668 | −0.0322 |

---

## 🔑 关键结论

1. **STF 是小样本最优**：222K 参数 + 1 epoch 早停，时空解耦架构在有限数据下参数效率最高。
2. **自适应邻接矩阵是最关键设计**（贡献 5.1% MAE），是 Env token（2.2%）的 2.3 倍。
3. **AGFormer 自适应图冷启动**：226 万参数在小样本下未充分训练，自适应邻接反而拖累收敛。
4. **STGCN 固定图天花板**：无法捕捉动态通勤关系，性能最差（MAE=429）。
5. **Epoch 22 断崖现象**：AGFormer_static 在 epoch 22 出现 train loss 骤降，印证小样本下自适应分支的优化不稳定性。

---

## ⚙️ 配置文件说明

`config.py` 中关键参数：

```python
seq_len = 48       # 输入序列长度（小时）
horizon = 48       # 预测步长（小时）
split = dict(train=2784, val=504, test=600)   # 3888h 全量
batch_size = 32
epochs = 30
patience = 12      # 早停耐心值
```

---

## 📋 模型注册机制

```python
from registry import get_trainer, list_models
print(list_models())
# ['stf', 'agformer', 'stgcn', 'stf_loc_only', 'agformer_static']

trainer = get_trainer("stf")
pred, gt = trainer.fit_predict(flow_4d, time_features, target="taxi_flow_total")
```

---

*交付时间：2026-07-17 | EC2 g4dn.xlarge 已停止*
