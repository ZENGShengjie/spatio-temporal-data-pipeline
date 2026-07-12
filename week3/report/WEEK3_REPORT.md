# Week 3 — 北京出租车流量预测：7 个模型基线对比报告

**任务**：用过去 24 小时（48 个时间片 × 30 分钟）预测未来 24 小时北京市 32×32 网格（共 1024 格点）的人流进出总量  
**数据**：3888 个时间片 / 训练 2784h / 验证 504h / 测试 600h（Tesla T4 GPU）  
**时间**：2026 年 7 月 12 日

---

## 一、实验设计

### 1.1 数据

| 数据集 | 路径 | 规格 |
|--------|------|------|
| 流量数据 | `cleaned_bj/taxi_p4_4d.npz` | (3888, 2, 32, 32) float32，inflow + outflow |
| 空间图 | `graph_bj/bj_hetero_graph.pt` | PyG HeteroData，spatial + similar 两种边 |
| 时间特征 | `features_bj/bj_features.parquet` | hour_sin/cos、is_weekend、is_holiday、weather_pressure_norm |
| 网格元数据 | `grid_bj/` | 32×32 网格定义 |

**归一化**：按训练集每格历史最大值反归一化（防极值干扰）  
**目标变量**：`taxi_flow_total = inflow + outflow`

### 1.2 七模型一览

| # | 模型 | 类型 | 多步 | 核心思想 |
|---|------|------|------|----------|
| 1 | ARIMA | 经典统计 | ❌ 1次 | 每格独立 ARIMA(1,0,1)，一次性外推 600 步 |
| 2 | Prophet | 统计+季节性 | ❌ 1次 | 同上，加周周期(6) + 日周期(8)，注入 5 个外生回归量 |
| 3 | LSTM | 深度 RNN | ✅ 48步 | 2层 LSTM(h=64)，全城 2048 维向量输入，共享编码器 |
| 4 | GRU | 深度 RNN | ✅ 48步 | 同架构换 GRU 单元，轻量快速 |
| 5 | GCN | 空间图卷积 | ✅ 48步 | ST-GCN（2层）→ 每格独立 GRU → MLP 解码头 |
| 6 | GAT | 注意力图卷积 | ✅ 48步 | 同架构换 GATConv（单头，ELU 激活） |
| 7 | GRU+ST-GCN 残差 | 混合 | ✅ 48步 | Phase1 训 GRU；Phase2 冻住，ST-GCN 学残差；合成预测 |

> **1次 vs 多步说明**：ARIMA/Prophet 输出 (600, 1024)，RNN/GNN 输出 (28800, 1024)。MAE 直接对比仅供参考。

### 1.3 共同训练配置

```python
seq_len=48, horizon=48, batch=16, epochs=50, lr=1e-3
optimizer=Adam, loss=SmoothL1Loss, early_stop_patience=7
hidden=64, layers=2, dropout=0.1, grad_clip=5.0
```

---

## 二、核心结果

### 2.1 六基线模型（V2）综合排名

| 排名 | 模型 | MAE | RMSE | MAPE(%) | Corr | 训练时间 | Epochs |
|------|------|-----|------|---------|------|----------|--------|
| 1 | **Prophet** | **93.66** | 166.62 | 56.89 | 0.9283 | ~5 min | — |
| 2 | ARIMA | 154.76 | 260.26 | 112.32 | 0.8102 | 3.7 min | — |
| 3 | **GRU** | **158.12** | 294.36 | **42.83** | **0.9452** | **0.9 min** | 13 |
| 4 | LSTM | 162.35 | 298.75 | 44.65 | 0.9445 | 1.3 min | 20 |
| 5 | GCN | 332.71 | 551.15 | 125.97 | 0.7931 | 7.8 min | 17 |
| 6 | GAT | 334.21 | 554.08 | 122.15 | 0.7942 | 12.7 min | 25 |

### 2.2 P2 混合模型（残差学习实验）

| 模型 | MAE | RMSE | Corr | Δ vs GRU |
|------|-----|------|------|----------|
| GRU（基准） | 158.12 | 294.36 | 0.9452 | — |
| **GRU+ST-GCN 残差** | 162.12 | 298.99 | 0.9450 | **+2.5%** ⚠️ |

> **P2 结论**：MAE 退化 2.5%，ST-GCN 残差阶段 val_loss 完全不动（0.02684 → 0.02686），是最重要的负面证据——图卷积无法从 GRU 残差中提取有意义的信号。

---

## 三、关键发现

### 3.1 图卷积在此数据上无效

GCN/GAT 的 MAE（~333）是 GRU（158）的 **2.1 倍**。即便经过 P1.1 V2 重构（per-timestep ST-GCN + residual + LayerNorm），改善极微（P1.1: 328.04 vs P0: 334.21）。

**原因推断**：
- 空间邻接/相似图的边权重信号太弱（相关性 < 噪声阈值）
- 1024 个格点 + 每步仅 7 维特征（2 通道 + 5 时序），GCNConv 每步传播一次即稀释殆尽
- 缺少"时间-空间"联合建模——真正可学的是"某时刻某格异常时，其邻近格在同时刻也异常"，但静态图无法表达

### 3.2 Prophet 出人意料地好

每个格独立建模（无空间信息）反而 MAE 最低（93.66），因为：
- 出租车流量的主要信号是**天周期 + 周周期**，Prophet 的季节性建模精准捕获
- 无空间传播误差累积，1次预测天然比多步滚动更稳定

### 3.3 GRU 是多步模型的甜点

- 比 LSTM 快（0.9 min vs 1.3 min），但 MAE 更低（158 vs 162）
- MAPE 43% 为全组最低，说明对高流量格点也有较好的比例控制
- Corr 0.9452 说明整体趋势拟合非常好

### 3.4 残差学习死锁

P2 混合模型训练日志：

```
[ST-Res  0] tr=0.02571  val=0.02695
[ST-Res  1] tr=0.02513  val=0.02687
[ST-Res  2] tr=0.02535  val=0.02686
[ST-Res  6] tr=0.02513  val=0.02684
[ST-Res early stop @ 13]  best_val=0.02684
```

val_loss 在 13 个 epoch 内**纹丝不动**，证明 ST-GCN 在残差上完全学不到东西，Phase-2 等效于输出常数 0。

---

## 四、模型实现要点

### 4.1 ARIMA（独立每格）
- `ARIMA(1,0,1)` — 1 阶自回归 + 0 阶差分 + 1 阶滑动平均
- `multiprocessing.Pool(n_jobs=min(16, cpu_count()))` 进程级并行
- 1次外推 600 步，无滚动

### 4.2 Prophet（独立每格）
- 加性季节性：weekly=6, daily=8
- 外生回归量：hour_sin, hour_cos, is_weekend, is_holiday, weather_pressure_norm
- L-BFGS 优化（Stan 内核）

### 4.3 LSTM / GRU（全城序列）
- 输入：(B, T_seq=48, F_in=2048+K_time=5)
- 2层 RNN → 末状态 → FC(64 → 48×1024) → reshape(B, 48, 1024)
- **同一隐状态在 1024 格间共享**，适合捕获全市宏观时序模式

### 4.4 GCN / GAT（时空图）
- 每步先做空间聚合：`(B, T, N, F=2+K)` → 逐时间片过 GCNConv/GATConv → concat + fuse + LayerNorm + 残差
- 2层 ST-Block 后 flatten → 每格独立 GRU 串时间 → LayerNorm → MLP 解码头
- 边类型：spatial（邻接）+ similar（流量相似性），correlated 边因 NaN 被剔除

### 4.5 GRU+ST-GCN 残差（混合）
- Phase1：训标准 GRU，checkpoint 后冻结
- Phase2：在归一化空间训练 ST-GCN，目标为 `y - gru_pred`
- Phase3：合成 `final = gru_pred_norm + stgcn_residual_norm`，统一反归一化

---

## 五、Bad Cells 分析（Top-20 Offenders）

所有模型一致预测最差的格点：

| 排名 | 格点 | 特征 | 根因 |
|------|------|------|------|
| 1 | #375 | 金融街区域 | 高流量绝对值大，误差放大 |
| 2 | #427 | 朝阳核心区 | 同上 |
| 3 | #302 | 海淀南部 | 同上 |
| 4 | #588 | 丰台区 | 同上 |
| 5 | #600 | 顺义/机场方向 | 突发性流量难以用历史模式预测 |

> 注：MAE 大的格点不一定 MAPE 高；Bad Cells 主要反映高流量区域的绝对误差放大效应。

---

## 六、后续方向（Week 4 建议）

| 方向 | 做法 | 预期收益 |
|------|------|----------|
| **A. 周期特征注入** | 给 GRU 增加 hour_sin/cos、dow_onehot、holiday_flag 等显式周期 embedding | 填补 GRU 不懂周期的盲区 |
| **B. Per-cell 偏差校正** | 用 Bad Cells 列表做混合：GRU 预测 + cell-level 线性偏移 | 针对性修正高频差格点 |
| **C. Spatial-Temporal Transformer** | 替换 GCN/GAT，用 multi-head self-attention 自适应学习格点间相关性 | 让模型自己决定哪些格点该相互注意 |

---

## 七、快速复现

```bash
# EC2 g4dn.xlarge 已就绪
ssh -i ~/.ssh/aws-spatio-key.pem ubuntu@3.235.182.223

# 安装依赖（如需）
bash /home/ubuntu/amazon/week3/scripts/install_torch.sh

# 跑全部 6 基线
cd /home/ubuntu/amazon/week3
bash scripts/run_all.sh

# 或单独跑某个模型
python3 run_week3.py --models gru --target taxi_flow_total --tag v2

# 跑 P2 残差实验
python3 run_week3.py --models gru_stgcn_residual --target taxi_flow_total --tag p2
```

---

*报告生成时间：2026-07-12 20:53 UTC+8*
