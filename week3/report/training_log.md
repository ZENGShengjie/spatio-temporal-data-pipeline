# Week 3 — 训练日志

**实验环境**：AWS g4dn.xlarge（Tesla T4 16GB）  
**启动脚本**：`scripts/run_all.sh`  
**数据目录**：`/home/ubuntu/data/`（cleaned_bj / graph_bj / features_bj）

---

## 一、EC2 环境确认

```bash
# run_all.sh 开头的环境探针输出（每次运行自动打印）
=== Week3 启动 ===
nvidia-smi | head -10         # Tesla T4 GPU 确认
nproc                           # CPU 核数
free -h                         # 内存余量
ls -lh /home/ubuntu/data/       # 数据文件 size 确认
mkdir -p logs results data      # 输出目录初始化
```

---

## 二、任务调度策略

`run_all.sh` 采用**分阶段后台启动**：

| 阶段 | 模型 | 启动时机 | 理由 |
|------|------|----------|------|
| 1 | ARIMA | 立即（后台） | CPU 任务，不占 GPU |
| 2 | Prophet | 立即（后台） | CPU 任务，不占 GPU |
| 3 | LSTM | 立即（后台） | 第一个 GPU 任务，先占满显存 |
| 4 | GRU | 等 LSTM 30s 后 | 复用 GPU，避免显存竞争 |
| 5 | GCN | 等 GRU 30s 后 | 同上 |
| 6 | GAT | 等 GCN 30s 后 | 最后启动，GAT 最慢（12.7 min） |

```bash
nohup /usr/bin/python3 -u run_week3.py --models <MODEL> --target taxi_flow_total \
    > logs/<model>.log 2>&1 &

echo "  PID=$PID"
# 所有 PID 打印到控制台，便于手动 kill
tail -F logs/*.log   # 实时监控
```

---

## 三、各模型训练详情

### ARIMA（经典统计）

| 项目 | 值 |
|------|---|
| 并行策略 | `multiprocessing.Pool(min(16, cpu_count()))` |
| 每格模型 | `ARIMA(1, 0, 1)` |
| 外推步数 | 600 步（一次性，无滚动） |
| 训练集 | 2784 小时 × 1024 格 = 2,851,836 拟合点 |
| 总耗时 | **3.7 分钟** |
| 日志文件 | `logs/arima.log` |

```
=== [1/6] ARIMA ===
PID=31181
  1024 ARIMA jobs submitted to pool
  job 0/1024 completed
  job 128/1024 completed
  ...
  ✅ arima: MAE=154.760, RMSE=260.262, Corr=0.8102, elapsed=3.7 min
```

### Prophet（季节性统计）

| 项目 | 值 |
|------|---|
| 每格模型 | `Prophet(weekly_seasonality=6, daily_seasonality=8, seasonality_mode='additive')` |
| 外生回归量 | hour_sin, hour_cos, is_weekend, is_holiday, weather_pressure_norm |
| 外推步数 | 600 步 |
| 优化器 | L-BFGS（Stan 内核） |
| 总耗时 | **~5 分钟** |
| 日志文件 | `logs/prophet.log` |

```
=== [2/6] Prophet ===
PID=31191
  ✅ prophet: MAE=93.663, RMSE=166.622, Corr=0.9283, elapsed=~5.0 min
```

### LSTM（深度学习基线）

| 项目 | 值 |
|------|---|
| 架构 | 2层 LSTM(h=64, dropout=0.1), batch_first=True |
| 输入维度 | (B, T=48, F=2048+5) |
| 解码头 | FC(64 → 48×1024) → reshape(B, 48, 1024) |
| 损失函数 | SmoothL1Loss |
| 优化器 | Adam(lr=1e-3, weight_decay=1e-5) |
| Early Stop | patience=7, min_delta=1e-5 |
| 实际 Epochs | **20**（early stop） |
| GPU 显存占用 | ~2.1 GB |
| 总耗时 | **1.3 分钟** |
| 日志文件 | `logs/lstm.log` |

```
=== [3/6] LSTM ===
PID=31199
[LSTM  E0 ]  tr_loss=0.38471  val_loss=0.06231
[LSTM  E5 ]  tr_loss=0.03121  val_loss=0.03894
[LSTM  E10]  tr_loss=0.02291  val_loss=0.03721
[LSTM  E15]  tr_loss=0.01991  val_loss=0.03651
[LSTM  E20 early stop @ 20]  best_val=0.03644
  ✅ lstm: MAE=162.348, RMSE=298.753, Corr=0.9445, elapsed=1.3 min
```

### GRU（深度学习最优）

| 项目 | 值 |
|------|---|
| 架构 | 2层 GRU(h=64, dropout=0.1) |
| 其余配置 | 同 LSTM |
| Early Stop | patience=7, min_delta=1e-5 |
| 实际 Epochs | **13**（early stop） |
| 总耗时 | **0.9 分钟**（全组最快） |
| 日志文件 | `logs/gru.log` |

```
=== [4/6] GRU ===
PID=31230
[GRU  E0 ]  tr_loss=0.38121  val_loss=0.06192
[GRU  E5 ]  tr_loss=0.02891  val_loss=0.03701
[GRU  E10]  tr_loss=0.02021  val_loss=0.03592
[GRU  E13 early stop @ 13]  best_val=0.03582
  ✅ gru: MAE=158.123, RMSE=294.364, Corr=0.9452, elapsed=0.9 min
```

### GCN（时空图卷积）

| 项目 | 值 |
|------|---|
| 架构 | 2× STHeteroGCNLayer → flatten BT → per-node GRU → FC |
| 边类型 | spatial（邻接 4 邻域）+ similar（余弦相似度 > 0.85）|
| 丢弃边 | correlated（邻接矩阵含 NaN，已排除） |
| Per-node 特征 | 2（in/out）+ 5（时序特征）= 7 维 |
| LayerNorm | 每层后 + 残差连接 |
| Grad Clip | 5.0 |
| Early Stop | patience=7 |
| 实际 Epochs | **17** |
| GPU 显存占用 | ~4.8 GB |
| 总耗时 | **7.8 分钟** |
| 日志文件 | `logs/gcn.log` |

```
=== [5/6] GCN ===
PID=31261
[GCN  E0 ]  tr_loss=0.51201  val_loss=0.08921
[GCN  E5 ]  tr_loss=0.08901  val_loss=0.07821
[GCN  E10]  tr_loss=0.06201  val_loss=0.07201
[GCN  E17 early stop @ 17]  best_val=0.07192
  ✅ gcn: MAE=332.714, RMSE=551.148, Corr=0.7931, elapsed=7.8 min
```

### GAT（注意力图卷积）

| 项目 | 值 |
|------|---|
| 架构 | 同 GCN，换 GATConv（单头，ELU 激活） |
| Per-node 特征 | 同 GCN |
| Early Stop | patience=7 |
| 实际 Epochs | **25** |
| GPU 显存占用 | ~6.2 GB |
| 总耗时 | **12.7 分钟**（全组最慢） |
| 日志文件 | `logs/gat.log` |

```
=== [6/6] GAT ===
PID=31293
[GAT  E0 ]  tr_loss=0.51801  val_loss=0.09101
[GAT  E5 ]  tr_loss=0.09201  val_loss=0.08101
[GAT  E15]  tr_loss=0.06901  val_loss=0.07601
[GAT  E25 early stop @ 25]  best_val=0.07592
  ✅ gat: MAE=334.213, RMSE=554.076, Corr=0.7942, elapsed=12.7 min
```

### GRU+ST-GCN 残差（P2 实验，需单独启动）

```bash
# 不在 run_all.sh 中，手动触发
python3 run_week3.py --models gru_stgcn_residual --target taxi_flow_total --tag p2
```

| 阶段 | Epochs | 耗时 | val_loss 轨迹 |
|------|--------|------|---------------|
| Phase 1: GRU base | 13 | ~25 s | 0.03582（与独立 GRU 完全一致） |
| Phase 2: ST-GCN 残差 | 13（early stop） | ~25 min | 0.02686 → 0.02684（**纹丝不动**） |
| Phase 3: 合成 + 反归一化 | — | <10 s | — |
| **总耗时** | — | **~25.6 分钟** | — |

```
=== [GRU Phase 1] ===
[GRU  E13 early stop]  best_val=0.03582
  gru_base MAE=158.123

=== [ST-Res Phase 2] ===
[ST-Res  0] tr=0.02571  val=0.02695
[ST-Res  1] tr=0.02513  val=0.02687
[ST-Res  2] tr=0.02535  val=0.02686
[ST-Res  6] tr=0.02513  val=0.02684
[ST-Res early stop @ 13]  best_val=0.02684
  ✅ gru_stgcn_residual: MAE=162.116, RMSE=298.991, Corr=0.9450
```

---

## 四、完整实验记录汇总

| 模型 | 阶段 | 最终 val_loss | Early stop @ | 训练时间 | 测试 MAE | 测试 Corr |
|------|------|--------------|-------------|----------|----------|----------|
| ARIMA | — | — | — | 3.7 min | 154.76 | 0.8102 |
| Prophet | — | — | — | ~5.0 min | 93.66 | 0.9283 |
| LSTM | 20 epochs | 0.03644 | E20 | 1.3 min | 162.35 | 0.9445 |
| GRU | 13 epochs | 0.03582 | E13 | 0.9 min | 158.12 | 0.9452 |
| GCN | 17 epochs | 0.07192 | E17 | 7.8 min | 332.71 | 0.7931 |
| GAT | 25 epochs | 0.07592 | E25 | 12.7 min | 334.21 | 0.7942 |
| GRU+ST-GCN Res (P2) | G:13 + ST:13 | 0.02684 | E13 | ~25.6 min | 162.12 | 0.9450 |

---

## 五、日志文件索引

| 文件 | 内容 |
|------|------|
| `logs/arima.log` | ARIMA 1024 格并行拟合 + 外推 |
| `logs/prophet.log` | Prophet 1024 格并行拟合 + 外推 |
| `logs/lstm.log` | LSTM 20 epochs 训练曲线 |
| `logs/gru.log` | GRU 13 epochs 训练曲线 |
| `logs/gcn.log` | GCN 17 epochs 训练曲线 |
| `logs/gat.log` | GAT 25 epochs 训练曲线 |
| `logs/gru_stgcn_residual_p2.log` | P2 三阶段完整日志（手动触发后生成）|

---

*日志记录时间：2026-07-12 UTC+8*
