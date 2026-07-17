# Week4 模型对比分析报告

## 任务概述

Week4 实现三个高级时空预测模型，与 Week3 基线模型（GCN、GAT）进行性能对比。

## 模型列表

| 模型 | 核心机制 | 空间建模 | 时序建模 |
|------|---------|---------|---------|
| **STGCN** | 空间GCNConv + 因果GLU-TCN 三明治Block | 异构图卷积（spatial+similar）| 因果GLU-TCN（kernel=3）|
| **AGFormer** | 自适应图Transformer + Top-K稀疏注意力 | 自适应邻接矩阵（静态图初始化）+ Top-K注意 | 交替空间+时序Transformer |
| **Spacetimeformer** | Env+Loc分离 + 交叉注意力 | Loc独立编码 + Env全局引导 | 1D-Conv时序汇聚 |

## 基线对比（Week3）

| 模型 | MAE | RMSE | MAPE | Corr | 训练时间 |
|------|-----|------|------|------|---------|
| GCN (Week3基线) | — | — | — | — | — |
| GAT (Week3基线) | — | — | — | — | — |
| **STGCN** | — | — | — | — | — |
| **AGFormer** | — | — | — | — | — |
| **Spacetimeformer** | — | — | — | — | — |

*待训练后填写上述结果*

## 实验配置

- **数据集**: 北京出租车流量 (TaxiBJ) — (3888, 2, 32, 32)
- **切分**: train=2784h / val=504h / test=600h
- **输入**: seq_len=48 (24h) → **输出**: horizon=48 (24h)
- **设备**: CUDA GPU
- **损失函数**: SmoothL1Loss
- **优化器**: Adam, lr=1e-3, weight_decay=1e-5
- **早停**: patience=7 epochs

## 消融实验

| 实验 | 模型变体 | 目的 |
|------|---------|------|
| A1 | AGFormer vs AGFormer_static | 验证自适应图的独立贡献 |
| A2 | STGCN (无空间卷积) vs STGCN | 验证图卷积的独立贡献 |
| A3 | Spacetimeformer vs STF_Loc_Only | 验证全局Env上下文注入的价值 |

## STGCN 架构细节

```
Input: (B, N=1024, F_in=2+K_time, T=48)
  ├── input_proj: Linear(F_in, hidden)
  ├── [STBlock × 4]
  │     ├── Spatial GCNConv: concat(spatial, similar) → Linear → LayerNorm
  │     └── causal GLU-TCN: pad-left → Conv1d → GLU(gate) → LayerNorm
  ├── Residual + LayerNorm
  ├── per-node GRU over T
  └── decoder: Linear → ReLU → Linear → (B, N, horizon=48)
```

## AGFormer 架构细节

```
Input: (B, N=1024, F_in, T=48)
  ├── input_proj: Linear(F_in, hidden)
  ├── [AGFormerBlock × 2]
  │     ├── TopK-SpatialAttn: adaptive_adj (static-init) + MultiHeadAttn
  │     ├── TemporalAttn: MultiHeadAttn over T
  │     └── FFN: Linear → GELU → Linear
  ├── per-node GRU over T
  └── decoder → horizon
```

## Spacetimeformer 架构细节

```
Input: (B, N=1024, F_in, T=48)
  ├── Env路径: mean_pool(N) → Linear(n_nodes, hidden) → Transformer(T, d)
  ├── Loc路径: Linear(F_in, hidden) per-node per-timestep
  ├── CrossAttn: Loc query × Env key/value → (B*N, T, d)
  ├── Conv1d temporal: (B*N, d, T) → Conv1d(3) → Conv1d(3) → (B*N, d, T)
  └── decoder → horizon
```

## 结论

*待训练后分析*
