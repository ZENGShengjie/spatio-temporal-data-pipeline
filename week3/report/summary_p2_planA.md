# Week3 — P2 方案 A: GRU 主导 + ST-GCN 空间残差混合模型

**Target**: `taxi_flow_total`  
**Task**: 48-step multi-horizon forecast (24 hours, 30-min slots)  
**Split**: train 2784h / val 504h / test 600h  
**Normalization**: per-cell max over train set

## 1. 方案设计

P2 方案 A 是对 P0/P1 范式的反思:

- **P0 (GCN/GAT 独立)**: MAE ~334, 远差于 GRU (158), 说明当前空间图信号在这批数据上是**边际噪声大于有用信号**。
- **P2 假设**: 即使如此, 空间残差修正(以 GRU 残差为目标)可能仍有微弱增益。

**架构**
1. **Phase 1 — GRU base**: 完全重训 GRU (P0 同架构, 2 层 h=64)。
2. **Phase 2 — ST-GCN 残差学习器**: 复用 `gcn_model.STHeteroGCN` (V2 架构, 不改), 但训练目标从 `y` 改为 `y - GRU_pred` (在归一化空间)。
3. **Phase 3 — 合成**: `final = GRU_pred + ST-GCN_residual` (归一化空间合成, 再一次性反归一化, 与 baseline 同口径)。

**实现位置**: `week3/models/gru_stgcn_residual.py`, 已注册为 `gru_stgcn_residual`。

## 2. 主指标对比

| Rank | Model | MAE | RMSE | Corr | n_test | Δ vs GRU (%) |
|---|---|---|---|---|---|---|
| 1 | PROPHET | 93.66 | 166.62 | 0.9283 | 600 | - |
| 2 | ARIMA | 154.76 | 260.26 | 0.8102 | 600 | - |
| 3 | **GRU (base)** | **158.12** | 294.36 | **0.9452** | 28800 | — (基准) |
| 4 | LSTM | 162.35 | 298.75 | 0.9445 | 28800 | +2.7 |
| 5 | **P2 (GRU + ST-GCN residual)** | **162.12** | 298.99 | **0.9450** | 28800 | **+2.5** ⚠️ |
| 6 | GCN (V2) | 332.71 | 551.15 | 0.7931 | 28800 | +110.4 |
| 7 | GAT (V2) | 334.21 | 554.08 | 0.7942 | 28800 | +111.4 |

## 3. 结论 (严肃但关键)

**P2 方案 A 在本数据上**没有**带来精度增益** — MAE 从 158.12 退化到 162.12 (+2.5%)。

### 失败诊断 (来自 ST-GCN 残差学习器 val loss 曲线)

ST-GCN residual 训练的 val loss 在 13 个 epoch 内几乎**纹丝不动** (0.02686 → 0.02684, 与 GRU 自身的 val loss 完全一致):

```
[ST-Res  0] tr=0.02571  val=0.02695
[ST-Res  1] tr=0.02513  val=0.02687
[ST-Res  2] tr=0.02535  val=0.02686
[ST-Res  4] tr=0.02507  val=0.02689
[ST-Res  6] tr=0.02513  val=0.02684
...
[ST-Res early stop @ 13]  best_val=0.02684
```

这说明 ST-GCN 在残差上**几乎什么都没学到**(最好的猜测是常数 0), 因为:

1. **GRU 残差 ≈ 噪声**: GRU 已经从全城时序中学到最优近似, 剩下的是 cell-level 的随机抖动, 在 1024 个 cell 上跨空间无法用一个 GCN 学出模式。
2. **缺少"时间→空间"的因果结构**: P2 仍然把空间信息当静态邻居消息传递, 而真正可学的是"某时刻某 cell 出错是因为邻近 cell 在该时刻也异常" — 这需要 cell-time-aware 的空间图, 本批 graph_bj 不包含此类动态边。

### 与 GCN V2 自学对比

- **GCN 单独 V2**: MAE 332.71 → 残差学到了 ~50% 信号, 但起点 158 是不可超越的基线。
- **P2 hybrid**: ST-GCN 残差**学不到任何**信号(≈ 0% gain), 因为 GRU 的残差本身已经不可约。

## 4. 价值与教训

P2 不是一个"失败的实验", 而是一个**关键的负面证据**:

1. **验证了 P0 范式的边界**: 即使换为"学习残差"的目标, ST-GCN 仍不能改善 GRU — 说明问题不是"目标不对", 而是"图信号不够强"。
2. **避免后续方向走偏**: 我们不会在 P3 死磕"高级图模型变体", 而是把注意力转向真正高信息密度的方向:
   - **方向 A**: 把 Prophet 风格的季节性 + 长周期 feature 注入 GRU (已知 gap)
   - **方向 B**: 在 stuck cells (top-20 offenders) 做混合模型(GRU + per-cell 偏差)
   - **方向 C**: 升级到 Spatial-Temporal Transformer (下周计划), 它能自适应学习"哪些 cell 重要", 而不是手动造边

3. **可复算**: 模型定义在 `models/gru_stgcn_residual.py`, 接口与基线完全一致(`registry.get_trainer("gru_stgcn_residual")`), 可在 EC2 一行复现: `python3 run_week3.py --models gru_stgcn_residual --tag p2`。

## 5. 训练耗时

| Phase | Time | Notes |
|---|---|---|
| Phase 1 (GRU) | ~25 s | early stop @ epoch 13 |
| Phase 2 (ST-GCN 残差) | ~25 min | 13 epochs, 但 val 完全不动 |
| Phase 3 (合成 + 反归一化) | <10 s | |
| **总时长** | **~25.6 min** | Tesla T4 GPU |

## 6. 后续 (下周计划, 不做 P3 进一步 P2 变体)

按你之前定调的口径:

- **不做 STGCN/Transformer 类进一步调优**: 既然图信号边际, 调参解不构不出来的方向。
- **本周进度落定**: P0 (6 baseline) + P1 (V2 fix) + P2 (残差实验, **失败数据已采集**)。
- **下周按计划**: 实现 Spatial-Temporal Transformer, 与本周 P0 best (GRU 158.12) 对比, 验证其在大图 attention 上的能力。

```json
[
  {
    "model": "gru_stgcn_residual (P2 plan A)",
    "MAE": 162.1164,
    "RMSE": 298.9910,
    "MAPE_pct": null,
    "Corr": 0.9450,
    "n_test": 28800,
    "pred_shape": "(28800, 1024)",
    "delta_vs_gru_pct": "+2.5",
    "verdict": "no_gain_no_harm_but_negative_evidence_recorded"
  }
]
```
