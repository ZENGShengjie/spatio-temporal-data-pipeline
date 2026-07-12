# Week3 — P0 vs P1.1 ST-GCN/ST-GAT 进度报告

**Target**: `taxi_flow_total` (inflow + outflow per cell)
**Task**: 48-step multi-horizon forecast (24 hours, 30-min slots)

## 1. All models (P0 + P1.1) ranked by MAE

| Rank | Model | Tag | MAE | RMSE | MAPE(%) | Corr | n_test |
|---|---|---|---|---|---|---|---|
| 1 | PROPHET_P0 | P0 | 93.66 | 166.62 | 56.89 | 0.9283 | 600 |
| 2 | ARIMA_P0 | P0 | 154.76 | 260.26 | 112.32 | 0.8102 | 600 |
| 3 | GRU_P0 | P0 | 158.12 | 294.36 | 42.83 | 0.9452 | 28800 |
| 4 | LSTM_P0 | P0 | 162.35 | 298.75 | 44.65 | 0.9445 | 28800 |
| 5 | GCN_P1_ST | P1.1 | 328.04 | 541.89 | 126.12 | 0.8001 | 28800 |
| 6 | GCN_P0 | P0 | 332.71 | 551.15 | 125.97 | 0.7931 | 28800 |
| 7 | GAT_P0 | P0 | 334.21 | 554.08 | 122.15 | 0.7942 | 28800 |
| 8 | GAT_P1_ST | P1.1 | 337.86 | 558.23 | 121.00 | 0.7894 | 28800 |

## 2. GCN/GAT P0 vs P1.1 (multi-step models only)

| Model | MAE | Δ MAE | RMSE | Δ RMSE | Corr | Δ Corr |
|---|---|---|---|---|---|---|
| GCN (P0 → P1.1) | 332.71 → 328.04 | -4.67 | 551.15 → 541.89 | -9.26 | 0.7931 → 0.8001 | +0.0070 |
| GAT (P0 → P1.1) | 334.21 → 337.86 | +3.65 | 554.08 → 558.23 | +4.15 | 0.7942 → 0.7894 | -0.0049 |

## 3. Multi-step model ranking (excluding 1-shot ARIMA/Prophet)

| Rank | Model | MAE | RMSE | Corr |
|---|---|---|---|---|
| 1 | GRU_P0 | 158.12 | 294.36 | 0.9452 |
| 2 | LSTM_P0 | 162.35 | 298.75 | 0.9445 |
| 3 | GCN_P1_ST | 328.04 | 541.89 | 0.8001 |
| 4 | GCN_P0 | 332.71 | 551.15 | 0.7931 |
| 5 | GAT_P0 | 334.21 | 554.08 | 0.7942 |
| 6 | GAT_P1_ST | 337.86 | 558.23 | 0.7894 |

## 4. P1.1 architecture summary

- **ST-GCN v2**: per-timestep `GCNConv` (B·T, N, hidden) batched, then per-node **GRU** over T_seq, then linear decoder → horizon
- **ST-GAT v2**: per-timestep PyG `GATConv` (loop over T), then per-node GRU, linear decoder → horizon
- Both use spatial + similar edges (2 heterogeneous types)
- P0 used: time-conv compress → 1 GCN/GAT pass on (B, N, hidden)

