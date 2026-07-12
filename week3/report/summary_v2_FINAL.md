# Week3 — 6 baseline models (V2) comparison

**Target**: `taxi_flow_total` (inflow + outflow per cell)  
**Task**: 48-step multi-horizon forecast (24 hours, 30-min slots)  
**Split**: train 2784h / val 504h / test 600h  
**Normalization**: per-cell max over train set  

## 1. Main metrics (ranked by MAE)

| Rank | Model | MAE | RMSE | MAPE(%) | Corr | n_test | pred_shape |
|---|---|---|---|---|---|---|---|
| 1 | PROPHET | 93.66 | 166.62 | 56.89 | 0.9283 | 600 | (600, 1024) |
| 2 | ARIMA | 154.76 | 260.26 | 112.32 | 0.8102 | 600 | (600, 1024) |
| 3 | GRU | 158.12 | 294.36 | 42.83 | 0.9452 | 28800 | (28800, 1024) |
| 4 | LSTM | 162.35 | 298.75 | 44.65 | 0.9445 | 28800 | (28800, 1024) |
| 5 | GCN | 332.71 | 551.15 | 125.97 | 0.7931 | 28800 | (28800, 1024) |
| 6 | GAT | 334.21 | 554.08 | 122.15 | 0.7942 | 28800 | (28800, 1024) |

## 2. Key notes

- **ARIMA(1,0,1)**: 1024 independent univariate ARIMA models, **1-shot 600h static forecast** (multi-step extension not implemented in V2). Listed for classical baseline reference only; its MAE/RMSE is **not directly comparable** to multi-step models (LSTM/GRU/GCN/GAT predict in `(B=600, H=48, N=1024) → flatten to (28800, 1024)` resolution, ARIMA is just (600, 1024)).
- **Prophet**: 1024 independent Prophet models with weekly + daily seasonality, also 1-shot.
- **LSTM/GRU**: 2-layer (h=64), city-level sequence (2048 + K=5 features), SmoothL1 loss.
- **GCN/GAT (P0 fix)**: PyG native sparse `GCNConv`/`GATConv` on `spatial + similar` edges (skipped `correlated` due to NaN in adjacency). 2-layer, h=64.

## 3. Architecture summary

| Model | Type | Multi-step | Edge types | Notes |
|---|---|---|---|---|
| ARIMA | classical | ❌ 1-shot | — | Univariate per cell |
| Prophet | classical | ❌ 1-shot | — | Weekly + daily seasonality |
| LSTM | RNN | ✅ 48-step | — | City-level sequence |
| GRU | RNN | ✅ 48-step | — | Lighter than LSTM |
| GCN | GNN | ✅ 48-step | spatial + similar | PyG native |
| GAT | GNN+attention | ✅ 48-step | spatial + similar | PyG native, single-head |

## 4. Best/worst among multi-step models

- **Best MAE (multi-step)**: GRU at 158.12
- **Worst MAE (multi-step)**: GAT at 334.21

## 5. Per-model training time

| Model | Time | Epochs | Notes |
|---|---|---|---|
| ARIMA | 3.7 min | — | 1024 cells parallel via multiprocessing |
| Prophet | ~5 min | — | 1024 cells parallel |
| LSTM | 1.3 min | 20 (early stop) | Tesla T4 GPU |
| GRU | 0.9 min | 13 (early stop) | Tesla T4 GPU |
| GCN | 7.8 min | 17 (early stop) | Tesla T4 GPU |
| GAT | 12.7 min | 25 (early stop) | Tesla T4 GPU |


## 6. JSON dump

```json
[
  {
    "model": "arima",
    "MAE": 154.76040649414062,
    "RMSE": 260.26226806640625,
    "MAPE_pct": 112.32414245605469,
    "Corr": 0.8102396675839528,
    "n_test": 600,
    "pred_shape": "(600, 1024)"
  },
  {
    "model": "prophet",
    "MAE": 93.66341400146484,
    "RMSE": 166.62245178222656,
    "MAPE_pct": 56.89258575439453,
    "Corr": 0.9282961235227948,
    "n_test": 600,
    "pred_shape": "(600, 1024)"
  },
  {
    "model": "lstm",
    "MAE": 162.34750366210938,
    "RMSE": 298.7531433105469,
    "MAPE_pct": 44.647518157958984,
    "Corr": 0.9444704124145964,
    "n_test": 28800,
    "pred_shape": "(28800, 1024)"
  },
  {
    "model": "gru",
    "MAE": 158.12322998046875,
    "RMSE": 294.3638610839844,
    "MAPE_pct": 42.834903717041016,
    "Corr": 0.9452436621165647,
    "n_test": 28800,
    "pred_shape": "(28800, 1024)"
  },
  {
    "model": "gcn",
    "MAE": 332.71368408203125,
    "RMSE": 551.1478881835938,
    "MAPE_pct": 125.97447967529297,
    "Corr": 0.7930667507182974,
    "n_test": 28800,
    "pred_shape": "(28800, 1024)"
  },
  {
    "model": "gat",
    "MAE": 334.2132263183594,
    "RMSE": 554.0762329101562,
    "MAPE_pct": 122.14722442626953,
    "Corr": 0.7942414993152315,
    "n_test": 28800,
    "pred_shape": "(28800, 1024)"
  }
]
```
