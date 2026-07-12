"""ARIMA baseline V2 — 保留 V1 的 1-shot 600h 静态预测 (方案 B)
任务: 1-shot 预测测试集 600h, 不参与多步对比, 仅作古典基线参考
"""
from __future__ import annotations
import os, time, warnings, multiprocessing as mp
from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import config as cfg

warnings.filterwarnings("ignore")


def _arima_one_cell(args):
    cell_idx, train_data, test_data, order = args
    try:
        model = ARIMA(train_data.astype(np.float64), order=order,
                      enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(method_kwargs={"warn_convergence": False})
        n_test = len(test_data)
        yhat = result.forecast(steps=n_test)
        preds = [float(v) for v in yhat]
        gts   = [float(v) for v in test_data]
        return (cell_idx, preds, gts)
    except Exception:
        last = float(train_data[-1])
        n = len(test_data)
        return (cell_idx, [last] * n, [float(v) for v in test_data])


def run_arima(flow_4d: np.ndarray, order=(1, 0, 1), n_jobs=-1) -> Tuple[np.ndarray, np.ndarray]:
    """V2: 保留 V1 1-shot 600h 静态预测 (方案 B)"""
    T, _, H, W = flow_4d.shape
    flow_total = (flow_4d[:, 0] + flow_4d[:, 1]).reshape(T, -1)
    train = flow_total[:cfg.SPLIT.train_end].astype(np.float64)
    test  = flow_total[cfg.SPLIT.val_end:].astype(np.float64)
    N = flow_total.shape[1]
    print(f"[ARIMA V2] cells={N}, train={len(train)}, test={len(test)}, order={order}")
    start = time.time()
    args_list = [(i, train[:, i], test[:, i], order) for i in range(N)]
    if n_jobs == -1:
        n_jobs = max(1, min(os.cpu_count() or 4, 16))
    if n_jobs == 1:
        results = [_arima_one_cell(a) for a in args_list]
    else:
        with mp.Pool(n_jobs) as pool:
            results = pool.map(_arima_one_cell, args_list)
    elapsed = time.time() - start
    print(f"[ARIMA V2] done in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    n_test = len(test)
    pred_arr = np.zeros((n_test, N), dtype=np.float32)
    gt_arr   = np.zeros((n_test, N), dtype=np.float32)
    for idx, preds, gts in results:
        pred_arr[:, idx] = preds
        gt_arr[:, idx]   = gts
    return pred_arr, gt_arr