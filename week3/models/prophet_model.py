"""Prophet baseline V2 — 1024 independent Prophet models
- 多步预测 (预测 test 区间每个时刻)
- 接入额外回归变量: hour_sin, hour_cos, is_weekend, is_holiday, weather_pressure_norm
"""
from __future__ import annotations
import os, time, warnings, multiprocessing as mp
import numpy as np
import pandas as pd
import logging
import config as cfg

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def _prophet_one_cell(args):
    """V2: 增加 extra_regressors (5个时间特征)"""
    cell_idx, train_ts, train_y, test_ts, test_extra, weekly_seas, daily_seas = args
    try:
        from prophet import Prophet
        train_df = pd.DataFrame({
            "ds": pd.to_datetime(train_ts),
            "y": train_y,
            "hour_sin": train_extra["hour_sin"],
            "hour_cos": train_extra["hour_cos"],
            "is_weekend": train_extra["is_weekend"],
            "is_holiday": train_extra["is_holiday"],
            "weather_pressure_norm": train_extra["weather_pressure_norm"],
        }) if train_extra is not None else pd.DataFrame({"ds": pd.to_datetime(train_ts), "y": train_y})

        m = Prophet(weekly_seasonality=weekly_seas, daily_seasonality=daily_seas,
                    yearly_seasonality=False, seasonality_mode="additive")
        # 注册额外回归量
        if train_extra is not None:
            for col in ["hour_sin", "hour_cos", "is_weekend", "is_holiday", "weather_pressure_norm"]:
                m.add_regressor(col)
        m.fit(train_df)

        future = pd.DataFrame({
            "ds": pd.to_datetime(test_ts),
            "hour_sin": test_extra["hour_sin"],
            "hour_cos": test_extra["hour_cos"],
            "is_weekend": test_extra["is_weekend"],
            "is_holiday": test_extra["is_holiday"],
            "weather_pressure_norm": test_extra["weather_pressure_norm"],
        })
        fc = m.predict(future)
        preds = fc["yhat"].values.astype(np.float32)
        return (cell_idx, preds)
    except Exception as e:
        last = float(train_y[-1]) if len(train_y) > 0 else 0.0
        return (cell_idx, np.full(len(test_ts), last, dtype=np.float32))


def run_prophet(flow_4d, weekly_seas=6, daily_seas=8, n_jobs=-1, use_regressors=True):
    """V2 Prophet — use_regressors=True 时接 5 个额外回归量"""
    from data_loader import load_feature_table, load_time_features
    df = load_feature_table()
    ts_raw = df["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    ts = pd.to_datetime(ts_raw)
    train_ts = ts.iloc[:cfg.SPLIT.train_end].values
    test_ts  = ts.iloc[cfg.SPLIT.val_end:cfg.SPLIT.test_end].values

    flow_total = (flow_4d[:, 0] + flow_4d[:, 1]).reshape(len(flow_4d), -1)
    train_y_all = flow_total[:cfg.SPLIT.train_end]
    N = flow_total.shape[1]
    n_test = cfg.SPLIT.test_end - cfg.SPLIT.val_end
    print(f"[Prophet V2] cells={N}, train={len(train_ts)}, test={n_test}, "
          f"use_regressors={use_regressors}")

    # V2: 准备 extra_regressors
    if use_regressors:
        tf = load_time_features()  # (T, 5)
        train_extra = {
            "hour_sin": tf[:cfg.SPLIT.train_end, 0],
            "hour_cos": tf[:cfg.SPLIT.train_end, 1],
            "is_weekend": tf[:cfg.SPLIT.train_end, 2],
            "is_holiday": tf[:cfg.SPLIT.train_end, 3],
            "weather_pressure_norm": tf[:cfg.SPLIT.train_end, 4],
        }
        test_extra = {
            "hour_sin": tf[cfg.SPLIT.val_end:cfg.SPLIT.test_end, 0],
            "hour_cos": tf[cfg.SPLIT.val_end:cfg.SPLIT.test_end, 1],
            "is_weekend": tf[cfg.SPLIT.val_end:cfg.SPLIT.test_end, 2],
            "is_holiday": tf[cfg.SPLIT.val_end:cfg.SPLIT.test_end, 3],
            "weather_pressure_norm": tf[cfg.SPLIT.val_end:cfg.SPLIT.test_end, 4],
        }
    else:
        train_extra = None
        test_extra = None

    start = time.time()
    args_list = [
        (i, train_ts, train_y_all[:, i].astype(np.float64),
         test_ts, test_extra, weekly_seas, daily_seas)
        for i in range(N)
    ]
    # train_extra 是共享的, 不能直接 pickle 单独传给 mp.Pool;
    # 我们把它 attach 到 test_extra 的姊妹位置
    if use_regressors:
        args_list = [
            (i, train_ts, train_y_all[:, i].astype(np.float64),
             test_ts, test_extra, weekly_seas, daily_seas, train_extra)
            for i in range(N)
        ]

    if n_jobs == -1:
        n_jobs = max(1, min(os.cpu_count() or 4, 12))
    if n_jobs == 1:
        results = [_prophet_one_cell_v2(a) for a in args_list]
    else:
        with mp.Pool(n_jobs) as pool:
            results = pool.map(_prophet_one_cell_v2, args_list)
    elapsed = time.time() - start
    print(f"[Prophet V2] done in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    pred_arr = np.zeros((n_test, N), dtype=np.float32)
    for idx, preds in results:
        pred_arr[:, idx] = preds
    gt = flow_total[cfg.SPLIT.val_end:cfg.SPLIT.test_end].astype(np.float32)
    return pred_arr, gt, test_ts


def _prophet_one_cell_v2(args):
    """V2: 支持 extra_regressors 的工作函数"""
    if len(args) == 8:
        cell_idx, train_ts, train_y, test_ts, test_extra, weekly_seas, daily_seas, train_extra = args
    else:
        cell_idx, train_ts, train_y, test_ts, test_extra, weekly_seas, daily_seas = args
        train_extra = None
    try:
        from prophet import Prophet
        train_df = pd.DataFrame({"ds": pd.to_datetime(train_ts), "y": train_y})
        if train_extra is not None:
            for col in ["hour_sin", "hour_cos", "is_weekend", "is_holiday", "weather_pressure_norm"]:
                train_df[col] = train_extra[col]

        m = Prophet(weekly_seasonality=weekly_seas, daily_seasonality=daily_seas,
                    yearly_seasonality=False, seasonality_mode="additive")
        if train_extra is not None:
            for col in ["hour_sin", "hour_cos", "is_weekend", "is_holiday", "weather_pressure_norm"]:
                m.add_regressor(col)
        m.fit(train_df)

        future = pd.DataFrame({"ds": pd.to_datetime(test_ts)})
        if test_extra is not None:
            for col in ["hour_sin", "hour_cos", "is_weekend", "is_holiday", "weather_pressure_norm"]:
                future[col] = test_extra[col]
        fc = m.predict(future)
        preds = fc["yhat"].values.astype(np.float32)
        return (cell_idx, preds)
    except Exception:
        last = float(train_y[-1]) if len(train_y) > 0 else 0.0
        return (cell_idx, np.full(len(test_ts), last, dtype=np.float32))