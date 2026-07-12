"""ARIMA runner — 包装我们写在 models/arima_model.py 的逻辑
通过 BaseTrainer 接口对外"""
import time
import numpy as np

from metrics import BaseTrainer
from models.arima_model import run_arima


class ARIMABaseTrainer(BaseTrainer):
    name = "arima"

    def __init__(self, order=(1, 0, 1), n_jobs=-1):
        self.order = order
        self.n_jobs = n_jobs

    def fit_predict(self, flow_4d, time_features=None, target="taxi_flow_total",
                    **kwargs):
        # 只在 flow_total 上跑 ARIMA
        t0 = time.time()
        pred, gt = run_arima(flow_4d, order=self.order, n_jobs=self.n_jobs)
        train_t = time.time() - t0
        test_t  = 0.0
        print(f"[ARIMA runner] pred={pred.shape} train_t={train_t:.1f}s")
        return pred, gt
