"""Prophet runner wrapper V2"""
import time
from metrics import BaseTrainer
from models.prophet_model import run_prophet


class ProphetBaseTrainer(BaseTrainer):
    name = "prophet"

    def __init__(self, weekly_seas=6, daily_seas=8, n_jobs=-1, use_regressors=True):
        self.weekly_seas = weekly_seas
        self.daily_seas = daily_seas
        self.n_jobs = n_jobs
        self.use_regressors = use_regressors

    def fit_predict(self, flow_4d, time_features=None, target="taxi_flow_total", **kwargs):
        t0 = time.time()
        pred, gt, test_ts = run_prophet(flow_4d, weekly_seas=self.weekly_seas,
                                         daily_seas=self.daily_seas,
                                         n_jobs=self.n_jobs,
                                         use_regressors=self.use_regressors)
        print(f"[Prophet runner V2] pred={pred.shape} train_t={time.time()-t0:.1f}s "
              f"use_regressors={self.use_regressors}")
        return pred, gt