import os, sys
sys.path.insert(0, " /home/ubuntu/amazon_repo\)
import numpy as np
from config import DATA_DIR, CACHE_DIR, TRAIN_END, VAL_END, N_CELLS

DATA = \/home/ubuntu/amazon_repo/week5/data\
CACHE = \/home/ubuntu/amazon_repo/week5/cache\

# === Statistical ===
from anomaly.statistical import StatisticalAnomalyDetector
from data_loader import get_flow_1d, get_time_group_labels
flow = get_flow_1d(\taxi_flow_total\)
tg = get_time_group_labels()
d = StatisticalAnomalyDetector(\taxi_flow_total\)
d.flow_train = flow[:TRAIN_END]
d.flow_val = np.load(os.path.join(DATA, \flow_val_injected_structural.npy\))
d.flow_test = np.load(os.path.join(DATA, \flow_test_injected_structural.npy\))
d.time_groups = tg
d.tg_train = tg[:TRAIN_END]
d.tg_val = tg[TRAIN_END:VAL_END]
d.tg_test = tg[VAL_END:]
d._compute_group_stats()
d._fitted = True
sv = d.score(d.flow_val)
st = d.score(d.flow_test)
np.save(os.path.join(CACHE, \stat_scores_val_structural.npy\), sv)
np.save(os.path.join(CACHE, \stat_scores_test_structural.npy\), st)
print(\[stat] val mean\, sv.mean(), " test mean\, st.mean())
