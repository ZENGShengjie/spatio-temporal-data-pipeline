# Week5 — Spatiotemporal Anomaly Detection
import os, sys

# 2026-07-28: Python310 (Streamlit-only env) 没有 torch；API 启动会失败。
# 这里给 import week5.* 的链路做防崩保护：
#   - config / data_loader / evaluation 用 try/except 兜底 torch
#   - anomaly 子模块的 torch 模型只在被调用时才加载，没 torch 就走 cache/history-mean baseline

try:
    from . import config
except Exception as _e:
    print(f"[week5 __init__] config import skipped: {_e}")

# data_loader 里有 torch.from_numpy（被预测器使用），但它本身只是数据类型
# 转换；可以用 numpy 替代。我们让 data_loader 在缺 torch 时给个 stub。
try:
    from . import data_loader as _dl
except Exception as _e:
    print(f"[week5 __init__] data_loader import skipped: {_e}")

try:
    from . import evaluation
except Exception as _e:
    print(f"[week5 __init__] evaluation import skipped: {_e}")