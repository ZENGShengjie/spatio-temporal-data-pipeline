"""Week5 V2 Smoke Test — 上传到 EC2 后运行此脚本"""
import os, sys, json, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
errors = []

print("=== Week5 V2 Smoke Test ===")
print(f"Python: {sys.version}")

# 1. Config
try:
    import config
    print(f"[OK] config: anomaly_ratio={config.INJECT_CFG.anomaly_ratio}")
    print(f"     VAL_HOURS={config.VAL_HOURS}, TEST_HOURS={config.TEST_HOURS}")
    print(f"     dropout={config.REC_CFG.dropout}, patience={config.REC_CFG.patience}")
    print(f"     REC device={config.REC_CFG.device}")
except Exception as e:
    errors.append(f"config FAILED: {e}")
    traceback.print_exc()

# 2. Data loader
try:
    from data_loader import get_flow_1d, get_time_features
    flow = get_flow_1d()
    print(f"[OK] data_loader: flow shape={flow.shape}, "
          f"val shape={flow[config.TRAIN_END:config.VAL_END].shape}, "
          f"test shape={flow[config.VAL_END:].shape}")
except Exception as e:
    errors.append(f"data_loader FAILED: {e}")
    traceback.print_exc()

# 3. Check V2 injection files
try:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    files_to_check = [
        "anomaly_labels_val.npy",
        "anomaly_labels_test.npy",
        "flow_val_injected.npy",
        "flow_test_injected.npy",
        "flow_test_clean.npy",
    ]
    for f in files_to_check:
        path = os.path.join(data_dir, f)
        exists = os.path.exists(path)
        size = os.path.getsize(path) // 1024 if exists else 0
        print(f"[{'OK' if exists else 'MISS'}] {f} ({size}KB)")
        if not exists:
            print(f"    (expected — run inject_anomalies.py first to generate)")
except Exception as e:
    errors.append(f"injection files check FAILED: {e}")
    traceback.print_exc()

# 4. Statistical detector
try:
    from anomaly.statistical import StatisticalAnomalyDetector
    det = StatisticalAnomalyDetector(target="taxi_flow_total")
    print("[OK] StatisticalAnomalyDetector instantiated")
except Exception as e:
    errors.append(f"statistical FAILED: {e}")
    traceback.print_exc()

# 5. Prediction detector
try:
    from anomaly.prediction import PredictionAnomalyDetector
    det = PredictionAnomalyDetector(target="taxi_flow_total")
    print("[OK] PredictionAnomalyDetector instantiated")
except Exception as e:
    errors.append(f"prediction FAILED: {e}")
    traceback.print_exc()

# 6. Fusion
try:
    from anomaly.fusion import AnomalyFusion
    fus = AnomalyFusion(target="taxi_flow_total")
    print("[OK] AnomalyFusion instantiated")
except Exception as e:
    errors.append(f"fusion FAILED: {e}")
    traceback.print_exc()

# 7. Transformer AE (check model)
try:
    import torch
    from anomaly.transformer_ae import TemporalAttentionAE
    model = TemporalAttentionAE(
        seq_len=48, d_model=64, num_heads=4, dropout=0.15
    )
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 48)
    with torch.no_grad():
        out, mask, _ = model(x, mask_ratio=0.25, return_masked=True)
    print(f"[OK] TAE model: {n_params} params, output shape={out.shape}, mask sum={mask.sum()}")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    errors.append(f"TAE model FAILED: {e}")
    traceback.print_exc()

# 8. Evaluation metrics
try:
    from evaluation.metrics import compute_point_metrics, compute_event_metrics
    import numpy as np
    pred = np.zeros((10, 1024), dtype=bool)
    gt = np.zeros((10, 1024), dtype=bool)
    gt[5, 100] = True
    m = compute_point_metrics(pred, gt)
    print(f"[OK] metrics: P={m.precision:.4f}, R={m.recall:.4f}")
except Exception as e:
    errors.append(f"metrics FAILED: {e}")
    traceback.print_exc()

print("\n=== Results ===")
if errors:
    print("ERRORS FOUND:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL smoke tests PASSED!")
    sys.exit(0)
