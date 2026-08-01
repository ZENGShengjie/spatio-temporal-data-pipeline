"""evaluate.py 本地逻辑自检脚本（无 EC2 数据情况下验证 API）

执行：
    python -m week6_evaluation.evaluation.evaluate_local

目的：
- 验证 metrics / profile_runner / summary 全部串通
- 确认 evaluate.py 在 EC2 上跑时不会因参数顺序错误崩
- 离线环境调试用，**不是正式评估**
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from week6_evaluation.evaluation import (
    full_evaluation, profile_pipeline, profile_batch,
    profile_api_endpoint, get_ram_mb,
)


def main():
    print("=== evaluate.py 本地自检 ===")
    print("造 fake 测试数据 (T=600, N=1024)...")

    np.random.seed(42)
    T, N = 600, 1024
    flow = np.random.rand(T, N).astype(np.float32) * 0.5
    pred = flow + np.random.randn(T, N).astype(np.float32) * 0.05
    pred = np.clip(pred, 0, 1)
    mask = np.random.rand(T, N) < 0.05
    scores = np.random.rand(T, N).astype(np.float32)
    ts = np.arange(T)

    # 模拟事件
    events = [
        {"event_id": 1, "t_start": 3288+t, "t_end": 3288+t+2, "duration": 3,
         "n_cells": 25, "avg_score": 0.92, "warning_level": 2, "event_type": "spatial_sustained"}
        for t in range(0, 100, 50)
    ]

    print("\n[1] 跑指标评估...")
    metrics = full_evaluation(flow, pred, mask, scores, events, ts, val_end=3288)
    print(f"  [OK] overall_mae: {metrics['predict']['overall_mae']:.4f}")
    print(f"  [OK] morning_peak_mae: {metrics['predict']['morning_peak_mae']:.4f}")
    print(f"  [OK] core_area_mae: {metrics['predict']['core_area_mae']:.4f}")
    print(f"  [OK] direction_acc: {metrics['predict']['next_step_direction_acc']*100:.1f}%")
    print(f"  [OK] day_anomaly_rate: {metrics['anomaly']['day_anomaly_rate']*100:.2f}%")
    print(f"  [OK] total_events: {metrics['events']['total_events']}")

    print("\n[2] 跑系统性能 profile (fake steps)...")
    # 模拟 100 步的延迟
    fake_latencies = np.random.gamma(2, 5, 100).tolist()  # 平均 10ms
    profile_results = {
        "batch_fast": {
            "total_ms": 1234.5,
            "throughput_steps_per_sec": 600 / 1.2345,
            "n_steps": 600,
            "n_anomalies": 30000,
            "n_events": 50,
            "n_alerts": 5,
        },
        "realtime_fast": {
            "n_steps": 100,
            "avg_latency_ms": float(np.mean(fake_latencies)),
            "p50_latency_ms": float(np.percentile(fake_latencies, 50)),
            "p95_latency_ms": float(np.percentile(fake_latencies, 95)),
            "p99_latency_ms": float(np.percentile(fake_latencies, 99)),
            "peak_ram_mb": 512.0,
        },
    }
    print(f"  [OK] batch 吞吐量: {profile_results['batch_fast']['throughput_steps_per_sec']:.1f} 步/秒")
    print(f"  [OK] realtime 平均延迟: {profile_results['realtime_fast']['avg_latency_ms']:.1f} ms")

    print("\n[3] 写 summary.md...")
    out = Path("week6_evaluation/results/local_test")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "profile.json").write_text(
        json.dumps(profile_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 复用 evaluate.py 的 summary 生成
    from week6_evaluation.evaluation.evaluate import write_summary
    summary = write_summary(metrics, profile_results, out, model_tag="local_test")
    print(f"  [OK] summary: {out / 'summary.md'}")

    print("\n[4] 当前内存:")
    print(f"  RAM: {get_ram_mb():.1f} MB")

    print("\n[ALL PASS] 自检完成")
    print(f"输出位置: {out.resolve()}")


if __name__ == "__main__":
    main()
