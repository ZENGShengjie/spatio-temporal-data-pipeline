"""Week6 性能打点器 — fast / structural 两种模式分别测

记录：
- 端到端延迟（API 请求处理时间）
- 内存峰值（psutil）
- GPU 显存峰值（torch.cuda，需 GPU）
- 单步推理延迟（pipeline.run_step 调用耗时）

结果输出 JSON，可被 evaluate.py 汇总。
"""
from __future__ import annotations
import time
import json
import gc
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class StepProfile:
    """单步推理的性能采样"""
    t: int                          # 时间步
    latency_ms: float               # 端到端耗时
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunProfile:
    """一次完整 profile 的汇总"""
    mode: str                       # "fast" | "structural"
    n_steps: int
    total_latency_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    peak_ram_mb: Optional[float] = None
    peak_gpu_mb: Optional[float] = None
    steps: List[StepProfile] = field(default_factory=list)


def _percentile(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(arr, p))


def get_ram_mb() -> Optional[float]:
    """当前进程 RSS"""
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    except ImportError:
        return None


def get_gpu_mb() -> Optional[float]:
    """当前 GPU 已分配显存（MB）"""
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / 1024 / 1024)
    except Exception:
        pass
    return None


def profile_pipeline(
    pipeline,
    t_start: int,
    t_end: int,
    mode: str = "fast",
    device: str = "cpu",
) -> RunProfile:
    """对 pipeline 在 [t_start, t_end) 区间逐步 profile

    Args:
        pipeline: SpatiotemporalPipeline 实例
        t_start: 起始时间步（含）
        t_end: 结束时间步（不含）
        mode: "fast" | "structural"
        device: "cpu" | "cuda"

    Returns:
        RunProfile
    """
    del device  # 暂未实现 gpu 单独分支

    latencies: List[float] = []
    steps: List[StepProfile] = []
    peak_ram = get_ram_mb() or 0.0
    peak_gpu = get_gpu_mb() or 0.0

    gc.collect()
    if get_gpu_mb() is not None:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    t_total_start = time.perf_counter()

    for t in range(t_start, t_end):
        t0 = time.perf_counter()
        # 模拟实时：单步推理
        result = pipeline.run_step()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        steps.append(StepProfile(
            t=int(result.get("t", t)),
            latency_ms=elapsed_ms,
            extra={"alert_level": result.get("alert").level if result.get("alert") else None},
        ))

        # 监控峰值
        cur_ram = get_ram_mb()
        if cur_ram and cur_ram > peak_ram:
            peak_ram = cur_ram
        cur_gpu = get_gpu_mb()
        if cur_gpu and cur_gpu > peak_gpu:
            peak_gpu = cur_gpu

    total_ms = (time.perf_counter() - t_total_start) * 1000

    return RunProfile(
        mode=mode,
        n_steps=len(latencies),
        total_latency_ms=total_ms,
        avg_latency_ms=float(np.mean(latencies)),
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
        peak_ram_mb=peak_ram,
        peak_gpu_mb=peak_gpu if peak_gpu > 0 else None,
        steps=steps,
    )


def profile_api_endpoint(
    endpoint_url: str,
    payload: Dict,
    n_requests: int = 10,
    method: str = "POST",
) -> Dict[str, float]:
    """对 API 端点做请求级 profile（端到端，含网络）

    Args:
        endpoint_url: 完整 URL
        payload: 请求体（POST）或 query（GET）
        n_requests: 重复请求次数
        method: "POST" | "GET"

    Returns:
        延迟统计
    """
    try:
        import requests
    except ImportError:
        return {"error": "requests not installed"}

    latencies: List[float] = []
    status_codes: List[int] = []

    for _ in range(n_requests):
        t0 = time.perf_counter()
        try:
            if method == "POST":
                r = requests.post(endpoint_url, json=payload, timeout=30)
            else:
                r = requests.get(endpoint_url, params=payload, timeout=30)
            status_codes.append(r.status_code)
        except Exception as e:
            status_codes.append(-1)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    return {
        "endpoint": endpoint_url,
        "n_requests": n_requests,
        "avg_ms": float(np.mean(latencies)),
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "success_rate": sum(1 for c in status_codes if 200 <= c < 300) / n_requests,
    }


def profile_batch(
    pipeline,
    split: str = "test",
    mode: str = "fast",
) -> Dict[str, float]:
    """测批量处理吞吐量

    Returns:
        耗时、吞吐量
    """
    gc.collect()
    pipeline._warning_engine.reset()
    t0 = time.perf_counter()
    result = pipeline.run_batch(split=split)
    elapsed = (time.perf_counter() - t0) * 1000

    T = len(result["flow"])
    return {
        "mode": mode,
        "split": split,
        "total_ms": elapsed,
        "throughput_steps_per_sec": T / (elapsed / 1000),
        "n_steps": T,
        "n_anomalies": int(result["anomaly_mask"].sum()),
        "n_events": len(result["events"]),
        "n_alerts": len(result["alerts"]),
    }


def save_profile(profile: RunProfile, path: str):
    """保存 profile 到 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, ensure_ascii=False, indent=2, default=str)


def save_api_profile(stats: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 自检：模块自身能 import + 跑通基本计算
    print("=== profile_runner.py 自检 ===")
    fake_latencies = [10.0, 12.0, 15.0, 100.0, 11.0, 13.0, 14.0, 200.0, 12.0, 11.0]
    print(f"p50: {_percentile(fake_latencies, 50):.2f} ms")
    print(f"p95: {_percentile(fake_latencies, 95):.2f} ms")
    print(f"p99: {_percentile(fake_latencies, 99):.2f} ms")
    print(f"avg: {np.mean(fake_latencies):.2f} ms")
    print(f"RAM: {get_ram_mb()} MB")
    print(f"GPU: {get_gpu_mb()} MB")
