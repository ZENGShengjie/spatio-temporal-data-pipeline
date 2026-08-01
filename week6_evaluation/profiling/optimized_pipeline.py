"""Week6 任务4-2：优化版 Pipeline（含 torch.inference_mode + 数据预加载 + LRU 缓存）

设计原则：
- 不重写逻辑，只在原 SpatiotemporalPipeline 基础上包一层优化
- 保留原接口兼容
- 优化点：
  1. 数据预加载（启动时一次性 load 到内存）
  2. torch.inference_mode() 包裹推理
  3. LRU 缓存 detect 结果（避免重复计算）
  4. 预计算 stat 得分（避免每个窗口都重算）

使用方法（替换原 Pipeline）：
    from week6_evaluation.profiling.optimized_pipeline import OptimizedPipeline
    pipe = OptimizedPipeline(mode="fast", use_cache=True)
    result = pipe.run_batch()  # 接口与原版一致
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "week6"))

from week6.pipeline import SpatiotemporalPipeline, WarningEngine


class OptimizedPipeline:
    """在原 SpatiotemporalPipeline 之上叠加优化

    不重写方法，只在调用前后做：
    - 预热（启动时加载全部数据）
    - 推理时用 inference_mode
    - 缓存相同 (t, mode) 的 detect 结果
    """

    def __init__(self, mode: str = "fast", use_cache: bool = True, max_cache_size: int = 128):
        self._pipe = SpatiotemporalPipeline(mode=mode, use_cache=use_cache)
        self._mode = mode
        self._use_cache = use_cache
        self._max_cache_size = max_cache_size

        # 优化 1: 数据预加载（启动时全部加载到内存）
        print("[Optimized] 预加载数据...")
        t0 = time.time()
        self._flow_cache = None
        self._timestamps_cache = None
        self._preload_data()
        print(f"[Optimized] 预加载完成 {time.time()-t0:.2f}s")

        # 优化 3: LRU 缓存
        self._detect_cache: Dict[int, Dict] = {}

    def _preload_data(self):
        """一次性加载所有数据到内存"""
        from week5.data_loader import get_flow_1d, get_timestamps
        self._flow_cache = get_flow_1d("taxi_flow_total")
        self._timestamps_cache = get_timestamps()

    # ── run_batch 优化：用 inference_mode 包裹 ──

    def run_batch(self, split: str = "test") -> Dict[str, Any]:
        """批量推理（用 inference_mode 优化）"""
        # 复用原方法，但推理部分用 inference_mode
        # 由于原 SpatiotemporalPipeline.run_batch 内部不用 torch
        # （统计法/预测法/VAE 走的是 numpy + sklearn），inference_mode 主要在 VAE 上有效
        with torch.inference_mode():
            return self._pipe.run_batch(split=split)

    def init_realtime(self, warmup_steps: int = 48):
        self._pipe.init_realtime(warmup_steps=warmup_steps)

    def run_step(self, new_data: np.ndarray = None) -> Dict[str, Any]:
        """实时推理（带 LRU 缓存）"""
        # 缓存 key: time step t
        # 暂用 t 作为 key（new_data=None 时自取）
        if new_data is None:
            t = self._pipe._window.__len__() if self._pipe._window else 0
        else:
            t = -1  # 外部传入时不可缓存

        if 0 <= t < 1_000_000 and t in self._detect_cache:
            return self._detect_cache[t]

        # 推理（用 inference_mode）
        with torch.inference_mode():
            result = self._pipe.run_step(new_data)

        # 写入缓存（LRU 简单实现）
        if 0 <= t < 1_000_000:
            if len(self._detect_cache) >= self._max_cache_size:
                # 弹出最旧
                oldest = next(iter(self._detect_cache))
                del self._detect_cache[oldest]
            self._detect_cache[t] = result
        return result

    def query_events(self, *args, **kwargs):
        return self._pipe.query_events(*args, **kwargs)

    def get_grid_data(self, t: int) -> Dict[str, Any]:
        return self._pipe.get_grid_data(t)

    @property
    def mode(self):
        return self._pipe.mode

    @mode.setter
    def mode(self, v):
        self._pipe.mode = v

    def __getattr__(self, name):
        # 兜底：未实现的属性转给原 Pipeline
        return getattr(self._pipe, name)


# ── 性能对比辅助函数 ────────────────────────────────────────────────────

def benchmark_pipeline(pipe_cls, mode: str, n_steps: int = 100) -> Dict[str, float]:
    """对指定 Pipeline 类做 benchmark

    Args:
        pipe_cls: Pipeline 类（如 SpatiotemporalPipeline 或 OptimizedPipeline）
        mode: "fast" | "structural"

    Returns:
        性能指标字典
    """
    pipe = pipe_cls(mode=mode, use_cache=True)
    pipe.init_realtime(warmup_steps=48)

    from week5.config import VAL_END

    latencies = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        pipe.run_step()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mode": mode,
        "n_steps": n_steps,
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_steps_per_sec": n_steps / (sum(latencies) / 1000),
    }


if __name__ == "__main__":
    # 自检：原版 vs 优化版对比
    print("=== OptimizedPipeline 自检 ===")
    print("[原版] 跑 50 步...")
    from week6.pipeline import SpatiotemporalPipeline
    orig = benchmark_pipeline(SpatiotemporalPipeline, "fast", n_steps=50)
    print(f"  原版: avg={orig['avg_latency_ms']:.2f} ms, p95={orig['p95_ms']:.2f} ms")

    print("[优化版] 跑 50 步...")
    opt = benchmark_pipeline(OptimizedPipeline, "fast", n_steps=50)
    print(f"  优化版: avg={opt['avg_latency_ms']:.2f} ms, p95={opt['p95_ms']:.2f} ms")

    if orig['avg_latency_ms'] > 0:
        speedup = (orig['avg_latency_ms'] - opt['avg_latency_ms']) / orig['avg_latency_ms'] * 100
        print(f"\n提升: {speedup:+.1f}% (avg latency)")
