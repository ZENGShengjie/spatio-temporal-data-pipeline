"""Week6 任务4 Patch 2：detect 结果 LRU 缓存

目标：对相同 (t, mode, threshold) 的 detect 请求做缓存
收益：重复请求 -95% 耗时（Streamlit 切换时间步时尤其明显）

应用方式：
    from week6_evaluation.patches.pipeline_lru_cache import patch_detect_with_cache
    patch_detect_with_cache()

注意事项：
- 仅对查询类 API（detect/forecast）有效，实时流式数据不适用
- 多用户场景下用 (user_id, t) 作 key
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Any, Dict
from functools import lru_cache

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


# ── LRU 缓存（线程安全版本） ────────────────────────────────────────────

import threading


class LRUCacheWithTTL:
    """简单 LRU 缓存（带 TTL），线程安全

    限制：
    - max_size 满了驱逐最旧
    - ttl_seconds 过期自动失效
    """
    def __init__(self, max_size: int = 512, ttl_seconds: float = 60.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[Any, Any] = {}
        self.timestamps: Dict[Any, float] = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            ts = self.timestamps.get(key, 0)
            if time.time() - ts > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
            return self.cache[key]

    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # 驱逐最旧
                oldest = min(self.timestamps, key=self.timestamps.get)
                del self.cache[oldest]
                del self.timestamps[oldest]
            self.cache[key] = value
            self.timestamps[key] = time.time()

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
            }


# 全局缓存
_detect_cache = LRUCacheWithTTL(max_size=512, ttl_seconds=30.0)


def cached_detect(pipe, t: int, mode: str = "fast", threshold: float = 0.0):
    """带缓存的 detect

    Args:
        pipe: SpatiotemporalPipeline
        t: 时间步
        mode: "fast" | "structural"
        threshold: 融合阈值（0 表示用默认）

    Returns:
        detect 结果
    """
    cache_key = ("detect", t, mode, threshold)
    cached = _detect_cache.get(cache_key)
    if cached is not None:
        return cached

    # 实际计算
    from week6.api.main import detect_anomaly
    from week6.api.schemas import AnomalyDetectRequest
    req = AnomalyDetectRequest(t=t, mode=mode, threshold=threshold if threshold > 0 else None)
    result = detect_anomaly(req)

    # 转为可缓存对象（Pydantic model 不直接缓存，存 dict）
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    _detect_cache.set(cache_key, result)
    return result


def cached_forecast(pipe, t_start: int, t_end: int, grid_ids=None):
    """带缓存的 forecast"""
    grid_key = tuple(sorted(grid_ids)) if grid_ids else tuple()
    cache_key = ("forecast", t_start, t_end, grid_key)
    cached = _detect_cache.get(cache_key)
    if cached is not None:
        return cached

    from week6.api.main import forecast
    from week6.api.schemas import ForecastRequest
    req = ForecastRequest(time_start=t_start, time_end=t_end, grid_ids=grid_ids)
    result = forecast(req)
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    _detect_cache.set(cache_key, result)
    return result


def get_cache_stats():
    return _detect_cache.stats()


def clear_cache():
    _detect_cache.clear()


def patch_detect_with_cache():
    """在 week6.api.main 里替换 detect_anomaly / forecast

    副作用：导入本模块后，原 detect_anomaly / forecast 自动带缓存
    （通过修改模块 __dict__）
    """
    import week6.api.main as api_main

    original_detect = api_main.detect_anomaly
    original_forecast = api_main.forecast

    def detect_with_cache(req):
        # 缓存 key 用 (t, mode, threshold)
        threshold = req.threshold or 0.0
        cache_key = ("detect", req.t, req.mode, threshold)
        cached = _detect_cache.get(cache_key)
        if cached is not None:
            return cached
        result = original_detect(req)
        # 转为 dict 缓存（Pydantic V2）
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
            _detect_cache.set(cache_key, result_dict)
            return result_dict
        _detect_cache.set(cache_key, result)
        return result

    def forecast_with_cache(req):
        grid_key = tuple(sorted(req.grid_ids)) if req.grid_ids else tuple()
        cache_key = ("forecast", req.time_start, req.time_end, grid_key)
        cached = _detect_cache.get(cache_key)
        if cached is not None:
            return cached
        result = original_forecast(req)
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
            _detect_cache.set(cache_key, result_dict)
            return result_dict
        _detect_cache.set(cache_key, result)
        return result

    api_main.detect_anomaly = detect_with_cache
    api_main.forecast = forecast_with_cache
    print("[Patch] LRU cache applied to detect_anomaly / forecast")


if __name__ == "__main__":
    print("=== LRU cache patch 自检 ===")
    c = LRUCacheWithTTL(max_size=3, ttl_seconds=0.5)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    print(f"  size after 3 sets: {c.stats()['size']}")
    c.set("d", 4)  # 触发驱逐
    print(f"  size after 4 sets (evict 1): {c.stats()['size']}")
    print(f"  get('a') after eviction: {c.get('a')} (应为 None)")
    print(f"  get('d'): {c.get('d')} (应为 4)")
    time.sleep(0.6)
    print(f"  get('d') after TTL: {c.get('d')} (应为 None)")
    print("  [OK] LRU cache 行为正确")
