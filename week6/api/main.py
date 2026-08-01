"""Week6 FastAPI 推理服务

接口说明：
  - GET  /api/health            健康检查
  - POST /api/forecast          区域人流预测
  - POST /api/anomaly/detect    单批次异常检测
  - GET  /api/anomaly/events    历史异常事件查询

默认快速模式：仅用统计法 + 预测法加权融合（0.9:0.1），不加载深度学习模型，
保证响应时间 < 1秒。

结构增强模式：通过请求参数 mode=structural 触发，懒加载 TAE/VAE，
执行全模型融合，针对结构性异常。
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    # 2026-07-28: Python310 装 torch 太慢；让 API 在无 torch 时仍可启动，
    # /api/health 里 cuda_available=False 即可。
    torch = None  # type: ignore

# ── 路径 ──────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Pipeline
from week6.pipeline import SpatiotemporalPipeline, WarningEngine
from week6.api.schemas import (
    ForecastRequest, ForecastResponse, CellForecast,
    AnomalyDetectRequest, AnomalyDetectResponse, AnomalyScore, AnomalyCell,
    EventsQueryRequest, EventsQueryResponse, AnomalyEvent,
    HealthResponse,
    TimeslotsRequest, TimeslotsResponse,
)

# ── 性能优化 patches（Week7 任务4 配套）──────────────────────────────────
# 启用: torch.inference_mode + LRU 缓存
# LRU 缓存只对相同 (t, mode, threshold) 的请求生效，跨请求、跨用户均生效
try:
    from week6.evaluation.patches.pipeline_inference_mode import patch_pipeline
    patch_pipeline()
except Exception as _e:
    print(f"[week6.api] WARN: inference_mode patch failed: {_e}")
try:
    from week6.evaluation.patches.pipeline_lru_cache import patch_detect_with_cache
    patch_detect_with_cache()
except Exception as _e:
    print(f"[week6.api] WARN: lru_cache patch failed: {_e}")

# ── 全局状态 ─────────────────────────────────────────────────────────────────
_pipeline: Optional[SpatiotemporalPipeline] = None
_warning_engine = WarningEngine()
_start_time = time.time()
_cache_loaded = False


def _get_pipeline() -> SpatiotemporalPipeline:
    if _pipeline is None:
        raise HTTPException(503, "Pipeline 未初始化，请等待启动完成")
    return _pipeline


# ── 启动/关闭生命周期 ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 启动时预热 Pipeline"""
    global _pipeline, _cache_loaded

    print("[API] 正在启动 Pipeline...")
    t0 = time.time()

    # 只预加载统计法和预测法（快速模式）
    try:
        _pipeline = SpatiotemporalPipeline(mode="fast", use_cache=True)
        # 提前触发数据加载
        _pipeline._ensure_stat()
        _pipeline._ensure_pred()
        _pipeline.run_batch(split="test")
        _cache_loaded = True
        print(f"[API] Pipeline 就绪，耗时 {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[API] WARNING: Pipeline 启动失败: {e}")
        _pipeline = SpatiotemporalPipeline(mode="fast", use_cache=False)

    yield

    print("[API] 关闭...")


# ── FastAPI 应用 ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="城市人流时空异常检测 API",
    description="""
## 系统概述

基于北京出租车 GPS 数据（32×32 网格，2015年7月-10月）的人流异常检测推理服务。

## 检测模式

- **快速模式 (fast)**：仅使用统计法 + 预测法加权融合（stat 0.9, pred 0.1），
  不加载任何深度学习重构模型，保证响应速度 < 1秒。
- **结构增强模式 (structural)**：通过请求参数 `mode=structural` 触发，
  懒加载 VAE/TAE 模型，执行全方法融合，专门针对连片区域、时序畸变类异常。

## 异常类型

- **突增 (surge)**：实际人流远高于预测
- **突降 (drop)**：实际人流远低于预测
- **持续型 (sustained)**：连续多个时间步的连片区域异常

## 预警等级

| 等级 | 名称 | 判定规则 |
|------|------|----------|
| 紧急 | 紧急 | 同连通片 ≥20格 且 该连通片连续 ≥3 步异常 |
| 重要 | 重要 | 连通片 ≥20格 |
| 一般 | 一般 | 连通片 ≥16格  或  任意 4×4 窗口内 ≥10/16 格密集异常 |
| 无预警 | - | 其他情况 |
""",
    version="1.0.0",
    lifespan=lifespan,
)

# 跨域（Streamlit 需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 接口实现 ─────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """健康检查接口 — 检查系统状态"""
    from week5.data_loader import get_period
    try:
        from week5.data_loader import get_period_info, get_splits
        periods = list(get_period_info().keys())
        _, val_end, test_end = get_splits()
    except Exception:
        periods = ["P4", "BJ13", "BJ14", "BJ15", "BJ16"]
        val_end, test_end = 3288, 3888
    return HealthResponse(
        status="ok" if _cache_loaded else "warming_up",
        cuda_available=(torch is not None and torch.cuda.is_available()),
        pipeline_mode=_pipeline.mode if _pipeline else "not_initialized",
        cache_loaded=_cache_loaded,
        period=get_period() if _cache_loaded else "P4",
        t_min=val_end,
        t_max=test_end,
        available_periods=periods,
    )


@app.post("/api/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    区域人流预测/历史回放接口

    返回指定时间范围内各网格的人流量序列。
    - 若 time_end 落在测试集区间：返回真实观测值（历史回放）
    - 若超出测试集：基于最后观测 + 同期均值外推（模拟预测）
    """
    t0 = time.time()
    pipe = _get_pipeline()

    try:
        from week5.data_loader import get_splits, get_period
        train_end, val_end, test_end = get_splits()
        period = get_period()
    except Exception:
        train_end, val_end, test_end = 2784, 3288, 3888
        period = "P4"

    try:
        cached = pipe._result_cache.get("flow")
        flow = cached if cached is not None else pipe.run_batch(split="test")["flow"]
    except Exception as e:
        raise HTTPException(500, f"数据加载失败: {e}")

    t_start = req.time_start
    t_end = min(req.time_end, test_end - 1)
    grid_ids = req.grid_ids or list(range(1024))

    horizon = t_end - t_start + 1
    if horizon <= 0 or horizon > 168:
        raise HTTPException(400, f"预测步数应在 1~168 之间，当前为 {horizon}")

    # 取时间戳（用于动画播放）
    from week5.data_loader import get_timestamps
    ts_all = get_timestamps()
    timestamps_iso = []
    for t in range(t_start, t_end + 1):
        t_local = t - val_end
        if 0 <= t_local < len(ts_all):
            try:
                timestamps_iso.append(str(np.datetime_as_string(ts_all[t_local], unit="m")))
            except Exception:
                timestamps_iso.append(str(ts_all[t_local]))
        else:
            timestamps_iso.append("")

    cells = []
    for gid in grid_ids:
        if gid < 0 or gid >= 1024:
            continue
        values = []
        for t in range(t_start, t_end + 1):
            t_local = t - val_end
            if 0 <= t_local < len(flow):
                values.append(float(flow[t_local, gid]))
            else:
                values.append(0.0)
        cells.append(CellForecast(grid_id=gid, values=values))

    # 当前 end 时刻的热力图
    t_local_end = t_end - val_end
    if 0 <= t_local_end < len(flow):
        heatmap = flow[t_local_end].reshape(GRID_H, GRID_W).tolist()
    else:
        heatmap = np.zeros((GRID_H, GRID_W)).tolist()

    elapsed_ms = (time.time() - t0) * 1000
    return ForecastResponse(
        time_start=t_start,
        time_end=t_end,
        horizon=horizon,
        cells=cells,
        heatmap=heatmap,
        timestamps=timestamps_iso,
        note=f"[{period}] horizon={horizon}，归一化人流量，耗时 {elapsed_ms:.0f}ms",
    )


# ── LRU 缓存（API 路径专用）──────────────────────────────────────────────
# 缓存 key: (mode, data_hash) → result dict
# 同一 mode + 同一 data 不重算 run_batch（批 600 步是耗时大头）
import hashlib
_RUN_BATCH_CACHE: Dict[str, Any] = {}
_RUN_BATCH_CACHE_MAX = 4  # 至多缓存 4 个 batch


def _run_batch_cached(pipe, mode: str, data_signature: str):
    """缓存 run_batch 结果：相同 mode + data 直接返回"""
    key = f"{mode}:{data_signature}"
    if key in _RUN_BATCH_CACHE:
        return _RUN_BATCH_CACHE[key], True  # (result, hit)
    result = pipe.run_batch(split="test")
    # LRU evict
    if len(_RUN_BATCH_CACHE) >= _RUN_BATCH_CACHE_MAX:
        oldest = next(iter(_RUN_BATCH_CACHE))
        del _RUN_BATCH_CACHE[oldest]
    _RUN_BATCH_CACHE[key] = result
    return result, False


@app.post("/api/anomaly/detect", response_model=AnomalyDetectResponse)
def detect_anomaly(req: AnomalyDetectRequest):
    """
    单时间步异常检测接口

    返回指定时间步的：
    - heatmap: 32×32 人流热力图
    - anomaly_mask: 32×32 异常标记矩阵
    - cells: 每网格详情（含经纬度，用于地图渲染）
    - 预警等级（基于该时间步异常结果实时判定）

    - **fast 模式**：仅用统计法 + 预测法融合（权重 0.9:0.1），不加载 DL 模型
    - **structural 模式**：懒加载 VAE/TAE，执行全量融合，针对连片异常

    **缓存策略**：同一 (mode, data) 的 batch 计算结果缓存在 _RUN_BATCH_CACHE，
    重复请求同一时间步 t 时跳过 run_batch（约 1s → <10ms）。
    """
    t0 = time.time()
    pipe = _get_pipeline()

    if req.data is not None:
        try:
            data = np.array(req.data, dtype=np.float32)
            T, N = data.shape
            data_signature = f"user:{T}x{N}"
        except Exception as e:
            raise HTTPException(400, f"数据格式错误: {e}")
    else:
        # 默认数据：signature 仅含 mode（不同 mode 不同结果）
        data_signature = "default"

    if req.mode == "structural":
        global _pipeline
        if _pipeline.mode != "structural":
            print("[API] 切换到 structural 模式，加载深度学习模型...")
            _pipeline.mode = "structural"

    try:
        result, cache_hit = _run_batch_cached(pipe, req.mode, data_signature)
        if cache_hit:
            print(f"[API] /detect cache HIT (mode={req.mode}, sig={data_signature})")
    except Exception as e:
        raise HTTPException(500, f"Pipeline 运行失败: {e}")

    fused = result["scores"]["fused"]
    mask = result["anomaly_mask"]
    flow = result["flow"]
    timestamps = result.get("timestamps")

    # 动态获取当前 period 的 split 边界
    try:
        from week5.data_loader import get_splits, get_period
        train_end, val_end, test_end = get_splits()
        period = get_period()
    except Exception:
        train_end, val_end, test_end = 2784, 3288, 3888
        period = "P4"

    T_steps = len(flow)
    if req.t is not None:
        t_global = req.t
        t_local = t_global - val_end
        if t_local < 0 or t_local >= T_steps:
            raise HTTPException(
                400,
                f"t={t_global} 超出测试集范围 [{val_end}, {val_end + T_steps - 1}]",
            )
    else:
        t_local = T_steps - 1
        t_global = val_end + t_local

    last_scores = fused[t_local]
    last_mask = mask[t_local]
    last_flow = flow[t_local]
    N = len(last_mask)
    threshold = req.threshold or pipe.fusion_threshold

    scores_out = []
    cells_out = []
    for gid in range(N):
        lon, lat = _GRID_CELLS[gid]
        r, c = gid // GRID_W, gid % GRID_W
        is_anom = bool(last_mask[gid])
        scores_out.append(AnomalyScore(
            grid_id=gid, score=float(last_scores[gid]), is_anomaly=is_anom,
        ))
        cells_out.append(AnomalyCell(
            grid_id=gid, row=r, col=c,
            lon=round(lon, 6), lat=round(lat, 6),
            flow=float(last_flow[gid]),
            score=float(last_scores[gid]),
            is_anomaly=is_anom,
        ))

    # 预警判定（仅对该步）
    alerts = _warning_engine.evaluate(last_mask, last_scores, t=t_global)
    top_alert = alerts[-1] if alerts else None

    # 时间戳转 ISO 字符串
    # 2026-07-31 BUG FIX: NPZ 文件里的 timestamps 数组按 60min 步长生成（162 天），
    # 但 flow 数据本身按 30min 步长（3888 帧 ≈ 81 天）。两者采样率不匹配。
    # 直接按 30min 步长重新生成 timestamps，从已知起点 2015-11-01 00:00 起。
    # 2026-08-01 BUG FIX: 之前用 t_local = t_global - val_end，但起点仍按全局 t=0 算，
    # 导致测试集时间戳偏早（差 3288 步 ≈ 42 天）。修复：用 t_global 而非 t_local。
    ts_str = None
    try:
        t_start = np.datetime64("2015-11-01T00:00:00")
        ts_correct = t_start + np.timedelta64(int(t_global) * 30, "m")
        ts_str = str(np.datetime_as_string(ts_correct, unit="m"))
    except Exception as e:
        ts_str = f"t={t_global}"

    elapsed_ms = (time.time() - t0) * 1000
    return AnomalyDetectResponse(
        mode=req.mode,
        threshold=threshold,
        t=t_global,
        timestamp=ts_str,
        period=period,
        t_min=val_end,
        t_max=test_end,
        n_anomaly_cells=int(last_mask.sum()),
        n_total_cells=N,
        anomaly_rate=float(last_mask.sum() / N),
        fused_scores=scores_out,
        heatmap=last_flow.reshape(GRID_H, GRID_W).tolist(),
        anomaly_mask=last_mask.reshape(GRID_H, GRID_W).astype(int).tolist(),
        cells=cells_out,
        warning_level=top_alert.level if top_alert else None,
        warning_name=top_alert.level_name if top_alert else None,
        processing_ms=elapsed_ms,
    )


@app.post("/api/anomaly/events", response_model=EventsQueryResponse)
def query_events(req: EventsQueryRequest | None = None):
    """
    历史异常事件查询接口

    按时间范围和最少网格数过滤，返回异常事件列表。
    不传参数（空body）时返回全部事件。
    """
    t0 = time.time()
    pipe = _get_pipeline()

    if not pipe._result_cache:
        pipe.run_batch(split="test")

    # 安全提取参数，None body 或缺省字段均走默认值
    p = req if req is not None else EventsQueryRequest()
    events = pipe.query_events(
        t_start=p.t_start,
        t_end=p.t_end,
        min_cells=p.min_cells,
        include_marginal=p.include_marginal,
    )

    # 预警等级过滤（API schema里缺省为None，None=不过滤）
    if p.level_filter is not None:
        events = [e for e in events if e["warning_level"] == p.level_filter]

    level_map = {3: "紧急", 2: "重要", 1: "一般", 0: "未知"}
    event_models = [
        AnomalyEvent(
            event_id=e["event_id"],
            t_start=e["t_start"],
            t_end=e["t_end"],
            duration=e["duration"],
            n_cells=e["n_cells"],
            center_row=e["center_row"],
            center_col=e["center_col"],
            event_type=e["event_type"],
            avg_score=e["avg_score"],
            warning_level=e["warning_level"],
            level_name=level_map.get(e["warning_level"], "未知"),
        )
        for e in events
    ]

    elapsed_ms = (time.time() - t0) * 1000
    return EventsQueryResponse(
        total=len(event_models),
        events=event_models,
    )


@app.post("/api/timeslots", response_model=TimeslotsResponse)
def query_timeslots(req: TimeslotsRequest):
    """
    动态查询典型时间槽：夜间低谷、早高峰、晚高峰对应的 t 值。

    实现逻辑：
    1. 拉取 t_min~t_max 范围内所有网格的平均流量
    2. 按时间步聚合，计算每步的平均流量
    3. 对 night_valley 找最低的 5 个时间点，返回中位数
    4. 对 morning_peak/evening_peak 找最高的 5 个时间点，返回中位数

    这样即使数据起点不是 00:00，也能准确找到对应的 t。
    """
    t0 = time.time()
    pipe = _get_pipeline()

    # 获取时间边界
    try:
        from week5.data_loader import get_splits, get_timestamps
        _, val_end, test_end = get_splits()
        timestamps = get_timestamps()
    except Exception:
        val_end, test_end = 3288, 3888
        timestamps = None

    t_min = req.t_min if req.t_min is not None else val_end
    t_max = req.t_max if req.t_max is not None else test_end - 1

    if t_max - t_min < 12:
        raise HTTPException(400, "时间范围至少需要 12 步")

    try:
        result = pipe.run_batch(split="test")
    except Exception as e:
        raise HTTPException(500, f"Pipeline 运行失败: {e}")

    flow = result["flow"]
    T_steps = len(flow)

    # 2026-07-31 BUG FIX：原始 NPZ 里 timestamps 是按 60min 步长生成的 (162 天)，
    # 但 flow 数据本身是 30min 步长（3888 帧 ≈ 81 天）。两者不能一一对应。
    # 重新按 30min 步长生成 timestamps（与 flow 索引真实语义一致）。
    t_start_ts = np.datetime64("2015-11-01T00:00:00")
    timestamps_30min = np.array(
        [t_start_ts + np.timedelta64(30 * i, "m") for i in range(len(timestamps))],
        dtype="datetime64[s]",
    )

    # 计算每步的平均流量
    step_flows = []
    valid_ts = []
    for t_global in range(t_min, t_max + 1):
        t_local = t_global - val_end
        if 0 <= t_local < T_steps:
            step_flows.append(float(np.nanmean(flow[t_local])))
            # 用重建的 30min 步长 timestamps 提取小时
            hour = None
            if t_global < len(timestamps_30min):
                try:
                    hour = int(pd.Timestamp(timestamps_30min[t_global]).hour)
                except Exception:
                    pass
            valid_ts.append((t_global, step_flows[-1], hour))
        else:
            step_flows.append(0.0)
            valid_ts.append((t_global, 0.0, None))

    if len(step_flows) == 0:
        raise HTTPException(500, "没有有效的时间步数据")

    # 按时序排序的 (t, flow, hour)
    sorted_by_flow = sorted(valid_ts, key=lambda x: x[1])
    sorted_desc = sorted_by_flow[::-1]

    if req.target == "night_valley":
        # 夜间低谷：限定 02~06 点之间找最低（避免误选中午 12:00 的极低流量点）
        _candidates = [c for c in valid_ts if c[2] is not None and 2 <= c[2] <= 6]
        if not _candidates:
            _candidates = sorted_by_flow[:10]
        # 按流量升序选 10 个，再用 hour 中位数筛出最贴近窗口中心的那个
        top10 = sorted(_candidates, key=lambda x: x[1])[:10]
        # 最佳 hour 应在窗口内（02~06），取中位数
        target_hour = int(np.median([c[2] for c in top10]))
        # 选 top10 中 hour 离 target_hour 最近的 t（避免 median 取整跨过去）
        def _score(c):
            return (abs(c[2] - target_hour), c[1])
        best = min(top10, key=_score)
        best_t = best[0]
        best_hour = best[2]
        candidates = top10
    elif req.target == "morning_peak":
        # 早高峰：限定 07~10 点之间找最高
        _candidates = [c for c in valid_ts if c[2] is not None and 7 <= c[2] <= 10]
        if not _candidates:
            _candidates = sorted_desc[:10]
        top10 = sorted(_candidates, key=lambda x: -x[1])[:10]
        target_hour = int(np.median([c[2] for c in top10]))
        def _score(c):
            return (abs(c[2] - target_hour), -c[1])
        best = min(top10, key=_score)
        best_t = best[0]
        best_hour = best[2]
        candidates = top10
    elif req.target == "evening_peak":
        # 晚高峰：限定 17~20 点（关键修复：之前全局最大容易撞到早晨通勤峰）
        _candidates = [c for c in valid_ts if c[2] is not None and 17 <= c[2] <= 20]
        if not _candidates:
            _candidates = sorted_desc[:10]
        top10 = sorted(_candidates, key=lambda x: -x[1])[:10]
        target_hour = int(np.median([c[2] for c in top10]))
        def _score(c):
            return (abs(c[2] - target_hour), -c[1])
        best = min(top10, key=_score)
        best_t = best[0]
        best_hour = best[2]
        candidates = top10
    else:
        raise HTTPException(400, f"未知的 target: {req.target}，支持: night_valley, morning_peak, evening_peak")

    elapsed_ms = (time.time() - t0) * 1000
    return TimeslotsResponse(
        target=req.target,
        t=best_t,
        hour_estimate=best_hour,
        t_min=t_min,
        t_max=t_max,
    )


# ── 常量导出（供内部使用）─────────────────────────────────────────────────────
from week5.config import VAL_END, TEST_HOURS


# ── 北京 32×32 网格地理边界 ──────────────────────────────────────────────────
# 来源：week2 step1_5_poi.py / verify_poi_rebuild.py / reverse_engineer_grid.py
GRID_LON_MIN = 116.10
GRID_LON_MAX = 116.60
GRID_LAT_MIN = 39.79
GRID_LAT_MAX = 40.05
GRID_H = GRID_W = 32


def _grid_cell_centers() -> list:
    """返回 1024 个网格中心的 (lon, lat) 列表（按 row-major）"""
    lon_step = (GRID_LON_MAX - GRID_LON_MIN) / GRID_W
    lat_step = (GRID_LAT_MAX - GRID_LAT_MIN) / GRID_H
    cells = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            lon = GRID_LON_MIN + (c + 0.5) * lon_step
            lat = GRID_LAT_MIN + (GRID_H - 1 - r + 0.5) * lat_step  # 北→南翻
            cells.append((lon, lat))
    return cells


_GRID_CELLS = _grid_cell_centers()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[API] 启动服务，端口 {port}，访问 http://0.0.0.0:{port}/docs 查看接口文档")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
