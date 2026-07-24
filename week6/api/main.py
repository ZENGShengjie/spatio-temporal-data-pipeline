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
import torch

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
)

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
        cuda_available=torch.cuda.is_available(),
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
    """
    t0 = time.time()
    pipe = _get_pipeline()

    if req.data is not None:
        try:
            data = np.array(req.data, dtype=np.float32)
            T, N = data.shape
        except Exception as e:
            raise HTTPException(400, f"数据格式错误: {e}")

    if req.mode == "structural":
        global _pipeline
        if _pipeline.mode != "structural":
            print("[API] 切换到 structural 模式，加载深度学习模型...")
            _pipeline.mode = "structural"

    try:
        result = pipe.run_batch(split="test")
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
    ts_str = None
    if timestamps is not None and 0 <= t_local < len(timestamps):
        try:
            ts_str = str(np.datetime_as_string(timestamps[t_local], unit="m"))
        except Exception:
            ts_str = str(timestamps[t_local])

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
