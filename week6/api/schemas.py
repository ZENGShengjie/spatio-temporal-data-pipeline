"""Week6 API 数据模型 — Pydantic schemas"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


# ── 请求模型 ─────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    """人流预测请求"""
    time_start: int = Field(..., ge=0, description="起始时间步（全局索引）")
    time_end: int = Field(..., ge=0, description="结束时间步")
    grid_ids: List[int] = Field(
        default=None,
        description="网格编号列表（0~1023），None=全量网格"
    )

    @field_validator("time_end")
    @classmethod
    def end_after_start(cls, v, info):
        if "time_start" in info.data and v < info.data["time_start"]:
            raise ValueError("time_end 必须 >= time_start")
        return v


class AnomalyDetectRequest(BaseModel):
    """异常检测请求"""
    t: Optional[int] = Field(
        default=None,
        description="指定时间步（全局索引）；None=用最后一个测试步",
    )
    data: Optional[List[List[float]]] = Field(
        default=None,
        description="时序数据矩阵，shape=(T, N)，T=时间步数，N=1024网格数。留空则使用测试集默认数据。"
    )
    mode: str = Field(
        default="fast",
        description="检测模式：fast=快速统计法+预测法融合，structural=全量融合"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="融合阈值（None=使用校准最优值）"
    )

    @field_validator("data")
    @classmethod
    def check_shape(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError("data 不能为空")
        T = len(v)
        N = len(v[0]) if v else 0
        if N != 1024:
            raise ValueError(f"data 列数必须为 1024，当前为 {N}")
        if T < 1 or T > 500:
            raise ValueError(f"data 行数应在 1~500 之间，当前为 {T}")
        return v


class EventsQueryRequest(BaseModel):
    """历史异常事件查询请求"""
    t_start: Optional[int] = Field(default=None, description="起始时间步")
    t_end: Optional[int] = Field(default=None, description="结束时间步")
    min_cells: int = Field(default=0, ge=0, description="最少涉及网格数")
    level_filter: Optional[int] = Field(
        default=None,
        description="预警等级过滤（1=一般 2=重要 3=紧急）"
    )
    include_marginal: bool = Field(
        default=False,
        description="包含兜底入库的零散/瞬时异常（patch_marginal / point_single），"
                    "默认关闭以保持列表简洁。与热力图实时异常格点一一对应。"
    )


# ── 响应模型 ─────────────────────────────────────────────────────────────────

class CellForecast(BaseModel):
    """单个网格的预测结果"""
    grid_id: int
    values: List[float]


class ForecastResponse(BaseModel):
    """人流预测响应"""
    time_start: int
    time_end: int
    horizon: int = Field(description="预测步数（小时）")
    cells: List[CellForecast]
    note: str = "values 为归一化后的人流量"
    heatmap: List[List[float]] = Field(
        default_factory=list,
        description="对应 time_end 时刻的 32×32 人流热力图",
    )
    timestamps: List[str] = Field(
        default_factory=list,
        description="每步对应的时间戳（ISO 格式）",
    )


class AnomalyScore(BaseModel):
    """异常得分"""
    grid_id: int
    score: float
    is_anomaly: bool


class AnomalyCell(BaseModel):
    """单个网格的热力图数据"""
    grid_id: int
    row: int
    col: int
    lon: float
    lat: float
    flow: float
    score: float
    is_anomaly: bool


class AnomalyDetectResponse(BaseModel):
    """异常检测响应"""
    mode: str
    threshold: float
    t: int = Field(..., description="当前时间步（全局索引）")
    timestamp: Optional[str] = None
    period: str = Field(default="P4", description="当前时间段标识")
    t_min: int = Field(default=0, description="当前时间段测试集起点")
    t_max: int = Field(default=3888, description="当前时间段测试集终点")
    n_anomaly_cells: int
    n_total_cells: int
    anomaly_rate: float
    fused_scores: List[AnomalyScore]
    heatmap: List[List[float]] = Field(
        default_factory=list,
        description="32×32 人流热力图矩阵",
    )
    anomaly_mask: List[List[int]] = Field(
        default_factory=list,
        description="32×32 异常标记矩阵 (0/1)",
    )
    cells: List[AnomalyCell] = Field(
        default_factory=list,
        description="每个网格的详情（包含经纬度，供地图可视化使用）",
    )
    warning_level: Optional[int] = None
    warning_name: Optional[str] = None
    processing_ms: float


class AnomalyEvent(BaseModel):
    """异常事件"""
    event_id: int
    t_start: int
    t_end: int
    duration: int
    n_cells: int
    center_row: int
    center_col: int
    event_type: str
    avg_score: float
    warning_level: int
    level_name: str


class EventsQueryResponse(BaseModel):
    """异常事件查询响应"""
    total: int
    events: List[AnomalyEvent]


# ── 健康检查 ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    pipeline_mode: str
    cache_loaded: bool
    period: str = "P4"
    t_min: int = 3288
    t_max: int = 3887
    available_periods: List[str] = Field(
        default_factory=lambda: ["P4", "BJ13", "BJ14", "BJ15", "BJ16"]
    )
    version: str = "1.1.0"
