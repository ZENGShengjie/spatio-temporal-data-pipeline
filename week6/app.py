"""Week6 Streamlit 可视化界面（修复版）

完整功能：
  1. 实时热力图（接真实 API 数据 + 32×32 Plotly）
  2. 24 小时预测动画（Plotly Frames 逐帧播放）
  3. 异常事件列表 + 历史查询
  4. 单网格时序分析（真实 API 数据）
  5. 地理地图可视化（MapLibre + 高德路网，无需 API key）
  6. 三级预警 + 弹窗 + 声音（WebAudio 合成）

依赖：streamlit, plotly, pandas, numpy, requests, maplibre-gl
"""
from __future__ import annotations
import io
import json
import os
import time
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="城市人流异常检测系统",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 常量（时间范围从 API health 动态获取）────────────────────────────────────
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")  # 可通过 export API_BASE=... 覆盖 EC2 IP
GRID_H = GRID_W = 32
N_CELLS = GRID_H * GRID_W

# 默认值（API 启动后会被 health check 结果覆盖）
TEST_T_MIN = 3288
TEST_T_MAX = 3887
CURRENT_PERIOD = "P4"

# 警示音 WAV（用 Python 动态生成的 base64，避免外部资源依赖）
def _beep_wav_b64(freq: int = 880, duration_ms: int = 250) -> str:
    """生成单频警示音 WAV 的 base64"""
    import struct
    import math
    sample_rate = 8000
    n_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        # 简单正弦波 + 衰减包络
        envelope = max(0.0, 1.0 - t / (duration_ms / 1000))
        val = int(32767 * 0.6 * envelope * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", val))
    data = b"".join(samples)
    wav = b"RIFF" + struct.pack("<I", 36 + len(data))
    wav += b"WAVEfmt "
    wav += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    wav += b"data" + struct.pack("<I", len(data)) + data
    return base64.b64encode(wav).decode()


# ── 侧边栏 ────────────────────────────────────────────────────────────────────
st.sidebar.title("系统配置")

mode = st.sidebar.radio(
    "检测模式",
    ["快速模式 (fast)", "结构增强模式 (structural)"],
    index=0,
    help="fast: 统计法+预测法，响应快；structural: 全量融合，针对连片异常",
)
mode_param = "structural" if "structural" in mode else "fast"

# ── 缓存层 ──────────────────────────────────────────────────────────────────
# 注：必须在 health-check 之前定义，否则第 116/117 行的调用会 NameError。


def _format_hour_label(h: int) -> str:
    """把 24h 小时数字格式化成带「凌晨/上午/下午/晚上」的中文标签。

    例如 3 -> "凌晨 03:00"；10 -> "上午 10:00"；17 -> "下午 17:00"；22 -> "晚上 22:00"。
    这样避免 03:00 被误读为下午 3 点、10:00 被误读为晚上 10 点。
    """
    if h is None:
        return "未知时间"
    h = int(h) % 24
    if h < 6:
        period = "凌晨"
    elif h < 12:
        period = "上午"
    elif h < 18:
        period = "下午"
    else:
        period = "晚上"
    return f"{period} {h:02d}:00"


@st.cache_data(show_spinner=False, ttl=300)
def _detect_cached(t: int, mode: str):
    """调用 detect 接口并把返回值缓存 5 分钟。

    接口一次返回 ~226KB 的 JSON；同 (t, mode) 不重复打 API，rerun/按钮点击瞬时返回。
    cache_data 默认按参数 hash；Streamlit rerun 时同名 cache miss 才向后端打。
    """
    return call_api(
        "/api/anomaly/detect", "POST",
        json_data={"t": t, "mode": mode},
    )


@st.cache_data(show_spinner=False, ttl=600)
def _fetch_timeslots_cached(t_min: int, t_max: int):
    """一次拿 3 个 timeslot 的 t 值，缓存 10 分钟。

    返回 dict: {"night_valley": int, "morning_peak": int, "evening_peak": int}
    """
    slots = {}
    for target in ("night_valley", "morning_peak", "evening_peak"):
        r = requests.post(
            f"{API_BASE}/api/timeslots",
            json={"target": target, "t_min": t_min, "t_max": t_max},
            timeout=15,
        )
        if r.status_code == 200:
            slots[target] = r.json().get("t")
        else:
            slots[target] = None
    return slots


@st.cache_data(show_spinner=False, ttl=600)
def _fetch_timeslots_hours_cached(t_min: int, t_max: int):
    """拿 3 个 timeslot 的 hour_estimate，缓存 10 分钟。"""
    hours = {}
    for target in ("night_valley", "morning_peak", "evening_peak"):
        r = requests.post(
            f"{API_BASE}/api/timeslots",
            json={"target": target, "t_min": t_min, "t_max": t_max},
            timeout=15,
        )
        if r.status_code == 200:
            hours[target] = r.json().get("hour_estimate")
        else:
            hours[target] = None
    return hours


st.sidebar.markdown("---")
st.sidebar.markdown(f"**API 地址**: `{API_BASE}`")

# 健康检查
health_ok = False
current_period = "P4"
available_periods = ["P4"]
t_min = TEST_T_MIN
t_max = TEST_T_MAX
CURRENT_PERIOD = "P4"
# 默认偏移（数据起点假设 00:00：凌晨3点=+6、上午8点=+16、下午18点=+36）
_fallback_t_night    = TEST_T_MIN + 6
_fallback_t_morning = TEST_T_MIN + 16
_fallback_t_evening = TEST_T_MIN + 36
try:
    resp = requests.get(f"{API_BASE}/api/health", timeout=5)
    if resp.status_code == 200:
        health = resp.json()
        st.sidebar.success(f"API 在线 ({health.get('status','?')})")
        st.sidebar.caption(f"CUDA: {health.get('cuda_available', False)}")
        health_ok = True
        # 动态获取时间范围和 period 信息
        current_period = health.get("period", "P4")
        available_periods = health.get("available_periods", ["P4"])
        t_min = health.get("t_min", 3288)
        t_max = health.get("t_max", 3887)
        TEST_T_MIN = t_min
        TEST_T_MAX = t_max - 1 if t_max else 3887

        # 覆盖默认偏移（health 已知准确的 t_min）
        _fallback_t_night    = TEST_T_MIN + 6
        _fallback_t_morning  = TEST_T_MIN + 16
        _fallback_t_evening  = TEST_T_MIN + 36

        # 动态获取三个典型时间槽（夜间低谷、早高峰、晚高峰）
        # 用 @st.cache_data 把 3 次 POST 缓存 10 分钟；rerun/按钮点都不再打 API
        # 注意：之前用过的 session_state 旧 hour 值可能跟新版不一致，
        # 每次都覆盖写入，避免历史脏值。
        _dynamic_slots = _fetch_timeslots_cached(t_min, t_max)
        _dynamic_hours = _fetch_timeslots_hours_cached(t_min, t_max)

        st.session_state["_t_night"]    = _dynamic_slots.get("night_valley")  or _fallback_t_night
        st.session_state["_t_morning"]  = _dynamic_slots.get("morning_peak")   or _fallback_t_morning
        st.session_state["_t_evening"]  = _dynamic_slots.get("evening_peak")   or _fallback_t_evening
        st.session_state["_h_night"]    = _dynamic_hours.get("night_valley")  or 3
        st.session_state["_h_morning"]  = _dynamic_hours.get("morning_peak")   or 10
        st.session_state["_h_evening"]  = _dynamic_hours.get("evening_peak")   or 17

        if len(_dynamic_slots) < 3:
            st.sidebar.warning("⚠️ 动态时间槽获取失败，使用默认偏移")
        else:
            st.sidebar.caption(
                f"时间槽: 夜间={_dynamic_slots.get('night_valley', '?')}, "
                f"早高峰={_dynamic_slots.get('morning_peak', '?')}, "
                f"晚高峰={_dynamic_slots.get('evening_peak', '?')}"
            )
        CURRENT_PERIOD = current_period
    else:
        st.sidebar.error(f"API 异常 ({resp.status_code})")
except Exception as e:
    st.sidebar.error(f"API 离线: {e}")
    st.info("请先启动 API 服务: `python -m week6.api.main`")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API**: `{API_BASE}`")
st.sidebar.markdown(f"**当前时段**: `{CURRENT_PERIOD}`")

# 时间段选择
st.sidebar.markdown("### 数据时间段")
period_labels = {
    "P4":   "P4 (2015-11~2016-04) — 默认",
    "BJ13": "BJ13 (2013-07~2013-10)",
    "BJ14": "BJ14 (2014-03~2014-06)",
    "BJ15": "BJ15 (2015-03~2015-06)",
    "BJ16": "BJ16 (2015-11~2016-04)",
}
selected_period = st.sidebar.selectbox(
    "选择分析时段",
    options=available_periods,
    index=available_periods.index(current_period) if current_period in available_periods else 0,
    format_func=lambda p: period_labels.get(p, p),
    help="切换时间段后需重启 API 服务生效",
)
st.sidebar.caption("注：切换时段需重启 EC2 上的 API 服务")
st.sidebar.markdown("---")
st.sidebar.caption("数据：北京出租车 GPS · 32×32 网格")


# ── Auto-refresh: st_autorefresh drives page rerender; jump logic checks timestamps ─
if "_next_jump_ts" not in st.session_state:
    st.session_state._next_jump_ts = None
if "_auto_refresh_at" not in st.session_state:
    st.session_state._auto_refresh_at = ""
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []
if "last_warning_level" not in st.session_state:
    st.session_state.last_warning_level = 0
if "_last_alert_t" not in st.session_state:
    st.session_state._last_alert_t = None
if "_audio_unlocked" not in st.session_state:
    # 浏览器自动播放策略：必须先有用户交互才能播放音频
    st.session_state._audio_unlocked = False
if "_audio_unlock_ts" not in st.session_state:
    st.session_state._audio_unlock_ts = 0.0


def _on_auto_refresh_change():
    if st.session_state.auto_refresh:
        st.session_state._next_jump_ts = time.time() + 8
    else:
        st.session_state._next_jump_ts = None


# Trigger auto-jump if timestamp has passed
now_ts = time.time()
next_ts = st.session_state._next_jump_ts
if (next_ts is not None) and (now_ts >= next_ts):
    st.session_state["t1_slider"] = min(
        st.session_state.get("t1_slider", TEST_T_MIN + 200) + 6, TEST_T_MAX
    )
    st.session_state._auto_refresh_at = datetime.now().strftime("%H:%M:%S")
    if st.session_state.auto_refresh:
        st.session_state._next_jump_ts = time.time() + 8
    else:
        st.session_state._next_jump_ts = None


# Invisible autorefresh: when auto_refresh is on, force rerun every 2s
if st.session_state.auto_refresh:
    st_autorefresh(interval=6000, key="autorefresh_dummy")


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

# 一次性初始化 session_state 中的 slider 默认值（避免 streamlit ≥1.30
# 因同时传 value= 和 session_state[key] 而发出冲突警告并静默终止按钮）。
# 在第一次 script 启动时执行，后续 rerun 不会再覆盖用户已经改过的值。
for _k, _default in [
    ("t1_slider",  TEST_T_MIN + 200),
]:
    st.session_state.setdefault(_k, _default)


@st.cache_resource
def _get_session() -> requests.Session:
    """单例 requests.Session，复用 TCP 连接、避免 socket 泄漏。"""
    s = requests.Session()
    return s


def call_api(endpoint: str, method: str = "GET", json_data=None, params=None):
    """调用 API。带连接复用、详细日志、超时。

    使用 requests.Session() 复用 TCP 连接，避免每次调用新建 socket。
    """
    url = f"{API_BASE}{endpoint}"
    t0 = time.time()
    sess = _get_session()
    try:
        if method == "GET":
            r = sess.get(url, params=params, timeout=10)
        else:
            r = sess.post(url, json=json_data, timeout=30)
        dt = (time.time() - t0) * 1000
        if r.status_code == 200:
            print(f"[call_api] {method} {endpoint} ok in {dt:.0f}ms "
                  f"({len(r.content)} bytes)")
            return r.json()
        print(f"[call_api] {method} {endpoint} -> {r.status_code} "
              f"in {dt:.0f}ms: {r.text[:200]}")
        st.error(f"API 错误 {r.status_code}: {r.text[:200]}")
        return None
    except Exception as e:
        dt = (time.time() - t0) * 1000
        print(f"[call_api] {method} {endpoint} EXC after {dt:.0f}ms: {e!r}")
        st.error(f"请求失败: {e}")
        return None


def play_alert_sound(level: int):
    """播放预警音（WebAudio 合成，无需外部文件）

    浏览器 autoplay 策略：必须先有用户交互才能播放音频。
    我们用 _audio_unlocked 状态标记；如未解锁则只在界面提示，不真正发声。
    """
    if not st.session_state.get("_audio_unlocked", False):
        # 没解锁：只显示视觉提示，不发声（避免无效操作）
        st.caption("🔇 浏览器禁止自动播放，请点击上方「🔔 启用警示音」按钮解锁")
        return
    freq_map = {3: 1200, 2: 880, 1: 660}
    freq = freq_map.get(level, 600)
    beep_b64 = _beep_wav_b64(freq=freq, duration_ms=300)
    # 用 HTML audio + autoplay 触发（现代浏览器需用户先交互后才能自动播放）
    components.html(
        f"""<audio autoplay>
        <source src="data:audio/wav;base64,{beep_b64}" type="audio/wav">
        </audio>""",
        height=0,
    )


def popup_alert(level: int, level_name: str, msg: str):
    """浏览器原生弹窗 + Streamlit 内提示"""
    if not st.session_state.get("_audio_unlocked", False):
        # 未解锁：用 st.toast 作为替代，更明显
        st.toast(f"[{level_name}] {msg}", icon="🚨" if level == 3 else ("⚠️" if level == 2 else "ℹ️"))
    else:
        components.html(
            f"""<script>
            alert('[{level_name}] {msg}');
            </script>""",
            height=0,
        )
    if level == 3:
        st.error(f"🚨 **{level_name}预警** — {msg}")
    elif level == 2:
        st.warning(f"⚠️ **{level_name}预警** — {msg}")
    else:
        st.info(f"ℹ️ **{level_name}预警** — {msg}")


def show_alert_banner(level: int | None, level_name: str | None, msg: str = "", t: int = None):
    """统一的预警横幅 + 触发弹窗/声音（按时间步去重，换步后重新触发）

    等级策略：
      - level >= 2：重要/紧急 → banner + 弹窗 + 声音
      - level == 1：一般     → 轻量 toast + 单声轻音（已解锁时）+ 黄色状态条
      - level is None         → 绿色无预警提示
    """
    if level is None:
        st.success("✅ 当前无预警")
        return

    if t is not None and st.session_state._last_alert_t != t:
        st.session_state.last_warning_level = 0
        st.session_state._last_alert_t = t

    if level >= 2:
        colors = {3: "🔴", 2: "🟠"}
        st.markdown(
            f"### {colors.get(level, '⚪')} {level_name}预警 (Level {level})"
        )
        st.write(msg or f"检测到 {level_name}级别异常")
        if level > st.session_state.last_warning_level:
            play_alert_sound(level)
            popup_alert(level, level_name, msg)
            st.session_state.last_warning_level = level
        elif level == 3:
            play_alert_sound(level)
    else:
        # 一般预警：已解锁音频时轻响一声，否则静默
        # streamlit 不支持 toast，用 info + 声音替代（声音已足够提示）
        if st.session_state.get("_audio_unlocked", False):
            play_alert_sound(level=1)
        st.markdown(
            "<span style='background:#fff3cd;padding:4px 10px;border-radius:4px;"
            "border-left:4px solid #ffc107;font-weight:bold'>"
            f"ℹ️ 一般预警：{msg or '检测到一般级别异常'}"
            "</span>",
            unsafe_allow_html=True,
        )


def plot_heatmap_32(flow_2d, anomaly_mask_2d, scores_2d, title):
    """Plotly 32×32 热力图 + 异常格点标记"""
    fig = go.Figure()

    zmin, zmax = _compute_z_bounds(flow_2d)
    fig.add_trace(go.Heatmap(
        z=flow_2d,
        colorscale="YlOrRd",
        zmin=zmin, zmax=zmax,             # 分位数动态色阶，避免饱和
        showscale=True,
        colorbar=dict(title="人流量", x=1.02),
        hovertemplate="row:%{y}<br>col:%{x}<br>flow:%{z:.3f}<extra></extra>",
        name="人流",
    ))

    if anomaly_mask_2d is not None and anomaly_mask_2d.any():
        rows, cols = np.where(anomaly_mask_2d.astype(bool))
        # Plotly Heatmap 在 yaxis autorange="reversed" 时，row 0 自动显示在图顶；
        # np.where 返回的 rows 就是矩阵行号，y=rows 直接正确。无需翻转。
        # (2026-07-28 修复：早期曾误改成 y=H-1-rows，导致红框偏转半张图，已回退)
        fig.add_trace(go.Scatter(
            x=cols, y=rows, mode="markers",
            marker=dict(
                size=14,                            # 加大红框，热点区域更醒目
                color="rgba(0,0,0,0)",
                symbol="square",
                line=dict(width=2.5, color="red"),
            ),
            name="异常格点",
            hovertemplate="异常 row:%{y} col:%{x}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        height=480,
        xaxis=dict(scaleanchor="y", constrain="domain", title="列"),
        yaxis=dict(scaleanchor="x", constrain="domain", title="行", autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_timeseries_real(grid_id, t_start, t_end):
    """拉取该网格的真实时序数据并绘制"""
    fc = call_api(
        "/api/forecast", "POST",
        json_data={"time_start": t_start, "time_end": t_end, "grid_ids": [grid_id]},
    )
    if not fc or not fc.get("cells"):
        return None, None, None, None
    cell = fc["cells"][0]
    values = np.array(cell["values"])
    T = len(values)
    ts_labels = fc.get("timestamps") or [str(i) for i in range(T)]
    row, col = grid_id // GRID_W, grid_id % GRID_W

    # 异常标记：拉取该区间内的异常判定（采样以提速）
    anomaly_marks = []
    step = max(1, (t_end - t_start + 1) // 60)  # 最多 60 次调用
    times_to_check = list(range(t_start, t_end + 1, step))
    anomaly_lookup = {}
    for t in times_to_check:
        det = call_api("/api/anomaly/detect", "POST", json_data={"t": t, "mode": mode_param})
        if det and det.get("fused_scores"):
            for fs in det["fused_scores"]:
                if fs["grid_id"] == grid_id:
                    anomaly_lookup[t] = fs.get("is_anomaly", False)
                    break
    # 填充所有时间步
    for t in range(t_start, t_end + 1):
        # 找最近的已检测时间
        closest_t = min(times_to_check, key=lambda x: abs(x - t), default=None)
        is_anom = anomaly_lookup.get(closest_t, False)
        anomaly_marks.append((t - t_start, "red" if is_anom else "lightgray"))

    fig = make_timeseries_fig(values, anomaly_marks, ts_labels, row, col)
    return fig, values, ts_labels, anomaly_marks


def make_timeseries_fig(values, anomaly_marks, ts_labels, row, col):
    from plotly.subplots import make_subplots
    T = len(values)
    x = list(range(T))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"网格 ({row},{col}) 人流时序",
            "异常标记",
        ),
    )

    fig.add_trace(go.Scatter(
        x=x, y=values,
        mode="lines+markers", name="实际人流",
        line=dict(color="steelblue", width=1.5),
        marker=dict(size=4),
        hovertemplate="t=%{x}<br>flow=%{y:.3f}<extra></extra>",
    ), row=1, col=1)

    colors_bar = [m[1] for m in anomaly_marks]
    fig.add_trace(go.Bar(
        x=x, y=[1] * T,
        marker_color=colors_bar,
        showlegend=False,
        hoverinfo="skip",
    ), row=2, col=1)
    fig.update_yaxes(range=[0, 1.2], showticklabels=False, row=2, col=1)

    fig.update_layout(
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="时间步", row=2, col=1)
    fig.update_yaxes(title_text="归一化人流量", row=1, col=1)
    return fig


def build_24h_frames(t_start: int, t_end: int):
    """拉取连续 24 步（每小时一步）数据 + 对应异常掩码，构造动画帧。

    异常掩码通过调用 /api/anomaly/detect 获取（每4帧调一次，中间帧复用相邻帧的 mask，
    避免 24 次 API 调用导致响应过慢）。
    返回 (frames, anomaly_masks, timestamps)，其中 anomaly_masks 与 frames 等长。
    """
    fc = call_api(
        "/api/forecast", "POST",
        json_data={"time_start": t_start, "time_end": t_end},
    )
    if not fc:
        return None

    timestamps = fc.get("timestamps") or []
    n_steps = fc.get("horizon", 0)
    if n_steps == 0:
        return None

    cells = fc["cells"]
    if not cells:
        return None

    frames = []
    for k in range(n_steps):
        z = np.zeros((GRID_H, GRID_W))
        for cell in cells:
            gid = cell["grid_id"]
            r, c = gid // GRID_W, gid % GRID_W
            if k < len(cell["values"]):
                z[r, c] = cell["values"][k]
        frames.append(z)

    # 获取每帧异常掩码：每4帧调用一次 /api/anomaly/detect，中间帧复用
    anomaly_masks = []
    sample_step = 4
    prev_mask = None
    for k in range(n_steps):
        if k % sample_step == 0 or k == n_steps - 1:
            # 走缓存：同 (t, mode) 5 分钟内不重复打 API，
            # 用户重复点「生成动画」从 ~8s 降到 ~1s
            det = _detect_cached(t_start + k, "fast")
            if det and "anomaly_mask" in det and det["anomaly_mask"]:
                mask_2d = np.array(det["anomaly_mask"], dtype=int)
                prev_mask = mask_2d
            elif det and "cells" in det:
                # 兜底：从 cells 的 is_anomaly 字段构造 mask
                mask_2d = np.zeros((GRID_H, GRID_W), dtype=int)
                for c in det["cells"]:
                    if c.get("is_anomaly"):
                        r, col = c["row"], c["col"]
                        if 0 <= r < GRID_H and 0 <= col < GRID_W:
                            mask_2d[r, col] = 1
                prev_mask = mask_2d
            else:
                # API 失败：尝试复用上一帧，否则用全零
                mask_2d = prev_mask if prev_mask is not None else np.zeros((GRID_H, GRID_W), dtype=int)
        else:
            # 中间帧复用上一帧的 mask
            mask_2d = prev_mask if prev_mask is not None else np.zeros((GRID_H, GRID_W), dtype=int)
        anomaly_masks.append(mask_2d)

    # 若无 timestamps，生成占位
    if not timestamps:
        timestamps = [f"+{i}h" for i in range(n_steps)]

    return frames, anomaly_masks, timestamps


def make_animation_fig(frames_data, anomaly_masks, timestamps, t_start,
                       flow_min=None, flow_max=None, anomaly_overlay: bool = False,
                       autoplay: bool = True,
                       frame_ms: int = 350, transition_ms: int = 350):
    """Plotly 动画热力图（默认**不**叠加异常红框 — 仅展示流量本身）

    动画连贯性说明：Plotly 没有视频播放器语义，frames 是「静态图序列」。
    通过以下方式让肉眼看起来像连续视频：
      - frame_ms 短（350ms/帧）→ 帧间紧凑
      - transition_ms > 0    → 帧间颜色/数值的 CSS 插值动画，让热度"流动"
      - redraw=True           → heatmap 数值重绘（数值变化连续，肉眼感觉平滑）

    Args:
        frames_data: List[np.ndarray]  每帧的 32×32 流量矩阵
        anomaly_masks: List[np.ndarray] 每帧的 32×32 异常掩码（来自 API，**不再使用**）
        timestamps: List[str]           每帧的时间戳标签
        t_start: int                    起点 t
        flow_min, flow_max: 分位数色阶边界
        anomaly_overlay: False=不显示异常红框（默认）；保留形参仅为向后兼容
        autoplay: True=自动播放
        frame_ms: 每帧停留毫秒数（默认 350，≈2.86 帧/秒）
        transition_ms: 帧间过渡毫秒数（默认 350，与 frame_ms 同长实现无缝循环）
    """
    if not frames_data:
        return None
    z0 = frames_data[0]
    if flow_min is None or flow_max is None:
        all_vals = np.concatenate([np.asarray(f, dtype=float).flatten() for f in frames_data])
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) >= 2:
            flow_min = float(np.nanpercentile(all_vals, 2))
            flow_max = float(np.nanpercentile(all_vals, 98))
        else:
            flow_min, flow_max = 0.0, 1.0
        if flow_max <= flow_min:
            flow_min, flow_max = 0.0, 1.0

    # 异常叠加：默认关闭（按用户要求 — Tab 2 仅展示流量本身，不显示异常红框）
    if anomaly_overlay and anomaly_masks:
        per_frame_overlay = []
        for mask in anomaly_masks:
            mask_arr = np.asarray(mask, dtype=int)
            rows, cols = np.where(mask_arr.astype(bool))
            per_frame_overlay.append((rows, cols))
        z_first_overlay = per_frame_overlay[0]
    else:
        per_frame_overlay = [(np.array([], int), np.array([], int))] * len(frames_data)
        z_first_overlay = per_frame_overlay[0]

    heatmap_trace = go.Heatmap(
        z=z0, colorscale="YlOrRd",
        zmin=flow_min, zmax=flow_max,
        colorbar=dict(title="人流量"),
    )
    # 不再画散点红框 — Tab 2 仅展示纯流量热力变化
    data_traces = [heatmap_trace]

    fig = go.Figure(
        data=data_traces,
        frames=[
            go.Frame(
                data=[go.Heatmap(z=f, zmin=flow_min, zmax=flow_max)],
                name=str(i),
                traces=[0],
            )
            for i, f in enumerate(frames_data)
        ],
    )
    steps = [
        dict(
            method="animate",
            args=[[str(i)], {"mode": "immediate",
                              "frame": {"duration": frame_ms, "redraw": True},
                              "transition": {"duration": transition_ms,
                                             "easing": "linear"}}],
            label=(timestamps[i][-8:-3] if timestamps and i < len(timestamps) and timestamps[i] else str(i)),
        )
        for i in range(len(frames_data))
    ]
    fig.update_layout(
        title=f"未来 24 小时人流预测动画（起点 t={t_start}）",
        height=560,
        # 关键：layout.transition 让 frames 之间产生连续插值动画（plotly 文档推荐）
        transition=dict(duration=transition_ms, easing="linear"),
        sliders=[dict(
            active=0, steps=steps,
            x=0.1, len=0.8, xanchor="left",
            currentvalue=dict(prefix="时间: ", visible=True, xanchor="right"),
            transition=dict(duration=transition_ms, easing="linear"),
        )],
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.15, x=0.0, xanchor="left",
            buttons=[
                dict(label="▶ 播放", method="animate",
                     args=[None, {"frame": {"duration": frame_ms, "redraw": True},
                                  "transition": {"duration": transition_ms, "easing": "linear"},
                                  "fromcurrent": True}]),
                dict(label="⏸ 暂停", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ],
        )],
        margin=dict(l=10, r=10, t=120 if autoplay else 80, b=10),
    )
    # autoplay: 标记按钮为「自动播放」并设置 fromcurrent
    if autoplay:
        try:
            fig.layout.updatemenus[0].buttons[0].args[1]["fromcurrent"] = True
            fig.add_annotation(
                text="▶ 自动播放中……（点击滑块或暂停按钮可停止）",
                xref="paper", yref="paper", x=0.5, y=1.13, showarrow=False,
                font=dict(size=11, color="gray"),
            )
        except Exception:
            pass
    fig.update_xaxes(title="列")
    fig.update_yaxes(title="行", autorange="reversed")
    return fig


def _compute_z_bounds(flow_2d):
    """计算分位数色阶边界，避免高峰/低谷饱和."""
    flat = np.asarray(flow_2d, dtype=float).flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) < 2:
        return 0.0, 1.0
    zmin = float(np.nanpercentile(flat, 2))
    zmax = float(np.nanpercentile(flat, 98))
    if zmax <= zmin:
        return 0.0, 1.0
    return zmin, zmax


def _flow_to_color(flow: float, zmin: float = 0.0, zmax: float = 1.0) -> list[int]:
    """流量 [zmin,zmax] → RGBA，线性插值，对齐 YlOrRd 色系."""
    if zmax <= zmin:
        v = 0.0
    else:
        v = (max(zmin, min(zmax, flow)) - zmin) / (zmax - zmin)
    # 线性映射: 0→淡黄, 1→深红
    r = 255
    g = int(round(255 - 240 * v))          # 255 → 15
    b = int(round(178 - 157 * v))          # 178 → 21
    alpha = int(round(40 + 80 * v))        # 40 → 120
    return [r, g, b, alpha]


def render_geo_map(cells, flow_min=None, flow_max=None, title="异常地理分布"):
    """地理地图：PyDeck ScatterplotLayer 渲染北京 32×32 网格点 + 异常高亮。"""

    if not cells:
        st.info("无网格数据")
        return
    df = pd.DataFrame(cells)
    df = df[df["lat"] > 0]
    df = df[df["lon"] > 0]

    # 异常 / 正常分开
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    anom = df[df["is_anomaly"]]
    normal = df[~df["is_anomaly"]]

    st.caption(f"正常 {len(normal)} 格 · 异常 {len(anom)} 格 · 总计 {len(df)} 格")

    # 北京主要地标（在底图上标注）
    landmarks = pd.DataFrame([
        {"name": "天安门", "lat": 39.907, "lon": 116.397},
        {"name": "故宫",   "lat": 39.913, "lon": 116.398},
        {"name": "奥体中心", "lat": 40.002, "lon": 116.397},
        {"name": "CBD",    "lat": 39.908, "lon": 116.458},
        {"name": "中关村", "lat": 39.983, "lon": 116.312},
        {"name": "望京",   "lat": 39.996, "lon": 116.470},
        {"name": "颐和园", "lat": 39.999, "lon": 116.275},
    ])

    # 地理范围（32×32 网格，对齐北京城市建成区）
    # 南：南四环 39.74°N | 北：北四环 40.05°N
    # 西：西五环 116.10°E | 东：东六环 116.60°E
    GRID_LON_MIN, GRID_LON_MAX = 116.10, 116.60
    GRID_LAT_MIN, GRID_LAT_MAX = 39.74,  40.05

    # 卫星底图文件（已 resize 到网格比例 1792×965）
    # 兼容多目录（streamlit 启动目录可能不在 e:/amazon/）：
    # 1. 当前工作目录 + 相对文件名
    # 2. e:/amazon/ 绝对路径
    # 3. __file__ 推导路径
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.getcwd(),
        "e:/amazon",
        "E:\\amazon",
        _here,
        os.path.dirname(_here),
        os.path.dirname(os.path.dirname(_here)),
    ]
    SAT_B64_PATH = None
    MPL_B64_PATH = None
    for base in _candidates:
        cand = os.path.join(base, "_bj_satellite_b64.txt")
        if os.path.exists(cand):
            SAT_B64_PATH = cand
            break
    for base in _candidates:
        cand = os.path.join(base, "_bj_base64.txt")
        if os.path.exists(cand):
            MPL_B64_PATH = cand
            break
    # 底图已裁剪对齐网格，四角完全贴合
    SAT_LON_MIN, SAT_LON_MAX = GRID_LON_MIN, GRID_LON_MAX
    SAT_LAT_MIN, SAT_LAT_MAX = GRID_LAT_MIN, GRID_LAT_MAX

    # ── P2: 显示/隐藏正常网格开关 ────────────────────────────────────────
    show_normal = st.checkbox(
        "显示人流热力格点（默认关闭，突出底图）",
        value=False,
        key="toggle_show_normal_geo",
    )

    cache_key = "bj_sat_mpl"
    if cache_key not in st.session_state:
        sat_b64, mpl_b64 = None, None
        if SAT_B64_PATH:
            try:
                with open(SAT_B64_PATH, "r") as f:
                    sat_b64 = f.read().strip()
                st.session_state["_sat_path"] = SAT_B64_PATH
            except Exception as e:
                st.session_state["_sat_err"] = str(e)
        if MPL_B64_PATH:
            try:
                with open(MPL_B64_PATH, "r") as f:
                    mpl_b64 = f.read().strip()
                st.session_state["_mpl_path"] = MPL_B64_PATH
            except Exception as e:
                st.session_state["_mpl_err"] = str(e)
        # 优先卫星底图（更直观），缺失时用 mpl 离线路网兜底
        st.session_state[cache_key] = sat_b64 or mpl_b64
        st.session_state["_bg_type"] = "sat" if sat_b64 else ("mpl" if mpl_b64 else "none")

    b64_bg = st.session_state.get(cache_key)
    bg_type = st.session_state.get("_bg_type", "mpl")

    # ── 调试面板：帮助确认底图是否加载成功 ──────────────────────────────────
    if bg_type == "sat":
        st.caption(
            f"🛰️ 卫星底图已加载 · {len(b64_bg)//1024} KB · 路径："
            f"`{st.session_state.get('_sat_path','?')}`"
        )
    elif bg_type == "mpl":
        st.caption(
            f"📍 离线底图已加载 · {len(b64_bg)//1024} KB · 路径："
            f"`{st.session_state.get('_mpl_path','?')}`"
        )
    else:
        st.error(
            "❌ 两个底图文件都未找到。已尝试路径：\n"
            f"- 卫星：`{SAT_B64_PATH}`\n"
            f"- 离线：`{MPL_B64_PATH}`\n\n"
            "请把 `_bj_satellite_b64.txt` 和 `_bj_base64.txt` 放到 streamlit 启动目录下。"
        )

    # ── Plotly 主图 ───────────────────────────────────────────────────────
    fig = go.Figure()

    if b64_bg and bg_type == "sat":
        # ── P1: 轴范围精确对齐网格四角 ──────────────────────────────────
        # 卫星底图完整填充整个 plot 区域（sizing=stretch 保持原始宽高比）
        fig.update_layout(
            images=[dict(
                source=f"data:image/png;base64,{b64_bg}",
                xref="x", yref="y",
                x=SAT_LON_MIN, y=SAT_LAT_MAX,
                sizex=SAT_LON_MAX - SAT_LON_MIN,
                sizey=SAT_LAT_MAX - SAT_LAT_MIN,
                sizing="fill",             # 图已 resize 到网格比例，不再变形
                opacity=1.0,
                layer="below",
            )],
            # P1: 精确取网格四角坐标（底图四角完全贴合网格角）
            xaxis=dict(range=[GRID_LON_MIN, GRID_LON_MAX],
                       showgrid=False,           # P0: 去掉网格线
                       showticklabels=False,
                       zeroline=False, fixedrange=False),
            yaxis=dict(range=[GRID_LAT_MIN, GRID_LAT_MAX],
                       showgrid=False,           # P0: 去掉网格线
                       showticklabels=False,
                       zeroline=False, fixedrange=False,
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="rgba(0,0,0,1)",
            paper_bgcolor="white",
            margin=dict(l=4, r=4, t=4, b=4),
            height=460,
            dragmode=False,
        )
    elif b64_bg:
        fig.update_layout(
            images=[dict(
                source=f"data:image/png;base64,{b64_bg}",
                xref="x", yref="y",
                x=GRID_LON_MIN, y=GRID_LAT_MAX,
                sizex=GRID_LON_MAX - GRID_LON_MIN,
                sizey=GRID_LAT_MAX - GRID_LAT_MIN,
                sizing="fill",
                opacity=1.0,
                layer="below",
            )],
            xaxis=dict(range=[GRID_LON_MIN, GRID_LON_MAX],
                       showgrid=False, zeroline=False, fixedrange=False),
            yaxis=dict(range=[GRID_LAT_MIN, GRID_LAT_MAX],
                       showgrid=False, zeroline=False, fixedrange=False,
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="rgba(244,244,244,1)",
            paper_bgcolor="white",
            margin=dict(l=4, r=4, t=4, b=4),
            height=460,
            dragmode=False,
        )
    else:
        fig.update_layout(
            xaxis=dict(range=[GRID_LON_MIN, GRID_LON_MAX],
                       showgrid=True, gridcolor="rgba(220,220,220,0.6)",
                       title="经度", tickformat=".2f", fixedrange=False),
            yaxis=dict(range=[GRID_LAT_MIN, GRID_LAT_MAX],
                       showgrid=True, gridcolor="rgba(220,220,220,0.6)",
                       title="纬度", tickformat=".2f",
                       scaleanchor="x", scaleratio=1, fixedrange=False),
            plot_bgcolor="rgba(248,248,248,1)",
            paper_bgcolor="white",
            margin=dict(l=50, r=20, t=20, b=50),
            height=460,
            dragmode=False,
        )

    # ── 正常格点：P0 大幅弱化——尺寸小、透明度低、只显趋势 ───────────────
    if len(normal) and show_normal:   # P2: 默认隐藏
        f_arr = normal["flow"].values.astype(float)
        f_min, f_max = f_arr.min(), f_arr.max()
        f_range = max(f_max - f_min, 1e-9)
        norm_colors = [
            f"rgb({int(20+t*235)},{int(80+t*120)},{int(200-t*150)})"
            for t in (f_arr - f_min) / f_range
        ]
        trace_normal = go.Scatter(
            x=normal["lon"].tolist(),
            y=normal["lat"].tolist(),
            mode="markers",
            marker=dict(
                size=8,                              # P0: 13→8，缩小
                color=norm_colors,
                opacity=0.30,                       # P0: 0.88→0.30，若隐若现
                line=dict(width=0),
            ),
            text=[f"({r.row},{r.col}) flow={r.flow:.4f} score={r.score:.4f}"
                  for _, r in normal.iterrows()],
            hoverinfo="text",
            name=f"正常({len(normal)})",
        )
        fig.add_trace(trace_normal)

    # ── 异常格点：P0 强制置顶，100%不透明，白边 ─────────────────────────
    if len(anom):
        trace_anom = go.Scatter(
            x=anom["lon"].tolist(),
            y=anom["lat"].tolist(),
            mode="markers",
            marker=dict(
                size=22,
                color="rgba(220,10,10,1.0)",
                line=dict(width=3, color="rgba(255,255,255,0.95)"),
            ),
            text=[f"({r.row},{r.col}) flow={r.flow:.4f} score={r.score:.4f} ⚠️"
                  for _, r in anom.iterrows()],
            hoverinfo="text",
            name=f"异常({len(anom)})",
        )
        fig.add_trace(trace_anom)

    # ── P2: 地标文字标注（白底黑字，高对比度）───────────────────────────
    for _, lm in landmarks.iterrows():
        fig.add_trace(go.Scatter(
            x=[lm["lon"]],
            y=[lm["lat"]],
            mode="markers",
            marker=dict(size=0, opacity=0),   # 不可见
            text=[lm["name"]],
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_annotation(
            x=lm["lon"], y=lm["lat"],
            text=f"<b>{lm['name']}</b>",
            showarrow=False,
            font=dict(size=10, color="rgba(255,255,255,0.95)",
                      family="sans-serif"),
            bgcolor="rgba(30,30,30,0.75)",
            borderpad=2,
            bordercolor="rgba(255,255,255,0.5)",
            borderwidth=1,
            align="center",
            yshift=8,
            yref="y", xref="x",
        )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.88)"),
        hovermode="closest",
    )

    st.plotly_chart(fig, width='stretch')

    # ── 色阶说明 ───────────────────────────────────────────────────────
    if bg_type == "sat":
        bg_label = "🛰️ 高德卫星底图（zoom 12）"
    else:
        bg_label = "📍 北京六环底图（离线绘制）"
    st.markdown(
        "<span style='font-size:12px;'>"
        "<span style='background:rgb(20,80,200);padding:2px 6px;color:white'>深蓝</span>"
        "=低人流 &nbsp;"
        "<span style='background:rgb(130,180,200);padding:2px 6px'>青</span>"
        "=中低 &nbsp;"
        "<span style='background:rgb(255,200,50);padding:2px 6px'>黄</span>"
        "=高人流 &nbsp;|&nbsp; "
        "<span style='background:rgba(220,10,10,1);padding:2px 6px;color:white'>红圆</span>"
        "=异常格点 &nbsp;|&nbsp; "
        + bg_label +
        "</span>",
        unsafe_allow_html=True,
    )


# ── 主标题 ────────────────────────────────────────────────────────────────────
st.title("🚨 城市人流时空异常检测系统")
st.markdown("**北京 32×32 网格 · 实时热力图 · 异常预警 · 24h 预测动画**")

if not health_ok:
    st.warning("API 未连接，下方功能可能不可用。")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 实时热力图",
    "🎬 24h 预测动画",
    "📋 异常事件",
    "🔍 单网格详情",
    "🗺️ 地理地图",
])


# ── Tab 1: 实时热力图 ────────────────────────────────────────────────────────
with tab1:
    st.subheader("实时热力图（接真实 API）")

    # 显著的全宽警示音启用提示（视觉权重最高）
    _audio_unlocked_tab = st.session_state.get("_audio_unlocked", False)
    if not _audio_unlocked_tab:
        st.warning(
            "🔇 **浏览器禁止自动播放声音**。"
            "**首次点击下方任一按钮**即可自动启用警示音（Tab 2/3/4 报警时也会发声）。",
            icon="⚠️",
        )
    else:
        st.success("🔔 警示音已启用 · Tab 2/3/4 报警时会发出提示音", icon="✅")

    col_t, col_info = st.columns([3, 1])

    with col_t:
        # 快捷按钮：跳到三个典型时段（在 slider 之前处理，否则无法修改已实例化的 widget）
        # 值来自 API /api/timeslots 动态获取；失败时回退到默认偏移（+6/+16/+36）
        t_night   = st.session_state.get("_t_night",   TEST_T_MIN + 6)
        t_morning = st.session_state.get("_t_morning", TEST_T_MIN + 16)
        t_evening = st.session_state.get("_t_evening", TEST_T_MIN + 36)
        # API 返回的 hour_estimate（动态），用于按钮标签和 slider 旁的小字
        h_night   = st.session_state.get("_h_night",   3)
        h_morning = st.session_state.get("_h_morning", 10)
        h_evening = st.session_state.get("_h_evening", 17)
        fallback_warn = not all(k in st.session_state for k in ["_t_night", "_t_morning", "_t_evening"])
        if fallback_warn:
            st.caption("⚠️ 动态时间槽获取失败，使用默认偏移")

        b1, b2, b3 = st.columns(3)
        if b1.button(f"夜间低谷 (~{_format_hour_label(h_night)})", key="t1_jump_night"):
            st.session_state["t1_slider"] = t_night
            st.session_state["_audio_unlocked"] = True
            st.session_state["_audio_unlock_ts"] = time.time()
            st.rerun()
        if b2.button(f"早高峰 (~{_format_hour_label(h_morning)})", key="t1_jump_morning"):
            st.session_state["t1_slider"] = t_morning
            st.session_state["_audio_unlocked"] = True
            st.session_state["_audio_unlock_ts"] = time.time()
            st.rerun()
        if b3.button(f"晚高峰 (~{_format_hour_label(h_evening)})", key="t1_jump_evening"):
            st.session_state["t1_slider"] = t_evening
            st.session_state["_audio_unlocked"] = True
            st.session_state["_audio_unlock_ts"] = time.time()
            st.rerun()

        col_step, col_auto, col_audio = st.columns([1, 2, 2])
        with col_step:
            st.checkbox("🔄 自动快进", key="auto_refresh", on_change=_on_auto_refresh_change)
        with col_auto:
            if st.button("▶ 单步推进", key="t1_step"):
                st.session_state["t1_slider"] = min(
                    st.session_state.get("t1_slider", TEST_T_MIN + 200) + 6, TEST_T_MAX
                )
                # 任何点击都视为"用户交互"，立即解锁音频
                st.session_state["_audio_unlocked"] = True
                st.session_state["_audio_unlock_ts"] = time.time()
                st.rerun()
        with col_audio:
            _unlocked = st.session_state.get("_audio_unlocked", False)
            if _unlocked:
                st.success("🔔 警示音已启用", icon="✅")
            else:
                if st.button("🔔 启用警示音（首次必须点击）", key="t1_unlock_audio"):
                    st.session_state["_audio_unlocked"] = True
                    st.session_state["_audio_unlock_ts"] = time.time()
                    # 立即播一声短的 660Hz 让浏览器 autoplay 通道开通
                    freq = 660
                    beep_b64 = _beep_wav_b64(freq=freq, duration_ms=120)
                    components.html(
                        f"""<audio autoplay>
                        <source src="data:audio/wav;base64,{beep_b64}" type="audio/wav">
                        </audio>""",
                        height=0,
                    )
                    st.rerun()

        # 默认 step=6（3 小时一跳），避免在同色系里逐帧拖看不出差异
        t_step = st.slider(
            "选择时间步",
            min_value=TEST_T_MIN, max_value=TEST_T_MAX,
            step=6, key="t1_slider",
        )
    with col_info:
        st.metric("当前 t", t_step)
        # 当前 t 在测试集内的偏移（只是索引，不代表真实小时数）
        st.caption(f"测试集偏移: t−t_min = {t_step - TEST_T_MIN} 步（每个步 30 分钟）")

    with st.spinner("拉取数据..."):
        det = _detect_cached(t_step, mode_param)

    if det:
        # 兼容新旧 API：若没有 heatmap/cells，从 fused_scores 构造
        if "heatmap" in det and det["heatmap"]:
            flow_2d = np.array(det["heatmap"])
            mask_2d = np.array(det.get("anomaly_mask") or np.zeros_like(flow_2d, dtype=int))
        else:
            flow_2d = np.zeros((GRID_H, GRID_W))
            mask_2d = np.zeros((GRID_H, GRID_W), dtype=int)
            for fs in det.get("fused_scores", []):
                gid = fs["grid_id"]
                r, c = gid // GRID_W, gid % GRID_W
                flow_2d[r, c] = fs["score"]
                mask_2d[r, c] = 1 if fs.get("is_anomaly") else 0

        # cells：用于经纬度
        cells = det.get("cells") or []
        if not cells and det.get("fused_scores"):
            for fs in det["fused_scores"]:
                gid = fs["grid_id"]
                lon = 116.10 + (gid % GRID_W + 0.5) * (116.60 - 116.10) / GRID_W
                lat = 39.79 + (GRID_H - 1 - gid // GRID_W + 0.5) * (40.05 - 39.79) / GRID_H
                cells.append({
                    "grid_id": gid, "row": gid // GRID_W, "col": gid % GRID_W,
                    "lon": round(lon, 6), "lat": round(lat, 6),
                    "flow": float(flow_2d[gid // GRID_W, gid % GRID_W]),
                    "score": fs["score"], "is_anomaly": fs.get("is_anomaly", False),
                })

        scores_2d = np.zeros_like(flow_2d)
        for c in cells:
            scores_2d[c["row"], c["col"]] = c["score"]

        zmin, zmax = _compute_z_bounds(flow_2d)

        ts_label = det.get("timestamp") or f"测试集第 {t_step - TEST_T_MIN} 小时"
        fig = plot_heatmap_32(
            flow_2d, mask_2d, scores_2d,
            title=f"时间步 {t_step}{(' · ' + ts_label) if ts_label else ''}",
        )
        st.plotly_chart(fig, width='stretch')

        # 预警状态（一般仅指标，紧急/重要才弹banner）
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("异常格点", f"{det['n_anomaly_cells']}", delta=f"{det['anomaly_rate']*100:.1f}%")
        c2.metric("异常率", f"{det['anomaly_rate']*100:.2f}%")
        c3.metric("预警等级", f"{det.get('warning_name') or '无'}")
        c4.metric("处理耗时", f"{det.get('processing_ms', 0):.0f}ms")
        st.markdown("---")
        show_alert_banner(
            det.get("warning_level"),
            det.get("warning_name"),
            f"异常格点 {det['n_anomaly_cells']}/{det['n_total_cells']}（{det['anomaly_rate']*100:.2f}%）",
            t=t_step,
        )





# ── Tab 2: 24h 预测动画 ──────────────────────────────────────────────────────
with tab2:
    st.subheader("未来 24 小时人流预测动画")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        anim_t_start = st.number_input(
            "起始时间步",
            min_value=TEST_T_MIN, max_value=TEST_T_MAX - 24,
            value=TEST_T_MIN + 100, step=1, key="anim_start",
        )
    with col_b:
        st.caption("动画长度：24 小时（每小时一帧）")

    if st.button("▶ 生成动画", key="gen_anim"):
        st.info("💡 提示：动画会自动播放，点击图表左下角 ⏸ 暂停按钮可停止，或用滑块跳到任意时刻")
        with st.spinner("拉取 24 步数据..."):
            res = build_24h_frames(int(anim_t_start), int(anim_t_start) + 23)
        if res:
            frames_data, anomaly_masks, timestamps = res
            all_vals = np.concatenate([np.asarray(f, dtype=float).flatten() for f in frames_data])
            all_vals = all_vals[~np.isnan(all_vals)]
            if len(all_vals) >= 2:
                flow_min = float(np.nanpercentile(all_vals, 2))
                flow_max = float(np.nanpercentile(all_vals, 98))
            else:
                flow_min, flow_max = 0.0, 1.0
            if flow_max <= flow_min:
                flow_min, flow_max = 0.0, 1.0
            fig = make_animation_fig(
                frames_data, anomaly_masks, timestamps,
                int(anim_t_start), flow_min, flow_max,
                anomaly_overlay=False,   # 按用户要求：24h 动画不显示异常红框，仅展示流量
            )
            if fig:
                # 用 components.html 注入一段小脚本：渲染后自动点「▶ 播放」
                # 这是 plotly 官方推荐的「自动播放」模式，比改 layout 更可靠
                st.plotly_chart(fig, width='stretch', key="anim_fig")
                # 真正的"循环播放" + 兜底自动点 ▶
                # Plotly 没有原生 loop（播完最后一帧就停），靠 JS 监听 plotly_afterplot
                # 在每帧切换时递增计数，到达 24 帧后调 gd.restart() 让动画从头播放
                components.html(
                    """<script>
                    (function(){
                      const tryHook = (attempt) => {
                        if (attempt > 60) return;
                        const iframes = window.parent.document.querySelectorAll('iframe');
                        for (const f of iframes) {
                          try {
                            const doc = f.contentDocument || f.contentWindow.document;
                            const gd = doc.querySelector('.js-plotly-plot');
                            if (!gd || !gd.data || !gd._fullLayout) {
                              setTimeout(() => tryHook(attempt+1), 250);
                              return;
                            }
                            const N = (gd.data[0].frames || []).length;
                            if (N === 0) {
                              setTimeout(() => tryHook(attempt+1), 250);
                              return;
                            }
                            // 启动播放
                            const playBtn = doc.querySelector('.modebar-btn[data-title="Play"]')
                                          || Array.from(doc.querySelectorAll('.modebar-btn'))
                                               .find(b => (b.getAttribute('data-title')||'').includes('Play'));
                            if (playBtn) playBtn.click();

                            // 监听"动画到最后一帧"→ 重置到 0 重新播
                            let frameIdx = 0;
                            gd.on('plotly_frame', () => {
                              frameIdx++;
                              if (frameIdx >= N) {
                                frameIdx = 0;
                                Plotly.animate(gd, [0], {mode: 'immediate',
                                                         frame: {duration: 0, redraw: false},
                                                         transition: {duration: 0}});
                                // 重新点 ▶ 让它继续循环
                                setTimeout(() => {
                                  const btn = doc.querySelector('.modebar-btn[data-title="Play"]')
                                           || Array.from(doc.querySelectorAll('.modebar-btn'))
                                                .find(b => (b.getAttribute('data-title')||'').includes('Play'));
                                  if (btn) btn.click();
                                }, 50);
                              }
                            });
                            return;
                          } catch(e) {}
                        }
                        setTimeout(() => tryHook(attempt+1), 300);
                      };
                      setTimeout(() => tryHook(0), 1500);
                    })();
                    </script>""",
                    height=0,
                )
                st.caption(f"已生成 {len(frames_data)} 帧 · 拖动下方滑块或点击播放按钮")
            else:
                st.error("动画构造失败")
        else:
            st.error("数据获取失败")
    else:
        st.info("点击「生成动画」按钮加载 24 步数据并播放")


# ── Tab 3: 异常事件 ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("历史异常事件查询")

    c1, c2, c3 = st.columns(3)
    with c1:
        t_start = st.number_input("起始 t", min_value=TEST_T_MIN, max_value=TEST_T_MAX,
                                   value=TEST_T_MIN, step=10, key="ev_t1")
    with c2:
        t_end = st.number_input("结束 t", min_value=TEST_T_MIN, max_value=TEST_T_MAX,
                                 value=TEST_T_MAX, step=10, key="ev_t2")
    with c3:
        min_cells = st.number_input("最少网格数", min_value=1, max_value=1024,
                                     value=9, step=1, key="ev_mincells")

    level_opts = st.multiselect(
        "预警等级",
        options=[(1, "一般"), (2, "重要"), (3, "紧急")],
        default=[(1, "一般"), (2, "重要"), (3, "紧急")],
        format_func=lambda x: f"{x[0]}级-{x[1]}",
        key="ev_levels",
    )

    include_marginal = st.checkbox(
        "包含零散/瞬时异常（patch_marginal / point_single）",
        value=False,
        key="ev_include_marginal",
        help="默认仅显示时空连续的聚合异常事件；开启后可查看所有兜底入库的零散异常，"
             "与热力图实时异常格点一一对应。",
    )

    st.caption(
        "💡 热力图红点为点位级原始检测结果全量展示；"
        "事件列表默认仅显示时空连续的聚合异常事件。"
        "零散/瞬时异常请开启上方选项查看。"
    )

    if st.button("🔍 查询事件", key="query_events"):
        with st.spinner("查询中..."):
            result = call_api("/api/anomaly/events", "POST", json_data={
                "t_start": t_start, "t_end": t_end, "min_cells": min_cells,
                "include_marginal": include_marginal,
            })
        if result and result.get("events"):
            events = result["events"]
            level_ids = [x[0] for x in level_opts]
            # 等级过滤仅对 spatial_sustained 生效，零散事件不受等级限制
            events = [e for e in events
                      if e.get("event_type") != "spatial_sustained"
                      or e.get("warning_level") in level_ids]

            st.success(f"找到 {len(events)} 个异常事件（默认仅显示时空聚合事件）")

            level_counts = {1: 0, 2: 0, 3: 0}
            for e in events:
                lv = e.get("warning_level", 0)
                if lv in level_counts:
                    level_counts[lv] += 1

            m1, m2, m3 = st.columns(3)
            m1.metric("一般", level_counts.get(1, 0))
            m2.metric("重要", level_counts.get(2, 0))
            m3.metric("紧急", level_counts.get(3, 0))

            # 事件类型映射（英文→中文）
            type_map = {
                "spatial_sustained": "时空聚合",
                "patch_marginal": "瞬时连片",
                "point_single": "单点高分",
            }
            df = pd.DataFrame([{
                "ID": e["event_id"],
                "起始": e["t_start"],
                "结束": e["t_end"],
                "持续(步)": e["duration"],
                "网格数": e["n_cells"],
                "中心": f"({e['center_row']},{e['center_col']})",
                "类型": type_map.get(e["event_type"], e["event_type"]),
                "均分": round(e["avg_score"], 3),
                "预警": f"{e['warning_level']}级-{e.get('level_name','')}",
            } for e in events])

            st.dataframe(df, width='stretch', hide_index=True)

            st.markdown(
                "<span style='font-size:12px;'>"
                "**类型说明：** "
                "<span style='background:#e8f4e8;padding:1px 6px'>时空聚合</span>=时空连续的正式异常事件 &nbsp;"
                "<span style='background:#fff3cd;padding:1px 6px'>瞬时连片</span>=≥12格/单步连片，兜底收录 &nbsp;"
                "<span style='background:#f8d7da;padding:1px 6px'>单点高分</span>=≥0.85分孤立点，兜底收录"
                "</span>",
                unsafe_allow_html=True,
            )
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下载 CSV", csv, "anomaly_events.csv",
                               "text/csv", key="dl_events")
        else:
            st.info("该区间内未找到异常事件")


# ── Tab 4: 单网格详情 ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("单网格时序分析")

    col_g, col_r = st.columns([1, 3])
    with col_g:
        grid_id = st.number_input("网格编号", min_value=0, max_value=1023,
                                   value=512, step=1, key="grid_id")
    with col_r:
        col_start, col_end = st.columns(2)
        with col_start:
            ts_start = st.number_input("起始 t", min_value=TEST_T_MIN,
                                        max_value=TEST_T_MAX - 1,
                                        value=TEST_T_MIN, step=10, key="ts_s")
        with col_end:
            ts_end = st.number_input("结束 t", min_value=TEST_T_MIN,
                                      max_value=TEST_T_MAX,
                                      value=min(TEST_T_MIN + 168, TEST_T_MAX),
                                      step=10, key="ts_e")

    if ts_end - ts_start > 168:
        st.warning("时间范围过大，已截断为 168 步")
        ts_end = ts_start + 168

    if st.button("📈 绘制时序图", key="plot_ts"):
        with st.spinner("拉取数据..."):
            fig, values, ts_labels, anomaly_marks = plot_timeseries_real(
                int(grid_id), int(ts_start), int(ts_end),
            )
        if fig:
            st.plotly_chart(fig, width='stretch')
            row, col = int(grid_id) // GRID_W, int(grid_id) % GRID_W
            st.caption(
                f"网格 ({row},{col}) · 时间范围 [{ts_start}, {ts_end}] · "
                f"{int(ts_end) - int(ts_start) + 1} 步 · 异常点 {sum(1 for _, c in anomaly_marks if c == 'red')}"
            )
        else:
            st.error("数据获取失败")


# ── Tab 5: 地理地图 ──────────────────────────────────────────────────────────
with tab5:
    st.subheader("异常地理分布（经纬度坐标散点图）")

    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        map_t = st.slider("选择时间步", min_value=TEST_T_MIN, max_value=TEST_T_MAX,
                           value=TEST_T_MIN + 200, step=1, key="map_t")
    with col_m2:
        st.caption("红色=异常格点")

    with st.spinner("加载地图..."):
        det = call_api("/api/anomaly/detect", "POST",
                        json_data={"t": map_t, "mode": mode_param})

    if det:
        cells = det.get("cells") or []
        st.caption(f"调试: API返回 {len(cells)} 个格点, keys={list(det.keys())[:6]}")

        if not cells and det.get("fused_scores"):
            for fs in det["fused_scores"]:
                gid = fs["grid_id"]
                lon = 116.10 + (gid % GRID_W + 0.5) * (116.60 - 116.10) / GRID_W
                lat = 39.79 + (GRID_H - 1 - gid // GRID_W + 0.5) * (40.05 - 39.79) / GRID_H
                cells.append({
                    "grid_id": gid, "row": gid // GRID_W, "col": gid % GRID_W,
                    "lon": round(lon, 6), "lat": round(lat, 6),
                    "flow": fs["score"],
                    "score": fs["score"], "is_anomaly": fs.get("is_anomaly", False),
                })
        if cells:
            render_geo_map(cells, title=f"t={map_t} 异常分布")
            st.caption(f"异常格点: {det['n_anomaly_cells']}/{det['n_total_cells']}")
        else:
            st.error("地图数据加载失败")


# ── 预警历史侧栏 ─────────────────────────────────────────────────────────────
with st.sidebar.expander("🚨 预警历史", expanded=False):
    if st.session_state.alert_history:
        st.table(pd.DataFrame(st.session_state.alert_history))
        if st.button("清空", key="clear_alerts"):
            st.session_state.alert_history = []
            st.rerun()
    else:
        st.caption("暂无预警记录")

if st.sidebar.button("🔇 重置预警状态"):
    st.session_state.last_warning_level = 0
    st.session_state._last_alert_t = None
    st.rerun()


# ── 页脚 ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "城市人流时空异常检测系统 · Week6 工程化交付 · "
    "基于 week1-5 算法积累 · FastAPI + Streamlit + MapLibre"
)