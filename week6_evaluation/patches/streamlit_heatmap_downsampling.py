"""Week6 任务4 Patch 3：Streamlit 热力图降采样渲染

⚠️ 已根据 [Visualization Bug Agent] (2026-07-28) 的诊断结论修订：
- Plotly Heatmap 的 yaxis.autorange="reversed" 会自动把矩阵 row 0 显示在图顶，
  np.where 返回的 rows 直接作为 y 坐标即可，**不需要翻转**。
- 早先版本曾误改成 y=H-1-rows，导致红框偏转半张图，已回退到原始公式。

应用方式：
    from week6_evaluation.patches.streamlit_heatmap_downsampling import plot_heatmap_32_fast
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def plot_heatmap_32_fast(
    flow_2d: np.ndarray,
    anomaly_mask_2d: Optional[np.ndarray] = None,
    scores_2d: Optional[np.ndarray] = None,
    title: str = "",
    downsample: int = 2,
):
    """32×32 → 16×16 降采样热力图 + 16×16 降采样异常格点

    Args:
        flow_2d: 32×32 流量矩阵
        anomaly_mask_2d: 32×32 异常掩码（可选）
        scores_2d: 32×32 异常分数（可选）
        title: 标题
        downsample: 降采样因子（2 = 32→16）

    Returns:
        Plotly Figure
    """
    import plotly.graph_objects as go

    H, W = flow_2d.shape
    ds = downsample
    # 流量降采样：max-pool（保留峰值信息）
    flow_ds = flow_2d.reshape(H // ds, ds, W // ds, ds).max(axis=(1, 3))

    flat = flow_ds.flatten()
    if len(flat) > 0:
        zmin = float(np.percentile(flat, 5))
        zmax = float(np.percentile(flat, 95))
    else:
        zmin, zmax = 0.0, 1.0

    H_ds = H // ds
    W_ds = W // ds

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=flow_ds,
        colorscale="YlOrRd",
        zmin=zmin, zmax=zmax,
        showscale=True,
        colorbar=dict(title="人流量", x=1.02, len=0.7),
        hovertemplate="row:%{y}<br>col:%{x}<br>flow:%{z:.3f}<extra></extra>",
        name="人流",
    ))

    # 异常格点：降采样 mask 到 16×16 后正常绘制
    if anomaly_mask_2d is not None and np.any(anomaly_mask_2d):
        mask_f = anomaly_mask_2d.astype(np.float32)
        mask_ds = mask_f.reshape(H_ds, ds, W_ds, ds).max(axis=(1, 3)) >= 0.5
        rows_ds, cols_ds = np.where(mask_ds)

        # hover 用降采样后的分数
        custom_data = None
        if scores_2d is not None:
            scores_ds = scores_2d.reshape(H_ds, ds, W_ds, ds).max(axis=(1, 3))
            custom_data = np.stack([scores_ds[rows_ds, cols_ds]], axis=-1)

        # y=rows_ds 直接正确（autorange=reversed 下 row 0 自动在顶）
        fig.add_trace(go.Scatter(
            x=cols_ds, y=rows_ds, mode="markers",
            marker=dict(
                size=12, color="rgba(0,0,0,0)",
                symbol="square",
                line=dict(width=2, color="red"),
            ),
            customdata=custom_data,
            hovertemplate="异常 row:%{y} col:%{x}<br>score:%{customdata[0]:.3f}<extra></extra>",
            name="异常格点",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        height=400,
        xaxis=dict(scaleanchor="y", constrain="domain", title="列"),
        yaxis=dict(scaleanchor="x", constrain="domain", title="行", autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def patch_app():
    """替换 week6/app.py 的 plot_heatmap_32"""
    import week6.app as app_module
    app_module.plot_heatmap_32 = plot_heatmap_32_fast
    print("[Patch] plot_heatmap_32 replaced with downsampling version")


if __name__ == "__main__":
    print("=== streamlit downsampling patch 自检 ===")
    np.random.seed(42)
    flow = np.random.rand(32, 32).astype(np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[5, 5] = mask[10, 15] = mask[20, 8] = True
    mask[0, 31] = mask[31, 0] = True
    scores = np.random.rand(32, 32).astype(np.float32)

    fig = plot_heatmap_32_fast(flow, mask, scores, "Test")
    print(f"  traces: {len(fig.data)}")
    scat = fig.data[1]
    print(f"  scatter x range: [{min(scat.x)}, {max(scat.x)}] (expect 0-15)")
    print(f"  scatter y range: [{min(scat.y)}, {max(scat.y)}] (expect 0-15)")