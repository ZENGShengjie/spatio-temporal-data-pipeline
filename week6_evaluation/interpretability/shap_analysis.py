"""Week6 任务3-2：SHAP 特征重要性分析

目标：
- 选 3 个代表性网格（核心高流量 / 郊区低流量 / 高频异常）
- 计算 SHAP 值
- 输出全局特征重要性 + 单样本瀑布图

设计原则（采纳你朋友方案 + 我们的调整）：
- 不用全量 1024 网格 × 600 步，只取代表性网格的子集
- 背景数据用训练集前 100 步，**不混测试集**（防数据泄露）
- 50-100 个测试样本足够
- 因果性：背景用训练集；样本用测试集但按时序取，保证没有"未来信息归因过去"

执行（EC2）：
    pip install shap
    python -m week6_evaluation.interpretability.shap_analysis \\
        --weights week4/weights/stf_optuna.pth \\
        --output week6_evaluation/results/interpretability/shap/
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "week4"))
sys.path.insert(0, str(_REPO / "week6_evaluation" / "interpretability"))

os.environ.setdefault("WEEK4_DIR", str(_REPO / "week4"))
os.environ.setdefault("BJ_DATA_DIR", "/home/ubuntu/data")

import week4.config as cfg
from week4.models.stf_model import SpacetimeformerLite
from week6_evaluation.optimization.optuna_stf import STFSearchData
from _shap_helpers import _reconstruct_x_node  # noqa: E402


# ── 特征分组 ────────────────────────────────────────────────────────────────

# 输入特征构成（参考 STF / SeqDataset）：
#   - 前 1024 维：flow_total (N 个网格的当前流量)
#   - 后续 5 维：time_features（hour_sin, hour_cos, is_weekend, is_holiday, weather_pressure_norm）
# 注：本任务的特征不是"per-step"，而是"全 48 步"——SHAP 是解释输入向量对单步预测的贡献
# 我们用 sequence flatten 后的 48*F 维作为 SHAP 输入

FEATURE_GROUPS = {
    "flow_recent_steps": (0, 48 * 1024),         # 历史流量（48 步 × 1024 节点）
    "time_hour_sin":     (48 * 1024, 48 * 1024 + 48),  # 每步的 hour_sin
    "time_hour_cos":     (48 * 1024 + 48, 48 * 1024 + 96),
    "time_weekend":      (48 * 1024 + 96, 48 * 1024 + 144),
    "time_holiday":      (48 * 1024 + 144, 48 * 1024 + 192),
    "time_weather":      (48 * 1024 + 192, 48 * 1024 + 240),
}


# ── 代表性网格选择 ──────────────────────────────────────────────────────────

def select_representative_grids(y_test: np.ndarray) -> Dict[str, int]:
    """根据平均流量选 3 个代表性网格

    Returns:
        {"high_flow": grid_id, "low_flow": grid_id, "anomaly_prone": grid_id}
    """
    # 1. 高流量网格（核心区）
    grid_means = y_test.mean(axis=0)  # (N,)
    high_flow_id = int(np.argmax(grid_means))
    print(f"  [SHAP] 高流量网格 {high_flow_id}: 平均流量 {grid_means[high_flow_id]:.3f}")

    # 2. 低流量网格（郊区，但非 0）
    sorted_ids = np.argsort(grid_means)
    # 跳过接近 0 的
    nonzero = sorted_ids[grid_means[sorted_ids] > 0.01]
    low_flow_id = int(nonzero[0]) if len(nonzero) > 0 else int(sorted_ids[0])
    print(f"  [SHAP] 低流量网格 {low_flow_id}: 平均流量 {grid_means[low_flow_id]:.3f}")

    # 3. 高频异常网格（异常率最高）
    # 简单代理：流量方差最大的
    grid_vars = y_test.var(axis=0)
    anomaly_prone_id = int(np.argmax(grid_vars))
    print(f"  [SHAP] 高波动网格 {anomaly_prone_id}: 方差 {grid_vars[anomaly_prone_id]:.4f}")

    return {
        "high_flow": high_flow_id,
        "low_flow": low_flow_id,
        "anomaly_prone": anomaly_prone_id,
    }


# ── 包装模型为单网格预测函数 ──────────────────────────────────────────────

class SingleCellWrapper(nn.Module):
    """把 STF 包装成"只输出指定 grid_id 预测"的模型"""
    def __init__(self, stf_model: SpacetimeformerLite, grid_id: int):
        super().__init__()
        self.stf = stf_model
        self.grid_id = grid_id

    def forward(self, x_node: torch.Tensor) -> torch.Tensor:
        out = self.stf(x_node)  # (B, N, 1)
        return out[:, self.grid_id, 0]  # (B,)


# ── SHAP 计算 ──────────────────────────────────────────────────────────────

def _aggregate_features(x_raw_batch: np.ndarray, n_nodes: int, target_grid: int) -> np.ndarray:
    """把 (B, 2N+5) raw 降维到 (B, K) 特征，便于 SHAP 计算

    特征构造（K=10）：
      - target_grid_in:        目标网格入流量
      - target_grid_out:       目标网格出流量
      - city_avg_in:           全城入流量均值
      - city_avg_out:          全城出流量均值
      - neighbor_in_mean:      目标格 3x3 邻域入流量均值
      - neighbor_out_mean:     目标格 3x3 邻域出流量均值
      - hour_sin, hour_cos, is_weekend, is_holiday: 时间特征
    """
    B = x_raw_batch.shape[0]
    in_flow = x_raw_batch[:, :n_nodes]                       # (B, N)
    out_flow = x_raw_batch[:, n_nodes:2*n_nodes]              # (B, N)
    tf = x_raw_batch[:, 2*n_nodes:2*n_nodes+5]                # (B, 5)

    # 邻域 3x3 索引（目标 grid_id 的 row, col 在 32x32 grid 中）
    side = int(np.sqrt(n_nodes))
    row = target_grid // side
    col = target_grid % side
    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r = row + dr
            c = col + dc
            if 0 <= r < side and 0 <= c < side:
                neighbors.append(r * side + c)
    neighbors = list(set(neighbors))
    neighbor_in = in_flow[:, neighbors].mean(axis=1)         # (B,)
    neighbor_out = out_flow[:, neighbors].mean(axis=1)

    agg = np.stack([
        in_flow[:, target_grid],
        out_flow[:, target_grid],
        in_flow.mean(axis=1),
        out_flow.mean(axis=1),
        neighbor_in,
        neighbor_out,
        tf[:, 0], tf[:, 1], tf[:, 2], tf[:, 3],
    ], axis=1)  # (B, 10)

    return agg


_FEATURE_LABELS = [
    "target_grid_in",
    "target_grid_out",
    "city_avg_in",
    "city_avg_out",
    "neighbor_in_mean",
    "neighbor_out_mean",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_holiday",
]


def _build_x_node(x_agg_batch: np.ndarray, n_nodes: int, target_grid: int, n_tf: int = 5) -> torch.Tensor:
    """把降维后的特征还原为 STF 输入 x_node: (B, N, 7, T=1)

    还原策略：
      - target_grid 的 in/out flow 用真实值
      - 其他节点的 in/out flow 用 city_avg
      - 邻域节点额外叠加 neighbor signal（按比例微调）
    """
    import torch
    from _shap_helpers import _reconstruct_x_node
    return _reconstruct_x_node(x_agg_batch, n_nodes, target_grid, n_tf)


def compute_shap_for_grid(
    model: SingleCellWrapper,
    n_nodes: int,
    x_train: np.ndarray,      # (T_train, 2N+5) raw from STFSearchData
    x_test: np.ndarray,       # (T_test, 2N+5) raw from STFSearchData
    n_background: int = 20,
    n_samples: int = 20,
):
    """用 shap.PermutationExplainer 计算正宗 SHAP 特征重要性

    Args:
        x_train: (T_train, 2N+5) 原始 STF 数据（用作 background）
        x_test:  (T_test, 2N+5) 原始 STF 数据（被解释的样本）
        n_background: SHAP 背景样本数（缩短以加速）
        n_samples: 被解释样本数
    """
    import shap
    import torch

    device = next(model.parameters()).device
    target_grid = model.grid_id

    # 1. 聚合到 K=10 维
    x_train_agg = _aggregate_features(x_train[:n_background], n_nodes, target_grid)
    x_test_agg = _aggregate_features(x_test[:n_samples], n_nodes, target_grid)

    # 2. 构造 numpy predict_fn（SHAP 要求 numpy 接口）
    def predict_np(x_agg_np: np.ndarray) -> np.ndarray:
        """(B, 10) -> (B,)"""
        x_agg_np = np.asarray(x_agg_np, dtype=np.float32)
        if x_agg_np.ndim == 1:
            x_agg_np = x_agg_np.reshape(1, -1)
        B = x_agg_np.shape[0]
        x_tensor = torch.from_numpy(x_agg_np).to(device)
        x_node = _build_x_node(x_tensor, n_nodes, target_grid).float().to(device)
        with torch.no_grad():
            pred = model(x_node)
        return pred.detach().cpu().numpy()

    # 3. SHAP: PermutationExplainer（对任意可调用 predict_fn 起效）
    print(f"  [SHAP] running PermutationExplainer: bg={x_train_agg.shape[0]}, test={x_test_agg.shape[0]}")
    explainer = shap.PermutationExplainer(predict_np, x_train_agg)
    # shap.Explainer 的 __call__ 返回 Explanation 对象
    # PermutationExplainer 是 shap.Explainer 子类
    shap_result = explainer(x_test_agg)  # 默认 nsamples = 2*K+1 = 21
    # shap_result: shap.Explanation 包含 .values (n_samples, K), .base_values (n_samples,)
    return shap_result, x_test_agg, _FEATURE_LABELS


# ── 可视化 ────────────────────────────────────────────────────────────────

def plot_shap_summary(shap_values: np.ndarray, feature_names: List[str], save_path: Path):
    """特征重要性条形图（按 mean |SHAP|）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_abs = np.abs(shap_values).mean(axis=0)
    # Top 20
    order = np.argsort(-mean_abs)[:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(order)), mean_abs[order][::-1], color="teal")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Top-20 Feature Importance (SHAP)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_shap_waterfall(shap_values: np.ndarray, sample_idx: int, feature_names: List[str],
                        base_value: float, save_path: Path):
    """单样本瀑布图（手绘版，避免依赖 shap.plot）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sv = shap_values[sample_idx]
    order = np.argsort(-np.abs(sv))[:10]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [feature_names[i] for i in order]
    values = sv[order]
    colors = ["red" if v > 0 else "blue" for v in values]
    ax.barh(range(len(order)), values, color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(f"Sample {sample_idx} Waterfall (base={base_value:.3f}, pred={base_value+sv.sum():.3f})", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="week4/weights/stf_optuna.pth")
    parser.add_argument("--output", default="week6_evaluation/results/interpretability/shap/")
    parser.add_argument("--n-samples", type=int, default=15)
    parser.add_argument("--n-background", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[SHAP] device={device}")

    # 加载模型
    print(f"[SHAP] 加载模型: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    params = ckpt.get("params", {"hidden": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.1})
    base_model = SpacetimeformerLite(
        in_dim=2 + 5, hidden=params["hidden"], horizon=1, n_nodes=1024,
        n_heads=params["n_heads"], n_layers=params["n_layers"], dropout=params["dropout"],
    ).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    base_model.eval()

    # 加载数据（直接传原始格式STFSearchData.x_train）
    data = STFSearchData(seq_len=48, horizon=1, batch_size=8)

    # 选代表性网格
    rep_grids = select_representative_grids(data.y_test)
    (output_dir / "representative_grids.json").write_text(
        json.dumps(rep_grids, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # x_train/x_test: (T, 2N+5) 直接传给 compute_shap_for_grid
    n_nodes = data.n_nodes
    x_train = data.x_train
    x_test = data.x_test

    results = {}
    for category, grid_id in rep_grids.items():
        print(f"\n[SHAP] 计算 {category} (grid_id={grid_id})...")
        wrapper = SingleCellWrapper(base_model, grid_id).to(device)
        wrapper.eval()

        shap_result, test_agg, feature_names = compute_shap_for_grid(
            wrapper, n_nodes, x_train, x_test,
            n_background=args.n_background, n_samples=args.n_samples,
        )
        # PermutationExplainer 返回 Explanation 对象
        if hasattr(shap_result, "values"):
            shap_values = np.asarray(shap_result.values)
            base_values = np.asarray(shap_result.base_values) if hasattr(shap_result, "base_values") else None
        else:
            shap_values = np.asarray(shap_result)
            base_values = None

        if base_values is None:
            base_value = float(shap_values.mean())
        elif base_values.ndim == 0:
            base_value = float(base_values)
        else:
            base_value = float(np.atleast_1d(base_values).mean())

        # 保存
        plot_shap_summary(
            shap_values, feature_names,
            output_dir / f"shap_summary_{category}_grid{grid_id}.png",
        )
        plot_shap_waterfall(
            shap_values, 0, feature_names, base_value,
            output_dir / f"shap_waterfall_{category}_grid{grid_id}.png",
        )

        # 数值摘要
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_features = sorted(
            [(feature_names[i], float(mean_abs[i])) for i in range(len(feature_names))],
            key=lambda x: -x[1],
        )[:10]
        results[category] = {
            "grid_id": int(grid_id),
            "top_10_features": top_features,
            "mean_abs_shap": float(mean_abs.mean()),
        }
        print(f"[SHAP] {category}: top feature={top_features[0][0]}, mean_abs={mean_abs.mean():.6f}")

    (output_dir / "shap_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n[SHAP] 完成 → {output_dir}")


if __name__ == "__main__":
    main()
