"""Week6 任务3-1：STF 注意力可视化

目标：
- 提取 Env Transformer 自注意力权重
- 提取 Cross-Attention 权重
- 筛选有效 head（按方差）
- 输出 32×32 空间注意力 + 48 步时间注意力

设计：
- 用 forward_hook 提取注意力，不修改模型代码
- 默认前 100 个测试样本（仅 100 步推理，几分钟完成）
- 选 Top-3 重要 head 分别可视化

执行（EC2）：
    python -m week6_evaluation.interpretability.attention_vis \\
        --weights week4/weights/stf_optuna.pth \\
        --output week6_evaluation/results/interpretability/attention/
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

os.environ.setdefault("WEEK4_DIR", str(_REPO / "week4"))
os.environ.setdefault("BJ_DATA_DIR", "/home/ubuntu/data")

import week4.config as cfg
from week4.models.stf_model import SpacetimeformerLite
from week6_evaluation.optimization.optuna_stf import STFSearchData, x_flat_to_x_node


# ── 注意力提取 hooks ────────────────────────────────────────────────────────

class AttentionHook:
    """在 self-attn 层注册 forward_hook 提取注意力权重"""
    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}

    def __call__(self, module, args, kwargs, output):
        # TransformerEncoderLayer 的 self_attn 接受 need_weights=True
        # 我们已经在外部设置过，会回传 (output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attn = output[1]
            if attn.is_sparse:
                attn = attn.to_dense()
            self.cache["self_attn"] = attn.detach().cpu()
        else:
            self.cache["self_attn"] = None


class CrossAttentionExtractor:
    """手动跑 cross-attention 并取权重"""
    def __init__(self, model: SpacetimeformerLite):
        self.model = model

    def forward_with_attn(self, x_node: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (model_output, cross_attn_weights)

        cross_attn_weights: (B*N, 1, 1) → reshape to (B, N, 1)
        """
        B, N, F_in, T = x_node.shape
        # env
        x_env = x_node.permute(0, 3, 1, 2)
        x_env = x_env.sum(dim=-1)
        x_env = self.model.env_proj(x_env) + self.model.env_pos
        env_seq = self.model.env_transformer(x_env)

        # loc proj
        x_loc = x_node.permute(0, 1, 3, 2).reshape(B*N, T, F_in)
        x_loc = self.model.loc_proj(x_loc) + self.model.loc_pos

        # cross attention with need_weights
        global_ctx = env_seq[:, -1:, :].repeat(1, N, 1).reshape(B*N, 1, self.model.hidden)
        loc_ca, attn_weights = self.model.cross_attn(
            x_loc, global_ctx, global_ctx, need_weights=True, average_attn_weights=False,
        )
        # attn_weights: (B*N, n_heads, 1, 1) — 每个节点对 1 个全局 token 的注意力
        return loc_ca, attn_weights


# ── 数据准备 ────────────────────────────────────────────────────────────────

def prepare_data():
    """准备测试样本（滑动窗口格式，与 STFSearchDataset 一致）"""
    data = STFSearchData(seq_len=48, horizon=1, batch_size=8)
    # 取测试集前 200 步，构造滑动窗口
    T_raw = min(200, len(data.x_test))
    x_raw = data.x_test[:T_raw]
    seq_len = data.seq_len  # 48
    # 构建滑动窗口 (n_samples, seq_len, 2N+5)
    n_samples = max(1, T_raw - seq_len)
    windows = np.stack([x_raw[i:i+seq_len] for i in range(n_samples)], axis=0)
    test_y = data.y_test[:T_raw]
    return data, windows, test_y


def get_input_node(test_x: np.ndarray, n_nodes: int, n_time: int, device: str) -> torch.Tensor:
    """构造 STF 输入 (B, N, F_in, T) — 与 week4 _batch_to_node 完全一致

    Args:
        test_x: (B, T, F_total) numpy — flat 数据
        n_nodes: 节点数 (1024)
        n_time: time features 维度 (5)
    """
    # 转成 (B, F_total, T) 再调 x_flat_to_x_node
    x_flat = torch.from_numpy(test_x).float().permute(0, 2, 1).to(device)
    return x_flat_to_x_node(x_flat, n_nodes, n_time)


# ── 注意力筛选 ──────────────────────────────────────────────────────────────

def top_k_heads(env_attn: np.ndarray, k: int = 3) -> List[int]:
    """按方差选 Top-K 重要 head

    Args:
        env_attn: (n_layers, n_heads, T, T)  — Env Transformer 自注意力
        k: 选 top k

    Returns:
        [(layer_idx, head_idx, variance), ...] 按 variance 降序
    """
    if env_attn.ndim != 4:
        return []
    n_layers, n_heads = env_attn.shape[:2]
    scores = []
    for l in range(n_layers):
        for h in range(n_heads):
            attn = env_attn[l, h]
            var = float(attn.var())
            scores.append((l, h, var))
    scores.sort(key=lambda x: -x[2])
    return scores[:k]


# ── 可视化（用纯 matplotlib，不依赖 plotly）──────────────────────────────────

def plot_spatial_attention(cross_attn: np.ndarray, title: str, save_path: Path):
    """32×32 空间注意力热力图

    Args:
        cross_attn: (N,)  — 每个节点对全局 token 的注意力分数
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = W = 32
    spatial = cross_attn.reshape(H, W)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(spatial, cmap="hot", interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_temporal_attention(env_attn_layer_head: np.ndarray, title: str, save_path: Path):
    """48 步时间注意力热力图

    Args:
        env_attn_layer_head: (T, T) — 一层一个 head 的自注意力
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(env_attn_layer_head, cmap="viridis", aspect="auto")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("key timestep")
    ax.set_ylabel("query timestep")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_attention_summary(layer_head_vars: List[Tuple[int, int, float]], save_path: Path):
    """Top-K 重要 head 柱状图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"L{l}H{h}" for l, h, _ in layer_head_vars]
    values = [v for _, _, v in layer_head_vars]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="steelblue")
    ax.set_title("Top-K Attention Heads by Variance", fontsize=12)
    ax.set_ylabel("Variance")
    ax.set_xlabel("Layer.Head")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "--checkpoint",
                        dest="weights",
                        default="week4/weights/stf_optuna.pth",
                        help="STF 模型权重路径")
    parser.add_argument("--output", default="week6_evaluation/results/interpretability/attention/")
    parser.add_argument("--n-samples", type=int, default=10, help="用几个 batch 提取注意力")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[AttentionVis] device={device}")

    # ── 加载模型 ──
    print(f"[AttentionVis] 加载模型: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    params = ckpt.get("params", {
        "hidden": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.1,
    })
    print(f"[AttentionVis] 权重元数据: params={params}, val_mae={ckpt.get('val_mae', 'N/A')}")

    # 先用占位 in_dim=2+n_time 构造，再 load（与 week4 训练一致）
    model = SpacetimeformerLite(
        in_dim=2 + 5, hidden=params["hidden"], horizon=1, n_nodes=1024,
        n_heads=params["n_heads"], n_layers=params["n_layers"], dropout=params["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # ── 准备数据 ──
    data, windows, _ = prepare_data()  # windows: (n_samples, seq_len, 2N+5)

    # 替换 TransformerEncoder 为带 output_attentions 的版本
    # PyTorch 的 TransformerEncoder 不直接支持，需手动包一层
    # 简化做法：用 hook 拦截
    print("[AttentionVis] 替换 Env Transformer 为支持 attention 输出...")
    model.env_transformer = _wrap_for_attention(model.env_transformer).to(device)

    # ── 提取注意力 ──
    print(f"[AttentionVis] 提取 {args.n_samples} 批次的注意力...")
    extractor = CrossAttentionExtractor(model)

    n_collected = 0
    n_batches = 0
    all_env_attn = []  # list of (n_layers, n_heads, T, T)
    all_cross_attn = []  # list of (B*N, n_heads, 1, 1)

    with torch.no_grad():
        for i in range(0, min(len(windows), args.n_samples * 8), 8):
            x = windows[i:i+8]
            x_node = get_input_node(x, data.n_nodes, data.n_time_features, device)

            # 提取 cross attention
            _, cross_attn_w = extractor.forward_with_attn(x_node)
            # 重新跑 forward 让 hook 拿到 env self-attn
            _ = model(x_node)

            # hook 缓存
            if hasattr(model.env_transformer, "_last_attn_weights"):
                env_attn = model.env_transformer._last_attn_weights
                if env_attn is not None:
                    all_env_attn.append(env_attn.cpu().numpy())

            all_cross_attn.append(cross_attn_w.cpu().numpy())
            n_batches += 1
            n_collected += x_node.shape[0]

            if n_batches >= args.n_samples:
                break

    print(f"[AttentionVis] 收集 {n_collected} 个样本, {n_batches} 批")
    if not all_env_attn:
        print("[AttentionVis] 警告: 未收集到 env attention（hook 失败）")
        return
    if not all_cross_attn:
        print("[AttentionVis] 警告: 未收集到 cross attention")
        return

    # 平均
    env_attn_mean = np.mean(np.stack(all_env_attn), axis=0)  # (n_layers, n_heads, T, T)
    cross_attn_mean = np.mean(np.stack(all_cross_attn), axis=0)  # (B*N, n_heads, 1, 1)

    # 1. 选 Top-K heads
    top_heads = top_k_heads(env_attn_mean, k=5)
    print(f"[AttentionVis] Top-5 heads: {top_heads}")

    plot_attention_summary(top_heads, output_dir / "head_variance.png")

    # 2. 每个 top head 一张时间注意力图（最多 3 张，但模型可能只有 1-2 个 head）
    for rank, (l, h, var) in enumerate(top_heads[:3]):
        title = f"Env Self-Attn  L{l}H{h}  var={var:.4f}"
        plot_temporal_attention(
            env_attn_mean[l, h],
            title,
            output_dir / f"temporal_attn_L{l}H{h}.png",
        )

    # 3. Cross attention 空间图（若收集到数据）
    # 真实 shape: (B*N, n_heads, T, 1) — T=48 是 key 的时间步
    # 取所有时间步均值得到每个节点对全局 token 的注意力分数
    if all_cross_attn:
        # 堆叠所有 batch：(n_batches, B*N, n_heads, T, 1)
        stacked = np.stack(all_cross_attn, axis=0)
        # 对 batch 维 + 时间维 求均值 → (B*N, n_heads)
        ca_mean = stacked.mean(axis=(0, 3, 4))  # (B*N, n_heads)
        # 取一个 batch 的 1024 节点
        if ca_mean.shape[0] >= 1024:
            ca_nodes = ca_mean[:1024]  # (1024, n_heads)
        else:
            ca_nodes = ca_mean
        n_heads = ca_nodes.shape[1] if ca_nodes.ndim == 2 else 1
        for h in range(min(n_heads, 4)):
            spatial_attn = ca_nodes[:, h] if ca_nodes.ndim == 2 else ca_nodes
            plot_spatial_attention(
                spatial_attn,
                f"Cross-Attn Head {h} (32x32 spatial, mean over time)",
                output_dir / f"spatial_attn_H{h}.png",
            )
    else:
        print("[AttentionVis] 跳过 cross attention 空间图（无数据）")

    # 4. 汇总
    summary = {
        "n_samples": n_collected,
        "n_batches": n_batches,
        "top_heads": [(int(l), int(h), float(v)) for l, h, v in top_heads],
        "env_attn_shape": list(env_attn_mean.shape),
        "cross_attn_shape": list(cross_attn_mean.shape),
    }
    (output_dir / "attention_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[AttentionVis] 完成 → {output_dir}")


def _wrap_for_attention(encoder: nn.TransformerEncoder) -> nn.Module:
    """包装 TransformerEncoder 让其输出 attention weights

    PyTorch 不原生支持 output_attentions，需要逐层提取。
    这里用一个简单 wrapper，把每层的 attn 输出累积到 self._last_attn_weights。
    """
    import copy

    class AttnCapturingEncoder(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.layers = original.layers
            self.norm = original.norm
            self._last_attn_weights = None

        def forward(self, src):
            # 逐层运行，每层开启 need_weights
            output = src
            all_attn = []
            for layer in self.layers:
                output, attn_w = layer.self_attn(
                    output, output, output, need_weights=True,
                    average_attn_weights=False,
                )
                if attn_w.is_sparse:
                    attn_w = attn_w.to_dense()
                all_attn.append(attn_w)  # (B, n_heads, T, T)
                # 完整 layer forward（含 FFN/LN）
                output = layer(output)
            if self.norm is not None:
                output = self.norm(output)
            # 堆叠所有层: (n_layers, B, n_heads, T, T) → permute to (n_layers, n_heads, T, T) via mean
            stacked = torch.stack(all_attn, dim=0)  # (n_layers, B, n_heads, T, T)
            self._last_attn_weights = stacked.mean(dim=1)  # mean over batch → (n_layers, n_heads, T, T)
            return output

    return AttnCapturingEncoder(encoder)


if __name__ == "__main__":
    main()
