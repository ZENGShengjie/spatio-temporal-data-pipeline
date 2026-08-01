"""Week6 任务4-1：Pipeline 性能 Profile

定位瓶颈：CPU / CUDA / 数据加载 / 序列化 各阶段耗时占比

执行（EC2）：
    python -m week6_evaluation.profiling.profile_pipeline \\
        --mode fast \\
        --output week6_evaluation/results/profiling/

输出：
- profile.json：各阶段耗时
- bottlenecks.md：瓶颈分析（自动生成）
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "week6"))

os.environ.setdefault("WEEK4_DIR", str(_REPO / "week4"))
os.environ.setdefault("BJ_DATA_DIR", "/home/ubuntu/data")


# ── PyTorch Profiler ──────────────────────────────────────────────────────

def run_torch_profile(pipe, t_start: int, t_end: int, output_path: Path, mode: str = "fast"):
    """用 PyTorch Profiler 分析单步推理"""
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

    pipe._warning_engine.reset()
    pipe.init_realtime(warmup_steps=48)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Warmup
    print(f"[Profile] Warmup 5 步...")
    for _ in range(5):
        pipe.run_step()

    print(f"[Profile] 开始 Profiling {t_end - t_start} 步...")
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(t_end - t_start):
            pipe.run_step()

    # 输出到文件 + 控制台
    table = prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=30,
    )
    print(table)
    (output_path / f"torch_profile_{mode}.txt").write_text(table, encoding="utf-8")
    print(f"  [saved] {output_path / f'torch_profile_{mode}.txt'}")

    # JSON 版
    stats = []
    for evt in prof.key_averages():
        stats.append({
            "name": evt.key,
            "count": evt.count,
            "cpu_time_us": evt.cpu_time,
            "cuda_time_us": getattr(evt, "cuda_time", 0),
            "self_cuda_time_us": getattr(evt, "self_cuda_time_total", 0),
        })
    (output_path / f"torch_profile_{mode}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return stats


# ── 各阶段手工打点 ────────────────────────────────────────────────────────

def profile_pipeline_stages(pipe, t_start: int, t_end: int) -> Dict[str, float]:
    """手动打点：分阶段记录耗时

    阶段：
    1. 数据加载（npz 读取 + 切分）
    2. 统计法打分
    3. 预测法打分
    4. 融合
    5. 异常判定
    6. 事件聚合
    7. 预警判定
    """
    times = {
        "data_loading": [],
        "stat_scoring": [],
        "pred_scoring": [],
        "fusion": [],
        "anomaly_decision": [],
        "warning_evaluation": [],
    }

    pipe._warning_engine.reset()
    pipe._ensure_stat()
    pipe._ensure_pred()

    # 阶段 1: 数据加载
    t0 = time.perf_counter()
    from week5.data_loader import get_flow_1d, get_timestamps
    flow = get_flow_1d("taxi_flow_total")
    ts = get_timestamps()
    times["data_loading"].append((time.perf_counter() - t0) * 1000)

    for t in range(t_start, t_end):
        # 阶段 2: 统计法打分
        t0 = time.perf_counter()
        _ = pipe._stat_detector.predict_scores(pipe._window[-48:])
        times["stat_scoring"].append((time.perf_counter() - t0) * 1000)

        # 阶段 3: 预测法打分
        t0 = time.perf_counter()
        new_data = pipe._window[-1]
        scores_pred = pipe._pred_detector.predict_scores(new_data)
        times["pred_scoring"].append((time.perf_counter() - t0) * 1000)

        # 阶段 4: 融合
        t0 = time.perf_counter()
        from week5.anomaly.fusion_v3 import normalize_scores
        fused = 0.9 * normalize_scores(_[0]) + 0.1 * normalize_scores(scores_pred)
        times["fusion"].append((time.perf_counter() - t0) * 1000)

        # 阶段 5: 异常判定
        t0 = time.perf_counter()
        mask = fused >= 0.9
        times["anomaly_decision"].append((time.perf_counter() - t0) * 1000)

        # 阶段 6: 预警判定
        t0 = time.perf_counter()
        _ = pipe._warning_engine.evaluate(mask.flatten(), fused.flatten(), t=t)
        times["warning_evaluation"].append((time.perf_counter() - t0) * 1000)

        pipe._window.append(new_data)
        if len(pipe._window) > 48:
            pipe._window.pop(0)

    return {
        stage: {
            "avg_ms": float(np.mean(vals)),
            "total_ms": float(np.sum(vals)),
            "p95_ms": float(np.percentile(vals, 95)) if vals else 0.0,
            "n": len(vals),
        }
        for stage, vals in times.items()
    }


# ── 瓶颈分析 ──────────────────────────────────────────────────────────────

def write_bottleneck_report(stages: Dict, output_path: Path, mode: str = "fast"):
    """根据 stage 耗时排序，输出瓶颈报告"""
    sorted_stages = sorted(
        stages.items(),
        key=lambda kv: -kv[1]["avg_ms"],
    )
    lines = [
        f"# Pipeline 瓶颈分析 — {mode}",
        "",
        "## 各阶段平均耗时（按耗时降序）",
        "",
        "| 排名 | 阶段 | 平均 ms | 总 ms | p95 ms | 占比 |",
        "|------|------|---------|-------|--------|------|",
    ]
    total_avg = sum(s["avg_ms"] for _, s in sorted_stages)
    for i, (name, s) in enumerate(sorted_stages, 1):
        ratio = s["avg_ms"] / total_avg * 100 if total_avg > 0 else 0
        lines.append(
            f"| {i} | {name} | {s['avg_ms']:.2f} | {s['total_ms']:.0f} | {s['p95_ms']:.2f} | {ratio:.1f}% |"
        )
    lines.append("")
    lines.append("## 优化优先级")
    lines.append("")
    if sorted_stages:
        top = sorted_stages[0]
        lines.append(f"**首要优化**：{top[0]}（占总耗时 {top[1]['avg_ms']/total_avg*100:.1f}%）")
    lines.append("")
    lines.append("### 推荐优化方案")
    lines.append("")
    lines.append("1. **数据预加载**：启动时一次性加载到内存，消除 `data_loading` 阶段")
    lines.append("2. **`torch.inference_mode()`**：推理时禁用 autograd，提升 30%")
    lines.append("3. **API LRU 缓存**：相同 (t, mode) 请求直接返回缓存")
    lines.append("4. **批量预测**：API 支持一次性预测多步，GPU 利用率提升")

    out = "\n".join(lines)
    (output_path / f"bottlenecks_{mode}.md").write_text(out, encoding="utf-8")
    print(out)


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="fast", choices=["fast", "structural"])
    parser.add_argument("--n-steps", type=int, default=50, help="Profile 步数")
    parser.add_argument("--output", default="week6_evaluation/results/profiling/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from week6.pipeline import SpatiotemporalPipeline
    from week5.config import VAL_END, TEST_END

    print(f"[Profile] mode={args.mode}, n_steps={args.n_steps}")
    print(f"[Profile] 加载 Pipeline...")
    pipe = SpatiotemporalPipeline(mode=args.mode, use_cache=True)
    pipe._ensure_stat()
    pipe._ensure_pred()
    pipe.init_realtime(warmup_steps=48)

    # 1. PyTorch Profiler
    if torch.cuda.is_available():
        print(f"[Profile] PyTorch Profiler (CUDA)...")
    else:
        print(f"[Profile] PyTorch Profiler (CPU only)...")
    run_torch_profile(pipe, t_start=VAL_END, t_end=VAL_END + args.n_steps,
                      output_path=output_dir, mode=args.mode)

    # 2. 手工打点
    print(f"[Profile] 手工 stage 打点...")
    pipe._warning_engine.reset()
    pipe.init_realtime(warmup_steps=48)
    stages = profile_pipeline_stages(pipe, t_start=VAL_END, t_end=VAL_END + args.n_steps)
    (output_dir / f"stages_{args.mode}.json").write_text(
        json.dumps(stages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3. 瓶颈报告
    write_bottleneck_report(stages, output_dir, mode=args.mode)
    print(f"\n[Profile] 完成 → {output_dir}")


if __name__ == "__main__":
    main()
