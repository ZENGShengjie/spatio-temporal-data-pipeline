"""Week6 任务4 工具：对比优化前后的 profile 数据

执行：
    python -m week6.evaluation.profiling.compare_results \\
        results/baseline/profile.json results/profiling/optimized/profile.json

输出：
- 对比表（Markdown）
- 关键指标变化
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any


def load_profile(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_change(before: float, after: float) -> str:
    """格式化变化百分比"""
    if before == 0:
        return "N/A"
    delta = (after - before) / before * 100
    if delta < 0:
        return f"-{abs(delta):.1f}% (优化)"
    elif delta > 0:
        return f"+{delta:.1f}% (变差)"
    return "0%"


def extract_metrics(profile: Dict[str, Any]) -> Dict[str, float]:
    """从 profile 字典提取关键指标"""
    metrics = {}

    # 批量
    if "batch_fast" in profile:
        b = profile["batch_fast"]
        metrics["batch_total_ms"] = b.get("total_ms", 0)
        metrics["batch_throughput"] = b.get("throughput_steps_per_sec", 0)

    # 实时
    if "realtime_fast" in profile:
        r = profile["realtime_fast"]
        metrics["realtime_avg_ms"] = r.get("avg_latency_ms", 0)
        metrics["realtime_p95_ms"] = r.get("p95_latency_ms", 0)
        metrics["realtime_p99_ms"] = r.get("p99_latency_ms", 0)
        metrics["realtime_ram_mb"] = r.get("peak_ram_mb", 0)

    # structural
    if "realtime_structural" in profile and "avg_latency_ms" in profile.get("realtime_structural", {}):
        r = profile["realtime_structural"]
        metrics["structural_avg_ms"] = r.get("avg_latency_ms", 0)
        metrics["structural_p95_ms"] = r.get("p95_latency_ms", 0)
        metrics["structural_gpu_mb"] = r.get("peak_gpu_mb", 0)

    # API
    if "api_detect" in profile and "avg_ms" in profile["api_detect"]:
        a = profile["api_detect"]
        metrics["api_detect_avg_ms"] = a.get("avg_ms", 0)
        metrics["api_detect_p95_ms"] = a.get("p95_ms", 0)

    return metrics


def compare_profiles(before_path: Path, after_path: Path, output_path: Path = None) -> str:
    """对比两个 profile 文件，输出 Markdown 报告"""
    before = load_profile(before_path)
    after = load_profile(after_path)

    before_m = extract_metrics(before)
    after_m = extract_metrics(after)

    lines = [
        f"# 性能优化对比报告",
        "",
        f"- **优化前**：`{before_path.name}`",
        f"- **优化后**：`{after_path.name}`",
        "",
        "## 关键指标对比",
        "",
        "| 指标 | 优化前 | 优化后 | 变化 |",
        "|------|--------|--------|------|",
    ]

    for key in sorted(set(before_m.keys()) | set(after_m.keys())):
        b_val = before_m.get(key)
        a_val = after_m.get(key)
        if b_val is None and a_val is None:
            continue
        b_str = f"{b_val:.1f}" if b_val is not None else "-"
        a_str = f"{a_val:.1f}" if a_val is not None else "-"
        if b_val is not None and a_val is not None:
            change = format_change(b_val, a_val)
        else:
            change = "新增"
        lines.append(f"| {key} | {b_str} | {a_str} | {change} |")

    lines.append("")
    lines.append("## 解读")
    lines.append("")

    # 自动给出关键解读
    if "realtime_avg_ms" in before_m and "realtime_avg_ms" in after_m:
        change = (after_m["realtime_avg_ms"] - before_m["realtime_avg_ms"]) / before_m["realtime_avg_ms"] * 100
        if change < 0:
            lines.append(f"- 实时平均延迟降低 **{abs(change):.1f}%**")
        else:
            lines.append(f"- 实时平均延迟增加 **{change:.1f}%**（需回滚优化）")

    if "batch_throughput" in before_m and "batch_throughput" in after_m:
        change = (after_m["batch_throughput"] - before_m["batch_throughput"]) / before_m["batch_throughput"] * 100
        if change > 0:
            lines.append(f"- 批量吞吐量提升 **{change:.1f}%**")
        else:
            lines.append(f"- 批量吞吐量降低 **{abs(change):.1f}%**（需回滚优化）")

    out = "\n".join(lines)
    if output_path:
        output_path.write_text(out, encoding="utf-8")
        print(f"  [saved] {output_path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", help="优化前 profile.json")
    parser.add_argument("after", help="优化后 profile.json")
    parser.add_argument("--output", help="输出 Markdown 报告路径")
    args = parser.parse_args()

    before = Path(args.before)
    after = Path(args.after)
    output = Path(args.output) if args.output else None

    if not before.exists():
        print(f"ERROR: {before} not found")
        return
    if not after.exists():
        print(f"ERROR: {after} not found")
        return

    report = compare_profiles(before, after, output)
    print(report)


if __name__ == "__main__":
    main()
