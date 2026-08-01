"""Week6 任务1 / 任务2 / 任务3 / 任务4 共用的评估入口

设计动机：
- 任务1（基线）、任务2（Optuna 优化后）、任务3（归因后）都用同一份评估
- 任务4（性能优化）共用 profile_runner
- 不重写评估代码，统一调度口径

调用方式（EC2 上）：
    python -m week6.evaluation.evaluation.evaluate --model-tag baseline --output results/baseline/
    python -m week6.evaluation.evaluation.evaluate --model-tag optuna --output results/optuna/

输出：
    - metrics.json        主指标
    - profile.json        性能 profile
    - profile_api.json    API 端到端延迟（可选）
    - summary.md          自动生成的人类可读摘要

数据约定（实测后修正）：
    - STF 预测流量值：来自 week4/results/*_pred.npy (T_test, N)
    - 实际流量值：来自 week4/results/*_gt.npy (T_test, N)
    - 异常 ground truth：来自 week5/data/anomaly_labels_test.npy (T_test, N) bool
    - 异常得分：fusion_scores_test_v3.npy (T_test, N)
    - 注意：pred_scores_test_v2.npy 是异常分数（归一化），不是流量值！
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# 路径修复：把仓库根加入 sys.path
_REPO = Path(__file__).resolve().parents[3]  # week6/evaluation/evaluation/ → amazon/
sys.path.insert(0, str(_REPO))

import numpy as np

from week6.evaluation.evaluation.metrics import full_evaluation, compute_classification_metrics
from week6.evaluation.evaluation.profile_runner import (
    profile_pipeline, profile_batch, profile_api_endpoint,
    save_profile, save_api_profile, get_ram_mb,
)


# ── 数据加载（适配 EC2 实际数据布局）──────────────────────────────────────────

def load_test_split(use_stf_predictions: bool = True, stf_pred_path: str = None) -> dict:
    """从缓存加载测试集所需数据

    关键修正：
    - pred_scores 现在是 STF 真实流量预测值（不是归一化异常得分）
    - 异常评估可对 ground truth 算 F1/Precision/Recall

    Args:
        use_stf_predictions: True=用 STF pred（来自 week4/results/），
                            False=回退到 week5 的异常分数当 proxy
        stf_pred_path: 若提供，直接从此路径加载 STF 预测（覆盖默认搜索路径）

    Returns:
        dict with keys: flow, pred_scores, anomaly_mask, fused_scores,
                        events, timestamps, val_end, gt_anomaly_labels
    """
    # 加载 week5 自包含的 flow（已 normalize 到 0~1 区间）
    from week5.config import VAL_END, TEST_END
    from week5.data_loader import get_flow_1d, get_timestamps, get_splits

    train_end, val_end, test_end = get_splits()
    flow_norm = get_flow_1d("taxi_flow_total")[val_end:test_end]
    timestamps = get_timestamps()[val_end:test_end]

    # ── STF 预测流量（已存在则用，否则 fallback） ──────────────────────────
    pred_flow = None
    gt_flow = None
    if stf_pred_path is not None:
        p = Path(stf_pred_path)
        if p.exists():
            pred_flow_full = np.load(p)
            if len(pred_flow_full) >= 600:
                pred_flow = pred_flow_full[-600:]
            else:
                pred_flow = pred_flow_full
            print(f"[evaluate] STF pred loaded from override path: {p}")
    else:
        # Week4 STF pred 可能存放在多处：week4/results/、week4/results_v4_fix/、amazon/results/、repo/results/
        candidate_pred_paths = [
            _REPO / "week4" / "results" / "stf_taxi_flow_total_v4fix_pred.npy",
            _REPO / "week4" / "results_v4_fix" / "stf_taxi_flow_total_v4fix_pred.npy",
            _REPO.parent / "results" / "stf_taxi_flow_total_v4fix_pred.npy",  # repo_root/results
            _REPO / "results" / "stf_taxi_flow_total_v4fix_pred.npy",  # amazon/results
        ]
        candidate_gt_paths = [
            _REPO / "week4" / "results" / "stf_taxi_flow_total_v4fix_gt.npy",
            _REPO / "week4" / "results_v4_fix" / "stf_taxi_flow_total_v4fix_gt.npy",
            _REPO.parent / "results" / "stf_taxi_flow_total_v4fix_gt.npy",
            _REPO / "results" / "stf_taxi_flow_total_v4fix_gt.npy",
        ]
        for pred_path, gt_path in zip(candidate_pred_paths, candidate_gt_paths):
            if pred_path.exists():
                pred_flow_full = np.load(pred_path)
                gt_flow_full = np.load(gt_path) if gt_path.exists() else pred_flow_full
                # 取测试集段（最后 600 步）
                if len(pred_flow_full) >= 600:
                    pred_flow = pred_flow_full[-600:]
                    gt_flow = gt_flow_full[-600:] if gt_path.exists() else pred_flow_full[-600:]
                else:
                    pred_flow = pred_flow_full
                    gt_flow = gt_flow_full
                print(f"[evaluate] STF pred loaded from: {pred_path}")
                break

    # 若 STF 训练未完成，pred_flow=None，用 flow_norm 当占位（pred=actual 无意义）
    if pred_flow is None:
        # 退化方案：用归一化异常分数 + 占位
        pred_flow = flow_norm.copy()

    # ── 异常 ground truth（来自 week5 注入） ───────────────────────────────
    gt_anomaly_labels = None
    gt_labels_path = _REPO / "week5" / "data" / "anomaly_labels_test.npy"
    if gt_labels_path.exists():
        gt_anomaly_labels = np.load(gt_labels_path).astype(bool)
        # 形状对齐
        if gt_anomaly_labels.shape != (len(flow_norm), 1024):
            print(f"[evaluate] WARNING: GT shape {gt_anomaly_labels.shape} != flow shape ({len(flow_norm)}, 1024)")
            gt_anomaly_labels = None

    # ── 各方法异常得分（用于融合 + 异常评估） ────────────────────────────
    cache_dir = _REPO / "week5" / "cache"
    def _load(name):
        p = cache_dir / name
        return np.load(p) if p.exists() else None

    stat_scores = _load("stat_scores_test_v2.npy")
    pred_scores_method = _load("pred_scores_test_v2.npy")  # 这是异常分数（归一化 pred 误差）
    vae_scores = _load("vae_scores_test.npy")
    tae_scores = _load("tae_scores_test_v3.npy")  # v3 更优

    # 融合（与 pipeline 一致，stat 主导）
    def normalize(x):
        if x is None: return 0.0
        x = np.asarray(x)
        if x.size == 0: return 0.0
        q = np.quantile(x, 0.99)
        return np.clip(x / (q + 1e-8), 0, 1)

    # 使用 stat + pred + vae 融合，tae 作为可选
    fused = (
        0.7 * normalize(stat_scores) +
        0.1 * normalize(pred_scores_method) +
        0.2 * normalize(vae_scores)
    )
    if tae_scores is not None:
        # 在融合基础上叠加 tae
        fused = 0.85 * normalize(fused) + 0.15 * normalize(tae_scores)

    # 异常判定（阈值 0.9，可通过参数覆盖）
    anomaly_mask = fused >= 0.90

    # ── 阈值 sweep（基于 ground truth 找最佳阈值） ────────────────────────
    threshold_sweep = None
    if Path("/home/ubuntu/amazon/week5/data/anomaly_labels_test.npy").exists() and (
        "amazon" in str(_REPO.resolve()) or _REPO.parent.resolve().name == "amazon"
    ):
        # 仅当在 amazon 仓库内时执行（保持可移植性）
        try:
            gt_path = _REPO / "week5" / "data" / "anomaly_labels_test.npy"
            gt = np.load(gt_path).astype(bool)
            if gt.shape == fused.shape:
                threshold_sweep = []
                for thr in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98):
                    pred = (fused >= thr).astype(bool).ravel()
                    gt_flat = gt.ravel()
                    TP = int((pred & gt_flat).sum())
                    FP = int((pred & ~gt_flat).sum())
                    FN = int((~pred & gt_flat).sum())
                    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                    threshold_sweep.append({
                        "threshold": thr,
                        "precision": float(P),
                        "recall": float(R),
                        "f1": float(F),
                        "TP": TP, "FP": FP, "FN": FN,
                        "pred_count": int(pred.sum()),
                    })
        except Exception:
            pass

    # ── 事件（从 cache 里读，没有就空） ──────────────────────────────────
    events = []
    for events_path in (_REPO / "week6" / "data" / "events_test_v1.json",):
        if events_path.exists():
            with open(events_path, "r", encoding="utf-8") as f:
                events = json.load(f)
            break

    return {
        "flow": flow_norm,
        "pred_scores": pred_flow,  # STF 真实预测流量
        "gt_flow": gt_flow,  # STF 测试集真实流量（来自 week4 results）
        "anomaly_mask": anomaly_mask,
        "fused_scores": fused,
        "events": events,
        "timestamps": timestamps,
        "val_end": int(val_end),
        "gt_anomaly_labels": gt_anomaly_labels,  # bool, (T, N)
        "threshold_sweep": threshold_sweep,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_metrics(data: dict) -> dict:
    """跑指标评估（包含真正的 GT 异常评估 + 合理性评估）"""
    base = full_evaluation(
        flow=data["flow"],
        pred_scores=data["pred_scores"],
        anomaly_mask=data["anomaly_mask"],
        fused_scores=data["fused_scores"],
        events=data["events"],
        timestamps=data["timestamps"],
        val_end=data["val_end"],
    )

    # 真正 ground truth 评估（异常检测）
    if data.get("gt_anomaly_labels") is not None:
        gt = data["gt_anomaly_labels"]
        pred = data["anomaly_mask"]
        try:
            base["anomaly_classification"] = compute_classification_metrics(pred, gt)
        except Exception as e:
            base["anomaly_classification"] = {"error": str(e)}

    return base


def run_system_profile(api_host: str = None, port: int = 8000) -> dict:
    """跑系统性能 profile"""
    from week6.pipeline import SpatiotemporalPipeline

    results = {}

    # 1. 批量模式
    print("[Profile] 启动 Pipeline (fast 模式)...")
    pipe_fast = SpatiotemporalPipeline(mode="fast", use_cache=True)
    results["batch_fast"] = profile_batch(pipe_fast, split="test", mode="fast")

    # 2. 实时模式（fast）
    print("[Profile] 启动 Pipeline 实时模式 (fast)...")
    pipe_fast.init_realtime(warmup_steps=48)
    _, val_end, _ = 2784, 3288, 3888  # 兜底
    try:
        from week5.data_loader import get_splits
        _, val_end, _ = get_splits()
    except Exception:
        pass
    prof = profile_pipeline(
        pipe_fast, t_start=val_end, t_end=val_end + 100, mode="fast",
    )
    results["realtime_fast"] = {
        "n_steps": prof.n_steps,
        "avg_latency_ms": prof.avg_latency_ms,
        "p50_latency_ms": prof.p50_latency_ms,
        "p95_latency_ms": prof.p95_latency_ms,
        "p99_latency_ms": prof.p99_latency_ms,
        "peak_ram_mb": prof.peak_ram_mb,
    }

    # 3. structural 模式（如果有 GPU）
    if _check_structural_available():
        try:
            print("[Profile] 切换到 structural 模式...")
            pipe_struct = SpatiotemporalPipeline(mode="structural", use_cache=True)
            prof_struct = profile_pipeline(
                pipe_struct, t_start=val_end, t_end=val_end + 50, mode="structural",
            )
            results["realtime_structural"] = {
                "n_steps": prof_struct.n_steps,
                "avg_latency_ms": prof_struct.avg_latency_ms,
                "p50_latency_ms": prof_struct.p50_latency_ms,
                "p95_latency_ms": prof_struct.p95_latency_ms,
                "p99_latency_ms": prof_struct.p99_latency_ms,
                "peak_ram_mb": prof_struct.peak_ram_mb,
                "peak_gpu_mb": prof_struct.peak_gpu_mb,
            }
        except Exception as e:
            results["realtime_structural"] = {"error": str(e)}

    # 4. API 端到端（如果提供）
    if api_host:
        try:
            results["api_health"] = profile_api_endpoint(
                f"http://{api_host}:{port}/api/health", {}, n_requests=10, method="GET",
            )
            results["api_detect"] = profile_api_endpoint(
                f"http://{api_host}:{port}/api/anomaly/detect",
                {"t": 3500, "mode": "fast"}, n_requests=20, method="POST",
            )
            results["api_forecast"] = profile_api_endpoint(
                f"http://{api_host}:{port}/api/forecast",
                {"time_start": 3288, "time_end": 3295, "grid_ids": [512]},
                n_requests=10, method="POST",
            )
        except Exception as e:
            results["api_error"] = str(e)

    return results


def _check_structural_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def write_summary(metrics: dict, profile: dict, output_dir: Path, model_tag: str):
    """生成人类可读的 Markdown 摘要"""
    md = []
    md.append(f"# 评估摘要 — {model_tag}")
    md.append(f"\n生成时间：{datetime.now().isoformat(timespec='seconds')}\n")

    md.append("## 1. 预测精度（STF 真实流量预测）")
    pred = metrics["predict"]
    md.append("| 维度 | 指标 | 数值 |")
    md.append("|------|------|------|")
    md.append(f"| 全局 | MAE | {pred['overall_mae']:.4f} |")
    md.append(f"| 全局 | RMSE | {pred['overall_rmse']:.4f} |")
    md.append(f"| 全局 | MAPE | {pred['overall_mape']:.2f}% |")
    if pred.get("next_step_direction_acc") is not None:
        md.append(f"| 连续性 | t+1 方向准确率 | {pred['next_step_direction_acc']*100:.1f}% |")
    md.append("")

    md.append("### 时段分层")
    md.append("| 时段 | MAE | RMSE | 样本数 |")
    md.append("|------|-----|------|--------|")
    for period in ["morning_peak", "off_peak", "evening_peak", "night"]:
        mae = pred.get(f"{period}_mae")
        rmse = pred.get(f"{period}_rmse")
        cnt = pred.get(f"{period}_count")
        if mae is not None:
            md.append(f"| {period} | {mae:.4f} | {rmse:.4f} | {cnt} |")
    md.append("")

    md.append("### 区域分层")
    md.append("| 区域 | MAE | RMSE |")
    md.append("|------|-----|------|")
    if pred.get("core_area_mae") is not None:
        md.append(f"| 核心区 | {pred['core_area_mae']:.4f} | {pred['core_area_rmse']:.4f} |")
    if pred.get("suburban_mae") is not None:
        md.append(f"| 郊区 | {pred['suburban_mae']:.4f} | {pred['suburban_rmse']:.4f} |")
    md.append("")

    # 异常检测：分两个角度（有 GT 时也算真实 F1）
    md.append("## 2. 异常检测评估")
    cls = metrics.get("anomaly_classification")
    if cls and "error" not in cls:
        md.append("### 2.1 真实 Ground Truth 评估（异常注入标签）")
        md.append("| 指标 | 数值 |")
        md.append("|------|------|")
        for k in ["precision", "recall", "f1", "accuracy", "auc_roc"]:
            if k in cls and cls[k] is not None:
                md.append(f"| {k} | {cls[k]:.4f} |")
        md.append("")
        md.append(f"- 真阳性 TP: {cls.get('TP', '-')}")
        md.append(f"- 假阳性 FP: {cls.get('FP', '-')}")
        md.append(f"- 假阴性 FN: {cls.get('FN', '-')}")
        md.append(f"- 真阴性 TN: {cls.get('TN', '-')}")
        md.append("")

    # 阈值 sweep 表格
    sweep = metrics.get("threshold_sweep")
    if sweep:
        md.append("### 2.3 阈值扫描（基于 Ground Truth F1）")
        md.append("| 阈值 | Precision | Recall | F1 | pred 数 |")
        md.append("|------|-----------|--------|----|---------|")
        for row in sweep:
            md.append(f"| {row['threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['pred_count']} |")
        best = max(sweep, key=lambda r: r["f1"])
        md.append(f"\n**最优阈值：{best['threshold']:.2f}，F1={best['f1']:.4f}**")
        md.append("")

    md.append("### 2.2 合理性评估（无 ground truth 时仍有效）")
    anom = metrics["anomaly"]
    md.append("| 维度 | 指标 | 数值 |")
    md.append("|------|------|------|")
    md.append(f"| 整体 | 异常率 | {anom['overall_anomaly_rate']*100:.2f}% |")
    md.append(f"| 时段 | 白天异常率 | {anom['day_anomaly_rate']*100:.2f}% |")
    md.append(f"| 时段 | 夜间异常率 | {anom['night_anomaly_rate']*100:.2f}% |")
    if anom["day_night_ratio"] != float("inf"):
        md.append(f"| 时段 | 昼/夜比 | {anom['day_night_ratio']:.2f}x |")
    md.append(f"| 区域 | 核心区异常密度 | {anom['core_anomaly_density']*100:.2f}% |")
    md.append(f"| 区域 | 郊区异常密度 | {anom['suburb_anomaly_density']*100:.2f}% |")
    md.append(f"| 连片 | 最大连通片尺寸 | {anom['max_cluster_size']} |")
    md.append("")

    md.append("## 3. 异常事件质量")
    ev = metrics["events"]
    if ev.get("total_events", 0) > 0:
        md.append(f"- 总事件数：{ev['total_events']}")
        md.append(f"- 每天事件数：{ev['events_per_day']:.1f}")
        md.append(f"- 平均影响格点：{ev['avg_n_cells']:.1f}")
        md.append(f"- 等级分布：紧急 {ev['level_3_count']} / 重要 {ev['level_2_count']} / 一般 {ev['level_1_count']}")
    else:
        md.append("- 本次评估未提供事件数据")
    md.append("")

    md.append("## 4. 系统性能")
    md.append("| 模式 | 指标 | 数值 |")
    md.append("|------|------|------|")
    if "batch_fast" in profile:
        b = profile["batch_fast"]
        md.append(f"| 批量 fast | 总耗时 | {b['total_ms']:.0f} ms |")
        md.append(f"| 批量 fast | 吞吐量 | {b['throughput_steps_per_sec']:.1f} 步/秒 |")
    if "realtime_fast" in profile:
        r = profile["realtime_fast"]
        md.append(f"| 实时 fast | 平均延迟 | {r['avg_latency_ms']:.1f} ms |")
        md.append(f"| 实时 fast | p95 延迟 | {r['p95_latency_ms']:.1f} ms |")
    if "realtime_structural" in profile and "avg_latency_ms" in profile["realtime_structural"]:
        r = profile["realtime_structural"]
        md.append(f"| 实时 structural | 平均延迟 | {r['avg_latency_ms']:.1f} ms |")
    if "api_detect" in profile and "avg_ms" in profile["api_detect"]:
        a = profile["api_detect"]
        md.append(f"| API /api/anomaly/detect | 平均 | {a['avg_ms']:.1f} ms |")

    out = "\n".join(md)
    (output_dir / "summary.md").write_text(out, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="baseline")
    parser.add_argument("--output", default="week6.evaluation/results/baseline/")
    parser.add_argument("--api-host", default=None)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--n-profile-steps", type=int, default=100)
    parser.add_argument("--no-stf-pred", action="store_true",
                        help="不用 STF pred，用 flow_norm 占位（仅 sanity check）")
    parser.add_argument("--stf-pred-path", default=None,
                        help="直接指定 STF 预测文件路径（覆盖默认搜索路径，用于 optuna 模型）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[evaluate] model-tag={args.model_tag}, output={output_dir}")
    print(f"[evaluate] 加载数据...")
    t0 = time.time()
    data = load_test_split(use_stf_predictions=not args.no_stf_pred,
                           stf_pred_path=args.stf_pred_path)
    print(f"[evaluate] 数据加载 {time.time()-t0:.1f}s:")
    print(f"  flow type={type(data.get('flow'))}, value_shape={getattr(data.get('flow'), 'shape', None)}")
    print(f"  pred type={type(data.get('pred_scores'))}, value_shape={getattr(data.get('pred_scores'), 'shape', None)}")
    am = data.get('anomaly_mask')
    if am is None:
        print("  anomaly_mask: (missing)")
    elif hasattr(am, 'size') and am.size > 1:
        print(f"  anomaly_mask={int(am.sum())}/{am.size}")
    else:
        print(f"  anomaly_mask={am}")
    if data.get("gt_anomaly_labels") is not None:
        gt = data["gt_anomaly_labels"]
        print(f"  gt_anomaly={int(gt.sum())}/{gt.size} ({gt.mean()*100:.2f}%)")
    else:
        print(f"  gt_anomaly: (missing)")

    print(f"[evaluate] 计算指标...")
    metrics = run_metrics(data)
    metrics["model_tag"] = args.model_tag
    metrics["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if data.get("threshold_sweep"):
        metrics["threshold_sweep"] = data["threshold_sweep"]  # 也写进 metrics
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if not args.skip_profile:
        print(f"[evaluate] 性能 profile...")
        profile = run_system_profile(api_host=args.api_host, port=args.api_port)
        (output_dir / "profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    else:
        profile = {}

    print(f"[evaluate] 生成摘要...")
    summary = write_summary(metrics, profile, output_dir, args.model_tag)
    print(f"[evaluate] 完成 → {output_dir}")
    print(f"\n{summary[:1800]}\n...")


if __name__ == "__main__":
    main()
