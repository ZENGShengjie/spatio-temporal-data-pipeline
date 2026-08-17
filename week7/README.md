# Week7 — 收尾交付 (2026-08-01)

> Week7 全部交付物的单一入口。从 week6/evaluation/results/ 拆出来独立成 week7/。

> **开发过程临时脚本**：本目录（含 `week7/optuna/`、`week7/interpretability/`、`week7/profiling/` 等）下若有 `_*.py` / `_*.sh` 前缀的脚本，多为开发过程的临时调试产物，**不在主入口复现流程内**。答辩使用主入口文件即可，临时脚本仅为过程证据保留。

## 目录结构

```
week7/
├── README.md                       ← 本文件
├── WEEK7_SUMMARY.md                ← 收尾汇总页（4 任务数字一览）
├── WEEK7_IMPLEMENTATION.md         ← 实现复盘（任务→做法→代码→困难→动机）
├── WEEK7_SANITY_REPORT.md          ← API E2E sanity 闸口（含时间戳修复）
├── REPORT_TASK1_BASELINE.md        ← 全面评估
├── REPORT_TASK2_OPTIMIZATION.md    ← Optuna 调参
├── REPORT_TASK2_OPTUNA.md          ← Optuna 详细 (study_result.json 解析)
├── REPORT_TASK3_INTERPRETABILITY.md← SHAP + Attention 可解释性
├── REPORT_TASK4_OPTIMIZATION.md    ← API 性能优化 (LRU 130x)
├── sanity_final_20260801.json      ← sanity 报告原始数据
├── api_benchmark_20260801_202832.json
├── api_benchmark_v3_20260801.json
├── baseline_live_v2_metrics.json   ← baseline_live_v2/ 顶层数据
├── baseline_live_v2_profile.json
├── baseline_live_v2_summary.md
├── baseline/                       ← baseline 评估子目录
├── baseline_live_v2/               ← 第二次 baseline 评估
├── local_smoke_test/               ← 本地 smoke test 产物
├── local_test/                     ← 本地 full test 产物
├── optuna/                         ← Optuna 调参 + STF 重训产物
│   ├── stf_optuna_pred.npy         ← Optuna 后最优 STF 在 test 上预测 (2.4MB)
│   └── stf_retrain/
│       └── stf_optuna.pth          ← Optuna 后重训的 STF 权重 (683KB)
├── interpretability/
│   ├── attention/                  ← 注意力头可视化
│   ├── attribution/                ← 异常归因报告
│   └── shap/                       ← SHAP summary / waterfall
└── profiling/                      ← pipeline 优化前后性能对比
```

## 4 个验收项 — 最终数字

| 任务 | 关键指标 | 最终值 | 报告 |
|------|----------|--------|------|
| ① 全面评估 | STF MAE / F1@0.7 | **597.70 / 0.7840** | [REPORT_TASK1_BASELINE.md](REPORT_TASK1_BASELINE.md) |
| ② Optuna | Best trial / Val MAE | **hidden=64, layer=1, head=2** / 0.10582 | [REPORT_TASK2_OPTUNA.md](REPORT_TASK2_OPTUNA.md) |
| ③ 可解释性 | SHAP top1 / Attn 头 | **target_grid_in=0.0196** / H1 比 H0 方差高 2.4x | [REPORT_TASK3_INTERPRETABILITY.md](REPORT_TASK3_INTERPRETABILITY.md) |
| ④ 性能优化 | API warm / cold | **14.4ms / 1276ms (88.6x)** | [REPORT_TASK4_OPTIMIZATION.md](REPORT_TASK4_OPTIMIZATION.md) |

> **附加 sanity 闸口**：[WEEK7_SANITY_REPORT.md](WEEK7_SANITY_REPORT.md) — 全链路 API E2E 走查。
> 含 S2 LRU / S3 事件 / S4 时间戳一致性，最终**修复了 API 时间戳索引 bug**（commit 9bb05fa：
> 之前用 `t_local*30min` 重算 NPZ 60min 真值，改为直接用 `pipe.run_batch()["timestamps"][t_global]`）。

## 关键 commits

- `9bb05fa` fix(api): use NPZ real timestamps instead of hardcoded 30min rebuild
- `f39c281` week7 sanity check: all 4 checks pass + fix timestamp offset bug
- `16c7020` week7 closeout: rerun baseline + fix LRU cache bug + final reports

## Week7 代码仍在 week6/evaluation/

Week7 的**代码框架**（evaluate.py / optuna_stf.py / shap_analysis.py / pipeline_inference_mode.py 等）
仍在 `week6/evaluation/` 下：evaluation/、optimization/、interpretability/、profiling/、patches/ 这 5 个子目录。
本目录 (`week7/`) 只放**产物（reports + JSON + PNG + 重训权重）**。

> 代码与产物分离原因：产物与时间戳、commit、运行环境强绑定；代码是 Week6 评估框架的自然延伸。

## 复现指引

```bash
# 1. 启动 API + Streamlit
cd week6 && python -m week6.api.main &
streamlit run week6/app.py --server.port 8501 &

# 2. 跑 Week7 4 项任务（每项约 1-3 min）
cd week6/evaluation
python evaluation/evaluate.py --mode baseline       # Task1
python optimization/optuna_stf.py --n-trials 30     # Task2
python interpretability/shap_analysis.py           # Task3
python profiling/profile_pipeline.py               # Task4

# 3. 看汇总
open week7/WEEK7_SUMMARY.md
```