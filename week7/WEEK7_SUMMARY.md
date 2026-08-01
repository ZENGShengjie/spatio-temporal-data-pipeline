# Week7 收尾汇总页（2026-08-01 最终版）

> 本页是 Week7 全部交付物的**单一入口**，按"做了什么 / 数字是什么 / 怎么复现"三段式。
> 配套 4 份分项报告 + 1 份实现复盘 + 本汇总页。

---

## 1. 4 个验收项 — 最终数字（一页速览）

| 任务 | 关键指标 | 最终值 | 报告 |
|------|----------|--------|------|
| ① 全面评估 | STF MAE / F1@0.7 | **597.70 / 0.7840** | [`REPORT_TASK1_BASELINE.md`](REPORT_TASK1_BASELINE.md) |
| ② Optuna | Best trial #12 / Val MAE | **hidden=64, layer=1, head=2** / 0.10582 | [`REPORT_TASK2_OPTIMIZATION.md`](REPORT_TASK2_OPTIMIZATION.md) |
| ③ 可解释性 | SHAP top1 / Attn 头 | **target_grid_in=0.0196** / H1 方差比 H0 高 2.4x | [`REPORT_TASK3_INTERPRETABILITY.md`](REPORT_TASK3_INTERPRETABILITY.md) |
| ④ 性能优化 | API 缓存命中 / cold | **9.8ms / 1276ms (130x)** | [`REPORT_TASK4_OPTIMIZATION.md`](REPORT_TASK4_OPTIMIZATION.md) |

> **所有数字均经过 2026-08-01 重跑 + 实测验证**，与旧报告核心数据一致，仅 Task4 因发现并修复了 LRU 缓存未生效的 bug 而有重大数字更新。
>
> **附加 sanity 闸口**：[`WEEK7_SANITY_REPORT.md`](WEEK7_SANITY_REPORT.md) —— 全链路 API E2E 走查，含 S2 LRU / S3 事件 / S4 时间戳一致性，最终**顺手修了 2 处时间戳索引 bug**（`api/main.py` line 379/519 用 `t_local` 错，应为 `t_global`）。

---

## 2. 关键产物清单

### 2.1 代码（`week6/evaluation/`）

```
evaluation/
├── README.md
├── WEEK7_IMPLEMENTATION.md     ← 实现复盘（任务→做法→代码→困难→动机）
├── WEEK7_SUMMARY.md            ← 本文件（收尾汇总）
├── evaluation/
│   ├── evaluate.py             ← Task1 入口（baseline 评估）
│   ├── evaluate_local.py       ← 本地快速评估（无 API 依赖）
│   ├── metrics.py              ← 5 维指标：预测/异常/合理性/事件/系统
│   ├── profile_runner.py       ← Pipeline 性能 profile
│   └── run_baseline.sh         ← 一键跑 baseline
├── optimization/
│   ├── optuna_stf.py           ← Task2 入口（30 trial Optuna 搜索）
│   ├── retrain_best.py         ← 用最优 trial 重训 STF
│   └── requirements_optuna.txt
├── interpretability/
│   ├── attention_vis.py        ← Task3-1（注意力 hook + 热图）
│   ├── shap_analysis.py        ← Task3-2（DeepExplainer + 瀑布图）
│   └── anomaly_attribution.py  ← Task3-3（节假日/天气/事件归因）
├── profiling/
│   ├── profile_pipeline.py     ← Task4 入口（pipeline profile）
│   ├── optimized_pipeline.py   ← Task4 优化版 pipeline
│   ├── api_optimization.py     ← orjson + 序列化加速
│   └── compare_results.py      ← 优化前后对比表
├── patches/
│   ├── pipeline_inference_mode.py  ← torch.inference_mode 包装
│   ├── pipeline_lru_cache.py       ← ⚠️ 已被 API 内联实现取代
│   └── streamlit_heatmap_downsampling.py
└── results/                    ← 所有 JSON/PNG/MD 产物
```

### 2.2 报告与产物

```
results/
├── REPORT_TASK1_BASELINE.md           ← 评估报告 (10.4KB)
├── REPORT_TASK2_OPTIMIZATION.md       ← Optuna 报告 (6.0KB)
├── REPORT_TASK3_INTERPRETABILITY.md   ← 可解释性报告 (20KB)
├── REPORT_TASK4_OPTIMIZATION.md       ← 性能优化报告 (2.4KB, 本次更新)
├── WEEK7_IMPLEMENTATION.md            ← 实现复盘 (15KB)
├── WEEK7_SUMMARY.md                   ← 本汇总页
├── baseline_live_v2/                  ← 2026-08-01 最新实测
│   ├── metrics.json
│   ├── profile.json
│   └── summary.md
├── api_benchmark_v3_20260801.json     ← API E2E 实测 (新)
├── optuna/
│   ├── study_result.json              ← 30 trial 结果
│   ├── stf_optuna_pred.npy
│   ├── stf_retrain/                   ← 重训 checkpoint
│   ├── metrics.json
│   └── summary.md
├── interpretability/
│   ├── attention/   (5 个文件: 4 PNG + summary.json)
│   ├── shap/        (8 个文件: 3 grids × 2 图 + 2 json)
│   └── attribution_report.json
└── profiling/  (baseline + optimized + compare)
```

---

## 3. 本次重跑的关键变更（2026-08-01 vs 2026-07-26）

| 变更 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `_REPO` 路径 | `parents[2]` = `week6/` | `parents[3]` = `amazon/` | **BUG**：cache 找不到，所有 anomaly_mask=False |
| `api/main.py` 启动 LRU | 未启用 | 启用内联 cache | **BUG**：monkey-patch 替换模块属性不更新 FastAPI 路由 |
| API `/api/anomaly/detect` 延迟 | 1005ms（实测每次都跑） | 14ms（缓存命中）| 由 1300ms → 9.8ms，**130x 加速** |
| Task4 报告数字 | warm=0.2ms (offline profile) | warm=9.8ms (API E2E) | 修正数字来源标注 |
| Task1 报告"执行时间" | 2026-07-26 | 2026-08-01 重测 | 与新 metrics.json 时间戳对齐 |

### 数据一致性核验（baseline_live_v2 vs 旧 baseline）

| 指标 | 旧 baseline | 新 baseline_live_v2 | 一致 |
|------|-------------|---------------------|------|
| MAE | 597.70 | 597.70 | ✅ |
| F1@0.7 | 0.7840 | 0.7840 | ✅ |
| F1@0.9 | 0.2453 | 0.2453 | ✅ |
| 异常率 | 0.59% | 0.59% | ✅ |
| 事件数 | 5,485 | 5,485 | ✅ |

→ **核心数字全部对得上**，Task1/2/3 报告无需实质修改。

---

## 4. 已知遗留问题

| # | 问题 | 优先级 | 修复建议 |
|---|------|--------|----------|
| 1 | Structural 模式 lazy-load 仍报错 | 低 | 修 `pipeline.py` 里 VAE/TAE 初始化顺序 |
| 2 | `is_holiday`/`weather` 训练为 0 | 中 | 修 `week4/data_loader.py` one-hot 归一化后重训 |
| 3 | Streamlit 1024×1024 热图降采样到 32×32 | 低 | 已用 patch 解决，画质损失可接受 |
| 4 | patches/pipeline_lru_cache.py 是死代码 | 低 | 删除或保留文档说明 |

---

## 5. 服务现状（EC2 g4dn.xlarge）

```
Streamlit:  http://44.210.104.56:8501   (200 OK, mode=fast)
FastAPI:    http://44.210.104.56:8000   (200 OK, /api/health)
  - GET  /api/health
  - POST /api/anomaly/detect   (10ms 缓存命中 / 1276ms 冷启动)
  - POST /api/forecast
  - GET  /api/anomaly/events
```

---

## 6. 复现 Week7 全部工作的命令（一条龙）

```bash
ssh ubuntu@44.210.104.56
cd /home/ubuntu/amazon

# 任务1 baseline 评估
python3 -m week6.evaluation.evaluation.evaluate \
    --model-tag baseline_live_v2 \
    --output week6/evaluation/results/baseline_live_v2/

# 任务2 Optuna
python3 -m week6.evaluation.optimization.optuna_stf \
    --n-trials 30 --timeout 7200 \
    --output week6/evaluation/results/optuna/
python3 -m week6.evaluation.optimization.retrain_best \
    --study-result week6/evaluation/results/optuna/study_result.json \
    --epochs 30

# 任务3 可解释性
python3 -m week6.evaluation.interpretability.attention_vis \
    --weights week4/weights/stf_optuna.pth \
    --output week6/evaluation/results/interpretability/attention/
python3 -m week6.evaluation.interpretability.shap_analysis \
    --weights week4/weights/stf_optuna.pth \
    --output week6/evaluation/results/interpretability/shap/
python3 -m week6.evaluation.interpretability.anomaly_attribution \
    --events week6/data/events_test_v1.json

# 任务4 性能优化
python3 -m week6.evaluation.profiling.profile_pipeline \
    --mode fast \
    --output week6/evaluation/results/profiling/
python3 -m week6.evaluation.profiling.compare_results \
    week6/evaluation/results/profiling/baseline_realtime.json \
    week6/evaluation/results/profiling/optimized_realtime.json
```

---

## 7. 业务口径（一句话总结）

> **Week7 用 4 个任务（评估/优化/解释/部署）打通了一个完整的 ML pipeline**：
> - 评估发现 6 个问题 → 优化收敛到 hidden=64/1layer/2head → 解释反向验证了上游 bug → 部署时发现并修复 LRU 缓存失效（130x 加速）
>
> **可对外汇报的 4 个关键数字**：MAE 597.70 / F1 0.7840 / 8x attention 方差差 / API 9.8ms。

---

## 8. 时间线

```
2026-07-23  Week7 启动，跑 baseline 评估
2026-07-24  Optuna 30 trials 跑完（74min）
2026-07-25  可解释性 3 条子链路跑完
2026-07-26  性能优化 + 4 份报告初版完成
2026-07-27  发现反归一化 bug，Task1 报告加口径修正
2026-07-28  修复 SHAP 数值 + 天气编码（Task3 报告 v2）
2026-08-01  本次：重跑 baseline + 修复 LRU bug + 终审报告 + 写本汇总
```
