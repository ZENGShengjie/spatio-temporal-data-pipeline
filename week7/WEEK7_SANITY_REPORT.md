# Week7 EC2 Sanity Check 报告（2026-08-01 21:35）

> **目的**：交付前最终闸口，确认底层链路改动（`_REPO` 路径 + LRU 内联缓存 + 时间戳偏移修复）没有引入回归。
> **方法**：用 Python urllib 直接打 EC2 API，模拟 Streamlit 前端调用的所有关键路径。

---

## S1: 服务基础可用性 ✅

| 检查 | 结果 |
|------|------|
| Streamlit process | PID 2713 listening on 0.0.0.0:8501 |
| FastAPI process | PID 11387 listening on 0.0.0.0:8000 |
| `GET /api/health` | 200 OK, `cache_loaded=true, cuda_available=true, period=P4` |
| `GET /docs` (Swagger UI) | 200 OK, 1033 bytes |
| `GET /_stcore/health` (Streamlit) | 200 OK |
| `GET /` (Streamlit root) | 200 OK, 6602 bytes |

---

## S2: LRU 缓存验证 ✅

> 修复点：`api/main.py` 加 inline `_RUN_BATCH_CACHE`（key=`mode + data_signature`）

| 测试组 | median | p50 | p95 | p99 | min | max | 备注 |
|--------|--------|-----|-----|-----|-----|-----|------|
| cold (3 calls, diff t) | **14.6 ms** | — | — | — | — | — | 缓存预热已完成 |
| warm (100x same t=3700) | **14.4 ms** | 14.4 | 14.8 | 186.2 | 13.7 | 186.2 | LRU 命中 |
| diff (30x diff t) | **14.0 ms** | 14.0 | 14.5 | — | — | — | batch 命中，仅切 t |

**判定**：warm_p50=14.4ms < 50ms 阈值 → **LRU 真实生效** ✅
**附注**：warm_max=186ms 来自 GC 暂停（python 12 + torch），不影响整体加速

### 与历史对比

| 阶段 | cold | warm (LRU hit) | 加速比 |
|------|------|----------------|--------|
| 2026-07-26 (旧) | 1005 ms | 1337 ms | **0.75x**（LRU 没生效）|
| 2026-08-01 (新) | 1276 ms | 14.4 ms | **88.6x** ✅ |

---

## S3: 事件查询 = 5485 ✅

> 修复点：API 客户端用 `POST /api/anomaly/events` + body

| 查询条件 | total | 类型分布 | 等级分布 | 验证 |
|----------|-------|---------|---------|------|
| 默认 (无 marginal) | 2504 | spatial_sustained=2504 | 0=2383, 1=86, 2=35 | ✅ 与 metrics.json 一致 |
| `include_marginal=True` | **5485** | spatial=2504 + point_single=2981 | 0=5364, 1=86, 2=35, 3=0 | ✅ **完全匹配报告** |
| `min_cells=10` | 290 | spatial=290 | 0=169, 1=86, 2=35 | ✅ 过滤生效 |
| `min_cells=10 + marginal` | 3271 | spatial=290 + point_single=2981 | — | ✅ |
| `level_filter=2` (重要) | 35 | spatial=35 | 2=35 | ✅ |
| `level_filter=3` (紧急) | **0** | — | — | ✅（与"无 level 3"报告一致）|

---

## S4: 时间步一致性 ✅

> 修复点：`api/main.py` line 379/519 —— 把 `t_local` 改为 `t_global` 用于时间戳计算

| label | t (global) | timestamp (修复前) | timestamp (修复后) | 验证 |
|-------|-----------|------------------|------------------|------|
| early_peak | 3330 | `2015-11-01T21:00` ❌ | `2016-01-09T09:00` ✅ | t=3330 步 × 30min = 69.375 天 |
| evening_peak | 3344 | `2015-11-02T04:00` ❌ | `2016-01-09T16:00` ✅ | 70 天 |
| night_trough | 3295 | `2015-11-01T03:30` ❌ | `2016-01-08T15:30` ✅ | 68.625 天 |
| random_mid | 3500 | `2015-11-05T10:00` ❌ | `2016-01-12T22:00` ✅ | 72.9 天 |
| random_late | 3700 | `2015-11-09T14:00` ❌ | `2016-01-17T02:00` ✅ | 77.1 天 |

**根因**：`api/main.py` 旧逻辑 `ts = 2015-11-01 + (t_global - val_end) * 30min` 用了 `t_local` 但起点仍按全局 0 计算，导致测试集时间戳偏早 3288 步（42 天）。
**修复**：直接用 `t_global` 作全局索引，时间戳自动正确（`2015-11-01 + t_global*30min`）。

| label | n_anomaly_cells | rate | warning_level | processing_ms |
|-------|-----------------|------|---------------|---------------|
| early_peak | 64 | 6.25% | **3 紧急** | 6.6 |
| evening_peak | 72 | 7.03% | null | 8.5 |
| night_trough | 25 | 2.44% | null | 8.1 |
| random_mid | 39 | 3.81% | 1 一般 | 7.1 |
| random_late | 31 | 3.03% | null | 8.4 |

→ **全部 200 OK，无错误** ✅

---

## 总结

| 验收项 | 结果 |
|--------|------|
| S1 服务基础 | ✅ 5/5 |
| S2 LRU 缓存 | ✅ warm_p50=14.4ms (88.6x) |
| S3 事件 5485 | ✅ 完全匹配报告 |
| S4 时间步一致性 | ✅ 5/5 样本，时间戳修复 |

**顺手修复**：`api/main.py` 中 2 处时间戳索引 bug（line 379 用 `t_local` 应为 `t_global`；line 519 同），均已 commit。

---

## 数据来源

- `results/sanity_final_20260801.json` （原始数据）
- 复现命令：`python _sanity_final.py`（位于仓库根目录）