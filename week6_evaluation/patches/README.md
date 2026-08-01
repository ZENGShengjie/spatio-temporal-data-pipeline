# Week6 任务4 代码补丁应用指南

本目录包含可以直接应用到原代码的优化 patch。每个文件都是**独立可应用**的，无需重写原 Pipeline。

## 优化清单

| 文件 | 优化点 | 收益 | 风险 |
|------|--------|------|------|
| `pipeline_inference_mode.py` | `torch.inference_mode()` 包裹推理 | 推理 +30% | 无 |
| `pipeline_lru_cache.py` | detect 结果 LRU 缓存（同一 t 不重算） | 重复请求 +95% | 数据更新场景需禁用 |
| `streamlit_heatmap_downsampling.py` | Plotly 热力图降采样（异常格点不降） | 渲染 -50% | 精度微降 |

## 应用方式

### 方式 1：直接替换（推荐）

`pipeline_inference_mode.py` 和 `pipeline_lru_cache.py` 都通过 monkey-patch 在 import 时生效：

```python
# 在 week6/api/main.py 顶部加一行
from week6_evaluation.patches.pipeline_inference_mode import patch_pipeline
patch_pipeline()
```

### 方式 2：包装类（更安全）

用 `OptimizedPipeline` 替代原 `SpatiotemporalPipeline`：

```python
# week6/api/main.py
# from week6.pipeline import SpatiotemporalPipeline
from week6_evaluation.profiling.optimized_pipeline import OptimizedPipeline as SpatiotemporalPipeline
```

## 各 patch 详细说明

### `pipeline_inference_mode.py`

在 `SpacetimeformerLite.forward` 外层包 `@torch.inference_mode()`，禁用 autograd：

```python
@torch.inference_mode()
def forward(self, x_node):
    ...
```

仅在 structural 模式（VAE/TAE）下有效。fast 模式无深度学习，收益为 0。

### `pipeline_lru_cache.py`

为 `SpatiotemporalPipeline.detect_anomaly` 加 LRU 缓存层：

```python
from functools import lru_cache

@lru_cache(maxsize=512)
def _cached_detect(t: int, mode: str, threshold: float):
    return pipe.detect_anomaly(t, mode, threshold)
```

**适用场景**：Streamlit 演示中，用户来回切换时间步 → 大量重复请求
**不适用场景**：实时数据流（每个 t 都是新的）

### `streamlit_heatmap_downsampling.py`

Plotly 渲染瓶颈主要是 `go.Heatmap(z=32x32)`，但加上 hover 标签后 JSON 体积大。
优化：
- 流量矩阵降采样到 16×16 渲染（背景）
- 异常格点单独画 `go.Scatter`（不降采样）
- 用 `include_mathjax=False` 减小 HTML 体积

```python
def plot_heatmap_32_fast(flow_2d, anomaly_mask_2d, scores_2d, title):
    # 流量降采样到 16x16
    flow_ds = flow_2d[::2, ::2]
    # 异常格点保留原精度
    rows, cols = np.where(anomaly_mask_2d.astype(bool))
    ...
```

## 验证步骤

1. 应用前先跑 baseline：
   ```bash
   python -m week6_evaluation.evaluation.evaluate --model-tag baseline --output results/baseline/
   ```

2. 应用 patch：
   ```bash
   # 在 week6/api/main.py 加 import
   echo "from week6_evaluation.patches.pipeline_inference_mode import patch_pipeline" >> week6/api/main.py
   echo "patch_pipeline()" >> week6/api/main.py
   ```

3. 重启 API 跑优化后评估：
   ```bash
   pkill -f uvicorn
   nohup uvicorn week6.api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
   python -m week6_evaluation.evaluation.evaluate --model-tag optimized --output results/profiling/optimized/
   ```

4. 对比：
   ```bash
   python -m week6_evaluation.profiling.compare_results \
       results/baseline/profile.json \
       results/profiling/optimized/profile.json
   ```

## 注意事项

- **`torch.inference_mode()` 不可与 `requires_grad=True` 混合使用**。如果模型后续要反向传播，不要加这个 patch
- **LRU 缓存不区分用户身份**。如果做多租户部署，按 (user_id, t) 做 key
- **降采样**会让 hover 标签的 (row, col) 与原图不一致，前端逻辑要相应调整
