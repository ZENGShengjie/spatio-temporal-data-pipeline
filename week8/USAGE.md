# USAGE — 深度使用说明

> 配合 QUICKSTART.md 使用。本文档解释每个模块的**设计理念 + API 用法**，
> 不重复 QUICKSTART 的"按这个命令跑"部分。

---

## 1. 总体目录约定

所有模块均遵循"配置 → 数据 → 模型 → 训练 → 评估"五段式：

```
week*/                                ← 每周一个独立子项目
├── README.md                         ← 该周总览
├── config.py                         ← 全局配置（数据路径、超参、随机种子）
├── data_loader.py                    ← 数据加载器（SeqDataset 等）
├── models/                           ← 模型定义
├── run_week*.py                      ← 主入口
├── results/                          ← 评估结果（.npy + summary.md）
├── weights/                          ← 训练好的 .pth
└── report/                           ← 报告文档
```

每个周之间通过**目录隔离**避免 import 冲突（不强制使用 `-m` 调用，但推荐）。

---

## 2. 数据层

### 2.1 数据加载

`week2/scripts/` 负责原始 H5 → 32×32 网格流量的 ETL，关键产物：

- `BJ16_M32x32_T30_InOut.h5`：P4（2015-11-01 → 2016-04-10）30 分钟粒度
- `taxi_flow_total`（流入 + 流出）/ `taxi_inflow` / `taxi_outflow`：可作为预测目标
- `weather_era5.npy`：7 维气象特征
- `poi_grid_2015_11.npy`：10 类 POI 密度

### 2.2 时空切分

| 集合 | 时间范围 | 步数 | 用途 |
|------|---------|------|------|
| train | [0, 2784) | 2784h | 模型训练 |
| val | [2784, 3288) | 504h | 阈值搜索 / 超参调优 / 融合权重 |
| test | [3288, 3888) | 600h | **仅**用于最终评估 |

切分由 `week*/config.py` 控制；**严禁**用 test 调参。

### 2.3 异常注入

Week5 的 `inject_anomalies.py` 在 val/test 上注入 4% 比例异常：

```bash
cd week5
python3 inject_anomalies.py --ratio 0.04 --mode surge_drop_sustained
```

注入类型：
- **surge**：突增 50%-200%，模拟活动散场
- **drop**：突降 30%-70%，模拟极端天气
- **sustained**：连续 2-4h 异常，模拟施工

---

## 3. 模型层

### 3.1 Week3 基线（7 个时序模型）

| 模型 | 文件 | 类别 |
|------|------|------|
| ARIMA | `week3/models/arima_model.py` | 统计 |
| Prophet | `week3/models/prophet_model.py` | 统计 |
| LSTM | `week3/models/lstm_model.py` | 深度时序 |
| GRU | `week3/models/gru_model.py` | 深度时序 |
| GCN | `week3/models/gcn_model.py` | 深度图 |
| GAT | `week3/models/gat_model.py` | 深度图 |
| GRU+STGCN 残差 | `week3/models/gru_stgcn_residual.py` | 混合 |

统一通过 `week3/registry.py: get_trainer(name)` 调用。

### 3.2 Week4 时空联合（3 个核心模型 + 2 消融）

| 模型 | 文件 | 参数量 | 主要特性 |
|------|------|--------|---------|
| **STF** | `week4/models/stf_model.py` | 222K | 时空分解 + Env Token，最优综合 |
| AGFormer | `week4/models/agformer_model.py` | 2.26M | 自适应邻接 + 多头注意力 |
| STGCN | `week4/models/stgcn_model.py` | 200K | 时空图卷积 |

消融：
- `stf_loc_only`：移除 Env Token + 跨节点注意力
- `agformer_static`：冻结自适应邻接矩阵

### 3.3 Week5 异常检测（4 范式 + 融合 V3）

| 方法 | 文件 | 输入 |
|------|------|------|
| 统计阈值 | `week5/anomaly/statistical.py` | 3σ + IQR 分时段 |
| 预测误差 | `week5/anomaly/prediction.py` | Week4 STF 预测 |
| VAE | `week5/anomaly/vae_v3.py` | 单网格重构 |
| Transformer AE | `week5/anomaly/transformer_ae_v3.py` | 多网格掩码重构 |
| **融合 V3** | `week5/anomaly/fusion_v3.py` | 加权组合 + 阈值搜索 |

融合 V3 通过验证集网格搜索确定最优权重 `(w_stat, w_pred, w_vae, w_tae)` + 阈值。

---

## 4. 训练

### 4.1 标准训练

```bash
cd week4
python3 run_week4.py --models stf --target taxi_flow_total --tag myrun
```

参数：
- `--models`（必填）：模型列表，空格分隔
- `--target`：`taxi_flow_total` / `taxi_inflow` / `taxi_outflow`
- `--tag`：结果文件后缀，便于并行实验
- `--ablation`：是否包含消融模型
- `--skip_baseline`：跳过 GCN/GAT 基线对比

### 4.2 自定义超参

修改 `week4/config.py`：

```python
BATCH_SIZE = 8
LEARNING_RATE = 1.176e-4   # Week7 Optuna 最优值
HIDDEN = 64
N_HEADS = 2
N_LAYERS = 1
DROPOUT = 0.137
EPOCHS = 50
PATIENCE = 12
```

### 4.3 GPU vs CPU

默认配置尝试 CUDA，否则 fallback CPU。
训练 STF 在 T4 GPU 上约 515 秒，CPU 上约 2-3 小时。

---

## 5. 评估

### 5.1 Week3 / Week4 指标

`metrics.py:evaluate_predictions()` 计算：

- **MAE**：平均绝对误差
- **RMSE**：均方根误差
- **MAPE**：平均绝对百分比误差
- **Corr**：相关系数（**跨粒度唯一可比指标**）
- **Direction Accuracy**：t+1 方向预测准确率

输出 `.md` 报告：`results/summary_<target>_<tag>.md`

### 5.2 Week5 异常检测指标

`run_v3_full_eval.py` 计算：

- Precision / Recall / F1（不同阈值）
- AUC-ROC
- 阈值扫描表（0.5 ~ 0.98）

输出 `.json` 和 `.md`。

### 5.3 Week7 综合评估

`week7/evaluation/` 提供：

- 5 维指标（预测 / 异常 / 合理性 / 事件 / 系统）
- API E2E 性能测试
- 可解释性产物

---

## 6. API 服务（Week6）

### 6.1 启动

```bash
cd week6
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 6.2 关键端点

#### `GET /api/health`
返回服务状态、GPU 可用性、缓存命中率。

#### `POST /api/anomaly/detect`

请求体：
```json
{
  "mode": "fast",                // fast | structural
  "time_range": [2784, 3288],    // 时间窗口（小时索引）
  "grid_ids": [0, 1, 2, ...]     // 待检测网格，空=全部
}
```

响应：
```json
{
  "scores": [[...], ...],       // 异常得分 (T, N)
  "labels": [[...], ...],       // 二值标签 (T, N)
  "metadata": {
    "threshold": 0.540,
    "weights": [0.5, 0.5, 0, 0],
    "elapsed_ms": 9.8
  }
}
```

#### `GET /api/forecast`

请求：`?horizon=48&grid_ids=0,1,2`

响应：
```json
{
  "predictions": [[...], ...],   // (H, N)
  "conf_low": [[...], ...],      // 95% 置信下界
  "conf_high": [[...], ...]      // 95% 置信上界
}
```

### 6.3 性能优化

- **LRU 缓存**：相同请求体 1000 条容量
- **orjson**：JSON 序列化加速
- **inference_mode**：禁用梯度跟踪
- **动态批处理**：根据请求队列自适应

---

## 7. Streamlit 可视化（Week6）

### 7.1 启动

```bash
cd week6
API_BASE=http://localhost:8000 streamlit run app.py
```

### 7.2 功能模块

1. **实时监控**：32×32 网格热力图 + 异常区域高亮
2. **24 小时动画**：Plotly Frames 逐帧播放
3. **事件列表**：历史异常事件按时间排序
4. **单网格分析**：时序图 + 异常事件标注
5. **地理地图**：MapLibre + 高德路网（无需 API key）
6. **三级预警**：弹窗 + WebAudio 声音

### 7.3 自定义 API 端点

修改 `week6/app.py` 第 38 行：
```python
API_BASE = os.environ.get("API_BASE", "http://your-server:8000")
```

---

## 8. Optuna 超参优化（Week7）

### 8.1 启动搜索

```bash
cd week7/optimization
python3 optuna_stf.py --n-trials 30 --timeout 7200
```

### 8.2 搜索空间

| 超参 | 范围 |
|------|------|
| hidden | [32, 256] |
| n_layers | [1, 4] |
| n_heads | [1, 8] |
| learning_rate | [1e-5, 1e-2]（对数） |
| batch_size | [4, 32] |
| dropout | [0.0, 0.5] |

### 8.3 重训最优参数

```bash
python3 retrain_best.py   # 读取 study_result.json 最优 trial，重训 STF
```

---

## 9. 可解释性（Week7）

### 9.1 注意力可视化

```bash
cd week7/interpretability
python3 attention_vis.py
```

输出：
- 空间注意力热力图（每个 head）
- 时间注意力分布

### 9.2 SHAP 分析

```bash
python3 shap_analysis.py
```

输出：
- 全局特征重要性排序
- 单样本 SHAP 值
- 瀑布图

---

## 10. 高级用法

### 10.1 自定义目标

在 `week*/config.py` 中修改：

```python
TARGET_OPTIONS = ["taxi_flow_total", "taxi_inflow", "taxi_outflow"]
```

### 10.2 自定义网格

修改 `week*/config.py`：

```python
GRID_H = 16
GRID_W = 16
```

注意：需要重新生成数据（`week2/scripts/`），旧模型权重不兼容。

### 10.3 多 GPU 训练

`week4/models/stf_model.py` 已支持 `torch.nn.DataParallel`：

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 run_week4.py --models stf --tag multi_gpu
```

---

## 11. 调试技巧

### 11.1 快速验证

```bash
# 仅加载数据不训练
python3 run_week3.py --data_only

# 1 epoch 快速跑（修改 config.EPOCHS = 1）
```

### 11.2 单元测试

各模块通过 `if __name__ == "__main__"` 提供快速 sanity check。

### 11.3 日志查看

```bash
tail -f logs/gru.log
tail -f logs/stf.log
```

---

## 12. 常见配置修改

| 修改项 | 位置 | 默认值 |
|--------|------|--------|
| 随机种子 | `week*/config.py` | 42 |
| Batch size | `week*/config.py` | 8 |
| 学习率 | `week*/config.py` | 1e-3 |
| 早停 patience | `week*/config.py` | 12 |
| 数据路径 | `week*/config.py` | `data/raw_bj/` |
| 结果输出 | `week*/config.py` | `results/` |
| 权重输出 | `week*/config.py` | `weights/` |

---

## 13. 复现清单（答辩前最后检查）

```bash
# 1. 数据：bit-for-bit 等于原始 H5
python3 docs/scripts/verify_p0_data.py

# 2. 模型：STF 权重可重新加载
python3 -c "
import torch
m = torch.load('week4/weights/stf_taxi_flow_total_v4fix.pth')
print('keys:', list(m.keys())[:3])
"

# 3. API：服务可启动 + 端点响应正常
curl http://localhost:8000/api/health

# 4. 复现数字：
# - GRU MAE ≈ 158.12（Week3）
# - STF MAE ≈ 327.19（Week4）
# - 融合 V3 F1 = 0.9165（Week5，统计+STF真实推理+VAE 三路融合，V3 注入测试集，4% 注入率）
```

---

## 下一步

- 阅读 `docs/技术报告.md` 了解完整设计与实验
- 阅读 `week5/README.md` 了解异常检测细节
- 阅读 `week7/README.md` 了解 Optuna 与可解释性
- 浏览 `docs/ppt.html` 看答辩演示