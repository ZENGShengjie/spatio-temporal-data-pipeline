# Week5 预测误差法 V2 — STF 真实推理修复报告

> **修复时间**：2026-08-17
> **修复范围**：`week5/anomaly/prediction.py` 的 `_generate_val_predictions_from_weights`
> **影响**：Fusion V3 异常检测 F1 从 baseline 0.7687 提升到 **0.9515**（+0.1828，相对 +23.8%）
> **实验条件**：V3 注入测试集，4% 注入率，混合突增/突降/持续模式（sustained_pct=20%）

---

## 1. 问题背景

`week5/anomaly/prediction.py` 是 Week5 异常检测的预测误差法（prediction-based）实现。
其 `_generate_val_predictions_from_weights` 原代码只是输出注释：

```python
# 构造模型（简化版：不依赖具体模型结构，用均值 baseline）
# 这里用均值 baseline，因为加载STF模型结构过于复杂
# 真正的实现应该在这里实例化 STF 并推理
print("[pred V2] NOTE: using history-mean baseline (STF model loading needs model class)")
return self._generate_history_mean_baseline()
```

也就是说，**整个预测误差法一直在用历史同期均值作为"STF 预测"**，从未真正加载 Week4 训练好的 STF 权重。

## 2. 根因

1. **本地 STF 权重缺失**：`week4/weights/` 在本地只有 `README.md`，没保存 `.pt` 权重。
2. **代码占位**：`_generate_val_predictions_from_weights` 上面写了"用均值 baseline 因为 STF 模型结构过于复杂"，是占位实现。
3. **伪装路径**：路径 A 找到了 `week5/cache/stf_val_predictions.npy`（2064512 字节，shape=504×1024），所以代码从来不进入 fallback 路径——看起来"成功了"，实际跑的是 V2 之前 history-mean 跑出来的缓存。

## 3. 修复过程

### 3.1 找到 STF 权重

```bash
ssh ubuntu@<EC2_PUBLIC_IP> 'ls /home/ubuntu/amazon/week4/weights/'
# → stf_taxi_flow_total_v4fix.pth (909670 bytes, epoch=12, n_params=222,576)
```

下载到本地：
```bash
scp ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/amazon/week4/weights/stf_taxi_flow_total_v4fix.pth \
    E:/amazon/week4/weights/
```

### 3.2 Checkpoint 结构

| 字段 | 值 |
|---|---|
| `model_name` | `stf` |
| `target` | `taxi_flow_total` |
| `tag` | `v4fix` |
| `best_epoch` | 12 |
| `n_params` | 222,576 |
| `loc_proj.weight` | `(64, 7)` ← **关键**：in_dim=7 = 2 路 flow + 5 时间特征 |

### 3.3 实现真正推理

`_generate_val_predictions_from_weights` 重写：

1. 加载归一化流量（仅用训练集 cell_max） + 时间特征（5 维）
2. 实例化 `SpacetimeformerLite(in_dim=2+K_time, hidden=64, horizon=48, n_heads=4, n_layers=2)`
3. **对每个 val 时间点 t_off ∈ [0, 504) 构造一个窗口**：
   - `t_end = VAL_START + t_off`
   - `x = flow[t_end - seq_len : t_end]` 形状 `(2*N + K_time, 48)`
   - 模型输出 `(N, 48)`，取 `[:, 0]`（horizon 第一步）作为 `val[t_off]` 的预测
4. 推理耗时：4.1 秒（504 窗口、batch=8）

### 3.4 关键技术点

- **路径冲突**：`week5/data_loader.py` 和 `week4/data_loader.py` 同名，必须先 `sys.path.insert(0, WEEK4)` 再绝对路径导入。
- **`STFBaseTrainer` 没有 `.model` 字段**：构造 trainer 后再手工实例化 `SpacetimeformerLite` 并挂到 `trainer.model = model`。
- **horizon 偏移**：训练时 `t_end = t - horizon + 1`，推理时输入窗口必须用 `[t_end - seq_len : t_end]`，不能用 `[t - seq_len : t]`。

## 4. 验证结果

### 4.1 STF 推理精度（验证集）

| 指标 | 值 |
|---|---|
| 验证集 MAE（归一化空间） | **0.4246** |
| 验证集 MAE（反归一化空间） | **430.79** |
| 最佳 epoch | 12 |
| 参数量 | 222,576 |

### 4.2 单路预测 F1（验证集）

| 阈值 | STF F1 | Baseline F1 |
|---|---|---|
| 0.5 | 0.168 | 0.650 |
| 0.7 | 0.176 | 0.780 |
| 0.8 | 0.228 | 0.803 |
| 0.9 | 0.304 | 0.796 |
| 1.5 | 0.632 | — |

**反直觉**：单路 STF 相对误差法反而不如 baseline。原因是 STF 拟合了正常模式的细粒度细节，异常突变被更大的正常误差方差掩盖。

### 4.3 Fusion V3 A/B 对比（关键）

| 模式 | BASELINE F1（融合前） | **STF F1（融合后）** | Δ F1 |
|---|---|---|---|
| **all**（三路：统计+预测+VAE） | 0.7687 | **0.9515** | **+0.1828** |
| **dual**（两路：统计+预测） | 0.7687 | **0.9515** | **+0.1828** |
| stat_only | 0.7929 | 0.7929 | 0 |

**关键发现**：
- baseline 时 fusion 抛弃 prediction 路（weight=0.0），只用 statistical
- STF 时 fusion 真的用上 prediction 路（weight=0.5），与 statistical 加权融合
- AUC 也从 0.9650 提升到 **0.9751**

虽然 STF 单路 F1 不如 baseline，但 STF 提供了**与 statistical 正交的信息**——两者加权融合后 F1 从 0.77 飙到 0.95。

### 4.4 融合权重

```
[fusion V3] search: weights={'statistical': 0.5, 'prediction': 0.5}, thresh=0.540
[fusion V3] all: P=0.9900 R=0.9159 F1=**0.9515** AUC=0.9751 （V3 注入测试集，三路融合：统计 0.5 + STF预测 0.5）
```

## 5. 部署

- API 路由：`POST /api/anomaly/detect`，mode=structural 返回 `processing_ms ≈ 92`（< 100ms SLA）
- Streamlit：`http://<EC2_PUBLIC_IP>:8501`
- 缓存：`week5/cache/pred_scores_*_v2.npy`（已覆盖为 STF 结果）
- 备份：`pred_scores_*_v2_BASELINE.npy` / `pred_scores_*_v2_STF.npy` 保留对比

## 6. 修改文件

| 文件 | 变更 |
|---|---|
| `week5/anomaly/prediction.py` | `_generate_val_predictions_from_weights` 真正实现 STF 推理 |
| `week5/anomaly/fusion_v3.py` | `from data_loader import ...` 改绝对引用（修 path 冲突） |
| `week4/weights/stf_taxi_flow_total_v4fix.pth` | 从 EC2 下载到本地 |
| `week5/cache/pred_scores_*_v2.npy` | 已覆盖为 STF 真实推理结果 |
| `week5/cache/fusion_scores_*_v3.npy` | 已重跑，all/dual 模式 F1=**0.9515**（vs 基线 0.7687，+23.8%） |

## 7. 数据泄露红线验证

- ✅ STF 训练时只用训练集 cell_max
- ✅ 验证集 / 测试集的真值都用注入后的 `flow_val_injected.npy` / `flow_test_*.npy`
- ✅ 阈值仅在验证集上搜索（F1 最大化）
- ✅ Fusion 权重仅在验证集上搜索

## 8. 后续建议

1. **融合再优化**：STF prediction 路单路 F1 不高，但融合后极强。可以研究 STF 残差的 top-k 异常检测（而非全局均值），可能进一步提升
2. **权重持久化**：fusion_v3 的 weights 现已写入 `week5/cache/fusion_params_v3.json`（含 `statistical: 0.5, prediction: 0.5`），但需要在重启服务时验证是否被正确加载
3. **STF 权重本地化**：以后 STF 训练完后，应自动同步 `.pt` 到本地 `week4/weights/`，避免再次发生"代码占位但缓存能用"的隐蔽 bug
