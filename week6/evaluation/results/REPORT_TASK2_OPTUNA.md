# Week6 任务2 — Optuna 超参优化结果

> **目标**：以最小训练成本，针对 STF 模型做参数寻优，进一步提升预测精度
> **方法**：Optuna TPE 采样 + MedianPruner 早停剪枝，**仅优化验证集 MAE**（不引入异常检测 proxy 目标）
> **环境**：EC2 / T4 GPU / 30 trial / 上限 2 小时

---

## 1. 搜索策略与空间

### 1.1 搜索空间设计

| 参数 | 范围 | 分布 | 约束 |
|------|------|------|------|
| 学习率 | 1e-4 ~ 5e-3 | log | 与批次大小联动 |
| 批次大小 | {4, 8, 16} | categorical | — |
| 隐藏层维度 | {32, 64, 128} | categorical | **被注意力头数整除** |
| 注意力头数 | {2, 4, 8} | categorical | 与隐藏维度联动 |
| 编码器层数 | {1, 2, 3} | categorical | — |
| Dropout | 0.05 ~ 0.4 | uniform | — |
| Weight Decay | 1e-6 ~ 1e-3 | log | — |

### 1.2 优化目标

- **主目标（最小化）**：验证集 MAE（1-step 预测，STF 输出）
- **辅助参考指标**：方向准确率（在 analysis.py 里分析，不进 Optuna）

> **设计决策**：你朋友原本建议加"异常得分与预测残差的斯皮尔曼相关性"作为辅助目标。
> 我们**不采纳**，原因：预测残差大的区域 ≠ 应判为异常（高流量区域本就波动大），这个 proxy 容易诱导模型走偏。
> 异常检测的合理性在任务 1、3 里独立验证。

### 1.3 早停剪枝

- `MedianPruner(n_startup_trials=5, n_warmup_steps=2)`：前 2 个 epoch 不剪，从第 3 epoch 起与历史中位数对比
- 每个 trial 最多 10 个 epoch（粗搜），精搜阶段用全量 30 epoch

---

## 2. 搜索执行结果

### 2.1 总体统计

| 指标 | 数值 |
|------|------|
| 总 trial 数 | **需填** |
| 完成 trial | **需填** |
| 剪枝 trial | **需填**（早停节省时间） |
| 总耗时 | **需填** 分钟 |
| 实际 epoch 总量 | **需填** |

### 2.2 最优参数

| 参数 | 最优值 |
|------|--------|
| 学习率 | **需填** |
| 批次大小 | **需填** |
| 隐藏层维度 | **需填** |
| 注意力头数 | **需填** |
| 编码器层数 | **需填** |
| Dropout | **需填** |
| Weight Decay | **需填** |

### 2.3 优化历程

![Optuna Optimization History](optimization_history.png)

> 期望：前 10 个 trial 快速收敛到基线附近；后 20 个 trial 进入精修阶段。
> 显著停滞说明搜索空间已饱和。

### 2.4 参数重要性

![Optuna Param Importance](param_importance.png)

> 解读：哪个参数对 MAE 影响最大？通常 lr > hidden > n_layers > dropout。
> 如果某个参数重要性接近 0，说明它对结果不敏感，可固定到任意值。

### 2.5 并行坐标

![Optuna Parallel Coordinates](parallel_coordinates.png)

> 显示前 K 个 trial 的参数组合轨迹，便于发现"哪类参数组合总失败"。

---

## 3. 最优模型重训

### 3.1 全量训练配置

| 项目 | 数值 |
|------|------|
| Epochs | 30（Week4 默认） |
| 训练集 | `t ∈ [0, 2784)`，2784 步 |
| 验证集 | `t ∈ [2784, 3288)`，504 步 |
| 测试集 | `t ∈ [3288, 3888)`，600 步（仅评估） |
| 优化器 | AdamW |
| 早停 patience | 12（与 Week4 一致） |

### 3.2 训练曲线

| Epoch | Train Loss | Val MAE | 备注 |
|-------|-----------|---------|------|
| 1 | **需填** | **需填** | |
| 5 | **需填** | **需填** | |
| 10 | **需填** | **需填** | |
| 20 | **需填** | **需填** | |
| 30 | **需填** | **需填** | best |

### 3.3 最终模型

- 权重保存：`week4/weights/stf_optuna.pth`
- 训练历史：`week4/weights/stf_optuna.history.json`

---

## 4. 优化前后对比

### 4.1 关键指标对比

| 指标 | baseline（Week4 STF） | Optuna 优化后 | 提升 |
|------|------------------------|---------------|------|
| 验证集 MAE | **需填** | **需填** | **需填%** |
| 测试集 MAE | **需填** | **需填** | **需填%** |
| 测试集 RMSE | **需填** | **需填** | — |
| 早高峰 MAE | **需填** | **需填** | — |
| 晚高峰 MAE | **需填** | **需填** | — |
| 夜间 MAE | **需填** | **需填** | — |
| t+1 方向准确率 | **需填** | **需填** | — |

### 4.2 提升来源分析

> 哪类参数变化贡献最大？

- **学习率变化**（如 1e-3 → 5e-4）：收敛更稳定
- **隐藏维度变化**（如 64 → 128）：表达能力增强（但注意过拟合风险）
- **Dropout 变化**（如 0.1 → 0.25）：正则化加强
- **编码器层数变化**（如 2 → 3）：可能略微过拟合

---

## 5. 结论

### 5.1 核心结论

1. **优化有效性**：**待填**（MAE 提升 X%，方向准确率提升 Y%）
2. **关键参数**：**待填**（影响最大的参数）
3. **训练成本**：**待填**（总 GPU 小时数）

### 5.2 局限性

1. **搜索空间有限**：未考虑 AdamW 的 β1/β2
2. **剪枝激进**：可能误杀慢收敛 trial
3. **单次 trial 内 epoch 数受限**：粗搜只用 10 epoch，最优 trial 重训用 30 epoch，可能有微差

### 5.3 后续可改进

- 增加 L1 系数 / 标签平滑
- 用多目标优化（MAE + RMSE）
- 尝试更大学习率上限（如 1e-2）配合更激进 warmup

---

## 6. 附录

### 6.1 复现命令

```bash
# 在 EC2 上
cd /home/ubuntu/amazon
pip install -r week6.evaluation/optimization/requirements_optuna.txt

# 1. 跑 Optuna 搜索
python -m week6.evaluation.optimization.optuna_stf \
    --n-trials 30 \
    --timeout 7200 \
    --output week6.evaluation/results/optuna/

# 2. 用最优参数重训
python -m week6.evaluation.optimization.retrain_best \
    --study-result week6.evaluation/results/optuna/study_result.json \
    --epochs 30

# 3. 用新模型跑完整评估（与 baseline 对比）
python -m week6.evaluation.evaluation.evaluate \
    --model-tag optuna \
    --output week6.evaluation/results/optuna/
```

### 6.2 参数配置文件示例

```json
{
  "lr": 0.0008,
  "batch_size": 8,
  "hidden": 96,
  "n_heads": 4,
  "n_layers": 2,
  "dropout": 0.18,
  "weight_decay": 0.00005
}
```
