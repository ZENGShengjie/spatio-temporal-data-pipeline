# Week4 最优模型权重

> 训练完成后自动保存在本目录。命名格式：`{model}_{target}_{tag}.pth`

## 文件清单

| 文件名 | 模型 | 描述 |
|--------|------|------|
| `stf_taxi_flow_total_v4fix.pth` | STF | 轻量时空解耦，最优 epoch=1 |
| `stf_loc_only_taxi_flow_total_v4fix.pth` | STF (消融-纯时序) | 去掉 Env+CrossAttn |
| `agformer_taxi_flow_total_v4fix.pth` | AGFormer | 自适应图 Transformer，2.26M 参数 |
| `agformer_static_taxi_flow_total_v4fix.pth` | AGFormer (消融-静态图) | 冻结自适应邻接矩阵 |
| `stgcn_taxi_flow_total_v4fix.pth` | STGCN | 时空图卷积网络 |

## 加载示例

```python
import sys
sys.path.insert(0, "week4")
from inference import load_model, SpacetimePredictor

# 加载最优 STF 模型
trainer, meta = load_model("stf", target="taxi_flow_total", tag="v4fix")
print(f"最优 epoch: {meta['best_epoch']}, 参数量: {meta['n_params']:,}")

# 单批次推理（归一化后输入）
import numpy as np
x_batch = np.random.randn(1024, 10, 48).astype(np.float32)  # (N, F, T)
pred = trainer._predict_batch(trainer.model,
                              torch.from_numpy(x_batch).unsqueeze(0),
                              N=1024, K_time=8)
# pred.shape = (1, 1024, 48) → (N, H)
```

## 训练命令

```bash
# 完整训练（含消融）
python week4/run_week4.py \
    --models stgcn agformer stf agformer_static stf_loc_only \
    --target taxi_flow_total --tag v4fix --ablation

# 仅消融对比
python week4/run_week4.py \
    --models stf stf_loc_only \
    --target taxi_flow_total --tag v4fix

python week4/run_week4.py \
    --models agformer agformer_static \
    --target taxi_flow_total --tag v4fix
```

## Checkpoint 格式

每个 `.pth` 文件是一个字典：

```
{
    "model_state_dict": {...},   # torch state_dict
    "model_name":      "stf",
    "target":          "taxi_flow_total",
    "tag":             "v4fix",
    "best_epoch":      1,
    "n_params":        222576
}
```

## 复现测试集指标

权重加载后，使用 `analysis_v4fix.py` 重新生成报告：

```bash
python week4/analysis_v4fix.py
```

该脚本会读取 `week4/results/*_v4fix_pred.npy` 和 `_gt.npy`，重新计算 MAE/RMSE/MAPE/Corr 并写入 `week4/results/v4fix/WEEK4_FINAL_REPORT.md`。
