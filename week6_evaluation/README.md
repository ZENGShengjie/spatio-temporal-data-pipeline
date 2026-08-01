# Week6 深度评估 — 4 任务交付包

本目录是 Week6「性能评估 / 超参优化 / 可解释性 / 系统优化」四大任务的代码与文档交付包。

## 目录结构

```
week6_evaluation/
├── README.md                          ← 当前文件
├── evaluation/                        ← 任务1：性能评估
│   ├── evaluate.py                    ← 评估入口（可复用）
│   ├── metrics.py                     ← 指标计算（分层）
│   ├── profile_runner.py              ← 性能打点器
│   └── run_baseline.sh                ← EC2 启动脚本
├── optimization/                      ← 任务2：Optuna 超参优化
│   ├── optuna_stf.py                  ← 搜索脚本
│   └── retrain_best.py                ← 用最优参数重训
├── interpretability/                  ← 任务3：可解释性
│   ├── attention_vis.py               ← 注意力可视化
│   ├── shap_analysis.py               ← SHAP 特征重要性
│   └── anomaly_attribution.py         ← 异常归因（节假日/天气）
├── profiling/                         ← 任务4：性能优化
│   ├── profile_pipeline.py            ← PyTorch Profiler
│   ├── optimized_pipeline.py          ← 优化后 Pipeline（含 patch）
│   └── api_optimization.py            ← API 层优化
├── patches/                           ← 可直接应用的代码补丁
│   ├── pipeline_inference_mode.py     ← torch.inference_mode()
│   ├── pipeline_lru_cache.py          ← LRU 缓存
│   └── streamlit_heatmap_downsampling.py  ← Plotly 降采样
└── results/
    ├── baseline/                      ← 任务1 基线结果
    ├── optuna/                        ← 任务2 优化结果
    ├── interpretability/              ← 任务3 可解释性图表
    └── profiling/                     ← 任务4 性能对比
```

## 执行前提

所有脚本默认在 **EC2 服务器** 上运行（拥有 GPU、数据、训练好的模型）。

```bash
ssh ubuntu@<EC2_PUBLIC_IP>
cd /home/ubuntu/amazon
```

数据/模型位置：
- 数据集：`week5/cache/*.npy`（stat_scores、pred_scores、vae_scores、tae_scores）
- 模型权重：`week4/weights/stf_*.pth` / `agformer_*.pth`
- 原始数据：`week5/data/TaxiBJ-P4.h5`

## 任务执行顺序

| 任务 | 脚本 | 触发命令 |
|------|------|---------|
| 1. 性能评估 | `evaluation/evaluate.py` | `python -m week6_evaluation.evaluation.evaluate --split test --mode fast --output results/baseline/` |
| 2. 超参优化 | `optimization/optuna_stf.py` | `python -m week6_evaluation.optimization.optuna_stf --n_trials 30 --timeout 7200` |
| 3. 可解释性 | `interpretability/{attention_vis,shap_analysis,anomaly_attribution}.py` | 三个脚本独立运行 |
| 4. 性能优化 | `profiling/profile_pipeline.py` + 应用 `patches/` | 见各文件 |

## 重要约束

1. **数据泄露红线**：所有评估、归因、可视化只针对 `t ∈ [VAL_END, TEST_END)` 测试集区间
2. **可复用**：任务1 的 `evaluate.py` 是任务2、3、4 的对标基准，不要重写评估代码
3. **EC2 跑**：所有重训练、Profile 大型任务在 EC2 上运行，本地开发仅做语法/接口验证
