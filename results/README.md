# Results

Experiment outputs and experiment reports.

## Structure

```
results/
├── README.md                          # 本文件
├── summary_taxi_flow_total_v4fix.md   # 6 模型性能汇总表 + 分时段 MAE
├── top_offenders_taxi_flow_total_v4fix.md  # STF Top-20 误差网格分析
└── v4fix/                            # Week4 v4fix 完整实验报告
    ├── WEEK4_FINAL_REPORT.md         # 周交付物：详细实验 + 完整训练日志 + 全量图表
    └── figures/                       # 5 张关键配图
        ├── fig1_dl_comparison.png    # 深度学习模型性能对比柱状图
        ├── fig2_segment_mae_heatmap.png   # 分时段 MAE 热力图
        ├── fig3_segment_corr_heatmap.png  # 分时段 Corr 热力图
        ├── fig4_timeseries_commercial.png # 商业区网格时序对比
        ├── fig4_timeseries_residential.png # 居民区网格时序对比
        └── fig5_stf_mae_heatmap.png   # STF 全网格空间误差热力图
```

## Quick Summary

| 排名 | 模型 | MAE | Corr | 参数量 |
|------|------|-----|------|--------|
| 🥇 | GRU (Week3) | 158.12 | 0.9452 | — |
| 🥈 | STF | 327.19 | 0.8043 | 222K |
| 🥉 | AGFormer | 386.91 | 0.6990 | 2.26M |
| 4 | STGCN | 429.18 | 0.6502 | 199K |

> 注：GRU 为 Week3 城市级序列预测结果，与 Week4 逐网格任务难度不同，仅供趋势参考。
> Prophet (MAE=93.66, Corr=0.928) 为 1-shot 基线，口径不同不参与同表对比。

## How to Reproduce

```bash
# 重新生成预测结果（需 EC2 g4dn.xlarge）
cd week4
python run_week4.py --models stgcn agformer stf --target taxi_flow_total --tag v4fix

# 重新生成分析报告与图表
python analysis_v4fix.py
```

## Note on Data Files

`*.npy` 预测文件（每个 ~113MB）未纳入版本控制，可通过上述命令重新生成。
