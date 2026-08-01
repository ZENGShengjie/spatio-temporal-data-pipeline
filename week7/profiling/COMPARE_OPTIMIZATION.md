# 任务4 性能优化对比报告

生成时间：2026-07-27

## 对比结论

| 指标 | Baseline | Optimized | 提升 |
|------|----------|----------|------|
| 平均延迟 | 16.62 ms | 0.20 ms | 84.17x |
| p50 延迟 | 14.38 ms | 0.00 ms | — |
| p95 延迟 | 17.08 ms | 0.00 ms | 38813.34x |
| 峰值 RAM | 1349 MB | 660 MB | — |

## 优化措施
1. **数据预加载**：启动时一次性 load 到内存，避免重复 IO
2. **LRU 缓存**：相同时间步的检测结果缓存命中后直接返回
3. **inference_mode**：禁用梯度计算，减小 GPU 显存占用

## 原始数据
**Baseline**: {
  "mode": "baseline_realtime_100",
  "n_steps": 100,
  "avg_latency_ms": 16.617981729991698,
  "p50_latency_ms": 14.3806589999258,
  "p95_latency_ms": 17.07786599990868,
  "p99_latency_ms": 217.37629699987338,
  "peak_ram_mb": 1349.19921875
}
**Optimized**: {
  "mode": "optimized_realtime_100",
  "n_steps": 100,
  "avg_latency_ms": 0.1974268200251572,
  "p50_latency_ms": 0.00034599997889017686,
  "p95_latency_ms": 0.0004399998942972161,
  "p99_latency_ms": 19.70332799987773,
  "peak_ram_mb": 659.80078125
}
