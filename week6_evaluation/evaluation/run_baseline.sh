#!/bin/bash
# Week6 任务1：基线评估启动脚本（在 EC2 上运行）
#
# 前置：
#   - 已 ssh 进入 EC2
#   - 已激活 miniforge/conda 环境
#   - week5/cache 目录有 stat_scores_test_v2.npy / pred_scores_test_v2.npy 等缓存
#   - week4/weights 下有 stf 模型权重（Pipeline 启动会用到）
#
# 执行：
#   cd /home/ubuntu/amazon
#   bash week6_evaluation/evaluation/run_baseline.sh

set -e

# 自动获取 EC2 主机（用于 API 端到端 latency test）
API_HOST="${API_HOST:-http://localhost:8000}"
OUTPUT_DIR="${OUTPUT_DIR:-week6_evaluation/results/baseline}"

echo "========================================"
echo "Week6 任务1：基线评估"
echo "API_HOST: $API_HOST"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "========================================"

# 1. 确认 API 服务在跑（用于 API profile）
if curl -s -f "$API_HOST/api/health" > /dev/null 2>&1; then
    echo "[✓] API 在线: $API_HOST"
    echo "[→] 评估完会同时测 API 端到端延迟"
    API_HOST_OPT="--api-host ${API_HOST#http://}"
else
    echo "[!] API 未启动或不可达"
    echo "[→] 跳过 API 端到端 latency 测试，仅跑 Pipeline 内部"
    API_HOST_OPT=""
fi

# 2. 跑评估
python -m week6_evaluation.evaluation.evaluate \
    --model-tag baseline \
    --output "$OUTPUT_DIR" \
    $API_HOST_OPT

echo ""
echo "========================================"
echo "完成。结果在: $OUTPUT_DIR"
echo "  - metrics.json       指标详情"
echo "  - profile.json       性能 profile"
echo "  - summary.md         人类可读摘要"
echo "========================================"
