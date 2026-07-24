#!/bin/bash
# ========================================
#   城市人流时空异常检测系统 - 启动脚本
# ========================================

set -e

echo "========================================"
echo "  城市人流时空异常检测系统"
echo "========================================"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 启动 API（后台）
echo "[1/2] 启动 API 服务（端口 8000）..."
nohup python -m week6.api.main --host 0.0.0.0 --port 8000 > api.log 2>&1 &
API_PID=$!
echo "  API PID: $API_PID"

sleep 3

# 启动 Streamlit（后台）
echo "[2/2] 启动可视化界面（端口 8501）..."
nohup streamlit run week6/app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
ST_PID=$!
echo "  Streamlit PID: $ST_PID"

echo ""
echo "========================================"
echo "  启动完成！"
echo "  API 文档:   http://localhost:8000/docs"
echo "  可视化界面: http://localhost:8501"
echo ""
echo "  停止服务："
echo "  kill $API_PID $ST_PID"
echo "========================================"
