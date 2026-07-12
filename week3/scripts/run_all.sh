#!/bin/bash
# 在 g4dn 上同时跑 ARIMA + Prophet + 4 个深度学习
# 数据: /home/ubuntu/data — 全部 BJ 数据
# 监控每模型进度
set -e
cd /home/ubuntu/amazon/week3

echo "=== Week3 启动 ==="
date
nvidia-smi | head -10
nproc
free -h
echo "=== Data check ==="
ls -lh /home/ubuntu/data/cleaned_bj/ /home/ubuntu/data/graph_bj/ /home/ubuntu/data/features_bj/

mkdir -p logs results data

# 启动 ARIMA (后台)
echo ""
echo "=== [1/6] ARIMA ==="
nohup /usr/bin/python3 -u run_week3.py --models arima --target taxi_flow_total \
    > logs/arima.log 2>&1 &
ARIMA_PID=$!
echo "  PID=$ARIMA_PID"

# 启动 Prophet (后台)
echo ""
echo "=== [2/6] Prophet ==="
nohup /usr/bin/python3 -u run_week3.py --models prophet --target taxi_flow_total \
    > logs/prophet.log 2>&1 &
PROPHET_PID=$!
echo "  PID=$PROPHET_PID"

# 启动 LSTM (后台, GPU)
echo ""
echo "=== [3/6] LSTM ==="
nohup /usr/bin/python3 -u run_week3.py --models lstm --target taxi_flow_total \
    > logs/lstm.log 2>&1 &
LSTM_PID=$!
echo "  PID=$LSTM_PID"

# 等 LSTM 启动占用好显存后再启动其余
sleep 30

echo "=== [4/6] GRU ==="
nohup /usr/bin/python3 -u run_week3.py --models gru --target taxi_flow_total \
    > logs/gru.log 2>&1 &
GRU_PID=$!
echo "  PID=$GRU_PID"

sleep 30

echo "=== [5/6] GCN ==="
nohup /usr/bin/python3 -u run_week3.py --models gcn --target taxi_flow_total \
    > logs/gcn.log 2>&1 &
GCN_PID=$!
echo "  PID=$GCN_PID"

sleep 30

echo "=== [6/6] GAT ==="
nohup /usr/bin/python3 -u run_week3.py --models gat --target taxi_flow_total \
    > logs/gat.log 2>&1 &
GAT_PID=$!
echo "  PID=$GAT_PID"

echo ""
echo "All jobs launched."
echo "  ARIMA_PID=$ARIMA_PID  PROPHET_PID=$PROPHET_PID"
echo "  LSTM_PID=$LSTM_PID    GRU_PID=$GRU_PID"
echo "  GCN_PID=$GCN_PID      GAT_PID=$GAT_PID"
echo ""
echo "Check: tail -F logs/*.log"
date
