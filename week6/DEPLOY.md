# Week6 系统部署指南

> 城市人流时空异常检测系统工程化交付 — 部署与运行手册

---

## 目录

1. [环境准备](#1-环境准备)
2. [依赖安装](#2-依赖安装)
3. [启动服务](#3-启动服务)
4. [验证与演示](#4-验证与演示)
5. [EC2 后台运行](#5-ec2-后台运行)
6. [常见问题](#6-常见问题)

---

## 1. 环境准备

### 1.1 推荐环境

| 项目 | 推荐配置 |
|------|----------|
| 系统 | Ubuntu 22.04 LTS (AWS EC2) |
| Python | 3.10+ |
| GPU | NVIDIA GPU (CUDA 11.8+) |
| 内存 | ≥ 16GB |
| 磁盘 | ≥ 50GB |

### 1.2 环境变量

在 `~/.bashrc` 或启动脚本中设置：

```bash
# 数据路径
export BJ_FLOW_NPZ="/home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz"

# Week4 目录（模型权重路径）
export WEEK4_DIR="/home/ubuntu/amazon/week4"

# 端口配置（可选）
export API_PORT=8000
export STREAMLIT_PORT=8501
```

---

## 2. 依赖安装

### 2.1 基础环境（Python）

```bash
# 创建独立 conda 环境（推荐）
conda create -n week6 python=3.10 -y
conda activate week6

# 安装核心依赖
pip install numpy pandas scipy scikit-learn torch --index-url https://download.pytorch.org/whl/cu118

# 安装 Week6 专有依赖
pip install -r week6/requirements.txt
```

### 2.2 数据与权重验证

```bash
# 验证数据文件存在
ls -lh $BJ_FLOW_NPZ

# 验证模型权重存在
ls -lh week4/weights/
ls -lh week5/cache/       # 预计算缓存
```

---

## 3. 启动服务

### 3.1 Pipeline 本地测试（验证安装正确）

```bash
cd /path/to/amazon
python -c "
from week6.pipeline import run_quick_demo
result = run_quick_demo()
print('anomalies:', result['anomaly_mask'].sum(), '/', result['anomaly_mask'].size)
print('events:', len(result['events']))
"
```

预期输出：
```
=== Week6 Pipeline Quick Demo ===
[stat] computed stats for 48 time groups
[pred V2] loaded val predictions from cache
[Pipeline] batch(test) done in X.Xs, anomalies=X/XXXXX, events=X, alerts=X
anomalies: XXX / 614400
events: XX
```

### 3.2 启动 FastAPI 服务

```bash
# 方式 A：前台运行（调试用）
cd /path/to/amazon
python -m week6.api.main

# 方式 B：后台运行（生产用）
nohup python -m week6.api.main > week6_api.log 2>&1 &
echo "API PID: $!"

# 方式 C：使用 gunicorn（生产推荐）
pip install gunicorn
nohup gunicorn -w 2 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    week6.api.main:app \
    > week6_api.log 2>&1 &
```

### 3.3 启动 Streamlit 可视化

```bash
# 方式 A：前台运行
cd /path/to/amazon
streamlit run week6/app.py --server.port 8501

# 方式 B：后台运行
nohup streamlit run week6/app.py \
    --server.port 8501 \
    --server.headless true \
    > week6_streamlit.log 2>&1 &

echo "Streamlit PID: $!"
```

---

## 4. 验证与演示

### 4.1 API 健康检查

```bash
curl http://localhost:8000/api/health
```

预期响应：
```json
{
  "status": "ok",
  "cuda_available": true,
  "pipeline_mode": "fast",
  "cache_loaded": true,
  "version": "1.0.0"
}
```

### 4.2 接口测试

```bash
# 测试异常检测接口
curl -X POST http://localhost:8000/api/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.1, 0.2, ...1024项...]], "mode": "fast"}'

# 测试预测接口
curl "http://localhost:8000/api/forecast?time_start=3288&time_end=3311"

# 测试事件查询
curl -X POST http://localhost:8000/api/anomaly/events \
  -H "Content-Type: application/json" \
  -d '{"t_start": 3288, "t_end": 3887, "min_cells": 1}'
```

### 4.3 API 文档页面

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

直接打开 `/docs` 页面即可交互式调试所有接口。

### 4.4 Streamlit 界面

访问 http://localhost:8501

功能说明：
| 页面 | 功能 |
|------|------|
| 全局热力图 | 32×32 网格人流热力图，时间滑块切换步数，红色标注异常格点 |
| 异常事件 | 按时间/等级过滤的历史异常事件列表，支持 CSV 导出 |
| 单网格详情 | 指定网格的时序曲线、异常区间标注 |

---

## 5. EC2 后台运行

### 5.1 使用 tmux（推荐）

```bash
# 创建 tmux 会话
tmux new -s week6

# 启动 API
python -m week6.api.main

# Ctrl+B, D 分离会话（后台运行）

# 重新连接
tmux attach -t week6
```

### 5.2 使用 systemd 服务（生产环境）

创建 `/etc/systemd/system/week6-api.service`：

```ini
[Unit]
Description=Week6 Anomaly Detection API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/amazon
Environment="PYTHONPATH=/home/ubuntu/amazon"
ExecStart=/home/ubuntu/miniconda3/envs/week6/bin/python -m week6.api.main
Restart=always
StandardOutput=append:/home/ubuntu/logs/week6_api.log
StandardError=append:/home/ubuntu/logs/week6_api_err.log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable week6-api
sudo systemctl start week6-api
sudo systemctl status week6-api
```

### 5.3 使用 screen

```bash
screen -S week6_api -dm bash -c 'cd /home/ubuntu/amazon && python -m week6.api.main'
screen -S week6_st -dm bash -c 'cd /home/ubuntu/amazon && streamlit run week6/app.py --server.port 8501'

# 查看运行状态
screen -ls
```

---

## 6. 常见问题

### Q1: API 启动报 "Pipeline 未初始化"

**原因**: FastAPI 启动时 Pipeline 预热失败
**解决**: 检查数据文件路径和环境变量

```bash
export BJ_FLOW_NPZ="/home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz"
python -c "import numpy as np; d=np.load('$BJ_FLOW_NPZ'); print(d.files, d['flow'].shape)"
```

### Q2: 接口响应超过 1 秒

**原因**: structural 模式加载了 VAE/TAE 模型，或缓存未命中
**解决**: 首次请求后模型已常驻显存，后续请求应 < 500ms；确认 `week5/cache/` 下缓存文件存在

```bash
ls -lh week5/cache/*.npy | head -20
```

### Q3: Streamlit 无法连接 API

**原因**: API 未启动，或 CORS 未配置
**解决**: 确认 API 已启动于 8000 端口，且 CORS 中间件已启用（代码中已配置 `allow_origins=["*"]`）

### Q4: 内存溢出 (CUDA OOM)

**原因**: 同时加载 STF + VAE + TAE 全部模型
**解决**: 使用 fast 模式（仅 stat + pred），或在 `api/main.py` 中注释掉不需要的模型加载

### Q5: 热力图显示空白

**原因**: 实时模拟模式数据源缺失
**解决**: 运行批量模式预计算缓存后重启 Streamlit

---

## 附录：接口响应时间参考

| 接口 | 模式 | 首次响应 | 后续响应 | 说明 |
|------|------|----------|----------|------|
| /api/health | - | < 50ms | < 50ms | 健康检查，无模型调用 |
| /api/forecast | fast | < 100ms | < 50ms | 读缓存，无推理 |
| /api/anomaly/detect | fast | < 500ms | < 100ms | stat+pred 融合 |
| /api/anomaly/detect | structural | < 5s | < 500ms | 首次加载 VAE/TAE |
| /api/anomaly/events | - | < 200ms | < 100ms | 读缓存结果 |
