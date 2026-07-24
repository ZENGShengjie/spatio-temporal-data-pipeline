# 城市人流时空异常检测系统

**Spatio-Temporal Urban Flow Anomaly Detection System**

基于北京出租车 GPS 数据，融合统计检测 + 预测残差 + 深度学习 (TransAE / VAE / Transformer-AE) 的城市级人流异常检测全栈系统。

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io/)

---

## 核心效果预览

> 进入系统后，你将看到以下 5 个功能页面：

| 页面 | 说明 |
|------|------|
| 📊 实时热力图 | 32×32 北京城区热力图，红色点标注异常格点，支持时间滑块切换 |
| 🎬 24h 预测动画 | 未来 24 小时人流预测滚动动画 |
| 📋 异常事件 | 历史异常事件查询，支持三级预警过滤、CSV 导出 |
| 🔍 单网格详情 | 任意网格编号的时序曲线 + 异常标记 |
| 🗺️ 地理地图 | 经纬度坐标散点图，展示异常地理分布 |

---

## 核心特性

- **支持快速 / 结构增强双检测模式**：fast 模式响应 < 500ms，structural 模式融合 TransAE + VAE
- **离线卫星地理底图**：预拼接高德瓦片，零网络依赖，演示永不白屏
- **FastAPI 全量 REST 接口**：覆盖预测、异常检测、事件查询、健康检查，自带 `/docs` 在线交互文档
- **三级分级预警**：一般（均分 ≥ 0.65）/ 重要（均分 ≥ 0.75）/ 紧急（均分 ≥ 0.85），支持弹窗告警
- **时空事件聚合**：三类事件（时空聚合 / 瞬时连片 / 单点高分），默认展示聚合事件，可展开全部零散异常
- **全链路真实数据驱动**：TaxiBJ P4 时段（2016年7-8月），训练/验证/测试集完整划分

---

## 项目架构

```
数据层                    模型推理层              API 服务层            前端可视化层
─────────────────────────────────────────────────────────────────────────────
TaxiBJ NPZ            statistical.py           FastAPI               Streamlit
(cleaned_bj/taxi_p4)  prediction.py            /api/health           app.py
                      fusion_v3.py             /api/forecast
                      transae.py               /api/anomaly/detect
                      vae.py                   /api/anomaly/events
                      trans_ae.py
```

数据流向：`taxi_p4_4d.npz` → `pipeline.py`（数据加载 + 特征提取）→ 三路异常检测器 → 融合打分 → API → 前端热力图

---

## 目录结构

```
amazon/
├── week6/                          # ★ 本项目主目录
│   ├── __init__.py
│   ├── config.py                   # 配置（复用 week5 config）
│   ├── pipeline.py                 # 核心 Pipeline（批量 + 实时）
│   ├── app.py                      # Streamlit 可视化界面
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI 服务入口
│   │   └── schemas.py              # Pydantic 数据模型
│   ├── weights/                    # 模型权重目录（需自行放置，见下方说明）
│   ├── requirements.txt             # 依赖清单
│   └── requirements_api.txt         # API 专用依赖
├── week5/                          # 算法层（异常检测核心逻辑，100% 复用）
│   ├── anomaly/
│   │   ├── statistical.py          # 统计异常检测器
│   │   ├── prediction.py           # 预测残差检测器
│   │   ├── fusion_v3.py            # 多源融合 + 事件聚合
│   │   ├── vae.py                  # VAE 深度检测器
│   │   ├── trans_ae.py              # Transformer-AE 检测器
│   │   └── transformer_ae.py        # TransAE 主模型
│   ├── data_loader.py              # 数据加载器
│   └── config.py                   # 配置（数据路径、阈值等）
├── data/                           # 数据集目录（需自行放置）
│   └── cleaned_bj/
│       └── taxi_p4_4d.npz          # ★ 核心数据集文件
└── docs/
    ├── 演示视频脚本.md
    └── README.md
```

---

## 🔰 小白向：从零到一完整运行教程

### 第一步：环境准备

**推荐 Python 3.10**，不要用 3.12（依赖报错率高）。

```bash
# 方法一：conda（推荐）
conda create -n taxi-anomaly python=3.10
conda activate taxi-anomaly

# 方法二：venv
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 第二步：放置数据和模型权重

> ⚠️ 数据集和模型权重未纳入 Git 仓库（文件过大），请手动放置。

| 文件 | 放置路径 | 说明 |
|------|---------|------|
| `taxi_p4_4d.npz` | `data/cleaned_bj/taxi_p4_4d.npz` | 北京出租车流量数据（需联系项目方获取） |
| Week5 模型权重 | `week5/cache/` | 预计算缓存（自动生成，首次运行 pipeline 后产生） |

目录创建命令：

```bash
# Windows PowerShell
mkdir -p data/cleaned_bj
mkdir -p week6/weights

# Linux / Mac
mkdir -p data/cleaned_bj
mkdir -p week6/weights
```

### 第三步：安装依赖

```bash
pip install -r week6/requirements.txt
```

**如安装 PyTorch 报错**，先单独安装：

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 第四步：启动后端 API 服务

```bash
# ★ 启动前确保已进入项目根目录
cd amazon

# Windows 前台运行
python -m week6.api.main --host 0.0.0.0 --port 8000

# Linux / 服务器后台常驻运行
nohup python -m week6.api.main --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

✅ **成功标志**：浏览器打开 `http://localhost:8000/docs`，看到 API 在线文档页面（Swagger UI）。

### 第五步：启动前端可视化界面

**另开一个终端窗口**（API 保持运行）：

```bash
# Windows
streamlit run week6/app.py --server.port 8501

# Linux / 服务器
streamlit run week6/app.py --server.port 8501 --server.headless true
```

✅ **成功标志**：浏览器自动打开 `http://localhost:8501`，侧边栏显示「**API 在线 (ok)**」，5 个功能 Tab 均可见。

### 第六步：一键启动（可选）

不想记命令？使用一键脚本（见下方「一键启动」小节）。

### 功能验证清单

完成后按以下顺序验证系统是否正常：

- [ ] `http://localhost:8000/docs` 能打开 API 文档
- [ ] Streamlit 侧边栏显示「API 在线 (ok)」，无红字警告
- [ ] Tab 1「实时热力图」拖动时间滑块，热力图随时间变化
- [ ] Tab 3「异常事件」点击「查询事件」，返回事件列表
- [ ] Tab 5「地理地图」点击「查询异常」，散点图正常渲染

---

## API 接口文档

所有接口均在 `http://localhost:8000/docs` 提供在线交互式调试。

### 健康检查

```
GET /api/health
```

返回示例：
```json
{
  "status": "ok",
  "cuda_available": true,
  "pipeline_mode": "fast",
  "cache_loaded": true,
  "period": "P4",
  "t_min": 3288,
  "t_max": 3888,
  "version": "1.1.0"
}
```

### 人流预测

```
POST /api/forecast
Body: { "time_start": 3288, "time_end": 3311 }
```

### 异常检测（核心接口）

```
POST /api/anomaly/detect
Body: {
  "t": 3500,          // 时间步（全局索引），可选
  "mode": "fast",     // fast=快速模式，structural=深度融合模式
  "threshold": null   // 阈值，可选
}
```

返回：热力图矩阵 + 异常标记 + 1024 个网格详情 + 三级预警等级

### 异常事件查询

```
POST /api/anomaly/events
Body: {
  "t_start": 3288,
  "t_end": 3888,
  "min_cells": 9,
  "include_marginal": false  // 是否包含零散异常
}
```

返回：事件列表（含 ID、起止时间、持续步数、网格数、预警等级）

---

## 异常检测设计说明

### 三级预警规则

| 等级 | 名称 | 均分阈值 | 说明 |
|------|------|---------|------|
| 1 级 | 一般 | ≥ 0.65 | 单格异常但时空不连续 |
| 2 级 | 重要 | ≥ 0.75 | 3×3 连片或持续 2 步以上 |
| 3 级 | 紧急 | ≥ 0.85 | 5×5 持续 3 步以上 |

### 事件类型说明

| 类型 | 说明 | 何时入库 |
|------|------|---------|
| `spatial_sustained`（时空聚合） | 时空连续的正式异常事件 | 核心业务事件，默认展示 |
| `patch_marginal`（瞬时连片） | ≥12 格单步连片 | 兜底收录，需开启选项查看 |
| `point_single`（单点高分） | ≥0.85 分孤立点 | 兜底收录，需开启选项查看 |

**为什么有"零散异常"？** 热力图全量标记所有异常格点（红点），但事件列表默认只展示时空连续的聚合事件，以避免噪声。开启「包含零散/瞬时异常」后，事件列表与热力图红点一一对应。

### 双模式说明

| 模式 | 使用的检测器 | 首次响应 | 后续响应 | 适用场景 |
|------|------------|---------|---------|---------|
| `fast`（默认） | stat 0.9 + pred 0.1 | < 500ms | < 100ms | 实时监控、演示 |
| `structural` | stat + pred + VAE + TAE 全量融合 | < 5s | < 500ms | 深度分析、报告生成 |

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 数据处理 | NumPy, SciPy, Pandas, scikit-learn |
| 深度学习 | PyTorch 2.0+ |
| 模型推理 | TransAE, VAE, Transformer-AE |
| 后端服务 | FastAPI 0.110+, Uvicorn, Pydantic 2.5+ |
| 前端界面 | Streamlit 1.35+, Plotly, pydeck |
| 环境管理 | Conda / venv |

---

## 一键启动脚本

### Windows（双击运行）

新建文件 `start.bat`，内容如下：

```bat
@echo off
chcp 65001 >nul
title 城市人流异常检测系统

echo ========================================
echo   城市人流时空异常检测系统
echo ========================================
echo.

:: 启动 API（后台）
echo [1/2] 启动 API 服务（端口 8000）...
start "API" cmd /k "python -m week6.api.main --host 0.0.0.0 --port 8000"

:: 等待 3 秒
timeout /t 3 /nobreak >nul

:: 启动 Streamlit
echo [2/2] 启动可视化界面（端口 8501）...
start "Streamlit" cmd /k "streamlit run week6/app.py --server.port 8501"

echo.
echo ========================================
echo   启动完成！
echo   API 文档:  http://localhost:8000/docs
echo   可视化界面: http://localhost:8501
echo ========================================
pause
```

### Linux / 服务器

新建文件 `start.sh`，内容如下：

```bash
#!/bin/bash
echo "========================================"
echo "  城市人流时空异常检测系统"
echo "========================================"

# 启动 API（后台）
echo "[1/2] 启动 API 服务（端口 8000）..."
nohup python -m week6.api.main --host 0.0.0.0 --port 8000 > api.log 2>&1 &

sleep 3

# 启动 Streamlit（后台）
echo "[2/2] 启动可视化界面（端口 8501）..."
nohup streamlit run week6/app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &

echo ""
echo "========================================"
echo "  启动完成！"
echo "  API 文档:   http://localhost:8000/docs"
echo "  可视化界面: http://localhost:8501"
echo "========================================"
```

```bash
chmod +x start.sh
./start.sh
```

---

## 关键技术决策与优化亮点

### 1. 地理地图稳定性方案

- **问题**：原始方案依赖境外在线地图瓦片，网络环境受限极易白屏，演示风险高
- **方案**：设计离线卫星底图方案，预拼接高德瓦片生成北京六环全域底图，base64 本地缓存，零外部网络依赖
- **效果**：彻底解决白屏问题，同时保留真实地理参照，支持人流格点按需显隐，演示稳定性 100%

### 2. 异常事件一致性方案

- **问题**：热力图全量标记异常格点，事件列表仅聚合时空连续异常，出现"热力图见红、事件查不到"的体感脱节
- **方案**：保留主聚合逻辑的同时，新增瞬时连片、单点高分两类兜底规则，前端增加「包含零散异常」筛选开关
- **效果**：默认展示核心聚合事件（降噪），展开可对应全部异常格点（完整）

### 3. 双模式推理架构

- **问题**：深度学习模型加载耗时长，无法满足实时监控场景 < 1s 响应要求
- **方案**：fast 模式（stat + pred 融合，无 GPU 推理）与 structural 模式（全部模型懒加载）并存，API 层按需切换
- **效果**：实时监控场景 100~500ms 出结果，深度分析场景一次性加载全部模型

---

## 常见问题 FAQ

### Q1：打开地理地图是空白的？
**A**：当前为离线底图方案，无需外网。若空白请检查底图缓存是否生成（首次加载需要 5~10 秒），刷新页面重试。

### Q2：侧边栏显示「API 离线」？
**A**：请确认 8000 端口 API 服务已正常启动。检查方法：
```bash
curl http://localhost:8000/api/health
```
若返回 JSON 说明 API 正常；若超时说明未启动，请回到第四步重新启动 API。

### Q3：异常事件列表为空？
**A**：
1. 默认查询区间为整个测试集（t=3288~3888），若时间段内无异常则为空，这是正常的
2. 可尝试缩小 `min_cells` 到 1，或开启「包含零散/瞬时异常」
3. 确认 API 侧边栏显示「API 在线」

### Q4：启动报错 `ModuleNotFoundError: No module named 'week6'`？
**A**：当前工作目录不是项目根目录。请先 `cd` 到 `amazon` 目录后再运行启动命令。

### Q5：启动报错提示找不到数据文件？
**A**：请确认 `data/cleaned_bj/taxi_p4_4d.npz` 文件已放置到正确路径。可运行以下命令验证：
```bash
python -c "import numpy as np; d=np.load('data/cleaned_bj/taxi_p4_4d.npz'); print('files:', d.files, 'flow shape:', d['flow'].shape)"
```

### Q6：接口响应超过 5 秒？
**A**：这是 structural 模式首次加载 VAE/TAE 模型的正常耗时。首次请求后模型常驻显存，后续请求应 < 500ms。

### Q7：PyTorch 报 CUDA 内存不足（OOM）？
**A**：同时加载了太多模型。使用 fast 模式（`mode="fast"`）可避免：
```bash
# 或在 API 请求中指定
curl -X POST http://localhost:8000/api/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{"t": 3500, "mode": "fast"}'
```

---

## 版本更新日志

### v1.1.0（本周交付）

- ✅ 全链路真实 TaxiBJ 数据接入，替换原有模拟数据，API 返回真实推理结果
- ✅ 地理地图模块重构：采用离线高德卫星底图方案，彻底解决在线瓦片白屏问题
- ✅ 异常事件聚合优化：新增瞬时连片、单点高分两类兜底规则，解决热力图与事件列表体感脱节
- ✅ 完善三级预警分级逻辑（一般/重要/紧急），优化连片判定规则
- ✅ 修复坐标轴比例拉伸、异常点遮挡等可视化问题
- ✅ API 版本升至 1.1.0，支持 5 个历史时段（BJ13~BJ16, P4）切换

---

## 接口调用示例（Python）

```python
import requests

API = "http://localhost:8000"

# 1. 健康检查
health = requests.get(f"{API}/api/health").json()
print("API 状态:", health["status"])

# 2. 异常检测
det = requests.post(f"{API}/api/anomaly/detect", json={
    "t": 3500,
    "mode": "fast"
}).json()
print(f"异常率: {det['anomaly_rate']:.2%}，预警: {det['warning_name']}")

# 3. 查询事件
events = requests.post(f"{API}/api/anomaly/events", json={
    "t_start": 3288,
    "t_end": 3888,
    "min_cells": 9,
    "include_marginal": False
}).json()
print(f"共 {events['total']} 个异常事件")
```

---

## 许可证

MIT License

---

*有任何问题欢迎提交 Issue，或联系项目维护者。*
