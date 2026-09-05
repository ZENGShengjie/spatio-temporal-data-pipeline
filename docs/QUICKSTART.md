# QUICKSTART — 5 分钟起跑指南

> 目标：从零到本地能跑（数据 + 训练 + 评估），无需 EC2。
> 适用：答辩复现 / 新成员 onboarding / 评审试跑。

---

## 0. 准备

### 系统要求

- **Python 3.10+**（项目用 3.10-3.12 测试）
- 内存 ≥ 8GB（CPU 训练）；推荐 **GPU**（任意 NVIDIA，≥4GB VRAM）
- 磁盘 ≥ 5GB（数据 + 权重 + 结果）
- OS：Linux / WSL / macOS（Windows 直接跑也行，部分脚本可能慢）

### 数据准备（重要）

本项目使用 **TaxiBJ P4** 公开数据集（微软亚研院）。

```bash
# 方式一：从原始 DeepST 仓库下载（推荐，需 ~400MB）
mkdir -p data/raw_bj/taxi_bj_gitee
cd data/raw_bj/taxi_bj_gitee
wget https://github.com/yoshitomo-matsubara/torchdistill/raw/main/datasets/cv_action/DeepST/taxibj/BJ16_M32x32_T30_InOut.h5
wget https://github.com/yoshitomo-matsubara/torchdistill/raw/main/datasets/cv_action/DeepST/taxibj/BJ_Meteorology.h5
wget https://github.com/yoshitomo-matsubara/torchdistill/raw/main/datasets/cv_action/DeepST/taxibj/BJ_Holiday.txt

# 方式二：用 Week1 的下载脚本（更完整）
cd week1/scripts
python3 01_download_taxibj.py
```

> 完整清单见 `docs/北京数据质量分析报告.md` 第 2 节。

---

## 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/ZENGShengjie/spatio-temporal-data-pipeline.git
cd spatio-temporal-data-pipeline

# 推荐：用 conda 创建独立环境
conda create -n st-pipeline python=3.10 -y
conda activate st-pipeline

# 安装核心依赖
pip install -r requirements.txt          # 若无此文件，按下面列安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn optuna shap streamlit plotly
pip install fastapi uvicorn orjson python-multipart
pip install torch-geometric              # 用于 GCN/STGCN
```

如果 `requirements.txt` 不存在，按上面前四行命令逐个装即可。

---

## 2. 跑 Week3 时序基线（30 秒看结果）

```bash
cd week3
python3 run_week3.py --models gru --target taxi_flow_total --tag demo
```

**预期输出**：
- `week3/results/gru_demo_pred.npy`（预测值）
- `week3/results/gru_demo_gt.npy`（真实值）
- `week3/results/summary_demo.md`（指标汇总）
- 控制台显示 MAE / RMSE / MAPE / Corr

**解读**：GRU 在 P4 测试集上 MAE ≈ 158.12，Corr = 0.9452（多步滚动预测基线）。

---

## 3. 跑 Week4 时空模型（推荐 STF，5-10 分钟）

```bash
cd week4
python3 run_week4.py --models stf --target taxi_flow_total --tag demo
```

**预期输出**：
- `week4/results/stf_demo_pred.npy`
- `week4/results/stf_demo_gt.npy`
- `week4/results/summary_taxi_flow_total_demo.md`
- `week4/weights/stf_taxi_flow_total_demo.pth`（训练好的权重）

**解读**：STF 测试集 MAE ≈ 327.19，Corr = 0.8043（仅 222K 参数，1 个 epoch 即收敛）。

进阶——三个时空模型一起跑 + 消融：

```bash
python3 run_week4.py --models stgcn agformer stf agformer_static stf_loc_only \
    --target taxi_flow_total --tag demo --ablation
```

---

## 4. 跑 Week5 异常检测融合（3-5 分钟）

**前置**：先跑完 Week4（产生 STF 预测值与权重）。

```bash
cd week5
python3 run_v3_full_eval.py
```

**预期输出**：
- `week5/report/v3_full_eval_<timestamp>.json`（所有方法 + 融合结果）
- `week5/report/v3_final_report.md`（人类可读报告）

**解读**：融合 V3（统计 + STF真实推理 + VAE）在 V3 注入测试集上 **F1 = 0.9165**，单路统计法约 0.791。
**实验条件**：V3 注入测试集，4% 注入率，混合突增/突降/持续模式（sustained_pct=20%）。

---

## 5. 启动 Week6 API 服务（可选）

```bash
# 终端 1：FastAPI
cd week6
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 终端 2：Streamlit（可选）
cd week6
API_BASE=http://localhost:8000 streamlit run app.py
```

打开浏览器：
- API 文档：http://localhost:8000/docs
- Streamlit：http://localhost:8501

**关键端点**：
- `GET  /api/health` 健康检查
- `POST /api/anomaly/detect` 异常检测（最常用）
- `GET  /api/forecast` 流量预测

---

## 6. Week7 评估套件（可选，耗时）

```bash
# Optuna 超参搜索（30 trial，约 30 分钟）
cd week7/optimization
python3 optuna_stf.py --n-trials 30 --timeout 7200

# SHAP 可解释性（约 10 分钟）
cd week7/interpretability
python3 shap_analysis.py
```

---

## 一键复现（高级）

如果你想一键跑完所有周：

```bash
# 顶层一键脚本（项目根目录）
bash reproduce_all.sh        # 若有；若无，按 2-5 步顺序跑
```

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: week4.models.stf_model` | 确认在 `week4/` 目录下，且 `python3` 是项目专用环境 |
| CUDA OOM | 减小 `batch_size`；或在 `week4/config.py` 改 `BATCH_SIZE = 4` |
| 数据文件找不到 | 检查 `data/raw_bj/` 路径，或修改 `week*/config.py` 的 `DATA_ROOT` |
| Streamlit 卡顿 | `API_BASE` 配错，改回 localhost:8000 |
| `pip install torch-geometric` 失败 | 先按 https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html 选对应 CUDA 版本 |

---

## 下一步

- 完整数据探索：`docs/北京数据质量分析报告.md`
- 完整模型评估：`week4/report/WEEK4_REPORT.md`
- 完整异常检测：`week5/README.md`
- 完整项目交付：`docs/技术报告.md`（10K+ 字）
- 答辩 PPT：`docs/ppt.html`（浏览器打开）