# Week 3 — 环境与依赖

## Python 版本
- Python 3.10+（在 g4dn 上为系统 `/usr/bin/python3`）

## 依赖包

```txt
# 深度学习核心
torch>=2.1
torch-geometric>=2.5
numpy>=1.24
pandas>=2.0

# 经典统计
statsmodels>=0.14
prophet>=1.1

# 可视化（可选）
matplotlib>=3.7
seaborn>=0.13
```

安装命令：
```bash
pip install -r scripts/requirements.txt
# 或一键安装脚本
bash scripts/install_torch.sh
```

## 硬件要求

| 模型 | 推荐 GPU 显存 | 最低 GPU 显存 | CPU |
|------|---------------|---------------|-----|
| ARIMA | N/A | N/A | 8+ 核推荐（Pool 并行）|
| Prophet | N/A | N/A | 8+ 核推荐 |
| LSTM | 2 GB | 1 GB | — |
| GRU | 2 GB | 1 GB | — |
| GCN | 5 GB | 3 GB | — |
| GAT | 7 GB | 4 GB | — |
| GRU+ST-GCN Res | 7 GB | 4 GB | — |

**已验证环境**：AWS g4dn.xlarge（Tesla T4 16GB，4 vCPU，16GB RAM）

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `BJ_DATA_DIR` | `/home/ubuntu/data` | 数据根目录 |
| `WEEK3_DIR` | `/home/ubuntu/amazon/week3` | 周3 工作目录 |
| `CUDA_OFF` | `0` | 设为 `1` 强制使用 CPU |

## 数据路径约定

```
$BJ_DATA_DIR/
├── cleaned_bj/taxi_p4_4d.npz       # 主流量数据 (3888, 2, 32, 32)
├── graph_bj/bj_hetero_graph.pt     # PyG HeteroData
├── features_bj/bj_features.parquet # 时序特征
└── grid_bj/                        # 网格元数据
```

## 一键部署

```bash
# g4dn 上首次部署
sudo apt update && sudo apt install -y python3-pip
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install --user torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip3 install --user numpy pandas statsmodels prophet

# 拉代码（如未挂载 EBS）
git clone <repo> ~/amazon

# 数据软链（假设数据在 /home/ubuntu/data）
ln -s /home/ubuntu/data ~/amazon/data

# 开始跑
cd ~/amazon/week3
bash scripts/run_all.sh
```