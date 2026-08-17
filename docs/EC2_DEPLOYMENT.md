# EC2 部署与 IP 入口

> **目的**：本仓库所有 .md 文档中的 EC2 公网 IP 均应引用本文件，不在文档中硬编码。
> **查询当前 IP**：`ssh ubuntu@<IP> 'curl -s ifconfig.me'`

---

## 1. 当前活跃实例

| 字段 | 值 |
|------|-----|
| **公网 IP** | `<EC2_PUBLIC_IP>` （演示前查实际值） |
| **用户名** | `ubuntu` |
| **SSH Key** | `~/.ssh/aws-spatio-key.pem` |
| **实例类型** | `g4dn.xlarge` (T4 GPU) |
| **OS** | Ubuntu 22.04 |
| **Python** | 3.12 (`.local/lib/python3.12`) |

## 2. 端口与服务

| 端口 | 服务 | 用途 |
|------|------|------|
| **22** | SSH | 远程登录 |
| **8000** | FastAPI | API 服务（`http://<EC2_PUBLIC_IP>:8000/docs`） |
| **8501** | Streamlit | 可视化界面（`http://<EC2_PUBLIC_IP>:8501`） |

## 3. 服务地址占位符

| 占位符 | 实际值 |
|--------|--------|
| `<EC2_PUBLIC_IP>` | 当前 EC2 公网 IP（每次重启可能变） |
| `http://<EC2_PUBLIC_IP>:8000` | API 根地址 |
| `http://<EC2_PUBLIC_IP>:8000/docs` | API 自动文档 |
| `http://<EC2_PUBLIC_IP>:8000/api/health` | 健康检查 |
| `http://<EC2_PUBLIC_IP>:8501` | Streamlit Web |

## 4. 快速 SSH 登录

```bash
# 1. 加载 EC2 配置（PowerShell 一次性）
. ./aws/EC2_IP.ps1

# 2. SSH 登录
ssh -i ~/.ssh/aws-spatio-key.pem ubuntu@<EC2_PUBLIC_IP>

# 3. 一键重启服务
bash ~/amazon/_restart_services.sh
```

## 5. 状态监控

```bash
# API 健康
curl -s http://<EC2_PUBLIC_IP>:8000/api/health | python3 -m json.tool

# Streamlit 健康
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://<EC2_PUBLIC_IP>:8501/_stcore/health

# 进程
ssh ubuntu@<EC2_PUBLIC_IP> "ps aux | grep -E 'uvicorn|streamlit' | grep -v grep"
```

## 6. EC2 项目目录

| 路径 | 内容 |
|------|------|
| `/home/ubuntu/amazon/` | **主工作区**（week1-7 完整） |
| `/home/ubuntu/amazon_repo/` | Git repo 镜像（早期工作区） |
| `/home/ubuntu/spatio-temporal-pipeline/` | 最早工作区（已废弃） |

## 7. 文档维护约定

- 所有 .md 文档提到 EC2 IP 时，**必须使用占位符 `<EC2_PUBLIC_IP>`**
- 答辩前运行 `aws/EC2_IP.ps1` 拿当前 IP，临时填入演示 PPT
- **不要**在代码（.py/.sh/.cmd/.ps1）中硬编码 IP（已通过 `EC2_IP.ps1` 变量化）

## 8. 历史 IP 索引（仅供追踪）

| 历史 IP | 备注 |
|---------|------|
| `44.210.104.56` | Week7 报告早期使用 |
| `34.236.170.122` | 根 README 早期使用 |
| `3.236.82.32` | 早期某次同步使用 |
| `35.174.62.169` | **当前活跃**（2026-08-17 更新） |

> 各 .md 文档中的历史 IP 已统一替换为 `<EC2_PUBLIC_IP>`，详见 `week8/AUDIT_REPORT.md` §2。
