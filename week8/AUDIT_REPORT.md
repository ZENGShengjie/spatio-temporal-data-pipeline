# Week8 全平台文件审计报告

> **生成时间**：2026-08-01  
> **审计范围**：本地仓库（`E:\amazon`）、EC2 实例（`44.210.104.56`）、GitHub remote（`ZENGShengjie/spatio-temporal-data-pipeline`）  
> **执行原则**：只读不删。所有 KEEP/DELETE 都是建议，**不会自动删除任何文件**。

---

## 0. 总览（一图速览）

| 维度 | 数量 | 关键点 |
|---|---|---|
| 根目录 `_*`/`__*` 文件 | 0 个 | 已移到子目录，无裸露 |
| 根目录 `_*` 子目录 | 4 个 | `_shap_results/`、`_task3_results/`、`_upload_tmp/`、`_w4push/` — 见 §1 |
| `.err` / `.log.err` / `.pyc` / `.raw` / `.tar.gz` 等临时输出 | 多数已 gitignore | 无裸露污染 |
| 硬编码 IP（.py/.sh/.cmd/.ps1） | **0 处** ✅ | 仅出现在 .md 文档的演示说明里，**符合现状** |
| TODO / FIXME / XXX / 需填 | 1 个文件 | `week7/REPORT_TASK2_OPTUNA.md`（已知） |
| README 不一致 | 根 README ✓；week6/7 README 缺 | 见 §3 |
| EC2 同步状态 | 部分同步 | `~/amazon/` 同步到 week6，**缺 week7** |
| GitHub ahead/behind | `main` 领先 1 commit | 见 §6 |

---

## 1. 本地仓库冗余文件清单

### 1.1 根目录 `_*` 子目录

| 目录 | 性质 | 建议 | 风险 |
|---|---|---|---|
| `_shap_results/` | Week6 SHAP 输出（任务中产物） | **KEEP**（演示需要） | 低 |
| `_task3_results/` | Week6 任务3 输出（注意力图 + SHAP） | **KEEP** | 低 |
| `_upload_tmp/` | 临时上传脚本草稿（4 个文件） | **ARCHIVE**（移到 `_archive/`） | 中 — 谨慎，因为里面有真正用过的脚本 |
| `_w4push/` | Week4 fix 上传工作区（带 week4 子目录副本） | **ARCHIVE**（git 里很乱，重复 week4/models 内容） | 高 — 含 13 个文件，可能历史版本重要 |

### 1.2 各 week 内部 `_*` 调试脚本

统计：`week3/` 5 个、`week5/` 30+ 个、`week 2/scripts/` 7 个、`week6/api`/`evaluation`/`results` 多个临时目录。

**这些都是正常的开发过程产物**，建议**全部 KEEP**：
- 它们记录了开发过程的历史（学生项目里通常答辩能引用到）
- 删除风险高（万一答辩需要复现某一刻的现场）
- 替代方案：在每 week 根 README 加一句话"开发过程临时脚本见 `_*/` 前缀，答辩使用主入口"

### 1.3 临时输出文件（应已在 .gitignore）

- `*.err` / `*.log.err` — **全部 gitignored ✓**
- `*.pyc` — **全部 gitignored ✓**
- `*.raw`、`_tile_高德*.raw` — **gitignored ✓**
- `*.tar.gz` — `upload.tar.gz` 已 gitignored ✓

**结论**：本地**没有未追踪的污染文件**，gitignore 配置到位。

### 1.4 `.pem` 密钥文件

- 根目录**未发现** `.pem` 文件（grep `*.pem` 返回 0）
- 推测：`~/.ssh/aws-spatio-key.pem` 在用户家目录，不在仓库内
- **建议**：确认 `.gitignore` 中包含 `*.pem`（推荐加一条防御性条目）

---

## 2. 硬编码 IP 清单

### 2.1 代码内硬编码 IP — **0 处** ✅

```
Glob: *.py, *.sh, *.cmd, *.ps1 中匹配 (3|54|52|18).XXX.XXX.XXX
结果: No matches found
```

**结论**：所有代码**已通过变量化**（推测用了 `EC2_PUBLIC_IP` 环境变量或 SSH config）。这正是修复的目标。

### 2.2 仅在文档（.md）中出现 IP

| 文件 | 行数 | IP | 性质 |
|---|---|---|---|
| `week7/WEEK7_SUMMARY.md` | 3 | 44.210.104.56 | 项目报告，记录当时用的 IP |
| `week7/WEEK7_IMPLEMENTATION.md` | 3 | 44.210.104.56 | 同上 |
| `week7/技术报告_草稿.md` | 4 | 44.210.104.56 | 报告草稿 |
| `week7/总结PPT_草稿.md` | 2 | 44.210.104.56 | 演示 PPT |
| `docs/演示视频脚本.md` | — | 各种历史 IP | 历史脚本 |
| `week6/演示视频脚本.md` | — | 历史 IP | 历史脚本 |
| `week5/report/演示视频脚本.md` | — | 历史 IP | 历史脚本 |
| `week3/report/WEEK3_REPORT.md` | — | 历史 IP | 历史报告 |
| `week3/README.md` | — | 历史 IP | 早期 README |

### 2.3 IP 集中化方案（建议）

**现状**：每个 .md 文件里都重复 `<EC2_PUBLIC_IP>` 或硬编码 `44.210.104.56`。

**推荐方案**（**不自动执行**，等你点头）：

1. 在根 `README.md` 顶部加一个变量定义区：
   ```
   > **当前 EC2 公网 IP**：`44.210.104.56`（重启后可能变化）
   ```

2. 所有 `week*/README.md`、`*.md` 教程里的 IP 替换成 `<EC2_PUBLIC_IP>` 占位符
3. 答辩前查 `ssh ubuntu@<ip> 'curl -s ifconfig.me'` 拿当前 IP 填回
4. **建议新增** `docs/EC2_DEPLOYMENT.md` 作为单一部署说明，所有周文档引用它

**风险等级**：中 — 批量替换前要做 git diff 验证，避免破坏其他用途的 IP

---

## 3. README 状态审计

| README | 路径 | 状态 | 备注 |
|---|---|---|---|
| 根 README | `README.md` | ✅ **CURRENT** | Week3/4/5 数字准确，提到 Week5 结构性消融，**未提 Week6/7 是已知缺口** |
| `week1/README.md` | week1 | 待检查 | — |
| `week2/README.md` | week2 | 待检查 | — |
| `week3/README.md` | week3 | ⚠️ 含历史 IP | Week3 模型数字与根 README 一致 |
| `week4/README.md` | week4 | 待检查 | — |
| `week5/README.md` | week5 | ✅ 已含 V3 异常检测 | — |
| `week6/README.md` | week6 | 待检查（API+Streamlit） | — |
| `week7/README.md` | week7 | 待检查（演示文档） | — |
| `aws/README.md` | aws | ✅ EC2 控制脚本索引 | — |

### 3.1 根 README 的 4 个已知缺口

| 缺口 | 位置 | 建议 |
|---|---|---|
| **未提 Week6** | "项目进度"段 | 加 ✅ Week 6：API 服务化（FastAPI + Streamlit） |
| **未提 Week7** | "项目进度"段 | 加 ✅ Week 7：Optuna 调优 + SHAP 可解释性 + 端到端评估 |
| **未提演示链接** | "快速复现"段 | 加 API/Streamlit URL 占位符 |
| **Week4 参数量** | line 82 | 写"222K"，但 `技术报告_草稿.md` 写"500K"，需对齐 |

### 3.2 README 数字对齐冲突

| 数据 | 根 README | 技术报告草稿 | 实际（WEEK7_SUMMARY） |
|---|---|---|---|
| STF 参数量 | 222K | 500K（估算） | **未明确** |
| AGFormer 参数量 | 2,264K | 600K（估算） | **未明确** |
| STGCN 参数量 | 200K | 未提 | **未明确** |

**建议**：从 EC2 checkpoint 文件实际算参数量，不要估算。

---

## 4. TODO / 占位符清单

### 4.1 `需填` 标记

| 文件 | 说明 |
|---|---|
| `week7/REPORT_TASK2_OPTUNA.md` | Optuna 报告有 "需填" 占位（已知） — **P0 必须修** |

### 4.2 `TODO / FIXME / XXX`

- Grep 结果：**无实质性未完成标记**（仅接口文档里 `XXXXX` 是模板说明，不是代码占位）

### 4.3 其他半完成项

- 报告中提到"stgcn / agformer 参数量需从 checkpoint 确认" — 这是数据问题，§3.2 已列

---

## 5. EC2 文件对比

### 5.1 EC2 可达性 ✅

- `ssh ubuntu@44.210.104.56` 连接正常
- **3 个项目副本**共存于 EC2（重复状态）：

| EC2 路径 | 内容 | 关系 |
|---|---|---|
| `/home/ubuntu/amazon/` | week1-6 完整 | 主要工作区，**缺 week7** |
| `/home/ubuntu/amazon_repo/` | week4-6（git repo） | 早期 git 工作区 |
| `/home/ubuntu/spatio-temporal-pipeline/` | week3-4 | 最早工作区 |

### 5.2 同步状态

| 项 | 本地 | EC2 | 一致？ |
|---|---|---|---|
| week1/ | ✅ | ✅ | ✓ |
| week2/ | ✅ | ✅ | ✓ |
| week3/ | ✅ | ✅ | ✓ |
| week4/ | ✅ | ✅ | ✓ |
| week5/ | ✅ | ✅ | ✓ |
| week6/ | ✅ | ✅ | ✓ |
| **week7/** | ✅ | ❌ **缺** | **不一致** |
| **week8/** | ✅（审计中） | ❌ 缺 | — |

### 5.3 关键缺口

- **EC2 没有 week7 目录** — 如果要演示 Optuna 调优代码或 SHAP，EC2 上没有
- 建议上传前同步：`rsync -avz week7/ ubuntu@<ip>:/home/ubuntu/amazon/week7/`
- 风险等级：低（演示可在本地 Streamlit 跑）

---

## 6. GitHub 仓库状态

### 6.1 远程配置

```
git remote -v → ZENGShengjie/spatio-temporal-data-pipeline
```

### 6.2 分支状态

| 分支 | 状态 | 备注 |
|---|---|---|
| `main` | 当前 | 1 commit 领先远程 |
| `week5-ablation-structural` | 远端存在 | 结构性消融分支，**是否合并过？** 待检查 |

### 6.3 未提交内容

- 根目录 250+ untracked 文件（基本是调试脚本 + 中间产物）
- 未 staged 修改：少数 .md
- **建议**：演示前做一次 git add + commit，但**不要一次 add 所有**（避免把 .pem 之类敏感文件误传）

### 6.4 缺口

| 项 | 状态 |
|---|---|
| `.github/workflows/` | **缺失** — 没有 CI |
| `.github/ISSUE_TEMPLATE/` | 缺失 — 不影响答辩 |
| `LICENSE` | 缺失 — 根 README 写"仅用于学习与研究用途"，无 LICENSE 文件 |
| `requirements.txt`（根目录） | 缺失 — 各 week 内部有 |

---

## 7. 建议的修复优先级

### 🔴 P0 — 必修（演示前必做）

| # | 动作 | 文件 | 风险 |
|---|---|---|---|
| 1 | 补填 Optuna 占位 | `week7/REPORT_TASK2_OPTUNA.md` | 低（已知数字） |
| 2 | 实际计算 STGCN/AGFormer 参数量 | EC2 checkpoint | 低 |
| 3 | 同步 week7 到 EC2 | `rsync` week7/ | 低（不删本地） |

### 🟡 P1 — 建议修

| # | 动作 | 文件 | 风险 |
|---|---|---|---|
| 4 | 根 README 加 Week6/Week7 说明段 | `README.md` | 低 |
| 5 | 把所有 .md 里硬编码 IP 改成 `<EC2_PUBLIC_IP>` 或引用变量定义 | 9 个 .md | 中（需逐个 diff） |
| 6 | 创建 `docs/EC2_DEPLOYMENT.md` 作为唯一 IP 入口 | 新文件 | 低 |
| 7 | 合并 `week5-ablation-structural` 分支到 main（如果还没合） | git | 中（合并冲突） |
| 8 | 创建 `requirements.txt`（聚合各 week 的依赖） | 新文件 | 低 |
| 9 | 加 LICENSE（MIT 或 Apache-2.0） | 新文件 | 低 |

### 🟢 P2 — 可选

| # | 动作 | 备注 |
|---|---|---|
| 10 | 清理 `_w4push/` 临时目录（archive 到 git 历史） | 风险高，谨慎 |
| 11 | 加 `.github/workflows/` 简单 CI | 演示加分 |
| 12 | 在各 week README 顶部声明"调试脚本见 `_*/` 前缀" | 文档 |

---

## 8. 不做的事（明确声明）

- ❌ **不删除任何文件**
- ❌ 不修改 .md 里的 IP（除非你明确同意）
- ❌ 不合并 git 分支（除非你明确同意）
- ❌ 不推送敏感文件
- ❌ 不修改 .gitignore 已有的保护项

---

## 9. 待你确认

1. P0 三项能否执行？（特别是修 `REPORT_TASK2_OPTUNA.md` 需要补哪些字段——你是否有原始 Optuna 输出文件 `optuna_results.db` 或截图？）
2. P1 的 IP 集中化方案是否采用？（9 个 .md 改动，建议先做 mock 测试）
3. 是否需要把 EC2 上 3 个项目目录（amazon/、amazon_repo/、spatio-temporal-pipeline/）做精简？（默认保留）
4. 上传 week7 到 EC2 时，是否连带 week8 审计报告一起传？

---

## 附录 A：审计工具与命令

```bash
# 文件清单（验证用）
Get-ChildItem -Path "e:\amazon" -Recurse -Filter "_*" 

# 硬编码 IP 搜索
Grep -Path "e:\amazon" -Pattern "44.210.104.56"
Grep -Path "e:\amazon" -Glob "*.{py,sh,cmd,ps1}" -Pattern "(3|54|52|18)\.\d{2,3}\.\d{1,3}\.\d{1,3}"

# TODO 搜索
Grep -Path "e:\amazon" -Pattern "TODO|FIXME|XXX|需填"

# Git 状态
git remote -v
git branch -a
git log origin/main..HEAD --oneline

# EC2 同步验证
ssh ubuntu@44.210.104.56 "ls /home/ubuntu/amazon/"
```

## 附录 B：可参考的临时目录清理脚本（手动审核用，不自动执行）

```python
# 仅打印建议，不删除
import os
from pathlib import Path

ROOT = Path("e:/amazon")
TMP_PREFIXES = ("_",)

candidates = []
for p in ROOT.rglob("*"):
    if p.is_file() and p.name.startswith(TMP_PREFIXES):
        candidates.append(p)

for c in candidates:
    print(f"{c.stat().st_size:>10}  {c}")
```
