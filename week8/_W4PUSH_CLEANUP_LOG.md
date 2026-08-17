# Week8 任务 P2-#10 — `_w4push/` 清理记录

> **清理时间**：2026-08-17
> **原则**：删除文件要谨慎（用户要求）。故归档后删除，保留 zip 作为安全网。

---

## 1. 处理流程

1. **列文件**：`dir /s /a-d` 列出 21 个文件 / 63 KB
2. **重复检查**：`fc /B` 比较 `_w4push/week4/models/*.py` 与 `week4/models/*.py`：
   - `agformer_model.py` 不同（13.0K vs 12.6K）
   - `stgcn_model.py` 不同（11.1K vs 10.7K）
   - `config.py` 不同
   - `metrics.py` 不同
   - `run_week4.py` 不同
   - **结论**：`_w4push/week4/models/` 是 Week4 fix 迭代早期的快照，不是多余重复。
3. **打包归档**：`make_archive('E:\amazon\_archive\_w4push_20260817', 'zip', 'E:\amazon\_w4push')` → 25,279 字节
4. **删除**：本地 `_w4push/` 已 `rmdir /S /Q`
5. **加固**：`.gitignore` 加 `_w4push/` 和 `_archive/`，避免未来误传

## 2. 归档位置

- 本地：`E:\amazon\_archive\_w4push_20260817.zip`（25,279 字节）
- 如需在 EC2 留底：`scp E:\amazon\_archive\_w4push_20260817.zip ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/_archive/`

## 3. 安全网

| 位置 | 内容 | 用途 |
|------|------|------|
| `E:\amazon\_archive\` | 已归档的 zip | 急救包（用户可手动 unzip 找回） |
| `.gitignore` 加 `_w4push/` | 阻止 git 误加 | 防意外 |
| `.gitignore` 加 `_archive/` | 阻止归档被传 git | 防意外 |

## 4. 未做的事（明确声明）

- ❌ **未删除** `_upload_tmp/`（虽然也含调试脚本，但有少量 shell，曾被 `git add` 过，且用户原话「删除文件要谨慎」，故 KEEP）
- ❌ **未删除** `_shap_results/` `_task3_results/`（Week6 正式产物，演示需要）
- ❌ **未删除** 双备份目录 `aws\*.sh` `aws\*.cmd`（用户可能误双开，不动）
