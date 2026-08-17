# Week8 任务 #2 — Week4 模型参数量实测

> **目的**：验证 Week4/README 中"STGCN/AGFormer 参数量需从 checkpoint 确认"的警告（来自 `week8/AUDIT_REPORT.md` §3.2）。

---

## 1. 背景

`week8/AUDIT_REPORT.md` §3.2 提到：

> | STF 参数量 | 222K | 500K（估算） | **未明确** |
> | AGFormer 参数量 | 2,264K | 600K（估算） | **未明确** |
> | STGCN 参数量 | 200K | 未提 | **未明确** |
>
> 建议：从 EC2 checkpoint 文件实际算参数量，不要估算。

但 EC2 上的 `week4/weights/` 只有 `stf_taxi_flow_total_v4fix.pth`（STGCN/AGFormer 权重未保存），所以无法直接读 checkpoint。**采用替代方案**：从模型代码直接构造并计算（参数定义与训练时一致）。

## 2. 实测方式

- 用 `week4/models/stf_model.py` / `stgcn_model.py` / `agformer_model.py` 的模型类
- 配置：Week4 训练时实际值（in_dim=7, hidden=64, horizon=48, N=1024, dropout=0.1）
- `sum(p.numel() for p in model.parameters())` 统计

脚本见 `_calc_params.py`（已运行在 EC2）。

## 3. 实测结果

| 模型 | 实际参数量 | 现有 README | 差距 |
|---|---|---|---|
| **STF** | **222,576** | 222K | ✓ |
| **STGCN** | **199,536** | 200K | ✓ |
| **AGFormer** | **2,263,536** | 2264K | ✓ |

**结论**：三个模型的参数量实测值与 README 现值一致，**无需修正**。

## 4. 误判原因

`week8/AUDIT_REPORT.md` §3.2 中提到的"500K / 600K 估算"来自 `week7/技术报告_草稿.md`。但那是**初稿估算**，可能基于某个早期版本或粗略公式。**实际**：

- STF 222K：环境编码器（n_nodes=1024 → 64 是大头）+ 局部编码器（in_dim=7 → 64）+ cross-attn + temporal conv + 4×decoder
- STGCN 200K：4 块 STGCN 块（每块 2 个 GCNConv + temporal conv + Linear fusion）+ GRU + decoder
- AGFormer 2.26M：自适应邻接矩阵 (1024×1024 = 1.05M) 占主要，加上 2 块 AGFormerBlock（每块 spatial + temporal + FFN）+ GRU + decoder

## 5. 后续动作

- ✅ `week4/README.md` 参数量表无需修改
- ✅ `week7/技术报告_草稿.md` 初稿待修正时，建议改成"STF 222K / STGCN 200K / AGFormer 2.26M"
- ⚠️ 建议：未来 STF/STGCN/AGFormer 训练后都应保存 `.pth`，便于审计（目前仅 STF 留存）
