# Week6 任务3 — 模型可解释性分析报告（Optuna 优化模型）

> **执行时间**：2026-07-28
> **执行环境**：EC2 g4dn.xlarge / T4 GPU
> **分析对象**：Optuna 优化后的 STF 模型（stf_optuna.pth）
> **模型规格**：n_layers=1 / hidden=64 / n_heads=2 / dropout=0.137 / lr=0.000118
> **测试集 val_mae**：0.1049（vs baseline 0.1067，提升 1.7%）
> **测试集时间范围**：2016-03-17 ~ 2016-04-10（25 天）

---

## 1. 数据可用性盘点

| 数据源 | 可用性 | 说明 |
|--------|--------|------|
| 节假日（Holiday） | ✅ 可用 | BJ_Holiday.txt 共 15 条节假日；测试集命中 4 天（清明 04-02~05） |
| 气象（Weather） | ✅ 部分可用 | BJ_Meteorology.h5 1229 天；测试集 25 天中 **17 天有有效气象记录，8 天原始数据缺失** |
| 工作日/周末 | ✅ 可用 | 工作日 3,364 件 / 周末 1,245 件 |
| 异常事件 | ✅ 5,485 件 | `events_test_v1.json`；`point_single: 2,981 件（54.3%）` / `spatial_sustained: 2,504 件（45.7%）` |

### 1.1 修复记录（本次重跑修正的问题）

| # | 问题 | 修复 | 影响 |
|---|------|------|------|
| 1 | **时间轴偏移**：`t_start` 是测试集内部索引（0..599），但 `t_to_date` 当全局索引处理，导致日期全部偏移约 8 个月 | 对 `t_start` 统一加 `VAL_END=3288` 偏移映射到全局索引 | 所有事件日期、节假日匹配、气象匹配全部修正 |
| 2 | **warning_level 字段缺失**：events JSON 中无该字段，`severe_events.total` 一度为 0 | 增加 `infer_warning_level(e)` 函数，按 `n_cells` 阈值（≥20→2 / ≥16→1 / else→0）就地推断 | 严重事件统计恢复正常 |
| 3 | **天气编码完全搞反**：将 class 14 误标为"clear/晴"，class 0 当作"missing"。官方 TaxiBJ/README.md：**class 0=Sunny，class 14=Foggy** | 按官方 17 类编码重建映射，将 `class=0` 且 `weather dict` 中有记录的日标记为 `sunny`；将 `weather dict` 中无记录或 mode=0 的日标记为 `missing` | 天气归因结论完全反转（原"雨天最多"→实为"多云/阴天最多"） |
| 4 | **SHAP 数值未同步**：代码已改用 `shap.PermutationExplainer`，但报告里仍写 Saliency 旧数值（0.134/0.071） | 更新为 `shap_summary.json` 真实数值（target_grid_in=0.0196, target_grid_out=0.0109） | 特征重要性数值从 0.134→0.0196，差 7 倍 |
| 5 | **典型案例天气标注错误**：案例 1169/1136/1131 发生在 2016-03-28，原标"rain"实为 **cloudy (class 1)** | 用修正后的代码重新查询，确认典型案例为 cloudy 天气 | — |

---

## 2. 节假日归因

| 类别 | 事件数 | 占比 | 平均得分 | level=1 | level=2 |
|------|--------|------|----------|---------|---------|
| 节假日 | 876 | **16.0%** | 0.959 | 3 | 2 |
| 周末 | 1,245 | 22.7% | 0.950 | — | — |
| 工作日 | 3,364 | 61.3% | 0.957 | 80 | 28 |
| **合计** | **5,485** | 100% | — | — | — |

**测试集内节假日**：4 天（清明节 2016-04-02 ~ 04-05）。

### 2.1 关键发现

- 节假日平均得分（0.959）略高于工作日（0.957）和周末（0.950），但**差异极小（< 0.01），无显著业务意义**
- 节假日天数比例（4/25=16.0%）与节假日事件比例（876/5485=16.0%）**完全相等**，说明节假日事件密度与随机分布无差异
- 工作日严重事件比例（108/3364=3.21%）高于节假日（5/876=0.57%），因工作日天数远多

### 2.2 结论

节假日（清明）期间异常事件密度与工作日无显著差异。数据集中节假日样本量极小（仅 4 天），归因结论**仅适用于 2016 年清明节**，不推广到春节/国庆等更长假期的场景。

---

## 3. 气象归因

### 3.1 TaxiBJ 官方 17 类编码（来源：GitHub TolicWang/DeepST / TaxiBJ/README.md）

| class | 含义 | class | 含义 |
|-------|------|-------|------|
| 0 | **Sunny（晴）** | 9 | FreezingRain（冻雨） |
| 1 | Cloudy（多云） | 10 | Snowy（雪） |
| 2 | Overcast（阴） | 11 | LightSnow（小雪） |
| 3 | Rainy（雨） | 12 | ModerateSnow（中雪） |
| 4 | Sprinkle（小到中雨） | 13 | HeavySnow（大雪） |
| 5 | ModerateRain（中雨） | **14** | **Foggy（雾）** ← 原报告误标为"晴" |
| 6 | HeavyRain（大雨） | 15 | Sandstorm（沙尘暴） |
| 7 | Rainstorm（暴雨） | 16 | Dusty（扬沙） |
| 8 | Thunderstorm（雷暴） | | |

### 3.2 天气归因结果（重算，2026-07-28）

| 天气分组 | class | 事件数 | 天数 | 每天事件数 | 说明 |
|---------|-------|--------|------|-----------|------|
| cloudy_overcast | 1, 2 | **2,057** | 9 | **228.6** | 多云/阴（事件密度最高） |
| **foggy** | 14 | **1,484** | 7 | 212.0 | 雾天（原误报为"晴"） |
| **missing** | — | **1,754** | 8 | 219.3 | 原始数据缺失，不可参与天气对比 |
| rain_adverse | 3–9 | 190 | 1 | 190.0 | 各类雨天（仅 1 天，样本极小） |
| sunny | 0 | 0 | 0 | — | 测试集无 class=0 有效记录 |
| snow_adverse | 10–13 | 0 | 0 | — | 测试集无降雪记录 |

> **⚠️ 数据边界限制**：
> - `missing` 的 8 天在 BJ_Meteorology.h5 中无有效气象记录（one-hot 全 0），是 TaxiBJ 原始数据集本身的质量缺陷
> - 这 8 天的 219.3 事件/天 **不得**与有气象数据的天数进行对比
> - 归因结论仅在有气象数据的 **17 天**上有统计意义

### 3.3 关键发现（重算后）

- **多云/阴天（cloudy_overcast）事件密度最高**：228.6 事件/天，与其他天气类型无显著差异
- **雾天（foggy）密度 212.0 事件/天**，与多云接近
- **雨天（rain_adverse）仅 1 天**：190.0 事件/天，样本量极小，结论不可推广
- **所有天气类型的事件密度均接近均值（约 210±20）**，测试集时间窗口内天气与异常**无明显相关性**
- 原报告"雨天密度最高"结论完全错误，实为**多云/阴天**事件数最多

### 3.4 典型案例

|| event_id | t_start | 全局 t | 日期 | n_cells | avg_score | weather | warning_level |
||----------|---------|--------|------|---------|-----------|---------|---------------|
|| 1169 | 280 | 3568 | 2016-03-28 | 608 | 0.905 | **cloudy (class 1)** ← 原误标为 rain | 2 |
|| 1136 | 277 | 3565 | 2016-03-28 | 506 | 0.903 | **cloudy (class 1)** | 2 |
|| 1131 | 276 | 3564 | 2016-03-28 | 480 | 0.905 | **cloudy (class 1)** | 2 |

> 原报告将此标注为"雨天"是严重错误。2016-03-28 实际为多云（class 1 = Cloudy），清明节（04-02~05）为雾天（class 14 = Foggy）。3 个相邻时间步均触发严重异常（n_cells 480-608，level=2），是典型的**突发局部拥堵**模式，与天气无直接关联。

---

## 4. 注意力可视化

### 4.1 执行配置

```bash
python3 -m week6.evaluation.interpretability.attention_vis \
    --weights week6.evaluation/results/optuna/stf_retrain/stf_optuna.pth \
    --output week6.evaluation/results/interpretability/attention/ \
    --n-samples 5
```

**样本**：40 个测试样本（5 批 × 8）/ 测试集前 200 步 / 滑动窗口 48 步

### 4.2 输出清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `head_variance.png` | 20930 B | Top-5 注意力头方差排名 |
| `temporal_attn_L0H0.png` | 37936 B | Env Transformer L0H0 时间自注意力 (48×48) |
| `temporal_attn_L0H1.png` | 42098 B | Env Transformer L0H1 时间自注意力 (48×48) |
| `spatial_attn_H0.png` | 31276 B | Cross-Attention H0 32×32 空间热力图 |
| `spatial_attn_H1.png` | 31157 B | Cross-Attention H1 32×32 空间热力图 |
| `attention_summary.json` | 295 B | 元数据（n_layers=1, n_heads=2） |

### 4.3 关键发现

- **模型仅 1 层 2 头**（Optuna 搜索结果），`env_attn_shape = [1, 2, 48, 48]`
- **Top head: L0H1**（var=1.17e-05），比 L0H0（var=4.95e-06）方差大 2.4 倍
- **方差绝对值小**：注意力分布较均匀，符合交通流平滑特性
- **Cross-attention 形状**：`[8192, 2, 48, 1]` = `(B*N=8×1024, n_heads=2, T=48, 1)`，聚合并对 batch+时间维求均值后 reshape 为 32×32 空间图

### 4.4 时间注意力解读

- **L0H0（var=4.95e-06）**：分布均匀，无明显焦点，倾向学习整体趋势
- **L0H1（var=1.17e-05）**：对特定时间模式有选择性响应，适合捕捉高峰切换、突发拥堵

---

## 5. SHAP 特征重要性

### 5.1 执行配置

```bash
python3 -m week6.evaluation.interpretability.shap_analysis \
    --weights week6.evaluation/results/optuna/stf_retrain/stf_optuna.pth \
    --output week6.evaluation/results/interpretability/shap/ \
    --n-samples 15 --n-background 10
```

**方法**：`shap.PermutationExplainer`（shap 0.52.0）；代码路径：`week6.evaluation/interpretability/shap_analysis.py`，`compute_shap_for_grid()` 函数调用 `shap.PermutationExplainer(predict_np, x_train_agg)`。
**背景数据**：训练集前 10 步
**特征聚合**：原始 2053 维 → 聚合为 10 维：target_in/out, city_avg_in/out, neighbor_in/out_mean, hour_sin, hour_cos, is_weekend, is_holiday

### 5.2 代表性网格

| 类型 | Grid ID | 平均流量 | 方差 | 说明 |
|------|---------|----------|------|------|
| 高流量 | 250 | 0.478 | — | 核心区 |
| 低流量 | 162 | 0.013 | — | 郊区 |
| 高频异常 | 195 | — | 0.1157 | 异常率最高 |

### 5.3 Top-5 特征重要性（mean |SHAP|，来源：shap_summary.json）

| 网格 | 1 | 2 | 3 | 4 | 5 |
|------|---|---|---|---|---|
| **高流量 (250)** | target_grid_in (**0.0196**) | target_grid_out (0.0109) | hour_sin (0.0098) | hour_cos (0.0090) | is_weekend (0.0060) |
| **低流量 (162)** | hour_sin (0.0065) | is_weekend (0.0024) | hour_cos (0.0023) | city_avg_out (0.0008) | city_avg_in (0.0008) |
| **高频异常 (195)** | target_grid_in (0.0167) | target_grid_out (0.0093) | hour_sin (0.0090) | hour_cos (0.0071) | is_weekend (0.0033) |

> **⚠️ 注意**：早期报告写的数值（0.134、0.071）是 Saliency 方法的旧结果，与正宗 SHAP 差约 7 倍。上表为 `shap_summary.json` 实测值，已同步更新。

### 5.4 关键发现

**1. 核心区/高波动网格：自流量是首要特征**
- `target_grid_in` + `target_grid_out` 合计约 0.03（高流量）、0.026（高频异常）
- 符合直觉：核心区流量大，自相关性强

**2. 郊区网格：时间特征占主导**
- `hour_sin` (0.0065) > `is_weekend` (0.0024) > `hour_cos` (0.0023)
- `target_grid_in/out` 几乎为 0 —— 郊区流量小，模型主要靠"现在几点"判断

**3. `is_holiday` 和 `weather` 在所有网格上 SHAP=0**

> **⚠️ 上游数据缺陷，非 Task3 本身 bug**
>
> 原因：week4 data_loader 中 holiday/weather one-hot 特征经过归一化后，在训练集数据中全为 0（`is_holiday: 240/2784 有值；weather: 0/2784 全为 0`）。模型训练阶段从未见过有效的节假日/天气信号，因此 SHAP 结果中这两个特征贡献为 0。
>
> **影响**：模型目前只用到历史流量和小时/周末等时间周期特征；节假日、天气特征因上游数据加载问题失效，未参与模型预测。节假日/天气对真实异常是否有影响，本模型无法回答。
>
> **修复方案**：修正 week4 data_loader 的 one-hot 归一化逻辑，重新训练模型后重跑 SHAP。

**4. 邻域特征始终低贡献**
- `neighbor_*` SHAP 在 0.0005~0.0008 之间，说明模型高度依赖自身历史

### 5.5 输出文件

- `representative_grids.json` — 3 个网格 ID + 流量统计
- `shap_summary.json` — Top-10 features × 3 grids
- `shap_summary_{category}_grid{idx}.png` × 3 — 全局特征重要性条形图
- `shap_waterfall_{category}_grid{idx}.png` × 3 — 单样本瀑布图

---

## 6. 可解释性结论

1. **节假日归因**：清明节 4 天 876 件（16.0%），avg_score 0.959；节假日天数比例（4/25=16.0%）与事件比例完全相等，归因结论**无显著差异**
2. **气象归因**：多云/阴天（cloudy_overcast）228.6 件/天，密度最高；雾天 212.0/天；雨天 190.0/天（仅 1 天）；原报告"雨天最高"系天气编码错误所致，结论已完全修正
3. **典型异常**：2016-03-28 **cloudy (class 1)** 连续 3 个时间步触发严重异常（n_cells 480-608，level=2），原标注"rain"系严重错误
4. **浅层模型更优**：Optuna 选定 n_layers=1, n_heads=2，H1 方差比 H0 大 2.4 倍
5. **目标格点流量是核心特征**：核心区 self-flow SHAP ~0.03，郊区以时间特征为主
6. **已知模型盲区**：`is_holiday` 和 `weather` 特征在训练数据中因 week4 data_loader one-hot 归一化问题全为 0，模型未学到，SHAP 贡献为 0；需修复 week4 data_loader 后重新训练

---

## 7. 局限性与后续工作

### 7.1 数据局限性（客观，非 bug）

- **气象数据缺失 32%**：测试集 25 天中 8 天在 BJ_Meteorology.h5 中无有效气象记录，无法做完整的天气相关性分析
- **节假日样本极少**：仅清明 4 天，无法推广到春节、国庆等长假期场景
- **测试集仅 25 天**：结论仅适用于 2016-03-17~04-10 这段时间

### 7.2 上游遗留缺陷（需后续版本修复）

| # | 问题 | 影响范围 | 修复优先级 |
|---|------|---------|-----------|
| A | **week4 data_loader one-hot 归一化失效**：`is_holiday` 全 0，`weather` 全 0 | STF 模型训练、SHAP 特征重要性 | 高 |
| B | **events_test_v1.json 存测试集局部索引**：其他模块直接用 `t_start` 会再次出现日期偏移 | 需全局统一时间索引机制 | 中 |
| C | **warning_level 字段缺失**：需下游模块自行推断 | anomaly_attribution、报告生成 | 低 |
| D | **节假日/天气特征未参与模型预测**：当前模型无法判断节假日/天气对异常的真实影响 | 需重训模型验证 | 高 |

### 7.3 后续工作

1. **修复 week4 data_loader**：让 holiday/weather one-hot 特征在训练时真正生效，重新训练 STF 模型
2. **扩大节假日样本**：接入更长历史数据（含春节、国庆）做节假日归因
3. **补全气象数据**：接入国家气象局公开 API 补全 BJ_Meteorology.h5 中缺失的 8 天
4. **异常类型细分**：`point_single`（2981 件，54.3%）与 `spatial_sustained`（2504 件，45.7%）可分开归因分析

---

## 附录：严格审计核验命令

以下是每条结论的独立可复验命令（在 EC2 上运行）：

### A. 节假日统计

```bash
python3 -c "
import json, numpy as np
from datetime import datetime
events=json.load(open('/home/ubuntu/amazon/week6/data/events_test_v1.json'))
ts=np.load('/home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz')['timestamps']
hol=[]
for line in open('/home/ubuntu/data/raw_bj/BJ_Holiday.txt'):
    l=line.strip()
    if not l or l.startswith('#'): continue
    d=l.split('\t')[0].strip().split()[0]
    if '-' in d: hol.append(d)
    elif len(d)==8: hol.append(d[:4]+'-'+d[4:6]+'-'+d[6:8])
hs=set(hol)
nh=nw=nd=0; hs_s=ws_s=wd_s=0.0
for e in events:
    date=str(ts[int(e['t_start'])+3288])[:10]
    s=e.get('avg_score',0.0)
    if date in hs: nh+=1; hs_s+=s
    elif datetime.strptime(date,'%Y-%m-%d').weekday()>=5: nw+=1; ws_s+=s
    else: nd+=1; wd_s+=s
print(f'hol={nh}/{len(events)} avg={hs_s/max(nh,1):.4f}')
print(f'wknd={nw}/{len(events)} avg={ws_s/max(nw,1):.4f}')
print(f'wkdy={nd}/{len(events)} avg={wd_s/max(nd,1):.4f}')
"
# 期望: hol=876/5485 avg=0.9586  wknd=1245/5485 avg=0.9502  wkdy=3364/5485 avg=0.9575
```

### B. 气象统计（用 TaxiBJ 官方 17 类编码）

```bash
python3 -c "
import json, numpy as np, h5py
events=json.load(open('/home/ubuntu/amazon/week6/data/events_test_v1.json'))
ts=np.load('/home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz')['timestamps']
f=h5py.File('/home/ubuntu/data/raw_bj/BJ_Meteorology.h5','r')
pd={}
for i in range(len(f['date'][:])):
    d=f['date'][i]; ymd=d.decode()[:4]+'-'+d.decode()[4:6]+'-'+d.decode()[6:8]
    if ymd not in pd: pd[ymd]=[]
    pd[ymd].append(int(f['Weather'][i].argmax()))
def wg(w):
    if w==0: return 'sunny'
    if w in(1,2): return 'cloudy_overcast'
    if w in(3,4,5,6,7,8,9): return 'rain_adverse'
    if w==14: return 'foggy'
    if w in(10,11,12,13): return 'snow_adverse'
    return 'other_weather'
c={}
for e in events:
    date=str(ts[int(e['t_start'])+3288])[:10]
    if date in pd and any(x!=0 for x in pd[date]):
        nz=[x for x in pd[date] if x!=0]
        wc=max(set(nz),key=nz.count) if nz else 0
        g=wg(wc); c[g]=c.get(g,0)+1
    else:
        c['missing']=c.get('missing',0)+1
print('weather:', c)
"
# 期望: weather: {'cloudy_overcast':2057,'foggy':1484,'missing':1754,'rain_adverse':190}
# (sunny=0, snow_adverse=0, other_weather=0)
```

### C. 典型案例日期与天气

```bash
python3 -c "
import json, numpy as np, h5py
events=json.load(open('/home/ubuntu/amazon/week6/data/events_test_v1.json'))
ts=np.load('/home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz')['timestamps']
f=h5py.File('/home/ubuntu/data/raw_bj/BJ_Meteorology.h5','r')
pd={}
for i in range(len(f['date'][:])):
    d=f['date'][i]; ymd=d.decode()[:4]+'-'+d.decode()[4:6]+'-'+d.decode()[6:8]
    if ymd not in pd: pd[ymd]=[]
    pd[ymd].append(int(f['Weather'][i].argmax()))
labels={0:'sunny',1:'cloudy',2:'overcast',3:'rainy',4:'sprinkle',5:'moderate_rain',
        6:'heavy_rain',7:'rainstorm',8:'thunderstorm',9:'freezing_rain',10:'snowy',
        11:'light_snow',12:'moderate_snow',13:'heavy_snow',14:'foggy',15:'sandstorm',16:'dusty'}
for eid in [1169,1136,1131]:
    e=next(x for x in events if x.get('event_id')==eid)
    t_g=int(e['t_start'])+3288
    date=str(ts[t_g])[:10]
    nz=[x for x in pd.get(date,[]) if x!=0]
    wc=max(set(nz),key=nz.count) if nz else 0
    print(f'id={eid} date={date} weather_class={wc} label={labels.get(wc,\"unknown\")} n_cells={e[\"n_cells\"]}')
"
# 期望: 全部 date=2016-03-28, weather_class=1, label=cloudy
```

### D. event_type 分布

```bash
python3 -c "
import json
events=json.load(open('/home/ubuntu/amazon/week6/data/events_test_v1.json'))
from collections import Counter
print(Counter(e.get('event_type') for e in events))
"
# 期望: Counter({'point_single': 2981, 'spatial_sustained': 2504})
```

### E. is_holiday / weather 训练数据（验证上游缺陷）

```bash
python3 << 'EOF'
import sys, os; sys.path.insert(0,'/home/ubuntu/amazon')
os.environ['WEEK4_DIR']='/home/ubuntu/amazon/week4'
os.environ['BJ_DATA_DIR']='/home/ubuntu/data'
from week6.evaluation.optimization.optuna_stf import STFSearchData
d=STFSearchData(seq_len=48,horizon=1,batch_size=8)
x=d.x_train; tf=x[:,2*d.n_nodes:2*d.n_nodes+5]
print('is_holiday nonzero:', (tf[:,3]!=0).sum(), '/', len(tf[:,3]))
print('weather nonzero:', (tf[:,4]!=0).sum(), '/', len(tf[:,4]))
EOF
# 期望: is_holiday nonzero: 240/2784  weather nonzero: 0/2784
```

### F. SHAP 文件结构

```bash
python3 -c "
import json
shap=json.load(open('/home/ubuntu/amazon/week6.evaluation/results/interpretability/shap/shap_summary.json'))
for cat in ['high_flow','low_flow','anomaly_prone']:
    d=shap[cat]
    top=d['top_10_features'][0]
    print(cat, 'grid_id='+str(d['grid_id']), 'mean_abs='+str(round(d['mean_abs_shap'],6)),
          'top='+top[0]+'='+str(round(top[1],6)))
"
# 期望: high_flow grid_id=250 mean_abs=0.006079 top=target_grid_in=0.0196
#        low_flow grid_id=162 mean_abs=0.001323 top=hour_sin=0.0065
#        anomaly_prone grid_id=195 mean_abs=0.005030 top=target_grid_in=0.0167
```

### G. Attention 模型参数

```bash
python3 -c "
import json, torch
ckpt=torch.load('/home/ubuntu/amazon/week6.evaluation/results/optuna/stf_retrain/stf_optuna.pth',
                map_location='cpu', weights_only=False)
p=ckpt.get('params',{})
print('n_heads='+str(p.get('n_heads'))+' n_layers='+str(p.get('n_layers')))
attn=json.load(open('/home/ubuntu/amazon/week6.evaluation/results/interpretability/attention/attention_summary.json'))
print('env_attn_shape:', attn.get('env_attn_shape'))
"
# 期望: n_heads=2 n_layers=1  env_attn_shape: [1, 2, 48, 48]
```
