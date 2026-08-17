# 项目总结 PPT 使用说明

> **文件位置**：`docs/ppt.html`
> **技术栈**：Reveal.js 单文件版（CDN，无需本地安装 npm）
> **浏览器要求**：Chrome / Edge / Firefox / Safari 任意现代浏览器

---

## 1. 打开方式

### 方法一：本地双击（最简单）

直接双击 `docs/ppt.html`，默认浏览器自动打开。

### 方法二：命令行打开（Windows）

```bash
start docs\ppt.html
```

### 方法三：从项目根目录用相对路径

若 PPT 已嵌入 README 引用，浏览器可直接 `file://` 访问。

---

## 2. 操作快捷键

| 键位 | 功能 |
|------|------|
| `→` / `Space` | 下一张 |
| `←` | 上一张 |
| `↑` / `↓` | 上下章节（带垂直章节时） |
| `Esc` / `O` | 进入总览模式（网格视图） |
| `F` | 进入全屏模式 |
| `Home` / `End` | 跳到首页 / 末页 |
| 鼠标点击右下角箭头 | 单步翻页 |

---

## 3. PPT 结构（21 张）

| # | 主题 |
|---|------|
| 1 | 标题页 |
| 2 | 项目概述（系统形态） |
| 3 | 业务场景（4 类） |
| 4 | 核心数字速览（表格） |
| 5 | 五大技术亮点 |
| 6 | 系统架构（mermaid） |
| 7 | 技术难点（4 大坑） |
| 8 | 数据流水线（mermaid） |
| 9 | 多源数据规模 |
| 10 | 时序基线（Week3 · 7 模型） |
| 11 | STF 模型架构（创新点） |
| 12 | 时空联合模型对比（Week4） |
| 13 | Optuna 超参优化（Week7） |
| 14 | 四范式异常检测 |
| 15 | 融合权重搜索 |
| 16 | 异常检测最终结果 |
| 17 | 可解释性 SHAP/Attn |
| 18 | API 性能 + 系统演示 |
| 19 | 项目交付清单 |
| 20 | 总结与未来方向 |
| 21 | 致谢 Q&A |

---

## 4. 注意事项

1. **首次加载**需要联网（Reveal.js 来自 jsDelivr CDN）。离线场景下需 `pip install reveal.js` 或预下载。
2. **mermaid 图**：浏览器需联网解析 mermaid CDN（已嵌入 Reveal.js 但 mermaid 模块是独立包）。
   若无法显示 mermaid，把 `<div class="mermaid">` 替换为预渲染 PNG。
3. **打印 / 导出 PDF**：`Ctrl+P` → 选择"另存为 PDF"。每一张幻灯片自动分页。

---

## 5. 转 PowerPoint（如需要 .pptx 文件）

```bash
# 需先 pip install
pip install revealjs-converter
revealjs-converter docs/ppt.html --output docs/项目总结.pptx
```

或手动：在浏览器全屏（F11）后录屏，再嵌入 PPT。
