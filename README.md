# FFA on Blackwell · MagiAttention SM100 算子教程

面向 GPU 算子/大模型系统工程师的中文教学网站，逐特性拆解
[MagiAttention](https://github.com/SandAI-org/MagiAttention) 的 Blackwell (SM100)
Flex-Flash-Attention kernel（`magi_attention/kernel/cutedsl/`，CuTe DSL）。

## 章节脉络（渐进式）

| # | id | 特性 | 主要源码 |
|---|----|------|----------|
| 00 | `attnslice` | AttnSlice 契约与 host 分发 | `kernel/cutedsl/flex_flash_attn.py` |
| 01 | `blackwell` | TMEM / tcgen05 UMMA / TMA | `ffa_fwd_sm100.py` `__init__`/`__call__` |
| 02 | `pipeline` | Warp 特化与六条流水线 | `ffa_fwd_sm100.py` `kernel()` |
| 03 | `mask` | 块级 Mask 三层防线与 R2P | `mask.py` + `block_info.py` |
| 04 | `softmax` | TMEM 上的 online softmax 与 ex2 仿真 | `softmax.py` + `softmax_step` |
| 05 | `correction` | Correction / epilogue / LSE | `correction_loop` |
| 06 | `scheduler` | LPT / L2 swizzle / CLC 调度 | `tile_scheduler.py` |
| 07 | `backward` | 反向 5-GEMM 与 dQ 原子归约 | `ffa_bwd_sm100.py` 等 |

每章包含：直觉 takeaway、可点击目录、SVG 架构图 + 交互源码工作台（点击图中节点
跳转对应源码块，行号对应仓库真实位置并链接 GitHub）、深入解析（含数学推导与
辅助示意图）、6 道练习（附提示与答案）、权威来源。

## 信源权重

1. **源码**（最高权重）：MagiAttention main 分支 2026-08 快照，代码块为忠实摘录，
   仅省略 debug 打印与其它架构的静态编译分支（以 `# …` 标注）。
2. 官方博客/文档：AttnSlice 表示、FFA_FA4/HSTU、attention sink 等设计动机。
3. 论文：MagiAttention (arXiv:2505.13211)、FlashAttention-2/3、online softmax。

注意：仓库存在两套 FFA 栈（functional/JIT 生产路径与 CuTe DSL 路径），课程以
CuTe DSL 的原生 SM100 kernel 为主角，并如实标注其当前兑现的语义子集
（cu_seqlens 等价 ranges、full/causal 两种 mask 类型）。

## 本地浏览

```bash
python3 -m http.server 8000
# 访问 http://localhost:8000/
```

学习进度只保存在浏览器 `localStorage`，不会上传。

## 校验

内容契约 + SVG 静态几何校验（连线不得穿过节点、平行轨道不得重叠、虚线必须带
标签、图中代码块 id 必须存在于 `assets/code.js`）：

```bash
node tools/validate.mjs
```

单图预览：`tools/preview.html?key=<章节id>` 或 `?aux=<辅图key>`。

## 文件结构

- `index.html` — 课程主页：学习地图、统计、章节卡片、来源
- `chapter.html?id=<id>` — 统一章节渲染页
- `assets/chapters.js` — 八章教学内容（takeaway/动机/解析/练习/来源）
- `assets/code.js` — 交互源码块（真实行号 + GitHub 深链）
- `assets/diagrams.js` — 8 张主架构图 + 7 张辅助示意图（JetBrains Mono、
  KaTeX 公式、内置几何校验）
- `assets/course.js` — 渲染器：目录、图-码联动工作台、进度
- `assets/styles.css` — editorial 设计系统（配色对齐调度示意图截图风格）
- `tools/validate.mjs` — 内容与几何校验器
- `tools/preview.html` — 单图预览harness
