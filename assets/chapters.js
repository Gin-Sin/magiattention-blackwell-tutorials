/* FFA on Blackwell — chapter contents.
 *
 * Bodies are hand-written HTML fragments. KaTeX runs over \( \) / \[ \]
 * delimiters after render. Facts follow the MagiAttention repository
 * (main branch, 2026-08 snapshot); file paths and line numbers refer to it.
 */
(function () {
  "use strict";

  var R = String.raw;

  window.MAGI_CHAPTERS = [
    /* ================================================================ *
     * 00 · AttnSlice
     * ================================================================ */
    {
      id: "attnslice",
      order: 0,
      title: "AttnSlice 契约",
      fullTitle: "The AttnSlice Contract",
      zhTitle: "算子对外承诺了什么",
      tag: "API · 语义",
      category: "dense",
      difficulty: "基础",
      source: "kernel/cutedsl/flex_flash_attn.py · 1537 行",
      deck: R`在读一行 kernel 代码之前，先弄清这个算子对外的承诺：任意注意力 mask 被分解为 <code>AttnSlice = (QRange, KRange, MaskType)</code> 三元组的集合；输出 <code>out</code> 与 <code>lse</code> 携带可累加、可合并的数学结构。Blackwell 上的 CuTe DSL kernel 是这份契约的最新实现——本章同时讲清它当前兑现了契约的哪个子集。`,
      takeaway: R`FFA 的本质是一次「语义降维」：把任意形状的 mask 拆成矩形切片 \(\mathrm{AttnSlice}=(Q\text{Range},K\text{Range},\text{MaskType})\)，每个切片只需 4 种基本几何（FULL / CAUSAL / INV-CAUSAL / BI-CAUSAL）。而让切片之间可以独立计算、事后合并的钥匙，是 <strong>fp32 累加的 out + 可合并的 LSE</strong>——这也是它能撑起分布式注意力的根本原因。`,
      intuitions: [
        { label: "分解", title: "任意 mask = 矩形切片之并", body: R`每个切片是一块连续的 2D Q–K 区域，内部只有一种简单几何。切片列表在 CP rank 间重新分配后依然合法。` },
        { label: "合并", title: "输出是半群", body: R`两个 partial 结果 \((out_1, lse_1)\)、\((out_2, lse_2)\) 可用封闭公式合并，与计算顺序无关（数学上）。` },
        { label: "现状", title: "Blackwell 兑现子集", body: "CuTe DSL SM100 kernel 目前吃 cu_seqlens 等价的 ranges + 单一 full/causal，完整四类型靠上层切片分解与 SM90 JIT 路径。" }
      ],
      motivation: [
        R`FlashAttention 系列 kernel 假设 mask 结构规则（causal、滑窗、varlen），对<strong>不规则、跨 rank 分布的 mask</strong>（例如 Magi-1 的 varlen block-causal 视频掩码）会产生碎片化、负载不均和多余通信。MagiAttention 的答案是 Flex-Flash-Attention（FFA）：不是给 kernel 加更多 mask 分支，而是改变 mask 的<strong>表示</strong>。`,
        R`仓库里有两套并行的 FFA 栈，教学时必须分清。<strong>生产主路径</strong>在 <code>magi_attention/functional/flex_flash_attn.py</code>，走 C++/CUTLASS JIT（SM90），支持任意重叠 ranges、四种 mask 类型、atomic 归约与 range merge。<strong>CuTe DSL 路径</strong>在 <code>magi_attention/kernel/cutedsl/</code>，源自 FA4/CUTLASS Blackwell FMHA 示例，是原生 SM100 kernel 所在地，也是本课程的主角。`,
        R`本章交互源码全部来自 CuTe DSL 的 host 侧入口 <code>_flex_flash_attn_fwd</code>：从 ranges 折叠、形状检查、tile/2-CTA/CLC 启发式，到 compile_key 与 tvm-ffi 启动。读懂它，就拿到了后面 7 章所有静态配置的来源。`
      ],
      diagram: {
        key: "attnslice",
        caption: "AttnSlice 契约与 host 侧一次前向的旅程：ranges 折叠 → 启发式决策 → compile_key → SM100 kernel 实例化 → tvm-ffi 启动。点击节点查看对应源码。"
      },
      explain: [
        {
          title: "AttnSlice 三元组与四种 mask 几何",
          body: [
            R`公共枚举 <code>AttnMaskType</code>（<code>magi_attention/common/enum.py</code>）定义了四种切片内几何：<code>FULL=0</code>、<code>CAUSAL=1</code>、<code>INVCAUSAL=2</code>、<code>BICAUSAL=3</code>。关键约定是<strong>对齐方向</strong>：CAUSAL 是「右下对齐」的下三角（Q 末端与 K 末端对齐），INV-CAUSAL 是「左上对齐」的上三角，BI-CAUSAL 是两者之交。`,
            R`这套约定使得 \(s_q \ne s_k\) 时语义依然唯一：当 \(s_q < s_k\)，CAUSAL 切片是一个梯形；当 \(s_q > s_k\)，BI-CAUSAL 可能为空集。滑动窗口等「按行收缩」的 mask 用 FULL+CAUSAL+INV-CAUSAL 的组合即可紧凑表达，不需要逐行枚举。`,
            R`任意 mask 的表达式：给定切片集合 \(\{(Q_i, K_i, T_i)\}\)，全局 mask \(M[q,k]=1\) 当且仅当存在 \(i\) 使 \(q\in Q_i,\ k\in K_i\)，且 \((q,k)\) 落在 \(T_i\) 的几何内。切片可以重叠——这正是需要「可累加输出」的原因。`
          ],
          svg: "attnslice-masktypes",
          formula: R`<p>以 CAUSAL 为例（右下对齐），切片内合法条件为</p>
\[ k - k_{\mathrm{start}} \;\le\; (q - q_{\mathrm{start}}) + (s_k - s_q), \qquad s_q = |Q_i|,\; s_k = |K_i| . \]
<p>INV-CAUSAL（左上对齐）为 \(k - k_{\mathrm{start}} \ge q - q_{\mathrm{start}}\)，BI-CAUSAL 取两式之交。四种类型都是「每行一个连续 K 区间」，区间端点随 \(q\) 线性移动——这是 kernel 能高效处理它们的根本原因。</p>`
        },
        {
          title: "out / lse 的可累加语义",
          body: [
            R`同一个 Q token 可能出现在多个切片（重叠 ranges）或多个 CP rank 的 partial attention 中，各处算出的是「局部 softmax」结果。要合并它们，kernel 必须输出两样东西：fp32 的 <code>out</code>（开 atomic reduction 时）和永远为 fp32 的 <code>lse</code>（log-sum-exp，空集初值 \(-\infty\)）。`,
            R`functional 层的默认策略：只要 Q ranges 可能重叠，就把 <code>out_type</code> 设为 <code>torch.float32</code> 并用 atomicAdd 归约；调用方显式声明「Q 不重叠」（<code>disable_fwd_atomic_reduction=True</code>）才允许按输入 dtype 直写。CuTe DSL 路径当前因为 ranges 等价于 cu_seqlens（天然不重叠），直接按 <code>q.dtype</code> 写出。`,
            R`Block 03 展示了 cutedsl 侧的另一处契约细节：空输入时 <code>out.zero_()</code>、<code>lse.fill_(-inf)</code>——\(-\infty\) 正是「对空集合做 log-sum-exp」的正确单位元，保证后续合并公式无需特判。`
          ],
          formula: R`<p>两段 partial 结果的合并（<code>magi_attention/functional/utils.py</code>，<code>correct_attn_out_lse</code>）：</p>
\[ \mathrm{lse} = \log\!\big(e^{\mathrm{lse}_1} + e^{\mathrm{lse}_2}\big) = \max(\mathrm{lse}_1,\mathrm{lse}_2) + \operatorname{softplus}\!\big(\min - \max\big), \]
\[ w_i = e^{\mathrm{lse}_i - \mathrm{lse}}, \qquad \mathrm{out} = w_1\,\mathrm{out}_1 + w_2\,\mathrm{out}_2 . \]
<p>由于 \(w_1 + w_2 = 1\) 且运算满足交换律与结合律，任意多段 partial attention 都能以任意顺序、任意分组合并——这就是「输出是半群」的精确含义，也是 MagiAttention 分布式 GroupReduce 的数学基础。</p>`
        },
        {
          title: "两套 FFA 栈：JIT 生产路径 vs CuTe DSL 路径",
          body: [
            R`<strong>functional/JIT 路径</strong>（<code>magi_attention/functional/flex_flash_attn.py</code>，SM90 C++ kernel）：支持任意重叠 <code>q_ranges/k_ranges</code> + 每 range 独立的 <code>attn_type_map</code>（0/1/2/3）、<code>merge_ranges</code> 自动去重（产出 <code>merge_q_ranges / fwd_qk_map / fwd_unique_count</code>）、三个 atomic 开关、<code>sm_margin</code> 给通信 kernel 留 SM。这是 MagiAttention CP 训练的主力。`,
            R`<strong>CuTe DSL 路径</strong>（本课程主角）：源码注释直言这是 "Step-1 hack"——<code>ranges_to_cu_seqlens</code> 要求 ranges 从 0 起、连续、不重叠；<code>MT_MAP</code> 只有 <code>full=0, causal=1</code>，<code>inv_causal/bi_causal</code> 留着 TODO。作为交换，它拿到了 SM90 JIT 没有的东西：原生 SM100 kernel、FlexAttention 风格的 <code>score_mod/mask_mod</code>、CSR 化的 block-sparse 表、PackGQA 与 PagedKV。`,
            R`历史脉络：v1.1.0 时代 Blackwell 支持靠 <code>FFA_FA4</code> 后端——fork 的 Flash-Attention 4 加上 HSTU Function 表示（把每行的合法 K 区间编码为分段函数）。in-tree 的 cutedsl SM100 kernel 是替代它的原生方案，正在逐步长出完整 AttnSlice 语义。`
          ]
        },
        {
          title: "host 侧一次前向的旅程",
          body: [
            R`Block 04–07 串起完整决策链。<strong>mask 折叠</strong>：单一 <code>mask_type</code> 折叠成 <code>causal</code> 布尔量供启发式使用。<strong>tile 选择</strong>：SM100 固定 <code>tile_m = tile_n = 128</code>；<code>q_stage = 2</code>（当 \(s_q^{\mathrm{packgqa}} > 128\)）意味着一个 CTA 同时处理两个 128 行的 Q 子块。<strong>2-CTA 判定</strong>：非 causal、非 varlen、非稀疏、head_dim padded 到 128/192 且 head_dim_v=128、序列足够长时启用（第 1 章详解）。<strong>CLC 判定</strong>：varlen-MHA 与 dense-noncausal 场景实测回退，被 host 侧启发式排除（第 6 章详解）。`,
            R`<strong>compile_key</strong> 是理解 CuTe DSL 工作方式的钥匙：dtype、head_dim、mask_type、tile、q_stage、pack_gqa、arch、2-CTA、CLC……所有会改变生成代码的量都进 key。同一 key 首次调用触发 <code>cute.compile(..., options="--enable-tvm-ffi")</code>，之后走缓存直接启动——tvm-ffi 消除了每次调用的 <code>from_dlpack</code> 元数据转换开销。这也解释了官方文档为何建议生产前预编译常见形状。`
          ]
        }
      ],
      warning: R`不要把「FFA 支持四种 mask 类型」理解为「Blackwell kernel 内部有四个分支」。当前 CuTe DSL kernel 只认 full/causal 两个整数；INV/BI-CAUSAL 由上层切片分解或 SM90 JIT 路径承接。引用能力表时务必注明是哪条栈。`,
      exercises: [
        {
          kind: "概念", level: "基础",
          q: R`一个 varlen causal mask（3 条序列，长度 512/1024/256，各自内部 causal）最少需要几个 AttnSlice？写出每个切片的 mask_type。`,
          hint: R`每条序列内部是一个右下对齐的 causal 矩形。`,
          answer: R`3 个切片：\((Q_i, K_i, \text{CAUSAL})\)，其中 \(Q_i = K_i\) 为每条序列自身的 range。varlen causal 恰是「每序列一个 CAUSAL 切片」的特例，这也是它能折叠成 cu_seqlens 的原因。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`证明 LSE 合并公式的数值安全性：为什么实现写成 \(\max + \operatorname{softplus}(\min-\max)\) 而不是直接 \(\log(e^{l_1}+e^{l_2})\)？当 \(l_1 = -\infty\) 时会发生什么？`,
          hint: R`考虑 \(l_i \approx 80\)（bf16 训练常见量级）时 \(e^{l_i}\) 的表示范围，以及 softplus 在 \(-\infty\) 处的取值。`,
          answer: R`\(e^{80}\approx 5.5\times10^{34}\) 已超出 fp32 上限（\(\sim3.4\times10^{38}\) 边缘），双段相加更容易溢出。改写后指数项恒为 \(e^{\min-\max}\le 1\)，绝不溢出。当 \(l_1=-\infty\)（空集）时 \(\operatorname{softplus}(-\infty)=0\)，公式退化为 \(\mathrm{lse}=l_2\)，即空集是合并的单位元——源码里 <code>safe_subtract</code> 专门保证 \(-\infty - (-\infty)\) 不产生 NaN。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`设一个 Q token 被 3 个重叠切片覆盖，各切片 partial 结果为 \((out_i, lse_i)\)。写出合并后的 \(out\)，并证明它等于「把三个切片的 K 集合并起来做一次完整 softmax」的结果（假设 K 集合两两不相交）。`,
          hint: R`把每个 \(out_i\) 写成 \(\frac{\sum_{k\in K_i} e^{s_k} v_k}{e^{lse_i}}\)。`,
          answer: R`\(out = \sum_i w_i\, out_i\)，\(w_i = e^{lse_i - lse}\)，\(lse = \log\sum_i e^{lse_i}\)。代入 \(out_i = e^{-lse_i}\sum_{k\in K_i} e^{s_k}v_k\) 得 \(out = e^{-lse}\sum_i \sum_{k\in K_i} e^{s_k} v_k\)，而 \(e^{lse} = \sum_i e^{lse_i} = \sum_{k\in\cup K_i} e^{s_k}\)。两者合起来正是全集 softmax 的定义。K 集不相交保证分子分母都不重复计数。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`compile_key 里为什么要放 <code>lse is None</code> 和 <code>sink is not None</code> 这类「张量是否存在」的布尔量,而不是张量本身？`,
          hint: R`想想 CuTe DSL 编译时 Optional 参数会发生什么。`,
          answer: R`它们是编译期分支（<code>const_expr(mLSE is not None)</code>）：LSE/sink 存在与否会改变生成的 kernel 代码（是否写 LSE、是否在尾声并入 sink 项）。张量的值不影响代码结构，但「有没有」影响，所以只有存在性进 key。同理 <code>cu_seqlens_q is None</code> 决定 varlen 分支的取舍。`
        },
        {
          kind: "系统", level: "挑战",
          q: R`functional 层在 Q ranges 重叠时强制 <code>out_type=fp32</code> + atomicAdd。分析：若允许 bf16 atomicAdd 直写,会引入哪两类问题？`,
          hint: R`一类关于硬件,一类关于数值。`,
          answer: R`(1) 数值：bf16 只有 8 位尾数,多个 partial 输出量级相近时逐次舍入误差累积,且加法顺序不定使误差不可复现;fp32 累加把舍入推迟到最后一次 cast。(2) 硬件/语义：bf16 的 atomic add 支持面窄（常需 CAS 模拟或 2 元素打包）,吞吐差且实现复杂。fp32 累加是用 2 倍显存换正确性与可移植性的经典折衷。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`如果要把 INV-CAUSAL 原生加进 SM100 kernel,除了 <code>MT_MAP</code> 加一个枚举值,至少还要动哪三处?（提示：分别在 BlockInfo、AttentionMask、softmax_loop）`,
          hint: R`INV-CAUSAL 的合法区随行号右移的是「左边界」。`,
          answer: R`(1) <code>BlockInfo.get_n_block_min_max</code>：INV-CAUSAL 收缩的是 n_block_min（左界随行增大）,需要新的跳块公式;(2) <code>AttentionMask.apply_mask_sm100</code>：加 col_limit_left 随行移动的分支（可复用 local mask 的双界 R2P）;(3) <code>softmax_loop</code> 的三段循环：partial 区出现在 KV 序列的「前端」而非「尾端」,遍历方向与提前释放逻辑要对称改写。这正好是第 3 章三层防线各改一层。`
        }
      ],
      sources: [
        { label: "MagiAttention 博客 · AttnSlice Representation 与四种 mask 类型", url: "https://sandai-org.github.io/MagiAttention/docs/blog/magi_attn/" },
        { label: "MagiAttention 论文 (arXiv:2505.13211)", url: "https://arxiv.org/abs/2505.13211" },
        { label: "源码 · kernel/cutedsl/flex_flash_attn.py（host 入口与 compile cache）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/flex_flash_attn.py" },
        { label: "源码 · functional/flex_flash_attn.py（生产 JIT 路径,四类型/atomic/merge）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/functional/flex_flash_attn.py" },
        { label: "官方博客 · Support Blackwell with FFA_FA4 Backend（HSTU 表示的历史方案）", url: "https://sandai-org.github.io/MagiAttention/docs/blog/blackwell_ffa_fa4/" }
      ]
    },

    /* ================================================================ *
     * 01 · Blackwell substrate
     * ================================================================ */
    {
      id: "blackwell",
      order: 1,
      title: "TMEM / UMMA / TMA",
      fullTitle: "The Blackwell Execution Substrate",
      zhTitle: "硬件给了 kernel 什么",
      tag: "SM100 · 硬件",
      category: "sparse",
      difficulty: "进阶",
      source: "kernel/cutedsl/ffa_fwd_sm100.py · __init__/__call__",
      deck: R`Hopper 的注意力 kernel 把累加器放在寄存器堆里；Blackwell 把它搬进了一块全新的片上存储 <strong>TMEM（Tensor Memory）</strong>，由 <strong>tcgen05</strong>（第五代 Tensor Core，俗称 UMMA）直接读写。理解 TMEM 的 512 列地图、单 warp 发射的 MMA、以及 TMA 的字节计账，是读懂整个 kernel 的前提。`,
      takeaway: R`Blackwell 给矩阵计算加了「第三层片上存储」：SMEM 喂数据，<strong>TMEM 存累加器</strong>（每 CTA 128 行 × 512 列 × 4B），tcgen05 MMA 由<strong>单个 warp 异步发射</strong>、完成靠 mbarrier 通知。于是「算」不再占用任何计算 warp 的寄存器——寄存器全部让给 softmax。FFA 把 S、P、O 全部塞进这 512 列，其中 <strong>P 直接叠放在 S 的空间上</strong>（bf16 只占 fp32 一半宽）。`,
      intuitions: [
        { label: "存储", title: "TMEM 是累加器的家", body: R`\(S = QK^\top\) 与 \(O = PV\) 的 fp32 累加器都住在 TMEM,MMA 写、T2R 读、R2T 改,不经过寄存器堆分配。` },
        { label: "计算", title: "MMA 只需一个 warp", body: "tcgen05 指令单线程发射、异步执行,一个 warp 就能喂满两个 CTA 的 Tensor Core;其余 15 个 warp 干别的。" },
        { label: "搬运", title: "TMA 按字节记账", body: R`bulk tensor copy 由硬件搬运,producer 只声明 tx_count 字节数,mbarrier 在数据到齐时自动翻转。` }
      ],
      motivation: [
        R`FlashAttention-3 在 Hopper 上的核心矛盾：WGMMA 的累加器占用寄存器,softmax 也要寄存器,两者互相挤压,只能靠 warpgroup 间乒乓调度缓解。Blackwell 的答案是架构级的：给累加器单独建一块存储（TMEM）,让 MMA 从「occupying-register 的集体操作」变成「单线程发射的异步引擎」。`,
        R`代价是显式管理:TMEM 只有 512 列（每列 128 行 × 32bit）,分配/释放要走 <code>tcgen05.alloc</code> 专用指令,读写要用专门的 T2R/R2T copy 原子,跨 warp 可见性要靠 <code>fence_view_async_tmem_*</code>。FFA 前向把这 512 列规划得严丝合缝——本章 Block 02 的 20 行代码就是整个 kernel 的「内存地图」。`,
        R`2-CTA 模式（<code>use_2cta_instrs</code>）再翻一倍:两个 CTA 组成 cluster,tcgen05 以 <code>CtaGroup.TWO</code> 发射,一次 MMA 覆盖 256 行,K/V 的 SMEM 各存一半、互相通过 DSMEM 读取。host 侧的启用条件（非 causal、非 varlen、head_dim ∈ {128,192}、序列够长）本质上都在回答一个问题:两个 CTA 的负载是否天然对称。`
      ],
      diagram: {
        key: "blackwell",
        caption: "SM100 执行基座：TMA 把 Q/K/V 搬进 SMEM，tcgen05 UMMA 消费 SMEM/TMEM 并把累加器写进 TMEM，T2R/R2T 供 softmax/correction warp 访问，epilogue 经 SMEM 用 TMA 写回。点击节点查看对应源码。"
        },
      explain: [
        {
          title: "TMEM 512 列地图",
          body: [
            R`Block 02 的规划（tileK=128、head_dim_v=128、q_stage=2 时）：S0 占列 [0,128)、S1 占 [128,256)、O0 占 [256,384)、O1 占 [384,512)，总共恰好 512 列。每个「列」是 128 行 × 32bit，正好对应 MMA tile 的一列 fp32 累加。`,
            R`最精巧的是 P 的安置：softmax 产出的 P 是 bf16，宽度只有 fp32 一半，所以 <code>tmem_p_offset = tmem_s_offset + 64</code>——P0 以 bf16 视图叠放在 S0 的后半空间 [64,192)。这不是巧合而是时序保证：softmax 先把 S 整块读进寄存器（T2R），再把 exp2 后的 P 写回（R2T），读写天然错开。row_max/row_sum 的向量缓冲同样复用 S 的空间（softmax 之后 S 不再被需要）。`,
            R`这张地图直接决定了第 2 章的流水线结构：S 槽位的「满/空」就是 MMA 与 softmax 之间的接力棒，O 槽位的「rescale 完成」就是 MMA 与 correction 之间的接力棒。`
          ],
          svg: "tmem-map"
        },
        {
          title: "tcgen05 UMMA：两个 GEMM、一种发射方式",
          body: [
            R`Block 03 用 <code>make_trivial_tiled_mma</code> 构造两个 TiledMMA：QK GEMM 的 A/B 都来自 SMEM；PV GEMM 的 A（即 P）声明 <code>OperandSource.TMEM</code>——tcgen05 可以直接从 TMEM 读 A 操作数，P 根本不用去 SMEM 绕一圈。注释里的 MMA Atom 形状 <code>(256,128,16)</code> 是 2-CTA 模式：ThrID 2:1 表示两个 CTA 各贡献一半。`,
            R`发射模型与 Hopper 截然不同：WGMMA 需要整个 warpgroup（128 线程）同步参与;tcgen05 由 MMA warp 中<strong>被选举的单个线程</strong>发射,指令进入异步队列,完成事件写到指定 mbarrier。这就是第 2 章「1 个 MMA warp 服务 15 个其他 warp」的硬件基础。`,
            R`2-CTA 下只有 leader CTA（cluster 内 rank 0）发射 UMMA,peer CTA 的 SMEM 通过 cluster 的分布式共享内存被读取;两个 CTA 各自拿到自己 128 行的累加结果。`
          ],
          formula: R`<p>一个 Q tile（q_stage=2 展开前的单 stage）在 TMEM 里完成的计算：</p>
\[ \underbrace{S_i}_{\mathrm{TMEM}[0,128)} = \underbrace{Q_i}_{\mathrm{SMEM}} \underbrace{K_j^{\mathsf T}}_{\mathrm{SMEM}}, \qquad \underbrace{O_i}_{\mathrm{TMEM}[256,384)} \mathrel{+}= \underbrace{P_{ij}}_{\mathrm{TMEM}[64,192)} \underbrace{V_j}_{\mathrm{SMEM}} . \]
<p>其中 \(P_{ij}\) 由 softmax warp 以 bf16 写回。O 在整个 KV 循环中驻留 TMEM 持续累加，只在末尾被 correction 读出一次。</p>`
        },
        {
          title: "TMA：搬运即记账",
          body: [
            R`Block 05 为 Q/K/V 各建一个 G2S TMA atom（<code>CopyBulkTensorTileG2SOp(cta_group)</code>），为 O 建一个 S2G atom。TMA 是描述符驱动的硬件引擎：kernel 里一条指令声明「把第 (i,j) 个 tile 搬到这个 SMEM 地址」，地址计算、边界裁剪、swizzle 全部由硬件完成。`,
            R`与流水线的接口是「事务字节数」：<code>tma_copy_bytes["K"]</code> 在 host 侧静态算出（Block 04 的 smem layout 尺寸 × cta_group_size），producer 发起 TMA 时把它挂到 mbarrier 上，硬件每搬到一批字节就向 mbarrier 记账，字节数到齐 barrier 自动翻转——消费者（MMA warp）看到的就是「sK[stage] 满了」。`,
            R`注意 K 与 V 复用同一块物理 SMEM（Block 04 里 <code>sV</code> 用 <code>recast_ptr</code> 指向 <code>sK</code> 的空间加偏移）,配合 kv_stage 环形缓冲：SMEM 预算 = 224KB 总量减去 Q/O 占用,除以单 stage 的 max(K,V) 尺寸,得出能开几级流水。`
          ]
        },
        {
          title: "TMEM 的生命周期管理",
          body: [
            R`Block 07：只有 MMA warp 执行 <code>tmem.allocate(512)</code>。分配结果（基地址）写到 SMEM 的约定位置,其他 warp 通过 <code>TmemPtr</code> named barrier 等待后 retrieve。释放更讲究:softmax 和 correction warp 完成各自最后一次 TMEM 访问后 arrive 同一个 barrier,MMA warp 等齐三方才敢 <code>dealloc</code>——否则会出现「一边释放一边还有 warp 在读」的竞态。`,
            R`这个模式值得记住:TMEM 没有自动生命周期,任何「谁分配、谁使用、谁见证释放」都要用 barrier 显式编排。第 2 章的 <code>tmem_alloc_barrier</code>（线程数 = mma + softmax×2 + correction 的总线程数）就是这套约定的实现。`
          ]
        }
      ],
      warning: R`TMEM 不是「更大的寄存器堆」也不是「另一块 SMEM」：它只能被 tcgen05 MMA 和专用 T2R/R2T copy 指令访问，普通 load/store 碰不到它；每 CTA 至多 512 列，且分配必须整块进行。把它理解为「Tensor Core 的专属累加器仓库」最不容易出错。`,
      exercises: [
        {
          kind: "计算", level: "基础",
          q: R`按 Block 02 的规划（tile 128×128、head_dim_v=128、q_stage=2）,验证 TMEM 恰好用满 512 列;若 head_dim_v=64,还剩多少列?`,
          hint: R`\(\text{total} = 2\times(\text{tileK} + \text{hd}_v)\)。`,
          answer: R`\(2\times(128+128)=512\),恰好用满。head_dim_v=64 时 \(2\times(128+64)=384\),剩 128 列。源码断言 <code>tmem_total <= tmem_alloc_cols</code>——这也解释了为何 head_dim 更大的配置需要不同的 TMEM 布局甚至砍 q_stage。`
        },
        {
          kind: "概念", level: "基础",
          q: R`为什么 P 能以 bf16 视图叠放在 S 的 TMEM 空间上而不产生数据竞争？给出时序上的理由。`,
          hint: R`看 softmax_step 里 T2R 和 R2T 的先后。`,
          answer: R`softmax 对每个 KV block 的处理是严格串行的：先把整块 S 从 TMEM 读进寄存器（T2R），再在寄存器里做 exp2，最后把 bf16 的 P 写回（R2T）。写 P 时 S 的值已全部离开 TMEM，且 P 只占 S 空间的一半宽（bf16 vs fp32），所以物理上是「先读空、再写入」的安全复用。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`2-CTA 模式下一次 QK UMMA 的有效吞吐是单 CTA 的两倍,但 K/V 的 SMEM 占用为什么反而减半?这如何影响 kv_stage?`,
          hint: R`MMA tiler M 翻倍,但每个 CTA 只存 B 操作数的一半。`,
          answer: R`2-CTA 的 MMA tiler 是 (256,128,·),两个 CTA 各出 128 行 Q;而 B 操作数（K/V tile）在 cluster 内切成两半、每 CTA 存一半（源码 <code>smem_size_kv_per_stage // cta_group_size</code>）。于是每 CTA 的 KV SMEM 减半,同样的 224KB 预算能开更多 kv_stage,流水更深——这是 2-CTA 除了 MMA 效率外的第二重收益。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`host 侧启用 2-CTA 要求「非 causal」。从负载对称性角度解释这条限制。`,
          hint: R`2-CTA 的两个 CTA 分到的是同一个 M tile 的上下两半。`,
          answer: R`2-CTA 把 M 方向 256 行绑成一个 MMA:上半 128 行给 CTA0,下半给 CTA1,两者必须迭代<strong>同一组</strong> KV block。causal 下上半行的合法 KV 少于下半行,同组迭代意味着上半 CTA 在大量 block 上空转 mask,负载天然不对称;非 causal 时两半负载完全相同。所以 causal 用 1-CTA + LPT 调度处理倾斜更划算。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`TMA 的 tx_count 记账为什么必须在 host 侧静态算好,而不能运行时按实际搬运量填?`,
          hint: R`想想 mbarrier 的 expected-tx 语义与编译期布局。`,
          answer: R`mbarrier 的 expect-tx 在 producer acquire 时一次性设置,硬件按到达字节数递减;若运行时才知道字节数,producer 无法在发起 TMA 前正确设置期望值。而 FFA 的 tile 尺寸、dtype、stage 布局全是编译期常量,每次 TMA 恰好搬一个完整 tile,字节数是纯静态量——变长部分（序列尾部）由 TMA 描述符的边界裁剪处理,搬运字节数不变,只是越界部分填充。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`假设未来出现 head_dim_v=256 的模型,q_stage=2 时 O 需要 2×256=512 列,S 还要 2×128=256 列,TMEM 装不下。参考本章内容,提出两种可行的重新规划方案并比较代价。`,
          hint: R`一种动 q_stage,一种动 O 的驻留策略。`,
          answer: R`方案 A：q_stage=1——S 128 列 + O 256 列 = 384 列,可行;代价是失去双 Q-stage 的 softmax/MMA 交错,流水变浅（源码正是这样处理大 head_dim 的:hd256 只在特定配置支持）。方案 B：O 分块驻留——把 O 沿 head_dim 切两半,PV GEMM 分两次做,每次只占 128 列;代价是 V 要读两遍或 P 要读两遍,增加 TMEM↔SMEM 流量。实践中 A 更常见,因为 O 的累加语义跨 KV 迭代,分块会把 correction 也复杂化。`
        }
      ],
      sources: [
        { label: "NVIDIA PTX ISA · tcgen05 指令族与 Tensor Memory", url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-family-instructions" },
        { label: "CUTLASS Example 77 · Blackwell FMHA（本 kernel 的直接蓝本）", url: "https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha" },
        { label: "源码 · ffa_fwd_sm100.py __init__（TMEM 规划与 MMA tiler）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/ffa_fwd_sm100.py" },
        { label: "CUTLASS CuTe DSL 文档（TiledMMA / TMA atom 的编程模型）", url: "https://docs.nvidia.com/cutlass/latest/" },
        { label: "FlashAttention-3 论文（Hopper 时代的寄存器矛盾,对照阅读）", url: "https://arxiv.org/abs/2407.08608" }
      ]
    },

    /* ================================================================ *
     * 02 · Warp specialization & pipelines
     * ================================================================ */
    {
      id: "pipeline",
      order: 2,
      title: "Warp 特化流水线",
      fullTitle: "Warp Specialization & Pipelines",
      zhTitle: "16 个 warp 的分工协作",
      tag: "SM100 · 骨架",
      category: "hybrid",
      difficulty: "进阶",
      source: "kernel/cutedsl/ffa_fwd_sm100.py · kernel()",
      deck: R`一个 CTA 的 512 个线程被切成 7 种角色：8 个 softmax warp、4 个 correction warp、1 个 MMA warp、1 个 load warp、1 个 epilogue warp、1 个 empty/CLC warp。它们共享一份 kernel 代码，靠 <code>warp_idx</code> 分流，用六条 mbarrier 流水线和一组 named barrier 接力。这是整个前向 kernel 的骨架。`,
      takeaway: R`Warp specialization 的本质是<strong>把寄存器和延迟按角色重新分配</strong>：softmax 是唯一的重计算角色，独享 176–192 个寄存器/线程；load/MMA/epilogue 是「发射即忘」的异步引擎驱动者，只留 48–80 个。角色之间不共享数据结构，只交换<strong>槽位的满/空状态</strong>——其中最精妙的一条流水线 <code>pipeline_s_p_o</code> 让同一个 mbarrier 槽位track 两种语义转移：S-full（MMA→softmax）与 P+O-empty（softmax+correction→MMA）。`,
      intuitions: [
        { label: "分工", title: "流水线工厂", body: "load 进料、MMA 冲压、softmax 精加工、correction 校准、epilogue 打包出货——每个工位只做一件事,靠传送带（mbarrier）衔接。" },
        { label: "资源", title: "512 寄存器的账本", body: R`\(512 = 2\times\text{softmax} + \text{correction} + \text{other}\)。setmaxregister 在运行时把寄存器从闲角色转移给 softmax。` },
        { label: "同步", title: "满/空是唯一语言", body: "warp 之间不传数据指针,只翻转 mbarrier 相位;数据永远在约定好的 TMEM/SMEM 槽位里。" }
      ],
      motivation: [
        R`FlashAttention-3 在 Hopper 上确立了「producer/consumer warpgroup」范式;SM100 把它推到极致:MMA 不再消耗任何 warpgroup 的寄存器（tcgen05 单线程发射）,于是角色可以切得更碎、更专。FFA 前向的 16 个 warp 里,真正「算数」的只有 softmax 的 8 个;其余 8 个全在编排数据流。`,
        R`角色表不是固定的（Block 01 后半）:q_stage=1 时 softmax1 整组让位;paged KV 非 TMA 时 load 扩成 2 个 warp;varlen 时 correction 兼任 epilogue,原 epilogue warp 变 empty。这种「按配置重排角色」是用一份代码覆盖多种形态的关键——所有分支都是 <code>const_expr</code> 编译期决议,运行时零开销。`,
        R`本章聚焦一条主线:一个 KV block 从 TMA 进 SMEM,到 S 进 TMEM,到 P 回 TMEM,到 O 累加、被 rescale,数据每换一次主人,靠哪条流水线交接。`
      ],
      diagram: {
        key: "pipeline",
        caption: "前向 kernel 的角色分工与流水线拓扑：左列为 warp 角色（含寄存器配额），右侧为它们之间的六条 mbarrier 流水线。点击节点查看源码。"
      },
      explain: [
        {
          title: "角色表与寄存器账本",
          body: [
            R`默认布局（Block 01）:warp 0–3 = softmax0（服务 Q-stage 0）,warp 4–7 = softmax1（Q-stage 1）,warp 8–11 = correction,warp 12 = MMA,warp 13 = epilogue,warp 14 = load,warp 15 = empty（CLC 开启时兼任调度 producer）。`,
            R`寄存器账本（Block 02）:SM100 每线程静态上限 512 个寄存器（按 warpgroup 记账）。kernel 启动后,load/MMA/epilogue/empty 立刻 <code>setmaxregister_decrease(num_regs_other)</code>（如 48–80）,softmax 随后 <code>increase(176–192)</code>,correction 取中间值 64–88。调参表 <code>_TUNING_CONFIG</code> 按 (2CTA, causal, head_dim, SM103) 精确到每 8 个寄存器——softmax 的寄存器直接决定它一次能驻留多少 S 元素,是全 kernel 最敏感的资源。`,
            R`一个容易忽略的细节:降寄存器的角色必须先降、升寄存器的角色后升,否则瞬时总量超限。源码里 empty/load/MMA/epilogue 的 decrease 都写在各自分支的第一行。`
          ]
        },
        {
          title: "六条流水线的全景",
          body: [
            R`<strong>pipeline_q / pipeline_kv</strong>（Block 03,TMA→UMMA 型）:load warp 是 producer,靠 TMA 硬件对 mbarrier 记账（tx_count 字节）;MMA warp 是 consumer。Q 有 q_stage 个槽位,KV 的槽位数由 SMEM 预算决定（通常 3–8 级）,K 和 V 交替占用。`,
            R`<strong>pipeline_s_p_o</strong>（Block 04,UMMA→Async 型）:全 kernel 最核心的一条。producer 是 MMA warp:QK GEMM 完成即 commit（S-full）;consumer 是 softmax+correction 的<strong>联合体</strong>——同一槽位要等 softmax（P 已写回）和 correction（O 已 rescale）双双 release,MMA 才能发下一次 PV GEMM。把两种完成事件编进一个 barrier,是「一个槽位、双重语义」的经典设计。`,
            R`<strong>pipeline_p_lastsplit</strong>:配合 split_P_arrive 优化（见第 4 节）。<strong>pipeline_o_acc</strong>（Block 05）:只在每个 Q tile 的最后一个 KV block 使用——主循环里 correction 无需等 O（GEMM 顺序保证了 O(i-1) 先于 S(i) 完成）,唯独尾声这个保证断裂,才需要显式流水线。<strong>pipeline_sm_stats + sm_stats_barrier</strong>（Block 06）:保护 sScale 槽位的一对 WAR/RAW 拍档,细节留给第 5 章。`
          ],
          svg: "pipeline-wave"
        },
        {
          title: "一个 KV block 的完整旅程",
          body: [
            R`把六条流水线串起来,追踪 KV block j 在 Q-stage i 上的一生:① load warp acquire <code>pipeline_kv</code> 空槽 → 发 TMA → 硬件记账至满;② MMA warp 等 K 满 → 发 QK UMMA（写 TMEM S_i）→ commit <code>pipeline_s_p_o</code>(S-full);③ softmax_i warpgroup 等 S-full → T2R → mask/row_max/exp2 → R2T 写 P → release(P 侧);④ 同时 correction 读到 corr_scale 后 rescale O_i(j-1) → release(O 侧);⑤ MMA 集齐双 release → 发 PV UMMA(P_i × V_j 累加进 O_i) → release <code>pipeline_kv</code> 的 V 槽;⑥ load warp 见空槽,继续搬 j+2 的 K。`,
            R`注意其中没有任何一步「等待计算」:每个角色只等槽位状态,计算本身全部异步。当 KV 足够长时,TMA 搬运、两个 GEMM、softmax、correction 五者完全重叠——这正是把 mask/调度做对之后,SM100 能接近 Tensor Core 峰值的原因。`
          ]
        },
        {
          title: "split_P_arrive：P 写一半就开跑",
          body: [
            R`softmax 把 128 列 P 写回 TMEM 需要多次 R2T copy。<code>split_P_arrive = 96</code>（tileK 的 3/4,取 32 倍数）时,softmax 写完前 96 列就提前 release <code>pipeline_s_p_o</code>,MMA 立即发射 PV GEMM;GEMM 硬件读到 96 列之后,需要等 <code>pipeline_p_lastsplit</code> 的信号才读最后 32 列——这个信号由 softmax 写完全部 P 后 commit。`,
            R`效果是把「写 P 的尾巴」藏进「PV GEMM 的头部」。这是一个典型的 Blackwell 式优化:因为 GEMM 是硬件异步引擎,可以让它先启动、在中途等一个 mbarrier,软件流水线因此能切得比指令粒度还细。`
          ]
        },
        {
          title: "s0_s1_barrier：两组 softmax 的写口错峰",
          body: [
            R`q_stage=2 时两组 softmax warpgroup 分别服务 S0/S1,它们的 exp2+R2T 阶段会争抢同一个 TMEM 写口带宽。<code>s0_s1_barrier</code> 开启时,用一条 2-stage 的 pipeline 强制两组的「exp2→写 P」区段互相串行（各自的其余阶段仍并行）,把写口冲突变成交替占用。`,
            R`加上 correction 对 <code>pipeline_sm_stats</code> 的 cross-release（第 5 章详解）,整个 kernel 呈现出一种「宏观并行、微观错峰」的节奏:两组 softmax 相位相差半个 KV block,correction 恰好插在两者的空隙里。`
          ]
        }
      ],
      warning: R`不要把 named barrier 与 pipeline 混为一谈：pipeline（mbarrier）管理「槽位的满/空 + 相位翻转」，适合跨迭代的环形缓冲；named barrier 是「一组线程到齐即放行」的会合点，无相位概念。sScale 同时需要两者（RAW 用 barrier、WAR 用 pipeline），正是因为它一个槽位上有两种不同生命周期的冒险。`,
      exercises: [
        {
          kind: "计算", level: "基础",
          q: R`hd128 非 causal 2-CTA 配置下 num_regs_softmax=176, num_regs_correction=88。验证账本并算出 other 角色的配额。`,
          hint: R`\(512 = 2\times s + c + o\)。`,
          answer: R`\(o = 512 - 2\times176 - 88 = 72\)。即 load/MMA/epilogue/empty 每线程 72 个寄存器——对「只发指令不算数」的角色绰绰有余。`
        },
        {
          kind: "概念", level: "基础",
          q: R`为什么 pipeline_s_p_o 的 consumer group 必须同时包含 softmax 和 correction 两组线程？只让 softmax release 会发生什么？`,
          hint: R`MMA 发 PV GEMM 前需要确认哪两件事？`,
          answer: R`PV GEMM 会（1）读 P、（2）向 O 累加。P 就绪由 softmax 保证，但 O 槽位可写的前提是上一轮的 O 已被 correction rescale 完。若只等 softmax，MMA 可能在 correction 还在读改 O 时就发起累加，产生数据竞争。所以「空」的定义 = P 写完 ∧ O 校准完，consumer 联合体的 arrive 计数正好编码了这个合取。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`主循环里 correction 修正 O(i-1) 时不等任何 O 流水线（注释：GEMM ordering guarantee）。写出这个保证的完整推理链，并指出它在哪一步断裂、由哪条流水线补救。`,
          hint: R`同一 MMA warp 发射的 UMMA 按序完成。`,
          answer: R`推理链：corr_scale(i) 可读 ⇒ softmax 已 T2R 读完 S(i) ⇒ S(i) 的 QK GEMM 已完成 ⇒ 而同一 MMA warp 上 O(i-1) 的 PV GEMM 先于 S(i) 的 QK GEMM 发射且按序完成 ⇒ O(i-1) 必已写好。断裂点在尾声：最后一个 KV block 之后没有「下一个 S」来传递这个信号，row_sum 由 softmax 直接发布，此时 O(-1) 的 GEMM 可能还在跑——所以 correction 在 epilogue 必须显式 <code>pipeline_o_acc.consumer_wait</code>。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`估算 split_P_arrive 的收益上界：设写回 128 列 P 耗时 \(t_P\)，PV GEMM 耗时 \(t_G\)，split 点在 3/4 处。理想情况下每个 KV block 能省多少时间？`,
          hint: R`被隐藏的是写 P 的最后 1/4。`,
          answer: R`无 split 时串行段为 \(t_P + t_G\)；split 后 GEMM 在 \(\tfrac34 t_P\) 时刻启动，读完前 96 列（约 \(\tfrac34 t_G\)）后若最后 32 列已就绪则无缝继续。只要 \(\tfrac14 t_P \le \tfrac34 t_G\)（几乎恒成立），总时长为 \(\tfrac34 t_P + t_G\)，节省 \(\tfrac14 t_P\)。即收益上界是「写 P 尾巴」的完全隐藏，约为每 block 时长的几个百分点——在 softmax 是瓶颈的配置下相当可观。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`varlen（is_varlen_q）时为什么让 correction 兼任 epilogue（use_correction_warps_for_epi），而不是保留独立 epilogue warp？`,
          hint: R`varlen 的 O 写回还能用整块 TMA 吗？`,
          answer: R`varlen 下每个 tile 的有效行数不定，尾块需要按 seqlen 谓词逐行写出，TMA 整 tile store 不再适用，写回退化为普通 S2G copy——这需要一个完整 warpgroup 的线程宽度才够带宽。correction 本来就要把 O 从 TMEM 读出来做 rescale，顺手写 gmem 省去一次 SMEM 中转和一条流水线（pipeline_o_epi 直接不创建）；1 个 warp 的独立 epilogue 反而喂不饱。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`如果把 correction 角色取消,把 rescale 合并进 softmax warp（算完 corr_scale 顺手改 O）,分析对寄存器账本和关键路径的双重影响。`,
          hint: R`O tile 是 128×128 fp32;softmax 的寄存器已经是全场最紧的。`,
          answer: R`寄存器:rescale 需要把 O 分块 T2R 进寄存器,即便 16 列一批也要 16×fp32/线程的额外驻留,softmax 只能压缩 S 的驻留或降低 unroll,直接拖慢 exp2 吞吐。关键路径:rescale 会插在「读 S」与「写 P」之间,MMA 等待 P 的时间变长,pipeline_s_p_o 的空转增加;而独立 correction 让 rescale 与下一块的 softmax 完全并行。这正是 FFA 选择 4+4+4 三组分工而非 FA2 式单一角色的原因——SM100 的寄存器转移机制让「多养一组低配 warp」几乎免费。`
        }
      ],
      sources: [
        { label: "源码 · ffa_fwd_sm100.py kernel()（角色分派与全部流水线定义）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/ffa_fwd_sm100.py" },
        { label: "源码 · kernel/cutedsl/pipeline.py（PipelineTmaUmma/PipelineUmmaAsync 定制）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/pipeline.py" },
        { label: "源码 · kernel/cutedsl/named_barrier.py（barrier 编号表）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/named_barrier.py" },
        { label: "FlashAttention-3 论文（warp specialization 与异步化的方法论源头）", url: "https://arxiv.org/abs/2407.08608" },
        { label: "CUTLASS · Blackwell FMHA 示例（同款角色划分的参考实现）", url: "https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha" }
      ]
    },

    /* ================================================================ *
     * 03 · Block-level mask
     * ================================================================ */
    {
      id: "mask",
      order: 3,
      title: "块级 Mask",
      fullTitle: "Block-level Masking",
      zhTitle: "三层防线消化任意几何",
      tag: "SM100 · mask",
      category: "dense",
      difficulty: "进阶",
      source: "kernel/cutedsl/mask.py + block_info.py",
      deck: R`mask 处理的代价结构是分层的：整块跳过（免费）≫ 整块无 mask（近免费）≫ 位掩码 R2P（便宜）≫ 逐元素谓词（昂贵）。FFA 用 BlockInfo 做粗粒度跳块、用三段主循环把 partial 块限制在几何边界上、用 R2P 把边界块的掩码压成位运算——mask 的开销从此不再随 \(n_{\mathrm{func}}\) 线性爆炸。`,
      takeaway: R`一条端对齐不等式统治一切：\(k \le q + (s_k - s_q)\)。它在 <strong>BlockInfo</strong> 里被抬升到块坐标（决定哪些 n_block 根本不迭代），在 <strong>softmax_loop</strong> 里把 KV 迭代切成 partial→full→partial 三段（只有边界段付 mask 钱），在 <strong>AttentionMask</strong> 里被翻译成每行的 <code>col_limit</code> 并经 <strong>R2P</strong> 一次性写成 32 位掩码。同一几何、三个粒度、一处定义。`,
      intuitions: [
        { label: "分层", title: "先跳块，再挑块，最后改元素", body: "能不算就不算（跳块），能整块算就别逐元素判（full 段），必须判时用位掩码批量判（R2P）。" },
        { label: "几何", title: "一条不等式三处使用", body: R`端对齐 causal 的列界 \(k \le q + (s_k - s_q)\) 同时驱动跳块、分段与写 \(-\infty\)，不会「跳过头」也不会「漏 mask」。` },
        { label: "开销", title: "mask 不该出现在快路径", body: "full 段的 softmax_step 根本不传 mask_fn——编译期就没有 mask 代码，这才是块级稀疏的意义。" }
      ],
      motivation: [
        R`朴素做法是每个元素都判一次 mask——FA4 博客给过量化：flex mask 逐元素检查需要约 \(128\times n_{\mathrm{func}}\) 条比较指令/块，softmax warp 的延迟甚至会超过两个 GEMM 之和。mask 一旦进入内层循环，就是纯粹的税。`,
        R`FFA 的策略是把 mask 从「内层判断」改造成「外层几何」：causal/local 的合法区边界是行号的线性函数，因此(1)整块合法/整块非法可以用块角点判定；(2)部分合法的块只出现在边界带上，数量是 \(O(\text{tile 对角带宽})\) 而非 \(O(n^2)\)。`,
        R`剩下必须逐块处理的边界块，SM100 路径用 <strong>R2P（register-to-predicate）</strong>收尾：把 32 列的 keep/drop 打进一个 uint32，硬件 R2P 指令一次把 32 个位散到 32 个谓词寄存器，SEL 指令按谓词选择 \(-\infty\)。比较指令数从 \(O(128\, n_{\mathrm{func}})\) 降到 \(O(\lceil 128/32\rceil)\) 次位运算。`
      ],
      diagram: {
        key: "mask",
        caption: "三层防线：BlockInfo 决定 n_block 迭代范围（跳块），softmax_loop 三段循环隔离 partial 块，AttentionMask 在 partial 块内用 R2P/谓词写 -inf。点击节点查看源码。"
      },
      explain: [
        {
          title: "第一层：BlockInfo 跳块",
          body: [
            R`Block 01 的 <code>get_n_block_min_max</code>：对 Q tile \([m\cdot128, (m{+}1)\cdot128)\)，用 tile 内<strong>最大行号</strong>代入端对齐不等式得到 \(n_{\mathrm{idx}} = m_{\mathrm{idx}}^{\max} + (s_k - s_q)\)，向上取整除以 tile_n 就是 <code>n_block_max</code>——之后的 KV block 整块非法，循环压根不迭代。local mask 的左窗对称地给出 <code>n_block_min</code>。`,
            R`注意所有坐标都是「切片内相对坐标」：<code>seqlen_q/k</code> 来自 <code>SeqlenInfoQK</code>（每 tile 开头读一次 cu_seqlens 缓存所有长度/offset），mask 比较用相对索引，gmem 寻址用 offset——两套坐标由同一个数据结构对齐，这是 varlen 正确性的基石。`,
            R`反向有对称的 <code>get_m_block_min_max</code>（固定 n_block 反推 Q 范围），第 7 章会用到。`
          ]
        },
        {
          title: "第二层：三段主循环",
          body: [
            R`Block 02 是 softmax_loop 的骨架。以 causal 为例：<code>get_n_block_min_causal_local_mask</code> 算出「对角带」的起点——从 n_block_max 往回数，只有落在对角带内的块才可能 partial。于是循环切成：<strong>Mainloop-1</strong>（对角带，带 <code>mask_fn</code>，从右往左）→ <strong>Mainloop-2</strong>（full 区，不传 mask_fn，编译出的 softmax_step 里没有任何 mask 代码）→ <strong>Mainloop-3</strong>（仅 local mask 的左窗带，再带 mask_fn）。`,
            R`prologue 单独处理最右一块并带 <code>mask_seqlen=True</code>：序列尾部的 OOB 列（\(k \ge s_k\)）只可能出现在最后一个 KV block，其余块连 seqlen 检查都省了。这种「把检查压到唯一可能出错的块」的手法贯穿全 kernel。`,
            R`对照第 0 章：这正是 FFA_FA4 博客里 Full/Partial/Empty 三分类的原生版——Empty 被 BlockInfo 消灭，Full 走 Mainloop-2，Partial 走 1/3 段。block-sparse 路径则由外部 CSR 表直接给出 full/partial 块清单（Block 06 的 mask_mod 分支配合 <code>is_full_block</code>）。`
          ],
          svg: "mask-segments"
        },
        {
          title: "第三层：行内列界与 R2P",
          body: [
            R`Block 04：partial 块内，每个线程拿到自己负责的行号 <code>row_idx</code>，算出 <code>col_limit_right = row_idx + causal_row_offset + 1</code>（其中 <code>causal_row_offset = s_k - n_block·tile_n - s_q</code> 把端对齐和块偏移一次性吸收）。local mask 再加一个 <code>col_limit_left</code>。`,
            R`Block 05 的 R2P 三件套：<code>r2p_bitmask_below(limit, s)</code> 用一次右移生成「第 s 个 32 列 chunk 中保留 < limit」的位掩码（inline PTX 避免移位宽度未定义行为）；<code>r2p_bitmask_above</code> 对称处理下界；<code>mask_r2p_lambda</code> 在编译期展开的循环里把位掩码逐位喂给 SEL——<code>range_constexpr</code> 是必须的，否则编译器无法生成 R2P 指令。local mask 的双界直接 AND 两个位掩码，优雅至极。`
          ],
          formula: R`<p>指令量对比（一个 128 列 tile、每行 \(n_{\mathrm{func}}\) 个合法区间）：</p>
\[ \text{逐元素：}\; \approx 128\, n_{\mathrm{func}}\ \text{条 ISETP} + 128\,(\tfrac{n_{\mathrm{func}}}{2}{+}1)\ \text{条 SEL}; \]
\[ \text{R2P：}\; \lceil 128/32 \rceil \times O(n_{\mathrm{func}})\ \text{条位运算} + 128\ \text{条 SEL} . \]
<p>比较与坐标加法几乎全部消失，SEL 固定 128 条。这是 FA4 fork 中验证过的优化（当时按 24 位一批），SM100 原生实现按 32 位 chunk 重写。</p>`
        },
        {
          title: "可编程出口：mask_mod 与 block sparse",
          body: [
            R`Block 06：当几何超出 causal/local 的表达力（如任意 document mask、NSA 式动态稀疏），走 FlexAttention 风格的 <code>mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors)</code> 逐元素谓词。它昂贵，所以搭配 block-sparse 表使用：CSR 化的 <code>mask_block_idx</code>（partial 块清单）与 <code>full_block_idx</code>（full 块清单）,full 块调用不带 mask_mod 的版本（<code>mask_fn_none</code>）,只做 OOB 检查。`,
            R`PackGQA 的坐标还原也在这里:折叠进 seqlen 维的行号先 <code>divmod(global_row, qhead_per_kvhead)</code> 拆回 (真实行, head 偏移),再喂给 mask_mod——可编程接口看到的是逻辑坐标,物理打包对用户透明。`,
            R`Block 07 回收第 0 章的伏笔:<code>MT_MAP</code> 只有 full/causal,<code>ranges_to_cu_seqlens</code> 断言 ranges 连续不重叠。INV/BI-CAUSAL 的原生支持（练习 6 设计过）与任意重叠 ranges,是这套三层防线接下来要长出的能力。`
          ]
        }
      ],
      warning: R`「causal」在本仓库里恒指<strong>端对齐</strong>（右下对齐）：\(k \le q + (s_k - s_q)\)。与「左上对齐」（\(k \le q\)）在 \(s_q \ne s_k\) 时完全不同。读任何 mask 相关代码前先确认对齐约定，否则 BlockInfo 的跳块公式会显得莫名其妙。`,
      exercises: [
        {
          kind: "计算", level: "基础",
          q: R`\(s_q = s_k = 1024\)、tile 128×128、causal。对 m_block=3 的 Q tile，写出 n_block 迭代范围和三段循环各自覆盖的块。`,
          hint: R`m_idx_max = 4×128 = 512。`,
          answer: R`n_block_max = ⌈512/128⌉ = 4，即迭代 n_block 0..3。对角带起点：tile 内最小行 384 对应列界 385，所处块 = 3，故 Mainloop-1 覆盖 n_block 3（唯一 partial 块，含对角线），Mainloop-2 覆盖 n_block 0..2（full），Mainloop-3 不存在。方阵 causal 恰好每个 Q tile 只有 1 个 partial 块。`
        },
        {
          kind: "计算", level: "基础",
          q: R`验证 r2p_bitmask_below：chunk s=1（列 32..63），col_limit_right=41，掩码值是多少？哪些列被保留？`,
          hint: R`m = (s+1)×32 − limit。`,
          answer: R`m = 64 − 41 = 23，掩码 = 0xFFFFFFFF >> 23 = 0x1FF，即低 9 位为 1——保留 chunk 内前 9 列（全局列 32..40），恰为「列 < 41」。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`统计方阵 causal（\(n\) 个 tile 边长）下 partial 块占总迭代块数的比例，并说明它如何随序列变长而变化。`,
          hint: R`partial 块只在对角线上。`,
          answer: R`总迭代块数 \(\sum_{m=1}^{n} m = \frac{n(n+1)}{2}\)，partial 块每行 1 个共 \(n\) 个，占比 \(\frac{2}{n+1}\)。序列越长占比越低：\(n=64\)（8K 序列）时仅 3%。这解释了为什么 mask 优化的重点是「让 full 段零开销」而非「让 partial 段更快」——但 flex/稀疏 mask 会推高 partial 占比，届时 R2P 成为主角。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`mask_r2p_lambda 里为什么内层循环必须 <code>range_constexpr</code>？如果写成运行时循环会怎样？`,
          hint: R`R2P 指令的语义是「寄存器位 → 谓词寄存器」。`,
          answer: R`R2P 要求编译器在编译期知道「哪一位对应哪条 SEL」，才能把 <code>mask & (1<<i)</code> 的序列模式识别成一条 R2P + 谓词化 SEL 序列。运行时循环里 i 是变量，每次迭代都是独立的移位+测试+分支，编译器只能生成逐元素代码——不仅没有 R2P，还多了循环开销。「编译期展开是优化的启用条件」是 CuTe DSL 编程的普遍规律。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`滑动窗口 mask（窗宽 w）下,三段循环的中段（full 区）长度是多少个块？什么条件下 full 区消失、退化为纯 partial 迭代？`,
          hint: R`full 区 = 右界带与左界带之间。`,
          answer: R`每行合法列数约 w，跨 ⌈w/128⌉ 个块；其中右端 1 块（causal 界穿过）与左端 1 块（左窗界穿过）是 partial，full 区 ≈ ⌈w/128⌉ − 2 块。当 w ≤ 2×128 = 256 时 full 区消失，每个 Q tile 全是 partial 块——此时窗口 mask 的 R2P 双界路径（below AND above）成为绝对主路径，这正是源码为 local 单独准备双界位掩码的原因。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`要在这套三层防线上支持每行多个合法区间（\(n_{\mathrm{func}} > 1\)，即 HSTU Function 语义），三层各需什么改动？R2P 部分如何推广？`,
          hint: R`参考 FA4 博客的区间 XOR 技巧。`,
          answer: R`第一层：n_block_min/max 改为按「所有区间的包络」计算，或直接改用 CSR 块表（现成的 block-sparse 机制）。第二层：full/partial 分类按每块与区间集的相交关系预计算（create_block_mask 类 kernel）。第三层：R2P 天然可推广——每个区间 \([a,b)\) 的位掩码是 <code>below(b) & above(a)</code>，多区间取 OR：\(\bigvee_i (\text{below}(b_i)\,\&\,\text{above}(a_i))\)，位运算次数 \(O(n_{\mathrm{func}})\)，SEL 仍固定 128 条。这正是 FA4 fork 中 R2P 批量谓词方案的一般形式。`
        }
      ],
      sources: [
        { label: "源码 · kernel/cutedsl/mask.py（AttentionMask 与 R2P）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/mask.py" },
        { label: "源码 · kernel/cutedsl/block_info.py（跳块几何）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/block_info.py" },
        { label: "源码 · kernel/cutedsl/seqlen_info.py（varlen 坐标系）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/seqlen_info.py" },
        { label: "官方博客 · FFA_FA4（Full/Partial/Empty 分类与 R2P 的量化分析）", url: "https://sandai-org.github.io/MagiAttention/docs/blog/blackwell_ffa_fa4/" },
        { label: "PyTorch FlexAttention（mask_mod/score_mod 编程模型的来源）", url: "https://pytorch.org/blog/flexattention/" }
      ]
    },

    /* ================================================================ *
     * 04 · Online softmax on TMEM
     * ================================================================ */
    {
      id: "softmax",
      order: 4,
      title: "Online Softmax",
      fullTitle: "Online Softmax on TMEM",
      zhTitle: "指数运算的工业化",
      tag: "SM100 · 数值",
      category: "linear",
      difficulty: "进阶",
      source: "kernel/cutedsl/softmax.py + softmax_step",
      deck: R`softmax 是注意力里唯一的非线性，也是 SM100 前向唯一「用 CUDA core 算数」的地方。本章从 online softmax 的递推公式出发，走完 softmax_step 的九个动作，最后解决一个 B200 特有的瓶颈：SFU 的 exp2 吞吐不够,怎么办？答案是用 FMA 管线<strong>多项式仿真 exp2</strong>。`,
      takeaway: R`Online softmax 把「全行归一化」拆成可流式的三元组 \((m, \ell, O)\)：新块只需更新 \(m\)、给出一个标量 <code>corr_scale</code> \(=2^{(m_{\mathrm{old}}-m_{\mathrm{new}})\cdot c}\)，归一化推迟到最后。SM100 实现的两个关键改写：一切指数用 <strong>exp2</strong>（把 \(\ln 2\) 吸进 scale），以及当 SFU 不够快时,按 <code>ex2_emu_freq</code> 的节奏把部分 exp2 换成 <strong>FMA 管线上的多项式仿真</strong>——两条发射端口同时吐指数。`,
      intuitions: [
        { label: "流式", title: "推迟归一化", body: R`不必等所有分数到齐才除以 \(\sum e^s\)：维护跑动的 max 和 sum，旧的 O 用一个标量补差价。` },
        { label: "换底", title: "exp2 是硬件的母语", body: R`\(e^x = 2^{x\log_2 e}\)。把 \(\log_2 e\) 乘进 softmax scale 后，内层循环只剩 FMA 和 ex2。` },
        { label: "双管线", title: "SFU 堵车就走 FMA", body: "MUFU.EX2 吞吐有限;多项式仿真用 packed f32x2 的 FMA 算 exp2,两条端口并行,总吞吐反超。" }
      ],
      motivation: [
        R`第 2 章讲了 softmax warp 拿着全场最多的寄存器（176–256/线程），本章解释这些寄存器在算什么。一个 softmax_step 处理一个 128×128 的 S 块：每线程分到一行的 128 个 fp32,要做 mask、求 max、减 max、乘 scale、exp2、转 bf16、写回、更新 row_sum——全部在寄存器里完成。`,
        R`数值稳定性的约束塑造了每一步:减 max 防上溢;\(-\infty\) 行的 row_max 用 0 兜底（<code>row_max_safe</code>）防 NaN;FP8 时还有 <code>max_offset=8</code> 把动态范围往上挪。这些「小心思」都在 <code>SoftmaxSm100</code> 的十几行里。`,
        R`最后是 B200 的现实:hd128 配置下 softmax 的 exp2 需求接近 SFU 峰值,而 SM103（B300）SFU 更快。所以调参表里 SM100 的 <code>ex2_emu_freq</code> 是 10–16（周期性仿真）,SM103 全部为 0（纯硬件）——一张表写尽两代芯片的性格差异。`
      ],
      diagram: {
        key: "softmax",
        caption: "softmax_step 的数据流：等 S 满 → T2R → score_mod/mask → row_max/corr_scale 发布 → 减 max 乘 scale → exp2（硬件/仿真混合）→ bf16 P 写回 TMEM → 背压与 row_sum。点击节点查看源码。"
      },
      explain: [
        {
          title: "Online softmax 的递推",
          body: [
            R`标准推导（FA2 一脉相承）:维护每行状态 \((m_i, \ell_i)\) 与未归一化输出 \(O_i\)。新块 \(S_i\) 到来时先并入 max,再把旧的 \(\ell\)、\(O\) 用同一个标量缩放,最后累加新块的贡献。全部块处理完后 \(O \leftarrow O/\ell\) 一次归一化,\(\mathrm{LSE} = m + \ln \ell\)。`,
            R`SM100 版的分工（对照 Block 03/07）:<code>update_row_max</code> 返回 \((m_{\mathrm{safe}}, \text{corr\_scale})\),corr_scale 经 sScale 发布给 correction warp（去改 O,第 5 章）;softmax 自己只负责 \(\ell\) 的缩放累加（<code>update_row_sum</code>,Block 05）。也就是说:<strong>三元组的三个分量由两组 warp 分头维护</strong>,凭 corr_scale 这一个标量保持一致。`,
            R`Block 07 里还有一个 <code>rescale_threshold</code> 门:若 \(\Delta m \cdot c\) 足够小（acc_scale ≈ 1）,干脆保留旧 max、令 corr_scale=1——correction 侧用 <code>vote_ballot_sync(corr_scale < 1.0)</code> 整 warp 投票,全票 ≈1 就跳过整次 rescale。数值上安全（exp 参数只是略偏离 0）,性能上省一次 128×128 的 TMEM 读改写。`
          ],
          formula: R`\[ m_i = \max(m_{i-1}, \operatorname{rowmax}(S_i)), \qquad \text{corr\_scale} = 2^{(m_{i-1} - m_i)\,c}, \quad c = \text{softmax\_scale}\cdot\log_2 e, \]
\[ \tilde P_i = 2^{\,S_i c \,-\, m_i c}, \qquad \ell_i = \ell_{i-1}\cdot \text{corr\_scale} + \operatorname{rowsum}(\tilde P_i), \qquad O_i = O_{i-1}\cdot \text{corr\_scale} + \tilde P_i V_i . \]
<p>最终 \(O = O_N / \ell_N\)，\(\mathrm{LSE} = (m_N c + \log_2 \ell_N)\cdot \ln 2\)。所有指数均为以 2 为底——scale 里的 \(\log_2 e\) 完成了换底。</p>`
        },
        {
          title: "softmax_step 逐行读",
          body: [
            R`Block 01–05 按执行顺序排列。① <code>consumer_wait(S-full)</code> + T2R:一次 <code>Ld32x32b</code> 系 copy 把本线程的 128 个 fp32 拉进寄存器。② 挂载点:score_mod（softcap 也从这里进来——host 把它包装成 score_mod）与 mask_fn（第 3 章的成果,full 段编译期为空）。③ <code>update_row_max</code> → 写 corr_scale 到 sScale → <code>sm_stats_barrier.arrive</code>:注意这发生在 exp2 <em>之前</em>——correction 越早拿到 scale,越早开始改 O,与本 warp 的 exp2 完全并行。`,
            R`④ <code>scale_subtract_rowmax</code>:一条 packed FMA 同时处理两个元素,\(s \cdot c + (\text{max\_offset} - m c)\)。⑤ <code>apply_exp2_convert</code>:exp2 + 转 bf16 一体完成（详见下节）。⑥ R2T 写 P（含 split_P_arrive 的提前放行,第 2 章讲过）。⑦ WAR acquire:确认 correction 已读走上一个 scale,才允许下一次覆写 sScale。⑧ <code>update_row_sum</code>——被刻意安排在 acquire <em>之后</em>,让这段纯寄存器计算填满等待窗口。`,
            R`一个值得玩味的细节:mask 存在的 step 会强制 <code>ex2_emu_freq=0</code>（Block 04 的三元式）——被 mask 的行里有 \(-\infty\),多项式仿真在极端输入下的行为不如硬件 ex2 可靠,宁可慢一点也要走硬件路径。`
          ]
        },
        {
          title: "ex2 仿真：把指数算在 FMA 管线上",
          body: [
            R`B200 的 SFU（MUFU.EX2）每周期每 SM 吞吐有限,而 softmax 每元素恰好一次 exp2——hd128 时 exp2 需求逼近 SFU 峰值,SFU 成了整条流水线的短板。解法:一部分 exp2 改在 FMA 管线上用多项式仿真,两条发射端口并行出活。`,
            R`Block 08 的 <code>ex2_emulation_2</code> 逐步拆解:(1) clamp 到 \([-127, \infty)\) 防下溢;(2) 加 magic number \(2^{23}+2^{22}\) 并用 <strong>round-down</strong> 模式取整——浮点加法的舍入行为免费完成 floor,整数部分 \(\lfloor x \rfloor\) 落在尾数低 8 位;(3) 减回 magic 得到 \(\lfloor x\rfloor\),原值相减得小数部分 \(f \in [0,1)\);(4) 3 次多项式估值 \(2^f\)（3 条 packed FMA 处理一对元素）;(5) <code>combine_int_frac_ex2</code> 把 \(\lfloor x\rfloor\) 直接加进指数位。全程无分支、无查表,packed f32x2 使每条指令处理两个元素。`,
            R`调度由三个旋钮控制（Block 06）:<code>ex2_emu_freq</code>——每多少对元素里安排一次仿真（值越大仿真越少,0=全硬件）;<code>ex2_emu_start_frg</code>/<code>ex2_emu_res</code> 控制从哪个 fragment 开始、每周期仿真几对。本质是在做<strong>双资源的静态负载均衡</strong>:让 SFU 与 FMA 的占用比恰好匹配它们的吞吐比。SM103 SFU 提速后全部归零——同一份代码,两种芯片,两套节奏。`
          ],
          formula: R`<p>仿真的数学骨架：设 \(x = \lfloor x \rfloor + f\)，\(f\in[0,1)\)，则</p>
\[ 2^x = 2^{\lfloor x\rfloor}\cdot 2^{f}, \qquad 2^{f} \approx p_3(f) = c_0 + c_1 f + c_2 f^2 + c_3 f^3, \]
<p>其中 \(2^{\lfloor x\rfloor}\) 通过把 \(\lfloor x\rfloor\) 加到 IEEE-754 指数域实现（一条整数加法），\(p_3\) 用 Horner 格式 3 条 FMA 完成。bf16 输出只有 8 位尾数，3 次多项式的相对误差（\(\sim10^{-4}\)）绰绰有余——这是「按输出精度定计算精度」的教科书案例。</p>`
        },
        {
          title: "FP8 与 max_offset",
          body: [
            R`FP8 输入时 P 也要量化到 FP8（e4m3 动态范围 ±448）。<code>max_offset=8</code> 把 exp2 的参数整体抬高 8：\(\tilde P = 2^{sc - mc + 8} = 256\cdot P\)，让 P 的典型值从 \((0,1]\) 挪到 \((0,256]\)，充分利用 e4m3 的表示密度；correction 尾声再除掉 \(2^{8}\)（藏在 <code>max_offset_scale</code> 与 v_descale 里）。LSE 计算同样要减回 offset（第 5 章 Block 06 里的 <code>- max_offset</code>）。`,
            R`这类「在指数域做平移」的技巧之所以廉价，是因为整条链路都以 2 为底：平移只是 FMA 的 bias 项改一个常数，没有任何额外指令。`
          ]
        }
      ],
      warning: R`corr_scale 的语义是「旧 max 相对新 max 的贬值率」，恒 ≤1；它乘在<strong>旧的</strong> \(\ell\) 和 \(O\) 上，而不是新块上。首块（is_first）没有旧状态，corr_scale 无意义也不发布——correction 的 prologue 直接放行。混淆新旧方向是手推 online softmax 最常见的错误。`,
      exercises: [
        {
          kind: "推导", level: "基础",
          q: R`两块序列 \(S_1, S_2\)，验证 online 递推给出的 \(\ell_2\) 与一次性计算 \(\sum_k e^{(s_k - m_2)c'}\)（\(c' = c\ln 2\) 意义下）一致。`,
          hint: R`把 \(\ell_1 = \sum_{k\in S_1} 2^{(s_k-m_1)c}\) 乘上 corr_scale 展开。`,
          answer: R`\(\ell_1 \cdot 2^{(m_1-m_2)c} = \sum_{k\in S_1} 2^{(s_k-m_1)c + (m_1-m_2)c} = \sum_{k\in S_1} 2^{(s_k-m_2)c}\)，再加上新块的 \(\sum_{k\in S_2} 2^{(s_k-m_2)c}\)，恰为全集在基准 \(m_2\) 下的和。递推只是不断「换基准并补差价」。`
        },
        {
          kind: "计算", level: "基础",
          q: R`用 magic number 法算 \(\lfloor 3.7 \rfloor\)：\(3.7 + (2^{23}+2^{22})\) 在 round-down 下的结果尾数低位是什么？减回 magic 后得到多少？`,
          hint: R`\(2^{23}+2^{22} = 12582912\)，和落在 \([2^{23}, 2^{24})\) 区间,ulp=1。`,
          answer: R`和为 12582915.7，该量级下 fp32 的 ulp 是 1，round-down 得 12582915，整数部分 3 编码在尾数低位。减回 magic 得 3.0——即 \(\lfloor 3.7\rfloor\)。round 模式若是 nearest 则 3.7 会进位成 4，这就是源码强调 <code>rnd="rm"</code> 的原因。`
        },
        {
          kind: "概念", level: "进阶",
          q: R`为什么 corr_scale 的发布（sm_stats_barrier.arrive）安排在 exp2 之前，而 row_sum 的更新安排在 WAR acquire 之后？两处安排各隐藏了什么延迟？`,
          hint: R`想想 correction warp 此刻在干什么、sScale 槽位何时才能复用。`,
          answer: R`前者：corr_scale 在 row_max 更新后立即可知,早发布一拍,correction 的 O-rescale 就能与本 warp 的整段 exp2+写 P 并行——隐藏的是 correction 的全部工作时长。后者：WAR acquire 是可能阻塞的等待,把无依赖的 row_sum 更新（纯寄存器 FMA）挪到 acquire 之后执行,等待窗口被计算填满——隐藏的是背压等待。两处都是「把必须等的和不必等的重新排序」。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`设 SFU 的 ex2 吞吐为 FMA 管线的 \(1/4\)（示意值），多项式仿真每个 exp2 耗 4 条 FMA 位。求最优的仿真比例 \(\alpha\)（仿真元素占比）使总吞吐最大，并解释 ex2_emu_freq 如何逼近它。`,
          hint: R`让两条管线同时跑满：硬件路径耗时 ∝ (1−α)/T_sfu，仿真路径 ∝ 4α/T_fma。`,
          answer: R`平衡条件 \((1-\alpha)/1 = 4\alpha/4\)（以各自吞吐归一）⇒ \(1-\alpha = \alpha\) ⇒ \(\alpha = 1/2\)？代入吞吐比：SFU 速率 1、FMA 速率 4，硬件路径时间 \((1-\alpha)\)、仿真时间 \(4\alpha/4 = \alpha\)，并行取 max，最优在 \(\alpha^* = 1/2\)，加速 2 倍（相对纯 SFU）。实际中 FMA 还要跑 scale/转换等其他活，最优 \(\alpha\) 更小——freq=10~16 意味着每 10–16 对元素仿真其中几对，α≈10–30%，正是把 FMA 的「剩余产能」精确填满而不反噬。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`带 mask 的 softmax_step 强制 ex2_emu_freq=0。从仿真算法的输入域假设出发解释原因。`,
          hint: R`看 ex2_emulation_2 的第一步 clamp 和被 mask 元素的值。`,
          answer: R`被 mask 的元素是 \(-\infty\)，减 max 乘 scale 后仍是 \(-\infty\)。仿真第一步 clamp 到 −127 后结果是 \(2^{-127}\)（非规格化边缘）而非精确 0；硬件 ex2 对 \(-\infty\) 直接返回 0。\(2^{-127}\) 乘 V 累加后一般无害,但在行全被 mask、row_sum 本应为 0 的边角情形会污染「空行检测」（row_sum==0 的判断）。与其为极端值加分支,不如 partial 块整体走硬件路径——反正 partial 块占比低（第 3 章练习 3）。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`若把 P 的输出精度从 bf16 降到 FP8（e4m3），多项式次数能否从 3 降到 2？给出误差预算分析。`,
          hint: R`e4m3 尾数 3 位，相对精度 ~2^-4;2 次多项式对 2^f 的最大相对误差约 10^-3。`,
          answer: R`可以。e4m3 的量化相对误差约 \(2^{-4} = 6.25\%\)（3 位尾数），而 2 次 minimax 多项式逼近 \(2^f, f\in[0,1)\) 的相对误差约 \(10^{-3}\)，比量化误差低 60 倍，完全被淹没。省 1 条 FMA/元素对，仿真成本降 25%，最优仿真比例相应提高。这是「输出精度决定中间精度」的又一次应用——但要注意 row_sum 用的是 exp2 的 fp32 值（转 FP8 前），它参与 LSE，误差预算按 fp32 路径单独评估，所以源码里 row_sum 累加永远用转换前的寄存器值。`
        }
      ],
      sources: [
        { label: "源码 · kernel/cutedsl/softmax.py（SoftmaxSm100）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/softmax.py" },
        { label: "源码 · kernel/cutedsl/cutedsl_utils.py（ex2_emulation 与 packed math）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/cutedsl_utils.py" },
        { label: "Milakov & Gimelshein · Online normalizer calculation for softmax", url: "https://arxiv.org/abs/1805.02867" },
        { label: "FlashAttention-2 论文（\\(m,\\ell,O\\) 三元组递推的出处）", url: "https://arxiv.org/abs/2307.08691" },
        { label: "MagiAttention 博客 · FA2 数学推导（官方版递推证明）", url: "https://sandai-org.github.io/MagiAttention/docs/blog/fa2_math_derivation/" }
      ]
    },

    /* ================================================================ *
     * 05 · Correction & epilogue
     * ================================================================ */
    {
      id: "correction",
      order: 5,
      title: "Correction 与归约",
      fullTitle: "Correction, Epilogue & LSE",
      zhTitle: "把 partial 结果变成承诺的输出",
      tag: "SM100 · 归约",
      category: "sparse",
      difficulty: "进阶",
      source: "kernel/cutedsl/ffa_fwd_sm100.py · correction_loop",
      deck: R`第 0 章承诺的「可累加 out + 可合并 lse」，由 correction warpgroup 兑现：主循环里它用 corr_scale 持续校准 TMEM 中的 O，尾声用 \(1/\ell\) 归一化、并入 sink、转换精度、写出 O 与 LSE。本章同时讲透它与 softmax 之间那对精妙的 RAW/WAR 握手。`,
      takeaway: R`Correction 是 online softmax 的「债务清算人」：softmax 每换一次 max 基准，旧 O 就欠一笔 \(\times\text{corr\_scale}\) 的债，correction 在<strong>别人的空隙里</strong>逐块还清；尾声一次性清算 \(\times 1/\ell\)。它与 softmax 只共享 <strong>sScale 一个槽位</strong>——RAW 用 named barrier、WAR 用 pipeline，而 correction 的 <strong>cross-release</strong>（释放对面 stage 的槽位）让单组 correction 以圆舞曲的节奏轮流服务两组 softmax。`,
      intuitions: [
        { label: "债务", title: "换基准就欠账", body: R`row_max 从 \(m_{old}\) 变 \(m_{new}\)，旧 O 的每个元素都高估了 \(2^{(m_{old}-m_{new})c}\) 倍的分母——乘 corr_scale 还账。` },
        { label: "错峰", title: "在空隙里干活", body: "O 的 rescale 与下一块的 softmax 完全无依赖;correction 的全部工作都藏在 softmax 与 MMA 的影子里。" },
        { label: "清算", title: "尾声一次归一", body: R`主循环维持未归一化的 \(O\)；最后 \(\times\,\mathrm{rcp}(\ell)\)、并 sink、转 dtype、写 gmem，LSE 同步产出。` }
      ],
      motivation: [
        R`online softmax 的三元组 \((m,\ell,O)\) 中,\(O\) 的体积是另外两者的 128 倍（128×128 fp32 vs 两个每行标量）。把「改 O」交给专职 warpgroup,是 SM100 版本相对 Hopper FA3 的关键重构——FA3 里 rescale 由算 softmax 的同一 warpgroup 顺手做,挤占寄存器且拉长关键路径。`,
        R`correction 的难点不在计算（乘一个标量）,在<strong>同步协议</strong>:corr_scale 经 SMEM 的 sScale 槽位从 softmax 流向 correction,一个槽位、两个写者时机（主循环 corr_scale / 尾声 row_sum）、两个读者时机,还要在两个 Q-stage 之间轮转。源码用「RAW 靠 barrier、WAR 靠 pipeline、轮转靠 cross-release」三招解决,值得逐行品味。`,
        R`尾声则是数值收口:除以 row_sum 之前要处理三件事——空行（row_sum=0 或 NaN,用 1.0 兜底避免除零,输出自然为 0）、learnable sink（在分母里补一项 \(2^{s_{\mathrm{sink}}\log_2 e - mc}\)）、FP8 的 v_descale 与 max_offset 回补。LSE 的合成公式则直接回应第 0 章的合并语义。`
      ],
      diagram: {
        key: "correction",
        caption: "correction 的两个阶段：主循环从 sScale 读 corr_scale 并 rescale TMEM 中的 O（T2R→乘→R2T）；尾声等最终 O 与 row_sum，做 1/ℓ 归一化并写出 O 与 LSE。点击节点查看源码。"
      },
      explain: [
        {
          title: "sScale：一个槽位的完整协议",
          body: [
            R`Block 01:sScale 每个 Q-stage 每行有 2 个槽——主循环的 corr_scale 与尾声的 row_sum/row_max 分开存放（后者在偏移 <code>q_stage*128</code> 处,见第 4 章 softmax_loop 尾声的写入）。`,
            R`RAW（数据就绪）:softmax 写完 → <code>sm_stats_barrier.arrive</code>（非阻塞）;correction <code>arrive_and_wait</code> 会合后才读。WAR（槽位可覆写）:correction 读完 → <code>pipeline_sm_stats.consumer_release</code>;softmax 在下一次写之前 <code>producer_acquire</code> 确认。一个槽位上两种冒险,分别交给两种原语——barrier 无相位、适合每次会合;pipeline 有相位、适合跨迭代的占用权转移。`,
            R`<strong>cross-release</strong>（Block 02 尾部）:主循环里 correction 释放的不是当前 stage 的槽位,而是 <code>q_stage-1-stage</code>——对面的。效果:softmax0 发布 scale 后,要等 correction 服务完 softmax1 才拿回自己槽位的写权。单组 correction 于是像圆舞曲一样在两组 softmax 之间交替,任何时刻只有一组 softmax 与 correction 重叠,另一组安静地停在 acquire 上——这同时也是两组 softmax 的天然错峰器（与第 2 章的 s0_s1_barrier 相辅相成）。尾声改回直接 release（不再轮转）,并在切换前补一次 release 排空主循环留下的「悬空槽」。`
          ],
          svg: "correction-handshake"
        },
        {
          title: "主循环：ballot 决定要不要还账",
          body: [
            R`Block 02:读到 corr_scale 后,<code>vote_ballot_sync(corr_scale < 1.0)</code> 让整个 warp 投票——只要有任一行的 scale 显著小于 1 就整块 rescale;全体 ≈1（配合第 4 章的 rescale_threshold 门控）则跳过。注意投票粒度是 warp（32 行）,以 warp 为单位跳过是 TMEM copy 原子宽度决定的。`,
            R`Block 03 的 rescale 本体:按 corrHD=16 列一批,T2R（<code>Ld32x32b</code>）→ packed f32x2 乘法 → R2T（<code>St32x32b</code>）,最后 <code>fence_view_async_tmem_store</code> 保证对 MMA warp 可见。128 列分 8 批流过寄存器,每线程瞬时只驻留 16 个 fp32——这就是 correction 只要 64–88 个寄存器的原因。`,
            R`随后的 <code>pipeline_s_p_o.consumer_release</code> 是第 2 章讲过的「O 侧放行」:MMA 由此得知 O(i-1) 已校准,可以发起向同一 TMEM 区域累加的下一次 PV GEMM。`
          ]
        },
        {
          title: "尾声：归一化、sink 与空行",
          body: [
            R`Block 04:读最终 row_sum/row_max → 若有 sink,分母补一项 \(2^{s\log_2 e - mc + \text{offset}}\)（learnable sink 等价于每行多一个 logit 为 \(s\) 的虚拟 token,它只进分母不贡献 V）→ 空行检测 <code>row_sum==0 || row_sum!=row_sum</code>,兜底 scale=1 → <code>rcp_approx(row_sum)</code> 得归一化系数（顺手乘 FP8 的 v_descale）→ 显式等 <code>pipeline_o_acc</code>（第 2 章练习 3 的「断裂点」）。`,
            R`Block 05:<code>correction_epilogue</code> 把 \(O \times \mathrm{rcp}(\ell)\)、转换为输出 dtype、写进 SMEM 的 sO;非 varlen 时 commit 给 epilogue warp 走 TMA store,varlen 时 correction 自己逐行写 gmem。`
          ]
        },
        {
          title: "LSE：写给「下一次合并」的收据",
          body: [
            R`Block 06:\(\mathrm{LSE} = (m\,c + \log_2 \ell - \text{max\_offset})\cdot\ln 2\)。这是自然对数意义下的 \(\log\sum e^{s\cdot\text{scale}}\),空行写 \(-\infty\)。每线程负责一行,PackGQA 时还要把折叠行号拆回 (head, 行) 散写。`,
            R`把第 0 章的合并公式与这里连起来,就看清了 MagiAttention 分布式的完整闭环:每个 rank 的 FFA kernel 产出 \((O^{(r)}, \mathrm{LSE}^{(r)})\),GroupReduce 用合并公式把它们归约成全局结果——kernel 内的 correction 是「块间归约」,分布式的 correction 是「rank 间归约」,数学上是同一个半群运算在两个尺度上的重复。`
          ],
          formula: R`<p>验证 kernel 内与 kernel 间归约的一致性：kernel 尾声输出 \(O = \frac{\sum_k \tilde p_k v_k}{\ell}\)、\(\mathrm{LSE} = mc' + \ln \ell\)（\(c'=c\ln2\)）。两个 rank 合并时</p>
\[ w_r = e^{\mathrm{LSE}_r - \mathrm{LSE}},\qquad O = \sum_r w_r O_r = \frac{\sum_r e^{\mathrm{LSE}_r} O_r}{e^{\mathrm{LSE}}} = \frac{\sum_r \sum_{k\in K_r} e^{s_k c'} v_k}{\sum_r \sum_{k \in K_r} e^{s_k c'}}, \]
<p>与把全部 K 交给单 kernel 的结果逐项相等。fp32 的 out 累加路径（functional 层 atomic 模式）走的正是分子分母同时累加的等价形式。</p>`
        }
      ],
      warning: R`corr_scale < 1 的 ballot 判断依赖第 4 章的 rescale_threshold 约定：softmax 侧把「接近 1」的 scale 直接规范成 1.0。若单独修改一侧的阈值逻辑，会出现 correction 频繁做无效 rescale 或漏掉必要 rescale 的隐性 bug——这对参数必须成对演化。`,
      exercises: [
        {
          kind: "概念", level: "基础",
          q: R`为什么主循环中 correction 修正的是 O(i-1) 而不是 O(i)？O(i) 此刻在哪里？`,
          hint: R`corr_scale(i) 的语义是「块 i 使基准变化后旧账的贬值率」。`,
          answer: R`corr_scale(i) 由块 i 的 row_max 更新产生,描述的是「块 0..i-1 累积的 O 相对新基准的高估」——所以乘在 O(i-1)（即累加到 i-1 为止的 O）上。而块 i 自己的贡献 \(\tilde P_i V_i\) 此刻还没算:P_i 刚被 softmax 写回 TMEM,PV GEMM 要等 correction 放行（O 侧 release）后才发射,其结果直接以新基准累加,无需修正。`
        },
        {
          kind: "计算", level: "基础",
          q: R`一行的处理经过 3 个 KV 块，row_max 依次为 2.0 → 5.0 → 5.0，scale_log2 = 1。写出两次 corr_scale 以及尾声前 O 携带的总校准系数。`,
          hint: R`corr_scale = 2^{(m_old − m_new)·c}。`,
          answer: R`第二块：\(2^{(2-5)\times1} = 2^{-3} = 0.125\)；第三块：\(2^{(5-5)} = 1\)（触发 rescale_threshold 短路，ballot 跳过）。块 1 的贡献总共被乘过 0.125×1 = 0.125，恰为 \(2^{(m_1 - m_3)c}\)——校准系数可迟到但不会算错，这是递推的伸缩性。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`cross-release 若改成 release 自己的 stage（直觉写法），两组 softmax 的执行会变成什么节奏？为什么反而更慢？`,
          hint: R`追踪 softmax0 连发两块的时序。`,
          answer: R`直接 release 下，softmax0 发布 scale 后立刻拿回槽位写权，可以连续冲刺多个 KV 块；softmax1 同样。两组会「同相位」推进——同时到达 exp2+写 P 段（TMEM 写口打架，s0_s1_barrier 强制其中一组干等）、同时向 correction 要服务（correction 串行处理,另一组排队）。cross-release 强制两组相位错开半拍：correction 服务 A 时 B 在算,服务 B 时 A 在算——资源冲突消失。慢的原因不是吞吐而是<strong>相位共振</strong>,cross-release 是一个用同步原语实现的「相位分离器」。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`推导 learnable sink 修正 row_sum 的公式：为何加的是 \(2^{s\cdot\log_2 e - m c + \text{offset}}\)？sink 为什么不需要对应的 V 贡献？`,
          hint: R`把 sink 视为一个 logit 恒为 \(s\)（自然对数域）的虚拟 token。`,
          answer: R`虚拟 token 的未归一化权重是 \(e^{s - m c'} \)（与真实 token 同基准）,换成以 2 为底:\(2^{s\log_2 e - mc}\),再补 max_offset 与其他项对齐——正是源码的表达式。sink 的语义是「允许注意力弃权」:它进分母稀释所有真实权重,但没有 value 向量,不进分子。所以只改 row_sum,O 不动;LSE 因分母变化自动变大,合并语义仍然自洽。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`空行（row_sum=0）时源码把归一化 scale 兜底为 1 而不是跳过写出。从「分布式消费者」的角度解释为什么必须写出 0 而不是留下垃圾值。`,
          hint: R`这个 out 可能马上被 GroupReduce 当作 partial 累加。`,
          answer: R`空行的 O 累加器本身为 0（没有任何 PV 贡献）,乘 1 后写出的是干净的 0,配合 LSE=−∞:合并公式中 \(w = e^{-\infty - \mathrm{lse}} = 0\),该 partial 对全局结果零贡献。若跳过写出,gmem 里是未初始化/上一轮的垃圾,而合并端并不知道哪些行「该被跳过」——除非引入额外的 valid 位图。写 0 + −∞ 让「空」成为合并代数里的合法元素,免掉所有特判。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`functional 层的 atomic 模式下,多个 kernel 实例向同一 out 累加的是「已按各自 lse 归一化的输出」吗？结合本章公式说明 atomic 直加成立的前提,以及为什么该路径要搭配后续的 lse 全局校正。`,
          hint: R`比较 \(\sum_r O_r\) 与 \(\sum_r w_r O_r\)。`,
          answer: R`不是。若各实例写出的是自归一化的 \(O_r\),直接 atomicAdd 得 \(\sum_r O_r\),而正确答案是 \(\sum_r w_r O_r\)——除非所有 \(w_r\) 已知且被预乘。FFA 的做法:重叠 ranges 场景下 kernel 写出的是<strong>按全局约定基准的未归一化分子</strong>（或等价地,消费端用各 range 的 lse 做 correct_attn_out_lse 校正）,fp32 累加保证分子求和的精度;归一化推迟到所有 partial 到齐后。也就是说 atomic 直加的前提是「加的是同一坐标系下的分子」,这正是 out+lse 成对出现的原因——lse 就是坐标系的记录。`
        }
      ],
      sources: [
        { label: "源码 · ffa_fwd_sm100.py correction_loop/correction_rescale/correction_epilogue", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/ffa_fwd_sm100.py" },
        { label: "源码 · functional/utils.py（correct_attn_out_lse 合并公式）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/functional/utils.py" },
        { label: "MagiAttention 博客 · Attention Sink（sink 的设计与实现）", url: "https://sandai-org.github.io/MagiAttention/docs/blog/attn_sink/" },
        { label: "MagiAttention 博客 · 主博文（GroupReduce 与分布式归约闭环）", url: "https://sandai-org.github.io/MagiAttention/docs/blog/magi_attn/" }
      ]
    },

    /* ================================================================ *
     * 06 · Tile scheduler
     * ================================================================ */
    {
      id: "scheduler",
      order: 6,
      title: "Tile Scheduler",
      fullTitle: "Persistent Scheduling & CLC",
      zhTitle: "把不均匀的工作喂满 SM",
      tag: "SM100 · 调度",
      category: "hybrid",
      difficulty: "进阶",
      source: "kernel/cutedsl/tile_scheduler.py · 1727 行",
      deck: R`mask 让每个 Q tile 的工作量天差地别：causal 下最后一个 tile 的计算量是第一个的 N 倍。调度层的使命是把这堆参差不齐的砖块严丝合缝地码进固定数量的 SM。FFA 备了四种调度器，外加 Blackwell 的硬件杀器——CLC（Cluster Launch Control）动态派工。`,
      takeaway: R`调度的三个正交决策：<strong>顺序</strong>（LPT——最重的块最先跑，装箱论的经典启发式）、<strong>亲和</strong>（L2 swizzle——让同时在跑的 CTA 尽量共享同一个 head 的 K/V）、<strong>分配方式</strong>（静态 grid-stride vs CLC 硬件动态派工）。三者在 <code>SingleTileLPTScheduler</code> 里合体：坐标映射管前两个，SchedulingMode 管第三个。而 host 侧的启发式提醒我们：<strong>动态调度不是免费午餐</strong>——负载本来均匀时，它只剩开销。`,
      intuitions: [
        { label: "顺序", title: "先搬大石头", body: "往桶里装石头要先大后小。causal 尾部的重 tile 若排在最后启动,其他 SM 只能干等它收尾。" },
        { label: "亲和", title: "邻居共享冰箱", body: "同一 head 的 K/V 是 CTA 之间唯一可共享的大宗数据;让相邻 tile_idx 落在同一 head,L2 命中率决定有效带宽。" },
        { label: "分配", title: "领任务 vs 派任务", body: "静态:每个 CTA 按固定步长自取。CLC:算完找硬件领下一份——空闲 SM 自动多领,负载自平衡。" }
      ],
      motivation: [
        R`先量化问题:方阵 causal 下第 \(m\) 个 Q tile 要迭代 \(m+1\) 个 KV block,最重与最轻差 \(n\) 倍。若按自然顺序静态分配,尾部的重 tile 最后才开始,长尾拖垮整卡利用率——这就是 LPT（Longest Processing Time first）要解的装箱问题。`,
        R`第二个问题是缓存:B200 一个 SM 同时只跑一两个 CTA,但全卡上百个 CTA 并发;它们的 K/V 读取都过同一块 L2。若并发的 CTA 分属不同 (head, batch),L2 里塞满互不相干的 K/V,命中率崩塌。调度器的坐标映射必须让「时间上相邻的 tile_idx」尽量「空间上共享 K/V」。`,
        R`第三个问题是不可预测性:varlen 下每条序列的块数不同,block-sparse 下有效块数取决于数据,静态映射再聪明也只是估计。Blackwell 的 CLC 给了硬件级答案:persistent CTA 每做完一个 tile,向硬件调度器申请下一个,响应写进 SMEM、mbarrier 通知——工作窃取的延迟被硬件压到微秒之下。`
      ],
      diagram: {
        key: "scheduler",
        caption: "调度层全景：host 启发式选定调度器类型；静态路径用 L2-swizzle + LPT 坐标映射；CLC 路径由调度 warp 预取、全体 warp 消费。点击节点查看源码。"
      },
      explain: [
        {
          title: "四种调度器与选型决策树",
          body: [
            R`Block 01 的决策链:varlen_q → <code>SingleTileVarlenScheduler</code>（warp 内前缀和 + ballot 定位 tile 属于哪条序列）;causal/local 或开 CLC → <code>SingleTileLPTScheduler</code>;dense 非 causal → <code>StaticPersistentTileScheduler</code>（CTA 数压到 SM 量级,grid-stride 轮询）;兜底 → <code>SingleTileScheduler</code>(一 CTA 一 tile,最简单)。`,
            R`「persistent」的含义:grid 不再等于 tile 数,而是 ≈ SM 数;每个 CTA 循环消费多个 tile。收益有二:省去大量 CTA 启动/收尾开销;让 prologue（TMA 预取、TMEM 分配）跨 tile 复用。所有角色 warp 的主循环都写成统一骨架:<code>work_tile = initial_work_tile_info(); while valid: ...; advance_to_next_work()</code>——调度策略被完全封装在这两个调用后面。`
          ]
        },
        {
          title: "LPT 与 L2 swizzle 的坐标算术",
          body: [
            R`Block 02:先估「一个 head 的 K/V 体积」\(= s_k\times(d+d_v)\times\text{elem}\),用 50MB 的 L2 预算除之,得到能同居的 head 数,向下取 2 的幂作为 <code>swizzle</code>。Block 03 的映射把线性 tile_idx 分解成「大节拍 × swizzle 内偏移」:同一节拍内的 CTA 共享 swizzle 个 head 的 K/V;末尾不足一节拍的余数单独处理（residual 分支）。`,
            R`LPT 只是最后一行:<code>block = num_block - 1 - block</code>——把块序反转,最重的（causal 下编号最大的）最先派发。配合 persistent 分配,轻重块自然穿插:先启动的 CTA 领走重块,做完时轻块还有剩,收尾整齐。`,
            R`调度理论背书:LPT 对 makespan 的近似比是 \(\tfrac{4}{3}-\tfrac{1}{3m}\)（Graham 1969）;对 causal 这种「工作量线性递增」的特殊分布,LPT + 动态领取几乎达到下界。`
          ],
          svg: "lpt-swizzle"
        },
        {
          title: "CLC：硬件动态派工",
          body: [
            R`CLC（Cluster Launch Control）是 Blackwell 的新硬件路径:kernel 声明一个逻辑 tile 网格（<code>clc_problem_shape</code>）,persistent CTA 通过 <code>clctrl.try_cancel</code> 类指令向 GPU 调度器申请「下一个尚未派发的 cluster 坐标」,响应异步写进 SMEM 的 16 字节 response buffer,mbarrier 记账通知。`,
            R`软件侧是一个标准 producer/consumer（Block 04/05/06）:调度 warp（empty warp 兼任,只在 leader CTA）循环 <code>prefetch_next_work</code>——acquire 空槽、把 mbarrier 地址交给硬件、advance;其余全部 warp 作为 consumer <code>consumer_wait → get_current_work → consumer_release</code>。sched_stages=1 时预取深度为 1:当前 tile 在算,下一个 tile 的坐标已在路上。`,
            R`CLC 返回的是原始 grid 坐标,<code>clc_work_to_coords</code> 再套用 LPT 反转与 split_kv 解包——但<strong>不做 L2 swizzle</strong>(注释:hardware decides order):派发顺序已由硬件决定,软件重排坐标只会破坏硬件的局部性假设。`
          ]
        },
        {
          title: "何时动态调度反而亏",
          body: [
            R`Block 07 的 host 启发式明确了两个回退场景。<strong>dense noncausal</strong>:所有 tile 等重,静态映射本来就完美均衡,CLC 只添 work-stealing 开销。<strong>varlen MHA</strong>(qhead_per_kvhead=1):不均衡序列使更多 K/V block 同时在飞,加上 CLC 打乱顺序,L2 压力雪上加霜——实测回退。`,
            R`这给出一个普适判断:<strong>动态调度买的是「против不可预测的不均衡」的保险</strong>。不均衡可预测(causal)时,LPT 静态排序已够;不均衡不可预测(block-sparse、CLC+LPT 组合)时保费才值得。工程上的验证路径:<code>MAGI_ATTENTION_FFA_CUTEDSL_CLC=1</code> 一键开关,benchmark 说话。`
          ]
        }
      ],
      warning: R`Persistent + CLC 模式下 grid 尺寸不再反映问题规模（只反映 SM 数）：任何「用 blockIdx 推断 tile 坐标」的直觉都失效，坐标必须一律经 tile_scheduler 获取。反向 kernel 的调度还有独立的一套（LPT/SPT + head_swizzle，为 deterministic 服务），不要与前向混用结论。`,
      exercises: [
        {
          kind: "计算", level: "基础",
          q: R`seqlen_k=8192、hd=hd_v=128、bf16。一个 KV head 的体积是多少？swizzle 取几？`,
          hint: R`\(8192\times256\times2\) 字节。`,
          answer: R`\(8192\times(128+128)\times2 = 4\,\mathrm{MB}\)。\(50\mathrm{MB}/4\mathrm{MB} = 12.5\)，向下取 2 的幂得 swizzle=8：同一节拍的 CTA 共享 8 个 head 的 K/V（32MB），留余量给 Q/O 流量。`
        },
        {
          kind: "推导", level: "基础",
          q: R`方阵 causal、n 个 Q tile、m 个 SM（n ≫ m）。自然顺序静态分配 vs LPT+动态领取，makespan 各约多少？（以单个 KV block 的处理时间为单位）`,
          hint: R`总工作量 \(\approx n^2/2\)；自然顺序的最后一个任务最重。`,
          answer: R`理想下界 \(n^2/(2m)\)。自然顺序静态：最重的 tile(耗时 n)最后才启动，makespan ≈ \(n^2/(2m) + n\)，尾巴 n 在 n≈m 量级时可占总时长的显著比例。LPT+动态：重块先行，收尾时剩的都是轻块，makespan ≈ \(n^2/(2m) + O(n/m)\)——尾巴缩小 m 倍。`
        },
        {
          kind: "概念", level: "进阶",
          q: R`为什么 CLC 路径不做 L2 swizzle，而静态路径必须做？两者的「顺序控制权」有什么本质区别？`,
          hint: R`谁决定「下一个 tile 给谁」？`,
          answer: R`静态路径中软件完全掌握 tile_idx→坐标映射,可以设计映射让并发 CTA 亲和同一 head——swizzle 是软件行使顺序控制权。CLC 下派发顺序由硬件的空闲状态实时决定,软件看到的只是「结果」;若再对返回坐标做 swizzle 重排,会把硬件按顺序派出的相邻工作打散到不同 head,亲和性反而被破坏。控制权只能有一个主人。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`varlen 调度器用「warp 内前缀和 + ballot」把线性 tile_idx 定位到 (batch, block)。为什么限制「最多 31 个 batch 的前缀和」？超过怎么办？`,
          hint: R`一个 warp 有几条 lane？ballot 返回什么？`,
          answer: R`前缀和由 warp 的 32 条 lane 并行持有——lane i 存放前 i 个 batch 的累计块数，ballot(tile_idx >= prefix) 的置位数直接给出所属 batch，一条指令完成二分。32 条 lane 减去边界哨位可覆盖 31 个 batch；更多 batch 时分多轮（每轮 31 个）迭代,或退化为循环查找。这是「用 warp 当 SIMD 查找表」的经典技巧。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`sched_stages=1 意味着 CLC 预取深度为 1。什么情况下值得把它提到 2？代价是什么？`,
          hint: R`比较 tile 计算时长与 CLC 往返延迟。`,
          answer: R`当单 tile 计算时间短于 CLC 请求往返延迟时（极小 seqlen、高稀疏、tile 大量为空），深度 1 会让 CTA 在 tile 之间露出等待气泡，深度 2 可再藏一层延迟。代价：response buffer 与 mbarrier 各多一份（SMEM 微增），以及「已预取但未消费」的 tile 在 CTA 退出时需要 producer_tail 妥善排空——排空逻辑复杂度随深度上升。FFA 的 tile 普遍够大（128×128×若干 KV block），深度 1 足矣。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`为 block-sparse 前向设计调度顺序：每个 Q tile 的有效 KV 块数 \(w_m\) 由 CSR 表给出（host 可见）。结合 LPT 与 L2 swizzle 的思想给出一个方案，并说明它比「纯 CLC 不排序」好在哪里。`,
          hint: R`LPT 需要的只是每个任务的重量估计,CSR 表恰好给了。`,
          answer: R`方案：host 侧按 \(w_m\)（CSR 的 cnt 数组即重量）对 tile 做桶排序生成派发顺序表；坐标映射先按 swizzle 分节拍（保 L2 亲和），节拍内按 \(w\) 降序（LPT）；分配方式仍可用 CLC——硬件派的是「顺序表的下标」而非原始坐标，兼得动态均衡与软件排序。相比纯 CLC：CLC 只解决「谁来做下一个」，不解决「下一个应该是谁」——重 tile 若排在表尾，动态分配也救不了长尾。排序表把两个自由度解耦，各自交给最擅长的一方。（反向 kernel 的 dq_write_order 已经在用类似的顺序表思想服务 deterministic。）`
        }
      ],
      sources: [
        { label: "源码 · kernel/cutedsl/tile_scheduler.py（全部调度器与 ClcState）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/tile_scheduler.py" },
        { label: "CUTLASS · Blackwell CLC persistent scheduler（ClcDynamicPersistentTileScheduler）", url: "https://github.com/NVIDIA/cutlass" },
        { label: "Graham (1969) · Bounds on Multiprocessing Timing Anomalies（LPT 近似比）", url: "https://epubs.siam.org/doi/10.1137/0117039" },
        { label: "NVIDIA PTX ISA · clusterlaunchcontrol 指令", url: "https://docs.nvidia.com/cuda/parallel-thread-execution/" }
      ]
    },

    /* ================================================================ *
     * 07 · Backward
     * ================================================================ */
    {
      id: "backward",
      order: 7,
      title: "反向传播",
      fullTitle: "Backward on SM100",
      zhTitle: "五个 GEMM 与一次全局归约",
      tag: "SM100 · bwd",
      category: "dense",
      difficulty: "挑战",
      source: "kernel/cutedsl/ffa_bwd_sm100.py · 6001 行",
      deck: R`反向是前向的镜像放大：计算量三倍（5 个 GEMM），依赖两倍（P 要重算，dS 是二级中间量），还多出一个前向没有的难题——dQ 的贡献散布在所有 K tile 上，必须跨 CTA 全局归约。本章用前六章建立的全部语言（TMEM、流水线、mask、调度、原子归约）读完这 6001 行的主干。`,
      takeaway: R`反向的organizing principle 是<strong>「以 K 为家」</strong>：固定一块 K/V，扫过所有相关 Q——于是 \(dK, dV\) 在 TMEM 本地累加（免归约），代价是 \(dQ\) 变成「客场作战」，靠 <strong>TMA bulk atomic-add 到 fp32 的 dQaccum</strong> 全局归约；deterministic 模式不取消 atomic，而是用<strong>信号量把归约顺序钉死</strong>。P 不存不传，用 LSE 重算：\(P = 2^{Sc - \mathrm{LSE}\log_2 e}\)——前向写 LSE 的每一分钱在这里收回。`,
      intuitions: [
        { label: "主场", title: "dK/dV 在家收快递", body: R`一个 n_block 的 \(dK, dV\) 只依赖本块 K/V 与流过的 Q/dO——固定 K 扫 Q,贡献全部本地累加。` },
        { label: "客场", title: "dQ 全局寄账", body: R`每块 K 都给每个 Q 一笔 \(dS\,K\) 的贡献;fp32 的 dQaccum + TMA atomic add 是跨 CTA 的记账本。` },
        { label: "重算", title: "LSE 是重算的钥匙", body: R`前向不存 P（那是 \(O(n^2)\) 的量）,反向用 \(S\) 与 LSE 三条指令重造它,换来显存的自由。` }
      ],
      motivation: [
        R`先写清数学(下方公式)。五个矩阵乘的依赖链是:S(重算)→P(重算)→[dV 用 P; dP→dS 用 P]→[dK, dQ 用 dS]。其中 \(D_i = \operatorname{rowsum}(dO_i \odot O_i)\) 这个每行标量,是 softmax 反向的雅可比收缩项,由独立的 preprocess kernel 提前算好。`,
        R`并行化的抉择:按 Q 分块(前向的方式)则 dQ 本地、dK/dV 全局归约(两个大张量要 atomic);按 K 分块则只有 dQ 要归约。FFA 选后者,而且 2-CTA 模式下 dK 的 GEMM 是 cluster 级的——两个 CTA 的 dS 各占一半,还需要一个专职 <strong>relay warp</strong> 把 peer CTA 的「dS 就绪」信号转发给 leader 的 MMA warp。`,
        R`warp 分工与前向同构但重心迁移:16 warp = 4 reduce(dQ 归约,前向没有的角色) + 8 compute(softmax 重算与 dS,两个 warpgroup) + 1 MMA + 1 load + 1 relay + 1 empty。TMEM 更挤:S/P 叠放、dP/dS 叠放、<strong>dQ 叠在 dP 上</strong>(1-CTA)或嵌在 S 右半(2-CTA)——每一处叠放都对应一条「先消费后覆写」的流水线约束。`
      ],
      diagram: {
        key: "backward",
        caption: "反向三段式：preprocess 产出 D 与 LSE·log2e 并清零 dQaccum；主 kernel 以 K-tile 为家做 5-GEMM 流水与 dQ 原子归约；postprocess 把 fp32 dQaccum 乘 scale 转 dtype。点击节点查看源码。"
      },
      explain: [
        {
          title: "反向的数学链",
          body: [
            R`从 \(O = PV\)、\(P = \operatorname{softmax}(Sc)\) 出发,softmax 的雅可比给出核心等式 \(dS = P \odot (dP - D)\),其中 \(D_i = \sum_j P_{ij}\, dP_{ij} = \operatorname{rowsum}(dO_i \odot O_i)\)——第二个等号是关键化简:不需要 P 就能算 D,所以 preprocess 在主 kernel 之前就能完成(Block 01)。若上游还传来 \(d\mathrm{LSE}\)(例如带 z-loss 的训练),只需 \(D' = D - d\mathrm{LSE}\),一并在 preprocess 处理。`,
            R`scale 的簿记值得单独说:S 的定义带 \(c = \text{softmax\_scale}\),链式法则会在 dQ/dK 上各留一个 \(c\)。实现把它推迟——dS 保持无 scale,dK 在 epilogue 乘 \(c\)(Block 上没展示,行 5598),dQ 在 postprocess 乘 \(c\)(Block 07)——避免在 \(O(n^2)\) 的 dS 上做 \(O(n^2)\) 次乘法,只在 \(O(n d)\) 的输出上乘。`
          ],
          formula: R`\[ D = \operatorname{rowsum}(dO \odot O), \qquad P = 2^{\,S c\,-\,\mathrm{LSE}\cdot\log_2 e}, \]
\[ dV = P^{\mathsf T}\, dO, \qquad dP = dO\, V^{\mathsf T}, \qquad dS = P \odot (dP - D), \]
\[ dQ = c\; dS\, K, \qquad dK = c\; dS^{\mathsf T} Q . \]
<p>验证 \(D\) 的化简：\(D_i = \sum_j P_{ij} dP_{ij} = \sum_j P_{ij} (dO_i \cdot V_j) = dO_i \cdot \sum_j P_{ij} V_j = dO_i \cdot O_i\)。一行点积换掉一次 \(O(n)\) 的归约，这正是 preprocess kernel 存在的理由。</p>`
        },
        {
          title: "以 K 为家的调度与 warp 分工",
          body: [
            R`主 kernel 的 work tile 是一个 <strong>n_block</strong>(K tile):K/V 只 load 一次,Q/dO/LSE/dPsum 沿 m_block 流水送入。<code>BlockInfo.get_m_block_min_max</code>(第 3 章的镜像)按 causal 几何裁掉该 K tile 看不到的 Q 区间。`,
            R`Block 02 列出 5 个 MMA tiler——注意全部写成转置形式(\(S^{\mathsf T} = K Q^{\mathsf T}\)):K 在家、Q 来访,转置让 K 稳坐 A 操作数的 M 维。dV 的 A 来自 TMEM 的 P,dK 的 A 来自 TMEM 的 dS,dQ 的 A 来自 SMEM 的 dS(要跨 CTA 交换,2-CTA 时经 DSMEM all2all)——同一个 dS 按消费者不同走两条路。`,
            R`Block 04 的软件流水是全 kernel 的心跳:<code>S(i) → dK(i-1) → dQ(i-1) → dP(i) → dV(i)</code>,五个 GEMM 首尾相接,每条注释都在声明「为什么这一步不用等」——全部是从 GEMM 发射顺序推出的免费保证,只有两处真正的 wait:dS 就绪(compute→MMA)与 tdQ 被 reduce 消费(TMEM 复用约束)。`
          ],
          svg: "bwd-tmem"
        },
        {
          title: "softmax 重算与 dS",
          body: [
            R`Block 05 上半:compute warpgroup 从 TMEM 读 S,一条 packed FMA 完成 \(F = Sc - \mathrm{LSE}\log_2 e\)(LSE 已被 preprocess 预乘 \(\log_2 e\),Block 01 尾部),一条 exp2 得 P,cvt 成 bf16 写回 S 的 TMEM 左半——与前向 P 叠 S 完全同款的复用。`,
            R`下半:等 dP 的 GEMM 完成,T2R 读 dP,减 dPsum(即 D)、乘 P,得 dS;R2T 写回 dP 的左半(给 dK 用),同时 R2S 写 SMEM(给 dQ 用)。mask 在这里同样只作用于重算的 S(<code>apply_mask_sm100_transposed</code>,swap_AB 版):被 mask 的位置 P=0,dS 自动为 0,梯度天然不泄漏。`,
            R`与前向的深刻区别:反向<strong>没有 online 递推</strong>——max 基准(LSE)是已知常量,P 一次算对,无 corr_scale、无 correction warp。反向的复杂度全部转移到了「五个 GEMM 的排程」与「dQ 的归约」上。`
          ]
        },
        {
          title: "dQ 归约：atomic 与确定性",
          body: [
            R`Block 06:reduce warpgroup 等 dQ 的 GEMM 完成,T2R→R2S 后由 TMA warp 发 <code>cp.async.bulk.reduce.add.f32</code>——TMA 引擎直接在 gmem 的 dQaccum 上做原子加,整 tile 一条指令,不占计算线程。`,
            R`deterministic 的实现哲学:<strong>不取消 atomic,只固定顺序</strong>。每个 (m_block) 配一个信号量,写者按约定的 lock_value 排队:<code>wait_eq(sem, lock)</code> → TMA reduce → <code>arrive_inc</code> 放行下一个。lock 顺序默认按 n_block 递增;causal+SPT 时反转(与 LPT 调度器的遍历方向一致);block-sparse 时由 host 预算的 <code>dq_write_order</code> 表给出。fp32 加法固定顺序 ⇒ 结果 bit 级可复现。代价是写者串行化——所以 deterministic 配套 LPT/head_swizzle 调度,让「同一 m_block 的写者们」尽量错峰到达。`,
            R`Block 07 的 postprocess 收尾:dQaccum(fp32,含全部贡献)→ 乘 softmax_scale → cast 到 bf16 → 经 SMEM 重排后合并写出。dK/dV 不需要这一步——它们在 TMEM 累加完就地转 dtype 写出(GQA 的多 Q-head 归约是例外,走各自的 fp32 accum + semaphore)。`
          ]
        }
      ],
      warning: R`「deterministic=True 消除了 atomic」是最常见的误读——原子加仍在，被固定的只是顺序。另外反向的 mask 是 <code>swap_AB</code> 版本（S 转置布局），行列角色互换；直接搬用前向 mask 的坐标公式会得到转置的错误结果。`,
      exercises: [
        {
          kind: "推导", level: "基础",
          q: R`完成 \(dS = P\odot(dP - D)\) 的推导：从 \(P_i = \operatorname{softmax}(z_i)\) 的雅可比 \(\frac{\partial P_{ij}}{\partial z_{ik}} = P_{ij}(\delta_{jk} - P_{ik})\) 出发。`,
          hint: R`\(dz_{ik} = \sum_j dP_{ij}\,\partial P_{ij}/\partial z_{ik}\)。`,
          answer: R`\(dz_{ik} = \sum_j dP_{ij} P_{ij}(\delta_{jk} - P_{ik}) = P_{ik} dP_{ik} - P_{ik}\sum_j P_{ij} dP_{ij} = P_{ik}(dP_{ik} - D_i)\)，其中 \(D_i = \sum_j P_{ij}dP_{ij}\)。矩阵形式即 \(dS = P\odot(dP - D\mathbf 1^{\mathsf T})\)。`
        },
        {
          kind: "计算", level: "基础",
          q: R`统计反向的 FLOPs：与前向的 2 个 GEMM 相比，5 个 GEMM 的总计算量是前向的几倍？（设 \(d = d_v\)，忽略逐元素项）`,
          hint: R`每个 GEMM 都是 \(2 n_q n_k d\) 量级。`,
          answer: R`前向 \(2\times 2 n_q n_k d\)（QK、PV）。反向 5 个 GEMM 中 S 重算、dP、dV、dK、dQ 各 \(2 n_q n_k d\)，共 5 份——反向 ≈ 前向的 2.5 倍。这与「训练一步 ≈ 3× 前向」的经验法则一致（前向 1 + 反向 2.5，加上重算摊销）。`
        },
        {
          kind: "概念", level: "进阶",
          q: R`如果反向也「以 Q 为家」（固定 Q 扫 K），需要对 dK/dV 做全局归约。对比两种取向的归约流量，说明何时「以 Q 为家」反而更优。`,
          hint: R`归约流量 ∝ 客场张量的大小 × 扫过它的 tile 数。`,
          answer: R`以 K 为家：dQ 归约,流量 ∝ \(n_q d \times n_k/\text{tile}\)。以 Q 为家：dK+dV 归约,流量 ∝ \(2 n_k d \times n_q/\text{tile}\)。比值 = \(n_q : 2n_k\)——当 \(n_q \gg 2 n_k\)(极不对称的 cross-attention、或 GQA 把 Q 折叠后)以 Q 为家更省。对自注意力 \(n_q = n_k\),以 K 为家流量减半,加上 dK/dV 是两个张量(两套 semaphore/accum),工程上也更繁,故 FFA 选 K。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`1-CTA 模式下 tdQ 与 tdP 完全共用 TMEM。据 Block 04 的流水注释，写出保证不冲突的事件顺序链。`,
          hint: R`dP(i) 发射前等的是什么？`,
          answer: R`顺序链：dQ(i-1) GEMM 完成（写 tdQ）→ reduce warpgroup T2R 读走 tdQ → 发 <code>pipeline_dQ.sync_object_empty</code> → MMA 的 dP(i) 才发射（向同一块 TMEM 写 dP）→ dP 被 compute 消费转成 dS → 下一轮 dQ 又写回来。即「dQ 写→dQ 读空→dP 写→dP 读空」的四拍循环，<code>pipeline_dQ</code> 的 full/empty 两个方向各守一拍。这就是注释 "tdQ is overlapped with tdP" 背后的完整契约。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`deterministic 模式的信号量把同一 m_block 的写者串行化。估算最坏情形的串行化代价，并解释 SPT（反转 lock 顺序）为什么能缓解它。`,
          hint: R`causal 下 m_block 大的 Q tile 被多少个 n_block 写？调度器按什么顺序派 n_block？`,
          answer: R`causal 下最后一个 m_block 被全部 \(n\) 个 n_block 写,若各写者同时到达,串行 TMA reduce 排队长 \(n\)。SPT 让 lock 顺序与 LPT 调度的派发顺序<strong>同向</strong>:先派发的 n_block(LPT 下是编号大的)恰好持有小 lock 值,先到先写;后派发的到达时前任大概率已完成——排队变成流水。顺序错配时(lock 升序、派发降序)最早到达的写者持最大 lock,要等所有后来者,退化为全串行。教训:确定性顺序必须与调度顺序协同设计。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`GQA(qhead_per_kvhead=8)时,8 个 Q head 的 dK/dV 贡献落到同一个 KV head 上。源码走「fp32 accum + 可选 semaphore」的 dKV_postprocess 路径。设计一个免全局归约的替代方案并分析代价。`,
          hint: R`归约的另一个维度是「调度」:能否让同一 KV head 的 8 个 Q head 由同一 CTA 处理？`,
          answer: R`替代方案:把 (n_block, kv_head) 作为 work tile,CTA 内沿 Q-head 维内层循环——8 个 Q head 的 dS 依次流过,dK/dV 在 TMEM 上跨 head 连续累加,写出时已是完整和,无需 atomic。代价:(1) 单 tile 工作量 ×8,persistent 负载粒度变粗,尾部均衡变差(可用 CLC 缓解);(2) Q/dO 的 load 量不变但访问跨 head 步长大,L2 亲和下降;(3) 寄存器/SMEM 中 LSE、dPsum 要按 head 轮换,流水线 stage 数受挤压。本质是拿「调度自由度」换「归约流量」——与前向 PackGQA 把 head 折进 seqlen 维是同一枚硬币的两面。`
        }
      ],
      sources: [
        { label: "源码 · kernel/cutedsl/ffa_bwd_sm100.py（主 kernel）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/ffa_bwd_sm100.py" },
        { label: "源码 · kernel/cutedsl/ffa_bwd_preprocess.py（D 与 LSE·log2e）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/ffa_bwd_preprocess.py" },
        { label: "源码 · kernel/cutedsl/ffa_bwd_postprocess.py（dQ 收尾）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/kernel/cutedsl/ffa_bwd_postprocess.py" },
        { label: "FlashAttention-2 论文 · 附录 B（backward 推导的标准出处）", url: "https://arxiv.org/abs/2307.08691" },
        { label: "MagiAttention 博客 · FA2 数学推导（含 backward 全链）", url: "https://sandai-org.github.io/MagiAttention/docs/blog/fa2_math_derivation/" }
      ]
    },

    /* ================================================================ *
     * 08 · Communication–Computation Overlap
     * ================================================================ */
    {
      id: "overlap",
      order: 8,
      title: "通算融合",
      fullTitle: "Communication–Computation Overlap",
      zhTitle: "通信如何藏进计算的影子里",
      tag: "分布式 · CP",
      category: "hybrid",
      difficulty: "挑战",
      source: "functional/dist_attn.py · 3806 行",
      deck: R`前八章的 kernel 无论多快，只解决单卡问题；百万 token 的序列必须切到几十上百个 CP rank 上，而注意力天生要「看见别人的 KV」。本章读 MagiAttention 分布式的心脏 <code>DistAttnRuntime</code>：GroupCast/GroupReduce 两条通信原语、多阶段 overlap 流水线、以及 <code>sm_margin</code> / KernelBarrier 这两种让通信 kernel 与 persistent 计算 kernel 分享 GPU 的方式——第 0 章埋下的每一个伏笔（可合并的 out/lse、sm_margin、GroupReduce）都在这里兑现。`,
      takeaway: R`通算融合的完整公式是<strong>「切阶段 + 三重奏 + 让 SM」</strong>：把远端 KV 按 overlap solver 的成本模型切成 \(d\) 个 stage；每个时刻让三件事并行——<strong>第 \(i{+}1\) 段的 GroupCast 预取、第 \(i\) 段的 FFA partial attention、第 \(i{-}1\) 段的归约</strong>；而这一切能真正并行的物理前提，是 FFA persistent kernel 通过 <code>sm_margin</code> 给通信 kernel 留出 SM（或 native grpcoll 用 KernelBarrier 保证发射顺序）。归约本身零成本创新：kernel 的 atomic 累加缓冲（<code>out_acc/lse_acc</code>）直接吃掉第 0 章的半群合并公式。`,
      intuitions: [
        { label: "原语", title: "GroupCast / GroupReduce", body: R`「一段数据发给多个 rank」与「多段 partial 结果按半群归约到属主」——不规则 mask 的通信不再是 ring，而是按依赖精确投递。` },
        { label: "流水", title: "三重奏错峰", body: R`prefetch(i+1) ∥ compute(i) ∥ reduce(i−1)：通信藏进计算的影子里,理想情况下端到端时间 = max(通信,计算) 而非两者之和。` },
        { label: "地皮", title: "SM 是要分的", body: R`NCCL/native 通信 kernel 也要 SM。persistent 的 FFA kernel 少开 sm_margin 个 CTA,通信才有地皮真正并行——overlap 不是免费的时间魔法。` }
      ],
      motivation: [
        R`Ring-Attention 一类环形 CP 方案假设每个 rank 需要「按固定拓扑轮转」所有 KV——对规则的 full/causal mask 尚可,对 Magi-1 的 varlen block-causal 这类<strong>不规则 mask</strong>会产生大量无效通信：很多 (Q,K) 对根本不在 mask 里,却仍要跟着环转一整圈。MagiAttention 的答案与 kernel 侧一脉相承：先用 AttnSlice 把依赖精确化（第 0 章）,dispatch solver 均衡负载后,每个 rank 需要哪些远端 KV 是一张<strong>精确的清单</strong>——通信原语只需按清单投递。这就是 GroupCast（KV 去程）与 GroupReduce（partial 结果回程）。`,
        R`有了原语还不够：一次性拉全所有远端 KV 再计算,通信完全暴露在关键路径上。<code>DistAttnRuntime</code> 把远端 KV 切成 <code>overlap_degree</code> 个 stage 组成流水线,用异步句柄 <code>WorkWithPostProcessFn</code> 把「等通信」推迟到「真正要用那份数据的前一刻」。切几段、每段装哪些 rank 的数据,由 <code>OverlapSolver</code> 按成本模型求解——这是一个和第 6 章 LPT 同款的「调度问题」,只是排的不是 CTA 而是通信段。`,
        R`最后一块拼图在 GPU 内部：通信 kernel（NCCL 的 a2av 或 native grpcoll 的收发 kernel）与计算 kernel 抢 SM。若 FFA kernel 占满全部 SM,异步发出的通信会被排队到计算之后——overlap 名存实亡。所以第 0 章见过的 <code>sm_margin</code> 在这里揭晓真身：FFA persistent kernel 故意少开若干 CTA,把 SM 留给通信;而 native grpcoll 路径反其道行之——通信 kernel 常驻 SM,用 <code>KernelBarrier</code> 计数器保证「先发射的计算 kernel 先拿到地皮」,此时 sm_margin 归零。`
      ],
      diagram: {
        key: "overlap",
        caption: "通算融合全景：meta 层（OverlapSolver 切 stage）→ 前向 overlap 环（prefetch/compute/reduce 三重奏）→ 通信原语层（group_cast 三种实现）→ SM 分配（sm_margin vs KernelBarrier）。点击节点查看对应源码。"
      },
      explain: [
        {
          title: "GroupCast / GroupReduce：为不规则依赖定制的集合通信",
          body: [
            R`两条原语的签名是理解一切的起点（Block 04）。<strong>group_cast</strong>：输入张量按 <code>input_split_sizes</code> 切段,每段带一个目的 rank 列表 <code>dst_indices[i]</code>（可多播）;输出侧 <code>output_split_sizes + src_index</code> 描述收到的段从谁来、按什么顺序排。<strong>group_reduce</strong> 是它的对偶（Block 05）：每段带一个目的 rank（<code>dst_index</code>）,属主侧按 <code>src_indices</code> 把多个来源的同一段<strong>归约</strong>——reduce_op 除了 sum/avg,还有专为注意力设计的 <code>"lse"</code>：用第 0 章的合并公式做 log-sum-exp 加权平均。`,
            R`默认实现把 group_cast <strong>降解到 NCCL all2all_v</strong>（Block 06）：多播段先在本地复制展开成「每个目的 rank 一份」的连续发送缓冲,a2av 一发;接收侧因为「同源段保序」,只需一次索引重排——这个重排被打包成 <code>post_process_fn</code>,挂在异步句柄上,等待完成后才执行。group_reduce 同理降解：a2av 收齐各来源的 partial 段,post-process 里做本地 sum/lse 归约。代价是显式的 pack/unpack 拷贝与多播复制——这正是 native grpcoll 存在的理由。`,
            R`<strong>native grpcoll</strong>（DeepEP 风格,Block 04 的第二分支）用 NVLink/RDMA 对称缓冲区直接做多播与归约,省去 pack/unpack,fp32 归约在通信 kernel 内完成（<code>comm_dtype</code> 可低精度传输、高精度累加）;<strong>hierarchical</strong> 分支把「节点内多播」与「跨节点传输」拆成两级 a2av,同一段数据跨节点只走一次 RDMA,到达后再在节点内 NVLink 扇出——三种实现共用一个签名,对上层 <code>DistAttnRuntime</code> 完全透明。`
          ],
          formula: R`<p>group_reduce 的 <code>"lse"</code> 归约在通信层复用第 0 章的半群公式：属主收到 \(r\) 个 partial \((out_k, lse_k)\) 后计算</p>
\[ \mathrm{lse} = \log\!\sum_k e^{\mathrm{lse}_k}, \qquad \mathrm{out} = \sum_k e^{\mathrm{lse}_k - \mathrm{lse}}\, out_k . \]
<p>交换律 + 结合律意味着：段怎么切、先到后到、在 kernel 里归约还是在通信里归约,结果数学上同一——这是整个多阶段流水线可以任意重排的根本许可。</p>`
        },
        {
          title: "多阶段 overlap 环：prefetch / compute / reduce 三重奏",
          body: [
            R`<code>DistAttnFunc.forward</code>（Block 01）是一台精确的三拍机器。<strong>host stage</strong>：先发起 stage-0 的远端 KV 预取,立即用本地 KV 算 partial attention——第一段通信藏在本地计算后面。<strong>主环</strong>第 \(i\) 拍做三件事：① <code>get_curr_q_kv_and_fetch_next</code> 等 stage-\(i\) 的 KV 到位、同时发起 stage-\(i{+}1\) 的预取（Block 02）;② 用 stage-\(i\) 的远端 KV 发射 FFA kernel;③ 把 stage-\(i{-}1\)（FFA 后立即）的 partial 结果交给 <code>reduce_partial_out_lse</code> 异步归约。三类工作分别压在通信流、计算流上,互相只以「异步句柄 + 等待点」耦合。`,
            R`归约有一个优雅的零通信特例：默认（KV 走通信,Q 不动）模式下,每个 rank 算的都是<strong>自己的 Q</strong> 对不同 KV 段的 partial 结果——它们本来就在本地！FFA backend 直接把上一拍的 \((out, lse)\) 作为 <strong>累加缓冲</strong>（<code>out_acc/lse_acc</code>）传入 kernel（Block 07）：atomic 归约在 kernel 尾声完成合并,连 <code>correct_attn_out_lse</code> 的显式调用都省了。GroupReduce 真正出场是在两处：反向的 dKV 必须回到 KV 属主（sum 归约）,以及开启 qo_comm 时 partial out/lse 回到 Q 属主（lse 归约）。`,
            R`预取的姿势由 <code>prefetch_stage_by_stage</code> 决定（Block 03）：默认在 host stage 一次把<strong>所有</strong> stage 的 group_cast 全部发出——persistent kernel + sm_margin 保证它们与后续计算并行;但当 <code>CUDA_DEVICE_MAX_CONNECTIONS=1</code>（硬件队列只有一条,先入队者阻塞后来者）或 native grpcoll（缓冲区按 stage 复用,不能同时在飞）时,退化为逐 stage 预取——每拍只提前一步。这是「调度自由度」与「资源占用」的经典折衷。`
          ],
          svg: "overlap-timeline"
        },
        {
          title: "OverlapSolver：切几段、每段装什么",
          body: [
            R`<code>OverlapConfig.degree</code> 的语义谱系（Block 08）值得背下来：<code>degree=0</code> 完全不 overlap——阻塞式 group_cast 拉全 KV、拼接后<strong>一次</strong> FFA 调用,换来的是彻底绕开 LSE 合并的精度损失（校验用基线）;<code>degree=1</code> 本地 + 1 个远端段（通信只能藏在 host 计算后面）;<code>degree=N</code> 静态多段;<code>degree=None</code> 动态模式——solver 逐个 degree 试解,取总时延最短者。`,
            R`成本模型（Block 09）把每个候选 chunk 记为 \((C^{\mathrm{comm}}_j, C^{\mathrm{calc}}_j)\)（通信量与 mask 面积各乘一个标定系数）,一个划分方案的总时延按「第 \(i\) 段通信与第 \(i{-}1\) 段计算完美互相掩盖」的假设估算（下方公式）。当前默认解法是朴素的均匀划分（<code>UniformOverlapAlg</code>）,贪心解法留着 TODO——但框架已把「求最优 stage 划分」形式化成了与第 6 章 LPT 同类的 makespan 极小化问题。`,
            R`划分的粒度下界由 <code>min_chunk_size=512</code>、<code>max_num_chunks=64</code> 控制：段切太碎,每段的 kernel 发射开销与 a2av 固定延迟会吃掉 overlap 的收益;切太粗,首段通信藏不进 host 计算。这组超参与 kernel 侧 tile=128 的角色完全同构——流水线的深度与粒度永远是一对矛盾。`
          ],
          formula: R`<p>overlap 划分 \(\{P_0,\dots,P_{d-1}\}\) 的总时延估计（<code>OverlapSolver._calc_overall_cost</code>）：</p>
\[ T \;=\; \max\!\Big(\textstyle\sum_{j\in P_0} C^{\mathrm{comm}}_j,\; C^{\mathrm{calc}}_{\mathrm{host}}\Big) \;+\; \sum_{i=1}^{d-1}\max\!\Big(\textstyle\sum_{j\in P_i} C^{\mathrm{comm}}_j,\; \sum_{j\in P_{i-1}} C^{\mathrm{calc}}_j\Big) \;+\; \sum_{j\in P_{d-1}} C^{\mathrm{calc}}_j . \]
<p>每一项都是 \(\max(\text{通信}_i, \text{计算}_{i-1})\)——相邻拍完美互掩的假设。理想的解让每个 \(\max\) 的两个参数相等：通信恰好被计算填满,首段通信恰好被 host 计算填满,只有最后一段计算裸露。这正是「线性可扩展」的数学条件：只要每 rank 的计算量随 CP 度不变（dispatch 均衡）且通信不超过计算,总时延与 CP 度无关。</p>`
        },
        {
          title: "SM 的分配：sm_margin 与 KernelBarrier",
          body: [
            R`异步 ≠ 并行。NCCL 的 a2av 也是 GPU kernel,要拿到 SM 才开始搬数据;FFA 是 persistent kernel（第 6 章）,默认按「每 SM 一个 CTA」铺满整卡。若不干预,通信 kernel 只能排在 FFA 之后——nsys 时间线上是「串行的两段」,overlap 完全落空。<code>fwd_sm_margin</code>（Block 03）是解法一：host 侧把 <code>sm_count - margin</code> 传给 tile scheduler,FFA 少开 margin 个 CTA,通信 kernel 从发射起就有地皮,真正与计算并行。margin 大小由环境变量标定——留太多,计算变慢;留太少,通信带宽上不去,是一个按硬件与模型形状实测的旋钮。`,
            R`native grpcoll 是解法二,哲学相反：通信 kernel 自己常驻 SM（<code>GrpCollConfig.num_sms</code>,默认 24,每 2 个 SM 一条 channel）,反而是<strong>发射顺序</strong>变成新风险——通信 kernel 若先于它依赖的计算 kernel 拿到 SM,会空转占地皮甚至死锁。<code>KernelBarrier</code>（Block 01 开头）是一个 GPU 上的计数器：计算 kernel 尾声 arrive,通信 kernel 开头 spin-wait 到目标值——用微型的「kernel 间 mbarrier」把发射顺序钉死,于是 <code>fwd_sm_margin</code> 返回 0：地皮已由通信 kernel 自带,无需再留。`,
            R`反向把同样的机器再跑一遍,多两个角色：dKV 的 GroupReduce（sum,回 KV 属主）与 dQ 本地累加,而且尾巴上有一个专门的优化——最后一个 stage 的 dKV 归约没有后续计算可掩,<code>save_tail_stage</code> 把前向存下的末段 KV 让反向「借尸还魂」：调换计算顺序,让末段归约与下一层的反向计算重叠（<code>_hide_tail_stage_reduce_backward</code>）。通算融合的每一处细节都在重复同一句话：<strong>找到还没被占用的影子,把通信塞进去</strong>。`
          ]
        }
      ],
      warning: R`不要把「overlap_degree 越大越好」当作结论——每加一段就多一次 kernel 发射、一次 a2av 固定延迟、一份 grpcoll 缓冲;当单段通信时间已小于计算时,加段只增开销。也不要混淆两条归约路径：默认 KV-comm 模式下前向的 out/lse 合并<strong>不走网络</strong>（kernel 累加缓冲完成）,GroupReduce 的网络流量在反向 dKV 与 qo_comm 模式——引用「通信量」数据时务必注明模式。`,
      exercises: [
        {
          kind: "概念", level: "基础",
          q: R`CP=4,rank 2 的某段 KV 同时被 rank 0 和 rank 3 的 Q 需要,而 rank 2 自己的 Q 不需要它。写出这段 KV 在 group_cast 的 <code>dst_indices</code> 中的条目,以及反向时对应 dKV 段在 group_reduce 中的 <code>src_indices</code> 条目。`,
          hint: R`group_reduce 是 group_cast 的镜像。`,
          answer: R`前向 group_cast（rank 2 视角）：该段的 <code>dst_indices[i] = [0, 3]</code>——一段数据多播给两个消费者。反向 group_reduce（rank 2 是属主）：该段的 <code>src_indices[i] = [0, 3]</code>——rank 0 和 rank 3 各算出一份 partial dK/dV,按 sum 归约回 rank 2。镜像对称正是「通信按依赖清单精确投递」的体现：不在 mask 里的 (Q,K) 对,一个字节都不会上网络。`
        },
        {
          kind: "推导", level: "基础",
          q: R`设 host 计算 8ms,三个远端段的（通信,计算）分别为 (6,7)、(5,6)、(4,5) ms。分别计算 degree=3 流水线与「先拉全再算」两种方案的总时延。`,
          hint: R`套用 _calc_overall_cost 的公式。`,
          answer: R`流水线：\(T = \max(6,8) + \max(5,7) + \max(4,6) + 5 = 8+7+6+5 = 26\) ms——三段通信全部藏进影子,总时延恰为纯计算 \(8+7+6+5=26\)。先拉全再算：通信 \(6+5+4=15\)（即使三路并发也受带宽约束按串行算）,再计算 \(8+7+6+5=26\),共 41ms。overlap 省下的正是全部 15ms 通信——前提是每段通信都不长于前一段计算。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`Block 01 的主环里,<code>reduce_partial_out_lse</code> 在默认 KV-comm 模式下并不发起任何通信（Block 07 显示合并已由 kernel 的 out_acc 完成）,但它仍然为每个 stage 压入一个 <code>WorkWithPostProcessFn</code> 并在 <code>prepare_reduced_local_out_lse</code> 里逐个 wait。解释这层「空壳异步」的设计意图。`,
          hint: R`看 qo_comm 模式下同一个调用点会发生什么。`,
          answer: R`这是接口的<strong>模式无关性</strong>设计：qo_comm 模式下同一调用点返回真正的 group_reduce 异步句柄（lse 归约在网络上跑）,KV-comm 模式下返回 no-op 句柄（post_process 恒等）。主环代码于是完全不感知模式差异——「每个 stage 交一个归约句柄、结束前统一 wait」的契约恒成立。空壳的代价是几个 Python 对象,换来的是 forward 主环零分支、以及未来加新归约路径（如 native grpcoll 的 fp32 缓冲归约）时不动主环。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`<code>CUDA_DEVICE_MAX_CONNECTIONS=1</code> 时,为什么「host stage 一次发出全部 prefetch」会几乎杀死 overlap？而留下的例外是「只有最后一个 stage 的 prefetch 还能重叠」——解释这个例外。`,
          hint: R`想象单条硬件队列里的入队顺序：cast0, cast1, cast2, ffa_host, ffa_0, ...`,
          answer: R`单 connection 下所有 stream 的工作折叠进一条硬件队列,queue 头部阻塞后续所有条目。一次发全 prefetch 的入队序是 [cast0, cast1, cast2, ffa_host, ...]：ffa_host 要等三个 cast 全部出队才开始——通信不是藏在计算影子里,而是把计算推后,恰好反转。逐 stage 预取的入队序是 [cast0, ffa_host, cast1, ffa_0, cast2, ...]：每个 cast 只挡住它<strong>本来就该等的</strong>下一拍计算的发射点,cast_i 与 ffa_{i-1} 的执行仍然重叠。「最后一个 stage 例外」：cast_{d-1} 之后没有更多 cast 入队,它后面的 ffa 发射不再被新通信阻塞,所以哪怕一次发全,最末段通信仍与它前面的计算天然重叠。`
        },
        {
          kind: "系统", level: "挑战",
          q: R`sm_margin 的两难：设 FFA 计算在 148 个 SM 上耗时 \(T_c\),通信在带宽不受 SM 数约束时耗时 \(T_m\)。留 margin \(s\) 后计算变为 \(T_c \cdot \frac{148}{148-s}\)（线性模型）。写出端到端时延 \(T(s)\) 并给出最优 margin 的判据;解释为什么 native grpcoll + KernelBarrier 能同时改善这个 trade-off 的两端。`,
          hint: R`s 太小通信排队（退化为串行）,s 够大才真正并行。`,
          answer: R`\(s>0\) 且通信 kernel 能被容纳时 \(T(s) = \max\!\big(T_c\frac{148}{148-s},\, T_m(s)\big)\)（\(T_m(s)\) 随 s 增大先降后平——通信自身也需要足够 SM 达到线速）;\(s=0\) 时退化为 \(T_c + T_m\)。最优 s 在两条曲线交点：计算膨胀恰好等于通信耗时。native grpcoll 改善两端：(1) 通信 kernel 是定制的收发/归约 kernel,单 SM 效率高于 NCCL 通用路径,\(T_m(s)\) 曲线整体下移、更小的 s 即达线速;(2) fp32 归约融合进通信 kernel,省掉 NCCL 路径上额外的本地归约 kernel（那也要抢 SM）;KernelBarrier 则消除了「怕通信 kernel 抢跑」而保守多留 margin 的需要——sm_margin 直接归零,计算不再膨胀。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`MagiAttention 论文声称「线性可扩展」：固定每 rank 序列长度,CP 度翻倍时迭代时间近似不变。用本章的成本模型写出线性可扩展成立的两个充分条件,并各举一个会破坏它的现实场景。`,
          hint: R`一个条件关于 dispatch,一个关于 comm/calc 之比。`,
          answer: R`条件一（计算侧）：dispatch solver 使每 rank 的 mask 面积（\(\sum C^{\mathrm{calc}}\)）与 CP 度无关——即负载均衡是完美的。破坏场景：极端不规则 mask 下最重的单个 AttnSlice 超过均值,minheap 也无法切开它（切片粒度下界）,该 rank 成为常驻掉队者。条件二（通信侧）：每个 stage 满足 \(\sum_{j\in P_i} C^{\mathrm{comm}}_j \le \sum_{j\in P_{i-1}} C^{\mathrm{calc}}_j\),即通信永远藏得进计算——由于 causal 类 mask 下计算 \(\propto\) 面积、通信 \(\propto\) 边长,序列越长该不等式越宽松。破坏场景：跨节点带宽骤降（如 RDMA 降级到 TCP）使 \(C^{\mathrm{comm}}\) 乘一个大系数,或 head_dim 很小、mask 很稀疏使单位通信量对应的计算量不足——通信从影子里露出来,scaling 曲线弯折。`
        }
      ],
      sources: [
        { label: "源码 · functional/dist_attn.py（DistAttnRuntime 与 overlap 主环）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/functional/dist_attn.py" },
        { label: "源码 · comm/primitive/grpcoll/_group_collective.py（group_cast/group_reduce 三层实现分发）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/comm/primitive/grpcoll/_group_collective.py" },
        { label: "源码 · meta/solver/overlap_solver.py（OverlapConfig 与成本模型）", url: "https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/meta/solver/overlap_solver.py" },
        { label: "MagiAttention 博客 · 主博文（GroupCast/GroupReduce 与 multi-stage overlap 的设计动机）", url: "https://sandai-org.github.io/MagiAttention/docs/blog/magi_attn/" },
        { label: "MagiAttention 论文 · A Distributed Attention Towards Linear Scalability", url: "https://arxiv.org/abs/2505.13211" },
        { label: "Ring Attention（对照方案：环形拓扑的通信重叠）", url: "https://arxiv.org/abs/2310.01889" }
      ]
    }
  ];
})();
