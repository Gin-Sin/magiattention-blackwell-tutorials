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
      deck: R`先不看 kernel（GPU 上执行的计算函数）代码，只看算子的输入和输出。MagiAttention 把复杂的注意力 mask（掩码）拆成若干 <code>AttnSlice</code>（注意力切片）。每个切片只说明三件事：哪些查询 Q、哪些键 K，以及它们之间使用哪种简单 mask。各切片可以分别计算，再用局部输出 <code>out</code> 和对数归一化项 <code>lse</code> 合并结果。本章还会明确 Blackwell SM100 上的 CuTe DSL 实现路径目前支持到哪一步。`,
      takeaway: R`把 <code>AttnSlice</code> 理解成一块矩形区域：<code>QRange</code> 选出若干 query（查询）行，<code>KRange</code> 选出若干 key（键）列，<code>MaskType</code> 决定矩形内部哪些位置有效。复杂 mask 可以由多块矩形拼成。每块单独计算后，利用 fp32（32 位浮点）<code>out</code> 和 LSE（log-sum-exp，对数指数和）即可合并。`,
      intuitions: [
        { label: "分解", title: "任意 mask = 矩形切片之并", body: R`每个切片是一块连续的二维 Q–K 区域，内部只有一种简单几何。切片列表在 CP（context parallelism，上下文并行）的各个 rank（进程编号）之间重新分配后依然合法。` },
        { label: "合并", title: "局部结果可以再合并", body: R`两份局部结果 \((out_1,lse_1)\) 和 \((out_2,lse_2)\) 有稳定的合并公式；其中下标 1、2 表示两个局部切片。实数运算下顺序不影响结果，浮点运算则可能有微小舍入差异。` },
        { label: "现状", title: "先区分两条实现路径", body: "SM100 CuTe DSL 路径目前要求 ranges（区间列表）连续且不重叠，并只原生处理 full/causal。完整四种类型由上层切片或 SM90 JIT（即时编译）路径处理。" }
      ],
      motivation: [
        R`规则 mask（如 causal，即因果掩码，或滑动窗口）容易直接写进 kernel；不规则 mask 会带来大量分支、负载不均和无效通信。Flex-Flash-Attention（FFA）不继续增加分支，而是先把 mask 拆成规则切片。kernel 只处理这些简单切片。`,
        R`仓库有两条 FFA 实现。<strong>functional/JIT 路径</strong>面向 SM90 生产使用，支持重叠 ranges 和四种 mask。<strong>CuTe DSL 路径</strong>面向 SM100，是本教程重点，但当前只实现了契约的一部分。后文出现“支持”时都会注明是哪条路径。`,
        R`本章从 CuTe DSL 的 host 入口 <code>_flex_flash_attn_fwd</code> 开始。host 指 CPU 侧的 Python 调用代码；它负责检查输入、选择 tile 和调度配置、生成编译缓存键，再启动 GPU kernel。`
      ],
      diagram: {
        key: "attnslice",
        caption: "AttnSlice 契约与 host 侧一次前向的旅程：ranges 折叠 → 启发式决策 → compile_key → SM100 kernel 实例化 → tvm-ffi 启动。点击节点查看对应源码。"
      },
      explain: [
        {
          title: "AttnSlice 三元组与四种 mask 几何",
          body: [
            R`<code>AttnMaskType</code> 定义四种矩形内部的形状：<code>FULL</code> 全部有效；<code>CAUSAL</code> 保留下三角；<code>INVCAUSAL</code> 保留上三角；<code>BICAUSAL</code> 取上下三角的交集。这里的“右下对齐”表示 Q 和 K 的末尾对齐，“左上对齐”表示二者开头对齐。`,
            R`设切片有 \(s_q\) 行 Q、\(s_k\) 列 K。若两者长度不同，对齐方向会改变三角形的位置。例如 \(s_q \lt s_k\) 时，右下对齐的 CAUSAL 区域会变成梯形。先明确长度和对齐方向，再看公式，就不会把它误认为普通的 \(k\le q\)。`,
            R`全局 mask 就是所有切片有效位置的并集。切片允许重叠，因此同一 Q 行可能得到多份局部结果；这些结果需要在最后合并。`
          ],
          svg: "attnslice-masktypes",
          formula: R`<p>先定义符号：\(q\) 和 \(k\) 是全局位置，\(q_{\mathrm{start}}\) 和 \(k_{\mathrm{start}}\) 是当前切片的起点；\(q'=q-q_{\mathrm{start}}\)、\(k'=k-k_{\mathrm{start}}\) 是切片内从 0 开始的相对位置；\(s_q\)、\(s_k\) 是切片内 Q、K 的长度。右下对齐的 CAUSAL 条件为</p>
\[ k' \le q' + (s_k-s_q). \]
<p>例：\(s_q=3,s_k=5\) 时，三行 Q 可看到的最大 K 下标依次为 2、3、4，正好让两者末尾对齐。INV-CAUSAL 为 \(k'\ge q'\)，BI-CAUSAL 同时满足两式。每行的有效 K 都是连续区间，因此 kernel 只需计算区间边界。</p>`
        },
        {
          title: "out / lse 的可累加语义",
          body: [
            R`同一个 Q token（序列位置）可能在多个切片或多个 CP rank 上分别计算。每处只得到局部 softmax。为了合并这些局部结果，需要同时保留输出 <code>out</code> 和归一化信息 <code>lse</code>。LSE 是 log-sum-exp，空集合记为 \(-\infty\)。`,
            R`functional（上层函数式接口）层的默认策略是：只要 Q ranges 可能重叠，就把输出类型 <code>out_type</code> 设为 <code>torch.float32</code>，并用 <code>atomicAdd</code>（原子加）归约；调用方显式声明「Q 不重叠」（<code>disable_fwd_atomic_reduction=True</code>）才允许按输入数据类型 <code>dtype</code> 直写。CuTe DSL 路径当前因为 ranges 等价于累积序列边界 <code>cu_seqlens</code>（天然不重叠），直接按 <code>q.dtype</code> 写出。`,
            R`Block 03 展示了 cutedsl 侧的另一处契约细节：空输入时 <code>out.zero_()</code>、<code>lse.fill_(-inf)</code>——\(-\infty\) 正是「对空集合做 log-sum-exp」的正确单位元，保证后续合并公式无需特判。`
          ],
          formula: R`<p>两段 partial 结果的合并（<code>magi_attention/functional/utils.py</code>，<code>correct_attn_out_lse</code>）：</p>
\[ \mathrm{lse} = \log\!\big(e^{\mathrm{lse}_1} + e^{\mathrm{lse}_2}\big) = \max(\mathrm{lse}_1,\mathrm{lse}_2) + \operatorname{softplus}\!\big(\min - \max\big), \]
\[ w_i = e^{\mathrm{lse}_i - \mathrm{lse}}, \qquad \mathrm{out} = w_1\,\mathrm{out}_1 + w_2\,\mathrm{out}_2 . \]
<p>在实数运算中，这个合并满足交换律和结合律，空结果 \((0,-\infty)\) 还是单位元，因此任意多段都能分组归约。实际 fp32 会因加法顺序不同产生很小的舍入差异。</p>`
        },
        {
          title: "两套 FFA 栈：JIT 生产路径 vs CuTe DSL 路径",
          body: [
            R`<strong>functional/JIT 路径</strong>支持重叠 ranges、四种 mask、自动去重和原子归约，是当前分布式训练主路径。`,
            R`<strong>CuTe DSL 路径</strong>原生运行在 SM100，并支持分数/掩码修改函数 <code>score_mod/mask_mod</code>、块稀疏、PackGQA（将分组查询注意力的多个 Q 头折叠打包）和 PagedKV（分页式 KV 缓存）。但 <code>ranges_to_cu_seqlens</code> 仍要求 ranges 从 0 开始、连续且不重叠，mask 类型也只有 full/causal。`,
            R`因此两条路径各有重点：前者语义更完整，后者展示 Blackwell 原生实现。不要把一条路径的能力直接套到另一条。`
          ]
        },
        {
          title: "host 侧一次前向的旅程",
          body: [
            R`host 先把掩码类型 <code>mask_type</code> 转成简单的因果标志 <code>causal</code>，再选择 tile（分块）大小、Q stage（流水线中的一组 Q 分块）、是否让两个 CTA 协作，以及是否启用 CLC（Cluster Launch Control，簇启动控制）调度。CTA 是一组协作线程；这里一个 Q stage 对应 128 行 Q。具体启用条件会在后续章节逐项解释。`,
            R`<code>compile_key</code> 是编译缓存键。凡是会改变生成代码的配置，例如 dtype、head_dim、mask、tile 和调度模式，都必须放进 key。某个 key 第一次出现时编译，之后直接复用缓存。`
          ]
        }
      ],
      warning: R`“FFA 支持四种 mask”说的是整体接口，不等于 SM100 CuTe DSL kernel 已原生实现四种分支。当前该 kernel 只识别 full/causal；引用能力时请注明实现路径。`,
      exercises: [
        {
          kind: "概念", level: "基础",
          q: R`一个 varlen（variable-length，变长）causal mask（3 条序列，长度 512/1024/256，各自内部 causal）最少需要几个 AttnSlice？写出每个切片的 <code>mask_type</code>（掩码类型）。`,
          hint: R`每条序列内部是一个右下对齐的 causal 矩形。`,
          answer: R`3 个切片：\((Q_i, K_i, \text{CAUSAL})\)，其中 \(Q_i = K_i\) 为每条序列自身的 range。varlen causal 恰是「每序列一个 CAUSAL 切片」的特例，这也是它能折叠成 cu_seqlens 的原因。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`证明 LSE 合并公式的数值安全性：为什么实现写成 \(\max + \operatorname{softplus}(\min-\max)\) 而不是直接 \(\log(e^{l_1}+e^{l_2})\)？当 \(l_1 = -\infty\) 时会发生什么？`,
          hint: R`考虑 \(l_i \approx 80\) 时 \(e^{l_i}\) 的表示范围。这里 \(\operatorname{softplus}(x)=\log(1+e^x)\)，且 \(\operatorname{softplus}(-\infty)=0\)。`,
          answer: R`\(e^{80}\approx 5.5\times10^{34}\) 在 fp32 中仍可表示，但 fp32 的自然指数约在 \(x>88.7\) 时上溢，因此直接计算依赖 LSE 的绝对大小。改写后唯一的指数项满足 \(e^{\min-\max}\le 1\)，不会上溢。当 \(l_1=-\infty\) 时，公式退化为 \(\mathrm{lse}=l_2\)，说明空集是合并单位元。源码中的 <code>safe_subtract</code> 还避免了 \(-\infty-(-\infty)\) 产生 NaN。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`设一个 Q token 被 3 个重叠切片覆盖，各切片 partial 结果为 \((out_i, lse_i)\)。写出合并后的 \(out\)，并证明它等于「把三个切片的 K 集合并起来做一次完整 softmax」的结果（假设 K 集合两两不相交）。`,
          hint: R`用 \(z_{qk}\) 表示 Q 行 \(q\) 与 K 列 \(k\) 的 attention score，把每个 \(out_i\) 写成 \(\frac{\sum_{k\in K_i} e^{z_{qk}} v_k}{e^{lse_i}}\)。`,
          answer: R`\(out = \sum_i w_i\, out_i\)，\(w_i = e^{lse_i-lse}\)，\(lse = \log\sum_i e^{lse_i}\)。代入 \(out_i=e^{-lse_i}\sum_{k\in K_i}e^{z_{qk}}v_k\)，得到 \(out=e^{-lse}\sum_i\sum_{k\in K_i}e^{z_{qk}}v_k\)；同时 \(e^{lse}=\sum_i e^{lse_i}=\sum_{k\in\cup K_i}e^{z_{qk}}\)。这正是对全部 K 做一次 softmax。K 集不相交可保证分子、分母都不重复计数。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`编译缓存键 <code>compile_key</code> 里为什么要放 <code>lse is None</code> 和 <code>sink is not None</code> 这类「张量是否存在」的布尔量，而不是张量本身？`,
          hint: R`想想 CuTe DSL 编译时 Optional 参数会发生什么。`,
          answer: R`它们是编译期分支（<code>const_expr(mLSE is not None)</code>）：LSE/sink 存在与否会改变生成的 kernel 代码（是否写 LSE、是否在尾声并入 sink 项）。张量的值不影响代码结构，但「有没有」影响，所以只有存在性进 key。同理 <code>cu_seqlens_q is None</code> 决定 varlen 分支的取舍。`
        },
        {
          kind: "系统", level: "挑战",
          q: R`functional 层在 Q ranges 重叠时强制 <code>out_type=fp32</code> 并用 <code>atomicAdd</code>（原子加）。分析：若允许 bf16（brain floating point 16）原子加直写，会引入哪两类问题？`,
          hint: R`一类关于硬件,一类关于数值。`,
          answer: R`(1) 数值：bf16 只有 8 位尾数,多个 partial 输出量级相近时逐次舍入误差累积,且加法顺序不定使误差不可复现;fp32 累加把舍入推迟到最后一次 cast。(2) 硬件/语义：bf16 的 atomic add 支持面窄（常需 CAS 模拟或 2 元素打包）,吞吐差且实现复杂。fp32 累加是用 2 倍显存换正确性与可移植性的经典折衷。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`如果要把 INV-CAUSAL（反向因果掩码）原生加进 SM100 kernel，除了给掩码类型映射 <code>MT_MAP</code> 增加一个枚举值，至少还要改哪三处？（提示：分别在 BlockInfo、AttentionMask、softmax_loop）`,
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
      deck: R`本章先建立三个硬件概念。<strong>SMEM</strong> 是线程块共享的片上存储；<strong>TMEM</strong> 是 Blackwell 专门存放矩阵乘累加结果的新空间；<strong>TMA</strong> 是负责在全局内存与 SMEM 之间搬运数据的硬件引擎。第五代 Tensor Core 指令 <strong>tcgen05</strong>（也常称 UMMA）直接读写 TMEM。`,
      takeaway: R`记住一条数据路线即可：TMA 把查询/键/值张量 Q/K/V 搬到 SMEM，tcgen05 完成矩阵乘并把结果累加到 TMEM，softmax 再通过专用指令读写 TMEM。这样，矩阵乘的累加器不再占普通寄存器，softmax 可以获得更多寄存器。`,
      intuitions: [
        { label: "存储", title: "TMEM 专门存累加结果", body: R`分数 \(S=QK^\top\)、概率 \(P=\operatorname{softmax}(S)\) 和输出 \(O=PV\) 中，S 与 O 的 fp32 累加器都放在 TMEM。T2R 表示 TMEM→寄存器，R2T 表示寄存器→TMEM。` },
        { label: "计算", title: "一个 warp 负责发射 MMA", body: "MMA（matrix multiply-accumulate，矩阵乘累加）指令 tcgen05 由一个线程发出后异步执行。负责发射的 warp 不必一直等待，其他 warp 可同时做 softmax、搬运和收尾。" },
        { label: "搬运", title: "TMA 用字节数判断完成", body: R`producer（生产者，即发起搬运的一方）预先登记本次应到达的字节数。TMA 搬完后更新 mbarrier（内存屏障对象），consumer（消费者）才开始读取。` }
      ],
      motivation: [
        R`在 Hopper 上，矩阵乘累加器和 softmax 都占寄存器，两者会争抢容量。Blackwell 把累加器移到 TMEM，缓解了这项冲突。`,
        R`TMEM 不是普通内存：它只有 512 列，必须显式分配和释放，也只能通过 tcgen05、T2R、R2T 等专用指令访问。因此 kernel 需要先画好每一列何时存 S、P 或 O。`,
        R`可选的 2-CTA 模式让两个 CTA（线程块）组成 cluster，共同处理 256 行 Q。两个 CTA 分摊 K/V 数据，并通过分布式 SMEM 互相读取。只有两边工作量接近时，这种协作才划算。`
      ],
      diagram: {
        key: "blackwell",
        caption: "SM100 执行基座：TMA 把 Q/K/V 搬进 SMEM，tcgen05 UMMA 消费 SMEM/TMEM 并把累加器写进 TMEM，T2R/R2T 供 softmax/correction warp 访问，epilogue 经 SMEM 用 TMA 写回。点击节点查看对应源码。"
        },
      explain: [
        {
          title: "TMEM 512 列地图",
          body: [
            R`以 K 分块宽度 <code>tileK=128</code>、value 单头维度 <code>head_dim_v=128</code>、Q 流水级数 <code>q_stage=2</code> 为例：S0、S1、O0、O1 各占 128 列，正好用完 512 列。S0/S1 是两组 Q 的分数，O0/O1 是对应输出。`,
            R`P 不需要额外 128 列。softmax 先把 S 读入寄存器，S 在 TMEM 中随即可以复用；随后 P 以 bf16 写回。bf16 每个元素占 fp32 的一半，因此 P 只占一半物理宽度。row_max 和 row_sum 也在 S 不再使用后复用其空间。`,
            R`后续流水线只需要维护这些槽位的状态：S 写完后通知 softmax，O 校准完后通知下一次矩阵乘。`
          ],
          svg: "tmem-map"
        },
        {
          title: "tcgen05 UMMA：两个 GEMM、一种发射方式",
          body: [
            R`kernel 有两个矩阵乘：QK 计算分数 S，PV 累加输出 O。QK 的输入来自 SMEM；PV 的 P 直接来自 TMEM，因此不需要先复制回 SMEM。`,
            R`tcgen05 指令由 MMA warp 中一个被选中的线程发射，随后异步执行，并在完成时更新 mbarrier。warpgroup 指 4 个 warp、共 128 个线程；与需要整个 warpgroup 参与的 Hopper WGMMA（warpgroup 级矩阵乘累加）相比，Blackwell 的发射端更轻。`,
            R`在 2-CTA 模式下，只有 leader CTA 发射指令，但硬件会读取两个 CTA 的 SMEM。两个 CTA 最终各得到自己的 128 行结果。`
          ],
          formula: R`<p>设 \(i\) 表示 Q stage，\(j\) 表示当前 KV block。一个 Q tile 在 TMEM 中完成：</p>
\[ \underbrace{S_i}_{\mathrm{TMEM}[0,128)} = \underbrace{Q_i}_{\mathrm{SMEM}} \underbrace{K_j^{\mathsf T}}_{\mathrm{SMEM}}, \qquad \underbrace{O_i}_{\mathrm{TMEM}[256,384)} \mathrel{+}= \underbrace{P_{ij}}_{\mathrm{TMEM}[64,192)} \underbrace{V_j}_{\mathrm{SMEM}} . \]
<p>\(P_{ij}\) 是 Q stage \(i\) 对 KV block \(j\) 的概率块，由 softmax warp 以 bf16 写回。\(O_i\) 在整个 KV 循环中驻留 TMEM，持续累加各个 \(j\) 的贡献。</p>`
        },
        {
          title: "TMA：搬运即记账",
          body: [
            R`TMA 的任务是批量搬运 tile（数据分块）。G2S 表示全局内存到 SMEM，S2G 表示反方向。地址计算、边界裁剪和布局转换由硬件描述符完成。`,
            R`发起搬运前，producer 把预期字节数登记到 mbarrier。TMA 完成对应字节后，barrier 才变为“就绪”，MMA warp 因而知道某个 K/V 槽位可以读取。`,
            R`K 和 V 在不同时间复用同一块 SMEM，并使用多个 stage 组成环形缓冲。可用 stage 数由 224KB SMEM 预算除以单个 K/V stage 的大小得到。`
          ]
        },
        {
          title: "TMEM 的生命周期管理",
          body: [
            R`只有 MMA warp 分配 TMEM，并把基地址写到 SMEM；其他 warp 等待 barrier 后再取这个地址。`,
            R`释放前，softmax、correction 和 MMA 三方都必须确认最后一次访问已结束。TMEM 没有自动生命周期，少一次同步就可能在仍有线程读写时提前释放。`
          ]
        }
      ],
      warning: R`TMEM 不能用普通 load/store 访问，也不能当作更大的 SMEM。把它理解成“Tensor Core 专用的累加器仓库”最准确。`,
      exercises: [
        {
          kind: "计算", level: "基础",
          q: R`按本章的规划（tile 为 128×128、value 单头维度 <code>head_dim_v=128</code>、Q 流水级数 <code>q_stage=2</code>），验证 TMEM 恰好用满 512 列；若 <code>head_dim_v=64</code>，还剩多少列？`,
          hint: R`\(\text{total}=2\times(\text{tileK}+\text{head\_dim\_v})\)，其中两项分别是 K 分块宽度与 V 单头维度。`,
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
          q: R`2-CTA 模式下一次 QK UMMA 的有效吞吐是单 CTA 的两倍，但 K/V 的 SMEM 占用为什么反而减半？这如何影响 K/V 流水级数 <code>kv_stage</code>？`,
          hint: R`MMA tiler 的 M（输出行）方向翻倍，但每个 CTA 只存右操作数 B（这里是 K/V）的一半。`,
          answer: R`2-CTA 的 MMA tiler 是 (256,128,·),两个 CTA 各出 128 行 Q;而 B 操作数（K/V tile）在 cluster 内切成两半、每 CTA 存一半（源码 <code>smem_size_kv_per_stage // cta_group_size</code>）。于是每 CTA 的 KV SMEM 减半,同样的 224KB 预算能开更多 kv_stage,流水更深——这是 2-CTA 除了 MMA 效率外的第二重收益。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`host 侧启用 2-CTA 要求「非 causal」。从负载对称性角度解释这条限制。`,
          hint: R`M 表示 Q 的行方向；2-CTA 的两个 CTA 分到同一个 M tile 的上下两半。`,
          answer: R`2-CTA 把 M 方向 256 行绑成一个 MMA：上半 128 行给 CTA0，下半给 CTA1，两者必须迭代<strong>同一组</strong> KV block。causal 下上半行的合法 KV 少于下半行，同组迭代意味着上半 CTA 在大量 block 上空转 mask，负载天然不对称；非 causal 时两半负载完全相同。所以 causal 使用 1-CTA 配合 LPT（Longest Processing Time first，最长处理时间优先）调度处理倾斜更划算。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`TMA 的传输字节计数 <code>tx_count</code> 为什么必须在 host 侧静态算好，而不能在运行时按实际搬运量填写？`,
          hint: R`想想 mbarrier 的 expected-tx 语义与编译期布局。`,
          answer: R`mbarrier 的 expect-tx 在 producer acquire 时一次性设置,硬件按到达字节数递减;若运行时才知道字节数,producer 无法在发起 TMA 前正确设置期望值。而 FFA 的 tile 尺寸、dtype、stage 布局全是编译期常量,每次 TMA 恰好搬一个完整 tile,字节数是纯静态量——变长部分（序列尾部）由 TMA 描述符的边界裁剪处理,搬运字节数不变,只是越界部分填充。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`假设未来出现 value 单头维度 <code>head_dim_v=256</code> 的模型，<code>q_stage=2</code> 时 O 需要 2×256=512 列，S 还要 2×128=256 列，TMEM 装不下。参考本章内容，提出两种可行的重新规划方案并比较代价。`,
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
      deck: R`一个 CTA（cooperative thread array，协作线程块）有 512 个线程，也就是 16 个 warp（每组 32 个线程）。kernel 不让所有 warp 做同一件事，而是把它们分成搬运、矩阵乘、softmax、校准和写回等角色。每个角色反复处理自己的步骤，并通过 barrier（屏障）交接数据槽位。`,
      takeaway: R`Warp specialization（warp 特化）就是“固定分工”。softmax 需要大量寄存器，因此分得最多；load、MMA（矩阵乘累加）和写回 warp 主要负责发起异步指令，只需较少寄存器。各角色不传递复杂对象，只约定某个 SMEM（共享内存）或 TMEM（张量内存）槽位何时“满”或“空”。`,
      intuitions: [
        { label: "分工", title: "流水线工厂", body: "load 进料、MMA 冲压、softmax 精加工、correction（输出校准）调整旧结果、epilogue（尾声）打包出货——每个工位只做一件事，靠传送带 mbarrier（内存屏障对象）衔接。" },
        { label: "资源", title: "按角色分配寄存器", body: R`代码用 512 作为四类角色的配额账本：两组 softmax、correction 和 other。<code>setmaxregister</code> 让轻量角色降低上限，把额度留给 softmax。` },
        { label: "同步", title: "只交接槽位状态", body: "数据始终放在约定的 TMEM/SMEM 地址。warp 只通过 mbarrier 告知对方“可以读”或“可以覆盖”。" }
      ],
      motivation: [
        R`producer 是写入槽位的一方，consumer 是读取并释放槽位的一方。SM100 的 MMA 只需轻量发射，因此可以把更多 warp 留给 softmax 和数据流编排。`,
        R`角色数量会随配置变化。例如只有一个 Q stage（流水线中的一组 Q 分块）时，不需要第二组 softmax；变长序列写回时，correction warp 会兼任 epilogue。由于这些选择在编译期确定，运行时不会为未选路径付分支开销。`,
        R`阅读本章只追踪一个 KV block（键/值分块）：TMA（Tensor Memory Accelerator，张量内存加速器）把它搬进 SMEM，MMA 生成分数块 S，softmax 生成概率块 P，下一次 MMA 累加输出块 O，correction 再校准旧 O。每次交接都对应一条流水线。`
      ],
      diagram: {
        key: "pipeline",
        caption: "前向 kernel 的角色分工与流水线拓扑：左列为 warp 角色（含寄存器配额），右侧为它们之间的六条 mbarrier 流水线。点击节点查看源码。"
      },
      explain: [
        {
          title: "角色表与寄存器账本",
          body: [
            R`默认分工是：warp 0–7 做两组 softmax，8–11 做 correction，12 发射 MMA，13 写回，14 搬运，15 空闲或负责 CLC（Cluster Launch Control，簇启动控制）调度。4 个连续 warp 称为一个 warpgroup。`,
            R`寄存器按 warpgroup 重新分配。load/MMA/epilogue 主动降低上限，softmax 再提高上限，correction 取中间值。softmax 的额度决定一次能在寄存器里保留多少分数，因此是最敏感的调参项。`,
            R`一个容易忽略的细节:降寄存器的角色必须先降、升寄存器的角色后升,否则瞬时总量超限。源码里 empty/load/MMA/epilogue 的 decrease 都写在各自分支的第一行。`
          ]
        },
        {
          title: "六条流水线的全景",
          body: [
            R`<strong>pipeline_q / pipeline_kv</strong> 管 SMEM 输入槽位。load warp 发起 TMA，MMA warp 等字节全部到达后读取。多个 stage 让搬运与计算重叠。`,
            R`<strong>pipeline_s_p_o</strong> 管 TMEM 中的 S、P 和 O。QK 完成表示 S 可读；下一次 PV 开始前，又必须同时满足 P 已写完、旧 O 已校准。一个槽位因此有两项释放条件。`,
            R`其余流水线只处理特殊等待：<code>pipeline_p_lastsplit</code> 通知 P 的最后一段已写完；<code>pipeline_o_acc</code> 等最后一次 O 累加；<code>pipeline_sm_stats</code> 与 named barrier 共同保护 softmax 和 correction 共享的缩放值。`
          ],
          svg: "pipeline-wave"
        },
        {
          title: "一个 KV block 的完整旅程",
          body: [
            R`一个 KV block 依次经历：① load warp 找到空槽并发起 TMA；② K 到齐后，MMA 计算 QK 并写 S；③ softmax 读取 S、应用 mask、计算 P；④ correction 同时校准上一轮 O；⑤ P 和 O 都就绪后，MMA 计算 PV；⑥ V 槽释放，供下一轮搬运复用。`,
            R`不同角色处理的是不同轮次的数据。例如 load 可能在搬第 \(j+2\) 块，softmax 正处理第 \(j\) 块，correction 在修正第 \(j-1\) 块。KV 足够长时，这些工作可以重叠。`
          ]
        },
        {
          title: "split_P_arrive：P 写一半就开跑",
          body: [
            R`写回 128 列 P 需要多条 R2T（寄存器到 TMEM）指令。写完前 96 列时，softmax 先通知 MMA 启动 PV；写完最后 32 列后，再发送第二个信号。MMA 读到尾部前若数据未就绪，会在硬件中等待。`,
            R`这样，P 尾部的写回与 PV 开头的计算重叠，缩短了两步之间的串行时间。`
          ]
        },
        {
          title: "s0_s1_barrier：两组 softmax 的写口错峰",
          body: [
            R`两组 softmax 同时写 P 会争用 TMEM 写入带宽。<code>s0_s1_barrier</code> 只把两组的“exp2 后写 P”阶段错开，其他阶段仍可并行。`,
            R`correction 的 cross-release 进一步让两组 softmax 相差约半个 KV block：一组写 P 时，另一组让出写口，correction 在两者之间工作。`
          ]
        }
      ],
      warning: R`pipeline 负责跨多轮复用的“槽位满/空”；named barrier（具名屏障）只负责“这些线程本轮是否都到齐”。两者用途不同。缩放值槽位 <code>sScale</code> 同时使用两者，是因为既要保证写完才能读（RAW，read after write），又要保证读完才能覆盖（WAR，write after read）。`,
      exercises: [
        {
          kind: "计算", level: "基础",
          q: R`单头维度为 128（hd128）的非 causal 2-CTA 配置下，softmax 与 correction 的每线程寄存器配额分别为 <code>num_regs_softmax=176</code>、<code>num_regs_correction=88</code>。验证账本并算出其他角色的配额。`,
          hint: R`令 \(s\)、\(c\)、\(o\) 分别表示一组 softmax、correction 和其他角色的配额，则 \(512 = 2\times s + c + o\)。`,
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
          q: R`用 \(O(i-1)\) 表示累计到第 \(i-1\) 个 KV block 的输出。主循环里 correction 修正 \(O(i-1)\) 时不等任何 O 流水线（注释：GEMM ordering guarantee）。写出这个保证的完整推理链，并指出它在哪一步断裂、由哪条流水线补救。`,
          hint: R`同一 MMA warp 发射的 UMMA 按序完成。`,
          answer: R`推理链：corr_scale(i) 可读 ⇒ softmax 已读完 S(i) ⇒ S(i) 的 QK GEMM 已完成 ⇒ 同一 MMA warp 上，上一轮 O 的 PV GEMM 发射更早且按序完成 ⇒ 上一轮 O 必已写好。最后一个 KV block 后没有“下一个 S”传递这个完成关系，因此最终 PV GEMM 可能仍在运行；correction 必须显式调用 <code>pipeline_o_acc.consumer_wait</code>。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`估算 split_P_arrive 的收益上界：设写回 128 列 P 耗时 \(t_P\)，PV GEMM 耗时 \(t_G\)，split 点在 3/4 处。理想情况下每个 KV block 能省多少时间？`,
          hint: R`被隐藏的是写 P 的最后 1/4。`,
          answer: R`无 split 时串行段为 \(t_P + t_G\)；split 后 GEMM 在 \(\tfrac34 t_P\) 时刻启动，读完前 96 列（约 \(\tfrac34 t_G\)）后若最后 32 列已就绪则无缝继续。只要 \(\tfrac14 t_P \le \tfrac34 t_G\)（几乎恒成立），总时长为 \(\tfrac34 t_P + t_G\)，节省 \(\tfrac14 t_P\)。即收益上界是「写 P 尾巴」的完全隐藏，约为每 block 时长的几个百分点——在 softmax 是瓶颈的配置下相当可观。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`Q 为变长布局（varlen，源码标志 <code>is_varlen_q</code>）时，为什么让 correction 兼任 epilogue（源码开关 <code>use_correction_warps_for_epi</code>），而不是保留独立 epilogue warp？`,
          hint: R`varlen 的 O 写回还能用整块 TMA 吗？`,
          answer: R`varlen 下每个 tile 的有效行数不定，尾块需要按序列长度（seqlen）谓词逐行写出，TMA 整 tile store 不再适用，写回退化为普通 S2G（SMEM 到全局内存）copy——这需要一个完整 warpgroup 的线程宽度才够带宽。correction 本来就要把 O 从 TMEM 读出来做 rescale，顺手写入 gmem（global memory，全局内存）可省去一次 SMEM 中转和一条写回流水线 <code>pipeline_o_epi</code>；1 个 warp 的独立 epilogue 反而喂不饱。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`如果取消 correction 角色，把 rescale（重新缩放）合并进 softmax warp，即算完缩放系数 <code>corr_scale</code> 后顺手修改输出 O，分析它对寄存器账本和关键路径的双重影响。`,
          hint: R`O tile 是 128×128 fp32;softmax 的寄存器已经是全场最紧的。`,
          answer: R`寄存器方面，rescale 需要分批把 O 读入寄存器；即使每次只处理 16 列，也会挤占 softmax 保存 S 的空间。关键路径方面，rescale 会插在“读 S”和“写 P”之间，使 MMA 更晚拿到 P。独立 correction 则让 8 个 softmax warp 与 4 个 correction warp 并行，其余 4 个 warp 负责 load、MMA、epilogue 和调度。代价是增加一组轻量 warp，收益是缩短 softmax 关键路径。`
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
      zhTitle: "Mask 如何分三层处理",
      tag: "SM100 · mask",
      category: "dense",
      difficulty: "进阶",
      source: "kernel/cutedsl/mask.py + block_info.py",
      deck: R`mask 越早处理，代价越低。最便宜的是整块跳过；其次是确认整块都有效；只有边界块才需要逐行判断。Flex-Flash-Attention（FFA）因此分三层处理：<code>BlockInfo</code>（块范围信息）跳过无效块，主循环区分完整块和边界块，R2P（register-to-predicate，寄存器位到谓词）再处理边界块内的具体元素。`,
      takeaway: R`先定义公式中的符号：\(q\) 是当前切片内的 Q 行号，\(k\) 是 K 列号，二者都从 0 开始；\(s_q\) 和 \(s_k\) 分别是 Q、K 的长度。右下对齐的 causal 条件是 \(k\le q+(s_k-s_q)\)。它的含义很直观：Q 与 K 的末尾对齐，每一行只能看见自己及其左侧位置。`,
      intuitions: [
        { label: "分层", title: "先跳块，再挑块，最后改元素", body: "能不算就不算（跳块），能整块算就别逐元素判（full 段），必须判断时再用位掩码批量处理（R2P）。" },
        { label: "几何", title: "先看一个具体例子", body: R`若 \(s_q=3,s_k=5\)，三行 Q 的最大合法 K 下标是 2、3、4。公式中的偏移 \(s_k-s_q=2\) 正是右下对齐产生的水平位移。` },
        { label: "开销", title: "mask 不该出现在快路径", body: "full 段的 <code>softmax_step</code> 根本不传掩码函数 <code>mask_fn</code>——编译期就没有 mask 代码，这才是块级稀疏的意义。" }
      ],
      motivation: [
        R`最直接的做法是对 128×128 tile 中每个元素检查一次 mask，但多数元素通常位于完全有效或完全无效的块中，这些比较是浪费。`,
        R`causal 和滑动窗口 mask 的边界随行号线性移动。因此只看一个 tile 的角点，就能判断它是全有效、全无效，还是边界穿过的 partial 块。只有第三类需要细查。`,
        R`partial 块使用 <strong>R2P（register-to-predicate，寄存器位到谓词）</strong>。代码先把连续 32 列的保留/丢弃结果编码进一个 uint32（32 位无符号整数），再一次转换成 32 个谓词，最后把无效分数写成 \(-\infty\)。`
      ],
      diagram: {
        key: "mask",
        caption: "三层防线：BlockInfo 决定 n_block 迭代范围（跳块），softmax_loop 三段循环隔离 partial 块，AttentionMask 在 partial 块内用 R2P/谓词写 -inf。点击节点查看源码。"
      },
      explain: [
        {
          title: "第一层：BlockInfo 跳块",
          body: [
            R`先看右边界。Q tile 编号 \(m_{\mathrm{block}}\) 覆盖 \(q\in[m_{\mathrm{block}}\cdot128,(m_{\mathrm{block}}+1)\cdot128)\)。源码使用开区间行上界 \(q_{\mathrm{end}}=(m_{\mathrm{block}}+1)\cdot128\)，由它得到 K 的开区间上界，再向上除以 128 得到 K block 的迭代上界 <code>n_block_max</code>；实际 K block 编号 \(n_{\mathrm{block}}\) 满足 \(n_{\mathrm{block}}<\text{n\_block\_max}\)。更右侧的 K block 一定全无效。`,
            R`这里比较 mask 时使用<strong>切片内相对坐标</strong>，即 Q 和 K 都从各自切片的 0 开始。真正访问全局内存时才加上切片起点 offset。<code>SeqlenInfoQK</code> 同时保存长度和 offset，避免混用两套坐标。`,
            R`反向有对称的 <code>get_m_block_min_max</code>（固定 n_block 反推 Q 范围），第 7 章会用到。`
          ]
        },
        {
          title: "第二层：三段主循环",
          body: [
            R`主循环按 K block 分成三段：右边界附近的 partial 段、完全有效的 full 段，以及滑动窗口才有的左边界 partial 段。partial 段调用 <code>mask_fn</code>，full 段在编译期完全移除 mask 代码。`,
            R`最右侧 block 还可能越过实际序列长度，即 \(k\ge s_k\)。因此 prologue 单独检查这一块；更左的 block 不可能越界，无需重复检查。`,
            R`可以把三类记成：Empty 由 BlockInfo 跳过，Full 直接计算，Partial 才逐元素处理。块稀疏路径不靠几何推断，而是由 CSR（compressed sparse row，压缩稀疏行）表直接列出 full/partial block；这种格式在每行中只记录非空块下标。`
          ],
          svg: "mask-segments"
        },
        {
          title: "第三层：行内列界与 R2P",
          body: [
            R`进入 partial block 后，<code>row_idx</code> 是当前 Q 行在切片内的相对下标，<code>n_block</code> 是当前 K block 编号。把全局条件 \(k\le q+(s_k-s_q)\) 改写成块内列条件，可得 <code>local_col &lt; col_limit_right</code>。因此右界需要加 1：<code>col_limit_right = row_idx + causal_row_offset + 1</code>；<code>causal_row_offset</code> 已吸收 \(s_k-s_q\) 和 K block 起点。`,
            R`<code>r2p_bitmask_below</code> 生成“保留右边界左侧列”的 32 位掩码，<code>r2p_bitmask_above</code> 生成“保留左边界右侧列”的掩码。滑动窗口把两者按位 AND。编译期展开循环后，硬件可用一条 R2P 指令把 32 位映射到 32 个谓词。`
          ],
          formula: R`<p>指令量对比：设一个 128 列 tile 中，每行有 \(n_{\mathrm{func}}\) 个连续合法区间；普通 causal 时 \(n_{\mathrm{func}}=1\)。ISETP 是整数比较并写谓词指令，SEL 是按谓词选择结果的指令。</p>
\[ \text{逐元素：}\; \approx 128\, n_{\mathrm{func}}\ \text{条 ISETP} + 128\,(\tfrac{n_{\mathrm{func}}}{2}{+}1)\ \text{条 SEL}; \]
\[ \text{R2P：}\; \lceil 128/32 \rceil \times O(n_{\mathrm{func}})\ \text{条位运算} + 128\ \text{条 SEL} . \]
<p>比较与坐标加法几乎全部消失，SEL 固定 128 条。这是 FA4 fork 中验证过的优化（当时按 24 位一批），SM100 原生实现按 32 位 chunk 重写。</p>`
        },
        {
          title: "可编程出口：mask_mod 与 block sparse",
          body: [
            R`任意 document mask 或动态稀疏无法只靠左右边界表达，此时使用 <code>mask_mod</code> 对元素求布尔值。为避免每个 block 都执行它，外部 CSR 表先列出 partial block 和 full block；full block 仍走无 mask 快路径。`,
            R`PackGQA（打包式 grouped-query attention，分组查询注意力）会把多个 Q head 折叠到行维。调用 <code>mask_mod</code> 前，代码用 <code>divmod</code> 还原真实行号和 head 编号，因此用户看到的仍是逻辑坐标。`,
            R`当前 SM100 CuTe DSL 路径仍只原生支持 full/causal 和连续不重叠 ranges。INV/BI-CAUSAL 与重叠 ranges 尚未完整接入这三层实现。`
          ]
        }
      ],
      warning: R`本仓库的 causal 是右下对齐。只有 \(s_q=s_k\) 时，它才退化为熟悉的 \(k\le q\)。若 Q、K 长度不同，必须使用偏移 \(s_k-s_q\)。`,
      exercises: [
        {
          kind: "计算", level: "基础",
          q: R`\(s_q=s_k=1024\)、tile 为 128×128、causal。对编号 <code>m_block=3</code> 的 Q tile，写出 K block 编号 <code>n_block</code> 的迭代范围和三段循环各自覆盖的块。`,
          hint: R`源码中的 <code>m_idx_max=4×128=512</code> 是开区间上界；最大合法 Q 行号是 511。`,
          answer: R`n_block_max = ⌈512/128⌉ = 4，即迭代 n_block 0..3。对角带起点：tile 内最小行 384 对应列界 385，所处块 = 3，故 Mainloop-1 覆盖 n_block 3（唯一 partial 块，含对角线），Mainloop-2 覆盖 n_block 0..2（full），Mainloop-3 不存在。方阵 causal 恰好每个 Q tile 只有 1 个 partial 块。`
        },
        {
          kind: "计算", level: "基础",
          q: R`验证 <code>r2p_bitmask_below</code>：32 列一组的 chunk 编号 \(s=1\)（列 32..63），右侧开区间列界 <code>col_limit_right=41</code>，掩码值是多少？哪些列被保留？`,
          hint: R`令 \(m\) 表示要从高位移除的位数，则 \(m=(s+1)\times32-\text{limit}\)。`,
          answer: R`m = 64 − 41 = 23，掩码 = 0xFFFFFFFF >> 23 = 0x1FF，即低 9 位为 1——保留 chunk 内前 9 列（全局列 32..40），恰为「列 < 41」。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`统计方阵 causal（每个维度各有 \(n\) 个 tile）下 partial 块占总迭代块数的比例，并说明它如何随序列变长而变化。`,
          hint: R`partial 块只在对角线上。`,
          answer: R`总迭代块数 \(\sum_{m=1}^{n} m = \frac{n(n+1)}{2}\)，partial 块每行 1 个共 \(n\) 个，占比 \(\frac{2}{n+1}\)。序列越长占比越低：\(n=64\)（8K 序列）时仅 3%。这解释了为什么 mask 优化的重点是「让 full 段零开销」而非「让 partial 段更快」——但 flex/稀疏 mask 会推高 partial 占比，届时 R2P 成为主角。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`R2P 掩码函数 <code>mask_r2p_lambda</code> 里，为什么内层循环必须使用编译期范围 <code>range_constexpr</code>？如果写成运行时循环会怎样？`,
          hint: R`R2P 指令的语义是「寄存器位 → 谓词寄存器」。`,
          answer: R`R2P 要求编译器在编译期知道「哪一位对应哪条 SEL」，才能把 <code>mask &amp; (1&lt;&lt;i)</code> 的序列模式识别成一条 R2P + 谓词化 SEL 序列。运行时循环里 i 是变量，每次迭代都是独立的移位+测试+分支，编译器只能生成逐元素代码——不仅没有 R2P，还多了循环开销。「编译期展开是优化的启用条件」是 CuTe DSL 编程的普遍规律。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`滑动窗口 mask（窗宽 \(w\)，即每行约有 \(w\) 个合法列）下，三段循环的中段（full 区）长度是多少个块？什么条件下 full 区消失、退化为纯 partial 迭代？`,
          hint: R`full 区 = 右界带与左界带之间。`,
          answer: R`每行合法列数约 w，跨 ⌈w/128⌉ 个块；其中右端 1 块（causal 界穿过）与左端 1 块（左窗界穿过）是 partial，full 区 ≈ ⌈w/128⌉ − 2 块。当 w ≤ 2×128 = 256 时 full 区消失，每个 Q tile 全是 partial 块——此时窗口 mask 的 R2P 双界路径（below AND above）成为绝对主路径，这正是源码为 local 单独准备双界位掩码的原因。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`要在这套三层防线上支持每行多个合法区间（\(n_{\mathrm{func}}>1\)，其中 \(n_{\mathrm{func}}\) 是每行的连续合法区间数，即 HSTU（Hierarchical Sequential Transduction Unit）的 Function 语义），三层各需什么改动？R2P 部分如何推广？`,
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
      zhTitle: "逐块完成稳定的 Softmax",
      tag: "SM100 · 数值",
      category: "linear",
      difficulty: "进阶",
      source: "kernel/cutedsl/softmax.py + softmax_step",
      deck: R`Softmax 通常需要先看完整一行才能归一化。Online softmax（在线 softmax）则按 KV block（键/值分块）逐块处理，只保存少量行统计量。本章先解释递推公式，再按执行顺序阅读 <code>softmax_step</code>，最后说明为何 B200 会用 FMA（fused multiply-add，融合乘加）多项式近似一部分以 2 为底的指数指令 <code>exp2</code>。`,
      takeaway: R`每行只需维护三个量：\(m\) 是目前见过的最大分数，\(\ell\) 是以 \(m\) 为基准的指数和，\(O\) 是尚未除以 \(\ell\) 的输出。新 block 若抬高 \(m\)，旧的 \(\ell\) 和 \(O\) 都乘同一个校准系数 <code>corr_scale</code> 后继续累加。最终只归一化一次。`,
      intuitions: [
        { label: "流式", title: "先累加，最后再除", body: R`不必等整行分数到齐。每处理一个 block，就更新 max、指数和与未归一化输出。` },
        { label: "换底", title: "统一使用以 2 为底的指数", body: R`利用 \(e^x=2^{x\log_2 e}\)，提前把 \(\log_2 e\) 合并进缩放系数，内层直接使用硬件 <code>exp2</code>。` },
        { label: "双管线", title: "硬件指数忙时借用 FMA", body: "SFU 是执行特殊函数的单元。它吞吐不足时，部分 exp2 改用 FMA 多项式近似，让两类执行单元并行工作。" }
      ],
      motivation: [
        R`一个 <code>softmax_step</code> 处理 128×128 的分数块 \(S\)。线程把自己负责的分数读进寄存器，依次应用 mask、求行最大值、计算指数、写出概率块 \(P=\operatorname{softmax}(S)\)，并更新行和。`,
        R`减去行最大值可防止指数上溢。全被 mask 的行最大值是 \(-\infty\)，代码临时用安全值替代，避免后续出现 NaN。FP8（8 位浮点）路径还会平移指数范围，后文单独解释。`,
        R`B200 上大量 <code>exp2</code> 可能压满 SFU（special function unit，特殊函数单元）。调参表会周期性地把一部分指数换成 FMA 多项式近似；SM103 的 SFU 更快，因此该优化关闭。`
      ],
      diagram: {
        key: "softmax",
        caption: "softmax_step 的数据流：等 S 满 → T2R → score_mod/mask → row_max/corr_scale 发布 → 减 max 乘 scale → exp2（硬件/仿真混合）→ bf16 P 写回 TMEM → 背压与 row_sum。点击节点查看源码。"
      },
      explain: [
        {
          title: "Online softmax 的递推",
          body: [
            R`处理第 \(i\) 个分数块 \(S_i\) 时，先得到新的最大值 \(m_i\)。若最大值变大，先把旧指数和 \(\ell_{i-1}\) 与旧输出 \(O_{i-1}\) 换算到新基准，再加入当前块。所有 block 结束后执行 \(O_N/\ell_N\)。`,
            R`SM100 让两组 warp 分工：softmax warp 更新 \(m\) 和 \(\ell\)，correction warp 缩放 \(O\)。二者通过同一个 <code>corr_scale</code> 保持相同基准。`,
            R`若最大值变化很小，<code>rescale_threshold</code> 会把 <code>corr_scale</code> 直接设为 1。correction warp 随后可整块跳过 O 的读改写，以少量可控近似换取更少工作。`
          ],
          formula: R`<p>下标 \(i\) 表示当前 KV block，\(c\) 是已换到以 2 为底指数域的 softmax 缩放系数：</p>
\[ m_i = \max(m_{i-1}, \operatorname{rowmax}(S_i)), \qquad \text{corr\_scale} = 2^{(m_{i-1} - m_i)\,c}, \quad c = \text{softmax\_scale}\cdot\log_2 e, \]
\[ \tilde P_i = 2^{\,S_i c \,-\, m_i c}, \qquad \ell_i = \ell_{i-1}\cdot \text{corr\_scale} + \operatorname{rowsum}(\tilde P_i), \qquad O_i = O_{i-1}\cdot \text{corr\_scale} + \tilde P_i V_i . \]
<p>其中 \(\tilde P_i\) 是尚未按行和归一化的当前概率块，\(V_i\) 是与它对应的 value 块。</p>
<p>最终 \(O = O_N / \ell_N\)，\(\mathrm{LSE} = (m_N c + \log_2 \ell_N)\cdot \ln 2\)。所有指数均为以 2 为底——scale 里的 \(\log_2 e\) 完成了换底。</p>`
        },
        {
          title: "softmax_step 逐行读",
          body: [
            R`执行顺序是：① 等 S 写完并读入寄存器；② 应用可选的 <code>score_mod</code> 和 mask；③ 更新行最大值，并尽早把 <code>corr_scale</code> 发给 correction；④ 计算 \(s\cdot c-mc\)；⑤ 执行 <code>exp2</code> 并转换 P 的 dtype；⑥ 把 P 写回 TMEM；⑦ 确认旧 scale 已被读取；⑧ 更新 row_sum。`,
            R`<code>corr_scale</code> 在指数计算前发布，使 correction 能与当前 warp 的 exp2 并行。行和 <code>row_sum</code> 则放在可能等待的 WAR（write after read，读后再写）acquire 之后，用无依赖计算填补等待时间。`,
            R`只要本步包含 mask，就强制使用硬件 <code>exp2</code>。原因是被 mask 的输入为 \(-\infty\)，而多项式近似会先截断输入，不能自然得到精确的 0。`
          ]
        },
        {
          title: "ex2 仿真：把指数算在 FMA 管线上",
          body: [
            R`B200 的 SFU 吞吐有限，而 softmax 每个元素都需要一次指数。这里 <code>ex2</code> 是底数为 2 的指数指令名称，与 <code>exp2</code> 同义。kernel 因此把一部分指数交给空闲的 FMA 管线近似计算，使 SFU 与 FMA 同时工作。`,
            R`近似过程先把 \(x\) 拆成整数部分 \(\lfloor x\rfloor\) 和小数部分 \(f\)。整数部分通过调整 IEEE-754 指数位实现 \(2^{\lfloor x\rfloor}\)，小数部分 \(2^f\) 用三次多项式估计。magic number 与向下舍入用于快速取得 \(\lfloor x\rfloor\)。`,
            R`近似频率参数 <code>ex2_emu_freq</code> 控制多长间隔使用一次近似；另外两个参数控制从哪个 fragment（线程持有的数据片段）开始及一次处理多少元素。目标不是全部替代 SFU，而是让 SFU 与 FMA 的工作量同时接近各自吞吐上限。`
          ],
          formula: R`<p>仿真的数学骨架：设 \(x = \lfloor x \rfloor + f\)，\(f\in[0,1)\)，则</p>
\[ 2^x = 2^{\lfloor x\rfloor}\cdot 2^{f}, \qquad 2^{f} \approx p_3(f) = c_0 + c_1 f + c_2 f^2 + c_3 f^3, \]
<p>其中 \(2^{\lfloor x\rfloor}\) 通过把 \(\lfloor x\rfloor\) 加到 IEEE-754 指数域实现（一条整数加法），\(p_3\) 用 Horner 格式 3 条 FMA 完成。bf16 输出只有 8 位尾数，3 次多项式的相对误差（\(\sim10^{-4}\)）绰绰有余——这是「按输出精度定计算精度」的教科书案例。</p>`
        },
        {
          title: "FP8 与 max_offset",
          body: [
            R`FP8（8 位浮点）的 P 若直接落在 \((0,1]\)，会浪费大部分 e4m3（4 位指数、3 位显式尾数）动态范围。指数平移量 <code>max_offset=8</code> 将指数参数加 8，相当于把 P 放大 \(2^8=256\) 倍；尾声和 LSE（log-sum-exp）再减回这项缩放。`,
            R`由于计算统一使用以 2 为底的指数，这个放大只需改一个加法常数，不增加额外指数指令。`
          ]
        }
      ],
      warning: R`<code>corr_scale</code> 只修正旧状态，不乘当前 block：softmax 用它缩放旧 \(\ell\)，correction 用它缩放旧 \(O\)。第一块没有旧状态，因此不需要发布该值。`,
      exercises: [
        {
          kind: "推导", level: "基础",
          q: R`两块序列 \(S_1, S_2\)，验证 online 递推给出的 \(\ell_2\) 与一次性计算 \(\sum_k e^{(s_k - m_2)c'}\)（\(c' = c\ln 2\) 意义下）一致。`,
          hint: R`把 \(\ell_1 = \sum_{k\in S_1} 2^{(s_k-m_1)c}\) 乘上 corr_scale 展开。`,
          answer: R`\(\ell_1 \cdot 2^{(m_1-m_2)c} = \sum_{k\in S_1} 2^{(s_k-m_1)c + (m_1-m_2)c} = \sum_{k\in S_1} 2^{(s_k-m_2)c}\)，再加上新块的 \(\sum_{k\in S_2} 2^{(s_k-m_2)c}\)，恰为全集在基准 \(m_2\) 下的和。递推只是不断「换基准并补差价」。`
        },
        {
          kind: "计算", level: "基础",
          q: R`用 magic number（借助大浮点常数取整）法计算 \(\lfloor3.7\rfloor\)：\(3.7+(2^{23}+2^{22})\) 在向下舍入模式 round-down 下的结果尾数低位是什么？减回 magic number 后得到多少？`,
          hint: R`\(2^{23}+2^{22}=12582912\)，和落在 \([2^{23},2^{24})\) 区间；此时 ULP（unit in the last place，末位单位）为 1。`,
          answer: R`和为 12582915.7，该量级下 fp32 的 ulp 是 1，round-down 得 12582915，整数部分 3 编码在尾数低位。减回 magic 得 3.0——即 \(\lfloor 3.7\rfloor\)。round 模式若是 nearest 则 3.7 会进位成 4，这就是源码强调 <code>rnd="rm"</code> 的原因。`
        },
        {
          kind: "概念", level: "进阶",
          q: R`为什么校准系数 <code>corr_scale</code> 的发布（<code>sm_stats_barrier.arrive</code>）安排在 <code>exp2</code> 之前，而行和 <code>row_sum</code> 的更新安排在 WAR（write after read，读后再写）acquire 之后？两处安排各隐藏了什么延迟？`,
          hint: R`想想 correction warp 此刻在干什么、sScale 槽位何时才能复用。`,
          answer: R`前者：corr_scale 在 row_max 更新后立即可知,早发布一拍,correction 的 O-rescale 就能与本 warp 的整段 exp2+写 P 并行——隐藏的是 correction 的全部工作时长。后者：WAR acquire 是可能阻塞的等待,把无依赖的 row_sum 更新（纯寄存器 FMA）挪到 acquire 之后执行,等待窗口被计算填满——隐藏的是背压等待。两处都是「把必须等的和不必等的重新排序」。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`设 SFU 的 <code>ex2</code> 吞吐为 FMA 管线的 \(1/4\)（示意值），多项式仿真每个 <code>exp2</code> 耗 4 条 FMA 指令。求最优的仿真比例 \(\alpha\)（仿真元素占比）使总吞吐最大，并解释近似频率 <code>ex2_emu_freq</code> 如何逼近它。`,
          hint: R`让两条管线同时跑满。记 \(T_{\mathrm{sfu}}\)、\(T_{\mathrm{fma}}\) 为两条管线的吞吐率，则硬件路径耗时正比于 \((1-\alpha)/T_{\mathrm{sfu}}\)，仿真路径耗时正比于 \(4\alpha/T_{\mathrm{fma}}\)。`,
          answer: R`平衡条件 \((1-\alpha)/1 = 4\alpha/4\)（以各自吞吐归一）⇒ \(1-\alpha = \alpha\) ⇒ \(\alpha = 1/2\)？代入吞吐比：SFU 速率 1、FMA 速率 4，硬件路径时间 \((1-\alpha)\)、仿真时间 \(4\alpha/4 = \alpha\)，并行取 max，最优在 \(\alpha^* = 1/2\)，加速 2 倍（相对纯 SFU）。实际中 FMA 还要跑 scale/转换等其他活，最优 \(\alpha\) 更小——freq=10~16 意味着每 10–16 对元素仿真其中几对，α≈10–30%，正是把 FMA 的「剩余产能」精确填满而不反噬。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`带 mask 的 <code>softmax_step</code> 强制 <code>ex2_emu_freq=0</code>，即关闭 <code>ex2</code> 多项式仿真。从仿真算法的输入域假设出发解释原因。`,
          hint: R`看 ex2_emulation_2 的第一步 clamp 和被 mask 元素的值。`,
          answer: R`被 mask 的元素是 \(-\infty\)，减 max 乘 scale 后仍是 \(-\infty\)。仿真第一步 clamp 到 −127 后结果是 \(2^{-127}\)（非规格化边缘）而非精确 0；硬件 ex2 对 \(-\infty\) 直接返回 0。\(2^{-127}\) 乘 V 累加后一般无害,但在行全被 mask、row_sum 本应为 0 的边角情形会污染「空行检测」（row_sum==0 的判断）。与其为极端值加分支,不如 partial 块整体走硬件路径——反正 partial 块占比低（第 3 章练习 3）。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`若把概率块 P 的输出精度从 bf16 降到 FP8（e4m3，即 4 位指数、3 位显式尾数），多项式次数能否从 3 降到 2？给出误差预算分析。`,
          hint: R`e4m3 尾数 3 位，相对精度约为 \(2^{-4}\)；令 \(f\in[0,1)\) 为指数参数的小数部分，二次多项式对 \(2^f\) 的最大相对误差约为 \(10^{-3}\)。`,
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
      zhTitle: "校准并写出 O / LSE",
      tag: "SM100 · 归约",
      category: "sparse",
      difficulty: "进阶",
      source: "kernel/cutedsl/ffa_fwd_sm100.py · correction_loop",
      deck: R`Correction warpgroup（输出校准线程组）负责两件事：主循环中用校准系数 <code>corr_scale</code> 把旧输出 O 换到新的最大值基准；所有 KV block（键/值分块）结束后，再除以行和 \(\ell\)，写出最终 O 和 LSE（log-sum-exp，对数指数和）。本章重点是这两步，以及 correction 与 softmax 如何安全共享缩放值。`,
      takeaway: R`每当 online softmax 的最大值 \(m\) 变大，尚未归一化的旧输出 \(O\) 都必须乘一次 <code>corr_scale</code>。correction warp 专门完成这项缩放，并与 softmax 并行。最后它再乘 \(1/\ell\) 完成归一化，其中 \(\ell\) 是以当前最大值为基准的指数和。`,
      intuitions: [
        { label: "换基准", title: "旧输出需要同步缩放", body: R`最大值从 \(m_{\text{old}}\) 变为 \(m_{\text{new}}\) 后，旧 O 乘 \(2^{(m_{\text{old}}-m_{\text{new}})c}\)，才能与新 block 使用同一基准；\(c=\text{softmax\_scale}\cdot\log_2e\) 是换底后的缩放系数。` },
        { label: "并行", title: "缩放与下一步 softmax 重叠", body: "correction 修改旧 O 时，softmax 可继续计算当前 block 的概率块 P；两者处理不同数据。" },
        { label: "收尾", title: "最后统一归一化", body: R`主循环中的 O 仍未除以 \(\ell\)。尾声执行 \(O/\ell\)，处理可选 sink，再写出 O 和 LSE。` }
      ],
      motivation: [
        R`\(m\) 和 \(\ell\) 每行只有一个标量，O 却有 128 列。若 softmax warp 同时修改 O，会增加寄存器压力并延长关键路径。SM100 因此安排独立 correction warpgroup。`,
        R`真正困难的是同步。softmax 通过 SMEM（shared memory，共享内存）缩放值槽位 <code>sScale</code> 发送 <code>corr_scale</code>；correction 读完前，softmax 不能覆盖它。RAW（read after write）表示“写完才能读”，WAR（write after read）表示“读完才能再写”。`,
        R`尾声还要处理空行、可选 attention sink 和 FP8（8 位浮点）缩放。Attention sink 可看成只有分数、没有 value 张量 V 的虚拟 token：它增加 softmax 分母，但不增加输出分子。`
      ],
      diagram: {
        key: "correction",
        caption: "correction 的两个阶段：主循环从 sScale 读 corr_scale 并 rescale TMEM 中的 O（T2R→乘→R2T）；尾声等最终 O 与 row_sum，做 1/ℓ 归一化并写出 O 与 LSE。点击节点查看源码。"
      },
      explain: [
        {
          title: "sScale：一个槽位的完整协议",
          body: [
            R`每个 Q stage（流水线中的一组 Q 分块）的 <code>sScale</code> 有两类位置：主循环存 <code>corr_scale</code>，尾声存行和 <code>row_sum</code> 与行最大值 <code>row_max</code>。二者使用不同偏移，不会互相覆盖。`,
            R`softmax 写完后在 named barrier 上 arrive；correction 等到后再读，这是 RAW 保护。correction 读完后释放 pipeline 槽位；softmax 下次写前先 acquire，这是 WAR 保护。`,
            R`<strong>cross-release</strong> 会释放另一组 softmax 的槽位，而非当前组。结果是 correction 在 softmax0 和 softmax1 之间交替服务，两组不会同时争抢 correction 或 TMEM 写口。尾声不再轮转，因此恢复直接 release。`
          ],
          svg: "correction-handshake"
        },
        {
          title: "主循环：ballot 决定是否缩放 O",
          body: [
            R`读到 <code>corr_scale</code> 后，一个 warp 用 <code>ballot</code>（线程束投票）对自己负责的 32 行汇总条件。只要有一行需要缩放，就处理整组；若全部为 1，则整组跳过。`,
            R`真正缩放时，每次只读 16 列 O 到寄存器，乘 scale 后写回。128 列分 8 批完成，避免一次占用过多寄存器。最后的 fence 确保 MMA warp 能看到更新结果。`,
            R`O 校准完成后，correction 释放 <code>pipeline_s_p_o</code> 的 O 条件。MMA 收到 P、O 两侧信号后，才可向同一 TMEM 区域累加下一次 PV。`
          ]
        },
        {
          title: "尾声：归一化、sink 与空行",
          body: [
            R`尾声先读取最终 row_sum 和 row_max。若启用 sink，就向分母增加一个虚拟 token 的指数权重。若 row_sum 为 0 或 NaN，则用 scale=1 避免除零；由于空行的 O 本身为 0，最终仍写出 0。`,
            R`随后等待最后一次 O 累加完成，计算 \(O/\ell\)，转换为输出 dtype。定长序列经 SMEM 和 TMA 写回；变长序列由 correction warp 按有效行直接写回。`
          ]
        },
        {
          title: "LSE：保留后续合并所需的归一化信息",
          body: [
            R`用 \(z_k\) 表示当前 Q 行对第 \(k\) 个 key 的 attention score。LSE 公式为 \(\mathrm{LSE}=(mc+\log_2\ell-\text{max\_offset})\ln2\)，其中 <code>max_offset</code> 是 FP8 路径使用的指数平移量；该式还原为自然对数下的 \(\log\sum_k e^{z_k\cdot\text{scale}}\)，空行写 \(-\infty\)。`,
            R`每个 rank（分布式进程编号）都输出一对 \((O,\mathrm{LSE})\)。分布式 GroupReduce（分组归约）再用第 0 章的公式合并这些局部结果。kernel 内按 KV block 合并、kernel 外按 rank 合并，使用的是同一套数学结构。`
          ],
          formula: R`<p>验证 kernel 内与 kernel 间归约的一致性：\(\tilde p_k\) 表示未按行和归一化的权重，kernel 尾声输出 \(O=\frac{\sum_k\tilde p_kv_k}{\ell}\)、\(\mathrm{LSE}=mc'+\ln\ell\)（\(c'=c\ln2\)）。两个 rank 合并时</p>
\[ w_r = e^{\mathrm{LSE}_r - \mathrm{LSE}},\qquad O = \sum_r w_r O_r = \frac{\sum_r e^{\mathrm{LSE}_r} O_r}{e^{\mathrm{LSE}}} = \frac{\sum_r \sum_{k\in K_r} e^{z_k c'} v_k}{\sum_r \sum_{k \in K_r} e^{z_k c'}}, \]
<p>与把全部 K 交给单 kernel 的结果逐项相等。fp32 的 out 累加路径（functional 层 atomic 模式）走的正是分子分母同时累加的等价形式。</p>`
        }
      ],
      warning: R`softmax 负责把“足够接近 1”的 <code>corr_scale</code> 规范为精确 1，correction 再据此跳过缩放。修改阈值时必须同时检查两侧，否则可能多做工作或漏做缩放。`,
      exercises: [
        {
          kind: "概念", level: "基础",
          q: R`用 \(O(i)\) 表示累计到第 \(i\) 个 KV block 后的输出。为什么主循环中 correction 修正的是 \(O(i-1)\) 而不是 \(O(i)\)？\(O(i)\) 此刻在哪里？`,
          hint: R`corr_scale(i) 的语义是「块 i 使基准变化后旧账的贬值率」。`,
          answer: R`corr_scale(i) 由块 i 的 row_max 更新产生,描述的是「块 0..i-1 累积的 O 相对新基准的高估」——所以乘在 O(i-1)（即累加到 i-1 为止的 O）上。而块 i 自己的贡献 \(\tilde P_i V_i\) 此刻还没算:P_i 刚被 softmax 写回 TMEM,PV GEMM 要等 correction 放行（O 侧 release）后才发射,其结果直接以新基准累加,无需修正。`
        },
        {
          kind: "计算", level: "基础",
          q: R`一行的处理经过 3 个 KV 块，行最大值 <code>row_max</code> 依次为 2.0 → 5.0 → 5.0，换底后的缩放系数 <code>scale_log2</code>（即 \(c\)）为 1。写出两次 <code>corr_scale</code> 以及尾声前 O 携带的总校准系数。`,
          hint: R`corr_scale = 2^{(m_old − m_new)·c}。`,
          answer: R`第二块：\(2^{(2-5)\times1}=2^{-3}=0.125\)；第三块：\(2^{(5-5)}=1\)（触发重缩放阈值 <code>rescale_threshold</code> 的短路，ballot 跳过）。块 1 的贡献总共被乘过 \(0.125\times1=0.125\)，恰为 \(2^{(m_1-m_3)c}\)——校准系数可迟到但不会算错，这是递推的伸缩性。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`若把 cross-release（释放另一组 softmax 槽位）改成释放自己的 stage（直觉写法），两组 softmax 的执行会变成什么节奏？为什么反而更慢？`,
          hint: R`追踪 softmax0 连发两块的时序。`,
          answer: R`直接 release 下，softmax0 发布 scale 后立刻拿回槽位写权，可以连续冲刺多个 KV 块；softmax1 同样。两组会「同相位」推进——同时到达 exp2+写 P 段（TMEM 写口打架，s0_s1_barrier 强制其中一组干等）、同时向 correction 要服务（correction 串行处理,另一组排队）。cross-release 强制两组相位错开半拍：correction 服务 A 时 B 在算,服务 B 时 A 在算——资源冲突消失。慢的原因不是吞吐而是<strong>相位共振</strong>,cross-release 是一个用同步原语实现的「相位分离器」。`
        },
        {
          kind: "推导", level: "进阶",
          q: R`推导 learnable sink（可学习注意力汇点）修正行和 <code>row_sum</code> 的公式：为何加的是 \(2^{s\cdot\log_2 e-mc+\text{offset}}\)？其中 \(s\) 是 sink logit，\(m\) 是行最大值，\(c\) 是换底后的缩放系数，<code>offset</code> 是 FP8 指数平移量。sink 为什么不需要对应的 V 贡献？`,
          hint: R`把 sink 视为一个 logit 恒为 \(s\)（自然对数域）的虚拟 token。`,
          answer: R`虚拟 token 的未归一化权重是 \(e^{s-mc'}\)（与真实 token 同基准），其中 \(c'=c\ln2\)。换成以 2 为底即为 \(2^{s\log_2e-mc}\)，再补指数平移量 <code>max_offset</code> 与其他项对齐——正是源码的表达式。sink 的语义是“允许注意力弃权”：它进入分母以稀释所有真实权重，但没有 value 向量，不进入分子。因此只修改 <code>row_sum</code>，O 不变；LSE 因分母变化自动增大，合并语义仍然自洽。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`空行（row_sum=0）时源码把归一化 scale 兜底为 1 而不是跳过写出。从「分布式消费者」的角度解释为什么必须写出 0 而不是留下垃圾值。`,
          hint: R`这个 out 可能马上被 GroupReduce 当作 partial 累加。`,
          answer: R`空行的 O 累加器本身为 0（没有任何 PV 贡献），乘 1 后写出的是干净的 0。配合 \(\mathrm{LSE}=-\infty\)，合并公式中 \(w=e^{-\infty-\mathrm{lse}}=0\)，该 partial 对全局结果零贡献。若跳过写出，gmem（global memory，全局内存）中会留下未初始化值或上一轮数据，而合并端并不知道哪些行“该被跳过”——除非引入额外的 valid 位图。写出 \(0\) 与 \(-\infty\) 让“空”成为合并代数中的合法元素，免去所有特判。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`functional 层的 atomic（原子累加）模式下，多个 kernel 实例向同一 <code>out</code> 累加的是“已按各自 LSE 归一化的输出”吗？结合本章公式说明原子直加成立的前提，以及为什么该路径要搭配后续的 LSE 全局校正。`,
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
      deck: R`一个 Q tile（查询分块）是一次调度任务。causal mask 下，靠后的 Q tile 能看到更多 K，因此任务大小不同。调度器要把这些轻重不一的任务分给有限数量的 SM（streaming multiprocessor，流式多处理器），尽量避免某些 SM 已空闲而另一些仍在处理大任务。`,
      takeaway: R`调度器做三件互相独立的事：LPT（Longest Processing Time first，最长处理时间优先）决定先做重任务还是轻任务；L2 swizzle（L2 缓存亲和的坐标重排）让同时运行的 CTA（线程块）尽量复用相同 K/V；静态分配或 CLC（Cluster Launch Control，簇启动控制）决定下一项工作由软件预先指定，还是由硬件动态派发。`,
      intuitions: [
        { label: "顺序", title: "先做最重的任务", body: "LPT 是 Longest Processing Time first。先启动重 tile，收尾时只剩轻 tile，更容易让所有 SM 同时结束。" },
        { label: "亲和", title: "让相邻任务复用缓存", body: "处理同一 head 的 CTA 会读取相同 K/V。把它们安排得更接近，可提高 L2 命中率。" },
        { label: "分配", title: "静态分配与动态领取", body: "静态模式预先确定每个 CTA 的任务序列；CLC 让 CTA 完成一项后再向硬件领取下一项。" }
      ],
      motivation: [
        R`方阵 causal 中，从 0 编号的第 \(m\) 个 Q tile 需要处理 \(m+1\) 个 KV block。若按从轻到重的自然顺序启动，最后才开始的重任务会形成长尾。LPT 通过反转顺序解决这个问题。`,
        R`调度还影响缓存。多个 CTA 若处理同一 head，会读取同一份 K/V；若把这些 CTA 安排在相近时间运行，L2 更可能保留所需数据。`,
        R`变长序列和块稀疏任务的大小更难提前预测。CLC（Cluster Launch Control）允许 persistent CTA 每完成一个 tile，就向硬件申请下一个未分配坐标，并通过 SMEM（共享内存）与 mbarrier（内存屏障对象）接收结果。`
      ],
      diagram: {
        key: "scheduler",
        caption: "调度层全景：host 启发式选定调度器类型；静态路径用 L2-swizzle + LPT 坐标映射；CLC 路径由调度 warp 预取、全体 warp 消费。点击节点查看源码。"
      },
      explain: [
        {
          title: "四种调度器与选型决策树",
          body: [
            R`选型顺序如下：变长 Q 使用 varlen（variable-length）调度器；causal/local（因果/局部窗口 mask）或 CLC 使用 LPT 调度器；稠密非 causal 使用静态 persistent 调度器；最简单的兜底方案是一 CTA 处理一个 tile。`,
            R`persistent 表示启动的 CTA 数量接近 SM 数，而不是 tile 总数；每个 CTA 在循环中连续处理多个 tile。这样可减少 CTA 启动开销，并复用部分初始化状态。各 warp 都通过同一组 <code>initial_work_tile_info</code> / <code>advance_to_next_work</code> 接口取得任务。`
          ]
        },
        {
          title: "LPT 与 L2 swizzle 的坐标算术",
          body: [
            R`先估算一个 head 的 K/V 占用：\(s_k(d+d_v)\times\)元素字节数，其中 \(s_k\) 是 K/V 序列长度，\(d\) 和 \(d_v\) 分别是 K 与 V 的单头维度。源码用固定 50MB 作为 L2 可用预算；这是调度启发式常量，不是硬件精确容量。用它除以单 head 占用，再向下取 2 的幂作为重排因子 <code>swizzle</code>。`,
            R`LPT 的核心实现只是反转 block 编号：<code>block = num_block - 1 - block</code>。causal 中编号越大工作越重，所以重块最先派发。`,
            R`makespan 指所有任务完成所需的总时间。一般任务上，LPT 的最坏情况有经典近似界；对工作量近似线性增长的 causal tile，它通常更接近理想均衡。`
          ],
          svg: "lpt-swizzle"
        },
        {
          title: "CLC：硬件动态派工",
          body: [
            R`CLC 模式下，kernel 先声明逻辑 tile 网格。调度 warp 向硬件请求一个尚未派发的坐标；硬件异步把响应写入 SMEM，并通过 mbarrier 通知。`,
            R`调度 warp 是 producer，其余 warp 是 consumer。预取深度为 1 时，当前 tile 正在计算，下一个坐标已在请求途中。`,
            R`CLC 返回原始 grid 坐标后，软件仍可做 LPT 反转和 split-KV（把一个 KV 任务拆成多个子任务）解包，但不再做 L2 swizzle。因为实际派发先后由硬件决定，软件无法再可靠控制同时运行的是哪些坐标。`
          ]
        },
        {
          title: "何时动态调度反而亏",
          body: [
            R`两个场景会回退。稠密非 causal 的 tile 工作量相同，静态分配已经均衡，CLC 只增加请求开销。varlen MHA（variable-length multi-head attention，变长多头注意力）中，CLC 还可能打乱 K/V 访问顺序并降低 L2 命中率。`,
            R`因此动态调度主要解决<strong>难以预先预测</strong>的负载不均。若不均衡可由 causal 几何准确估计，LPT 往往已经足够。最终仍应通过环境变量开关做 benchmark。`
          ]
        }
      ],
      warning: R`Persistent + CLC 模式下，grid 大小接近 SM 数，不等于 tile 数。不能再从 <code>blockIdx</code> 直接推断任务坐标，必须通过 <code>tile_scheduler</code> 获取。`,
      exercises: [
        {
          kind: "计算", level: "基础",
          q: R`K/V 序列长度 <code>seqlen_k=8192</code>、K 与 V 的单头维度 <code>hd=hd_v=128</code>、数据类型为 bf16（brain floating point 16）。一个 KV head 的体积是多少？L2 重排因子 <code>swizzle</code> 取几？`,
          hint: R`\(8192\times256\times2\) 字节。`,
          answer: R`\(8192\times(128+128)\times2 = 4\,\mathrm{MB}\)。\(50\mathrm{MB}/4\mathrm{MB} = 12.5\)，向下取 2 的幂得 swizzle=8：同一节拍的 CTA 共享 8 个 head 的 K/V（32MB），留余量给 Q/O 流量。`
        },
        {
          kind: "推导", level: "基础",
          q: R`方阵 causal、\(n\) 个 Q tile、\(m\) 个 SM（\(n\gg m\)）。自然顺序静态分配与 LPT 动态领取的 makespan（全部任务完成时间）各约多少？（以单个 KV block 的处理时间为单位）`,
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
          q: R`varlen 调度器用“warp 内前缀和 + ballot（线程束投票）”把线性任务编号 <code>tile_idx</code> 定位到 <code>(batch, block)</code>。为什么限制“最多 31 个 batch 的前缀和”？超过怎么办？`,
          hint: R`一个 warp 有几条 lane？ballot 返回什么？`,
          answer: R`前缀和由 warp 的 32 条 lane 并行持有——lane \(i\) 存放前 \(i\) 个 batch 的累计块数，<code>ballot(tile_idx &gt;= prefix)</code> 的置位数直接给出所属 batch，一条指令完成定位。32 条 lane 减去边界哨位可覆盖 31 个 batch；更多 batch 时分多轮（每轮 31 个）迭代，或退化为循环查找。这是“用 warp 当 SIMD（single instruction, multiple data）查找表”的经典技巧。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`调度流水级数 <code>sched_stages=1</code> 意味着 CLC 预取深度为 1。什么情况下值得把它提高到 2？代价是什么？`,
          hint: R`比较 tile 计算时长与 CLC 往返延迟。`,
          answer: R`当单 tile 计算时间短于 CLC 请求往返延迟时（极小 seqlen、高稀疏、tile 大量为空），深度 1 会让 CTA 在 tile 之间露出等待气泡，深度 2 可再藏一层延迟。代价：response buffer 与 mbarrier 各多一份（SMEM 微增），以及「已预取但未消费」的 tile 在 CTA 退出时需要 producer_tail 妥善排空——排空逻辑复杂度随深度上升。FFA 的 tile 普遍够大（128×128×若干 KV block），深度 1 足矣。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`为 block-sparse 前向设计调度顺序：每个 Q tile \(m\) 的有效 KV 块数记为 \(w_m\)，由 CSR（压缩稀疏行）表给出且 host 可见。结合 LPT 与 L2 swizzle 的思想给出一个方案，并说明它比“纯 CLC 不排序”好在哪里。`,
          hint: R`LPT 需要的只是每个任务的重量估计,CSR 表恰好给了。`,
          answer: R`方案：host 侧按 \(w_m\)（CSR 的计数数组 <code>cnt</code> 即任务重量）对 tile 做桶排序，生成派发顺序表；坐标映射先按 swizzle 分节拍以保持 L2 亲和，节拍内再按 \(w_m\) 降序（LPT）。分配方式仍可使用 CLC——硬件派发的是“顺序表的下标”而非原始坐标，从而兼得动态均衡与软件排序。相比纯 CLC，CLC 只解决“谁来做下一个”，不解决“下一个应该是谁”；重 tile 若排在表尾，动态分配也救不了长尾。排序表把两个自由度解耦，各自交给最擅长的一方。反向 kernel 的 dQ 写入顺序表 <code>dq_write_order</code> 已在用类似思路保证 deterministic（确定性）模式的顺序。`
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
      deck: R`反向需要五次矩阵乘：重算分数 \(S=QK^\mathsf{T}\)，计算概率梯度 \(dP\)、value 梯度 \(dV\)、分数梯度 \(dS\)，之后再得到 query 梯度 \(dQ\) 和 key 梯度 \(dK\)；这里 \(dX\) 表示损失对张量 \(X\) 的梯度。前向没有保存概率矩阵 P，因此反向用 S 和 LSE（log-sum-exp）重算它。另一个难点是：同一 dQ 会收到多个 K tile 的贡献，需要跨 CTA（线程块）求和。`,
      takeaway: R`kernel 固定一个 K/V tile，再遍历所有相关 Q tile。这样 dK、dV 可在当前 CTA 的 TMEM（Tensor Memory，张量内存）中累加并一次写出；dQ 则由多个 CTA 分别贡献，统一原子加到 fp32（32 位浮点）梯度累加缓冲 <code>dQaccum</code>。确定性模式仍使用原子加，只是用信号量固定加法顺序。`,
      intuitions: [
        { label: "局部累加", title: "dK/dV 留在当前 CTA", body: R`固定 K/V 后，所有依次流过的 Q 与输出梯度 dO 都为同一块 dK、dV 提供贡献，因此可在 TMEM 中连续累加。` },
        { label: "全局累加", title: "dQ 需要跨 CTA 求和", body: R`每个 K tile 都会贡献一部分 \(dQ=dS\,K\)。TMA（Tensor Memory Accelerator）原子加把这些部分汇总到 fp32 缓冲。` },
        { label: "重算", title: "用计算换显存", body: R`序列长度为 \(n\) 时，P 有 \(O(n^2)\) 个元素，前向不保存它。反向从分数 S 和每行 LSE 恢复 P，减少显存读写。` }
      ],
      motivation: [
        R`依赖顺序是：重算 \(S\) 和 \(P\)；由 P、输出梯度 dO 得到 dV；由 dO、V 得到 dP；再由 P、dP 和 softmax 反向所需的行标量 D 得到 dS；最后由 dS 得到 dQ、dK。D 在 preprocess（预处理）kernel 中提前计算。`,
        R`若固定 Q，dQ 可本地累加，但 dK、dV 两个张量都要跨 CTA 归约。固定 K 后只剩 dQ 需要全局归约，因此 Flex-Flash-Attention（FFA）选择后者。2-CTA 模式还需要 relay warp（中继线程束），把另一个 CTA 的 dS 就绪信号转发给发射 MMA（矩阵乘累加）的 leader。`,
        R`16 个 warp 分为 reduce、compute、MMA、load、relay 和空闲角色。TMEM 中 S/P、dP/dS、dQ/dP 会在不同时间复用同一位置，因此每次覆盖前都必须确认旧数据已被消费。`
      ],
      diagram: {
        key: "backward",
        caption: "反向三段式：preprocess 产出 D 与 LSE·log2e 并清零 dQaccum；主 kernel 以 K-tile 为家做 5-GEMM 流水与 dQ 原子归约；postprocess 把 fp32 dQaccum 乘 scale 转 dtype。点击节点查看源码。"
      },
      explain: [
        {
          title: "反向的数学链",
          body: [
            R`记 \(dX\) 为损失对 X 的梯度，\(\odot\) 为逐元素乘法。Softmax 反向可写成 \(dS=P\odot(dP-D)\)，其中每行标量 \(D_i=\sum_jP_{ij}dP_{ij}\)。利用 \(dP=dO\,V^\mathsf{T}\) 和 \(O=PV\)，它又等于 \(\operatorname{rowsum}(dO_i\odot O_i)\)，因此不需要 P 就能在预处理阶段算出。`,
            R`softmax 缩放系数 \(c\) 最终要乘到 dQ 和 dK。设序列长度为 \(n\)、单头维度为 \(d\)，实现不在 \(O(n^2)\) 大小的 dS 上逐元素乘，而是在写出 \(O(nd)\) 大小的 dQ/dK 时再乘，结果相同但工作更少。`
          ],
          formula: R`\[ D = \operatorname{rowsum}(dO \odot O), \qquad P = 2^{\,S c\,-\,\mathrm{LSE}\cdot\log_2 e}, \]
\[ dV = P^{\mathsf T}\, dO, \qquad dP = dO\, V^{\mathsf T}, \qquad dS = P \odot (dP - D), \]
\[ dQ = c\; dS\, K, \qquad dK = c\; dS^{\mathsf T} Q . \]
<p>验证 \(D\) 的化简：\(D_i = \sum_j P_{ij} dP_{ij} = \sum_j P_{ij} (dO_i \cdot V_j) = dO_i \cdot \sum_j P_{ij} V_j = dO_i \cdot O_i\)。一行点积换掉一次 \(O(n)\) 的归约，这正是 preprocess kernel 存在的理由。</p>`
        },
        {
          title: "以 K 为家的调度与 warp 分工",
          body: [
            R`一个工作 tile 对应一个 K block（源码编号为 <code>n_block</code>）。K/V 只加载一次，Q、dO、LSE 和 D 按 Q block（源码编号为 <code>m_block</code>）依次流过。<code>BlockInfo</code> 会跳过当前 K block 无法看到的 Q 范围。`,
            R`矩阵乘按 \(S^\mathsf{T}=KQ^\mathsf{T}\) 的转置形式组织，使固定的 K 位于主操作数方向。P 和 dS 的部分消费者从 TMEM 读取；dQ 路径需要从 SMEM 读取 dS，以便 2-CTA 时交换数据。`,
            R`五次 GEMM 交错为 <code>S(i) → dK(i-1) → dQ(i-1) → dP(i) → dV(i)</code>。多数依赖可由同一 MMA warp 的发射顺序保证；显式等待主要用于确认 dS 已生成，以及复用 TMEM 前确认 dQ 已被 reduce warp 读走。`
          ],
          svg: "bwd-tmem"
        },
        {
          title: "softmax 重算与 dS",
          body: [
            R`compute warp 从 TMEM 读取 S，计算 \(P=2^{Sc-\mathrm{LSE}\log_2e}\)，再把 bf16 P 写回已经读空的 S 区域。`,
            R`dP 完成后，代码读取 dP，减去每行的 D，再乘 P 得到 dS。dS 一份写回 TMEM 供 dK 使用，另一份写到 SMEM 供 dQ 使用。被 mask 的位置有 \(P=0\)，因此 dS 也为 0。`,
            R`反向已知完整行的 LSE，不需要 online 递推，也没有 <code>corr_scale</code> 或 correction warp。主要复杂度转为五次 GEMM 的排程和 dQ 归约。`
          ]
        },
        {
          title: "dQ 归约：atomic 与确定性",
          body: [
            R`dQ GEMM 完成后，reduce warp 把 tile 从 TMEM 搬到 SMEM，再由 TMA 对全局内存中的 fp32 <code>dQaccum</code> 做批量原子加。`,
            R`确定性模式为每个 Q block 配置信号量。写者按预定顺序执行“等待→原子加→放行下一位”，使 fp32 加法顺序固定，从而获得 bit 级可复现结果。代价是同一 Q block 的写者必须串行。`,
            R`postprocess（后处理）最后把 <code>dQaccum</code> 乘 softmax scale，再转换为输出数据类型 <code>dtype</code>。dK/dV 已在当前 CTA 内累加完成，通常可直接写出；GQA（grouped-query attention，分组查询注意力）的多 Q-head 归约是例外。`
          ]
        }
      ],
      warning: R`<code>deterministic=True</code> 不会取消原子加，只会固定原子加顺序。反向按 \(S^\mathsf{T}\) 布局工作，mask 的行列已交换，不能直接复制前向坐标公式。`,
      exercises: [
        {
          kind: "推导", level: "基础",
          q: R`完成 \(dS=P\odot(dP-D)\) 的推导：从第 \(i\) 行概率 \(P_i=\operatorname{softmax}(z_i)\) 的雅可比 \(\frac{\partial P_{ij}}{\partial z_{ik}}=P_{ij}(\delta_{jk}-P_{ik})\) 出发；\(z_i\) 是该行 logits，\(\delta_{jk}\) 是 Kronecker delta。`,
          hint: R`\(dz_{ik} = \sum_j dP_{ij}\,\partial P_{ij}/\partial z_{ik}\)。`,
          answer: R`\(dz_{ik} = \sum_j dP_{ij} P_{ij}(\delta_{jk} - P_{ik}) = P_{ik} dP_{ik} - P_{ik}\sum_j P_{ij} dP_{ij} = P_{ik}(dP_{ik} - D_i)\)，其中 \(D_i = \sum_j P_{ij}dP_{ij}\)。矩阵形式即 \(dS = P\odot(dP - D\mathbf 1^{\mathsf T})\)。`
        },
        {
          kind: "计算", level: "基础",
          q: R`统计反向的 FLOPs（浮点运算次数）：与前向的 2 个 GEMM（通用矩阵乘）相比，5 个 GEMM 的总计算量是前向的几倍？（设 Q/K 单头维度 \(d\) 等于 V 单头维度 \(d_v\)，忽略逐元素项）`,
          hint: R`记 Q、K 的序列长度分别为 \(n_q\)、\(n_k\)，每个 GEMM 都是 \(2n_qn_kd\) 量级。`,
          answer: R`前向 \(2\times 2 n_q n_k d\)（QK、PV）。反向 5 个 GEMM 中 S 重算、dP、dV、dK、dQ 各 \(2 n_q n_k d\)，共 5 份——反向 ≈ 前向的 2.5 倍。这与「训练一步 ≈ 3× 前向」的经验法则一致（前向 1 + 反向 2.5，加上重算摊销）。`
        },
        {
          kind: "概念", level: "进阶",
          q: R`如果反向也“以 Q 为家”（固定 Q 扫 K），需要对 dK/dV 做全局归约。记 Q、K 的序列长度为 \(n_q\)、\(n_k\)，对比两种取向的归约流量，说明何时“以 Q 为家”反而更优。`,
          hint: R`归约流量 ∝ 客场张量的大小 × 扫过它的 tile 数。`,
          answer: R`以 K 为家：dQ 归约,流量 ∝ \(n_q d \times n_k/\text{tile}\)。以 Q 为家：dK+dV 归约,流量 ∝ \(2 n_k d \times n_q/\text{tile}\)。比值 = \(n_q : 2n_k\)——当 \(n_q \gg 2 n_k\)(极不对称的 cross-attention、或 GQA 把 Q 折叠后)以 Q 为家更省。对自注意力 \(n_q = n_k\),以 K 为家流量减半,加上 dK/dV 是两个张量(两套 semaphore/accum),工程上也更繁,故 FFA 选 K。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`1-CTA 模式下，TMEM 中的 dQ tile（源码名 <code>tdQ</code>）与 dP tile（<code>tdP</code>）完全共用空间。根据流水线注释，写出保证不冲突的事件顺序链。`,
          hint: R`dP(i) 发射前等的是什么？`,
          answer: R`顺序链：dQ(i-1) GEMM 完成（写 tdQ）→ reduce warpgroup T2R 读走 tdQ → 发 <code>pipeline_dQ.sync_object_empty</code> → MMA 的 dP(i) 才发射（向同一块 TMEM 写 dP）→ dP 被 compute 消费转成 dS → 下一轮 dQ 又写回来。即「dQ 写→dQ 读空→dP 写→dP 读空」的四拍循环，<code>pipeline_dQ</code> 的 full/empty 两个方向各守一拍。这就是注释 "tdQ is overlapped with tdP" 背后的完整契约。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`deterministic（确定性）模式的信号量把同一 Q block（<code>m_block</code>）的写者串行化。估算最坏情形的串行化代价，并解释“让 lock 顺序与 LPT（最长处理时间优先）派发顺序同向”为何能缓解它。`,
          hint: R`设 K 方向共有 \(n\) 个 block，其编号为 <code>n_block</code>。causal 下，编号较大的 Q tile 会被多少个 <code>n_block</code> 写入？调度器按什么顺序派发这些 block？`,
          answer: R`causal 下最后一个 m_block 会被全部 \(n\) 个 n_block 写；若写者同时到达，最坏需要串行执行 \(n\) 次 TMA reduce。把 lock 顺序与 LPT 派发顺序设为同向后，先派发、通常也先到达的 n_block 持有较小 lock，可立即写入；后到达者再依次接续。若两种顺序相反，最早到达的写者反而要等待所有后来者，容易形成长队。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`GQA（分组查询注意力）的每个 KV head 对应 8 个 Q head（<code>qhead_per_kvhead=8</code>）时，8 份 dK/dV 贡献落到同一个 KV head 上。源码走“fp32 累加缓冲 + 可选信号量”的 <code>dKV_postprocess</code> 路径。设计一个免全局归约的替代方案并分析代价。`,
          hint: R`归约的另一个维度是「调度」:能否让同一 KV head 的 8 个 Q head 由同一 CTA 处理？`,
          answer: R`可把 (n_block, kv_head) 作为 work tile，并在一个 CTA 内依次处理 8 个 Q head。这样 dK/dV 可跨 head 留在 TMEM 中累加，最后一次写出，无需全局原子归约。代价是：单 tile 工作量增至 8 倍，任务粒度更粗；跨 head 访问可能降低 L2 局部性；LSE 和 D 需要按 head 轮换，可能减少流水线 stage。这个方案用较少归约换取较少调度自由度。`
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
      deck: R`CP（context parallelism）把长序列分到多个进程，进程在代码中称为 rank。每个 rank 持有部分 Q/K/V，但计算本地 Q 时可能需要远端 K/V。本章说明数据如何按需发送、局部结果如何合并，以及通信如何与计算重叠。`,
      takeaway: R`通算融合先把 mask 依赖变成精确的收发清单，再用全局一致的 <code>overlap_degree</code> 将各 rank 的远端接收列切成多拍；前向隐藏预取，反向同时编排 cast 与 reduce，OverlapSolver 负责在通信、计算和发射开销之间选取分段。只有异步队列与 SM 资源同时就绪，时间线才会真正重叠。`,
      intuitions: [
        { label: "通信", title: "按依赖发送和归约", body: R`GroupCast（一对多分组发送）把一段数据发给多个需要它的 rank；GroupReduce（分组归约）把多个 rank 的局部结果送回属主并合并。` },
        { label: "流水", title: "三拍并行", body: R`理想状态下同时执行预取 <code>prefetch(i+1)</code>、计算 <code>compute(i)</code> 与归约 <code>reduce(i-1)</code>，每拍耗时接近通信与计算中的较大者。` },
        { label: "资源", title: "异步不等于能同时运行", body: R`通信 kernel 也需要 SM。计算 kernel 若占满 GPU，通信即使已异步入队，也可能只能排队等待。` }
      ],
      motivation: [
        R`环形方案通常让 KV 按固定拓扑依次经过各 rank。遇到不规则 mask 时，许多根本不会参与计算的 KV 也会被传输。MagiAttention 先由 AttnSlice（注意力切片）得出精确依赖清单，只把某段 KV 发给真正需要它的 rank。`,
        R`若先收齐全部远端 KV 再计算，通信时间会完整暴露。分布式注意力运行时 <code>DistAttnRuntime</code> 将远端 KV 切成多个 stage，并把等待推迟到该 stage 真正要使用前。重叠策略求解器 <code>OverlapSolver</code> 根据通信量和计算量选择切分方式。`,
        R`GPU 内部还要分配执行资源。NCCL（NVIDIA Collective Communications Library）或 native grpcoll（原生分组集合通信）的通信 kernel 都要占 SM。NCCL 路径用 SM 预留量 <code>sm_margin</code> 让 Flex-Flash-Attention（FFA）少启动若干 CTA；native grpcoll 使用常驻通信 kernel，并以 <code>KernelBarrier</code>（kernel 间计数屏障）协调发射顺序。`
      ],
      diagram: {
        key: "overlap",
        caption: "通算融合全景涵盖四层：依赖驱动的 GroupCast/GroupReduce、OverlapSolver 对接收列的分段、前向与反向 overlap 流水，以及 sm_margin/KernelBarrier 的 SM 资源协调。点击节点查看对应源码。"
      },
      explain: [
        {
          title: "通信语义：谁把什么发给谁",
          body: [
            R`<strong>CP（context parallelism，上下文并行）</strong>把长序列的 token 均匀分片，或按计算负载分配到各个 rank；分布式层采用 packed 布局 <code>[num_tokens, num_heads, head_dim]</code>，三个维度依次表示 token 数、注意力头数和单头维度。默认 <strong>KV-comm</strong> 模式让查询张量 Q 常驻，而键/值张量 K/V 移动。不同于固定环形轮转，<code>AttnSlice</code>（注意力切片分析器）根据 mask 依赖生成 <code>CommMeta</code>（通信元数据）；其中 <code>dst_indices[c]</code> 表示 KV 分片 \(c\) 的目标 rank 清单，例如贯穿本章的 rank 2 分片对应 <code>dst_indices[c]=[0,3]</code>，因此每段 KV 只发给真正需要它的 rank。`,
            R`前向中，<code>GroupCast</code>（按目标清单执行的一对多发送）依据 <code>dst_indices</code> 多播每段 K/V；各 rank 再用本地 Q 与收到的远端 K/V 计算 partial attention，产生局部输出 <code>out</code> 和对数归一化项 <code>lse</code>。在 FFA backend 中，<code>out_acc</code> 与 <code>lse_acc</code> 分别是跨 KV stage 累积局部输出和 LSE 的本地缓冲区，因此前向默认不为 out/LSE 发起网络归约。`,
            R`反向沿同一依赖走对偶路径：正向目标表 <code>dst_indices</code> 对应为反向来源表 <code>src_indices</code>，后者记录每段梯度来自哪些 rank；<code>GroupReduce(reduce_op="sum")</code> 中的 <code>reduce_op</code> 指定求和归约，把各处产生的 K/V 梯度 dK/dV 加回 KV 属主，而 Q 的梯度 dQ 留在 Q 属主本地累加。可选的 <strong><code>qo_comm</code></strong>（Q/O 通信模式）交换角色：<code>GroupCast</code> 将 Q 发往 KV rank，<code>GroupReduce(reduce_op="lse")</code> 用注意力专用的 LSE 归约把 partial out/LSE 合并回 Q 属主。实现中的 <code>cp_group_gc</code> 和 <code>cp_group_gr</code> 分别是 cast 与 reduce 的独立通信组，使两条路径能够重叠。`,
            {
              card: true,
              tone: "fact",
              label: "FACT · 通信量级",
              title: R`<strong>512 KiB</strong> 的 KV 对 <strong>8 MiB</strong> 的 partial out`,
              body: [
                R`取批大小 \(B=1\)、序列长度 \(S=4096\)、\(CP=4\)、Q/KV 头数 \(H_q=16,\ H_{kv}=4\)、单头维度 \(D=128\)。张量采用每元素 2 字节的 bf16，每个 rank 持有 \(S/CP=1024\) 个 token。一个 256-token 的 KV 分片发往一个目标时，K 和 V 各为 <code>[256,4,128]</code>，通信量为 \(2\times256\times4\times128\times2=512\ \mathrm{KiB}\)。`,
                R`对应的 fp32 partial out 为 <code>[1024,16,128]</code>，每元素 4 字节，占 \(1024\times16\times128\times4=8\ \mathrm{MiB}\)，恰为上述 KV 的 16 倍。默认 KV-comm 因而只让较小的 KV 上网，把较大的 partial out 留在本地合并。`
              ]
            }
          ],
          svg: "cp-communication"
        },
        {
          title: "依赖矩阵的两个端点：full 对称，causal 偏斜",
          body: [
            R`<strong>例 1：full mask。</strong>每个 rank 的 Q 都需要全部 KV，因此每段 KV 的 <code>dst_indices</code> 都包含其余所有 rank；在 \(CP=4\) 时，每个 rank 发出 3 份自己的 KV 拷贝，也收到 3 段远端 KV，单向通信量均为 \((CP-1)/CP\) 份全量 KV。这个总字节数与 Ring Attention 环形轮转 \(CP-1\) 步<strong>完全相同</strong>：GroupCast 在 full mask 下不省字节，差别只在一步直达而非逐跳转发的拓扑，以及更自由的重叠调度。`,
            R`<strong>例 2：causal mask，按 token 连续均匀分片的教学假设。</strong>rank \(r\) 的 Q 只依赖 rank \(0,\ldots,r\) 的 KV，所以 <code>dst_indices[KV_r] = {r+1,…,CP−1}</code>；\(CP=4\) 时发送份数为 3/2/1/0、接收段数为 0/1/2/3，总通信量恰为 full 的一半。计算量又随所覆盖的下三角 mask 面积增长，形成通信与计算的双重偏斜：rank 0 早早算完干等，rank 3 则又算又收。这里先看清依赖矩阵，后面的 causal 每拍表会把这种偏斜展开到 chunk 粒度。`,
            R`反向 dKV 的传输矩阵是前向矩阵的<strong>转置</strong>：谁收过 KV，谁就把 partial dKV 发回其属主，因此 full 仍然对称，而上述 causal 例子变成 rank 0 只收、rank 3 只发。这两个例子构成两个端点；mask 越稀疏、越不规则，矩阵上的“洞”就越多，例如上一小节中 rank 2 的分片只被 rank 0/3 需要，此时 GroupCast 相对固定轮转省下的冗余通信也越多。`
          ],
          svg: "cp-comm-examples"
        },
        {
          title: "GroupCast / GroupReduce：为不规则依赖定制的集合通信",
          body: [
            R`<strong>group_cast</strong> 先按 <code>input_split_sizes</code> 切段，再用 <code>dst_indices[i]</code> 指定每段要发给哪些 rank。<strong>group_reduce</strong> 做反向操作：把多个来源的对应段送回属主，并按 sum、avg 或注意力专用的 <code>"lse"</code> 规则合并。`,
            R`默认 NCCL 实现会先把多播数据打包成每个目标 rank 一份，再调用变长 all-to-all（a2av，all-to-all-v）。接收后执行索引重排；group_reduce 则在接收后做本地 sum/LSE 合并。这条路径简单通用，但需要额外 pack/unpack。`,
            R`<strong>native grpcoll</strong> 直接在 NVLink 或 RDMA（remote direct memory access，远程直接内存访问）对称缓冲区中多播和归约，减少中间拷贝。<strong>hierarchical</strong>（分层）路径先跨节点传一次，再在目标节点内扇出。三种实现使用相同接口。`
          ],
          formula: R`<p>group_reduce 的 <code>"lse"</code> 归约复用第 0 章的结果合并公式。属主收到 \(r\) 个局部结果 \((out_k,lse_k)\) 后计算</p>
\[ \mathrm{lse} = \log\!\sum_k e^{\mathrm{lse}_k}, \qquad \mathrm{out} = \sum_k e^{\mathrm{lse}_k - \mathrm{lse}}\, out_k . \]
<p>实数运算下该操作满足交换律和结合律，因此可以分阶段合并。实际浮点结果仍可能随归约顺序产生微小舍入差异。</p>`
        },
        {
          title: "从接收列到每一拍：overlap_degree 的含义",
          body: [
            R`<code>overlap_degree</code> 的本义，是把<strong>每个 rank 要接收的远端 KV</strong>按 chunk 粒度切成多少段；chunk 是按 <code>chunk_size</code> 个 token 划分的连续段。切分采用接收方视角，也就是切开通信矩阵中属于本 rank 的远端接收列；每个 stage 对应一轮 <code>group_cast</code> 集合通信和一次 FFA 计算。各接收方的划分投影回发送侧后，才决定发送方每一拍发什么、发给谁，并记录在 <code>CommMeta</code> 的逐 stage 清单中。`,
            R`图中的 full mask 取 \(CP=4\)、<code>overlap_degree=3</code>：每个 rank 的 6 个远端 chunk 被均分成 3 拍，每拍接收 2 个 chunk，错峰置换使每拍每个 rank 单发单收。第 \(i\) 拍先用一轮 <code>group_cast</code> 收取 stage \(i\) 并预取 stage \(i+1\)，随后 FFA 让本地 Q 完整遍历本拍的 KV 子集，得到 partial <code>(out, lse)</code>。这些局部结果经 <code>out_acc/lse_acc</code> 按合并公式跨拍累积，正对应 FlashAttention 在 kernel 内用 \(m/\ell/O\) 递推的分布式重现。整个前向中，每个远端 KV chunk 恰好传输并遍历一次，而本地 Q 会被重读 <code>degree+1</code> 次：一次 host 拍，加上每个远端拍各一次。`,
            {
              card: true,
              tone: "source",
              label: "SOURCE · OverlapConfig",
              title: R`<code>degree</code> 的四种取值与两道切分护栏`,
              body: [
                R`源码默认 <code>degree=1</code>，表示本地段加一个远端段；<code>degree=0</code> 是用户侧的阻塞式无重叠写法，初始化时归一化为 <code>degree=1</code>，并把 <code>max_num_chunks</code> 压到 1；<code>degree=N≥2</code> 表示静态 \(N\) 段；<code>degree=None</code> 表示动态搜索，并受 <code>dynamic_max_degree</code> 约束。`,
                R`默认 <code>min_chunk_size=512</code> 与 <code>max_num_chunks=64</code> 分别限制 chunk 的最小 token 数和最大数量，防止切分过细。阻塞模式将远端 KV 合并后只调用一次 FFA，也避免分段 LSE 合并带来的精度损失。`
              ]
            },
            R`<code>degree</code> 与 \(CP-1\) 没有绑定，图中取 3 只是为了对齐属主边界。每多一拍会增加一次 kernel/通信发射和一轮 a2av 固定延迟。接收缓冲峰值则由预取策略决定：逐段预取的双缓冲约为 \(2\times(\mathrm{全部远端\ KV}/\mathrm{degree})\)，一次性预取全部 stage 时等于全部远端 KV、与 <code>degree</code> 无关。`
          ],
          svg: "overlap-degree-schedule"
        },
        {
          title: "causal 的每一拍：不等长接收列与空拍",
          body: [
            R`<strong>causal 会同时破坏 full 的两种均匀性。</strong>在连续分片的教学假设下，chunk 顺序为 \(c_0 \lt \cdots \lt c_7\)，rank \(r\) 持有 \(c_{2r},c_{2r+1}\)。当 \(CP=4\) 时，四个 rank 的远端接收列长度依次为 0/2/4/6 个 chunk，这是 rank 之间的第一重不均；同一接收列内，各 chunk 的计算量又随 causal mask 面积而异，越靠前的 chunk 被越多 Q 行依赖。于是 rank 3 不仅接收最多，也承担最多计算。`,
            {
              card: true,
              tone: "recipe",
              label: "RECIPE · 三步口诀",
              title: R`<strong>列清单 → host 占位均分 → 不够补空拍</strong>`,
              body: [
                R`<ol><li><strong>第 1 步 · 列清单：</strong>站在接收方视角，远端 chunk 按 token 顺序排列。causal 连续分片下，rank \(r\) 的清单是 \(c_0,\ldots,c_{2r-1}\)，长度 \(L=2r\)。</li><li><strong>第 2 步 · host 占位 + 均分余数靠前：</strong>清单最前加入 host 占位符，本地计算占第 0 拍一个名额；\(L+1\) 项按有效段数均分，余数从前往后每组多摊 1 项。移除占位符后，每组剩余的远端 chunk 就是该拍所收。</li><li><strong>第 3 步 · 不够补空拍：</strong>\(L<\mathrm{degree}\) 时，有效 degree 为 \(\min(\mathrm{degree},L)\)，缺少的尾拍留空；空拍 rank 仍参与该轮集合通信。</li></ol>`,
                R`这套切法自然产生四条规律：<strong>拍 0 总是轻一格</strong>，因为它只能依靠 host 计算掩护通信；所有 rank 的清单都从 \(c_0\) 开始，前缀 chunk 因而在拍 0 集中多播，且 causal 下越靠后的 chunk 出场越晚、扇出越小；有效 degree 为 \(\min(\mathrm{degree},L)\)，缺口由空拍补齐；每拍接收块数约为 \((L+1)/\mathrm{degree}\)，可用于估算单拍通信量。`,
                R`图中 rank 2 的 \(L=4\)：<code>[host,c₀,c₁,c₂,c₃]</code> 共 5 项分 3 组，组大小为 \(2/2/1\)，两个余项靠前分摊；扣除第 0 组中的 host 后，三拍远端块数为 \(1/2/1\)，即 \(c_0/(c_1c_2)/c_3\)。`
              ]
            },
            R`<code>group_cast</code> 是集合操作，所以 <code>overlap_degree=3</code> 意味着所有 rank 都执行全局一致的 3 拍。图中的完整接收表为：rank 0 收 \(\varnothing/\varnothing/\varnothing\)，rank 1 收 \(c_0/c_1/\varnothing\)，rank 2 收 \(c_0/(c_1c_2)/c_3\)，rank 3 收 \((c_0c_1)/(c_2c_3)/(c_4c_5)\)。源码 <code>_solve_with_uniform</code> 通过 <code>overlap_degree_idle</code> 在 <code>partitions</code> 末尾补空分组；空拍中的 rank 仍照常参与本轮集合通信，只是不接收数据，因而不会破坏其他 rank 的收发次序。`,
            R`实际训练的顺序是 <code>dispatch solver</code> 先重新分片，尽量抹平各 rank 的计算量与接收列长度，再由 overlap 策略切拍；因此图中的空拍和双重偏斜是<strong>配平之前</strong>的形态。配平后，列内 chunk 的计算量仍可能不同：uniform 按块数均分只能让每拍通信量接近，greedy 等算法才会按估计耗时做装箱。取值直觉来自 causal 的尺度关系——计算量正比于 mask 面积，通信量正比于边长，长序列下较小的 <code>degree=1~2</code> 往往已经足以隐藏通信；只有通信相对计算变贵时，才更需要较大的 degree。`
          ],
          svg: "overlap-degree-causal"
        },
        {
          title: "前向流水：prefetch / compute / reduce 三重奏",
          body: [
            R`开始时先预取 stage 0，同时用本地 KV 计算。主循环第 \(i\) 拍：等待 stage \(i\) 到位并预取 \(i+1\)；用 stage \(i\) 启动 FFA；把上一拍结果交给异步归约。通信流与计算流只在必要等待点同步。`,
            R`默认 KV-comm 模式下，Q 留在本 rank，不同 KV stage 的局部 out/LSE 也都留在本地。backend 可把上一拍结果作为 <code>out_acc/lse_acc</code> 传给后续 kernel 合并。真正跨网络的 GroupReduce 主要用于反向 dKV，或 qo_comm 模式下把局部 out/LSE 送回 Q 属主。`,
            R`通常可提前发出多个 stage 的预取。但当 <code>CUDA_DEVICE_MAX_CONNECTIONS=1</code> 或 native grpcoll 复用同一缓冲区时，只能逐 stage 预取，以免队列阻塞或缓冲区冲突。`
          ],
          svg: "overlap-timeline"
        },
        {
          title: "反向流水：cast 与 reduce 分道而行",
          body: [
            R`反向流水先用本地 KV 完成 host 段反向，并预取远端 stage 0。进入主循环后，第 \(i\) 拍会用 GroupCast 预取下一 stage 的远端 KV，在<strong>单独的 CUDA stream</strong> 上对上一 stage 的 partial dKV 发起 <code>GroupReduce(sum)</code>，再发射当前 stage 的 FFA 反向 kernel。除最后一段 dKV 归约外，这些通信都可被本层反向计算遮蔽。`,
            R`这里与前向有一个关键区别：默认 KV-comm 的前向 reduce 拍只是空壳句柄，out/LSE 已由 kernel 的 <code>out_acc/lse_acc</code> 在本地合并。反向 reduce 拍则是真正的网络归约，依据 <code>src_indices</code> 镜像清单，把散落在各消费 rank 上的 partial dK/dV 求和送回 KV 属主。dQ 仍在 Q 属主本地累加，不会上网。在 PyTorch 中，一个进程组（process group）对应一条集合通信 CUDA 流；若 cast 与 reduce 共用同一组，两类通信 kernel 就会在同一流上串行，因此实现分别使用 <code>cp_group_gc</code> 承载 GroupCast、<code>cp_group_gr</code> 承载 GroupReduce。`,
            R`最后一段 dKV 归约之后，本层已经没有计算可以继续遮蔽它，因此这段<strong>尾段归约</strong>会暴露在时间线末端。<code>save_tail_stage</code> 通过调整发射顺序，尝试把尾段归约藏进下一层的反向计算。`
          ],
          svg: "overlap-timeline-bwd"
        },
        {
          title: "OverlapSolver：切几段、每段装什么",
          body: [
            R`明确 degree 与每拍的语义后，<code>OverlapSolver</code> 还要决定候选 degree 和各 stage 的装箱。对候选 chunk \(j\)，\(C_j^{\mathrm{comm}}\) 表示估计通信时间，\(C_j^{\mathrm{calc}}\) 表示估计计算时间；模型假设第 \(i\) 段通信可与第 \(i-1\) 段计算重叠，并据此估算总时延。`,
            {
              card: true,
              tone: "source",
              label: "SOURCE · _solve_with_uniform",
              title: R`源码实际切分的是 <code>[host, remote₀, remote₁, …]</code>`,
              body: [
                R`<code>stage_costs</code> 的第 0 项是 host 本地计算，后面是按自然顺序排列的远端 chunk。默认 <code>random_costs=false</code>；开启后只先打乱远端项，再把 host 放回清单最前，因此 host 始终位于 <code>partition 0</code> 并占一个名额。`,
                R`设远端项数为 \(L\)、请求 degree 为 \(d\)。源码先取有效切分段数 \(d_{\mathrm{eff}}=\min(d,L)\)；当 \(L>0\) 时，把 \(L+1\) 个清单项分成 \(d_{\mathrm{eff}}\) 组，每组先取 \(\lfloor(L+1)/d_{\mathrm{eff}}\rfloor\) 项，余下的 \((L+1)\bmod d_{\mathrm{eff}}\) 项从前往后各组多摊 1 项。host 因而固定挤掉第 0 拍的一个远端名额。`,
                R`当 \(L \lt d\) 时，<code>overlap_degree_idle=d-L</code> 个空分组追加在 <code>partitions</code> 末尾，使结果仍有全局一致的 \(d\) 拍；\(L=0\) 时只有 host 留在第 0 分组，其余视图全为空拍。源码 docstring 将 uniform 明确称为 “a dummy but feasible solution ... instead of production”；<code>greedy</code> 是按估计耗时装箱的备选算法。`
              ]
            },
            R`host 占位使第 0 拍少装一个远端 chunk，正好对应前文三步口诀中的“拍 0 轻一格”：首拍通信只能由 host 计算遮蔽。uniform 平衡的是清单项数和近似通信量，不保证由 mask 面积决定的计算耗时相等；因此相同 chunk 数的两拍仍可能成为不同长度的关键路径。`,
            R`段太碎会增加 kernel 发射和 a2av 的固定开销；段太粗又难以把首段通信藏进本地计算。因此 solver 不是追求最大的 degree，而是在首段暴露、逐拍重叠、额外发射与缓冲策略之间折中。`
          ],
          formula: R`<p>设远端数据被切成 \(d\) 个 stage，\(P_i\) 是第 \(i\) 个 stage 包含的 chunk 集合；\(C_j^{\mathrm{comm}}\) 和 \(C_j^{\mathrm{calc}}\) 分别是 chunk \(j\) 的估计通信、计算时间；\(C_{\mathrm{host}}^{\mathrm{calc}}\) 是本地 KV 的计算时间。总时延估计为：</p>
\[ T \;=\; \max\!\Big(\textstyle\sum_{j\in P_0} C^{\mathrm{comm}}_j,\; C^{\mathrm{calc}}_{\mathrm{host}}\Big) \;+\; \sum_{i=1}^{d-1}\max\!\Big(\textstyle\sum_{j\in P_i} C^{\mathrm{comm}}_j,\; \sum_{j\in P_{i-1}} C^{\mathrm{calc}}_j\Big) \;+\; \sum_{j\in P_{d-1}} C^{\mathrm{calc}}_j . \]
<p>每个 \(\max\) 表示同一拍中的通信和计算并行，较慢的一项决定该拍时长。该公式是理想模型：它没有显式计入队列阻塞、固定发射开销和资源竞争，这些因素需要实测校准。</p>`
        },
        {
          title: "SM 的分配：sm_margin 与 KernelBarrier",
          body: [
            R`NCCL a2av 也是 GPU kernel。若 persistent（常驻式）FFA 占满所有 SM，通信会等计算结束。前向 SM 预留量 <code>fwd_sm_margin</code> 让 FFA 少启动若干 CTA，给通信保留资源。margin 太大会拖慢计算，太小又不足以支撑通信带宽，因此必须按硬件和形状实测。`,
            R`native grpcoll 使用固定数量的常驻通信 SM。此时主要风险变成发射顺序：通信 kernel 不能在依赖的计算尚未完成时占住资源等待。<code>KernelBarrier</code> 用 GPU 计数器协调两类 kernel，因此这条路径不再额外设置 <code>fwd_sm_margin</code>。`,
            R`反向中 GroupCast 与 GroupReduce 各占一条通信流，会同时与 FFA 争用 SM，因此 margin 与 <code>KernelBarrier</code> 的资源权衡比前向更紧。两条通信流的流水细节见前文反向 overlap 小节。`
          ]
        }
      ],
      warning: R`<code>overlap_degree</code> 不是越大越好。每多一段都会增加一次 kernel/通信发射和一轮 a2av 固定延迟；接收缓冲峰值取决于预取深度，而不是由 degree 单独决定。还要区分模式：默认 KV-comm 的前向 out/LSE 在本地合并；反向 dKV 和 qo_comm 才使用网络 GroupReduce。`,
      exercises: [
        {
          kind: "概念", level: "基础",
          q: R`CP（上下文并行）度为 4。rank 2 的某段 KV 同时被 rank 0 和 rank 3 的 Q 需要，而 rank 2 自己的 Q 不需要它。写出这段 KV 在 <code>group_cast</code> 目标表 <code>dst_indices</code> 中的条目，以及反向时对应 dKV 段在 <code>group_reduce</code> 来源表 <code>src_indices</code> 中的条目。`,
          hint: R`group_reduce 是 group_cast 的镜像。`,
          answer: R`前向 group_cast（rank 2 视角）：该段的 <code>dst_indices[i] = [0, 3]</code>——一段数据多播给两个消费者。反向 group_reduce（rank 2 是属主）：该段的 <code>src_indices[i] = [0, 3]</code>——rank 0 和 rank 3 各算出一份 partial dK/dV,按 sum 归约回 rank 2。镜像对称正是「通信按依赖清单精确投递」的体现：不在 mask 里的 (Q,K) 对,一个字节都不会上网络。`
        },
        {
          kind: "排程", level: "基础",
          q: R`连续分片的 causal 教学例中，\(CP=4\)、<code>overlap_degree=3</code>，四个 rank 的远端接收列长为 0/2/4/6 个 chunk。已知非空 partitions 为：rank 1 的 \([c_0],[c_1]\)，rank 2 的 \([c_0],[c_1,c_2],[c_3]\)，rank 3 的 \([c_0,c_1],[c_2,c_3],[c_4,c_5]\)，rank 0 没有远端 chunk。补出四个 rank 的三拍接收表，统计每个 rank 的空拍数，解释为什么空拍 rank 不能跳过该轮 <code>group_cast</code>，并用“三步口诀”推导 rank 2 的 \(2/2/1\) 清单分组及 \(1/2/1\) 远端块数。`,
          hint: R`rank 2 先列出 <code>[host,c₀,c₁,c₂,c₃]</code>；余数靠前分摊。<code>overlap_degree_idle</code> 在不足三组时向末尾补空组。`,
          answer: R`接收表的三拍分别为：rank 0 是 \(\varnothing/\varnothing/\varnothing\)，rank 1 是 \(c_0/c_1/\varnothing\)，rank 2 是 \(c_0/(c_1c_2)/c_3\)，rank 3 是 \((c_0c_1)/(c_2c_3)/(c_4c_5)\)；空拍数依次为 3、1、0、0。rank 2 的口诀推导是：<strong>第 1 步列清单</strong>，\(L=4\)，远端项为 \(c_0,c_1,c_2,c_3\)；<strong>第 2 步 host 占位并均分</strong>，<code>[host,c₀,c₁,c₂,c₃]</code> 共 5 项分 3 组，\(5=1\times3+2\)，两个余项靠前分摊，所以组大小为 \(2/2/1\)，具体是 <code>[host,c₀]/[c₁,c₂]/[c₃]</code>，扣除 host 后得到 \(1/2/1\) 个远端块；<strong>第 3 步补空拍</strong>，此处 \(L=4\ge3\)，无需补空。对 rank 1，\(L=2<3\)，末尾才补一拍空组。<code>group_cast</code> 是集合操作，空拍 rank 仍须参与同一轮调用以维持全局一致的收发次序；空拍只表示该 rank 不接收数据。这张表描述的是 dispatch solver 重新分片配平之前的形态。`
        },
        {
          kind: "推导", level: "基础",
          q: R`设本地 KV（host 段）计算耗时 8 ms，三个远端段的（通信，计算）耗时分别为 (6,7)、(5,6)、(4,5) ms。分别计算切分度 <code>degree=3</code> 的流水线与“先拉全再算”两种方案的总时延。`,
          hint: R`套用 _calc_overall_cost 的公式。`,
          answer: R`流水线：\(T=\max(6,8)+\max(5,7)+\max(4,6)+5=26\) ms，等于纯计算时间。先拉全再算：在“各段共享同一通信带宽、通信时间相加”的教学假设下，通信为 \(6+5+4=15\) ms，再加 26ms 计算，共 41ms。此例中每段通信都短于前一段计算，因此可被完全隐藏。`
        },
        {
          kind: "源码", level: "进阶",
          q: R`前向主循环中，局部结果归约入口 <code>reduce_partial_out_lse</code> 在默认 KV-comm 模式下并不发起通信，因为合并已由 kernel 的累加缓冲 <code>out_acc</code> 完成；但它仍为每个 stage 压入一个异步工作包装器 <code>WorkWithPostProcessFn</code>，并在结果准备函数 <code>prepare_reduced_local_out_lse</code> 中逐个等待。解释这层“空壳异步”的设计意图。`,
          hint: R`看 qo_comm 模式下同一个调用点会发生什么。`,
          answer: R`这是接口的<strong>模式无关性</strong>设计：qo_comm 模式下同一调用点返回真正的 group_reduce 异步句柄（lse 归约在网络上跑）,KV-comm 模式下返回 no-op 句柄（post_process 恒等）。主环代码于是完全不感知模式差异——「每个 stage 交一个归约句柄、结束前统一 wait」的契约恒成立。空壳的代价是几个 Python 对象,换来的是 forward 主环零分支、以及未来加新归约路径（如 native grpcoll 的 fp32 缓冲归约）时不动主环。`
        },
        {
          kind: "系统", level: "进阶",
          q: R`CUDA 硬件工作队列连接数 <code>CUDA_DEVICE_MAX_CONNECTIONS=1</code> 时，为什么“host stage 一次发出全部 prefetch（预取）”会几乎消除 overlap？而留下的例外是“只有最后一个 stage 的 prefetch 还能重叠”——解释这个例外。`,
          hint: R`想象单条硬件队列里的入队顺序：cast0, cast1, cast2, ffa_host, ffa_0, ...`,
          answer: R`单 connection 下所有 stream 的工作折叠进一条硬件队列,queue 头部阻塞后续所有条目。一次发全 prefetch 的入队序是 [cast0, cast1, cast2, ffa_host, ...]：ffa_host 要等三个 cast 全部出队才开始——通信不是藏在计算影子里,而是把计算推后,恰好反转。逐 stage 预取的入队序是 [cast0, ffa_host, cast1, ffa_0, cast2, ...]：每个 cast 只挡住它<strong>本来就该等的</strong>下一拍计算的发射点,cast_i 与 ffa_{i-1} 的执行仍然重叠。「最后一个 stage 例外」：cast_{d-1} 之后没有更多 cast 入队,它后面的 ffa 发射不再被新通信阻塞,所以哪怕一次发全,最末段通信仍与它前面的计算天然重叠。`
        },
        {
          kind: "系统", level: "挑战",
          q: R`SM 预留量 <code>sm_margin</code> 的两难：设 FFA 计算在 148 个 SM 上耗时 \(T_c\)，通信在带宽不受 SM 数约束时耗时 \(T_m\)。预留 \(s\) 个 SM 后，计算耗时变为 \(T_c\frac{148}{148-s}\)（线性模型）。写出端到端时延 \(T(s)\) 并给出最优 margin 的判据；解释为什么 native grpcoll 与 <code>KernelBarrier</code> 能同时改善这一权衡的两端。`,
          hint: R`s 太小通信排队（退化为串行）,s 够大才真正并行。`,
          answer: R`\(s>0\) 且通信 kernel 能被容纳时 \(T(s) = \max\!\big(T_c\frac{148}{148-s},\, T_m(s)\big)\)（\(T_m(s)\) 随 s 增大先降后平——通信自身也需要足够 SM 达到线速）;\(s=0\) 时退化为 \(T_c + T_m\)。最优 s 在两条曲线交点：计算膨胀恰好等于通信耗时。native grpcoll 改善两端：(1) 通信 kernel 是定制的收发/归约 kernel,单 SM 效率高于 NCCL 通用路径,\(T_m(s)\) 曲线整体下移、更小的 s 即达线速;(2) fp32 归约融合进通信 kernel,省掉 NCCL 路径上额外的本地归约 kernel（那也要抢 SM）;KernelBarrier 则消除了「怕通信 kernel 抢跑」而保守多留 margin 的需要——sm_margin 直接归零,计算不再膨胀。`
        },
        {
          kind: "设计", level: "挑战",
          q: R`MagiAttention 论文声称“线性可扩展”：固定每个 rank 的序列长度，CP 度翻倍时迭代时间近似不变。用本章的成本模型写出线性可扩展成立的两个充分条件，并各举一个会破坏它的现实场景。`,
          hint: R`一个条件关于 dispatch（任务派发），另一个关于 comm/calc（通信/计算）之比。`,
          answer: R`条件一（计算侧）：dispatch solver（任务派发求解器）使每个 rank 的 mask 面积 \(\sum C^{\mathrm{calc}}\) 与 CP 度无关，即负载均衡是完美的。破坏场景：极端不规则 mask 下，最重的单个 AttnSlice 超过均值，最小堆 <code>minheap</code> 也无法切开它（切片粒度下界），该 rank 会持续掉队。条件二（通信侧）：每个 stage 满足 \(\sum_{j\in P_i}C^{\mathrm{comm}}_j\le\sum_{j\in P_{i-1}}C^{\mathrm{calc}}_j\)，即通信始终能够藏入计算；由于 causal 类 mask 下计算量正比于面积、通信量正比于边长，序列越长，该不等式越宽松。破坏场景：跨节点带宽骤降（如 RDMA 降级到 TCP）使 \(C^{\mathrm{comm}}\) 乘上较大系数，或 <code>head_dim</code> 很小、mask 很稀疏，使单位通信量对应的计算量不足——通信会从计算的“影子”中露出，扩展曲线随之弯折。`
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
