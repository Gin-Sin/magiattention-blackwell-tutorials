/* Interactive optimization-lineage map for the course homepage.
 *
 * Renders a layered dependency graph of the kernel/distributed optimization
 * points: hover previews the upstream/downstream chain, click pins a node and
 * fills the side panel with the "why it exists / what it enables" story plus
 * a deep link into the owning chapter.
 */
(function () {
  "use strict";

  var NODES = [
    /* L0 · 契约层 */
    {
      id: "rep", x: 140, y: 46, w: 250, tone: "compute", no: "00",
      title: "AttnSlice 切片表示", sub: "任意 mask → 矩形切片 × 4 种几何",
      chapter: "attnslice",
      detail: "把 mask 从「形状」变成「清单」：任意注意力模式分解为 (QRange, KRange, MaskType) 三元组的集合。切片内部只有 FULL/CAUSAL/INV-CAUSAL/BI-CAUSAL 四种几何——kernel 不再枚举 mask 分支，分布式不再盲目全交换。"
    },
    {
      id: "semi", x: 660, y: 46, w: 260, tone: "remote", no: "00",
      title: "out / LSE 可合并半群", sub: "任意顺序、任意分组合并",
      chapter: "attnslice",
      detail: "两个 partial 结果 (out₁,lse₁)、(out₂,lse₂) 可用封闭公式合并，且满足交换律与结合律。这一数学性质是切片可重叠、块可乱序、rank 可分布的统一许可——单卡与分布式共享同一个代数结构。"
    },

    /* L1 · 硬件基座 */
    {
      id: "tmem", x: 120, y: 164, w: 220, tone: "dispatch", no: "01",
      title: "TMEM 常驻累加", sub: "S / P / O 不占寄存器堆",
      chapter: "blackwell",
      detail: "Blackwell 的 Tensor Memory 让 128×512 fp32 的累加器脱离寄存器堆常驻片上：S、P、O 通过 T2R/R2T 精确搬运，寄存器全部让给 softmax 的逐元素计算。"
    },
    {
      id: "engines", x: 370, y: 164, w: 250, tone: "dispatch", no: "01",
      title: "TMA / UMMA 异步引擎", sub: "单线程发射 · 硬件记账",
      chapter: "blackwell",
      detail: "数据搬运（TMA）与矩阵乘（tcgen05 UMMA）都变成「单线程发射、硬件异步完成」的引擎。计算与搬运的并行不再靠人海战术，而靠少数驾驶员 warp 与 mbarrier 记账。"
    },

    /* L2 · kernel 执行 */
    {
      id: "mask", x: 40, y: 282, w: 210, tone: "compute", no: "03",
      title: "三层 Mask 防线", sub: "跳块 → 三段循环 → R2P",
      chapter: "mask",
      detail: "同一条「列界随行号线性移动」的几何驱动三层机制：BlockInfo 整块跳过（免费）、三段主循环隔离 partial 带（近免费）、R2P 位掩码逐元素写 -inf（只在对角带付钱）。"
    },
    {
      id: "pipe", x: 290, y: 282, w: 230, tone: "communication", no: "02",
      title: "Warp 特化 · 六流水线", sub: "槽位状态是唯一协调语言",
      chapter: "pipeline",
      detail: "16 个 warp 按角色分派（load/MMA/softmax/correction/epilogue），六条 mbarrier 流水线管理槽位的满/空。warp 之间不传数据，只翻转状态——数据永远躺在约定好的 TMEM/SMEM 里。"
    },
    {
      id: "softmax", x: 560, y: 282, w: 220, tone: "compute", no: "04",
      title: "Online Softmax", sub: "驻留 TMEM · exp2 双管线",
      chapter: "softmax",
      detail: "以 2 为底换底后内层只剩 packed FMA 与 ex2；row_max 更新即早发布 corr_scale，P 写到 3/4 提前放行 PV GEMM——递推的每一步都在为并行让路。"
    },
    {
      id: "corr", x: 820, y: 282, w: 240, tone: "remote", no: "05",
      title: "Correction 收口", sub: "块间归约 · 写出 (out, lse)",
      chapter: "correction",
      detail: "主循环用 corr_scale 还账、尾声用 1/ℓ 清算，最后写出 fp32 的 LSE——那是给「下一次合并」的收据：kernel 内块间归约与分布式 rank 间归约，是同一半群运算在两个尺度上的重复。"
    },

    /* L3 · 调度与反向 */
    {
      id: "sched", x: 90, y: 400, w: 230, tone: "dispatch", no: "06",
      title: "LPT · L2 swizzle · CLC", sub: "顺序 × 亲和 × 分配",
      chapter: "scheduler",
      detail: "causal 负载沿 Q tile 线性递增：LPT 反转派发让重块先行；L2 swizzle 让时间相邻的 CTA 共享 head 的 K/V；CLC 把分配权交还硬件。三个正交自由度各司其职。"
    },
    {
      id: "persist", x: 370, y: 400, w: 230, tone: "dispatch", no: "06",
      title: "Persistent Kernel", sub: "CTA 常驻 · tile 软件派发",
      chapter: "scheduler",
      detail: "每 SM 常驻一个 CTA，work tile 由软件（或 CLC 硬件）逐个派发。CTA 数量成为一个可调参数——这正是后来 sm_margin 能「少开几个、让出地皮」的结构性前提。"
    },
    {
      id: "bwd", x: 650, y: 400, w: 250, tone: "compute", no: "07",
      title: "反向 5-GEMM", sub: "以 K 为家 · dQ 原子归约",
      chapter: "backward",
      detail: "固定 K/V 扫过所有 Q：dK/dV 本地累加，dQ 用 TMA bulk atomic-add 全局记账；P 不存不传，用 LSE 重算。分布式反向还要把 dK/dV 送回 KV 属主——这笔通信交给 GroupReduce。"
    },

    /* L4 · 分布式 */
    {
      id: "margin", x: 60, y: 518, w: 240, tone: "orange", no: "08",
      title: "sm_margin / KernelBarrier", sub: "给通信 kernel 让出 SM",
      chapter: "overlap",
      detail: "异步 ≠ 并行：通信 kernel 也要 SM。NCCL 路径让 FFA persistent kernel 少开 sm_margin 个 CTA 留出地皮；native grpcoll 路径通信 kernel 自带 SM，用 KernelBarrier 钉死发射顺序、margin 归零。"
    },
    {
      id: "overlap", x: 350, y: 518, w: 250, tone: "communication", no: "08",
      title: "多阶段 Overlap 流水线", sub: "prefetch ∥ compute ∥ reduce",
      chapter: "overlap",
      detail: "远端 KV 按成本模型切成 d 个 stage，每一拍并行三件事：预取第 i+1 段、计算第 i 段、归约第 i−1 段。通信藏进计算的影子——只有最后一段计算裸露，这是「线性可扩展」的工程形态。"
    },
    {
      id: "grpcoll", x: 650, y: 518, w: 260, tone: "communication", no: "08",
      title: "GroupCast / GroupReduce", sub: "按依赖清单精确投递",
      chapter: "overlap",
      detail: "「一段发多家」与「多家归一段」两条原语，输入就是 AttnSlice 分析出的依赖清单：不在 mask 里的 (Q,K) 对一个字节都不上网络。reduce_op=\"lse\" 让归约本身执行半群合并公式。"
    }
  ];

  var EDGES = [
    { from: "rep", to: "semi", d: "M390 75H660", label: "切片可重叠、跨 rank ⇒ 输出必须可合并" },
    { from: "rep", to: "mask", d: "M155 104V140H90V282", label: "四种几何的列界都是行号的线性函数" },
    { from: "semi", to: "corr", d: "M900 104V140H1010V282", label: "kernel 内块间合并 = 同一半群运算" },
    { from: "semi", to: "grpcoll", d: "M900 104V126H1080V498H850V518", label: "reduce_op=\"lse\" 直接实现合并公式" },
    { from: "tmem", to: "softmax", d: "M230 222V252H620V282", label: "S/P 常驻片上 · T2R/R2T 免搬运" },
    { from: "engines", to: "pipe", d: "M450 222V282", label: "异步引擎需要 warp 特化来驾驶" },
    { from: "pipe", to: "softmax", d: "M520 311H560", label: "S-full / P-release 槽位协议供数" },
    { from: "pipe", to: "corr", d: "M470 282V262H900V282", label: "sScale 握手 · cross-release 错峰" },
    { from: "mask", to: "sched", d: "M160 340V400", label: "causal 负载线性递增 ⇒ LPT 反转派发" },
    { from: "corr", to: "grpcoll", d: "M930 340V486H880V518", label: "(out, lse) 是 GroupReduce 的载荷" },
    { from: "sched", to: "persist", d: "M320 429H370", label: "同一个 while work_tile 消费骨架" },
    { from: "persist", to: "margin", d: "M420 458V488H240V518", label: "CTA 数可调 ⇒ 少开即让出 SM" },
    { from: "margin", to: "overlap", d: "M300 547H350", label: "通信有地皮，预取才真正并行" },
    { from: "grpcoll", to: "overlap", d: "M650 547H600", label: "按清单投递的原语是流水线的积木" },
    { from: "bwd", to: "overlap", d: "M700 458V492H520V518", label: "反向 dKV 归约同样进 overlap 环" }
  ];

  var CHAPTER_NAMES = {
    attnslice: "第 00 章 · AttnSlice 契约",
    blackwell: "第 01 章 · TMEM / UMMA / TMA",
    pipeline: "第 02 章 · Warp 特化流水线",
    mask: "第 03 章 · 块级 Mask",
    softmax: "第 04 章 · Online Softmax",
    correction: "第 05 章 · Correction",
    scheduler: "第 06 章 · Tile Scheduler",
    backward: "第 07 章 · 反向传播",
    overlap: "第 08 章 · 通算融合"
  };

  function esc(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  var byId = {};
  NODES.forEach(function (node) { byId[node.id] = node; });

  var upOf = {};
  var downOf = {};
  NODES.forEach(function (node) {
    upOf[node.id] = [];
    downOf[node.id] = [];
  });
  EDGES.forEach(function (edge) {
    downOf[edge.from].push(edge);
    upOf[edge.to].push(edge);
  });

  function collectChain(id) {
    var nodes = {};
    var edges = {};
    nodes[id] = true;
    var stack = [id];
    while (stack.length) {
      var current = stack.pop();
      upOf[current].forEach(function (edge) {
        edges[edge.from + ">" + edge.to] = true;
        if (!nodes[edge.from]) {
          nodes[edge.from] = true;
          stack.push(edge.from);
        }
      });
    }
    stack = [id];
    while (stack.length) {
      var cursor = stack.pop();
      downOf[cursor].forEach(function (edge) {
        edges[edge.from + ">" + edge.to] = true;
        if (!nodes[edge.to]) {
          nodes[edge.to] = true;
          stack.push(edge.to);
        }
      });
    }
    return { nodes: nodes, edges: edges };
  }

  function nodeMarkup(node) {
    var h = 58;
    var cx = node.x + node.w / 2;
    return (
      '<g class="lineage-node lineage-node--' + node.tone +
      '" data-node="' + node.id + '" role="button" tabindex="0" aria-label="' +
      esc(node.title + "，属于" + (CHAPTER_NAMES[node.chapter] || "")) + '">' +
      '<rect x="' + node.x + '" y="' + node.y + '" width="' + node.w +
      '" height="' + h + '" rx="10"/>' +
      '<circle cx="' + (node.x + 16) + '" cy="' + (node.y + 15) + '" r="9.5"/>' +
      '<text class="lineage-node__no" x="' + (node.x + 16) + '" y="' + (node.y + 18) +
      '" text-anchor="middle">' + esc(node.no) + "</text>" +
      '<text class="lineage-node__title" x="' + cx + '" y="' + (node.y + 27) +
      '" text-anchor="middle">' + esc(node.title) + "</text>" +
      '<text class="lineage-node__sub" x="' + cx + '" y="' + (node.y + 45) +
      '" text-anchor="middle">' + esc(node.sub) + "</text></g>"
    );
  }

  function edgeMarkup(edge) {
    return (
      '<g class="lineage-edge" data-edge="' + edge.from + ">" + edge.to + '">' +
      '<path d="' + edge.d + '" fill="none" marker-end="url(#lineage-arrow)"/></g>'
    );
  }

  function buildSvg() {
    var body = "";
    body += "<defs>" +
      '<marker id="lineage-arrow" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" ' +
      'markerWidth="8" markerHeight="8" refX="7.2" refY="4" orient="auto">' +
      '<path d="M0 0L8 4L0 8Z" fill="context-stroke"/></marker></defs>';
    body += EDGES.map(edgeMarkup).join("");
    body += NODES.map(nodeMarkup).join("");
    /* legend */
    body += '<g class="lineage-legend" aria-hidden="true">';
    var legend = [
      ["compute", "计算路径"],
      ["dispatch", "硬件 / 调度"],
      ["communication", "流水线 / 通信"],
      ["remote", "归约 / 收口"],
      ["orange", "资源分配"]
    ];
    var lx = 60;
    legend.forEach(function (item) {
      body += '<rect class="lineage-legend__swatch lineage-node--' + item[0] +
        '" x="' + lx + '" y="606" width="11" height="11" rx="2.5"/>' +
        '<text class="lineage-legend__label" x="' + (lx + 17) + '" y="615">' +
        esc(item[1]) + "</text>";
      lx += item[1].length * 11 + 66;
    });
    body += '<text class="lineage-legend__hint" x="1060" y="615" text-anchor="end">悬停预览依赖链 · 点击固定并查看解读</text>';
    body += "</g>";
    return (
      '<svg viewBox="0 0 1100 636" role="group" aria-label="优化点依赖脉络图" ' +
      'font-family="JetBrains Mono" xmlns="http://www.w3.org/2000/svg">' + body + "</svg>"
    );
  }

  function renderDetail(panel, node) {
    if (!node) {
      panel.innerHTML =
        '<span class="lineage-detail__label">Optimization Lineage · 怎么读</span>' +
        "<h3>点击任意优化点</h3>" +
        "<p>左图是整门课的因果骨架：每个节点是一个优化点，每条箭头读作「因为有了 A，B 才成立 / 才必要」。</p>" +
        "<p>悬停可以预览一个优化点的完整上下游链条；点击固定后，这里会展开它的解读、依赖关系与所属章节入口。</p>" +
        '<p class="lineage-detail__tip">推荐路线：从 <strong>AttnSlice 切片表示</strong> 出发走到 <strong>多阶段 Overlap 流水线</strong>，正好穿过全部 9 章。</p>';
      return;
    }
    var ups = upOf[node.id].map(function (edge) {
      return "<li><strong>" + esc(byId[edge.from].title) + "</strong><span>" +
        esc(edge.label) + "</span></li>";
    }).join("");
    var downs = downOf[node.id].map(function (edge) {
      return "<li><strong>" + esc(byId[edge.to].title) + "</strong><span>" +
        esc(edge.label) + "</span></li>";
    }).join("");
    panel.innerHTML =
      '<span class="lineage-detail__label">' + esc(CHAPTER_NAMES[node.chapter] || "") + "</span>" +
      "<h3>" + esc(node.title) + "</h3>" +
      "<p>" + esc(node.detail) + "</p>" +
      (ups ? '<h4>站在谁的肩上</h4><ul class="lineage-detail__list">' + ups + "</ul>" : "") +
      (downs ? '<h4>成就了什么</h4><ul class="lineage-detail__list">' + downs + "</ul>" : "") +
      '<a class="button primary lineage-detail__go" href="chapter.html?id=' +
      esc(node.chapter) + '">进入' + esc(CHAPTER_NAMES[node.chapter] || "章节") + " ↗</a>";
  }

  function init() {
    var host = document.getElementById("lineage-map");
    var panel = document.getElementById("lineage-detail");
    if (!host || !panel) return;
    host.innerHTML = buildSvg();
    renderDetail(panel, null);

    var svg = host.querySelector("svg");
    var nodeEls = {};
    host.querySelectorAll(".lineage-node").forEach(function (el) {
      nodeEls[el.getAttribute("data-node")] = el;
    });
    var edgeEls = {};
    host.querySelectorAll(".lineage-edge").forEach(function (el) {
      edgeEls[el.getAttribute("data-edge")] = el;
    });

    var pinned = null;

    function applyChain(id) {
      if (!id) {
        svg.classList.remove("has-focus");
        Object.keys(nodeEls).forEach(function (key) {
          nodeEls[key].classList.remove("is-chain", "is-focus");
        });
        Object.keys(edgeEls).forEach(function (key) {
          edgeEls[key].classList.remove("is-chain");
        });
        return;
      }
      var chain = collectChain(id);
      svg.classList.add("has-focus");
      Object.keys(nodeEls).forEach(function (key) {
        nodeEls[key].classList.toggle("is-chain", !!chain.nodes[key]);
        nodeEls[key].classList.toggle("is-focus", key === id);
      });
      Object.keys(edgeEls).forEach(function (key) {
        edgeEls[key].classList.toggle("is-chain", !!chain.edges[key]);
      });
    }

    function pin(id) {
      pinned = id;
      applyChain(id);
      renderDetail(panel, id ? byId[id] : null);
    }

    Object.keys(nodeEls).forEach(function (id) {
      var el = nodeEls[id];
      el.addEventListener("mouseenter", function () {
        if (!pinned) applyChain(id);
      });
      el.addEventListener("mouseleave", function () {
        if (!pinned) applyChain(null);
      });
      el.addEventListener("click", function (event) {
        event.stopPropagation();
        pin(pinned === id ? null : id);
      });
      el.addEventListener("keydown", function (event) {
        if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") return;
        event.preventDefault();
        pin(pinned === id ? null : id);
      });
    });

    svg.addEventListener("click", function () {
      if (pinned) pin(null);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && pinned) pin(null);
    });
  }

  window.addEventListener("DOMContentLoaded", init);
})();
