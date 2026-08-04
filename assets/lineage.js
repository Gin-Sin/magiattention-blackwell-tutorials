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
      id: "attnslice", x: 420, y: 46, w: 260, tone: "compute", no: "00",
      title: "AttnSlice 契约", sub: "切片语义 · 可合并输出",
      chapter: "attnslice",
      detail: "AttnSlice（注意力切片）把任意 mask 拆成 (QRange, KRange, MaskType) 清单，切片内只保留 FULL、CAUSAL、INV-CAUSAL、BI-CAUSAL 四种几何。局部 out 与 LSE（对数归一项）可按固定公式合并，让 kernel 块间归约和跨 rank 汇合共享同一结果契约；不同浮点顺序可能带来微小舍入差异。"
    },

    /* L1 · 硬件基座 */
    {
      id: "blackwell", x: 420, y: 164, w: 260, tone: "dispatch", no: "01",
      title: "Blackwell 基座", sub: "TMEM 常驻 · 双引擎异步",
      chapter: "blackwell",
      detail: "TMEM（Tensor Memory）专门承载矩阵乘累加结果，S、P、O 通过 T2R/R2T 与寄存器交换，减少对普通寄存器的占用。TMA 负责搬运，tcgen05 UMMA 负责矩阵乘；warp 异步发射后，由 mbarrier（硬件屏障）报告完成。"
    },

    /* L2 · kernel 执行 */
    {
      id: "mask", x: 40, y: 282, w: 230, tone: "compute", no: "03",
      title: "三层 Mask 防线", sub: "跳块 · 快路 · 边界修正",
      chapter: "mask",
      detail: "BlockInfo（块元数据）先跳过全无效块，主循环让全有效块直接走快路径。只有边界穿过的 partial 块才使用 R2P，把无效分数写成 -inf。"
    },
    {
      id: "pipe", x: 310, y: 282, w: 230, tone: "communication", no: "02",
      title: "Warp 特化流水线", sub: "角色分工 · 槽位状态协调",
      chapter: "pipeline",
      detail: "16 个 warp 按 load、MMA、softmax、correction、epilogue 分工，六条 mbarrier 流水线管理槽位的满与空。warp 之间只翻转状态，不搬运数据；数据始终留在约定的 TMEM 或 SMEM（共享内存）中。"
    },
    {
      id: "softmax", x: 580, y: 282, w: 230, tone: "compute", no: "04",
      title: "在线 Softmax", sub: "TMEM 驻留 · 递推早放行",
      chapter: "softmax",
      detail: "Softmax 换为以 2 为底后，内层只剩 packed FMA（融合乘加）与 ex2。row_max 更新后立即发布 corr_scale（校正缩放），P 写到 3/4 就提前放行 PV GEMM，让递推与矩阵乘重叠。"
    },
    {
      id: "corr", x: 850, y: 282, w: 210, tone: "remote", no: "05",
      title: "Correction", sub: "块间校正 · 写出 out/LSE",
      chapter: "correction",
      detail: "主循环用 corr_scale 把旧 O 校正到新的最大值基准，尾声再除以行和 ℓ，并写出 fp32 out/LSE。块间与 rank 间沿用同一合并公式，Correction 因而成为 kernel 到分布式归约的收口。"
    },

    /* L3 · 调度与反向 */
    {
      id: "sched", x: 180, y: 400, w: 260, tone: "dispatch", no: "06",
      title: "持久化 Tile 调度", sub: "LPT · L2 亲和 · CLC 分配",
      chapter: "scheduler",
      detail: "Causal 负载沿 Q tile 线性递增：LPT（最长任务优先）让重块先行，L2 swizzle 保持同一 head 的 K/V 亲和，CLC 把分配交回硬件。Persistent kernel（持久化内核）让每个 CTA 连续消费多个 tile；CTA 数量可调，才可用 sm_margin 给通信让出 SM。"
    },
    {
      id: "bwd", x: 650, y: 400, w: 260, tone: "compute", no: "07",
      title: "反向 5-GEMM", sub: "K 为家 · dQ 原子归约",
      chapter: "backward",
      detail: "固定 K/V 扫过所有 Q：dK/dV 本地累加，dQ 用 TMA bulk atomic-add（批量原子加）全局记账。P 不保存也不传输，而由 LSE 重算；分布式反向还需把 dK/dV 送回 KV 属主。"
    },

    /* L4 · 分布式 */
    {
      id: "overlap", x: 420, y: 518, w: 260, tone: "communication", no: "08",
      title: "通算融合", sub: "让出 SM · 分阶段重叠",
      chapter: "overlap",
      detail: "异步入队不等于同时运行：NCCL 路径用 sm_margin 留出 SM，native grpcoll 以常驻通信 SM 配合 KernelBarrier 协调发射。GroupCast/GroupReduce（组播/组归约）按依赖清单精确投递，并用 out/LSE 公式合并结果。远端 KV 再按成本模型切成多个 stage，使预取、计算与归约形成重叠；收益仍取决于队列和 SM 资源。"
    }
  ];

  var EDGES = [
    { from: "attnslice", to: "mask", d: "M470 104V136H155V282", label: "切片几何限定 Mask" },
    { from: "attnslice", to: "corr", d: "M630 104V136H955V282", label: "可合并约定要求收口" },
    { from: "attnslice", to: "overlap", d: "M420 75H20V547H420", label: "切片清单限定通信" },
    { from: "blackwell", to: "pipe", d: "M425 222V282", label: "异步引擎需角色驾驶" },
    { from: "pipe", to: "softmax", d: "M540 311H580", label: "槽位协议供数" },
    { from: "pipe", to: "corr", d: "M470 282V262H900V282", label: "缩放握手驱动校正" },
    { from: "mask", to: "sched", d: "M155 340V370H255V400", label: "负载差异需要 LPT" },
    { from: "softmax", to: "corr", d: "M810 311H850", label: "递推结果归约收口" },
    { from: "softmax", to: "bwd", d: "M695 340V400", label: "LSE 支撑反向重算" },
    { from: "corr", to: "overlap", d: "M955 340V478H660V518", label: "可合并结果进入归约" },
    { from: "sched", to: "overlap", d: "M310 458V498H540V518", label: "CTA 可调才能让出 SM" },
    { from: "bwd", to: "overlap", d: "M780 458V488H600V518", label: "dK/dV 需送回属主" }
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
      ["remote", "归约 / 收口"]
    ];
    var lx = 60;
    legend.forEach(function (item) {
      body += '<rect class="lineage-legend__swatch lineage-node--' + item[0] +
        '" x="' + lx + '" y="606" width="11" height="11" rx="2.5"/>' +
        '<text class="lineage-legend__label" x="' + (lx + 17) + '" y="615">' +
        esc(item[1]) + "</text>";
      lx += item[1].length * 11 + 66;
    });
    body += '<text class="lineage-legend__hint" x="1060" y="615" text-anchor="end">悬停预览依赖链 · 点击固定解读 · 双击进入章节</text>';
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
        "<h3>先看因果，再进章节</h3>" +
        "<p>每个节点对应一章；每条箭头都读作「因为有 A，B 才成立或才有必要」。</p>" +
        "<p>悬停节点可预览完整上下游；点击固定依赖链并在这里查看解读；双击节点直接进入对应章节。</p>" +
        '<p class="lineage-detail__tip">首次阅读按章节顺序：从 <strong>AttnSlice 契约</strong> 一路走到 <strong>通算融合</strong>，正好穿过全部 9 章。</p>';
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
      el.addEventListener("dblclick", function (event) {
        event.preventDefault();
        event.stopPropagation();
        window.location.href = "chapter.html?id=" + byId[id].chapter;
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
