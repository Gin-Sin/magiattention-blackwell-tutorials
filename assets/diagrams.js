/* FFA on Blackwell — SVG diagram builders.
 *
 * Every main diagram is built from axis-aligned boxes and orthogonal
 * connectors, then passed through a static geometry validator that rejects
 * connectors crossing node interiors, overlapping rails, unlabelled dashed
 * edges, and code-block ids missing from window.MAGI_CODE.
 */
(function () {
  "use strict";

  var P = {
    canvas: "#fcfcfb",
    paper: "#ffffff",
    ink: "#2e3338",
    muted: "#5f676d",
    rule: "#d9dade",
    compute: "#dbe5f6",
    computeStroke: "#7189b8",
    control: "#dfebd6",
    controlStroke: "#7ba36c",
    state: "#f6dcd7",
    stateStroke: "#c07a6f",
    gather: "#e0daf4",
    gatherStroke: "#8b7cc2",
    cyan: "#dceaf8",
    cyanStroke: "#2b6cb8",
    orange: "#fbe8cd",
    orangeStroke: "#c78f4a"
  };

  function toneFill(tone) {
    return P[tone] || P.paper;
  }

  function toneStroke(tone) {
    return P[tone + "Stroke"] || P.muted;
  }

  function escapeText(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function M(tex, fallback) {
    return { tex: tex, fallback: fallback || tex };
  }

  function fallbackLabel(value) {
    if (value && typeof value === "object" && value.fallback) return value.fallback;
    return value == null ? "" : String(value);
  }

  function textLabel(x, y, value, fontSize, color, weight) {
    return '<text x="' + x + '" y="' + (y + fontSize * 0.34) +
      '" text-anchor="middle" font-family="JetBrains Mono" font-size="' + fontSize +
      '" font-weight="' + (weight || 400) + '" fill="' + color + '">' +
      escapeText(value) + "</text>";
  }

  function mathLabel(x, y, width, height, value, fontSize, color, weight) {
    var fallback = escapeText(value.fallback);
    var tex = escapeText(value.tex);
    var left = x - width / 2;
    var top = y - height / 2;
    return (
      '<switch class="svg-math-switch" role="img" aria-label="' + fallback + '">' +
      '<foreignObject class="svg-math-label-wrap" x="' + left + '" y="' + top +
      '" width="' + width + '" height="' + height +
      '" requiredExtensions="http://www.w3.org/1999/xhtml">' +
      '<div xmlns="http://www.w3.org/1999/xhtml" class="svg-math-label" style="color:' + color +
      ";font-size:" + fontSize + "px;font-weight:" + (weight || 400) +
      '"><span aria-hidden="true">\\(' + tex + "\\)</span></div>" +
      "</foreignObject>" +
      '<text class="svg-math-fallback" x="' + x + '" y="' + (y + fontSize * 0.34) +
      '" text-anchor="middle" font-family="JetBrains Mono" font-size="' + fontSize +
      '" font-weight="' + (weight || 400) + '" fill="' + color + '">' + fallback +
      "</text></switch>"
    );
  }

  function labelMarkup(x, y, width, height, value, fontSize, color, weight) {
    if (value && typeof value === "object" && value.tex) {
      return mathLabel(x, y, width, height, value, fontSize, color, weight);
    }
    return textLabel(x, y, value, fontSize, color, weight);
  }

  function defs(rootId) {
    var tones = ["compute", "control", "state", "gather", "cyan", "orange", "muted"];
    return "<defs>" + tones.map(function (tone) {
      var color = tone === "muted" ? P.muted : toneStroke(tone);
      return '<marker id="' + rootId + "-arrow-" + tone +
        '" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" ' +
        'refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="' + color + '"/></marker>';
    }).join("") + "</defs>";
  }

  function panel(x, y, w, h, title, tone, dashed) {
    var stroke = toneStroke(tone);
    return (
      '<g class="diagram-panel"><rect x="' + x + '" y="' + y + '" width="' + w +
      '" height="' + h + '" rx="14" fill="' + toneFill(tone) +
      '" fill-opacity=".18" stroke="' + stroke + '" stroke-opacity=".55" stroke-width="1.2" ' +
      (dashed ? 'stroke-dasharray="7 6" ' : "") + "/>" +
      '<rect x="' + (x + 12) + '" y="' + (y - 9) + '" width="' +
      Math.max(124, title.length * 7.4 + 20) + '" height="20" fill="' + P.canvas + '"/>' +
      '<text x="' + (x + 20) + '" y="' + (y + 5) +
      '" font-family="JetBrains Mono" font-size="10.5" font-weight="600" fill="' +
      stroke + '">' + escapeText(title) + "</text></g>"
    );
  }

  /* Interactive (or plain, when codeBlockId is null) node box. */
  function box(x, y, w, h, title, sub, tone, n, codeBlockId, options) {
    options = options || {};
    if (x < 24 || y < 24 || x + w > 1076) {
      throw new Error("Diagram node violates margins: " + fallbackLabel(title));
    }
    var fill = toneFill(tone);
    var stroke = toneStroke(tone);
    var number = n != null
      ? '<circle cx="' + (x + 14) + '" cy="' + (y + 14) + '" r="9" fill="' + stroke + '"/>' +
        textLabel(x + 14, y + 14, n, 8.5, P.paper, 700)
      : "";
    var titleY = y + h / 2 - (sub ? 6 : 0);
    var inner =
      '<rect class="diagram-node-box" x="' + x + '" y="' + y + '" width="' + w +
      '" height="' + h + '" rx="9" fill="' + fill + '" stroke="' + stroke +
      '" stroke-width="1.35" ' + (options.dashed ? 'stroke-dasharray="6 5" ' : "") + "/>" +
      number +
      labelMarkup(x + w / 2, titleY, w - 22, 34, title, options.titleSize || 11.5, P.ink, 600) +
      (sub
        ? labelMarkup(x + w / 2, y + h / 2 + 15, w - 20, 26, sub,
            options.subSize || 8.8, P.muted, 500)
        : "");
    if (codeBlockId == null) {
      return '<g class="diagram-plain-node">' + inner + "</g>";
    }
    var aria = escapeText("查看代码块 " + codeBlockId + "：" + fallbackLabel(title));
    return (
      '<g class="diagram-code-node" data-code-block="' + escapeText(codeBlockId) +
      '" role="button" tabindex="0" aria-label="' + aria + '">' + inner + "</g>"
    );
  }

  function ortho(x1, y1, x2, y2, axis, turn) {
    if (x1 === x2) return "M" + x1 + " " + y1 + "V" + y2;
    if (y1 === y2) return "M" + x1 + " " + y1 + "H" + x2;
    if (axis === "y") {
      var bendY = turn == null ? (y1 + y2) / 2 : turn;
      return "M" + x1 + " " + y1 + "V" + bendY + "H" + x2 + "V" + y2;
    }
    var bendX = turn == null ? (x1 + x2) / 2 : turn;
    return "M" + x1 + " " + y1 + "H" + bendX + "V" + y2 + "H" + x2;
  }

  function edge(rootId, d, label, tone, dashed) {
    tone = tone || "muted";
    if (dashed && !label) {
      throw new Error("Dashed diagram edge requires an explicit label");
    }
    var color = tone === "muted" ? P.muted : toneStroke(tone);
    return (
      '<g><path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.5" ' +
      (dashed ? 'stroke-dasharray="6 5" ' : "") +
      'stroke-linecap="square" stroke-linejoin="round" marker-end="url(#' +
      rootId + "-arrow-" + tone + ')"/>' +
      (label
        ? labelMarkup(label[0], label[1], label[3] || 190, 24, label[2], label[4] || 8.8, color, 600)
        : "") +
      "</g>"
    );
  }

  function baseSvg(rootId, diagramKey, height, body, label) {
    return (
      '<svg viewBox="0 0 1100 ' + height + '" role="img" aria-label="' + escapeText(label) +
      '" data-diagram-key="' + escapeText(diagramKey) +
      '" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono">' +
      defs(rootId) +
      '<rect width="1100" height="' + height + '" fill="' + P.canvas + '"/>' +
      body +
      "</svg>"
    );
  }

  /* ---------------- static geometry validator ---------------- */

  function validateStaticGeometry(svg, diagramKey) {
    function attributes(tag) {
      var result = {};
      var match;
      var pattern = /([\w:-]+)="([^"]*)"/g;
      while ((match = pattern.exec(tag))) result[match[1]] = match[2];
      return result;
    }

    function pathPoints(d) {
      var points = [];
      var x = 0;
      var y = 0;
      var match;
      var commands = /([MHV])\s*(-?[\d.]+)(?:\s+(-?[\d.]+))?/g;
      while ((match = commands.exec(d))) {
        if (match[1] === "M") {
          x = Number(match[2]);
          y = Number(match[3]);
        } else if (match[1] === "H") {
          x = Number(match[2]);
        } else {
          y = Number(match[2]);
        }
        points.push({ x: x, y: y });
      }
      return points;
    }

    function crossesInterior(a, z, node) {
      var epsilon = 0.75;
      if (a.x === z.x) {
        return a.x > node.x + epsilon && a.x < node.x + node.w - epsilon &&
          Math.max(Math.min(a.y, z.y), node.y + epsilon) <
          Math.min(Math.max(a.y, z.y), node.y + node.h - epsilon);
      }
      if (a.y === z.y) {
        return a.y > node.y + epsilon && a.y < node.y + node.h - epsilon &&
          Math.max(Math.min(a.x, z.x), node.x + epsilon) <
          Math.min(Math.max(a.x, z.x), node.x + node.w - epsilon);
      }
      throw new Error(diagramKey + ": connector is not orthogonal");
    }

    var boxes = [];
    var boxMatch;
    var boxPattern = /<rect class="diagram-node-box"[^>]*>/g;
    while ((boxMatch = boxPattern.exec(svg))) {
      var boxAttrs = attributes(boxMatch[0]);
      boxes.push({
        x: Number(boxAttrs.x),
        y: Number(boxAttrs.y),
        w: Number(boxAttrs.width),
        h: Number(boxAttrs.height)
      });
    }

    var edges = [];
    var edgeMatch;
    var edgePattern = /<path d="([^"]+)"[^>]*marker-end/g;
    while ((edgeMatch = edgePattern.exec(svg))) {
      edges.push({ d: edgeMatch[1], points: pathPoints(edgeMatch[1]) });
    }

    edges.forEach(function (item) {
      for (var i = 1; i < item.points.length; i += 1) {
        for (var j = 0; j < boxes.length; j += 1) {
          if (crossesInterior(item.points[i - 1], item.points[i], boxes[j])) {
            throw new Error(diagramKey + ": connector traverses node: " + item.d);
          }
        }
      }
    });

    var RAIL_GAP = 3.5;
    var OVERLAP_LIMIT = 6;
    function sharedSpan(a1, a2, b1, b2) {
      return Math.min(Math.max(a1, a2), Math.max(b1, b2)) -
        Math.max(Math.min(a1, a2), Math.min(b1, b2));
    }
    for (var ei = 0; ei < edges.length; ei += 1) {
      for (var ej = ei + 1; ej < edges.length; ej += 1) {
        var A = edges[ei].points;
        var B = edges[ej].points;
        var sharedSource = A[0].x === B[0].x && A[0].y === B[0].y;
        for (var si = 1; si < A.length; si += 1) {
          for (var sj = 1; sj < B.length; sj += 1) {
            if (sharedSource && si === 1 && sj === 1) continue;
            var a0 = A[si - 1];
            var a1 = A[si];
            var b0 = B[sj - 1];
            var b1 = B[sj];
            var overlap = -1;
            if (a0.x === a1.x && b0.x === b1.x && Math.abs(a0.x - b0.x) <= RAIL_GAP) {
              overlap = sharedSpan(a0.y, a1.y, b0.y, b1.y);
            } else if (a0.y === a1.y && b0.y === b1.y && Math.abs(a0.y - b0.y) <= RAIL_GAP) {
              overlap = sharedSpan(a0.x, a1.x, b0.x, b1.x);
            }
            if (overlap > OVERLAP_LIMIT) {
              throw new Error(diagramKey + ": overlapping rails between edges: " +
                edges[ei].d + " | " + edges[ej].d);
            }
          }
        }
      }
    }

    var dashedMatch;
    var dashedPattern = /<g><path\b[^>]*stroke-dasharray="[^"]+"[^>]*\/>([\s\S]*?)<\/g>/g;
    while ((dashedMatch = dashedPattern.exec(svg))) {
      if (!/<(?:text|switch)\b/.test(dashedMatch[1])) {
        throw new Error(diagramKey + ": dashed connector lacks a label");
      }
    }

    var impl = window.MAGI_CODE && window.MAGI_CODE[diagramKey];
    if (impl) {
      var validIds = {};
      impl.blocks.forEach(function (block) { validIds[block.id] = true; });
      var idMatch;
      var idPattern = /data-code-block="([^"]+)"/g;
      while ((idMatch = idPattern.exec(svg))) {
        if (!validIds[String(idMatch[1]).padStart(2, "0")]) {
          throw new Error(diagramKey + ": invalid code block " + idMatch[1]);
        }
      }
    }
  }

  /* ================================================================ *
   * Main diagrams
   * ================================================================ */

  function attnsliceDiagram(rootId) {
    var b = "";
    b += panel(40, 52, 1020, 128, "INPUTS · 调用方视角", "compute");
    b += box(70, 84, 250, 68, "Q / K / V", M("[B,S,H,d]\\ \\text{or packed}", "[B,S,H,d] or packed"), "compute", 1, "01");
    b += box(390, 84, 280, 68, "q_ranges / k_ranges", "[N,2] int32 · [start,end)", "control", 2, "01");
    b += box(740, 84, 280, 68, "mask_type · sink · softcap", "MT_MAP: full=0 causal=1", "control", 3, "01");

    b += box(180, 236, 330, 66, "ranges_to_cu_seqlens", "Step-1: 连续不重叠 → cu_seqlens", "gather", 4, "02");
    b += box(620, 236, 330, 66, "out / lse 分配", M("\\text{lse} \\equiv \\text{fp32},\\ \\text{init} -\\infty", "lse=fp32, init -inf"), "state", 5, "03");

    b += box(180, 366, 590, 66, "启发式决策", "tile 128×128 · q_stage · 2-CTA · CLC 开关", "compute", 6, "04");
    b += box(830, 366, 190, 66, "空输入短路", "out=0, lse=-inf", "state", null, "03", { dashed: true });

    b += box(180, 496, 330, 64, "compile_key 元组", "全部静态分支进 key", "cyan", 7, "05");
    b += box(620, 496, 330, 64, "FFAFwdSm100(...)", "cache miss 时实例化", "compute", 8, "06");

    b += box(180, 626, 590, 66, "cute.compile --enable-tvm-ffi", "编译一次 · 缓存 · 低开销启动", "gather", 9, "07");
    b += box(830, 626, 190, 66, "out, lse 返回", "供 GroupReduce 合并", "state", 10, "03");

    b += edge(rootId, ortho(345, 152, 345, 236), null, "control");
    b += edge(rootId, "M195 152V200H230V236", null, "compute");
    b += edge(rootId, "M880 152V200H785V236", null, "control");
    b += edge(rootId, ortho(345, 302, 345, 366), null, "gather");
    b += edge(rootId, "M785 302V334H475V366", null, "state");
    b += edge(rootId, ortho(345, 432, 345, 496), null, "compute");
    b += edge(rootId, "M510 528H620", null, "cyan");
    b += edge(rootId, "M785 560V593H475V626", null, "compute");
    b += edge(rootId, ortho(475, 692, 830, 692), null, "gather");
    return {
      svg: baseSvg(rootId, "attnslice", 730, b,
        "AttnSlice contract and host-side forward journey"),
      title: "AttnSlice 契约 · host 侧一次前向",
      badges: ["蓝 = 计算/实例化", "绿 = ranges 与 mask", "玫瑰 = 输出契约", "紫 = 折叠/启动", "青 = 编译缓存"],
      notes: [
        ["入口三元组", "q/k/v 张量 + [N,2] 的 q_ranges/k_ranges + 单一 mask_type;sink/softcap 是可选修饰。"],
        ["Step-1 折叠", "当前 CuTe DSL 路径要求 ranges 等价于 cu_seqlens 分段:从 0 起、连续、不重叠。"],
        ["输出契约", "lse 恒为 fp32、空集初值 -inf;functional 层在 ranges 可能重叠时把 out 升为 fp32 并 atomic 累加。"],
        ["启发式决策", "tile=128×128;q_stage 由序列长度定;2-CTA 要求非 causal/varlen/稀疏且 head_dim 合规;CLC 排除两类回退场景。"],
        ["编译缓存", "所有静态分支进 compile_key;首次 cute.compile,之后 tvm-ffi 直接启动,免去元数据转换。"]
      ],
      memory: "一句话:ranges 进、(out,lse) 出;中间全是编译期决策。"
    };
  }

  function blackwellDiagram(rootId) {
    var b = "";
    b += panel(40, 52, 500, 120, "GMEM", "orange");
    b += box(70, 84, 200, 64, "Q · K · V", "bf16 / fp16", "orange", 1, "05");
    b += box(310, 84, 200, 64, "O (输出)", "TMA S2G 写回", "orange", null, "05");

    b += panel(40, 232, 500, 150, "SMEM · 224KB 预算", "compute");
    b += box(70, 264, 200, 94, "sQ", "swizzled · q_stage 级", "compute", 2, "04");
    b += box(310, 264, 200, 94, "sK / sV 复用", "kv_stage 环形缓冲", "compute", 3, "04");

    b += panel(600, 52, 460, 330, "TMEM · 128行 × 512列 fp32", "state");
    b += box(630, 96, 190, 60, "S0 | S1", "[0,128) | [128,256)", "state", 6, "02");
    b += box(850, 96, 180, 60, "O0 | O1", "[256,384) | [384,512)", "state", 7, "02");
    b += box(630, 196, 190, 60, "P0 / P1 叠放", "bf16 视图 · S 空间复用", "state", 8, "06");
    b += box(850, 196, 180, 60, "vec 复用", "row_max/sum 暂存", "state", null, "02", { dashed: true });
    b += box(790, 296, 240, 60, "TMEM alloc / free", "仅 MMA warp", "gather", 9, "07");

    b += panel(40, 442, 1020, 130, "tcgen05 UMMA · 单线程发射 · 2-CTA 可选", "gather");
    b += box(90, 476, 400, 66, M("S = Q K^{\\mathsf T}", "S = Q K^T"), "tiled_mma_qk · A/B 来自 SMEM", "gather", 4, "03");
    b += box(590, 476, 400, 66, M("O \\mathrel{+}= P V", "O += P V"), "tiled_mma_pv · P 来自 TMEM", "gather", 5, "03");

    b += edge(rootId, ortho(170, 148, 170, 264), [218, 190, "TMA G2S · tx_count", 168], "orange");
    b += edge(rootId, "M240 148V200H410V264", null, "orange");
    b += edge(rootId, "M170 358V420H240V476", null, "compute");
    b += edge(rootId, "M340 358V476", null, "compute");
    b += edge(rootId, "M410 358V430H620V476", [520, 416, "sV → PV GEMM", 130], "compute");
    b += edge(rootId, "M490 509H540V126H630", [560, 300, "写 S 累加器", 120], "state");
    b += edge(rootId, "M660 256V476", [700, 400, "读 P (bf16)", 108], "gather");
    b += edge(rootId, "M940 256V296", null, "state");
    b += edge(rootId, "M940 156V196", null, "state");
    b += edge(rootId, "M990 476V420H1044V126H1030", null, "state");
    b += edge(rootId, "M900 96V40H410V84", [660, 28, "correction → sO → TMA S2G 写回", 250], "orange");
    return {
      svg: baseSvg(rootId, "blackwell", 600, b,
        "Blackwell execution substrate: TMA, SMEM, TMEM and tcgen05 UMMA"),
      title: "SM100 执行基座 · 数据只向前流",
      badges: ["橙 = GMEM/TMA", "蓝 = SMEM", "玫瑰 = TMEM", "紫 = UMMA"],
      notes: [
        ["三层存储", "GMEM --TMA--> SMEM --UMMA--> TMEM;累加器全程驻留 TMEM,不占寄存器堆。"],
        ["TMEM 地图", "S0/S1 占 [0,256),O0/O1 占 [256,512);bf16 的 P 叠放在 S 后半,row_max/sum 向量复用 S 空间。"],
        ["两个 GEMM", "QK 的 A/B 都在 SMEM;PV 的 A(即 P)声明 OperandSource.TMEM,免去 SMEM 中转。"],
        ["生命周期", "仅 MMA warp 分配/释放 TMEM;softmax+correction 通过 TmemPtr barrier 见证释放安全。"],
        ["2-CTA", "cluster 内两个 CTA 组队:MMA tiler M 翻倍到 256,K/V 的 SMEM 各存一半,kv_stage 翻倍。"]
      ],
      memory: "SMEM 喂数据、TMEM 存累加、UMMA 单 warp 发射——寄存器全部让给 softmax。"
    };
  }

  function pipelineDiagram(rootId) {
    var b = "";
    b += panel(40, 52, 300, 620, "WARP 角色 · 512 线程", "compute");
    b += box(70, 92, 240, 64, "load warp 14", "regs: other(48-80)", "compute", 1, "01");
    b += box(70, 192, 240, 64, "mma warp 12", "regs: other · 发 UMMA", "compute", 2, "01");
    b += box(70, 312, 240, 76, "softmax0/1 · warp 0-7", "regs ↑ 176-192", "compute", 3, "02");
    b += box(70, 442, 240, 76, "correction · warp 8-11", "regs 64-88", "compute", 4, "02");
    b += box(70, 582, 240, 64, "epilogue 13 · empty 15", "empty 可兼 CLC producer", "compute", null, "01");

    b += panel(420, 52, 640, 620, "PIPELINES · mbarrier 接力", "gather");
    b += box(470, 92, 250, 64, "pipeline_q / kv", "TMA→UMMA · tx 字节记账", "gather", 5, "03");
    b += box(470, 192, 250, 64, "pipeline_s_p_o", "S-full / P+O-empty 双义", "gather", 6, "04");
    b += box(780, 192, 240, 64, "p_lastsplit", "P 后半就绪 → GEMM 硬件", "gather", null, "04", { dashed: true });
    b += box(470, 312, 250, 76, "sScale RAW/WAR", "barrier + pipeline 双保险", "state", 7, "06");
    b += box(780, 312, 240, 76, "pipeline_o_acc", "仅尾块 · O-full 显式等待", "state", 8, "05");
    b += box(470, 442, 250, 76, "pipeline_o_epi", "correction → epilogue", "gather", null, "04", { dashed: true });
    b += box(780, 442, 240, 76, "cross-release", "单 correction 轮转两 softmax", "state", null, "06", { dashed: true });
    b += box(470, 582, 550, 64, "warp_idx 分派 · 一份代码七种人生", "setmaxregister ↓↑ 后进入各自 loop", "cyan", 9, "07");

    b += edge(rootId, "M310 124H470", [382, 110, "producer", 76, 8.2], "compute");
    b += edge(rootId, "M310 224H470", [382, 210, "S-full", 76, 8.2], "compute");
    b += edge(rootId, "M310 350H470", [382, 336, "P release", 76, 8.2], "compute");
    b += edge(rootId, "M310 480H470", [382, 466, "scale / O", 76, 8.2], "compute");
    b += edge(rootId, "M720 224H780", null, "gather");
    b += edge(rootId, "M595 256V312", null, "gather");
    b += edge(rootId, "M720 350H765V478H780", null, "state");
    b += edge(rootId, "M780 366H735V480H720", null, "state");
    b += edge(rootId, "M190 518V582", null, "compute");
    b += edge(rootId, "M595 518V582", null, "gather");
    return {
      svg: baseSvg(rootId, "pipeline", 700, b,
        "Warp specialization roles and mbarrier pipelines"),
      title: "16 个 warp · 六条流水线",
      badges: ["蓝 = warp 角色", "紫 = 流水线", "玫瑰 = sScale/O 同步", "青 = 分派"],
      notes: [
        ["角色即资源", "softmax 独享高寄存器配额;load/MMA/epilogue 是异步引擎的驾驶员,只留最低配额。"],
        ["双义槽位", "pipeline_s_p_o 一个槽位 track 两种转移:MMA commit S-full;softmax(P 写完)+correction(O 校准完)联合 release。"],
        ["sScale 协议", "RAW 用 named barrier(数据就绪),WAR 用 pipeline(槽位可覆写);correction 的 cross-release 让两组 softmax 错峰。"],
        ["尾块特例", "主循环靠 GEMM 顺序免等 O;最后一块该保证断裂,pipeline_o_acc 补上显式等待。"],
        ["统一骨架", "所有角色都是同一个 while work_tile 循环;warp_idx 分派 + 编译期 const_expr 让分支零开销。"]
      ],
      memory: "warp 之间不传数据,只翻转槽位状态;数据永远躺在约定好的 TMEM/SMEM 里。"
    };
  }

  function maskDiagram(rootId) {
    var b = "";
    b += box(70, 60, 430, 64, "Q tile (m_block) + SeqlenInfoQK", "相对坐标系 · cu_seqlens 读一次", "compute", 1, "07");
    b += box(600, 60, 420, 64, "端对齐几何", M("k \\le q + (s_k - s_q)", "k <= q + (sk - sq)"), "cyan", null, "01");

    b += panel(40, 176, 1020, 128, "第一层 · BlockInfo 跳块(免费)", "control");
    b += box(90, 210, 400, 66, "get_n_block_min_max", "整块非法的 n_block 不迭代", "control", 2, "01");
    b += box(560, 210, 440, 66, "get_n_block_min_causal_local_mask", "定位 partial 带的边界", "control", 3, "01");

    b += panel(40, 356, 1020, 128, "第二层 · 三段主循环(近免费)", "compute");
    b += box(90, 390, 280, 66, "Mainloop-1 · partial", "带 mask_fn · 从右往左", "orange", 4, "02");
    b += box(410, 390, 280, 66, "Mainloop-2 · full", "不传 mask_fn · 零 mask 代码", "control", 5, "02");
    b += box(730, 390, 280, 66, "Mainloop-3 · partial", "仅 local 左窗带", "orange", null, "02");

    b += panel(40, 536, 1020, 128, "第三层 · 元素级写 -inf(边界块专属)", "state");
    b += box(90, 570, 280, 66, "apply_mask_sm100", "col_limit 由行号线性给出", "state", 6, "03");
    b += box(410, 570, 280, 66, "R2P 位掩码", "32 列一条 uint32", "state", 7, "05");
    b += box(730, 570, 280, 66, "mask_mod / 双界", "可编程谓词 · below&above", "state", 8, "06");

    b += edge(rootId, ortho(285, 124, 285, 210), null, "compute");
    b += edge(rootId, "M810 124V162H780V210", null, "cyan");
    b += edge(rootId, ortho(290, 276, 230, 390), "", "control");
    b += edge(rootId, "M780 276V330H550V390", null, "control");
    b += edge(rootId, ortho(230, 456, 230, 570), null, "orange");
    b += edge(rootId, "M370 603H410", null, "state");
    b += edge(rootId, "M690 603H730", null, "state");
    b += edge(rootId, "M870 456V570", [960, 500, "仅 partial 块付钱", 150], "orange");
    return {
      svg: baseSvg(rootId, "mask", 700, b,
        "Three-layer masking: block skipping, segmented loop, element predicates"),
      title: "块级 Mask · 三层防线",
      badges: ["绿 = 跳块/full", "橙 = partial 带", "玫瑰 = 元素级", "青 = 几何"],
      notes: [
        ["一条不等式", "端对齐 causal 的列界是行号的线性函数,同一几何驱动三层机制。"],
        ["第一层免费", "BlockInfo 用 tile 角点代入不等式,整块非法的 KV block 循环压根不进。"],
        ["第二层近免费", "三段循环把 partial 块隔离在对角带;full 段的 softmax_step 编译期就没有 mask 代码。"],
        ["第三层高效", "R2P 把 32 列的 keep/drop 打进一个 uint32,一次散到 32 个谓词寄存器;local 双界取 AND。"],
        ["可编程出口", "超出 causal/local 表达力时走 mask_mod 谓词,配合 CSR block-sparse 表限定范围。"]
      ],
      memory: "先跳块,再挑块,最后才逐位改元素——mask 的钱只花在对角带上。"
    };
  }

  function softmaxDiagram(rootId) {
    var b = "";
    b += box(70, 60, 300, 64, "wait S-full · T2R", "128 fp32 / 线程进寄存器", "compute", 1, "01");
    b += box(70, 176, 300, 64, "score_mod / mask_fn", "softcap 也从此进", "control", 2, "02");
    b += box(70, 292, 300, 72, "update_row_max", M("\\text{corr\\_scale}=2^{(m_{old}-m_{new})c}", "corr_scale"), "compute", 3, "03");
    b += box(70, 428, 300, 64, "corr_scale → sScale", "barrier arrive · 通知 correction", "state", 4, "03");

    b += box(470, 60, 300, 64, "scale_subtract_rowmax", M("s\\cdot c + (\\text{off} - mc)", "packed FMA"), "compute", 5, "04");
    b += box(470, 176, 300, 72, "exp2 · 硬件/仿真混合", "ex2_emu_freq 定节奏", "gather", 6, "04");
    b += box(470, 292, 300, 72, "R2T 写 P (bf16)", "split_P_arrive 提前放行", "gather", 7, "04");
    b += box(470, 428, 300, 64, "WAR acquire + row_sum", "等待窗口被计算填满", "compute", 8, "05");

    b += panel(850, 44, 210, 464, "支撑", "orange");
    b += box(870, 92, 170, 76, "调参表", "(2cta,causal,hd,sm103)", "orange", 9, "06");
    b += box(870, 216, 170, 76, "update_row_max 源码", "rescale_threshold 门", "orange", 10, "07");
    b += box(870, 340, 170, 76, "ex2_emulation_2", "多项式 + packed FMA", "orange", 11, "08");

    b += edge(rootId, ortho(220, 124, 220, 176), null, "compute");
    b += edge(rootId, ortho(220, 240, 220, 292), null, "control");
    b += edge(rootId, ortho(220, 364, 220, 428), null, "compute");
    b += edge(rootId, "M370 328H420V92H470", [420, 40, "减 max 后进入指数段", 190], "compute");
    b += edge(rootId, ortho(620, 124, 620, 176), null, "compute");
    b += edge(rootId, ortho(620, 248, 620, 292), null, "gather");
    b += edge(rootId, ortho(620, 364, 620, 428), null, "gather");
    b += edge(rootId, "M870 130H820V212H770", null, "orange");
    b += edge(rootId, "M870 378H795V248H770", null, "orange");
    return {
      svg: baseSvg(rootId, "softmax", 560, b,
        "softmax_step data flow on SM100"),
      title: "softmax_step · 一个 KV block 的九个动作",
      badges: ["蓝 = 计算", "绿 = 挂载点", "紫 = 指数与写回", "玫瑰 = 发布", "橙 = 支撑机制"],
      notes: [
        ["早发布", "corr_scale 在 row_max 更新后立即写 sScale 并 arrive——correction 的 O-rescale 与本 warp 的 exp2 全程并行。"],
        ["换底", "所有指数以 2 为底:scale_log2 = softmax_scale · log2(e),内层只剩 packed FMA 和 ex2。"],
        ["双管线指数", "SFU 吞吐不够时按 ex2_emu_freq 把部分 exp2 换成 FMA 管线的多项式仿真;SM103 SFU 快,全部归零。"],
        ["提前放行", "P 写到 3/4 就 release,PV GEMM 先启动,读尾部前等 p_lastsplit 信号。"],
        ["错峰收尾", "WAR acquire 之后才更新 row_sum:把无依赖的寄存器计算塞进可能阻塞的等待窗口。"]
      ],
      memory: "减 max、exp2、写 P 是明线;corr_scale 的早发布和 row_sum 的晚更新是暗线。"
    };
  }

  function correctionDiagram(rootId) {
    var b = "";
    b += box(390, 56, 320, 64, "sScale · 每行两槽", "corr_scale | row_sum/max", "state", 1, "01");

    b += panel(40, 176, 490, 400, "主循环 · 每个 KV block", "compute");
    b += box(80, 216, 410, 64, "barrier wait → 读 corr_scale", "RAW 会合", "compute", 2, "02");
    b += box(80, 320, 410, 64, "ballot(corr_scale < 1)?", "全票 ≈1 则整块跳过", "control", 3, "02");
    b += box(80, 424, 410, 64, "correction_rescale", "T2R → ×scale → R2T · 16列/批", "gather", 4, "03");
    b += box(80, 494, 410, 0 + 64, "release s_p_o(O侧) + cross-release", "放行 MMA · 轮转 softmax", "state", null, "02", { dashed: true });

    b += panel(570, 176, 490, 400, "尾声 · 每个 Q tile 一次", "orange");
    b += box(610, 216, 410, 64, "读 row_sum/row_max → sink 并入", M("\\ell \\mathrel{+}= 2^{s\\log_2 e - mc}", "sink into rowsum"), "orange", 5, "04");
    b += box(610, 320, 410, 64, "rcp(row_sum) · 空行兜底", "wait pipeline_o_acc", "compute", 6, "04");
    b += box(610, 424, 410, 64, "correction_epilogue → sO", "×1/ℓ · 转 dtype · TMA/S2G", "gather", 7, "05");
    b += box(610, 494, 410, 64, "LSE 写出", M("(mc+\\log_2\\ell-\\text{off})\\ln 2", "LSE formula"), "cyan", 8, "06");

    b += edge(rootId, "M470 120V168H285V216", [352, 148, "主循环槽", 100], "state");
    b += edge(rootId, "M630 120V168H815V216", [732, 148, "尾声槽", 90], "state");
    b += edge(rootId, ortho(285, 280, 285, 320), null, "compute");
    b += edge(rootId, ortho(285, 384, 285, 424), null, "control");
    b += edge(rootId, ortho(815, 280, 815, 320), null, "orange");
    b += edge(rootId, ortho(815, 384, 815, 424), null, "compute");
    b += edge(rootId, ortho(815, 488, 815, 494), null, "gather");
    return {
      svg: baseSvg(rootId, "correction", 620, b,
        "Correction main loop and epilogue"),
      title: "Correction · 还账与清算",
      badges: ["蓝 = 计算", "绿 = 门控", "紫 = TMEM 读改写", "玫瑰 = 同步", "橙 = 尾声", "青 = LSE"],
      notes: [
        ["一个槽位两种货", "sScale 主循环装 corr_scale,尾声装 row_sum/row_max(偏移 q_stage×128)。"],
        ["ballot 门控", "warp 整体投票,全票 ≈1 就跳过整块 rescale——与 softmax 侧 rescale_threshold 成对。"],
        ["免等 O 的推理", "scale 可读 ⇒ S(i) GEMM 已完 ⇒ 同 warp 上 O(i-1) GEMM 更早完成;唯尾块需显式等 o_acc。"],
        ["cross-release", "释放对面 stage 的槽位,单组 correction 圆舞曲式轮转服务两组 softmax。"],
        ["数值收口", "空行 scale 兜底 1、sink 只进分母、FP8 的 descale/max_offset 一并在 rcp 处回补。"]
      ],
      memory: "主循环还 corr_scale 的债,尾声用 1/ℓ 清算;LSE 是写给下一次合并的收据。"
    };
  }

  function schedulerDiagram(rootId) {
    var b = "";
    b += box(300, 56, 500, 64, "调度器选型(编译期)", "varlen → LPT → persistent → single", "cyan", 1, "01");
    b += box(70, 56, 190, 64, "host 回退启发式", "varlen-MHA/dense 关 CLC", "orange", 2, "07");

    b += panel(40, 176, 500, 330, "STATIC 路径 · 软件排序", "compute");
    b += box(80, 216, 420, 70, "L2 swizzle 参数", "50MB / head 体积 → 2 的幂", "compute", 3, "02");
    b += box(80, 330, 420, 70, "LPT 坐标映射", "divmod 链 + block 反转", "compute", 4, "03");
    b += box(80, 424, 420, 60, "get_current_work", "tile_idx → (block,head,batch)", "control", null, "03");

    b += panel(580, 176, 480, 330, "CLC 路径 · 硬件派工", "gather");
    b += box(620, 216, 400, 70, "ClcState", "response buf + mbarrier 管线", "gather", 5, "04");
    b += box(620, 330, 400, 70, "producer / consumer", "prefetch ↔ wait/release", "gather", 6, "05");
    b += box(620, 424, 400, 60, "clc_scheduler_warp", "empty warp · 仅 leader CTA", "control", 7, "06");

    b += box(300, 566, 500, 64, "while work_tile.is_valid: 统一消费骨架", "所有角色 warp 同一循环", "control", 8, "06");

    b += edge(rootId, "M260 88H300", null, "orange");
    b += edge(rootId, "M430 120V168H290V216", [350, 148, "STATIC", 80], "compute");
    b += edge(rootId, "M670 120V168H820V216", [750, 148, "CLC", 60], "gather");
    b += edge(rootId, ortho(290, 286, 290, 330), null, "compute");
    b += edge(rootId, ortho(290, 400, 290, 424), null, "compute");
    b += edge(rootId, ortho(820, 286, 820, 330), null, "gather");
    b += edge(rootId, ortho(820, 400, 820, 424), null, "gather");
    b += edge(rootId, "M290 484V538H480V566", null, "control");
    b += edge(rootId, "M820 484V538H620V566", null, "control");
    return {
      svg: baseSvg(rootId, "scheduler", 680, b,
        "Tile scheduling: static LPT with L2 swizzle vs CLC dynamic"),
      title: "调度层 · 顺序 × 亲和 × 分配方式",
      badges: ["青 = 选型", "蓝 = 静态排序", "紫 = CLC 硬件", "绿 = 统一骨架", "橙 = 回退"],
      notes: [
        ["选型决策树", "varlen → 前缀和调度;causal/local/CLC → LPT;dense persistent → grid-stride;兜底 single-tile。"],
        ["L2 swizzle", "估算 L2 能同居几个 head 的 K/V,让时间相邻的 CTA 空间上共享——命中率即有效带宽。"],
        ["LPT", "块序反转,最重的 causal 尾块最先派发;Graham 定理保证 4/3 近似,线性递增负载下近乎最优。"],
        ["CLC", "persistent CTA 向硬件申请下一份工作,响应经 mbarrier 通知;软件不再做 swizzle——顺序控制权只能有一个主人。"],
        ["回退智慧", "dense noncausal 本来均衡、varlen-MHA 伤 L2:动态调度是保险,均匀负载不值得付保费。"]
      ],
      memory: "LPT 管顺序,swizzle 管亲和,CLC 管分配——三个正交自由度各司其职。"
    };
  }

  function backwardDiagram(rootId) {
    var b = "";
    b += box(70, 56, 440, 66, "preprocess", M("D=\\mathrm{rowsum}(dO\\odot O),\\ \\mathrm{LSE}\\cdot\\log_2 e", "D & LSE·log2e"), "control", 1, "01");
    b += box(590, 56, 430, 66, "dQaccum 清零", "fp32 全局记账本", "state", null, "01");

    b += panel(40, 178, 1020, 420, "主 kernel · 以 K-tile 为家 (n_block)", "compute");
    b += box(80, 218, 300, 70, "16 warps · 5 tilers", "reduce4+compute8+mma+load+relay", "compute", 2, "02");
    b += box(420, 218, 300, 70, "TMEM 复用地图", "S/P 叠 · dP/dS 叠 · dQ 叠 dP", "state", 3, "03");
    b += box(760, 218, 260, 70, "load: K/V 一次", "Q/dO/LSE/dPsum 流水", "compute", null, "02");

    b += box(80, 340, 640, 70, "5-GEMM 软件流水", "S(i) → dK(i-1) → dQ(i-1) → dP(i) → dV(i)", "gather", 4, "04");
    b += box(760, 340, 260, 70, "softmax 重算 + dS", M("P=2^{Sc-\\mathrm{LSE}},\\ dS=P(dP-D)", "recompute"), "compute", 5, "05");

    b += box(80, 470, 440, 80, "dQ · TMA atomic reduce", "cp.async.bulk.reduce.add.f32", "state", 6, "06");
    b += box(580, 470, 440, 80, "deterministic 信号量", "wait_eq → 写 → arrive_inc", "orange", null, "06", { dashed: true });

    b += box(70, 650, 440, 66, "postprocess", M("dQ=\\mathrm{cast}(dQ_{acc}\\times c)", "dQ cast+scale"), "control", 7, "07");
    b += box(590, 650, 430, 66, "dK/dV 就地写出", "TMEM 累加完 · epilogue ×scale", "gather", null, "04");

    b += edge(rootId, ortho(290, 122, 290, 218), [352, 160, "D / LSE 流入", 118], "control");
    b += edge(rootId, "M805 122V170H890V218", null, "state");
    b += edge(rootId, ortho(230, 288, 230, 340), null, "compute");
    b += edge(rootId, ortho(570, 288, 570, 340), null, "state");
    b += edge(rootId, ortho(890, 288, 890, 340), null, "compute");
    b += edge(rootId, "M760 388H720", null, "compute");
    b += edge(rootId, ortho(300, 410, 300, 470), null, "gather");
    b += edge(rootId, "M520 510H580", null, "orange");
    b += edge(rootId, ortho(290, 550, 290, 650), null, "state");
    b += edge(rootId, "M700 410V430H1044V683H1020", [880, 620, "dK/dV epilogue", 140], "gather");
    return {
      svg: baseSvg(rootId, "backward", 740, b,
        "Backward pass: preprocess, K-centric 5-GEMM kernel, dQ atomic reduce, postprocess"),
      title: "反向三段式 · 以 K 为家",
      badges: ["绿 = pre/post", "蓝 = 计算", "紫 = GEMM 流水", "玫瑰 = 归约", "橙 = 确定性"],
      notes: [
        ["预处理", "D = rowsum(dO⊙O) 免去存 P;LSE 预乘 log2(e) 服务 exp2;顺手清零 dQaccum。"],
        ["以 K 为家", "固定 n_block,扫所有相关 m_block:dK/dV 在 TMEM 本地累加,免全局归约。"],
        ["五连环", "S→dK→dQ→dP→dV 首尾相接;仅两处真 wait:dS 就绪、tdQ 被 reduce 消费(TMEM 复用约束)。"],
        ["dQ 归约", "TMA 引擎直接在 gmem 的 fp32 dQaccum 上原子加;deterministic 不去掉 atomic,只用信号量钉死顺序。"],
        ["收尾", "postprocess 把 dQaccum 乘 softmax_scale 转 dtype;dK 在 epilogue 乘 scale——缩放只花在 O(nd) 的输出上。"]
      ],
      memory: "dK/dV 在家收快递,dQ 客场寄账;P 不存不传,LSE 一到就能重造。"
    };
  }

  /* ================================================================ *
   * Aux diagrams (illustrative, non-interactive)
   * ================================================================ */

  function grid(x0, y0, cell, rows, cols, keepFn, tone) {
    var out = "";
    for (var r = 0; r < rows; r += 1) {
      for (var c = 0; c < cols; c += 1) {
        var keep = keepFn(r, c);
        out += '<rect x="' + (x0 + c * cell) + '" y="' + (y0 + r * cell) +
          '" width="' + (cell - 2) + '" height="' + (cell - 2) + '" rx="2" fill="' +
          (keep ? toneFill(tone) : "#f0f0ee") + '" stroke="' +
          (keep ? toneStroke(tone) : "#d8d8d4") + '" stroke-width="0.8"/>';
      }
    }
    return out;
  }

  function auxMaskTypes(rootId) {
    var b = "";
    var n = 8;
    var cell = 18;
    var configs = [
      ["FULL", function () { return true; }, "control"],
      ["CAUSAL", function (r, c) { return c <= r; }, "compute"],
      ["INV-CAUSAL", function (r, c) { return c >= r; }, "gather"],
      ["BI-CAUSAL", function (r, c) { return c === r; }, "state"]
    ];
    configs.forEach(function (config, index) {
      var x0 = 60 + index * 260;
      b += textLabel(x0 + (n * cell) / 2, 64, config[0], 12, P.ink, 600);
      b += grid(x0, 84, cell, n, n, config[1], config[2]);
      b += textLabel(x0 + (n * cell) / 2, 84 + n * cell + 22,
        index === 1 ? "右下对齐 tril" : index === 2 ? "左上对齐 triu" : index === 3 ? "两者之交" : "全 1", 9.5, P.muted, 500);
    });
    b += textLabel(550, 300, "s_q = s_k 时的四种切片几何;s_q ≠ s_k 时按各自对齐方向平移(CAUSAL 变梯形,BI-CAUSAL 可为空)。", 10.5, P.muted, 500);
    return {
      svg: baseSvg(rootId, "attnslice-masktypes", 330, b, "Four AttnSlice mask type geometries"),
      caption: "四种 AttnSlice mask 类型的几何（8×8 示意，行=Q、列=K）：着色格为合法注意力位置。"
    };
  }

  function auxTmemMap(rootId) {
    var b = "";
    var x0 = 70;
    var w = 960;
    var y0 = 90;
    var h = 70;
    var colw = w / 512;
    function seg(c0, c1, tone, label, sub, yy, hh, dashed) {
      var x = x0 + c0 * colw;
      var ww = (c1 - c0) * colw;
      return '<rect x="' + x + '" y="' + yy + '" width="' + ww + '" height="' + hh +
        '" rx="6" fill="' + toneFill(tone) + '" stroke="' + toneStroke(tone) +
        '" stroke-width="1.3" ' + (dashed ? 'stroke-dasharray="6 5" ' : "") + "/>" +
        textLabel(x + ww / 2, yy + hh / 2 - (sub ? 7 : 0), label, 11, P.ink, 600) +
        (sub ? textLabel(x + ww / 2, yy + hh / 2 + 13, sub, 8.5, P.muted, 500) : "");
    }
    b += textLabel(550, 48, "TMEM · 128 行 × 512 列 (fp32)", 12.5, P.ink, 600);
    b += seg(0, 128, "compute", "S0", "[0,128)", y0, h);
    b += seg(128, 256, "compute", "S1", "[128,256)", y0, h);
    b += seg(256, 384, "state", "O0", "[256,384)", y0, h);
    b += seg(384, 512, "state", "O1", "[384,512)", y0, h);
    b += seg(64, 192, "gather", "P0 (bf16 视图)", "[64,192) 叠放", y0 + 96, 54, true);
    b += seg(192, 320, "gather", "P1 (bf16 视图)", "[192,320) 叠放", y0 + 96, 54, true);
    b += seg(0, 64, "orange", "vec", "row_max/sum", y0 + 96, 54, true);
    [0, 64, 128, 192, 256, 320, 384, 448, 512].forEach(function (c) {
      b += '<line x1="' + (x0 + c * colw) + '" y1="' + (y0 + h + 4) + '" x2="' +
        (x0 + c * colw) + '" y2="' + (y0 + h + 12) + '" stroke="' + P.muted + '" stroke-width="1"/>';
      b += textLabel(x0 + c * colw, y0 + h + 24, String(c), 8.5, P.muted, 500);
    });
    b += textLabel(550, 292, "P 以 bf16 视图叠放在 S 的空间上（先 T2R 读空 S，再 R2T 写 P）；row_max/sum 向量复用 S 前段。", 10.5, P.muted, 500);
    return {
      svg: baseSvg(rootId, "tmem-map", 320, b, "TMEM 512-column allocation map"),
      caption: "前向 TMEM 地图（tile 128×128、head_dim_v=128、q_stage=2）：上排为常驻分配，下排虚线为时间复用的叠放视图。"
    };
  }

  function auxPipelineWave(rootId) {
    var b = "";
    var lanes = [
      ["TMA load", "orange"],
      ["MMA (UMMA)", "gather"],
      ["Softmax", "compute"],
      ["Correction", "state"]
    ];
    var laneY = {};
    lanes.forEach(function (lane, i) {
      var y = 84 + i * 74;
      laneY[lane[0]] = y;
      b += textLabel(120, y + 22, lane[0], 10.5, P.ink, 600);
      b += '<line x1="210" y1="' + (y + 44) + '" x2="1050" y2="' + (y + 44) +
        '" stroke="' + P.rule + '" stroke-width="1"/>';
    });
    function tile(lane, x, w, label, tone) {
      var y = laneY[lane];
      b += '<rect x="' + x + '" y="' + y + '" width="' + w +
        '" height="44" rx="7" fill="' + toneFill(tone) + '" stroke="' + toneStroke(tone) +
        '" stroke-width="1.2"/>' + textLabel(x + w / 2, y + 22, label, 9.5, P.ink, 600);
    }
    /* KV blocks j, j+1, j+2 flowing through the ladder. */
    tile("TMA load", 230, 110, "K/V j+2", "orange");
    tile("TMA load", 360, 110, "K/V j+3", "orange");
    tile("TMA load", 490, 110, "K/V j+4", "orange");
    tile("MMA (UMMA)", 230, 120, "QK → S(j)", "gather");
    tile("MMA (UMMA)", 370, 130, "PV → O(j-1)", "gather");
    tile("MMA (UMMA)", 520, 120, "QK → S(j+1)", "gather");
    tile("MMA (UMMA)", 660, 130, "PV → O(j)", "gather");
    tile("Softmax", 370, 200, "T2R·exp2·R2T P(j)", "compute");
    tile("Softmax", 660, 200, "P(j+1)", "compute");
    tile("Correction", 440, 150, "rescale O(j-1)", "state");
    tile("Correction", 730, 150, "rescale O(j)", "state");
    b += textLabel(630, 400, "同一时刻:TMA 在搬 j+3,MMA 在做 j+1 的 QK,softmax 在写 P(j),correction 在改 O(j-1)。", 10.5, P.muted, 500);
    b += textLabel(630, 424, "错峰由 pipeline_s_p_o 双义槽位 + sScale cross-release 共同维持。", 10.5, P.muted, 500);
    return {
      svg: baseSvg(rootId, "pipeline-wave", 450, b, "Overlapped execution wave across warp roles"),
      caption: "四个角色在时间轴上的重叠推进（示意）：每个角色处理不同的 KV block，槽位状态是唯一的协调语言。"
    };
  }

  function auxMaskSegments(rootId) {
    var b = "";
    var cell = 24;
    var cols = 32;
    var x0 = 80;
    var y0 = 90;
    /* one Q tile row of KV blocks: skipped | full | partial(diag) */
    for (var c = 0; c < cols; c += 1) {
      var tone, label;
      if (c > 22) { tone = null; }
      else if (c >= 20) { tone = "orange"; }
      else { tone = "control"; }
      var x = x0 + c * (cell + 4);
      if (tone == null) {
        b += '<rect x="' + x + '" y="' + y0 + '" width="' + cell + '" height="' + cell +
          '" rx="4" fill="#f0f0ee" stroke="#d8d8d4" stroke-width="0.8" stroke-dasharray="3 3"/>';
      } else {
        b += '<rect x="' + x + '" y="' + y0 + '" width="' + cell + '" height="' + cell +
          '" rx="4" fill="' + toneFill(tone) + '" stroke="' + toneStroke(tone) + '" stroke-width="1"/>';
      }
    }
    b += textLabel(x0 + 10 * 28, y0 - 26, "一个 Q tile 的 KV block 轴 (n_block →)", 10.5, P.ink, 600);
    b += textLabel(x0 + 10 * 28, y0 + 56, "绿色 full 块 · Mainloop-2 · 无 mask 代码", 10, P.controlStroke, 600);
    b += textLabel(x0 + 21.2 * 28, y0 + 82, "橙色 partial 块 · Mainloop-1 · mask_fn + R2P", 10, P.orangeStroke, 600);
    b += textLabel(x0 + 25.5 * 28 + 40, y0 - 26, "BlockInfo 跳过", 10, P.muted, 600);
    b += '<path d="M' + (x0 + 20 * 28) + " " + (y0 + 40) + "V" + (y0 + 66) +
      '" stroke="' + P.orangeStroke + '" stroke-width="1.2" fill="none"/>';
    b += textLabel(550, 210, "causal 下 partial 带宽度 ≈ 1 块;窗宽 w 的滑窗左右各一条 partial 带,中间 full 区宽 ≈ w/128 − 2。", 10.5, P.muted, 500);
    return {
      svg: baseSvg(rootId, "mask-segments", 240, b, "Full/partial/skipped KV block segmentation"),
      caption: "三段循环的几何：只有对角带（橙）付 mask 的钱，full 区（绿）编译期即无 mask 代码，尾部（虚线）被 BlockInfo 整块跳过。"
    };
  }

  function auxCorrectionHandshake(rootId) {
    var b = "";
    var rows = [
      ["softmax0 (S0)", "compute", 84],
      ["correction", "state", 174],
      ["softmax1 (S1)", "compute", 264]
    ];
    rows.forEach(function (row) {
      b += textLabel(120, row[2] + 20, row[0], 10.5, P.ink, 600);
      b += '<line x1="215" y1="' + (row[2] + 40) + '" x2="1050" y2="' + (row[2] + 40) +
        '" stroke="' + P.rule + '" stroke-width="1"/>';
    });
    function tile(y, x, w, label, tone) {
      b += '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="40" rx="7" fill="' +
        toneFill(tone) + '" stroke="' + toneStroke(tone) + '" stroke-width="1.2"/>' +
        textLabel(x + w / 2, y + 20, label, 9.3, P.ink, 600);
    }
    tile(84, 230, 170, "step(j): scale→sScale0", "compute");
    tile(84, 480, 200, "exp2 + 写 P0(j)", "compute");
    tile(84, 760, 170, "step(j+1) 等 WAR", "cyan");
    tile(174, 300, 180, "rescale O0(j-1)", "state");
    tile(174, 560, 180, "rescale O1(j-1)", "state");
    tile(174, 820, 180, "rescale O0(j)", "state");
    tile(264, 340, 170, "scale→sScale1", "compute");
    tile(264, 590, 200, "exp2 + 写 P1(j)", "compute");
    b += textLabel(630, 356, "correction 释放的是「对面」stage 的槽位(cross-release):S0 发布后要等 correction 服务完 S1", 10.2, P.muted, 500);
    b += textLabel(630, 380, "才拿回写权——两组 softmax 被强制错开半拍,任意时刻只有一组与 correction 重叠。", 10.2, P.muted, 500);
    return {
      svg: baseSvg(rootId, "correction-handshake", 410, b, "sScale handshake and cross-release staggering"),
      caption: "sScale 握手时序（示意）：RAW 由 barrier 保证「写完才读」，WAR 由 pipeline 保证「读完才覆写」，cross-release 制造两组 softmax 的相位差。"
    };
  }

  function auxLptSwizzle(rootId) {
    var b = "";
    var x0 = 80;
    var y0 = 96;
    var n = 8;
    var bw = 44;
    b += textLabel(x0 + n * (bw + 8) / 2, 56, "causal 负载:第 m 个 Q tile 的 KV 块数 ∝ m+1", 11, P.ink, 600);
    for (var m = 0; m < n; m += 1) {
      var height = 18 + m * 14;
      var x = x0 + m * (bw + 8);
      b += '<rect x="' + x + '" y="' + (y0 + (n * 14 + 18) - height) + '" width="' + bw +
        '" height="' + height + '" rx="5" fill="' + toneFill("compute") +
        '" stroke="' + toneStroke("compute") + '" stroke-width="1.1"/>';
      b += textLabel(x + bw / 2, y0 + n * 14 + 40, "m=" + m, 9, P.muted, 500);
      b += textLabel(x + bw / 2, y0 + (n * 14 + 18) - height - 12, "#" + (n - m), 9.5, P.orangeStroke, 700);
    }
    b += textLabel(x0 + n * (bw + 8) / 2, y0 + n * 14 + 66, "橙色编号 = LPT 派发顺序(最重的 m=7 第一个跑)", 10, P.orangeStroke, 600);

    var gx = 640;
    b += textLabel(gx + 180, 56, "L2 swizzle:tile_idx 先按节拍分组", 11, P.ink, 600);
    for (var g = 0; g < 2; g += 1) {
      b += panel(gx + g * 190, 86, 170, 150, "节拍 " + g + " · head " + (g * 2) + "-" + (g * 2 + 1), g === 0 ? "control" : "gather");
      for (var i = 0; i < 4; i += 1) {
        var yy = 108 + i * 28;
        b += '<rect x="' + (gx + 18 + g * 190) + '" y="' + yy + '" width="134" height="22" rx="4" fill="' +
          toneFill(g === 0 ? "control" : "gather") + '" stroke="' + toneStroke(g === 0 ? "control" : "gather") +
          '" stroke-width="1"/>' +
          textLabel(gx + 18 + g * 190 + 67, yy + 11, "tile " + (g * 4 + i) + " · head " + (g * 2 + (i % 2)), 8.5, P.ink, 500);
      }
    }
    b += textLabel(gx + 180, 262, "同节拍的 CTA 共享 swizzle 个 head 的 K/V → L2 命中", 10, P.muted, 500);
    return {
      svg: baseSvg(rootId, "lpt-swizzle", 300, b, "LPT ordering and L2 swizzle grouping"),
      caption: "左：causal 工作量沿 Q tile 线性递增，LPT 反转派发顺序让重块先行。右：L2 swizzle 把时间相邻的 tile 绑到同一小组 head 上共享 K/V。"
    };
  }

  function auxBwdTmem(rootId) {
    var b = "";
    var x0 = 70;
    var w = 960;
    var y0 = 84;
    var cols = 512;
    var colw = w / cols;
    function seg(c0, c1, tone, label, sub, yy, hh, dashed) {
      var x = x0 + c0 * colw;
      var ww = (c1 - c0) * colw;
      return '<rect x="' + x + '" y="' + yy + '" width="' + ww + '" height="' + hh +
        '" rx="6" fill="' + toneFill(tone) + '" stroke="' + toneStroke(tone) +
        '" stroke-width="1.3" ' + (dashed ? 'stroke-dasharray="6 5" ' : "") + "/>" +
        textLabel(x + ww / 2, yy + hh / 2 - (sub ? 7 : 0), label, 10.5, P.ink, 600) +
        (sub ? textLabel(x + ww / 2, yy + hh / 2 + 13, sub, 8.3, P.muted, 500) : "");
    }
    b += textLabel(550, 48, "反向 TMEM (1-CTA · tile 128 · hd 128)", 12, P.ink, 600);
    b += seg(0, 128, "compute", "S", "[0,128)", y0, 64);
    b += seg(128, 256, "state", "dV 累加", "[128,256)", y0, 64);
    b += seg(256, 384, "compute", "dP", "[256,384)", y0, 64);
    b += seg(384, 512, "state", "dK 累加", "[384,512)", y0, 64);
    b += seg(0, 64, "gather", "P (bf16)", "叠 S 左半", y0 + 90, 50, true);
    b += seg(256, 320, "gather", "dS (bf16)", "叠 dP 左半", y0 + 90, 50, true);
    b += seg(256, 384, "orange", "dQ", "完全叠在 dP 上(分时)", y0 + 168, 50, true);
    b += textLabel(550, 350, "dQ 与 dP 分时共享同一空间:dQ(i-1) 被 reduce 读走 → 槽位变空 → dP(i) 才写入——pipeline_dQ 的 full/empty 各守一拍。", 10.2, P.muted, 500);
    return {
      svg: baseSvg(rootId, "bwd-tmem", 380, b, "Backward TMEM overlap map"),
      caption: "反向 TMEM 复用：dV/dK 常驻累加，P 叠 S、dS 叠 dP，dQ 与 dP 分时共享——每处叠放都对应一条「先消费后覆写」的流水线约束。"
    };
  }

  /* ---------------- registry ---------------- */

  var mains = {
    attnslice: attnsliceDiagram,
    blackwell: blackwellDiagram,
    pipeline: pipelineDiagram,
    mask: maskDiagram,
    softmax: softmaxDiagram,
    correction: correctionDiagram,
    scheduler: schedulerDiagram,
    backward: backwardDiagram
  };

  var auxes = {
    "attnslice-masktypes": auxMaskTypes,
    "tmem-map": auxTmemMap,
    "pipeline-wave": auxPipelineWave,
    "mask-segments": auxMaskSegments,
    "correction-handshake": auxCorrectionHandshake,
    "lpt-swizzle": auxLptSwizzle,
    "bwd-tmem": auxBwdTmem
  };

  var buildSerial = 0;

  function build(key) {
    var builder = mains[key];
    if (!builder) throw new Error("Unknown diagram: " + key);
    buildSerial += 1;
    var rootId = "magi-" + key + "-" + buildSerial;
    var report = builder(rootId);
    validateStaticGeometry(report.svg, key);
    return report;
  }

  function buildAux(key) {
    var builder = auxes[key];
    if (!builder) return null;
    buildSerial += 1;
    var report = builder("magi-aux-" + key + "-" + buildSerial);
    validateStaticGeometry(report.svg, key);
    return report;
  }

  window.MagiDiagrams = { build: build, buildAux: buildAux };
})();
