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

    function touchesNodeBoundary(point, node) {
      var epsilon = 0.8;
      var onVertical = (
        Math.abs(point.x - node.x) <= epsilon ||
        Math.abs(point.x - (node.x + node.w)) <= epsilon
      ) && point.y >= node.y - epsilon && point.y <= node.y + node.h + epsilon;
      var onHorizontal = (
        Math.abs(point.y - node.y) <= epsilon ||
        Math.abs(point.y - (node.y + node.h)) <= epsilon
      ) && point.x >= node.x - epsilon && point.x <= node.x + node.w + epsilon;
      return onVertical || onHorizontal;
    }

    edges.forEach(function (item) {
      if (item.points.length < 2) {
        throw new Error(diagramKey + ": connector has too few points: " + item.d);
      }
      var source = item.points[0];
      var target = item.points[item.points.length - 1];
      if (!boxes.some(function (node) { return touchesNodeBoundary(source, node); })) {
        throw new Error(diagramKey + ": connector source is detached from nodes: " + item.d);
      }
      if (!boxes.some(function (node) { return touchesNodeBoundary(target, node); })) {
        throw new Error(diagramKey + ": connector target is detached from nodes: " + item.d);
      }
    });

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

    b += edge(rootId, "M530 152V194H345V236", null, "control");
    b += edge(rootId, "M195 152V200H785V236", null, "compute");
    b += edge(rootId, "M880 152V200H1040V348H740V366", null, "control");
    b += edge(rootId, ortho(345, 302, 345, 366), null, "gather");
    b += edge(rootId, "M785 302V334H475V366", null, "state");
    b += edge(rootId, ortho(345, 432, 345, 496), null, "compute");
    b += edge(rootId, "M770 399H830", null, "state");
    b += edge(rootId, "M510 528H620", null, "cyan");
    b += edge(rootId, "M785 560V593H475V626", null, "compute");
    b += edge(rootId, "M770 659H830", null, "gather");
    b += edge(rootId, "M1020 399H1040V659H1020", null, "state");
    return {
      svg: baseSvg(rootId, "attnslice", 730, b,
        "AttnSlice contract and host-side forward journey"),
      title: "AttnSlice 契约 · host 侧一次前向",
      badges: ["蓝 = 计算/实例化", "绿 = ranges 与 mask", "玫瑰 = 输出契约", "紫 = 折叠/启动", "青 = 编译缓存"],
      notes: [
        ["输入", "q_ranges/k_ranges 用 [start,end) 选出 Q、K 区间，mask_type 指定区间内部的几何。"],
        ["当前限制", "CuTe DSL 路径要求 ranges 从 0 开始、连续且不重叠，因此可转换为 cu_seqlens。"],
        ["输出", "LSE 始终为 fp32，空集记为 -inf；可能重叠时，functional 路径还会用 fp32 out 做原子累加。"],
        ["选择配置", "host 根据序列长度、mask 和 head_dim 选择 Q stage、2-CTA 与 CLC。"],
        ["编译缓存", "所有会改变生成代码的配置都进入 compile_key；相同 key 只需编译一次。"]
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
    b += box(70, 250, 200, 116, "sQ", "swizzled · q_stage 级", "compute", 2, "04");
    b += box(310, 250, 200, 58, "sK / sV 复用", "kv_stage 环形缓冲", "compute", 3, "04");
    b += box(310, 322, 200, 44, "sO · epilogue", "TMA S2G 前的写回槽", "compute", null, "05", { titleSize: 10.8, subSize: 8.0 });

    b += panel(600, 52, 460, 330, "TMEM · 128行 × 512列 fp32", "state");
    b += box(630, 96, 190, 60, "S0 | S1", "[0,128) | [128,256)", "state", 6, "02");
    b += box(850, 96, 180, 60, "O0 | O1", "[256,384) | [384,512)", "state", 7, "02");
    b += box(630, 196, 190, 60, "P0 / P1 叠放", "bf16 视图 · S 空间复用", "state", 8, "06");
    b += box(850, 196, 180, 60, "vec 复用", "row_max/sum 暂存", "state", null, "02", { dashed: true });
    b += box(790, 296, 240, 60, "TMEM alloc / free", "仅 MMA warp", "gather", 9, "07");

    b += panel(40, 442, 1020, 130, "tcgen05 UMMA · 单线程发射 · 2-CTA 可选", "gather");
    b += box(90, 476, 400, 66, M("S = Q K^{\\mathsf T}", "S = Q K^T"), "tiled_mma_qk · A/B 来自 SMEM", "gather", 4, "03");
    b += box(590, 476, 400, 66, M("O \\mathrel{+}= P V", "O += P V"), "tiled_mma_pv · P 来自 TMEM", "gather", 5, "03");

    b += edge(rootId, ortho(170, 148, 170, 250), [218, 190, "TMA G2S · Q", 130], "orange");
    b += edge(rootId, "M240 148V200H410V250", [350, 188, "TMA G2S · K/V", 130], "orange");
    b += edge(rootId, "M170 366V420H240V476", null, "compute");
    b += edge(rootId, "M310 279H290V420H340V476", null, "compute");
    b += edge(rootId, "M510 279H590V430H620V476", [535, 416, "V · SMEM B operand", 140], "compute");
    b += edge(rootId, "M490 509H540V126H630", [560, 300, "写 S 累加器", 120], "state");
    b += edge(rootId, "M660 256V476", [710, 400, "P · TMEM A operand", 140], "gather");
    b += edge(rootId, "M725 156V196", null, "state");
    b += edge(rootId, "M820 126H835V226H850", null, "state");
    b += edge(rootId, "M990 509H1044V126H1030", [1015, 300, "写 O 累加器", 112, 8.2], "state");
    b += edge(rootId, "M940 156V180H1070V400H410V366", [900, 388, "correction epilogue → sO", 190], "state");
    b += edge(rootId, "M510 344H570V180H410V148", [490, 168, "TMA S2G · O", 120], "orange");
    return {
      svg: baseSvg(rootId, "blackwell", 600, b,
        "Blackwell execution substrate: TMA, SMEM, TMEM and tcgen05 UMMA"),
      title: "SM100 执行基座 · 数据只向前流",
      badges: ["橙 = GMEM/TMA", "蓝 = SMEM", "玫瑰 = TMEM", "紫 = UMMA"],
      notes: [
        ["数据路线", "TMA 把 GMEM 数据搬到 SMEM，UMMA 计算后把累加结果放进 TMEM。"],
        ["TMEM 地图", "S0/S1 和 O0/O1 各占 128 列；S 被读走后，其空间再存 bf16 P 与行统计量。"],
        ["两个矩阵乘", "QK 从 SMEM 读取 Q/K；PV 直接从 TMEM 读取 P，省去一次 SMEM 中转。"],
        ["生命周期", "MMA warp 分配和释放 TMEM；释放前必须等待 softmax 与 correction 完成最后一次访问。"],
        ["2-CTA", "两个 CTA 共同处理 256 行 Q，并各存一半 K/V。只有双方负载接近时才启用。"]
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
    b += box(470, 582, 550, 64, "warp_idx 分派 · 一份代码七种角色", "setmaxregister 调整后进入各自循环", "cyan", 9, "07");

    b += edge(rootId, "M310 124H470", [382, 110, "producer", 76, 8.2], "compute");
    b += edge(rootId, "M310 224H470", [382, 210, "S-full", 76, 8.2], "compute");
    b += edge(rootId, "M310 338H390V238H470", [390, 286, "P release", 86, 8.2], "compute");
    b += edge(rootId, "M310 362H470", [382, 348, "写 corr_scale", 104, 8.2], "compute");
    b += edge(rootId, "M470 368H410V480H310", [408, 418, "读 scale", 76, 8.2], "state");
    b += edge(rootId, "M310 468H370V244H470", [370, 292, "O release", 80, 8.2], "state");
    b += edge(rootId, "M310 492H470", [382, 478, "O / sO", 76, 8.2], "compute");
    b += edge(rootId, "M720 224H780", null, "gather");
    b += edge(rootId, "M310 504H400V550H900V518", null, "state");
    b += edge(rootId, "M780 366H735V480H720", null, "state");
    b += edge(rootId, "M190 518V582", null, "compute");
    return {
      svg: baseSvg(rootId, "pipeline", 700, b,
        "Warp specialization roles and mbarrier pipelines"),
      title: "16 个 warp · 六条流水线",
      badges: ["蓝 = warp 角色", "紫 = 流水线", "玫瑰 = sScale/O 同步", "青 = 分派"],
      notes: [
        ["角色与资源", "softmax 获得最多寄存器；load、MMA、epilogue 主要发起异步指令，只需较少寄存器。"],
        ["S/P/O 槽位", "S 写完后 softmax 才能读；P 写完且旧 O 校准完后，MMA 才能开始 PV。"],
        ["sScale 协议", "RAW 保证写完才读，WAR 保证读完才覆盖；cross-release 让两组 softmax 错峰。"],
        ["最后一个 block", "主循环可由 GEMM 顺序推断旧 O 已完成；最后一块没有后继信号，必须显式等待。"],
        ["统一循环", "所有角色都处理同一个 work_tile 循环，只由 warp_idx 选择各自工作。"]
      ],
      memory: "warp 之间不传数据,只翻转槽位状态;数据永远躺在约定好的 TMEM/SMEM 里。"
    };
  }

  function maskDiagram(rootId) {
    var b = "";
    b += box(70, 60, 430, 64, "Q tile (m_block) + SeqlenInfoQK", "相对坐标系 · cu_seqlens 读一次", "compute", 1, "07");
    b += box(600, 60, 420, 64, "右下对齐 causal · 坐标均从 0 开始", M("k \\le q + (s_k - s_q)", "k <= q + (sk - sq)"), "cyan", null, "01");

    b += panel(40, 176, 1020, 128, "第一层 · BlockInfo 跳块(免费)", "control");
    b += box(90, 210, 400, 66, "get_n_block_min_max", "整块非法的 n_block 不迭代", "control", 2, "01");
    b += box(560, 210, 440, 66, "get_n_block_min_causal_local_mask", "定位 partial 带的边界", "control", 3, "01");
    b += box(410, 318, 280, 38, "n_block 分段结果", "partial-right | full | partial-left", "cyan", null, null, { titleSize: 10.5, subSize: 7.8 });

    b += panel(40, 356, 1020, 128, "第二层 · 三段主循环(近免费)", "compute");
    b += box(90, 390, 280, 66, "Mainloop-1 · partial", "带 mask_fn · 从右往左", "orange", 4, "02");
    b += box(410, 390, 280, 66, "Mainloop-2 · full", "不传 mask_fn · 零 mask 代码", "control", 5, "02");
    b += box(730, 390, 280, 66, "Mainloop-3 · partial", "仅 local 左窗带", "orange", null, "02");

    b += panel(40, 536, 1020, 128, "第三层 · 元素级写 -inf(边界块专属)", "state");
    b += box(90, 570, 280, 66, "apply_mask_sm100", "col_limit 由行号线性给出", "state", 6, "03");
    b += box(410, 570, 280, 66, "R2P 位掩码", "32 列一条 uint32", "state", 7, "05");
    b += box(730, 570, 280, 66, "mask_mod · block-sparse", "任意谓词 · 独立慢路径", "state", 8, "06");

    b += edge(rootId, ortho(285, 124, 285, 210), null, "compute");
    b += edge(rootId, "M500 92H530V190H650V210", null, "compute");
    b += edge(rootId, "M810 124V162H780V210", null, "cyan");
    b += edge(rootId, "M290 276V298H480V318", null, "control");
    b += edge(rootId, "M780 276V298H620V318", null, "control");
    b += edge(rootId, "M470 356V372H230V390", null, "cyan");
    b += edge(rootId, "M550 356V390", null, "cyan");
    b += edge(rootId, "M630 356V372H870V390", null, "cyan");
    b += edge(rootId, ortho(230, 456, 230, 570), null, "orange");
    b += edge(rootId, "M370 603H410", [390, 589, "uses", 52, 8.2], "state");
    b += edge(rootId, "M870 456V510H550V570", [700, 498, "local 双界", 100, 8.2], "orange");
    return {
      svg: baseSvg(rootId, "mask", 700, b,
        "Three-layer masking: block skipping, segmented loop, element predicates"),
      title: "块级 Mask · 三层防线",
      badges: ["绿 = 跳块/full", "橙 = partial 带", "玫瑰 = 元素级", "青 = 几何"],
      notes: [
        ["符号", "q/k 是切片内从 0 开始的行列下标，s_q/s_k 是 Q/K 长度；长度相等时公式退化为 k≤q。"],
        ["先跳过", "BlockInfo 用 tile 角点判断整块是否无效；无效 K block 不进入循环。"],
        ["再分段", "边界块走 partial 段并应用 mask；完全有效块走不含 mask 代码的 full 段。"],
        ["最后处理元素", "R2P 把连续 32 列的保留结果编码成一个 uint32，再转换为 32 个谓词。"],
        ["任意 mask", "无法用左右边界表示时，CSR 表先列出相关块，partial 块再调用 mask_mod。"]
      ],
      memory: "先跳过无效块，再直接计算完整块，最后只处理边界块内的元素。"
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
    b += edge(rootId, "M870 254H820V400H370V364", [600, 388, "rescale_threshold", 132, 8.2], "orange");
    b += edge(rootId, "M870 378H795V248H770", null, "orange");
    return {
      svg: baseSvg(rootId, "softmax", 560, b,
        "softmax_step data flow on SM100"),
      title: "softmax_step · 一个 KV block 的九个动作",
      badges: ["蓝 = 计算", "绿 = 挂载点", "紫 = 指数与写回", "玫瑰 = 发布", "橙 = 支撑机制"],
      notes: [
        ["尽早发布", "行最大值更新后立即发送 corr_scale，使 correction 可与当前 exp2 并行缩放旧 O。"],
        ["以 2 为底", "提前把 log2(e) 合入 softmax scale，内层直接使用 exp2。"],
        ["分担指数计算", "B200 的 SFU 忙时，部分 exp2 改由 FMA 多项式近似；SM103 不需要该优化。"],
        ["提前启动 PV", "P 写完前 3/4 后即可启动 PV；读取最后 1/4 前再等待完整就绪信号。"],
        ["填补等待", "确认旧 scale 已读走后再更新 row_sum，用寄存器计算覆盖潜在等待时间。"]
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
    b += edge(rootId, ortho(285, 488, 285, 494), null, "gather");
    b += edge(rootId, ortho(815, 280, 815, 320), null, "orange");
    b += edge(rootId, ortho(815, 384, 815, 424), null, "compute");
    b += edge(rootId, "M1020 248H1040V526H1020", [1000, 402, "row stats", 82, 8.2], "cyan");
    return {
      svg: baseSvg(rootId, "correction", 620, b,
        "Correction main loop and epilogue"),
      title: "Correction · 校准与归一化",
      badges: ["蓝 = 计算", "绿 = 门控", "紫 = TMEM 读改写", "玫瑰 = 同步", "橙 = 尾声", "青 = LSE"],
      notes: [
        ["分开存放", "sScale 的主循环区域存 corr_scale，另一片区域存尾声所需的 row_sum/row_max。"],
        ["整 warp 跳过", "若 32 行的 scale 都为 1，整个 warp 不再读改 O。"],
        ["O 的完成条件", "主循环可由 GEMM 发射顺序推断旧 O 已完成；最后一块必须显式等待。"],
        ["cross-release", "释放另一 stage 的槽位，强制 correction 在两组 softmax 之间交替服务。"],
        ["最终归一化", "空行写 0/-inf；sink 只增加分母；FP8 缩放也在此还原。"]
      ],
      memory: "主循环用 corr_scale 校准旧 O，尾声除以行和，并写出可继续合并的 O/LSE。"
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
        ["选型", "varlen 使用前缀和定位；causal/local 使用 LPT；稠密均匀负载使用静态 persistent；另有简单兜底。"],
        ["L2 swizzle", "估算 L2 可同时容纳多少个 head 的 K/V，并让相邻 CTA 尽量复用这些数据。"],
        ["LPT", "反转 causal block 顺序，让工作最重的尾部 block 最先开始。"],
        ["CLC", "persistent CTA 向硬件领取下一项工作，响应经 mbarrier 到达；实际派发顺序由硬件决定。"],
        ["何时回退", "负载均匀时动态请求只有额外开销；varlen MHA 还可能因顺序变化降低 L2 命中率。"]
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

    b += edge(rootId, "M510 89H590", null, "control");
    b += edge(rootId, ortho(290, 122, 290, 218), [352, 160, "D / LSE 流入", 118], "control");
    b += edge(rootId, ortho(230, 288, 230, 340), null, "compute");
    b += edge(rootId, ortho(570, 288, 570, 340), null, "state");
    b += edge(rootId, ortho(890, 288, 890, 340), null, "compute");
    b += edge(rootId, "M760 388H720", null, "compute");
    b += edge(rootId, ortho(300, 410, 300, 470), null, "gather");
    b += edge(rootId, "M520 510H580", null, "orange");
    b += edge(rootId, ortho(290, 550, 290, 650), null, "state");
    b += edge(rootId, "M700 410V430H1044V620H805V650", [950, 608, "dK/dV epilogue", 140], "gather");
    return {
      svg: baseSvg(rootId, "backward", 740, b,
        "Backward pass: preprocess, K-centric 5-GEMM kernel, dQ atomic reduce, postprocess"),
      title: "反向三段式 · 以 K 为家",
      badges: ["绿 = pre/post", "蓝 = 计算", "紫 = GEMM 流水", "玫瑰 = 归约", "橙 = 确定性"],
      notes: [
        ["预处理", "先算每行 D=rowsum(dO⊙O)，预处理 LSE，并清零 fp32 dQaccum。"],
        ["固定 K tile", "遍历相关 Q tile 时，dK/dV 一直在当前 CTA 的 TMEM 中累加。"],
        ["五次矩阵乘", "重算 S/P，得到 dP/dV，再形成 dS，最后计算 dQ/dK；多步按流水线交错。"],
        ["dQ 归约", "多个 K tile 都贡献 dQ，因此 TMA 原子加到全局 fp32 缓冲；确定性模式只固定加法顺序。"],
        ["收尾", "postprocess 对 dQ 乘 softmax scale 并转换 dtype；dK 在自己的 epilogue 中做同类缩放。"]
      ],
      memory: "dK/dV 在家收快递,dQ 客场寄账;P 不存不传,LSE 一到就能重造。"
    };
  }

  function overlapDiagram(rootId) {
    var b = "";
    b += panel(40, 52, 1020, 128, "META · 切阶段（一次求解,整个训练复用）", "control");
    b += box(70, 84, 290, 66, "OverlapConfig", "degree: 0 / 1 / N / None", "control", 1, "08");
    b += box(400, 84, 300, 66, "OverlapSolver 成本模型", M("\\textstyle\\sum_i \\max(C^{comm}_i, C^{calc}_{i-1})", "Σ max(comm_i, calc_i-1)"), "control", 2, "09");
    b += box(740, 84, 280, 66, "CommMeta 投递清单", "每 stage: split_sizes + dst/src", "cyan", 3, "04");

    b += panel(40, 232, 1020, 190, "前向 OVERLAP 环 · DistAttnFunc.forward", "compute");
    b += box(70, 264, 220, 70, "host: 本地 FFA", "掩护 stage-0 预取", "compute", 4, "01");
    b += box(330, 264, 250, 70, "wait i · prefetch i+1", "get_curr_q_kv_and_fetch_next", "gather", 5, "02");
    b += box(620, 264, 200, 70, "FFA stage i", "out_acc/lse_acc 原子累加", "compute", 6, "07");
    b += box(860, 264, 160, 70, "reduce i", "空壳 或 GroupReduce", "state", 7, "05");
    b += box(330, 356, 490, 50, "for ith_overlap_stage in range(overlap_degree)", "三件事各睡各的流 · 句柄相连", "compute", null, "01", { dashed: true });

    b += panel(40, 472, 640, 150, "通信原语 · 三层实现", "gather");
    b += box(70, 508, 250, 70, "group_cast / group_reduce", "hier | native | a2av 分发", "gather", 8, "04");
    b += box(370, 508, 280, 70, "a2av 降解", "pack → all2all_v → post_process", "gather", 9, "06");

    b += panel(720, 472, 340, 150, "SM 分配", "orange");
    b += box(750, 508, 280, 70, "sm_margin / KernelBarrier", "预留 SM 或协调发射顺序", "orange", 10, "03");

    b += edge(rootId, "M360 117H400", null, "control");
    b += edge(rootId, "M700 117H740", null, "control");
    b += edge(rootId, "M880 150V200H455V264", [660, 188, "清单交给 runtime", 150], "cyan");
    b += edge(rootId, "M290 299H330", null, "compute");
    b += edge(rootId, "M580 299H620", null, "gather");
    b += edge(rootId, "M820 299H860", null, "compute");
    b += edge(rootId, "M940 334V381H820", null, "state");
    b += edge(rootId, "M330 381H305V320H330", [258, 348, "下一拍", 66, 8.2], "compute", true);
    b += edge(rootId, "M455 334V356", null, "compute");
    b += edge(rootId, "M430 406V440H195V508", [320, 428, "发起 GroupCast", 130], "gather");
    b += edge(rootId, "M890 508V464H840V320H820", [852, 446, "少开 CTA", 80, 8.2], "orange");
    return {
      svg: baseSvg(rootId, "overlap", 660, b,
        "Communication-computation overlap: solver, pipeline loop, comm primitives and SM budget"),
      title: "通算融合 · 三重奏与它的地基",
      badges: ["绿 = 求解/配置", "蓝 = 计算", "紫 = 通信原语", "玫瑰 = 归约", "橙 = SM 分配", "青 = 元数据"],
      notes: [
        ["先切 stage", "OverlapSolver 根据估计通信和计算时间切分远端 KV；每个 stage 的发送/接收清单可重复使用。"],
        ["三拍流水", "计算第 i 段时预取 i+1，并处理 i-1 的结果；每拍时长由通信和计算中较慢者决定。"],
        ["默认前向", "KV-comm 模式下 partial out/LSE 留在本地并由后续 kernel 合并；反向 dKV 与 qo_comm 才走网络归约。"],
        ["三种实现", "同一 group_cast/group_reduce 接口可走 NCCL a2av、native grpcoll 或节点分层实现。"],
        ["分配 SM", "NCCL 路径用 sm_margin 留出 CTA；native 路径让通信 kernel 常驻，并用 KernelBarrier 协调顺序。"]
      ],
      memory: "切成多个 stage，预取/计算/归约错峰执行，并确保通信 kernel 真正拿得到 SM。"
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

  function auxOverlapTimeline(rootId) {
    var b = "";
    var lanes = [
      ["GroupCast 通信流", "gather"],
      ["FFA 计算 (persistent)", "compute"],
      ["归约 / 句柄", "state"]
    ];
    var laneY = {};
    lanes.forEach(function (lane, i) {
      var y = 84 + i * 74;
      laneY[lane[0]] = y;
      b += textLabel(120, y + 22, lane[0], 10, P.ink, 600);
      b += '<line x1="215" y1="' + (y + 44) + '" x2="1060" y2="' + (y + 44) +
        '" stroke="' + P.rule + '" stroke-width="1"/>';
    });
    function tile(lane, x, w, label, tone) {
      var y = laneY[lane];
      b += '<rect x="' + x + '" y="' + y + '" width="' + w +
        '" height="44" rx="7" fill="' + toneFill(tone) + '" stroke="' + toneStroke(tone) +
        '" stroke-width="1.2"/>' + textLabel(x + w / 2, y + 22, label, 9.3, P.ink, 600);
    }
    tile("GroupCast 通信流", 230, 150, "cast stage-0", "gather");
    tile("GroupCast 通信流", 400, 150, "cast stage-1", "gather");
    tile("GroupCast 通信流", 570, 150, "cast stage-2", "gather");
    tile("FFA 计算 (persistent)", 230, 150, "host 本地 attn", "compute");
    tile("FFA 计算 (persistent)", 400, 150, "FFA stage-0", "compute");
    tile("FFA 计算 (persistent)", 570, 150, "FFA stage-1", "compute");
    tile("FFA 计算 (persistent)", 740, 150, "FFA stage-2", "compute");
    tile("归约 / 句柄", 570, 150, "acc(out,lse) S0", "state");
    tile("归约 / 句柄", 740, 150, "acc S1", "state");
    tile("归约 / 句柄", 910, 145, "acc S2 · wait 全部", "state");
    b += textLabel(630, 330, "同一时刻三行并行:cast(i+1) 在通信流上搬数据,FFA(i) 在 SM−margin 个 SM 上算,归约(i−1) 收上一拍。", 10.3, P.muted, 500);
    b += textLabel(630, 354, "端到端 ≈ Σ max(通信ᵢ, 计算ᵢ₋₁) + 最后一段计算——只有 FFA stage-2 的尾巴是裸露的。", 10.3, P.muted, 500);
    return {
      svg: baseSvg(rootId, "overlap-timeline", 380, b, "Multi-stage overlap timeline across comm and compute streams"),
      caption: "overlap_degree=3 的时间线（示意）：GroupCast、FFA 计算、归约三条泳道错峰推进；通信藏在计算影子里，理想情况下每拍耗时 = max(通信, 计算)。"
    };
  }

  function auxOverlapTimelineBwd(rootId) {
    var b = "";
    var lanes = [
      ["GroupCast 流 · cp_group_gc", "gather"],
      ["FFA 反向计算 (persistent)", "compute"],
      ["GroupReduce 流 · cp_group_gr", "state"]
    ];
    var laneY = {};
    lanes.forEach(function (lane, i) {
      var y = 84 + i * 74;
      laneY[lane[0]] = y;
      b += textLabel(120, y + 22, lane[0], 9.2, P.ink, 600);
      b += '<line x1="215" y1="' + (y + 44) + '" x2="1060" y2="' + (y + 44) +
        '" stroke="' + P.rule + '" stroke-width="1"/>';
    });
    function tile(lane, x, w, label, tone, dashed) {
      var y = laneY[lane];
      b += '<rect x="' + x + '" y="' + y + '" width="' + w +
        '" height="44" rx="7" fill="' + toneFill(tone) + '" stroke="' + toneStroke(tone) +
        '" stroke-width="1.2" ' + (dashed ? 'stroke-dasharray="6 5" ' : "") + "/>" +
        textLabel(x + w / 2, y + 22, label, 9.3, P.ink, 600);
    }
    tile("GroupCast 流 · cp_group_gc", 230, 150, "cast KV stage-0", "gather");
    tile("GroupCast 流 · cp_group_gc", 400, 150, "cast KV stage-1", "gather");
    tile("GroupCast 流 · cp_group_gc", 570, 150, "cast KV stage-2", "gather");
    tile("FFA 反向计算 (persistent)", 230, 150, "host 本地反向", "compute");
    tile("FFA 反向计算 (persistent)", 400, 150, "bwd stage-0", "compute");
    tile("FFA 反向计算 (persistent)", 570, 150, "bwd stage-1", "compute");
    tile("FFA 反向计算 (persistent)", 740, 150, "bwd stage-2", "compute");
    tile("FFA 反向计算 (persistent)", 910, 145, "下一层反向…", "compute", true);
    tile("GroupReduce 流 · cp_group_gr", 570, 150, "reduce dKV S0", "state");
    tile("GroupReduce 流 · cp_group_gr", 740, 150, "reduce dKV S1", "state");
    tile("GroupReduce 流 · cp_group_gr", 910, 145, "reduce dKV S2 · 尾段", "orange");
    b += textLabel(630, 330, "与前向不同,反向的 reduce 拍是真正的网络归约:cast 与 reduce 走两个独立通信组,各占一条通信流,互不串行。", 10.3, P.muted, 500);
    b += textLabel(630, 354, "尾段 dKV 归约(橙)之后没有本层计算可遮蔽——save_tail_stage 把它推迟到与下一层反向(虚线)重叠。", 10.3, P.muted, 500);
    return {
      svg: baseSvg(rootId, "overlap-timeline-bwd", 380, b,
        "Backward multi-stage overlap timeline with dual communication streams"),
      caption: "反向 overlap_degree=3 的时间线（示意）：GroupCast 预取下一段 KV、FFA 反向计算当前段、GroupReduce 在独立通信组上归约上一段 partial dKV，三条泳道错峰推进；只有尾段 dKV 归约露在本层之外，由 save_tail_stage 藏进下一层反向。"
    };
  }

  function auxCpCommExamples(rootId) {
    var b = "";
    var CP = 4;
    var cell = 50;

    function commMatrix(x0, y0, sendFn, sendCounts, recvCounts) {
      var out = "";
      out += textLabel(x0 + CP * cell / 2, y0 - 40, "接收方 →", 9, P.muted, 500);
      for (var t = 0; t < CP; t += 1) {
        out += textLabel(x0 + t * cell + cell / 2, y0 - 18, "rank " + t, 9, P.ink, 600);
      }
      out += textLabel(x0 - 92, y0 - 18, "KV 属主 ↓", 9, P.muted, 500);
      for (var r = 0; r < CP; r += 1) {
        out += textLabel(x0 - 62, y0 + r * cell + cell / 2, "KV" + r + " (rank " + r + ")", 8.6, P.ink, 600);
        for (var c = 0; c < CP; c += 1) {
          var x = x0 + c * cell;
          var y = y0 + r * cell;
          if (r === c) {
            out += '<rect x="' + x + '" y="' + y + '" width="' + (cell - 4) + '" height="' + (cell - 4) +
              '" rx="6" fill="#f0f0ee" stroke="#d8d8d4" stroke-width="0.9"/>' +
              textLabel(x + cell / 2 - 2, y + cell / 2 - 2, "本地", 8.2, P.muted, 500);
          } else if (sendFn(r, c)) {
            out += '<rect x="' + x + '" y="' + y + '" width="' + (cell - 4) + '" height="' + (cell - 4) +
              '" rx="6" fill="' + toneFill("gather") + '" stroke="' + toneStroke("gather") + '" stroke-width="1.1"/>';
          } else {
            out += '<rect x="' + x + '" y="' + y + '" width="' + (cell - 4) + '" height="' + (cell - 4) +
              '" rx="6" fill="#fbfbf9" stroke="#e2e2de" stroke-width="0.9" stroke-dasharray="3 3"/>';
          }
        }
        out += textLabel(x0 + CP * cell + 32, y0 + r * cell + cell / 2, "发 " + sendCounts[r], 8.8, P.gatherStroke, 600);
      }
      for (var c2 = 0; c2 < CP; c2 += 1) {
        out += textLabel(x0 + c2 * cell + cell / 2, y0 + CP * cell + 16, "收 " + recvCounts[c2], 8.8, P.computeStroke, 600);
      }
      return out;
    }

    b += panel(40, 52, 500, 400, "例 1 · Full mask：全对称", "control");
    b += commMatrix(210, 150, function (r, c) { return r !== c; }, [3, 3, 3, 3], [3, 3, 3, 3]);
    b += textLabel(290, 392, "每个 rank 发 3 份、收 3 份，总量 = (CP−1)/CP × 全量 KV，", 9.6, P.muted, 500);
    b += textLabel(290, 414, "与环形轮转相同——full mask 下按依赖投递没有可省的字节。", 9.6, P.muted, 500);

    b += panel(560, 52, 500, 400, "例 2 · Causal mask：连续分片（教学假设）", "orange");
    b += commMatrix(730, 150, function (r, c) { return c > r; }, [3, 2, 1, 0], [0, 1, 2, 3]);
    b += textLabel(810, 392, "只剩上三角：发送随 rank 递减、接收随 rank 递增，通信量减半，", 9.6, P.muted, 500);
    b += textLabel(810, 414, "但计算量 ∝ rank+1——实际由 dispatch solver 重新分片抹平。", 9.6, P.muted, 500);

    b += textLabel(550, 470,
      "着色格 = 属主（行）把该段 K/V 发给接收方（列）一份；反向 dKV 的传输矩阵是它的转置：full 依旧对称，causal 变成 rank 0 只收、rank 3 只发。",
      10, P.muted, 500);
    return {
      svg: baseSvg(rootId, "cp-comm-examples", 500, b,
        "GroupCast communication matrices for full and causal masks at CP=4"),
      caption: "CP=4 的前向通信矩阵（行 = KV 属主，列 = 接收方）。左：full mask 全对称，每个 rank 发 3 份、收 3 份。右：causal mask 按 token 连续分片时只剩严格上三角——通信量减半但通信与计算都随 rank 偏斜，这正是 dispatch solver 要抹平的形态。"
    };
  }

  function auxOverlapDegreeSchedule(rootId) {
    var b = "";
    /* chunk owner tones: rank0 = orange (traced), rank1/2/3 distinct */
    var ownerTone = ["orange", "compute", "state", "control"];

    function chunkBox(x, y, w, h, label, owner) {
      return '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h +
        '" rx="7" fill="' + toneFill(ownerTone[owner]) +
        '" stroke="' + toneStroke(ownerTone[owner]) + '" stroke-width="1.1"/>' +
        textLabel(x + w / 2, y + h / 2, label, 9.4, P.ink, owner === 0 ? 700 : 500);
    }

    /* ---- Panel 1: per-beat receive schedule ---- */
    b += panel(40, 52, 1020, 352, "每一拍谁收什么 · full mask · CP=4 · overlap_degree=3", "control");
    b += textLabel(550, 96,
      "每个 rank 本地持有 2 个 chunk：rank 0 = c₀ c₁ ｜ rank 1 = c₂ c₃ ｜ rank 2 = c₄ c₅ ｜ rank 3 = c₆ c₇；", 9.8, P.muted, 500);
    b += textLabel(550, 118,
      "对任一 rank，远端共 6 个 chunk，默认按块数均分为 3 拍 × 2 chunk（颜色 = chunk 属主，橙 = rank 0 的 c₀ c₁）。", 9.8, P.muted, 500);

    var cellW = 150;
    var cellH = 54;
    var gx0 = 260;
    var gy0 = 176;
    /* recvOwner[beat][receiver] = owner rank whose 2 chunks arrive (staggered permutation) */
    var recvOwner = [
      [1, 2, 3, 0],
      [2, 3, 0, 1],
      [3, 0, 1, 2]
    ];
    for (var t = 0; t < 4; t += 1) {
      b += textLabel(gx0 + t * cellW + (cellW - 12) / 2, gy0 - 16, "rank " + t + " 收", 9.2, P.ink, 600);
    }
    for (var r = 0; r < 3; r += 1) {
      b += textLabel(gx0 - 56, gy0 + r * cellH + (cellH - 10) / 2, "拍 " + r, 9.8, P.ink, 600);
      for (var c = 0; c < 4; c += 1) {
        var owner = recvOwner[r][c];
        b += chunkBox(gx0 + c * cellW, gy0 + r * cellH, cellW - 12, cellH - 10,
          "c" + (2 * owner) + " c" + (2 * owner + 1), owner);
      }
    }
    b += textLabel(550, 356, "拍 i = 一轮 group_cast（收 stage i 的 2 个 chunk，同时预取 stage i+1）+ 一次 FFA（本地 Q 完整遍历本拍 chunk）；", 9.7, P.muted, 500);
    b += textLabel(550, 380, "partial (out, lse) 经 out_acc/lse_acc 跨拍累积：KV 全程只被遍历一次，Q 每拍重读一次。", 9.7, P.muted, 500);

    /* ---- Panel 2: degree = how many slices of the receive column ---- */
    b += panel(40, 434, 1020, 232, "overlap_degree = 把「要接收的远端 KV 列」切成几段（rank 0 视角）", "gather");

    function chunkStrip(y, groups, degLabel, beatLabelY) {
      var out = "";
      out += textLabel(140, y + 20, degLabel, 10, P.ink, 700);
      var chunks = [
        ["c₂", 1], ["c₃", 1], ["c₄", 2], ["c₅", 2], ["c₆", 3], ["c₇", 3]
      ];
      chunks.forEach(function (chunk, i) {
        out += chunkBox(230 + i * 100, y, 88, 40, chunk[0], chunk[1]);
      });
      groups.forEach(function (group, gi) {
        var x1 = 230 + group[0] * 100 - 6;
        var x2 = 230 + group[1] * 100 + 88 + 6;
        out += '<rect x="' + x1 + '" y="' + (y - 7) + '" width="' + (x2 - x1) +
          '" height="54" rx="10" fill="none" stroke="' + P.gatherStroke +
          '" stroke-width="1.3" stroke-dasharray="6 5"/>';
        out += textLabel((x1 + x2) / 2, beatLabelY, "拍 " + gi, 9, P.gatherStroke, 600);
      });
      return out;
    }

    b += chunkStrip(492, [[0, 1], [2, 3], [4, 5]], "degree=3", 476);
    b += chunkStrip(578, [[0, 2], [3, 5]], "degree=2", 562);

    b += textLabel(550, 646,
      "同一列可切成不同段数：uniform 算法按块数均分（6/3=2、6/2=3）。degree 与 CP−1 无绑定：源码默认 degree=1（远端整体一段），或 None 交给 solver 搜索。",
      9.9, P.muted, 500);
    return {
      svg: baseSvg(rootId, "overlap-degree-schedule", 680, b,
        "Per-beat receive schedule and receive-column partitions under different overlap degrees"),
      caption: "上：full mask、CP=4、chunk 粒度下 overlap_degree=3 的每拍收取表——每拍一轮 group_cast 收 2 个 chunk，FFA 用本地 Q 完整遍历本拍子集，橙色追踪 rank 0 两个 chunk 的去向。下：overlap_degree 的本义是把接收方的远端 KV 列切成几段（rank 0 视角），同一列既可切 3 段也可切 2 段——degree 与 CP−1 没有绑定关系。"
    };
  }

  function auxCpCommunication(rootId) {
    var b = "";

    /* Left: canonical Ring Attention layout (paper Fig.2) as the reference. */
    b += panel(40, 52, 490, 370, "对照 · Ring Attention 固定环形轮转", "control");
    b += box(100, 108, 150, 64, "rank 0", "Q₀ 常驻 · KV 轮转", "control", null, null, { subSize: 8.4 });
    b += box(340, 108, 150, 64, "rank 1", "Q₁ 常驻 · KV 轮转", "control", null, null, { subSize: 8.4 });
    b += box(340, 248, 150, 64, "rank 2", "Q₂ 常驻 · KV 轮转", "control", null, null, { subSize: 8.4 });
    b += box(100, 248, 150, 64, "rank 3", "Q₃ 常驻 · KV 轮转", "control", null, null, { subSize: 8.4 });
    b += edge(rootId, "M250 140H340", [295, 126, "K/V", 44, 8.4], "gather");
    b += edge(rootId, "M415 172V248", null, "gather");
    b += edge(rootId, "M340 280H250", null, "gather");
    b += edge(rootId, "M175 248V172", null, "gather");
    b += textLabel(285, 352, "每步把手上的 K/V 分片发给下一 rank，同时接收上一 rank 的分片；", 9.8, P.muted, 500);
    b += textLabel(285, 374, "CP−1 步后每个 Q 见过全部 KV——mask 用不到的分片也要绕行一圈。", 9.8, P.muted, 500);

    /* Right: MagiAttention replaces rotation with dependency-driven multicast. */
    b += panel(560, 52, 500, 370, "MagiAttention · GroupCast 按依赖多播", "gather");
    b += box(580, 170, 100, 64, "rank 0", "需要分片 c", "compute", null, null, { subSize: 8.4 });
    b += box(705, 170, 100, 64, "rank 1", "不需要 · 不收", "compute", null, null, { dashed: true, subSize: 8.4 });
    b += box(830, 170, 100, 64, "rank 2", "分片 c 属主", "control", null, null, { subSize: 8.4 });
    b += box(955, 170, 100, 64, "rank 3", "需要分片 c", "compute", null, null, { subSize: 8.4 });
    b += edge(rootId, "M880 170V118H630V170", [750, 104, "前向 GroupCast：K_c/V_c 一步多播", 250, 8.6], "gather");
    b += edge(rootId, "M930 202H955", null, "gather");
    b += edge(rootId, "M630 234V292H880V234", [748, 306, "反向 GroupReduce(sum)：dK_c/dV_c 加和回属主", 330, 8.6], "state", true);
    b += edge(rootId, "M1005 234V264H905V234", [1004, 278, "dK_c/dV_c", 88, 8.2], "state", true);
    b += textLabel(810, 352, "依赖清单来自 AttnSlice/CommMeta：dst_indices[c]=[0,3]，一步直达；", 9.8, P.muted, 500);
    b += textLabel(810, 374, "rank 1 不在清单上，一个字节也不收；Q 与 partial out/LSE 留在本地合并。", 9.8, P.muted, 500);

    b += textLabel(550, 452,
      "两者都把通信藏进计算的影子；差别在通信量：环形轮转固定传全量 KV，GroupCast 只传 mask 真正依赖的分片。",
      10.2, P.muted, 500);
    return {
      svg: baseSvg(rootId, "cp-communication", 484, b,
        "Ring Attention rotation versus MagiAttention dependency-driven GroupCast"),
      caption: "左：Ring Attention 的经典画法——Q 常驻，K/V 分片沿固定环逐步轮转。右：MagiAttention 不轮转，GroupCast 沿依赖清单把分片一步多播给需要它的 rank，反向用 GroupReduce 沿同一清单把 dK/dV 加和回属主；Q 与 partial out/LSE 始终留在本地。"
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
    backward: backwardDiagram,
    overlap: overlapDiagram
  };

  var auxes = {
    "attnslice-masktypes": auxMaskTypes,
    "tmem-map": auxTmemMap,
    "pipeline-wave": auxPipelineWave,
    "mask-segments": auxMaskSegments,
    "correction-handshake": auxCorrectionHandshake,
    "lpt-swizzle": auxLptSwizzle,
    "bwd-tmem": auxBwdTmem,
    "overlap-timeline": auxOverlapTimeline,
    "overlap-timeline-bwd": auxOverlapTimelineBwd,
    "cp-communication": auxCpCommunication,
    "cp-comm-examples": auxCpCommExamples,
    "overlap-degree-schedule": auxOverlapDegreeSchedule
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
