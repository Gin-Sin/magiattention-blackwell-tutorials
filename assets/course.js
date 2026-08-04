(function () {
  "use strict";

  var chapters = window.MAGI_CHAPTERS || [];
  var codeMap = window.MAGI_CODE || {};
  var STORAGE_KEY = "magi_blackwell_completed";
  var REPO_BLOB = "https://github.com/SandAI-org/MagiAttention/blob/main/";

  function esc(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /* Chapter bodies carry hand-written inline HTML (strong/code/KaTeX), so
     they are inserted as-is; plain metadata still goes through esc(). */
  function raw(value) {
    return value == null ? "" : String(value);
  }

  function readProgress() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    } catch (_) {
      return [];
    }
  }

  function writeProgress(ids) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
    } catch (_) {
      /* Local files or privacy mode may disable storage. */
    }
  }

  function renderMath(scope) {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(scope || document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false,
      strict: "ignore"
    });
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      try {
        if (!document.execCommand("copy")) throw new Error("copy failed");
        resolve();
      } catch (error) {
        reject(error);
      } finally {
        document.body.removeChild(textarea);
      }
    });
  }

  /* ---------------- code workbench ---------------- */

  function sourceUrl(path, block) {
    var url = REPO_BLOB + ((block && block.path) || path);
    if (block) url += "#L" + block.start + "-L" + block.end;
    return url;
  }

  function renderCodePane(impl) {
    if (!impl || !(impl.blocks || []).length) {
      return '<aside class="code-ide code-ide--unavailable" role="status">' +
        "<strong>交互源码未能载入</strong>" +
        "<p>架构图仍可阅读；请检查 assets/code.js 是否可访问。</p></aside>";
    }
    var blocks = impl.blocks;
    var initial = blocks[0];
    var options = blocks.map(function (block) {
      return '<option value="' + esc(block.id) + '"' +
        (block.id === initial.id ? " selected" : "") + ">Block " + esc(block.id) +
        " · " + esc(block.title) + "</option>";
    }).join("");
    return '<aside class="code-ide" data-architecture-ide tabindex="-1" aria-label="Kernel source workbench">' +
      '<div class="code-ide__titlebar"><span class="code-ide__traffic" aria-hidden="true">' +
      '<i></i><i></i><i></i></span><a class="code-ide__meta" data-workbench-srclink target="_blank" rel="noreferrer" href="' +
      esc(sourceUrl(impl.path, initial)) + '" title="在 GitHub 上查看源码">' + esc(impl.path) + "</a>" +
      '<span class="code-ide__icons">' +
      '<button class="code-ide__icon" type="button" data-workbench-copy title="复制当前代码" aria-label="复制当前代码">⧉</button>' +
      "</span></div>" +
      '<div class="code-ide__toolbar">' +
      '<select class="code-ide__jump" data-workbench-select aria-label="选择实现代码块" title="跳转到指定代码块">' +
      options + '</select><span class="code-ide__lines" data-workbench-lines>Lines ' +
      esc(initial.start) + "–" + esc(initial.end) + "</span></div>" +
      '<p class="code-ide__status" data-workbench-status aria-live="polite">' +
      "点击架构图节点，或从列表跳转到对应源码块。行号对应仓库真实源码。</p>" +
      '<pre class="code-ide__editor language-python line-numbers" data-workbench-pre data-start="' +
      esc(initial.start) + '"><code class="language-python" data-workbench-editor>' +
      esc(initial.code) + "</code></pre></aside>";
  }

  function renderDiagram(config, impl) {
    if (!window.MagiDiagrams) {
      return '<section class="warning diagram-load-failure" role="alert">' +
        "<strong>架构图加载失败</strong><p>请检查 assets/diagrams.js 是否可访问。</p></section>";
    }
    var report = window.MagiDiagrams.build(config.key);
    var badges = report.badges.map(function (badge) {
      return "<span>" + esc(badge) + "</span>";
    }).join("");
    var notes = report.notes.map(function (note, index) {
      return '<li><span class="diagram-guide__num">' + String(index + 1).padStart(2, "0") +
        '</span><div><strong>' + esc(note[0]) + "</strong><p>" + esc(note[1]) + "</p></div></li>";
    }).join("");
    return '<figure class="report-figure report-figure--workbench" id="architecture-block">' +
      '<div class="diagram-header"><div><span class="diagram-header__eyebrow">Kernel Deconstruction</span>' +
      "<strong>" + esc(report.title) + '</strong></div><div class="diagram-header__tools"><div class="diagram-legend">' +
      badges + '</div><button class="diagram-expand" type="button" data-diagram-expand aria-expanded="false">⤢ 放大查看</button></div></div>' +
      '<div class="architecture-workbench"><div class="architecture-pane"><div class="diagram diagram--report"><div class="diagram-canvas">' +
      report.svg + "</div></div></div>" + renderCodePane(impl) +
      '</div><figcaption class="figcaption">' + esc(config.caption) +
      '</figcaption><aside class="diagram-memory"><span>One-line Memory · 一眼记住</span><p>' +
      esc(report.memory) + '</p></aside><ol class="diagram-guide">' + notes + "</ol></figure>";
  }

  function initDiagramExpand(scope) {
    var figure = scope.querySelector(".report-figure");
    var button = scope.querySelector("[data-diagram-expand]");
    if (!figure || !button) return;
    function setExpanded(expanded) {
      figure.classList.toggle("is-expanded", expanded);
      document.body.classList.toggle("diagram-is-open", expanded);
      button.setAttribute("aria-expanded", expanded ? "true" : "false");
      button.textContent = expanded ? "× 关闭大图" : "⤢ 放大查看";
    }
    button.addEventListener("click", function () {
      setExpanded(!figure.classList.contains("is-expanded"));
      window.dispatchEvent(new Event("resize"));
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && figure.classList.contains("is-expanded")) {
        setExpanded(false);
        window.dispatchEvent(new Event("resize"));
      }
    });
  }

  function initWorkbenchHeightSync(scope) {
    var workbench = scope.querySelector(".architecture-workbench");
    if (!workbench) return;
    var pane = workbench.querySelector(".architecture-pane");
    var ide = workbench.querySelector(".code-ide");
    if (!pane || !ide) return;
    function apply() {
      workbench.classList.remove("is-height-synced");
      workbench.style.removeProperty("--workbench-height");
      if (!window.matchMedia("(min-width: 1500px)").matches) return;
      var unified = pane.offsetHeight;
      if (!isFinite(unified) || unified < 200) return;
      workbench.style.setProperty("--workbench-height", unified + "px");
      workbench.classList.add("is-height-synced");
    }
    var pending = 0;
    function schedule() {
      window.cancelAnimationFrame(pending);
      pending = window.requestAnimationFrame(apply);
    }
    window.addEventListener("resize", schedule);
    window.addEventListener("load", schedule);
    window.setTimeout(schedule, 400);
    apply();
  }

  function initWorkbench(scope, impl) {
    var workbench = scope.querySelector(".architecture-workbench");
    var ide = scope.querySelector("[data-architecture-ide]");
    if (!workbench || !ide || !impl || !(impl.blocks || []).length) return;

    var byId = {};
    impl.blocks.forEach(function (block) { byId[block.id] = block; });

    var select = ide.querySelector("[data-workbench-select]");
    var copyButton = ide.querySelector("[data-workbench-copy]");
    var lines = ide.querySelector("[data-workbench-lines]");
    var status = ide.querySelector("[data-workbench-status]");
    var pre = ide.querySelector("[data-workbench-pre]");
    var code = ide.querySelector("[data-workbench-editor]");
    var srcLink = ide.querySelector("[data-workbench-srclink]");
    var diagramNodes = workbench.querySelectorAll("svg [data-code-block]");
    var currentId = impl.blocks[0].id;
    var displayedCode = impl.blocks[0].code;

    function normalizeId(value) {
      var id = String(value || "").trim();
      return /^\d+$/.test(id) ? id.padStart(2, "0") : id;
    }

    function highlightCode() {
      code.textContent = displayedCode;
      if (!window.Prism || typeof window.Prism.highlightElement !== "function") return;
      try {
        window.Prism.highlightElement(code);
      } catch (_) {
        code.textContent = displayedCode;
      }
    }

    function syncDiagramNodes(sourceNode) {
      var primary = sourceNode || null;
      diagramNodes.forEach(function (node) {
        var matches = normalizeId(node.getAttribute("data-code-block")) === currentId;
        node.classList.remove("is-code-active", "is-code-related");
        node.setAttribute("aria-pressed", "false");
        if (!matches) return;
        if (!primary) primary = node;
        if (node !== primary) node.classList.add("is-code-related");
      });
      if (primary) {
        primary.classList.add("is-code-active");
        primary.setAttribute("aria-pressed", "true");
      }
    }

    function revealIdeIfNeeded() {
      var rect = ide.getBoundingClientRect();
      if (rect.top >= 0 && rect.bottom <= window.innerHeight) return;
      try {
        ide.focus({ preventScroll: true });
      } catch (_) {
        ide.focus();
      }
      var reduceMotion = window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      ide.scrollIntoView({
        behavior: reduceMotion ? "auto" : "smooth",
        block: "nearest",
        inline: "nearest"
      });
    }

    function selectBlock(id, options) {
      var normalized = normalizeId(id);
      var block = byId[normalized];
      if (!block) return;
      currentId = normalized;
      select.value = currentId;
      lines.textContent = (block.path ? block.path.split("/").pop() + " · " : "") +
        "Lines " + block.start + "–" + block.end;
      displayedCode = block.code;
      pre.setAttribute("data-start", block.start);
      pre.scrollTop = 0;
      if (srcLink) srcLink.setAttribute("href", sourceUrl(impl.path, block));
      syncDiagramNodes(options && options.sourceNode);
      highlightCode();
      status.textContent = "Block " + currentId + " · " + block.title +
        " · 源码行 " + block.start + "–" + block.end + "（点击文件名跳转 GitHub）。";
      if (options && options.reveal) revealIdeIfNeeded();
    }

    diagramNodes.forEach(function (node) {
      var id = normalizeId(node.getAttribute("data-code-block"));
      var block = byId[id];
      if (!block) return;
      node.setAttribute("tabindex", "0");
      node.setAttribute("role", "button");
      node.setAttribute("aria-label", "查看代码块 " + id + "：" + block.title);
      node.addEventListener("click", function () {
        selectBlock(id, { reveal: true, sourceNode: node });
      });
      node.addEventListener("keydown", function (event) {
        if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") return;
        event.preventDefault();
        selectBlock(id, { reveal: true, sourceNode: node });
      });
    });

    select.addEventListener("change", function () {
      selectBlock(select.value);
    });
    copyButton.addEventListener("click", function () {
      copyText(displayedCode).then(function () {
        copyButton.textContent = "✓";
        status.textContent = "当前代码块已复制。";
      }).catch(function () {
        copyButton.textContent = "✕";
        status.textContent = "复制失败，请手动选择代码。";
      }).finally(function () {
        window.setTimeout(function () { copyButton.textContent = "⧉"; }, 1500);
      });
    });

    selectBlock(currentId);
  }

  /* ---------------- chapter sections ---------------- */

  function renderCards(items, className) {
    return '<div class="' + className + '-grid">' + items.map(function (item) {
      return '<article class="' + className + '"><span class="label">' + esc(item.label) +
        "</span><strong>" + raw(item.title) + "</strong><p>" + raw(item.body) + "</p></article>";
    }).join("") + "</div>";
  }

  /* A body item may be a plain paragraph string, or an info-card object:
     { card: true, tone?: "recipe"|"source"|"fact", label?, title?, body: string|string[] } */
  function renderBodyItem(item) {
    if (item && typeof item === "object" && item.card) {
      var cardBody = Array.isArray(item.body) ? item.body : [item.body];
      return '<aside class="info-card info-card--' + esc(item.tone || "note") + '">' +
        (item.label ? '<span class="info-card__label">' + esc(item.label) + "</span>" : "") +
        (item.title ? '<strong class="info-card__title">' + raw(item.title) + "</strong>" : "") +
        cardBody.map(function (paragraph) {
          return '<p class="info-card__body">' + raw(paragraph) + "</p>";
        }).join("") +
        "</aside>";
    }
    return "<p>" + raw(item) + "</p>";
  }

  function renderExplainSections(chapter) {
    return (chapter.explain || []).map(function (section, index) {
      var body = (section.body || []).map(renderBodyItem).join("");
      var figure = "";
      if (section.svg && window.MagiDiagrams) {
        var aux = window.MagiDiagrams.buildAux(section.svg);
        if (aux) {
          figure = '<figure class="report-figure report-figure--aux"><div class="diagram diagram--report"><div class="diagram-canvas">' +
            aux.svg + '</div></div><figcaption class="figcaption">' + esc(aux.caption) +
            "</figcaption></figure>";
        }
      }
      var formula = section.formula
        ? '<article class="formula"><span class="formula-label">Math · 数学推导</span><div>' +
          raw(section.formula) + "</div></article>"
        : "";
      return '<section class="explain-section" id="sec-explain-' + index + '">' +
        '<h3 class="explain-title"><span>' + String(index + 1).padStart(2, "0") +
        "</span>" + esc(section.title) + "</h3>" + body + figure + formula + "</section>";
    }).join("");
  }

  function renderExercises(chapter) {
    return chapter.exercises.map(function (exercise, index) {
      return '<article class="exercise"><div class="diagram-legend" aria-label="练习分类与难度"><span>Kind · ' +
        esc(exercise.kind) + "</span><span>Level · " + esc(exercise.level) +
        "</span></div><h3>练习 " + (index + 1) + '</h3><div class="exercise-body"><p>' +
        raw(exercise.q) + "</p><details><summary>提示</summary><p>" + raw(exercise.hint) +
        "</p></details><details><summary>答案</summary><p>" + raw(exercise.answer) +
        "</p></details></div></article>";
    }).join("");
  }

  var TOC_STORAGE_KEY = "magi_blackwell_toc_collapsed";

  function initChapterToc(scope) {
    var toc = scope.querySelector(".chapter-toc");
    if (!toc) return;

    /* Floating panel collapse: panel <-> edge fab. */
    var fab = scope.querySelector("[data-toc-expand]");
    var collapseButton = toc.querySelector("[data-toc-collapse]");
    if (fab && collapseButton) {
      var readCollapsed = function () {
        try {
          return localStorage.getItem(TOC_STORAGE_KEY);
        } catch (_) {
          return null;
        }
      };
      var setCollapsed = function (collapsed, persist, moveFocus) {
        toc.classList.toggle("is-collapsed", collapsed);
        fab.hidden = !collapsed;
        if (persist) {
          try {
            localStorage.setItem(TOC_STORAGE_KEY, collapsed ? "1" : "0");
          } catch (_) {
            /* Storage may be unavailable in privacy mode. */
          }
        }
        if (moveFocus) {
          try {
            (collapsed ? fab : toc).focus({ preventScroll: true });
          } catch (_) {
            (collapsed ? fab : toc).focus();
          }
        }
      };
      var stored = readCollapsed();
      var wide = window.matchMedia("(min-width: 1360px)").matches;
      setCollapsed(stored == null ? !wide : stored === "1", false, false);
      collapseButton.addEventListener("click", function () {
        setCollapsed(true, true, true);
      });
      fab.addEventListener("click", function () {
        setCollapsed(false, true, true);
      });
      document.addEventListener("keydown", function (event) {
        if (event.key !== "Escape") return;
        if (toc.classList.contains("is-collapsed")) return;
        if (!toc.contains(document.activeElement)) return;
        setCollapsed(true, true, true);
      });
    }

    var links = Array.prototype.slice.call(toc.querySelectorAll("[data-toc-link]"));
    var pairs = links.map(function (link) {
      var id = link.getAttribute("href").slice(1);
      return { link: link, target: document.getElementById(id) };
    }).filter(function (pair) {
      return pair.target;
    });
    if (!pairs.length) return;

    function setActive(activeLink) {
      toc.querySelectorAll(".chapter-toc__item").forEach(function (item) {
        item.classList.remove("is-ancestor");
      });
      links.forEach(function (link) {
        var active = link === activeLink;
        link.classList.toggle("is-active", active);
        if (active) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
      var item = activeLink.parentElement;
      if (item && item.classList.contains("chapter-toc__item--child")) {
        var ancestor = item.parentElement && item.parentElement.parentElement;
        if (ancestor) ancestor.classList.add("is-ancestor");
      }
    }

    function update() {
      var marker = window.innerHeight * 0.28;
      var active = pairs[0].link;
      pairs.forEach(function (pair) {
        if (pair.target.getBoundingClientRect().top <= marker) active = pair.link;
      });
      if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4) {
        active = pairs[pairs.length - 1].link;
      }
      setActive(active);
    }

    var pending = 0;
    function schedule() {
      if (pending) return;
      pending = window.requestAnimationFrame(function () {
        pending = 0;
        update();
      });
    }

    if ("IntersectionObserver" in window) {
      var observer = new IntersectionObserver(schedule, {
        rootMargin: "-20% 0px -70% 0px",
        threshold: [0, 1]
      });
      pairs.forEach(function (pair) { observer.observe(pair.target); });
    }
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    update();
  }

  function renderChapter() {
    var root = document.getElementById("chapter-root");
    if (!root) return;
    var id = new URLSearchParams(window.location.search).get("id") || chapters[0].id;
    var index = chapters.findIndex(function (chapter) { return chapter.id === id; });
    if (index < 0) index = 0;
    var c = chapters[index];
    document.title = c.title + " · FFA on Blackwell";
    document.body.classList.add(c.category || "dense");

    var impl = codeMap[c.id];
    var motivation = c.motivation.map(function (paragraph) {
      return "<p>" + raw(paragraph) + "</p>";
    }).join("");
    var sources = c.sources.map(function (source) {
      return '<li><span><a href="' + esc(source.url) +
        '" target="_blank" rel="noreferrer">' + esc(source.label) + "</a></span></li>";
    }).join("");

    var tocEntries = [
      ["01", "直觉 Takeaway", "sec-takeaway"],
      ["02", "问题从哪里来", "sec-motivation"],
      ["03", "架构图与交互源码", "architecture-block"],
      ["04", "深入解析", "sec-explain"]
    ];
    (c.explain || []).forEach(function (section, sectionIndex) {
      tocEntries.push([
        "04." + (sectionIndex + 1),
        section.title,
        "sec-explain-" + sectionIndex,
        "sec-explain"
      ]);
    });
    tocEntries.push(["05", "练习与答案", "sec-exercises"]);
    tocEntries.push(["06", "参考来源", "sec-sources"]);
    var topEntries = tocEntries.filter(function (entry) { return !entry[3]; });
    var toc = '<nav class="chapter-toc" aria-label="本章目录" tabindex="-1">' +
      '<div class="chapter-toc__head"><span class="chapter-toc__label">Contents · 本章目录</span>' +
      '<button class="chapter-toc__toggle" type="button" data-toc-collapse aria-label="折叠目录" title="折叠目录">✕</button></div>' +
      '<ol class="chapter-toc__list">' + topEntries.map(function (entry) {
        var children = tocEntries.filter(function (candidate) {
          return candidate[3] === entry[2];
        });
        return '<li class="chapter-toc__item"><a class="chapter-toc__link" data-toc-link href="#' +
          entry[2] + '"><i aria-hidden="true">' + entry[0] + '</i><span>' + esc(entry[1]) + "</span></a>" +
          (children.length
            ? '<ol class="chapter-toc__sublist">' + children.map(function (child) {
              return '<li class="chapter-toc__item chapter-toc__item--child"><a class="chapter-toc__link" data-toc-link href="#' +
                child[2] + '"><i aria-hidden="true">' + child[0] + '</i><span>' +
                esc(child[1]) + "</span></a></li>";
            }).join("") + "</ol>"
            : "") + "</li>";
      }).join("") + "</ol></nav>" +
      '<button class="chapter-toc-fab" type="button" data-toc-expand aria-label="展开目录" title="展开目录" hidden>' +
      '<span aria-hidden="true">☰</span><i>目 录</i></button>';

    var prev = chapters[index - 1];
    var next = chapters[index + 1];
    var nav = '<nav class="chapter-nav">' +
      (prev
        ? '<a href="?id=' + prev.id + '"><small>上一章</small>' + esc(prev.title + " · " + prev.zhTitle) + "</a>"
        : "<span></span>") +
      (next
        ? '<a class="next" href="?id=' + next.id + '"><small>下一章</small>' + esc(next.title + " · " + next.zhTitle) + "</a>"
        : '<a class="next" href="index.html"><small>课程完成</small>回到课程地图</a>') +
      "</nav>";

    root.innerHTML =
      '<nav class="breadcrumbs"><a href="index.html">FFA on Blackwell</a> &nbsp;/&nbsp; Chapter ' +
      String(c.order).padStart(2, "0") + " &nbsp;/&nbsp; " + esc(c.title) + "</nav>" +
      '<header class="chapter-hero"><p class="eyebrow">Chapter ' + String(c.order).padStart(2, "0") +
      " · " + esc(c.fullTitle) + "</p><h1>" + esc(c.title) + '</h1><p class="chapter-deck">' +
      esc(c.zhTitle) + "。 " + raw(c.deck) + '</p><div class="chapter-meta"><b>' + esc(c.tag) +
      "</b><span>难度 · " + esc(c.difficulty) + "</span><span>" + esc(c.source) + "</span></div></header>" +
      '<aside class="takeaway" id="sec-takeaway"><span>Intuition Takeaway · 直觉要点</span><p>' +
      raw(c.takeaway) + "</p></aside>" +
      renderCards(c.intuitions, "intuition") + toc +
      '<main class="chapter-main">' +
      '<h2 data-no="02" id="sec-motivation">问题从哪里来</h2>' + motivation +
      '<h2 data-no="03" id="sec-architecture">架构图与交互源码</h2>' + renderDiagram(c.diagram, impl) +
      '<h2 data-no="04" id="sec-explain">深入解析</h2>' + renderExplainSections(c) +
      '<div class="warning"><strong>边界与误区：</strong> ' + raw(c.warning) + "</div>" +
      '<h2 data-no="05" id="sec-exercises">练习与答案</h2>' + renderExercises(c) +
      '<h2 data-no="06" id="sec-sources">参考来源</h2><ol class="source-list">' + sources + "</ol>" +
      '<button class="button" id="complete-chapter" type="button">标记本章完成</button>' +
      nav + "</main>";

    var done = readProgress();
    var button = document.getElementById("complete-chapter");
    function syncButton() {
      var completed = done.indexOf(c.id) >= 0;
      button.textContent = completed ? "✓ 已完成 · 点击撤销" : "标记本章完成";
      button.classList.toggle("primary", completed);
    }
    button.addEventListener("click", function () {
      var at = done.indexOf(c.id);
      if (at >= 0) done.splice(at, 1);
      else done.push(c.id);
      writeProgress(done);
      syncButton();
    });
    syncButton();
    initDiagramExpand(root);
    initWorkbench(root, impl);
    renderMath(root);
    initWorkbenchHeightSync(root);
    initChapterToc(root);
  }

  function populateHome() {
    var grid = document.getElementById("chapter-grid");
    if (grid) {
      grid.innerHTML = chapters.map(function (c) {
        return '<a class="chapter-card ' + (c.category || "dense") + '" href="chapter.html?id=' + c.id +
          '"><span class="card-kicker">Chapter ' + String(c.order).padStart(2, "0") +
          '</span><span class="year">' + esc(c.tag) + "</span><h3>" + esc(c.title) +
          "</h3><p>" + esc(c.zhTitle) + '</p><span class="arrow">↗</span></a>';
      }).join("");
    }
    var done = readProgress();
    document.querySelectorAll("[data-progress-count]").forEach(function (el) {
      el.textContent = done.length + " / " + chapters.length;
    });
    document.querySelectorAll("[data-chapter-count]").forEach(function (el) {
      el.textContent = chapters.length;
    });
    var exerciseTotal = chapters.reduce(function (sum, chapter) {
      return sum + chapter.exercises.length;
    }, 0);
    document.querySelectorAll("[data-exercise-count]").forEach(function (el) {
      el.textContent = exerciseTotal;
    });
    renderMath(document.body);
  }

  window.addEventListener("DOMContentLoaded", function () {
    renderChapter();
    populateHome();
  });
})();
