(function () {
  const payload = JSON.parse(document.getElementById("data").textContent);
  const data = payload.results || [];
  const cfgAnalysis = (payload.config && payload.config.analysis) || {};

  const elements = {
    tbody: document.getElementById("rows"),
    headerRow: document.getElementById("header-row"),
    summary: document.getElementById("summary"),
    loading: document.getElementById("loading"),
    loadingText: document.getElementById("loading-text"),
    tableWrap: document.getElementById("table-wrap"),
    exportBtn: document.getElementById("export"),
    lightbox: document.getElementById("lightbox"),
    lightboxImg: document.getElementById("lightbox-img"),
    lightboxPrev: document.getElementById("lightbox-prev"),
    lightboxNext: document.getElementById("lightbox-next"),
    lightboxClose: document.getElementById("lightbox-close"),
  };

  const ROW_HEIGHT_ESTIMATE = 64;
  const BUFFER_ROWS = 20;
  let renderToken = 0;
  const state = {
    sort: { key: "path", dir: "asc" },
    currentOrder: [],
    orderLookup: new Map(),
    lightboxPos: null,
    rowHeight: ROW_HEIGHT_ESTIMATE,
    rowHeightLocked: false,
    visibleRange: null,
    pendingHighlight: null,
    highlightRowId: null,
    highlightUntil: 0,
    requestRender: null,
    scrollHandler: null,
    resizeHandler: null,
  };

  const thresholds = {
    sharpnessMin: cfgAnalysis.sharpness_min,
    tenengradMin: cfgAnalysis.tenengrad_min,
    motionRatioMin: cfgAnalysis.motion_ratio_min,
    noiseStdMax: cfgAnalysis.noise_std_max,
    brightnessMin: cfgAnalysis.brightness_min,
    brightnessMax: cfgAnalysis.brightness_max,
    qualityScoreMin: cfgAnalysis.quality_score_min,
    shadowsMin: cfgAnalysis.shadows_min,
    shadowsMax: cfgAnalysis.shadows_max,
    highlightsMin: cfgAnalysis.highlights_min,
    highlightsMax: cfgAnalysis.highlights_max,
  };

  const groups = buildDuplicateGroups(data);

  const hasValue = (value) =>
    value !== undefined && value !== null && !Number.isNaN(value);

  function buildDuplicateGroups(items) {
    const map = new Map();
    items.forEach((item, idx) => {
      if (item.duplicate_group === undefined) return;
      const groupId = item.duplicate_group;
      if (!map.has(groupId)) map.set(groupId, []);
      map.get(groupId).push(idx);
    });
    return map;
  }

  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  function scoreColorFromRatio(ratio) {
    const r = Math.max(0, ratio);
    const capped = Math.min(1, Math.max(0, r - 1));
    const hue = r < 1 ? 0 : capped * 120;
    const light = r < 1 ? 18 + 28 * r : 46 + 14 * capped;
    return `hsl(${Math.round(hue)}, 75%, ${Math.round(light)}%)`;
  }

  function metricRatio(metric, value) {
    const safe = (v) => (Number.isFinite(v) ? v : null);
    switch (metric) {
      case "sharpness": {
        const min = safe(thresholds.sharpnessMin);
        if (!min || min <= 0) return null;
        return (value || 0) / min;
      }
      case "tenengrad": {
        const min = safe(thresholds.tenengradMin);
        if (!min || min <= 0) return null;
        return (value || 0) / min;
      }
      case "motion_ratio": {
        const min = safe(thresholds.motionRatioMin);
        if (!min || min <= 0) return null;
        return (value || 0) / min;
      }
      case "noise": {
        const max = safe(thresholds.noiseStdMax);
        if (!max || max <= 0) return null;
        return max / Math.max(value || 0, 1e-6);
      }
      case "brightness": {
        const lo = safe(thresholds.brightnessMin);
        const hi = safe(thresholds.brightnessMax);
        if (lo === null || hi === null || value === undefined || value === null || Number.isNaN(value)) {
          return null;
        }
        const mid = (lo + hi) / 2;
        const half = (hi - lo) / 2;
        if (half <= 0) return null;
        return clamp01(1 - Math.abs(value - mid) / Math.max(half, 1e-6));
      }
      case "quality_score": {
        const min = safe(thresholds.qualityScoreMin);
        if (!min || min <= 0) return null;
        return (value || 0) / min;
      }
      case "shadows": {
        const lo = safe(thresholds.shadowsMin) ?? 0;
        const hi = safe(thresholds.shadowsMax);
        if (hi === null || hi <= lo) return null;
        const span = Math.max(hi - lo, 1e-6);
        const mid = lo + span / 2;
        const dist = Math.abs((value ?? 0) - mid);
        const norm = clamp01(1 - dist / (span / 2));
        return 1 + norm;
      }
      case "highlights": {
        const lo = safe(thresholds.highlightsMin) ?? 0;
        const hi = safe(thresholds.highlightsMax);
        if (hi === null || hi <= lo) return null;
        const span = Math.max(hi - lo, 1e-6);
        const mid = lo + span / 2;
        const dist = Math.abs((value ?? 0) - mid);
        const norm = clamp01(1 - dist / (span / 2));
        return 1 + norm;
      }
      default:
        return null;
    }
  }

  function badgeStyle(metric, value) {
    const ratio = metricRatio(metric, value);
    if (ratio === null) return null;
    const backgroundColor = scoreColorFromRatio(ratio);
    const color = ratio > 0.65 ? "#0b1223" : "#f8fafc";
    return { backgroundColor, color };
  }

  function createBadge(text, metric, value) {
    const div = document.createElement("div");
    div.className = "metric-badge me-1";
    if (hasValue(value) && metric !== null && metric !== undefined) {
      const style = badgeStyle(metric, value);
      if (style) Object.assign(div.style, style);
    }
    div.textContent = text;
    return div;
  }

  const columns = [
    { key: "path", label: "Filename", sortable: true, type: "string" },
    {
      key: "capture",
      label: "Capture time",
      sortable: true,
      type: "number",
      getter: (item) => {
        if (Number.isFinite(item.capture_ts)) return item.capture_ts;
        const parsed = Date.parse(item.capture_time || "");
        return Number.isFinite(parsed) ? parsed : null;
      },
    },
    { key: "preview", label: "Preview", sortable: false },
    { key: "decision", label: "Decision", sortable: true, type: "string" },
    { key: "reasons", label: "Reasons", sortable: false },
    {
      key: "quality_score",
      label: "Quality score",
      sortable: true,
      type: "number",
      metric: "quality_score",
      format: (v) => v.toFixed(2),
    },
    { key: "sharpness", label: "Sharpness", sortable: true, type: "number", metric: "sharpness", format: (v) => v.toFixed(1) },
    { key: "sharpness_center", label: "Center sharpness", sortable: true, type: "number", metric: "sharpness", format: (v) => v.toFixed(1) },
    { key: "tenengrad", label: "Contrast", sortable: true, type: "number", metric: "tenengrad", format: (v) => v.toFixed(0) },
    { key: "motion_ratio", label: "Motion ratio", sortable: true, type: "number", metric: "motion_ratio", format: (v) => v.toFixed(2) },
    {
      key: "noise",
      label: "Noise",
      sortable: true,
      type: "number",
      metric: "noise",
      format: (v) => v.toFixed(1),
    },
    {
      key: "brightness",
      label: "Brightness",
      sortable: true,
      type: "number",
      metric: "brightness",
      getter: (item) => item.brightness?.mean,
      format: (v) => v.toFixed(2),
    },
    {
      key: "shadows",
      label: "Shadows",
      sortable: true,
      type: "number",
      metric: "shadows",
      getter: (item) => item.brightness?.shadows,
      format: (v) => `${(v * 100).toFixed(0)}%`,
    },
    {
      key: "highlights",
      label: "Highlights",
      sortable: true,
      type: "number",
      metric: "highlights",
      getter: (item) => item.brightness?.highlights,
      format: (v) => `${(v * 100).toFixed(0)}%`,
    },
    { key: "composition", label: "Composition", sortable: true, type: "number", format: (v) => v.toFixed(2) },
    { key: "faces", label: "Faces", sortable: true, type: "number", getter: (item) => item.faces?.count ?? 0 },
    { key: "face_sharpness", label: "Face sharpness", sortable: true, type: "number", getter: (item) => item.faces?.best_sharpness, format: (v) => v.toFixed(1) },
  ];

  const columnsByKey = Object.fromEntries(columns.map((c) => [c.key, c]));

  function buildHeader() {
    elements.headerRow.innerHTML = "";
    columns.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col.label;
      if (col.key === "path") {
        th.classList.add("filename-col");
      }
      if (col.key === "decision") {
        th.classList.add("decision-col");
      }
      if (col.key === "reasons") {
        th.classList.add("reasons-col");
      }
      if (col.sortable !== false) {
        th.classList.add("sortable");
        const indicator = document.createElement("span");
        indicator.className = "sort-indicator";
        if (state.sort.key === col.key) {
          indicator.textContent = state.sort.dir === "asc" ? "\u25B2" : "\u25BC";
        } else {
          indicator.textContent = "\u2195";
        }
        th.appendChild(indicator);
        th.onclick = () => toggleSort(col.key);
      }
      elements.headerRow.appendChild(th);
    });
  }

  function toggleSort(key) {
    const col = columnsByKey[key];
    if (!col || col.sortable === false) return;
    if (state.sort.key === key) {
      state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
    } else {
      state.sort = { key, dir: "asc" };
    }
    render();
  }

  function valueForSort(item, col) {
    if (!col) return item.path ?? "";
    if (col.getter) return col.getter(item);
    return item[col.key];
  }

  function compareValues(a, b, col) {
    const direction = state.sort.dir === "asc" ? 1 : -1;
    if (a === undefined || a === null) return (b === undefined || b === null) ? 0 : 1;
    if (b === undefined || b === null) return -1;
    if (col && col.type === "string") {
      return a.toString().localeCompare(b.toString()) * direction;
    }
    if (a > b) return 1 * direction;
    if (a < b) return -1 * direction;
    return 0;
  }

  function hasOtherKeep(groupId, excludeIdx) {
    const list = groups.get(groupId);
    if (!list) return false;
    return list.some((i) => i !== excludeIdx && data[i].decision === "keep");
  }

  function updateSummary(stats) {
    const keepPct = stats.total ? Math.round((stats.keep / stats.total) * 100) : 0;
    const discardPct = stats.total ? Math.round((stats.discard / stats.total) * 100) : 0;
    elements.summary.textContent = `Keeps: ${keepPct}% (${stats.keep}/${stats.total}) | Discards: ${discardPct}% (${stats.discard}/${stats.total})`;
  }

  function updateLoadingProgress(current, total) {
    if (!elements.loadingText) return;
    const pct = total ? Math.round((current / total) * 100) : 0;
    elements.loadingText.textContent = `Loading... ${pct}%`;
  }

  function triggerRowHighlight(row) {
    row.classList.remove("row-highlight");
    void row.offsetWidth;
    row.classList.add("row-highlight");
  }

  function scrollToRow(targetIdx) {
    const container = elements.tableWrap;
    if (!container) return;
    const targetPos = state.orderLookup.get(targetIdx);
    if (targetPos === undefined) return;
    const rowHeight = state.rowHeight || ROW_HEIGHT_ESTIMATE;
    const targetOffset = rowHeight * targetPos;
    state.pendingHighlight = targetIdx;
    state.highlightRowId = targetIdx;
    state.highlightUntil = 0;
    const header = elements.headerRow?.parentElement;
    const headerHeight = header ? header.getBoundingClientRect().height : 0;
    if (container.scrollHeight <= container.clientHeight + 1) {
      const containerTop = container.getBoundingClientRect().top + window.scrollY;
      const targetScroll = Math.max(
        0,
        containerTop + targetOffset - ((window.innerHeight - headerHeight) / 2 - rowHeight / 2)
      );
      window.scrollTo({ top: targetScroll, behavior: "smooth" });
    } else {
      const targetScroll = Math.max(
        0,
        targetOffset - ((container.clientHeight - headerHeight) / 2 - rowHeight / 2)
      );
      container.scrollTo({ top: targetScroll, behavior: "smooth" });
    }
    if (state.requestRender) state.requestRender();
  }

  function buildDuplicateBadge(item) {
    if (!item.duplicate_of) return null;
    const badge = document.createElement("div");
    badge.className = "badge bg-light text-dark border duplicate-badge";
    const dupName = item.duplicate_of.split(/[/\\]/).pop();
    const targetIdx = Number.isInteger(item.duplicate_group)
      ? item.duplicate_group
      : undefined;
    badge.appendChild(document.createTextNode("Duplicate of: "));
    if (targetIdx !== undefined) {
      const link = document.createElement("a");
      link.href = `#photo-${targetIdx}`;
      link.className = "duplicate-link";
      link.textContent = dupName;
      link.onclick = (e) => {
        e.preventDefault();
        scrollToRow(targetIdx);
      };
      badge.appendChild(link);
    } else {
      badge.appendChild(document.createTextNode(dupName));
    }
    return badge;
  }

  function buildRow(item, idx) {
    const tr = document.createElement("tr");
    tr.id = `photo-${idx}`;
    tr.dataset.row = "true";
    if (state.highlightRowId === idx && Date.now() < state.highlightUntil) {
      tr.classList.add("row-highlight");
    }

    columns.forEach((col) => {
      const td = document.createElement("td");
      switch (col.key) {
        case "preview": {
          const img = document.createElement("img");
          img.className = "preview";
          img.src = (item.preview || "").replace(/\\/g, "/");
          img.loading = "lazy";
          img.onclick = () => {
            const pos = state.orderLookup.get(idx) ?? 0;
            openLightbox(pos);
          };
          td.appendChild(img);
          break;
        }
        case "path": {
          td.classList.add("filename-col");
          const filename = (item.path || "").split(/[/\\]/).pop();
          td.textContent = filename;
          td.title = item.path || filename;
          break;
        }
        case "capture": {
          td.textContent = formatCapture(item);
          break;
        }
        case "decision": {
          const decisionClass = item.decision === "keep" ? "decision-keep" : "decision-discard";
          td.className = `decision-cell decision-col ${decisionClass}`;
          const controls = document.createElement("div");
          controls.className = "controls d-flex gap-2";
          const btnKeep = document.createElement("button");
          btnKeep.textContent = "Keep";
          btnKeep.className = "btn btn-success btn-sm";
          btnKeep.onclick = () => {
            item.decision = "keep";
            render();
          };
          const btnDrop = document.createElement("button");
          btnDrop.textContent = "Discard";
          btnDrop.className = "btn btn-danger btn-sm";
          btnDrop.onclick = () => {
            if (item.duplicate_group !== undefined && !hasOtherKeep(item.duplicate_group, idx)) {
              alert("At least one photo in a duplicate set must be kept.");
              return;
            }
            item.decision = "discard";
            render();
          };
          controls.appendChild(btnKeep);
          controls.appendChild(btnDrop);
          td.appendChild(controls);
          break;
        }
        case "reasons": {
          td.classList.add("reasons-col");
          const duplicateBadge = buildDuplicateBadge(item);
          if (duplicateBadge) {
            td.appendChild(duplicateBadge);
          }
          const reasonText = item.reasons && item.reasons.length ? item.reasons.join(", ") : "-";
          const reasonNode = document.createElement("div");
          reasonNode.textContent = reasonText;
          td.appendChild(reasonNode);
          if (reasonText !== "-") {
            td.title = reasonText;
          }
          break;
        }
        default: {
          const value = col.getter ? col.getter(item) : item[col.key];
          const present = hasValue(value);
          const display = present ? (col.format ? col.format(value, item) : value) : "n/a";
          if (col.metric) {
            td.appendChild(createBadge(display, col.metric, value));
          } else {
            td.textContent = display;
          }
        }
      }
      tr.appendChild(td);
    });

    return tr;
  }

  function formatCapture(item) {
    const iso = item.capture_time;
    if (!iso) return "n/a";
    const d = new Date(iso);
    if (!Number.isNaN(d.getTime())) {
      const y = d.getFullYear();
      const m = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      const hh = String(d.getHours()).padStart(2, "0");
      const mm = String(d.getMinutes()).padStart(2, "0");
      return `${y}-${m}-${day} ${hh}:${mm}`;
    }
    if (typeof iso === "string" && iso.length >= 16) {
      return iso.replace("T", " ").slice(0, 16);
    }
    return iso.toString();
  }

  function openLightbox(pos) {
    if (!state.currentOrder.length) return;
    if (pos < 0 || pos >= state.currentOrder.length) return;
    const idx = state.currentOrder[pos];
    const item = data[idx];
    if (!item) return;
    state.lightboxPos = pos;
    elements.lightboxImg.src = (item.preview || "").replace(/\\/g, "/");
    elements.lightbox.classList.add("open");
  }

  function closeLightbox() {
    elements.lightbox.classList.remove("open");
    state.lightboxPos = null;
  }

  function stepLightbox(delta) {
    if (state.lightboxPos === null) return;
    const nextPos = (state.lightboxPos + delta + state.currentOrder.length) % state.currentOrder.length;
    openLightbox(nextPos);
  }

  function bindLightbox() {
    elements.lightbox.onclick = (e) => {
      if (e.target === elements.lightbox) closeLightbox();
    };
    elements.lightboxPrev.onclick = (e) => {
      e.stopPropagation();
      stepLightbox(-1);
    };
    elements.lightboxNext.onclick = (e) => {
      e.stopPropagation();
      stepLightbox(1);
    };
    elements.lightboxClose.onclick = (e) => {
      e.stopPropagation();
      closeLightbox();
    };
  }

  function render() {
    const token = ++renderToken;
    buildHeader();
    const colForSort = columnsByKey[state.sort.key];

    const sortedIdx = data
      .map((_, idx) => idx)
      .sort((a, b) => {
        const av = valueForSort(data[a], colForSort);
        const bv = valueForSort(data[b], colForSort);
        const cmp = compareValues(av, bv, colForSort);
        if (cmp !== 0) return cmp;
        return (data[a].path ?? "").localeCompare(data[b].path ?? "");
      });

    const stats = { total: data.length, keep: 0, discard: 0 };
    data.forEach((item) => {
      if (item.decision === "keep") {
        stats.keep += 1;
      } else {
        stats.discard += 1;
      }
    });

    state.currentOrder = sortedIdx;
    state.orderLookup = new Map(sortedIdx.map((id, pos) => [id, pos]));
    state.lightboxPos = null;
    state.visibleRange = null;
    const container = elements.tableWrap || elements.tbody.parentElement;
    if (!container) return;
    const useWindowScroll = container.scrollHeight <= container.clientHeight + 1;

    let scheduled = false;
    let initialPaint = false;
    const requestRender = () => {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        renderVisible();
      });
    };
    state.requestRender = requestRender;

    state.scrollHandler = () => requestRender();
    if (useWindowScroll) {
      if (state.scrollHandler) container.removeEventListener("scroll", state.scrollHandler);
      if (state.scrollHandler) window.removeEventListener("scroll", state.scrollHandler);
      window.addEventListener("scroll", state.scrollHandler, { passive: true });
    } else {
      if (state.scrollHandler) window.removeEventListener("scroll", state.scrollHandler);
      if (state.scrollHandler) container.removeEventListener("scroll", state.scrollHandler);
      container.addEventListener("scroll", state.scrollHandler, { passive: true });
    }

    if (state.resizeHandler) window.removeEventListener("resize", state.resizeHandler);
    state.resizeHandler = () => requestRender();
    window.addEventListener("resize", state.resizeHandler);

    const spacerRow = (height) => {
      const tr = document.createElement("tr");
      tr.className = "spacer-row";
      const td = document.createElement("td");
      td.colSpan = columns.length;
      td.style.height = `${Math.max(0, height)}px`;
      tr.appendChild(td);
      return tr;
    };

    const getScrollMetrics = () => {
      if (!useWindowScroll) {
        return {
          scrollTop: container.scrollTop || 0,
          viewportHeight: container.clientHeight || 600,
        };
      }
      const containerTop = container.getBoundingClientRect().top + window.scrollY;
      const scrollTop = Math.max(0, window.scrollY - containerTop);
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 600;
      return { scrollTop, viewportHeight };
    };

    const renderVisible = () => {
      if (token !== renderToken) return;
      const rowHeight = state.rowHeight || ROW_HEIGHT_ESTIMATE;
      const metrics = getScrollMetrics();
      const viewportHeight = Math.max(metrics.viewportHeight, rowHeight);
      const scrollTop = Math.max(0, metrics.scrollTop);
      const visibleCount = Math.ceil(viewportHeight / rowHeight);
      let startIdx = Math.floor(scrollTop / rowHeight) - BUFFER_ROWS;
      if (startIdx < 0) startIdx = 0;
      const endIdx = Math.min(sortedIdx.length, startIdx + visibleCount + BUFFER_ROWS * 2);
      const sameRange =
        state.visibleRange &&
        state.visibleRange.start === startIdx &&
        state.visibleRange.end === endIdx;

      if (sameRange) {
        if (state.pendingHighlight !== null) {
          const row = document.getElementById(`photo-${state.pendingHighlight}`);
          if (row) {
            state.highlightUntil = Date.now() + 1200;
            triggerRowHighlight(row);
            state.pendingHighlight = null;
          }
        }
        return;
      }
      state.visibleRange = { start: startIdx, end: endIdx };

      const fragment = document.createDocumentFragment();
      fragment.appendChild(spacerRow(startIdx * rowHeight));
      for (let i = startIdx; i < endIdx; i += 1) {
        const idx = sortedIdx[i];
        fragment.appendChild(buildRow(data[idx], idx));
      }
      fragment.appendChild(spacerRow((sortedIdx.length - endIdx) * rowHeight));
      elements.tbody.replaceChildren(fragment);

      const firstRow = elements.tbody.querySelector("tr[data-row]");
      if (firstRow) {
        const measured = Math.round(firstRow.getBoundingClientRect().height);
        if (!state.rowHeightLocked && measured > 0 && measured !== state.rowHeight) {
          state.rowHeight = measured;
          state.rowHeightLocked = true;
          requestRender();
          return;
        }
      }

      if (state.pendingHighlight !== null) {
        const row = document.getElementById(`photo-${state.pendingHighlight}`);
        if (row) {
          state.highlightUntil = Date.now() + 1200;
          triggerRowHighlight(row);
          state.pendingHighlight = null;
        }
      }

      if (!initialPaint) {
        initialPaint = true;
        updateSummary(stats);
        updateLoadingProgress(sortedIdx.length, sortedIdx.length);
        if (elements.loading && elements.loading.parentNode) {
          elements.loading.parentNode.removeChild(elements.loading);
        }
      }
    };

    elements.tbody.replaceChildren();
    requestRender();
  }

  function bindExport() {
    if (!elements.exportBtn) return;
    elements.exportBtn.onclick = () => {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "decisions.json";
      a.click();
      URL.revokeObjectURL(url);
    };
  }

  bindLightbox();
  bindExport();
  render();
})();
