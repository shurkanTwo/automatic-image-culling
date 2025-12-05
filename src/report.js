(function () {
  const payload = JSON.parse(document.getElementById("data").textContent);
  const data = payload.results || [];
  const cfgAnalysis = (payload.config && payload.config.analysis) || {};

  const elements = {
    tbody: document.getElementById("rows"),
    headerRow: document.getElementById("header-row"),
    summary: document.getElementById("summary"),
    loading: document.getElementById("loading"),
    exportBtn: document.getElementById("export"),
    lightbox: document.getElementById("lightbox"),
    lightboxImg: document.getElementById("lightbox-img"),
    lightboxPrev: document.getElementById("lightbox-prev"),
    lightboxNext: document.getElementById("lightbox-next"),
    lightboxClose: document.getElementById("lightbox-close"),
  };

  const state = {
    sort: { key: "path", dir: "asc" },
    currentOrder: [],
    orderLookup: new Map(),
    lightboxPos: null,
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
    const hasValue = value !== undefined && value !== null && !Number.isNaN(value);
    if (hasValue && metric !== null && metric !== undefined) {
      const style = badgeStyle(metric, value);
      if (style) Object.assign(div.style, style);
    }
    div.textContent = text;
    return div;
  }

  const columns = [
    { key: "path", label: "Filename", sortable: true, type: "string" },
    { key: "capture", label: "Capture time", sortable: true, type: "string" },
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
    buildHeader();
    const fragment = document.createDocumentFragment();
    const stats = { total: data.length, keep: 0, discard: 0 };
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

    state.currentOrder = sortedIdx;
    state.orderLookup = new Map(sortedIdx.map((id, pos) => [id, pos]));
    state.lightboxPos = null;

    sortedIdx.forEach((idx) => {
      const item = data[idx];
      const tr = document.createElement("tr");

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
            const filename = (item.path || "").split(/[/\\]/).pop();
            td.textContent = filename;
            if (item.duplicate_of) {
              const badge = document.createElement("div");
              badge.className = "badge bg-light text-dark border";
              const dupName = item.duplicate_of.split(/[/\\]/).pop();
              badge.textContent = "Duplicate of: " + dupName;
              td.appendChild(badge);
            }
            break;
          }
          case "capture": {
            td.textContent = formatCapture(item);
            break;
          }
          case "decision": {
            const decisionClass = item.decision === "keep" ? "decision-keep" : "decision-discard";
            td.className = `decision-cell ${decisionClass}`;
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
            if (item.reasons && item.reasons.length) {
              td.textContent = item.reasons.join(", ");
            } else {
              td.textContent = "-";
            }
            break;
          }
          default: {
            const value = col.getter ? col.getter(item) : item[col.key];
            const hasValue = value !== undefined && value !== null && !Number.isNaN(value);
            const display = hasValue ? (col.format ? col.format(value, item) : value) : "n/a";
            if (col.metric) {
              td.appendChild(createBadge(display, col.metric, value));
            } else {
              td.textContent = display;
            }
          }
        }
        tr.appendChild(td);
      });
      fragment.appendChild(tr);
      if (item.decision === "keep") {
        stats.keep += 1;
      } else {
        stats.discard += 1;
      }
    });

    elements.tbody.replaceChildren(fragment);

    const keepPct = stats.total ? Math.round((stats.keep / stats.total) * 100) : 0;
    const discardPct = stats.total ? Math.round((stats.discard / stats.total) * 100) : 0;
    elements.summary.textContent = `Keeps: ${keepPct}% (${stats.keep}/${stats.total}) | Discards: ${discardPct}% (${stats.discard}/${stats.total})`;

    if (elements.loading && elements.loading.parentNode) {
      elements.loading.parentNode.removeChild(elements.loading);
    }
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


