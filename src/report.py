import json
import pathlib
from typing import Dict, List


def write_html_report(results: List[Dict], path: pathlib.Path) -> None:
    data_json = json.dumps(results)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ARW Analysis Report</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QL6gUmHUh3Y0P7A5YkT2sHPyOdrFblOaUkL81P+GqjI7OU15m5uW2li31QnQ+8bN" crossorigin="anonymous">
  <style>
    body {{ background: #f6f7fb; color: #1f2937; }}
    img.preview {{ max-width: 200px; height: auto; border-radius: 6px; cursor: zoom-in; }}
    .table-alt tbody tr:nth-child(odd) {{ background-color: #f4f7fb; }}
    .table-alt tbody tr:nth-child(even) {{ background-color: #ffffff; }}
    .table-alt tbody tr:hover {{ background-color: #e8f0ff; }}
    .table-alt td, .table-alt th {{ text-align: center; vertical-align: middle; padding: 0.75rem 1rem; }}
    .decision-cell {{ font-weight: 600; }}
    .decision-keep {{ background-color: #d4edda; color: #0f5132; }}
    .decision-discard {{ background-color: #f8d7da; color: #842029; }}
    .content-wrap {{ padding: 2rem 2.5rem; }}
    .lightbox {{
      position: fixed; inset: 0; background: rgba(0,0,0,0.8);
      display: flex; align-items: center; justify-content: center;
      opacity: 0; pointer-events: none; transition: opacity 0.2s ease;
      z-index: 1000;
    }}
    .lightbox.open {{ opacity: 1; pointer-events: all; }}
    .lightbox img {{ max-width: 95vw; max-height: 95vh; border-radius: 6px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
    #lightbox-prev, #lightbox-next {{
      position: absolute;
      top: 50%;
      transform: translateY(-50%);
      opacity: 0.9;
    }}
    #lightbox-prev {{ left: 16px; }}
    #lightbox-next {{ right: 16px; }}
    .controls {{ display: flex; gap: 0.4rem; align-items: center; }}
    .lightbox-nav {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      pointer-events: none;
      padding: 0 1rem;
    }}
    .lightbox-nav button {{
      pointer-events: auto;
      opacity: 0.9;
    }}
    .lightbox-nav {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      pointer-events: none;
      padding: 0 1rem;
    }}
    .lightbox-nav button {{
      pointer-events: auto;
      opacity: 0.85;
    }}
    th.sortable {{ cursor: pointer; user-select: none; }}
    th.sortable .sort-indicator {{ margin-left: 0.25rem; opacity: 0.6; font-size: 0.8rem; }}
    .summary {{ margin: 0.25rem 0 0.75rem 0; font-weight: 600; }}
  </style>
</head>
<body>
    <div class="container-lg py-4 content-wrap">
    <h1 class="mb-1">ARW Analysis Report</h1>
    <p class="mb-3">Toggle decisions per photo and export as JSON.</p>
    <div id="summary" class="summary"></div>
    <div class="mb-3">
      <button id="export" class="btn btn-primary btn-sm">Download decisions JSON</button>
    </div>
    <div id="loading" class="d-flex align-items-center justify-content-center"><div class="spinner"></div>Loading.</div>
    <div class="lightbox" id="lightbox">
      <button id="lightbox-prev" class="btn btn-light position-absolute top-50 start-0 translate-middle-y ms-3">‹</button>
      <img id="lightbox-img" src="" alt="Preview" />
      <button id="lightbox-next" class="btn btn-light position-absolute top-50 end-0 translate-middle-y me-3">›</button>
    </div>
    <div class="table-responsive mt-3">
      <table class="table table-hover align-middle table-alt">
        <thead>
          <tr id="header-row"></tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </div>
  <script id="data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById("data").textContent);
    const tbody = document.getElementById("rows");
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox-img");
    const btnPrev = document.getElementById("lightbox-prev");
    const btnNext = document.getElementById("lightbox-next");
    let currentOrder = [];
    let orderLookup = new Map();
    let currentLightboxPos = null;
    function openLightbox(pos) {{
      if (!currentOrder.length) return;
      if (pos < 0 || pos >= currentOrder.length) return;
      const idx = currentOrder[pos];
      const item = data[idx];
      if (!item) return;
      currentLightboxPos = pos;
      lightboxImg.src = item.preview.replace(/\\\\/g, "/");
      lightbox.classList.add("open");
    }}
    function closeLightbox() {{
      lightbox.classList.remove("open");
      currentLightboxPos = null;
    }}
    function stepLightbox(delta) {{
      if (currentLightboxPos === null) return;
      const nextPos = (currentLightboxPos + delta + currentOrder.length) % currentOrder.length;
      openLightbox(nextPos);
    }}
    lightbox.onclick = (e) => {{
      if (e.target === lightbox) closeLightbox();
    }};
    btnPrev.onclick = (e) => {{ e.stopPropagation(); stepLightbox(-1); }};
    btnNext.onclick = (e) => {{ e.stopPropagation(); stepLightbox(1); }};
    const groups = {{}};
    data.forEach((item, idx) => {{
      if (item.duplicate_group !== undefined) {{
        const g = item.duplicate_group;
        if (!groups[g]) groups[g] = [];
        groups[g].push(idx);
      }}
    }});

    const thresholds = {{
      sharpnessMin: 8,
      tenengradMin: 200,
      motionRatioMin: 0.02,
      noiseStdMax: 25,
      brightnessMin: 0.08,
      brightnessMax: 0.92,
      highlightsMax: 0.92,
      shadowsMax: 0.5,
    }};

    const clamp01 = (v) => Math.min(1, Math.max(0, v));
    const scoreColor = (score) => `hsl(${{Math.round(score * 120)}}, 70%, 45%)`;
    const minScore = (value, min, span = 2) => clamp01((value - min) / (min * span));
    const maxScore = (value, max, span = 0.6) => clamp01((max - value) / (max * span));
    const rangeScore = (value, min, max) => {{
      const mid = (min + max) / 2;
      const half = (max - min) / 2;
      const distance = Math.abs(value - mid);
      return clamp01(1 - distance / half);
    }};
    function badgeStyle(metric, value) {{
      let score = null;
      switch (metric) {{
        case "sharpness":
          score = minScore(value, thresholds.sharpnessMin, 2.5);
          break;
        case "tenengrad":
          score = minScore(value, thresholds.tenengradMin, 2.5);
          break;
        case "motion_ratio":
          score = minScore(value, thresholds.motionRatioMin, 3);
          break;
        case "noise":
          score = maxScore(value, thresholds.noiseStdMax, 1);
          break;
        case "brightness":
          score = rangeScore(value, thresholds.brightnessMin, thresholds.brightnessMax);
          break;
        case "shadows":
          score = maxScore(value, thresholds.shadowsMax, 1);
          break;
        case "highlights":
          score = maxScore(value, thresholds.highlightsMax, 1);
          break;
        default:
          score = null;
      }}
      if (score === null) return null;
      const backgroundColor = scoreColor(score);
      const color = score > 0.55 ? "#0b1223" : "#f8fafc";
      return {{ backgroundColor, color }};
    }}

    function createBadge(text, metric, value) {{
      const div = document.createElement("div");
      div.className = "badge bg-secondary-subtle text-dark-emphasis border me-1";
      const hasValue = value !== undefined && value !== null && !Number.isNaN(value);
      if (metric !== null && metric !== undefined && hasValue) {{
        const style = badgeStyle(metric, value);
        if (style) {{
          Object.assign(div.style, style);
          div.classList.add("metric-badge");
        }}
      }}
      div.textContent = text;
      return div;
    }}

    const columns = [
      {{
        key: "path",
        label: "File",
        sortable: true,
        type: "string",
        getter: (item) => item.path?.toLowerCase() ?? "",
      }},
      {{
        key: "capture",
        label: "Captured",
        sortable: true,
        type: "number",
        getter: (item) => item.capture_ts ?? 0,
        format: (_v, item) => item.capture_time ?? "n/a",
      }},
      {{ key: "preview", label: "Preview", sortable: false }},
      {{ key: "sharpness", label: "Sharpness", sortable: true, type: "number", metric: "sharpness", format: (v) => v.toFixed(1) }},
      {{ key: "tenengrad", label: "Tenengrad", sortable: true, type: "number", metric: "tenengrad", format: (v) => v.toFixed(0) }},
      {{ key: "motion_ratio", label: "Motion ratio", sortable: true, type: "number", metric: "motion_ratio", format: (v) => v.toFixed(2) }},
      {{ key: "noise", label: "Noise std", sortable: true, type: "number", metric: "noise", format: (v) => v.toFixed(1) }},
      {{
        key: "brightness",
        label: "Brightness",
        sortable: true,
        type: "number",
        metric: "brightness",
        getter: (item) => item.brightness?.mean,
        format: (v) => `${{(v * 100).toFixed(0)}}%`,
      }},
      {{
        key: "shadows",
        label: "Shadows",
        sortable: true,
        type: "number",
        metric: "shadows",
        getter: (item) => item.brightness?.shadows,
        format: (v) => `${{(v * 100).toFixed(0)}}%`,
      }},
      {{
        key: "highlights",
        label: "Highlights",
        sortable: true,
        type: "number",
        metric: "highlights",
        getter: (item) => item.brightness?.highlights,
        format: (v) => `${{(v * 100).toFixed(0)}}%`,
      }},
      {{ key: "composition", label: "Composition", sortable: true, type: "number", format: (v) => v.toFixed(2) }},
      {{ key: "faces", label: "Faces", sortable: true, type: "number", getter: (item) => item.faces?.count ?? 0 }},
      {{ key: "face_sharpness", label: "Face sharpness", sortable: true, type: "number", getter: (item) => item.faces?.best_sharpness, format: (v) => v.toFixed(1) }},
      {{ key: "decision", label: "Decision", sortable: true, type: "string", getter: (item) => item.decision ?? "" }},
      {{ key: "reasons", label: "Reasons", sortable: false }},
    ];
    const columnsByKey = Object.fromEntries(columns.map((c) => [c.key, c]));
    const headerRow = document.getElementById("header-row");
    let sortState = {{ key: "path", dir: "asc" }};

    function buildHeader() {{
      headerRow.innerHTML = "";
      columns.forEach((col) => {{
        const th = document.createElement("th");
        th.textContent = col.label;
        if (col.sortable !== false) {{
          th.classList.add("sortable");
          const indicator = document.createElement("span");
          indicator.className = "sort-indicator";
          indicator.textContent =
            sortState.key === col.key ? (sortState.dir === "asc" ? "▲" : "▼") : "⇅";
          th.appendChild(indicator);
          th.onclick = () => toggleSort(col.key);
        }}
        headerRow.appendChild(th);
      }});
    }}

    function toggleSort(key) {{
      const col = columnsByKey[key];
      if (!col || col.sortable === false) return;
      if (sortState.key === key) {{
        sortState.dir = sortState.dir === "asc" ? "desc" : "asc";
      }} else {{
        sortState = {{ key, dir: "asc" }};
      }}
      render();
    }}

    function valueForSort(item, col) {{
      if (!col) return item.path ?? "";
      if (col.getter) return col.getter(item);
      return item[col.key];
    }}

    function compareValues(a, b, col) {{
      const direction = sortState.dir === "asc" ? 1 : -1;
      if (a === undefined || a === null) return (b === undefined || b === null) ? 0 : 1;
      if (b === undefined || b === null) return -1;
      if (col && col.type === "string") {{
        return a.toString().localeCompare(b.toString()) * direction;
      }}
      if (a > b) return 1 * direction;
      if (a < b) return -1 * direction;
      return 0;
    }}

    function hasOtherKeep(groupId, excludeIdx) {{
      if (!groups[groupId]) return false;
      return groups[groupId].some(i => i !== excludeIdx && data[i].decision === "keep");
    }}

    function render() {{
      buildHeader();
      tbody.innerHTML = "";
      const stats = {{ total: data.length, keep: 0, discard: 0 }};
      const colForSort = columnsByKey[sortState.key];
      const sortedIdx = data.map((_, idx) => idx).sort((a, b) => {{
        const av = valueForSort(data[a], colForSort);
        const bv = valueForSort(data[b], colForSort);
        const cmp = compareValues(av, bv, colForSort);
        if (cmp !== 0) return cmp;
        return (data[a].path ?? "").localeCompare(data[b].path ?? "");
      }});
      currentOrder = sortedIdx;
      orderLookup = new Map(sortedIdx.map((id, pos) => [id, pos]));
      currentLightboxPos = null;

      sortedIdx.forEach((idx) => {{
        const item = data[idx];
        const tr = document.createElement("tr");

        columns.forEach((col) => {{
          const td = document.createElement("td");
          switch (col.key) {{
            case "preview": {{
              const img = document.createElement("img");
              img.className = "preview";
              img.src = item.preview.replace(/\\\\/g, "/");
              img.loading = "lazy";
              img.onclick = () => {{
                const pos = orderLookup.get(idx) ?? 0;
                openLightbox(pos);
              }};
              td.appendChild(img);
              break;
            }}
            case "path": {{
              const filename = (item.path || "").split(/[/\\\\]/).pop();
              td.textContent = filename;
              if (item.duplicate_of) {{
                const badge = document.createElement("div");
                badge.className = "badge";
                const dupName = item.duplicate_of.split(/[/\\\\]/).pop();
                badge.textContent = "Duplicate of: " + dupName;
                td.appendChild(badge);
              }}
              break;
            }}
            case "capture": {{
              td.textContent = item.capture_time ?? "n/a";
              break;
            }}
            case "decision": {{
              const decisionClass = item.decision === "keep" ? "decision-keep" : "decision-discard";
              td.className = `decision-cell ${{decisionClass}}`;
              const controls = document.createElement("div");
              controls.className = "controls d-flex gap-2";
              const btnKeep = document.createElement("button");
              btnKeep.textContent = "Keep";
              btnKeep.className = "btn btn-success btn-sm";
              btnKeep.onclick = () => {{ item.decision = "keep"; render(); }};
              const btnDrop = document.createElement("button");
              btnDrop.textContent = "Discard";
              btnDrop.className = "btn btn-danger btn-sm";
              btnDrop.onclick = () => {{
                if (item.duplicate_group !== undefined && !hasOtherKeep(item.duplicate_group, idx)) {{
                  alert("At least one photo in a duplicate set must be kept.");
                  return;
                }}
                item.decision = "discard";
                render();
              }};
              controls.appendChild(btnKeep);
              controls.appendChild(btnDrop);
              td.appendChild(controls);
              break;
            }}
            case "reasons": {{
              if (item.reasons && item.reasons.length) {{
                td.textContent = item.reasons.join(", ");
              }} else {{
                td.textContent = "—";
              }}
              break;
            }}
            default: {{
              const value = col.getter ? col.getter(item) : item[col.key];
              const hasValue = value !== undefined && value !== null && !Number.isNaN(value);
              const display = hasValue
                ? (col.format ? col.format(value, item) : value)
                : "n/a";
              if (col.metric) {{
                td.appendChild(createBadge(display, col.metric, value));
              }} else {{
                td.textContent = display;
              }}
            }}
          }}
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);

        if (item.decision === "keep") stats.keep += 1; else stats.discard += 1;
      }});

      const keepPct = stats.total ? Math.round((stats.keep / stats.total) * 100) : 0;
      const discardPct = stats.total ? Math.round((stats.discard / stats.total) * 100) : 0;
      const summaryDiv = document.getElementById("summary");
      summaryDiv.textContent = `Keeps: ${{keepPct}}% (${{stats.keep}}/${{stats.total}}) • Discards: ${{discardPct}}% (${{stats.discard}}/${{stats.total}})`;

      document.getElementById("loading").style.display = "none";
    }}

    render();

    document.getElementById("export").onclick = () => {{
      const blob = new Blob([JSON.stringify(data, null, 2)], {{type: "application/json"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "decisions.json";
      a.click();
      URL.revokeObjectURL(url);
    }};
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")




