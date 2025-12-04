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
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; }}
    h1 {{ margin-top: 0; }}
    img.preview {{ max-width: 200px; height: auto; border-radius: 6px; cursor: zoom-in; }}
    .lightbox {{
      position: fixed; inset: 0; background: rgba(0,0,0,0.8);
      display: flex; align-items: center; justify-content: center;
      opacity: 0; pointer-events: none; transition: opacity 0.2s ease;
      z-index: 1000;
    }}
    .lightbox.open {{ opacity: 1; pointer-events: all; }}
    .lightbox img {{ max-width: 95vw; max-height: 95vh; border-radius: 6px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
    .controls button {{ margin-right: 0.25rem; }}
    .keep {{ color: #22c55e; font-weight: 600; }}
    .discard {{ color: #ef4444; font-weight: 600; }}
    .badge {{ padding: 0.1rem 0.4rem; border-radius: 4px; background: #1f2937; margin-right: 0.3rem; }}
    .metric-badge {{ border: 1px solid rgba(15,23,42,0.3); font-weight: 700; }}
    th.sortable {{ cursor: pointer; user-select: none; }}
    th.sortable .sort-indicator {{ margin-left: 0.25rem; opacity: 0.6; font-size: 0.8rem; }}
    .reasons {{ margin-top: 0.35rem; color: #94a3b8; font-size: 0.9rem; }}
    .summary {{ margin: 0.25rem 0 0.75rem 0; font-weight: 600; }}
    #loading {{
      position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
      background: rgba(0,0,0,0.7); color: #e2e8f0; z-index: 2000;
      font-size: 1.2rem; gap: 0.5rem;
    }}
    .spinner {{
      width: 24px; height: 24px; border-radius: 999px; border: 3px solid #1e293b; border-top-color: #38bdf8;
      animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body class="bg-dark text-light">
    <div class="container-fluid py-3">
    <h1 class="mb-1">ARW Analysis Report</h1>
    <p class="mb-3">Toggle decisions per photo and export as JSON.</p>
    <div id="summary" class="summary"></div>
    <div class="mb-3">
      <button id="export" class="btn btn-primary btn-sm">Download decisions JSON</button>
    </div>
    <div id="loading" class="d-flex align-items-center justify-content-center"><div class="spinner"></div>Loading.</div>
    <div class="lightbox" id="lightbox">
      <img id="lightbox-img" src="" alt="Preview" />
    </div>
    <div class="table-responsive">
      <table class="table table-dark table-striped align-middle table-hover">
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
    lightbox.onclick = () => {{ lightbox.classList.remove("open"); }};
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
      div.className = "badge bg-secondary text-light me-1";
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
      {{ key: "preview", label: "Preview", sortable: false }},
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
            sortState.key === col.key ? (sortState.dir === "asc" ? "^" : "v") : "";
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
                lightboxImg.src = img.src;
                lightbox.classList.add("open");
              }};
              td.appendChild(img);
              break;
            }}
            case "path": {{
              td.textContent = item.path;
              if (item.duplicate_of) {{
                const badge = document.createElement("div");
                badge.className = "badge";
                badge.textContent = "Duplicate of: " + item.duplicate_of;
                td.appendChild(badge);
              }}
              break;
            }}
            case "capture": {{
              td.textContent = item.capture_time ?? "n/a";
              break;
            }}
            case "decision": {{
              td.className = item.decision === "keep" ? "table-success text-dark" : "table-danger text-dark";
              const status = document.createElement("div");
              status.className = item.decision === "keep" ? "keep" : "discard";
              status.textContent = item.decision ? item.decision.toUpperCase() : "N/A";
              const controls = document.createElement("div");
              controls.className = "controls d-flex gap-2 mt-2";
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
              td.appendChild(status);
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




