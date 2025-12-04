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
  <style>
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 1rem; }}
    h1 {{ margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 0.5rem; border-bottom: 1px solid #1e293b; vertical-align: top; }}
    img.preview {{ max-width: 200px; height: auto; border-radius: 4px; cursor: zoom-in; }}
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
    .row {{ background: #0b1223; }}
    .row:nth-child(odd) {{ background: #0c162b; }}
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
<body>
  <h1>ARW Analysis Report</h1>
  <p>Toggle decisions per photo and export as JSON.</p>
  <div id="summary" class="summary"></div>
  <div style="margin-bottom: 1rem;">
    <button id="export">Download decisions JSON</button>
  </div>
  <div id="loading"><div class="spinner"></div>Loading…</div>
  <div class="lightbox" id="lightbox">
    <img id="lightbox-img" src="" alt="Preview" />
  </div>
  <table>
    <thead>
      <tr>
        <th>Preview</th>
        <th>File</th>
        <th>Scores</th>
        <th>Decision</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
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
      div.className = "badge";
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

    function hasOtherKeep(groupId, excludeIdx) {{
      if (!groups[groupId]) return false;
      return groups[groupId].some(i => i !== excludeIdx && data[i].decision === "keep");
    }}

    function render() {{
      tbody.innerHTML = "";
      const stats = {{ total: data.length, keep: 0, discard: 0 }};
      data.forEach((item, idx) => {{
        const tr = document.createElement("tr");
        tr.className = "row";

        const tdImg = document.createElement("td");
        const img = document.createElement("img");
        img.className = "preview";
        img.src = item.preview.replace(/\\\\/g, "/");
        img.loading = "lazy";
        img.onclick = () => {{
          lightboxImg.src = img.src;
          lightbox.classList.add("open");
        }};
        tdImg.appendChild(img);

        const tdFile = document.createElement("td");
        tdFile.innerHTML = `<div>{'{'}${{item.path}}{'}'}</div>`;
        if (item.duplicate_of) {{
          const badge = document.createElement("div");
          badge.className = "badge";
          badge.textContent = "Duplicate of: " + item.duplicate_of;
          tdFile.appendChild(badge);
        }}

        const tdScores = document.createElement("td");
        const addBadge = (label, metricKey, value, formatter = (v) => v) => {{
          const hasValue = value !== undefined && value !== null && !Number.isNaN(value);
          const display = hasValue ? formatter(value) : "n/a";
          tdScores.appendChild(createBadge(`${{label}}: ${{display}}`, hasValue ? metricKey : null, value));
        }};

        addBadge("Sharpness", "sharpness", item.sharpness, (v) => v.toFixed(1));
        addBadge("Tenengrad", "tenengrad", item.tenengrad, (v) => v.toFixed(0));
        addBadge("Motion ratio", "motion_ratio", item.motion_ratio, (v) => v.toFixed(2));
        addBadge("Noise std", "noise", item.noise, (v) => v.toFixed(1));

        if (item.faces) {{
          addBadge("Faces", null, item.faces.count, (v) => v);
          addBadge("Face sharpness", null, item.faces.best_sharpness, (v) => v.toFixed(1));
        }}

        if (item.brightness) {{
          addBadge("Brightness", "brightness", item.brightness.mean, (v) => `${{(v * 100).toFixed(0)}}%`);
          addBadge("Shadows", "shadows", item.brightness.shadows, (v) => `${{(v * 100).toFixed(0)}}%`);
          addBadge("Highlights", "highlights", item.brightness.highlights, (v) => `${{(v * 100).toFixed(0)}}%`);
        }}

        addBadge("Composition", null, item.composition, (v) => v.toFixed(2));

        const tdDecision = document.createElement("td");
        const status = document.createElement("div");
        status.className = item.decision === "keep" ? "keep" : "discard";
        status.textContent = item.decision.toUpperCase();
        const controls = document.createElement("div");
        controls.className = "controls";
        const btnKeep = document.createElement("button");
        btnKeep.textContent = "Keep";
        btnKeep.onclick = () => {{ item.decision = "keep"; render(); }};
        const btnDrop = document.createElement("button");
        btnDrop.textContent = "Discard";
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
        tdDecision.appendChild(status);
        tdDecision.appendChild(controls);
        if (item.reasons && item.reasons.length) {{
          const reasons = document.createElement("div");
          reasons.className = "reasons";
          reasons.textContent = "Reasons: " + item.reasons.join(", ");
          tdDecision.appendChild(reasons);
        }}

        tr.appendChild(tdImg);
        tr.appendChild(tdFile);
        tr.appendChild(tdScores);
        tr.appendChild(tdDecision);
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
