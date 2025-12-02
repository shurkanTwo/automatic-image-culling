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

        const faceBadges = item.faces
          ? `<div class="badge">Faces: ${{item.faces.count}}</div><div class="badge">Face sharpness: ${{item.faces.best_sharpness?.toFixed(1) ?? 'n/a'}}</div>`
          : "";
        const tdScores = document.createElement("td");
        tdScores.innerHTML = `
          <div class="badge">Sharpness: ${{item.sharpness.toFixed(1)}}</div>
          <div class="badge">Tenengrad: ${{item.tenengrad?.toFixed(0) ?? 'n/a'}}</div>
          <div class="badge">Motion ratio: ${{item.motion_ratio?.toFixed(2) ?? 'n/a'}}</div>
          <div class="badge">Noise std: ${{item.noise?.toFixed(1) ?? 'n/a'}}</div>
          ${{faceBadges}}
          <div class="badge">Brightness: ${{(item.brightness.mean * 100).toFixed(0)}}%</div>
          <div class="badge">Shadows: ${{(item.brightness.shadows * 100).toFixed(0)}}%</div>
          <div class="badge">Highlights: ${{(item.brightness.highlights * 100).toFixed(0)}}%</div>
          <div class="badge">Composition: ${{item.composition.toFixed(2)}}</div>
        `;

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
