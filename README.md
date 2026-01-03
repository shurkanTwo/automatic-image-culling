# Automatic Image Culling

Small toolchain to scan input directory for Sony `.ARW` photos, generate previews automatically, and score/flag keepers with an HTML report.

Proprietary — no copying, redistribution, or derivative works without permission.

## What it does

- Scans the input directory for `.ARW` files, excluding output folders.
- Builds lightweight previews (configurable size/format) for fast analysis.
- Computes quality metrics (sharpness, motion, noise, brightness, composition).
- Groups likely duplicates using perceptual hashes and time windows.
- Produces a JSON results file and a standalone HTML report for browsing.
- Optionally detects faces to include in the analysis results.

## How it works

1. **Scan input directory**: walks `input_dir` and reads EXIF timestamps to order frames.
2. **Preview**: uses `rawpy` + `Pillow` to extract or demosaic a thumbnail.
3. **Analyze**: computes metrics, scores each frame, and labels duplicates.
4. **Report**: writes `analysis.json` + `report.html` (with CSS/JS sidecars).
5. **Decisions**: applies `analysis/decisions.json` to move or copy files.

The HTML report is static and self-contained with local assets. It is safe to
open directly in a browser.

## Quick start

### Windows (PowerShell)

1. `py -3.12 -m venv .venv-3.12`
2. `.\\.venv-3.12\\Scripts\\Activate.ps1`
3. git bash: source .venv-3.12/Scripts/activate
4. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
5. `uv pip install -r requirements.txt`
6. Run `python -m src.gui`, open the Configuration tab, set `input_dir`, and Save (writes `config.yaml`)
7. `python -m src.main analyze --config config.yaml`
   - Outputs land in `input_dir\\analysis` (`analysis.json`, `report.html`, `report.css`, `report.js`), with Windows paths for opening from the host.

### Linux / WSL (Ubuntu)

1. `sudo apt update && sudo apt install python3.12 python3.12-venv build-essential python3.12-dev`
2. `python3.12 -m venv .venv-3.12 && source .venv-3.12/bin/activate`
3. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` then `source ~/.profile`
4. `uv pip install -r requirements.txt`
5. Run `python -m src.gui`, open the Configuration tab, set `input_dir` (use `/mnt/c/...` for Windows storage), and Save (writes `config.yaml`)
6. `python -m src.main analyze --config config.yaml`
   - Outputs land in `input_dir/analysis`; paths in the report are converted to Windows form for host opening.

## Commands

- `python -m src.main scan --config config.yaml` — scan input directory for `.ARW` files (+EXIF if `--json`).
- `python -m src.main analyze --config config.yaml` — score frames, mark duplicates, and write the report.
- `python -m src.main decisions --apply` — move files into keep/discard subfolders based on `analysis/decisions.json`.

## Outputs

By default, outputs are stored under `input_dir`:

- `analysis/analysis.json` — full analysis data per frame.
- `analysis/report.html` — static report with sortable tables and previews.
- `analysis/report.css`, `analysis/report.js` — report assets.
- `analysis/decisions.json` — editable decisions file used by the `decisions` command.
- `previews/` — generated preview images.
- `output/keep` and `output/discard` — results of applying decisions.

## GUI (optional)

- `python -m src.gui` — launch a simple desktop UI to run scan input directory, analyze, and decisions steps (analysis generates previews automatically).
- Use the Configuration tab to edit settings (thresholds, face detection, etc.) and Save to `config.yaml`.

## Configuration

- Defaults live in the app/GUI; use the Configuration tab to save `config.yaml`.
- `preview` block controls preview size/format; `analysis` block tunes thresholds.
- Analysis outputs are written under `input_dir/analysis`; previews under `input_dir/previews` and keep/discard moves under `input_dir/output`.
- Under WSL, set `input_dir` to the host path (`C:\...`); the analyzer writes Windows-style paths into outputs so you can open `report.html` from the host.

## Face detection (optional)

- Mediapipe (default) and InsightFace are installed via `requirements.txt`; enable with `analysis.face.enabled: true`.
- GPU users on Windows may need CUDA/CUDNN on `PATH` (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin` and CUDNN `bin`).

## Dependencies and optional features

- `rawpy`, `Pillow` — preview generation (required for analysis).
- `exifread` — EXIF parsing for timestamps and orientation.
- `scikit-image` — optional SSIM/PSNR similarity details in duplicate reasons.
- `mediapipe` or `insightface` — optional face detection backend.

## Style and testing

- Follow `CODING_STANDARDS.md`; format with `black src`.
- Fast sanity check: `python3 -m compileall src`.

## Credits

- Built collaboratively with OpenAI GPT-5.1-Codex-Max via vibe coding sessions.
