# Automatic Image Culling

Small toolchain to scan Sony `.ARW` photos, generate previews, and score/flag keepers with an HTML report.

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

- `python -m src.main scan --config config.yaml` — list discovered `.ARW` files (+EXIF if `--json`).
- `python -m src.main previews --config config.yaml` — generate and cache previews.
- `python -m src.main analyze --config config.yaml` — score frames, mark duplicates, and write the report.
- `python -m src.main decisions --apply` — move files into keep/discard subfolders based on `analysis/decisions.json`.

## GUI (optional)

- `python -m src.gui` — launch a simple desktop UI to run discover, previews, analyze, and decisions steps.
- Use the Configuration tab to edit settings (thresholds, face detection, etc.) and Save to `config.yaml`.

## Configuration

- Defaults live in the app/GUI; use the Configuration tab to save `config.yaml`.
- `preview` block controls preview size/format; `analysis` block tunes thresholds.
- Analysis outputs are written under `input_dir/analysis`; previews under `input_dir/previews` and keep/discard moves under `input_dir/output`.
- Under WSL, set `input_dir` to the host path (`C:\...`); the analyzer writes Windows-style paths into outputs so you can open `report.html` from the host.

## Face detection (optional)

- Mediapipe (default) and InsightFace are installed via `requirements.txt`; enable with `analysis.face.enabled: true`.
- GPU users on Windows may need CUDA/CUDNN on `PATH` (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin` and CUDNN `bin`).

## Style and testing

- Follow `CODING_STANDARDS.md`; format with `black src`.
- Fast sanity check: `python3 -m compileall src`.

## Credits

- Built collaboratively with OpenAI GPT-5.1-Codex-Max via vibe coding sessions.
