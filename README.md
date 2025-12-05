## Hinweis:

- This project was entirely build with vibe coding using OpenAIs GPT-5.1-Codex-Max

## Enter venv in git bash:

Windows:
py -3.112 -m venv .venv-3.12

source .venv-3.12/Scripts/activate

uv pip install -r ./requirements.txt

python -m src.main <command> --config config.yaml
python -m src.main analyze

Ubuntu:
sudo apt install python3.12 python3.12-venv
python3.12 -m venv .venv-3.12
source .venv-3.12/bin/activate
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.profile
uv --version
uv pip install -r requirements.txt

sudo apt update
sudo apt install build-essential
sudo apt install python3.12-dev

uv pip install black
python3 -m black src

## Code standards

See `CODING_STANDARDS.md` for required style and quality guidelines.

### Face detection/recognition prerequisites

- Install build tools on Windows if you run into compilation errors (e.g., Visual Studio Build Tools with MSVC; but wheels are provided for required libs).
- Mediapipe and InsightFace are installed via `requirements.txt` (wheels available for most Python versions).
- For CPU-only: nothing beyond the above (default).
- If you see protobuf-related errors, ensure `protobuf` stays on the pinned version from `requirements.txt` (>=4.25,<5).

export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin:$PATH"

export PATH="/c/Program Files/NVIDIA/CUDNN/v9.16/bin/12.9:$PATH"
where cudnn64_9.dll
