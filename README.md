## Enter venv in git bash:

py -3.11 -m venv .venv-3.11

source .venv-3.11/Scripts/activate

uv pip install -r ./requirements.txt

python -m src.main <command> --config config.yaml
