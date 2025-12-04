## Enter venv in git bash:

py -3.112 -m venv .venv-3.12

source .venv-3.12/Scripts/activate

uv pip install -r ./requirements.txt

python -m src.main <command> --config config.yaml
