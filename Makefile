VENV = .venv
CONFIG = configs/misc

all: venv

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: pyproject.toml
	@uv sync
	@uv run pre-commit install --config=$(CONFIG)/pre-commit.yaml
	@uv run dvc pull

data:
	@uv run -m src.data

upgrade:
	@uv sync --upgrade

format:
	@uv run ruff format

check:
	@uv run ruff check

check-fix:
	@uv run ruff check --fix

clean:
	@uvx pyclean .

test:
	@uv run pytest

notebooks:
	@uv run jupyter notebook notebooks/

nb-clean:
	@uv run nb-clean clean -n notebooks/

nb-update:
	@uv run -m scripts.nb.main

dvc-status:
	@uv run dvc data status --granular

dvc-pull:
	@uv run dvc pull

dvc-push:
	@uv run dvc add data/ models/
	@uv run dvc push
	@git add ./*.dvc

dvc-clean:
	@uv run dvc gc -w
	@uv run dvc gc -w -c -r gdrive-data

.PHONY: all venv upgrade format check check-fix clean test \
		notebooks nb-clean nb-update dvc-status dvc-pull \
		dvc-push dvc-clean data
