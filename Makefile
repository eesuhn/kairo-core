VENV = .venv
CONFIG = configs/misc

all: venv

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: pyproject.toml
	@uv sync
	@uv run pre-commit install --config=$(CONFIG)/pre-commit.yaml
	@uv run dvc pull

format:
	@uv run ruff format

check:
	@uv run ruff check

check-fix:
	@uv run ruff check --fix

test:
	@uv run pytest

notebooks:
	@uv run jupyter notebook notebooks/

upgrade:
	@uv sync --upgrade

nb-clean:
	@uv run nb-clean clean -n notebooks/

pull-dvc:
	@uv run dvc pull

push-dvc:
	@uv run dvc add data/ models/
	@uv run dvc push
	@git add ./*.dvc

clean-dvc:
	@uv run dvc gc -w
	@uv run dvc gc -w -c -r gdrive-data

status-dvc:
	@uv run dvc data status --granular

clean:
	@uvx pyclean -v .

.PHONY: all venv format check check-fix test \
		notebooks upgrade nb-clean pull-dvc \
		push-dvc clean-dvc status-dvc clean
