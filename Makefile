VENV = .venv
CONFIG = configs/misc

all: venv

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: pyproject.toml
	@uv sync
	@uv run pre-commit install --config=$(CONFIG)/pre-commit.yaml

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

.PHONY: all venv format check check-fix test notebooks upgrade
