VENV = .venv
CONFIG = configs
OS := $(shell uname -s)

all: venv

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: pyproject.toml
	@uv sync
	@uv run pre-commit install --config=$(CONFIG)/pre-commit.yaml

%:
	@:

server:
	@uv run -m src.server

data:
	@uv run -m src.inter_data_handler

ner-train-local:
	@uv run -m scripts.train.ner --mode new

upgrade:
	@uv sync --upgrade
	@git add uv.lock

format:
	@uv run ruff format

check:
	@uv run ruff check

check-fix:
	@uv run ruff check --fix

clean:
	rm -rf .pytest_cache/ .ruff_cache/
	@uvx pyclean .

test:
	@uv run pytest $(filter-out $@,$(MAKECMDGOALS))

notebooks:
	@uv run jupyter notebook notebooks/

nb-clean:
	@uv run nb-clean clean -n notebooks/

nb-update:
	@uv run -m scripts.nb_update

dvc-setup:
	@if [ "$(OS)" = "Darwin" ]; then \
		rm -rf ~/Library/Caches/pydrive2fs/; \
	elif [ "$(OS)" = "Linux" ]; then \
		rm -rf ~/.cache/pydrive2fs/; \
	else \
		rm -rf "$$LOCALAPPDATA/pydrive2fs" 2>/dev/null || true; \
	fi
	@uv run dvc pull

dvc-status:
	@uv run dvc data status --granular

dvc-pull:
	@uv run dvc pull

dvc-push:
	@uv run dvc add data/ models/
	@uv run dvc push
	@git add ./*.dvc

dvc-clean:
	@echo y | uv run dvc gc -w
	@echo y | uv run dvc gc -w -c -r gdrive-data

.PHONY: all venv upgrade format check check-fix clean test \
		notebooks nb-clean nb-update dvc-status dvc-pull \
		dvc-push dvc-clean data dvc-setup ner-train-local \
		server
