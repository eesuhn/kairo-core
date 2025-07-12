VENV = .venv
CONFIG = configs/misc

all: venv

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: pyproject.toml
	@uv sync
	@uv run pre-commit install --config=$(CONFIG)/pre-commit.yaml

%:
	@:

data:
	@uv run -m src.data_handler

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
		dvc-push dvc-clean data
