PY   ?= python3
VENV ?= .venv
BIN  := $(VENV)/bin

.PHONY: setup test lint format data run clean

setup:            ## create .venv and install the package + dev tools
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test:             ## run the test suite (unimplemented TODOs show as skipped)
	$(BIN)/pytest

lint:             ## check style without changing files
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:           ## auto-fix style
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

data:             ## download + reproject the DEMs for every region in configs/regions.yaml
	$(BIN)/python scripts/download_dem.py --all

run:              ## run the default experiment (synthetic terrain unless config says otherwise)
	$(BIN)/python scripts/run_experiment.py --config configs/default.yaml

clean:
	rm -rf results/ .pytest_cache .ruff_cache build src/*.egg-info
