PY   ?= python3
VENV ?= .venv
BIN  := $(VENV)/bin
JOBS ?= 10
EXPERIMENTS := e1 e1@synth e2 e3 e4 e5 e6 e7

.PHONY: setup test lint format data survey terrain run demo media experiments quick smoke figures \
        benchmark visuals report clean

setup:            ## create .venv and install the package + dev tools
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test:             ## run the test suite
	$(BIN)/pytest

lint:             ## check style without changing files
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:           ## auto-fix style
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

data:             ## download + reproject the regions in configs/regions.yaml, and the COP90 survey
	$(BIN)/python scripts/download_dem.py --all
	$(BIN)/python scripts/survey_regions.py

survey:           ## re-run the (pre-registered) region selection and rewrite configs/regions.yaml
	$(BIN)/python scripts/survey_regions.py --write

terrain:          ## terrain statistics table + overview figure
	$(BIN)/python scripts/terrain_stats.py

run:              ## quick-look run of configs/default.yaml (synthetic terrain)
	$(BIN)/python scripts/run_experiment.py --config configs/default.yaml

demo:             ## live animated window: particle cloud, truth and estimates over the terrain
	$(BIN)/python scripts/live_demo.py

media:            ## README animations in assets/ (live demo rendered to video, then GIF; needs ffmpeg)
	@mkdir -p results/media assets
	$(BIN)/python scripts/live_demo.py --save results/media/demo.mp4
	$(BIN)/python scripts/live_demo.py --sigma0 1000 --run 0 --save results/media/false_lock.mp4
	ffmpeg -loglevel error -y -t 6 -i results/media/demo.mp4 -vf "fps=7,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:reserve_transparent=0[p];[b][p]paletteuse=dither=none" assets/demo.gif
	ffmpeg -loglevel error -y -t 6 -i results/media/false_lock.mp4 -vf "fps=7,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:reserve_transparent=0[p];[b][p]paletteuse=dither=none" assets/false_lock.gif

experiments:      ## every experiment, full size (resumable; logs in results/<exp>.log)
	@for e in $(EXPERIMENTS); do \
	  echo "== $$e"; \
	  $(BIN)/python scripts/sweep.py --experiment $$e --jobs $(JOBS) 2>&1 | tee results/$$e.log || exit 1; \
	done

quick:            ## every experiment in --quick mode (3 runs, short flights) + its figures
	@mkdir -p results
	@for e in $(EXPERIMENTS); do \
	  echo "== $$e (quick)"; \
	  $(BIN)/python scripts/sweep.py --experiment $$e --quick --jobs $(JOBS) || exit 1; \
	  $(BIN)/python scripts/make_figures.py --experiment $$e --quick || exit 1; \
	done

smoke:            ## pipeline check on synthetic terrain (no download needed)
	$(BIN)/python scripts/sweep.py --experiment smoke --quick --jobs $(JOBS)
	$(BIN)/python scripts/make_figures.py --experiment smoke --as e1 --quick

benchmark:        ## runtime per filter step (single process; run on an idle machine)
	$(BIN)/python scripts/benchmark.py

figures:          ## every figure, table and macro from saved results (no re-running)
	$(BIN)/python scripts/terrain_stats.py
	$(BIN)/python scripts/extract_listings.py
	$(BIN)/python scripts/validation.py
	$(BIN)/python scripts/make_figures.py --all

report:           ## build the PDF if latexmk exists (otherwise compile report/latex on Overleaf)
	@if command -v latexmk >/dev/null; then \
	  cd report/latex && latexmk -pdf -interaction=nonstopmode -jobname=220059J_mini_project_report main.tex; \
	else echo "latexmk not found: upload report/latex (with report/figures, tables, generated) to Overleaf"; fi

clean:
	rm -rf .pytest_cache .ruff_cache build src/*.egg-info
