.PHONY: setup lint fmt typecheck test test-fast smoke bench clean help

UV ?= uv
PY ?= $(UV) run

help:
	@echo "setup      bootstrap the local dev environment (WSL2 / Linux)"
	@echo "lint       ruff check + format check"
	@echo "fmt        ruff format (writes)"
	@echo "typecheck  mypy --strict on src/datum"
	@echo "test       full pytest run"
	@echo "test-fast  unit tests only, parallel, no slow markers"
	@echo "smoke      end-to-end pipeline on the sample match"
	@echo "bench      run the benchmark suite"
	@echo "clean      drop caches and build artifacts"

setup:
	./scripts/bootstrap_wsl.sh

lint:
	$(PY) ruff check src tests
	$(PY) ruff format --check src tests

fmt:
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

typecheck:
	$(PY) mypy

test:
	$(PY) pytest

test-fast:
	$(PY) pytest tests/unit -n auto -m "not slow"

smoke:
	$(PY) datum pipeline run \
		--match data/samples/sample_match.mp4 \
		--config configs/pipelines/smoke.yaml

bench:
	$(PY) python -m datum.eval.bench

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
