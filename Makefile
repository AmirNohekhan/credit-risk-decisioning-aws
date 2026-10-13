PYTHON ?= python
.PHONY: setup data train evaluate backtest policy-simulate demo serve test lint format typecheck clean
setup:
	$(PYTHON) -m pip install -e ".[dev]"
data:
	$(PYTHON) scripts/generate_data.py
train evaluate backtest policy-simulate demo:
	$(PYTHON) scripts/run_demo.py
serve:
	$(PYTHON) -m uvicorn credit_risk.serving:app --host 0.0.0.0 --port 8000
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check src tests scripts
format:
	$(PYTHON) -m ruff format src tests scripts
typecheck:
	$(PYTHON) -m mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
