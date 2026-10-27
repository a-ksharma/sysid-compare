.PHONY: lint format test data train evaluate figures api dashboard

PYTHON_PATH_PREFIX = set PYTHONPATH=src &&

lint:
	ruff check src tests api scripts
	ruff format --check src tests api scripts
	$(PYTHON_PATH_PREFIX) mypy src

format:
	ruff format src tests api scripts

test:
	pytest

data:
	$(PYTHON_PATH_PREFIX) python scripts/make_datasets.py

train:
	$(PYTHON_PATH_PREFIX) python scripts/train_all.py

evaluate:
	$(PYTHON_PATH_PREFIX) python scripts/run_experiments.py

figures:
	$(PYTHON_PATH_PREFIX) python scripts/make_figures.py

api:
	$(PYTHON_PATH_PREFIX) uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py
