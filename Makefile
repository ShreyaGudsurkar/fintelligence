VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.PHONY: venv install run test clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) -m streamlit run src/web_app/app.py

test:
	$(PYTHON) tests/test_imports.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf $(VENV)
