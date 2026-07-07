.PHONY: install install-dev run cli test test-all lint fmt docker-build docker-run

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	streamlit run streamlit_app.py

cli:
	python cli.py index $(REPO)

test:
	pytest

test-all:
	pytest -m "integration or not integration"

lint:
	ruff check .

fmt:
	ruff check --fix .

docker-build:
	docker build -t codebase-mentor .

docker-run:
	docker compose up --build
