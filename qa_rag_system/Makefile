.PHONY: install dev-install test lint format run docker-build docker-run clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check .
	mypy src/

format:
	ruff format .

run:
	streamlit run app.py

docker-build:
	docker build -t qa-rag-system .

docker-run:
	docker run -p 8501:8501 --env-file .env qa-rag-system

docker-compose-up:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf chroma_db

