.PHONY: help install install-dev test lint format type-check clean docs build

help:  ## Show this help message
	@echo \"Available commands:\"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = \":.*?## \"}; {printf \"\\033[36m%-20s\\033[0m %s\\n\", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e \".[dev]\"
	pre-commit install

install-gpu:  ## Install with GPU support
	pip install -e \".[dev,gpu]\"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint:  ## Run linting
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code
	black src/ tests/
	isort src/ tests/

type-check:  ## Run type checking
	mypy src/

quality:  ## Run all quality checks
	make lint
	make type-check
	make test

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name \"*.pyc\" -delete

docs:  ## Build documentation
	cd docs && make html

build:  ## Build package
	python -m build

benchmark:  ## Run benchmarks
	python scripts/run_benchmarks.py

profile:  ## Profile training performance
	python scripts/profile_training.py