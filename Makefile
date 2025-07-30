.PHONY: help install install-dev test test-unit test-integration test-e2e lint format typecheck docs clean build upload

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run all tests
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

test-e2e:  ## Run end-to-end tests only
	pytest tests/e2e/ -v

lint:  ## Run linting checks
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code
	black src/ tests/
	isort src/ tests/
	ruff check src/ tests/ --fix

typecheck:  ## Run type checking
	mypy src/

docs:  ## Build documentation
	sphinx-build docs/ docs/_build/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

upload:  ## Upload to PyPI (requires credentials)
	python -m twine upload dist/*

benchmark:  ## Run performance benchmarks
	python scripts/benchmark.py

monitor:  ## Start monitoring stack
	docker-compose -f docker/monitoring.yml up -d

dev-setup:  ## Complete development setup
	make install-dev
	pre-commit install
	echo "Development environment ready!"