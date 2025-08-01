.PHONY: help install install-dev test test-unit test-integration test-e2e lint format typecheck docs clean build upload docker-build docker-push security-scan

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

docker-build:  ## Build all Docker images
	docker build --target development -t dgfrl:dev .
	docker build --target production -t dgfrl:prod .
	docker build --target gpu -t dgfrl:gpu .

docker-push:  ## Push Docker images to registry
	docker tag dgfrl:prod $(REGISTRY)/dgfrl:latest
	docker tag dgfrl:prod $(REGISTRY)/dgfrl:$(VERSION)
	docker push $(REGISTRY)/dgfrl:latest
	docker push $(REGISTRY)/dgfrl:$(VERSION)

security-scan:  ## Run security scans
	bandit -r src/
	safety check
	pip-audit

docker-scan:  ## Scan Docker images for vulnerabilities
	docker scout quickview dgfrl:prod
	trivy image dgfrl:prod

ci-test:  ## Run CI test suite
	make lint
	make typecheck
	make test
	make security-scan

release:  ## Create a release build
	make clean
	make ci-test
	make build
	make docker-build

k8s-deploy:  ## Deploy to Kubernetes
	kubectl apply -f infrastructure/kubernetes/

k8s-delete:  ## Delete Kubernetes deployment
	kubectl delete -f infrastructure/kubernetes/