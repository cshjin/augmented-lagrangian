.PHONY: help install install-dev test lint format clean build upload docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e .
	pip install -r requirements-dev.txt

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 src tests examples
	mypy src --ignore-missing-imports

format: ## Format code
	black src tests examples

format-check: ## Check code formatting
	black --check src tests examples

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

isort: ## Sort imports
	isort src tests examples

isort-check: ## Check import sorting
	isort --check-only src tests examples
	
build: clean ## Build package
	python -m build

check-build: build ## Check package build
	twine check dist/*

upload-test: build ## Upload to Test PyPI
	twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	twine upload dist/*

docs: ## Generate documentation
	sphinx-build -b html docs docs/_build/html

run-examples: ## Run example scripts
	python examples/01_single_const.py
	python examples/02_multi_const.py