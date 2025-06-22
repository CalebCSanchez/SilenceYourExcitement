.PHONY: help dev run build test lint clean install
.DEFAULT_GOAL := help

# Variables
PYTHON := python
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/Scripts/python
VENV_PIP := $(VENV_DIR)/Scripts/pip

help: ## Show this help message
	@echo "Rage-Reducer Development Commands"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev: ## Set up development environment
	@echo "Setting up development environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip setuptools wheel
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PIP) install -e .
	@echo "Development environment ready!"
	@echo "Activate with: $(VENV_DIR)/Scripts/activate"

install: ## Install package in development mode
	$(VENV_PIP) install -e .

run: ## Run the application
	$(VENV_PYTHON) -m rage_reducer

run-debug: ## Run the application with debug logging
	$(VENV_PYTHON) -m rage_reducer --debug

test: ## Run tests
	$(VENV_PYTHON) -m pytest tests/ -v --cov=rage_reducer --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	$(VENV_PYTHON) -m pytest tests/ -v

lint: ## Run linting tools
	$(VENV_PYTHON) -m black rage_reducer/ tests/ --check
	$(VENV_PYTHON) -m isort rage_reducer/ tests/ --check-only
	$(VENV_PYTHON) -m ruff rage_reducer/ tests/
	$(VENV_PYTHON) -m mypy rage_reducer/

format: ## Format code
	$(VENV_PYTHON) -m black rage_reducer/ tests/
	$(VENV_PYTHON) -m isort rage_reducer/ tests/

build: ## Build executable with PyInstaller
	@echo "Building executable..."
	$(VENV_PYTHON) -m PyInstaller \
		--noconsole \
		--onefile \
		--add-data "rage_reducer/assets;assets" \
		--name "RageReducer" \
		--icon "rage_reducer/assets/icon.ico" \
		rage_reducer/__main__.py
	@echo "Executable built: dist/RageReducer.exe"

build-debug: ## Build executable with console for debugging
	@echo "Building debug executable..."
	$(VENV_PYTHON) -m PyInstaller \
		--console \
		--onefile \
		--add-data "rage_reducer/assets;assets" \
		--name "RageReducer-Debug" \
		--icon "rage_reducer/assets/icon.ico" \
		rage_reducer/__main__.py
	@echo "Debug executable built: dist/RageReducer-Debug.exe"

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	-rm -rf build/
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf htmlcov/
	-rm -rf .coverage
	-rm -rf .mypy_cache/
	-rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"

clean-venv: ## Remove virtual environment
	@echo "Removing virtual environment..."
	-rm -rf $(VENV_DIR)
	@echo "Virtual environment removed!"

setup-pre-commit: ## Setup pre-commit hooks
	$(VENV_PIP) install pre-commit
	$(VENV_PYTHON) -m pre_commit install

# Windows-specific commands
setup-win: ## Setup for Windows development
	@echo "Setting up for Windows..."
	python -m venv $(VENV_DIR)
	$(VENV_DIR)\Scripts\pip install --upgrade pip setuptools wheel
	$(VENV_DIR)\Scripts\pip install -r requirements.txt
	$(VENV_DIR)\Scripts\pip install -e .
	@echo "Windows setup complete!"

run-win: ## Run on Windows
	$(VENV_DIR)\Scripts\python -m rage_reducer

# Docker commands (future enhancement)
docker-build: ## Build Docker image
	docker build -t rage-reducer .

docker-run: ## Run in Docker container
	docker run --rm -it rage-reducer

# Package info
info: ## Show package information
	@echo "Rage-Reducer Development Info"
	@echo "============================="
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Virtual env: $(VENV_DIR)"
	@echo "Package location: $(shell pwd)"
	@echo "Dependencies: requirements.txt" 