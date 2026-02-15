ifneq (,$(wildcard ./.env))
    include .env
    export
endif

PYTHON := python
PIP := pip
DC := docker-compose

SRC_DIR := src
TEST_DIR := tests

QDRANT_HOST ?= localhost
QDRANT_PORT ?= 6333
QDRANT_URL := http://$(QDRANT_HOST):$(QDRANT_PORT)

.DEFAULT_GOAL := help

.PHONY: help install infra-up infra-down clean-data test test-full lint format run-ingest clean

help: ## Show available commands
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed!"

infra-up: ## Start Qdrant server
	$(DC) up -d
	@echo "⏳ Waiting for Qdrant..."
	@sleep 5
	@echo "✅ Qdrant is ready at $(QDRANT_URL)"

infra-down: ## Stop Qdrant server
	$(DC) down

clean-data: ## Wipe Qdrant storage (Warning: Destructive!)
	@echo "🧹 Cleaning Qdrant volume data..."
	rm -rf data/qdrant_storage/*
	@echo "✨ Storage wiped clean."
	
test: ## 🧪 Run tests with coverage and XML reports
	@echo "🧪 Running tests..."
	PYTHONPATH=$(SRC_DIR) pytest $(TEST_DIR) \
		-vv \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=xml:coverage.xml \
		--junitxml=test-results.xml

lint: ## Check code style with Ruff
	@echo "🔍 Checking code..."
	ruff check $(SRC_DIR)

lint-fix: ## Auto-fix code style issues with Ruff
	@echo "🛠️ Fixing code issues..."
	ruff check --fix $(SRC_DIR)

format: ## 🎨 Auto-format code with Ruff (replaces black)
	@echo "🎨 Formatting code..."
	ruff format $(SRC_DIR)
	ruff check --select I --fix $(SRC_DIR)

run-ingest: ## Run the ingestion pipeline
	cd $(SRC_DIR) && $(PYTHON) -m rag_pipeline.ingest

run-server: ## Start the API server (Placeholder)
	@echo "🚧 Server not implemented yet."

clean: ## Clean cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✨ Project cleaned."

test-full: infra-down clean-data infra-up test infra-down ## Run full cycle: Clean -> Up -> Test -> Down
check-all: lint format test-full ## 🏆 Run EVERYTHING: Lint, Format, and Full Test Cycle
	@echo "✨ All checks passed! Code is ready to be pushed."