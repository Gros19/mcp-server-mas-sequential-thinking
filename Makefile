# Makefile for MCP Sequential Thinking Server
# Provides convenient commands for testing and development

.PHONY: help test test-unit test-integration test-security test-config test-fast test-debug test-quiet test-parallel test-coverage test-junit check-types lint check-all clean

# Default target: show help
help:
	@echo "MCP Sequential Thinking Server - Test Commands"
	@echo ""
	@echo "Test Execution:"
	@echo "  make test              - Run all tests with coverage (default)"
	@echo "  make test-unit         - Run only unit tests"
	@echo "  make test-integration  - Run only integration tests"
	@echo "  make test-security     - Run only security tests"
	@echo "  make test-config       - Run only configuration tests"
	@echo "  make test-fast         - Run tests without coverage (faster)"
	@echo "  make test-debug        - Run tests in debug mode (verbose)"
	@echo "  make test-quiet        - Run tests with minimal output"
	@echo "  make test-parallel     - Run tests in parallel"
	@echo ""
	@echo "Coverage & Reports:"
	@echo "  make test-coverage     - Run tests with HTML coverage report"
	@echo "  make test-junit        - Run tests with JUnit XML report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make check-types       - Run mypy type checking"
	@echo "  make lint              - Run ruff linting"
	@echo "  make check-all         - Run all quality checks (tests + types + lint)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Remove test artifacts and cache files"

# Run all tests with coverage (default behavior)
test:
	@echo "Running all tests with coverage..."
	@uv run pytest tests/ --cov=src/mcp_server_mas_sequential_thinking --cov-report=term-missing -v --tb=short
	@$(MAKE) -s check-types
	@$(MAKE) -s lint

# Run only unit tests
test-unit:
	@echo "Running unit tests..."
	@uv run pytest tests/unit/ --cov=src/mcp_server_mas_sequential_thinking --cov-report=term-missing -v --tb=short
	@$(MAKE) -s check-types
	@$(MAKE) -s lint

# Run only integration tests
test-integration:
	@echo "Running integration tests..."
	@uv run pytest tests/integration/ --cov=src/mcp_server_mas_sequential_thinking --cov-report=term-missing -v --tb=short
	@$(MAKE) -s check-types
	@$(MAKE) -s lint

# Run only security tests (using markers)
test-security:
	@echo "Running security tests..."
	@uv run pytest -m security --cov=src/mcp_server_mas_sequential_thinking --cov-report=term-missing -v --tb=short
	@$(MAKE) -s check-types
	@$(MAKE) -s lint

# Run only configuration tests (using markers)
test-config:
	@echo "Running configuration tests..."
	@uv run pytest -m config --cov=src/mcp_server_mas_sequential_thinking --cov-report=term-missing -v --tb=short
	@$(MAKE) -s check-types
	@$(MAKE) -s lint

# Run tests without coverage (faster)
test-fast:
	@echo "Running tests without coverage (fast mode)..."
	@uv run pytest tests/ -v --tb=short --no-cov

# Run tests in debug mode (verbose, no capture, long tracebacks)
test-debug:
	@echo "Running tests in debug mode..."
	@uv run pytest tests/ -vv -s --tb=long --no-cov

# Run tests with minimal output
test-quiet:
	@echo "Running tests (quiet mode)..."
	@uv run pytest tests/ -q --tb=short --no-cov

# Run tests in parallel
test-parallel:
	@echo "Running tests in parallel..."
	@uv run pytest tests/ --cov=src/mcp_server_mas_sequential_thinking --cov-report=term-missing -v --tb=short -n auto
	@$(MAKE) -s check-types
	@$(MAKE) -s lint

# Run tests with HTML coverage report
test-coverage:
	@echo "Running tests with HTML coverage report..."
	@uv run pytest tests/ --cov=src/mcp_server_mas_sequential_thinking --cov-report=term-missing --cov-report=html -v --tb=short
	@echo "Coverage report generated in htmlcov/index.html"
	@$(MAKE) -s check-types
	@$(MAKE) -s lint

# Run tests with JUnit XML report
test-junit:
	@echo "Running tests with JUnit XML report..."
	@uv run pytest tests/ --cov=src/mcp_server_mas_sequential_thinking --cov-report=term-missing -v --tb=short --junit-xml=test-results.xml
	@echo "JUnit report generated: test-results.xml"
	@$(MAKE) -s check-types
	@$(MAKE) -s lint

# Run mypy type checking
check-types:
	@echo "Type checking with mypy..."
	@uv run mypy src --ignore-missing-imports

# Run ruff linting
lint:
	@echo "Linting with ruff..."
	@uv run ruff check src tests

# Run all quality checks (tests, type checking, linting)
check-all: test check-types lint
	@echo "All quality checks passed!"

# Clean up test artifacts and cache files
clean:
	@echo "Cleaning test artifacts and cache files..."
	@rm -rf .pytest_cache .coverage htmlcov test-results.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"
