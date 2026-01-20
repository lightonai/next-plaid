.PHONY: all build test lint fmt check clean doc example install-hooks compare-reference lint-python fmt-python evaluate-scifact evaluate-scifact-cached compare-scifact compare-scifact-cached benchmark-scifact-update benchmark-scifact-api benchmark-scifact-docker benchmark-scifact-docker-keep benchmark-fastplaid-compat benchmark-fastplaid-compat-keep benchmark-api-encoding benchmark-onnx-api benchmark-onnx-api-cuda benchmark-onnx-api-gte benchmark-onnx-api-gte-int8 benchmark-onnx-vs-pylate ci-api ci-onnx ci-cli test-api-integration test-api-rate-limit onnx-setup onnx-export onnx-export-all onnx-benchmark onnx-benchmark-rust onnx-compare onnx-lint onnx-fmt docker-build docker-build-cuda docker-up docker-up-cuda docker-down docker-logs kill-api docs-serve docs-build docs-deploy

all: fmt lint test

# Build the project
build:
	cargo build

# Build in release mode
release:
	cargo build --release

# Run all tests
test:
	cargo test

# Run tests in release mode
test-release:
	cargo test --release

# Run linting (clippy + format check)
lint: fmt-check clippy

# Run clippy
clippy:
	cargo clippy --all-targets -- -D warnings

# Check formatting
fmt-check:
	cargo fmt --all -- --check

# Format code
fmt:
	cargo fmt --all

# Type check without building
check:
	cargo check

# Build documentation
doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# Open documentation in browser
doc-open:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --open

# Clean build artifacts
clean:
	cargo clean

# Run the basic example
example:
	cargo run --example basic --release

# Run all CI checks locally
ci: doc ci-index ci-api ci-onnx ci-cli lint-python
	@echo "All CI checks passed!"

# Kill any existing API process on port 8080
kill-api:
	@if lsof -t -i:8080 >/dev/null 2>&1; then \
		echo "Killing process on port 8080..."; \
		kill -9 $$(lsof -t -i:8080) 2>/dev/null || true; \
		sleep 2; \
	fi

ci-index:
	cd next-plaid && cargo fmt --all -- --check
	cd next-plaid && cargo clippy --all-targets -- -D warnings
	cd next-plaid && cargo test

# Run CI checks for api crate (using cross-platform features only)
ci-api:
	cd next-plaid-api && cargo fmt --all -- --check
	cd next-plaid-api && cargo clippy --all-targets -- -D warnings
	cd next-plaid-api && cargo test

# Run CI checks for onnx crate (using cross-platform features only)
ci-onnx:
	cd next-plaid-onnx && cargo fmt --all -- --check
	cd next-plaid-onnx && cargo clippy --all-targets -- -D warnings
	cd next-plaid-onnx && cargo test
	cd next-plaid-onnx/python && uv run --extra dev ruff check .
	cd next-plaid-onnx/python && uv run --extra dev python -m pytest tests/

# Run CI checks for cli crate
ci-cli:
	cd next-plaid-cli && cargo fmt --all -- --check
	cd next-plaid-cli && cargo clippy --all-targets -- -D warnings
	cd next-plaid-cli && cargo test

# Run API integration tests (starts server, runs Python tests, cleans up)
test-api-integration:
	./scripts/run-api-tests.sh

# Run API rate limiting tests (must run serially due to timing sensitivity)
test-api-rate-limit:
	cd next-plaid-api && cargo test test_rate_limiting --features accelerate -- --test-threads=1

# Install git hooks
install-hooks:
	./scripts/install-hooks.sh

# Lint Python code
lint-python:
	cd benchmarks && uv run --extra dev ruff check .

# Format Python code
fmt-python:
	cd benchmarks && uv run --extra dev ruff format .


launch-api-debug:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf next-plaid-api/indices
	cd next-plaid-api && RUST_LOG=debug cargo run --release

# =============================================================================
# ONNX ColBERT targets
# =============================================================================

# Set up ONNX Python environment
onnx-setup:
	cd next-plaid-onnx/python && uv sync

# Export default model (GTE-ModernColBERT-v1) to ONNX
onnx-export:
	cd next-plaid-onnx/python && uv run python export_onnx.py

# Export all supported models to ONNX
onnx-export-all:
	cd next-plaid-onnx/python && uv run python export_onnx.py --all

# Lint ONNX Python code
onnx-lint:
	cd next-plaid-onnx/python && uv run --extra dev ruff check .

# Format ONNX Python code
onnx-fmt:
	cd next-plaid-onnx/python && uv run --extra dev ruff format .

# Benchmark SciFact via Docker container with server-side encoding
# Uses next-plaid SDK, starts docker compose, runs benchmark, then stops container
benchmark-scifact-docker:
	cd benchmarks && uv sync --extra eval && uv run python benchmark_scifact_docker.py --batch-size 30

# Benchmark SciFact via Docker container (keeps container running after)
benchmark-scifact-docker-keep:
	cd benchmarks && uv sync --extra eval && uv run python benchmark_scifact_docker.py --batch-size 30 --keep-running

# Benchmark fast-plaid index format compatibility with next-plaid-api
# Creates a fast-plaid format index, loads it with next-plaid-api, and validates ndcg@10 ~ 0.74
benchmark-fastplaid-compat:
	cd benchmarks && uv sync --extra eval && uv run python benchmark_fastplaid_compat.py

# =============================================================================
# Docker targets
# =============================================================================

# Build Docker image (CPU with model support, default)
docker-build:
	docker build -t next-plaid-api -f next-plaid-api/Dockerfile .

# Build Docker image with CUDA support (GPU encoding)
docker-build-cuda:
	docker build -t next-plaid-api:cuda -f next-plaid-api/Dockerfile --target runtime-cuda .

# Start Docker Compose (CPU with model support)
docker-up:
	docker compose up -d

# Start Docker Compose with CUDA support (GPU encoding)
# Requires: NVIDIA Container Toolkit
docker-up-cuda: docker-build-cuda
	docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d

# Stop Docker Compose
docker-down:
	docker compose down

# View Docker Compose logs
docker-logs:
	docker compose logs -f

# =============================================================================
# Documentation targets
# =============================================================================

# Serve documentation locally (hot-reload) on an available port
docs-serve:
	pip install mkdocs-material -q
	@PORT=$$(python -c "import socket; s=socket.socket(); s.bind(('',0)); port=s.getsockname()[1]; s.close(); print(port)"); \
	echo "Starting docs server on port $$PORT..."; \
	mkdocs serve -a 127.0.0.1:$$PORT

# Build documentation
docs-build:
	pip install mkdocs-material -q
	mkdocs build

# Deploy documentation to GitHub Pages
docs-deploy:
	pip install mkdocs-material -q
	mkdocs gh-deploy --force

start-cpu-docker-build:
	docker build -t next-plaid-api -f next-plaid-api/Dockerfile --target runtime-cpu .

start-cuda-docker-build:
	docker build -t next-plaid-api:cuda -f next-plaid-api/Dockerfile --target runtime-cuda .