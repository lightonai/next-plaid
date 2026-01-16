.PHONY: all build test lint fmt check clean bench doc example install-hooks compare-reference lint-python fmt-python evaluate-scifact evaluate-scifact-cached compare-scifact compare-scifact-cached benchmark-scifact-update benchmark-scifact-api benchmark-scifact-docker benchmark-scifact-docker-keep benchmark-api-encoding benchmark-onnx-api benchmark-onnx-api-cuda benchmark-onnx-api-gte benchmark-onnx-api-gte-int8 benchmark-onnx-vs-pylate ci-api ci-onnx test-api-integration test-api-rate-limit onnx-setup onnx-export onnx-export-all onnx-benchmark onnx-benchmark-rust onnx-compare onnx-lint onnx-fmt docker-build docker-build-cuda docker-up docker-up-cuda docker-down docker-logs kill-api docs-serve docs-build docs-deploy

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

# Run benchmarks
bench:
	cargo bench

# Compile benchmarks without running
bench-check:
	cargo bench --no-run

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
ci: fmt-check clippy test doc bench-check ci-api ci-onnx
	@echo "All CI checks passed!"

# Kill any existing API process on port 8080
kill-api:
	@if lsof -t -i:8080 >/dev/null 2>&1; then \
		echo "Killing process on port 8080..."; \
		kill -9 $$(lsof -t -i:8080) 2>/dev/null || true; \
		sleep 2; \
	fi

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

# Benchmark SciFact with updates (batch size 800) - compares fast-plaid and next-plaid
benchmark-scifact-update:
	cargo build --release --example benchmark_cli
	cd benchmarks && uv sync --extra eval --extra fast-plaid && uv run python benchmark_scifact_update.py --batch-size 800

# Benchmark SciFact via REST API (uses cached embeddings, with accelerate)
# Starts the server in background, runs benchmark, then stops server
benchmark-scifact-api:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf next-plaid-api/indices
	cargo build --release -p next-plaid-api  --features accelerate
	./target/release/next-plaid-api -h 127.0.0.1 -p 8080 -d ./next-plaid-api/indices & \
	API_PID=$$!; \
	sleep 2; \
	cd benchmarks && uv sync --extra eval && uv run python benchmark_scifact_api.py --batch-size 80; \
	EXIT_CODE=$$?; \
	kill $$API_PID 2>/dev/null || true; \
	exit $$EXIT_CODE

# Benchmark Next-Plaid API (uses cached embeddings, with accelerate)
# Starts the server in background, runs benchmark, then stops server
# Benchmark also the delete operations
benchmark-next-plaid-api:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf next-plaid-api/indices
	cargo build --release -p next-plaid-api  --features accelerate
	./target/release/next-plaid-api -h 127.0.0.1 -p 8080 -d ./next-plaid-api/indices & \
	API_PID=$$!; \
	sleep 2; \
	cd benchmarks && uv sync --extra eval && uv run python benchmark_next_plaid_api.py --batch-size 80; \
	EXIT_CODE=$$?; \
	kill $$API_PID 2>/dev/null || true; \
	exit $$EXIT_CODE

# Benchmark SciFact with server-side encoding (API encodes text internally)
# Requires model feature and a downloaded ONNX model
benchmark-api-encoding:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf next-plaid-api/indices
	cargo build --release -p next-plaid-api --features model,cuda
	./target/release/next-plaid-api -h 127.0.0.1 -p 8080 -d ./next-plaid-api/indices --model lightonai/GTE-ModernColBERT-v1-onnx --int8 & \
	API_PID=$$!; \
	sleep 3; \
	cd benchmarks && uv sync --extra eval && uv run python benchmark_scifact_api_encoding.py --batch-size 100; \
	EXIT_CODE=$$?; \
	kill $$API_PID 2>/dev/null || true; \
	exit $$EXIT_CODE

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

# Run Python benchmark (PyLate vs ONNX-Python)
onnx-benchmark:
	cd next-plaid-onnx/python && uv run python generate_reference.py --benchmark

# Run Rust ONNX benchmark (compares against PyLate)
onnx-benchmark-rust: onnx-benchmark
	cd next-plaid-onnx && cargo run --release --bin benchmark_encoding -- --model-dir models/GTE-ModernColBERT-v1

# Compare Rust ONNX embeddings against PyLate reference
onnx-compare:
	cd next-plaid-onnx/python && uv run python generate_reference.py
	cd next-plaid-onnx && cargo run --release --bin compare_pylate

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

# Benchmark ONNX vs PyLate embeddings (asserts equivalence)
# Compares embeddings from Rust ONNX against PyLate for the same texts
benchmark-onnx-vs-pylate:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf next-plaid-api/indices
	cargo build --release -p next-plaid-api --features model
	./target/release/next-plaid-api -h 127.0.0.1 -p 8080 -d ./next-plaid-api/indices --model lightonai/GTE-ModernColBERT-v1-onnx & \
	API_PID=$$!; \
	sleep 3; \
	cd benchmarks && uv sync --extra eval && uv run python benchmark_onnx_vs_pylate.py --batch-size 10; \
	EXIT_CODE=$$?; \
	kill $$API_PID 2>/dev/null || true; \
	exit $$EXIT_CODE

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