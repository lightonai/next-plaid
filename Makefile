.PHONY: all build test lint fmt check clean bench doc example install-hooks compare-reference lint-python fmt-python evaluate-scifact evaluate-scifact-cached compare-scifact compare-scifact-cached benchmark-scifact-update benchmark-scifact-api benchmark-api-encoding benchmark-onnx-api benchmark-onnx-api-cuda benchmark-onnx-api-gte benchmark-onnx-api-gte-int8 ci-api ci-onnx test-api-integration test-api-rate-limit onnx-setup onnx-export onnx-export-all onnx-benchmark onnx-benchmark-rust onnx-compare onnx-lint onnx-fmt docker-build docker-build-model docker-build-cuda docker-up docker-up-model docker-up-cuda docker-down docker-logs

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

# Run clippy (using cross-platform features only)
clippy:
	cargo clippy --all-targets --features npy,filtering -- -D warnings

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

# Build documentation (using cross-platform features only)
doc:
	cargo doc --no-deps --features npy,filtering

# Open documentation in browser
doc-open:
	cargo doc --no-deps --features npy,filtering --open

# Clean build artifacts
clean:
	cargo clean

# Run the basic example
example:
	cargo run --example basic --release

# Run all CI checks locally
ci: fmt-check clippy test doc bench-check ci-api test-api-integration ci-onnx
	@echo "All CI checks passed!"

# Run CI checks for api crate (using cross-platform features only)
ci-api:
	cd api && cargo fmt --all -- --check
	cd api && cargo clippy --all-targets -- -D warnings
	cd api && cargo test

# Run CI checks for onnx crate (using cross-platform features only)
ci-onnx:
	cd onnx && cargo fmt --all -- --check
	cd onnx && cargo clippy --all-targets -- -D warnings
	cd onnx && cargo test
	cd onnx/python && uv run --extra dev ruff check .
	cd onnx/python && uv run --extra dev pytest tests/

# Run API integration tests (starts server, runs Python tests, cleans up)
test-api-integration:
	./scripts/run-api-tests.sh

# Run API rate limiting tests (must run serially due to timing sensitivity)
test-api-rate-limit:
	cd api && cargo test test_rate_limiting --features accelerate -- --test-threads=1

# Install git hooks
install-hooks:
	./scripts/install-hooks.sh

# Lint Python code
lint-python:
	cd docs && uv run --extra dev ruff check .

# Format Python code
fmt-python:
	cd docs && uv run --extra dev ruff format .

# Compare with fast-plaid reference implementation
compare-reference:
	cargo build --release --features npy --example benchmark_cli
	cd docs && uv sync --extra fast-plaid && uv run python compare_reference.py --skip-cross

# Benchmark SciFact with updates (batch size 800) - compares fast-plaid and lategrep
benchmark-scifact-update:
	cargo build --release --features npy,accelerate --example benchmark_cli
	cd docs && uv sync --extra eval --extra fast-plaid && uv run python benchmark_scifact_update.py --batch-size 800

# Benchmark SciFact via REST API (uses cached embeddings, with accelerate)
# Starts the server in background, runs benchmark, then stops server
benchmark-scifact-api:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf api/indices
	cargo build --release -p lategrep-api 
	./target/release/lategrep-api -h 127.0.0.1 -p 8080 -d ./api/indices & \
	API_PID=$$!; \
	sleep 2; \
	cd docs && uv sync --extra eval && uv run python benchmark_scifact_api.py --batch-size 80; \
	EXIT_CODE=$$?; \
	kill $$API_PID 2>/dev/null || true; \
	exit $$EXIT_CODE

# Benchmark SciFact with server-side encoding (API encodes text internally)
# Requires model feature and a downloaded ONNX model
benchmark-api-encoding:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf api/indices
	cargo build --release -p lategrep-api --features model,cuda
	./target/release/lategrep-api -h 127.0.0.1 -p 8080 -d ./api/indices --model ./onnx/models/GTE-ModernColBERT-v1 & \
	API_PID=$$!; \
	sleep 3; \
	cd docs && uv sync --extra eval && uv run python benchmark_scifact_api_encoding.py --batch-size 100; \
	EXIT_CODE=$$?; \
	kill $$API_PID 2>/dev/null || true; \
	exit $$EXIT_CODE

launch-api-debug:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf api/indices
	cd api && RUST_LOG=debug cargo run --release

# =============================================================================
# ONNX ColBERT targets
# =============================================================================

# Set up ONNX Python environment
onnx-setup:
	cd onnx/python && uv sync

# Export default model (GTE-ModernColBERT-v1) to ONNX
onnx-export:
	cd onnx/python && uv run python export_onnx.py

# Export all supported models to ONNX
onnx-export-all:
	cd onnx/python && uv run python export_onnx.py --all

# Run Python benchmark (PyLate vs ONNX-Python)
onnx-benchmark:
	cd onnx/python && uv run python generate_reference.py --benchmark

# Run Rust ONNX benchmark (compares against PyLate)
onnx-benchmark-rust: onnx-benchmark
	cd onnx && cargo run --release --bin benchmark_encoding -- --model-dir models/GTE-ModernColBERT-v1

# Compare Rust ONNX embeddings against PyLate reference
onnx-compare:
	cd onnx/python && uv run python generate_reference.py
	cd onnx && cargo run --release --bin compare_pylate

# Lint ONNX Python code
onnx-lint:
	cd onnx/python && uv run --extra dev ruff check .

# Format ONNX Python code
onnx-fmt:
	cd onnx/python && uv run --extra dev ruff format .

# =============================================================================
# Docker targets
# =============================================================================

# Build Docker image (CPU only, default)
docker-build:
	docker build -t lategrep-api -f api/Dockerfile .

# Build Docker image with model support (CPU encoding)
docker-build-model:
	docker build -t lategrep-api:model -f api/Dockerfile --target runtime-model .

# Build Docker image with CUDA support (GPU encoding)
docker-build-cuda:
	docker build -t lategrep-api:cuda -f api/Dockerfile --target runtime-cuda .

# Start Docker Compose (CPU only)
docker-up:
	docker compose up -d

# Start Docker Compose with model support (CPU encoding)
# Requires models to be present in ./onnx/models/
docker-up-model: docker-build-model
	docker compose -f docker-compose.yml -f docker-compose.model.yml up -d

# Start Docker Compose with CUDA support (GPU encoding)
# Requires: NVIDIA Container Toolkit and models in ./onnx/models/
docker-up-cuda: docker-build-cuda
	docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d

# Stop Docker Compose
docker-down:
	docker compose down

# View Docker Compose logs
docker-logs:
	docker compose logs -f
