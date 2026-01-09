.PHONY: all build test lint fmt check clean bench doc example install-hooks compare-reference lint-python fmt-python evaluate-scifact evaluate-scifact-cached compare-scifact compare-scifact-cached benchmark-scifact-update benchmark-scifact-api benchmark-onnx-api ci-api test-api-integration test-api-rate-limit

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
	cargo clippy --all-targets --all-features -- -D warnings

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
	cargo doc --no-deps --all-features

# Open documentation in browser
doc-open:
	cargo doc --no-deps --all-features --open

# Clean build artifacts
clean:
	cargo clean

# Run the basic example
example:
	cargo run --example basic --release

# Run all CI checks locally
ci: fmt-check clippy test doc bench-check ci-api test-api-integration
	@echo "All CI checks passed!"

# Run CI checks for api crate
ci-api:
	cd api && cargo fmt --all -- --check
	cd api && cargo clippy --all-targets --all-features -- -D warnings
	cd api && cargo test

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
	cargo build --release -p lategrep-api --features accelerate
	./target/release/lategrep-api -h 127.0.0.1 -p 8080 -d ./api/indices & \
	API_PID=$$!; \
	sleep 2; \
	cd docs && uv sync --extra eval && uv run python benchmark_scifact_api.py --batch-size 80; \
	EXIT_CODE=$$?; \
	kill $$API_PID 2>/dev/null || true; \
	exit $$EXIT_CODE

# Benchmark SciFact with ONNX embeddings via REST API
# Uses ONNX (Rust) for encoding and Lategrep REST API for indexing/search
benchmark-onnx-api:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf api/indices
	cargo build --release -p lategrep-api --features accelerate
	./target/release/lategrep-api -h 127.0.0.1 -p 8080 -d ./api/indices & \
	API_PID=$$!; \
	sleep 2; \
	cd docs && uv sync --extra eval && uv run python onnx_benchmark.py --skip-encoding --batch-size 80; \
	EXIT_CODE=$$?; \
	kill $$API_PID 2>/dev/null || true; \
	exit $$EXIT_CODE

launch-api-debug:
	-kill -9 $$(lsof -t -i:8080) 2>/dev/null || true
	rm -rf api/indices
	cd api && RUST_LOG=debug cargo run --release
