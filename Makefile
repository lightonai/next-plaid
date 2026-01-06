.PHONY: all build test lint fmt check clean bench doc example install-hooks compare-reference lint-python fmt-python

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
ci: fmt-check clippy test doc bench-check
	@echo "All CI checks passed!"

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
