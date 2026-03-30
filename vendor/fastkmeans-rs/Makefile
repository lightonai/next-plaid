.PHONY: all build test lint fmt check clean bench doc example install-hooks compare-reference benchmark-comparison

# Auto-detect BLAS feature based on OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    BLAS_FEATURE := accelerate
else
    BLAS_FEATURE := openblas
endif

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

# Compare with Python fastkmeans reference implementation
compare-reference:
	cargo build --release --features "npy,$(BLAS_FEATURE)" --bin compare-kmeans
	uv run benches/compare_reference.py

# Run performance benchmark comparison
benchmark-comparison:
	cargo build --release --features "npy,$(BLAS_FEATURE)" --bin compare-kmeans
	uv run benches/benchmark_comparison.py

# Build with BLAS acceleration
build-blas:
	cargo build --release --features $(BLAS_FEATURE)

# Test with BLAS acceleration
test-blas:
	cargo test --features $(BLAS_FEATURE)
