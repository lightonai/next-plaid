# Contributing

Thank you for your interest in contributing to NextPlaid!

## Development Setup

### Prerequisites

- Rust 1.70+ ([rustup](https://rustup.rs/))
- Python 3.10+ with [uv](https://docs.astral.sh/uv/)
- Docker 20.10+
- OpenBLAS (Linux) or Accelerate (macOS)

### Clone and Setup

```bash
git clone https://github.com/lightonai/next-plaid.git
cd next-plaid

# Install git hooks
make install-hooks

# Install OpenBLAS (Linux)
sudo apt-get install -y libopenblas-dev

# Run all CI checks
make ci
```

---

## Project Structure

```
next-plaid/
├── next-plaid/           # Core library (Rust)
├── next-plaid-api/       # REST API server (Rust)
│   ├── python-sdk/       # Python client SDK
│   └── Dockerfile        # Multi-stage Docker build
├── next-plaid-onnx/      # ONNX encoding support (Rust)
│   └── python/           # pylate-onnx-export CLI tool
├── benchmarks/           # Evaluation benchmarks
└── .github/workflows/    # CI/CD pipelines
```

---

## Development Workflow

### Building

```bash
# Debug build
make build

# Release build
make release

# Build with specific features
cargo build --release -p next-plaid --features "npy,filtering,openblas"
```

### Testing

```bash
# Run all Rust tests
make test

# Run with specific features
cargo test --features "npy,filtering,openblas"

# Run API integration tests
make test-api

# Run Python SDK tests
cd next-plaid-api/python-sdk && pytest tests/
```

### Linting and Formatting

```bash
# Format code
make fmt

# Run clippy
make lint

# Format and lint Python
make fmt-python
make lint-python
```

### Documentation

```bash
# Build Rust documentation
make doc

# Open in browser
cargo doc --open
```

---

## Docker Development

### Building Images

```bash
# CPU-only image
make docker-build

# With CUDA support
docker build -t next-plaid-api:cuda \
  -f next-plaid-api/Dockerfile \
  --target runtime-cuda .
```

### Running Locally

```bash
# Start with Docker Compose
make docker-up

# View logs
make docker-logs

# Stop
make docker-down
```

---

## Benchmarking

```bash
# Run SciFact benchmark
make benchmark-scifact-update

# Compare with FastPlaid
make compare-scifact-cached

# Evaluate retrieval quality
make evaluate-scifact-cached
```

---

## Pull Request Guidelines

1. **Create a feature branch** from `main`
2. **Write tests** for new functionality
3. **Run CI checks locally**: `make ci`
4. **Keep commits focused** - one logical change per commit
5. **Write clear commit messages**

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples:**

```
feat(api): add batch encoding endpoint
fix(search): correct score normalization for empty queries
docs: update installation instructions
chore: bump dependencies
```

---

## Code Style

### Rust

- Follow standard Rust conventions
- Run `cargo fmt` before committing
- Address all `clippy` warnings
- Document public APIs with doc comments

### Python

- Follow PEP 8
- Use type hints
- Format with `black` and `isort`
- Lint with `ruff`

---

## Testing Requirements

### Unit Tests

- Test individual functions and modules
- Use descriptive test names
- Cover edge cases

### Integration Tests

- Test API endpoints end-to-end
- Test Python SDK against running server
- Use fixtures for test data

### Benchmarks

- Run benchmarks before and after changes
- Document performance impact in PR

---

## Release Process

Releases are automated via GitHub Actions when a version tag is pushed.

### What Gets Published

| Package | Registry |
|---------|----------|
| `next-plaid` | crates.io |
| `next-plaid-onnx` | crates.io |
| `next-plaid-client` | PyPI |
| `pylate-onnx-export` | PyPI |
| `next-plaid-api` | GitHub Container Registry |

### Creating a Release

1. **Update version numbers** in:
   - `/Cargo.toml` (workspace version)
   - `/next-plaid-api/python-sdk/pyproject.toml`
   - `/next-plaid-onnx/python/pyproject.toml`

2. **Commit the version bump**:
   ```bash
   git add -A
   git commit -m "chore: release v0.2.0"
   ```

3. **Create and push the tag**:
   ```bash
   git tag v0.2.0
   git push origin main --tags
   ```

4. **Monitor the release** in GitHub Actions

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/lightonai/next-plaid/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lightonai/next-plaid/discussions)

---

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
