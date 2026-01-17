# Installation

Choose the installation method that best fits your use case.

## Docker (Recommended)

Run NextPlaid in production using Docker containers. Both images support model inference for text encoding. The PLAID index and search always run on CPU, while model inference runs on CPU or GPU depending on the image.

### Pre-built Images

=== "CPU"

    ```bash
    docker pull ghcr.io/lightonai/next-plaid-api:latest

    docker run -d \
      --name next-plaid-api \
      -p 8080:8080 \
      -v ~/.local/share/next-plaid:/data/indices \
      -v next-plaid-models:/models \
      ghcr.io/lightonai/next-plaid-api:latest \
      --model lightonai/GTE-ModernColBERT-v1-onnx
    ```

=== "CUDA"

    ```bash
    docker pull ghcr.io/lightonai/next-plaid-api:latest-cuda

    docker run -d \
      --name next-plaid-api \
      --gpus all \
      -p 8080:8080 \
      -v ~/.local/share/next-plaid:/data/indices \
      -v next-plaid-models:/models \
      ghcr.io/lightonai/next-plaid-api:latest-cuda \
      --model lightonai/GTE-ModernColBERT-v1-onnx
    ```

### Docker Compose

Clone the repository and use Docker Compose:

```bash
git clone https://github.com/lightonai/next-plaid.git
cd next-plaid

# CPU only
docker compose up -d

# With CUDA support
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
```

### Image Tags

The index and search always run on CPU. Model inference runs on CPU or GPU depending on the image.

| Tag | Description | Model Inference |
|-----|-------------|-----------------|
| `latest` | Latest CPU release | CPU |
| `X.Y.Z` | Specific version (CPU) | CPU |
| `latest-cuda` | Latest CUDA release | GPU |
| `X.Y.Z-cuda` | Specific version (CUDA) | GPU |

---

## Python SDK

Install the Python client library:

```bash
pip install next-plaid-client
```

### Requirements

- Python 3.8 - 3.12
- httpx (installed automatically)

### Verify Installation

```python
from next_plaid_client import NextPlaidClient

client = NextPlaidClient("http://localhost:8080")
print(client.health())
```

---

## Rust Crate

For Rust developers who want to embed NextPlaid directly:

### Basic Installation

```toml
[dependencies]
next-plaid = { git = "https://github.com/lightonai/next-plaid" }
```

### With BLAS Acceleration (Recommended)

=== "macOS (Accelerate)"

    ```toml
    [dependencies]
    next-plaid = {
      git = "https://github.com/lightonai/next-plaid",
      features = ["accelerate"]
    }
    ```

=== "Linux (OpenBLAS)"

    First install OpenBLAS:

    ```bash
    # Ubuntu/Debian
    sudo apt-get install -y libopenblas-dev

    # Fedora
    sudo dnf install openblas-devel
    ```

    Then add to Cargo.toml:

    ```toml
    [dependencies]
    next-plaid = {
      git = "https://github.com/lightonai/next-plaid",
      features = ["openblas"]
    }
    ```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `accelerate` | macOS BLAS acceleration |
| `openblas` | Linux OpenBLAS acceleration |

---

## Model Export Tool

Install the CLI tool for exporting ColBERT models to ONNX:

```bash
pip install pylate-onnx-export
```

### Requirements

- Python 3.10 - 3.12
- PyTorch
- Transformers
- ONNX Runtime

### Verify Installation

```bash
pylate-onnx-export --help
```

See [Model Export](model-export.md) for usage details.

---

## Building from Source

### Prerequisites

- Rust 1.70+
- Python 3.10+ with [uv](https://docs.astral.sh/uv/)
- OpenBLAS (Linux) or Accelerate (macOS)

### Build Steps

```bash
# Clone the repository
git clone https://github.com/lightonai/next-plaid.git
cd next-plaid

# Build the API server (Linux)
cargo build --release -p next-plaid-api --features "openblas"

# Build the API server (macOS)
cargo build --release -p next-plaid-api --features "accelerate"

# Run the server
./target/release/next-plaid-api --index-dir ./indices
```

### Build Docker Images Locally

```bash
# CPU variant
docker build -t next-plaid-api:local \
  -f next-plaid-api/Dockerfile \
  --target runtime-cpu .

# CUDA variant
docker build -t next-plaid-api:local-cuda \
  -f next-plaid-api/Dockerfile \
  --target runtime-cuda .
```
