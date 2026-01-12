# =============================================================================
# Next-Plaid API Dockerfile
# =============================================================================
# Supports three build variants using --target:
#   1. runtime-cpu (default): Search API with OpenBLAS, no model
#   2. runtime-model: + Text encoding on CPU
#   3. runtime-cuda: + GPU-accelerated encoding
#
# Build examples:
#   docker build -t next-plaid-api -f next-plaid-api/Dockerfile .
#   docker build -t next-plaid-api:model -f next-plaid-api/Dockerfile --target runtime-model .
#   docker build -t next-plaid-api:cuda -f next-plaid-api/Dockerfile --target runtime-cuda .
#
# Run examples (with persistent storage):
#   # CPU only - indices stored at ~/.local/share/next-plaid/<index-name>/
#   docker run -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices next-plaid-api
#
#   # With model support (CPU encoding)
#   docker run -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices -v models:/models \
#     next-plaid-api:model --model /models/colbert
#
#   # With CUDA support (GPU encoding)
#   docker run -p 8080:8080 --gpus all -v ~/.local/share/next-plaid:/data/indices -v models:/models \
#     next-plaid-api:cuda --model /models/colbert --cuda
#
# Auto-download model from HuggingFace Hub:
#   docker run -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices -v models:/models \
#     next-plaid-api:cuda --model lightonai/GTE-ModernColBERT-v1-onnx --cuda
#
#   # With INT8 quantization (faster):
#   docker run -p 8080:8080 -v ~/.local/share/next-plaid:/data/indices -v models:/models \
#     next-plaid-api:cuda --model lightonai/GTE-ModernColBERT-v1-onnx --int8 --cuda
#
# Vector Database Persistence:
#   Default location: ~/.local/share/next-plaid/<index-name>/
#   Customize via NEXT_PLAID_DATA env var or mount any path to /data/indices
#   On container restart, existing indices are automatically loaded when accessed.
#
# Environment variables (optional):
#   HF_TOKEN: Token for private HuggingFace models
#   MODELS_DIR: Directory to store downloaded models (default: /models)
# =============================================================================

# =============================================================================
# Builder stage - CPU only (no model support)
# =============================================================================
FROM rust:1.83-bookworm AS builder-cpu

WORKDIR /app

# Install dependencies for building (including OpenBLAS for BLAS acceleration)
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the workspace files (excluding tests which are in .dockerignore)
COPY Cargo.toml Cargo.lock ./
COPY next-plaid ./next-plaid
COPY next-plaid-api ./next-plaid-api
COPY next-plaid-onnx ./next-plaid-onnx

# Build the API in release mode with OpenBLAS
RUN cargo build --release --package next-plaid-api --features openblas

# =============================================================================
# Builder stage - CPU with model support
# =============================================================================
FROM rust:1.83-bookworm AS builder-model

WORKDIR /app

# Install dependencies for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the workspace files
COPY Cargo.toml Cargo.lock ./
COPY next-plaid ./next-plaid
COPY next-plaid-api ./next-plaid-api
COPY next-plaid-onnx ./next-plaid-onnx

# Build with model support (CPU)
RUN cargo build --release --package next-plaid-api --features "openblas,model"

# =============================================================================
# Builder stage - CUDA with model support
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder-cuda

WORKDIR /app

# Install Rust and build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libopenblas-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Download ONNX Runtime GPU (version must match ort crate - requires >= 1.23.x)
ARG ORT_VERSION=1.23.0
RUN mkdir -p /opt/ort_gpu && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz && \
    cp -r onnxruntime-linux-x64-gpu-${ORT_VERSION}/lib/* /opt/ort_gpu/ && \
    rm -rf onnxruntime-linux-x64-gpu-${ORT_VERSION} onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz

# Copy the workspace files
COPY Cargo.toml Cargo.lock ./
COPY next-plaid ./next-plaid
COPY next-plaid-api ./next-plaid-api
COPY next-plaid-onnx ./next-plaid-onnx

# Set environment for CUDA build
ENV ORT_DYLIB_PATH=/opt/ort_gpu/libonnxruntime.so.${ORT_VERSION}
ENV LD_LIBRARY_PATH=/opt/ort_gpu:${LD_LIBRARY_PATH}

# Build with CUDA support
RUN cargo build --release --package next-plaid-api --features "openblas,cuda"

# =============================================================================
# Runtime stage - CPU only (default, smallest image)
# =============================================================================
FROM debian:bookworm-slim AS runtime-cpu

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash nextplaid

# Create indices directory
RUN mkdir -p /data/indices && chown -R nextplaid:nextplaid /data

# Copy binary from builder
COPY --from=builder-cpu /app/target/release/next-plaid-api /usr/local/bin/next-plaid-api

# Switch to non-root user
USER nextplaid

# Expose API port
EXPOSE 8080

# Default environment variables
ENV RUST_LOG=info
ENV INDEX_DIR=/data/indices

# Health check - aggressive settings for faster freeze detection
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=2 \
    CMD curl -f --max-time 5 http://localhost:8080/health || exit 1

# Run the API
ENTRYPOINT ["next-plaid-api"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--index-dir", "/data/indices"]

# =============================================================================
# Runtime stage - CPU with model support
# =============================================================================
FROM debian:bookworm-slim AS runtime-model

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash nextplaid

# Create directories for indices and models
RUN mkdir -p /data/indices /models && chown -R nextplaid:nextplaid /data /models

# Copy binary from builder
COPY --from=builder-model /app/target/release/next-plaid-api /usr/local/bin/next-plaid-api

# Copy entrypoint script
COPY --chmod=755 next-plaid-api/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER nextplaid

# Expose API port
EXPOSE 8080

# Default environment variables
ENV RUST_LOG=info
ENV INDEX_DIR=/data/indices
# Set HF_MODEL_ID to auto-download model from HuggingFace Hub
# Example: ENV HF_MODEL_ID=lightonai/GTE-ModernColBERT-v1-onnx
# Set MODEL_QUANTIZED=true to use INT8 quantized model

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=2 \
    CMD curl -f --max-time 5 http://localhost:8080/health || exit 1

# Run the API via entrypoint script (handles model download)
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--index-dir", "/data/indices"]

# =============================================================================
# Runtime stage - CUDA with model support
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime-cuda

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash nextplaid

# Create directories for indices, models, and ONNX Runtime
RUN mkdir -p /data/indices /models /opt/ort_gpu && chown -R nextplaid:nextplaid /data /models

# Copy ONNX Runtime GPU libraries from builder
COPY --from=builder-cuda /opt/ort_gpu /opt/ort_gpu

# Copy binary from builder
COPY --from=builder-cuda /app/target/release/next-plaid-api /usr/local/bin/next-plaid-api

# Copy entrypoint script
COPY --chmod=755 next-plaid-api/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Set library paths for CUDA and ONNX Runtime
ARG ORT_VERSION=1.23.0
ENV ORT_DYLIB_PATH=/opt/ort_gpu/libonnxruntime.so.${ORT_VERSION}
ENV LD_LIBRARY_PATH=/opt/ort_gpu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Switch to non-root user
USER nextplaid

# Expose API port
EXPOSE 8080

# Default environment variables
ENV RUST_LOG=info
ENV INDEX_DIR=/data/indices
# Set HF_MODEL_ID to auto-download model from HuggingFace Hub
# Example: ENV HF_MODEL_ID=lightonai/GTE-ModernColBERT-v1-onnx
# Set MODEL_QUANTIZED=true to use INT8 quantized model

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=2 \
    CMD curl -f --max-time 5 http://localhost:8080/health || exit 1

# Run the API via entrypoint script (handles model download)
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--index-dir", "/data/indices"]
