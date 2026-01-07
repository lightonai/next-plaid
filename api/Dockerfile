# Build stage
FROM rust:1.83-bookworm AS builder

WORKDIR /app

# Install dependencies for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the workspace files (excluding tests which are in .dockerignore)
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY api ./api
COPY benches ./benches
COPY examples ./examples
# Create empty tests directory to satisfy Cargo.toml references
RUN mkdir -p tests

# Build the API in release mode
RUN cargo build --release --package lategrep-api

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash lategrep

# Create indices directory
RUN mkdir -p /data/indices && chown -R lategrep:lategrep /data

# Copy binary from builder
COPY --from=builder /app/target/release/lategrep-api /usr/local/bin/lategrep-api

# Switch to non-root user
USER lategrep

# Expose API port
EXPOSE 8080

# Default environment variables
ENV RUST_LOG=info
ENV INDEX_DIR=/data/indices

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the API
ENTRYPOINT ["lategrep-api"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--index-dir", "/data/indices"]
