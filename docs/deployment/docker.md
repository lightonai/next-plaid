# Docker Deployment

Deploy NextPlaid using Docker containers.

## Quick Start

```bash
docker pull ghcr.io/lightonai/next-plaid-api:latest

docker run -d \
  --name next-plaid-api \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  -v ~/.cache/huggingface/next-plaid:/models \
  ghcr.io/lightonai/next-plaid-api:latest \
  --model lightonai/GTE-ModernColBERT-v1-onnx
```

Verify:

```bash
curl http://localhost:8080/health
```

---

## Image Variants

The PLAID index and search always run on CPU. Model inference for text encoding can run on CPU or GPU depending on the image.

| Tag           | Description    | Index & Search | Model Inference |
| ------------- | -------------- | -------------- | --------------- |
| `latest`      | CPU image      | CPU            | CPU             |
| `X.Y.Z`       | Versioned CPU  | CPU            | CPU             |
| `latest-cuda` | CUDA image     | CPU            | GPU             |
| `X.Y.Z-cuda`  | Versioned CUDA | CPU            | GPU             |

---

## Building from Source

The Dockerfile supports two build targets:

```bash
# CPU build (default)
docker build -t next-plaid-api -f next-plaid-api/Dockerfile .

# CUDA build (GPU encoding)
docker build -t next-plaid-api:cuda -f next-plaid-api/Dockerfile --target runtime-cuda .
```

---

## CLI Options

The API accepts the following command-line options:

| Option              | Default                              | Description                               |
| ------------------- | ------------------------------------ | ----------------------------------------- |
| `--model <path>`    | -                                    | HuggingFace model ID or local path        |
| `--int8`            | `false`                              | Use INT8 quantized model (~2x faster CPU) |
| `--cuda`            | `false`                              | Use CUDA for inference (GPU builds only)  |
| `--parallel <N>`    | `1`                                  | Number of parallel ONNX sessions          |
| `--batch-size <N>`  | `1`                                  | Batch size per session                    |
| `--threads <N>`     | auto                                 | Threads per ONNX session                  |
| `--query-length`    | `48`                                 | Max query length in tokens                |
| `--document-length` | `300`                                | Max document length in tokens             |
| `--host`            | `127.0.0.1`                          | Host to bind                              |
| `--port`            | `8080`                               | Port to bind                              |
| `--index-dir`       | `/data/indices`                      | Directory for index storage               |

### Model Download

When `--model` contains a HuggingFace model ID (e.g., `lightonai/GTE-ModernColBERT-v1-onnx`), the entrypoint script automatically downloads the model to `/models/`. Set `HF_TOKEN` for private models.

---

## Docker Compose

### CPU Configuration

```yaml
# docker-compose.yml
services:
  next-plaid-api:
    build:
      context: .
      dockerfile: next-plaid-api/Dockerfile
      target: runtime-cpu
    ports:
      - "8080:8080"
    volumes:
      # Persistent index storage
      - ${NEXT_PLAID_DATA:-~/.local/share/next-plaid}:/data/indices
      # Persistent model cache (downloaded from HuggingFace)
      - ${NEXT_PLAID_MODELS:-~/.cache/huggingface/next-plaid}:/models
    environment:
      - RUST_LOG=info
      # Rate limiting (configurable)
      - RATE_LIMIT_PER_SECOND=${RATE_LIMIT_PER_SECOND:-50}
      - RATE_LIMIT_BURST_SIZE=${RATE_LIMIT_BURST_SIZE:-100}
      - CONCURRENCY_LIMIT=${CONCURRENCY_LIMIT:-100}
      # Document processing
      - MAX_QUEUED_TASKS_PER_INDEX=${MAX_QUEUED_TASKS_PER_INDEX:-10}
      - MAX_BATCH_DOCUMENTS=${MAX_BATCH_DOCUMENTS:-300}
      - BATCH_CHANNEL_SIZE=${BATCH_CHANNEL_SIZE:-100}
      # Encode batching
      - MAX_BATCH_TEXTS=${MAX_BATCH_TEXTS:-64}
      - ENCODE_BATCH_CHANNEL_SIZE=${ENCODE_BATCH_CHANNEL_SIZE:-256}
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8080"
      - --index-dir
      - /data/indices
      - --model
      - lightonai/GTE-ModernColBERT-v1-onnx
      - --parallel
      - "4"
      - --batch-size
      - "12"
      - --query-length
      - "48"
      - --document-length
      - "300"
    healthcheck:
      test: ["CMD", "curl", "-f", "--max-time", "5", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 2
      start_period: 120s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 4G
```

### CUDA Configuration (GPU Encoding)

Use as an overlay with `docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d`:

```yaml
# docker-compose.cuda.yml
services:
  next-plaid-api:
    build:
      context: .
      dockerfile: next-plaid-api/Dockerfile
      target: runtime-cuda
    volumes:
      - ${NEXT_PLAID_DATA:-~/.local/share/next-plaid}:/data/indices
      - ${NEXT_PLAID_MODELS:-~/.cache/huggingface/next-plaid}:/models
    environment:
      - RUST_LOG=info
      - NVIDIA_VISIBLE_DEVICES=all
      # Rate limiting (configurable)
      - RATE_LIMIT_PER_SECOND=${RATE_LIMIT_PER_SECOND:-50}
      - RATE_LIMIT_BURST_SIZE=${RATE_LIMIT_BURST_SIZE:-100}
      - CONCURRENCY_LIMIT=${CONCURRENCY_LIMIT:-100}
      # Document processing
      - MAX_QUEUED_TASKS_PER_INDEX=${MAX_QUEUED_TASKS_PER_INDEX:-10}
      - MAX_BATCH_DOCUMENTS=${MAX_BATCH_DOCUMENTS:-300}
      - BATCH_CHANNEL_SIZE=${BATCH_CHANNEL_SIZE:-100}
      # Encode batching
      - MAX_BATCH_TEXTS=${MAX_BATCH_TEXTS:-64}
      - ENCODE_BATCH_CHANNEL_SIZE=${ENCODE_BATCH_CHANNEL_SIZE:-256}
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "8080"
      - --index-dir
      - /data/indices
      - --model
      - lightonai/GTE-ModernColBERT-v1-onnx
      - --cuda
      - --batch-size
      - "64"
      - --query-length
      - "48"
      - --document-length
      - "300"
    healthcheck:
      start_period: 120s
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**CPU vs CUDA defaults:**

- **CPU**: `--parallel 4 --batch-size 12` (multiple parallel sessions for throughput)
- **CUDA**: `--cuda --batch-size 64` (single session, GPU handles parallelism with large batches)

---

## Running Docker Compose

```bash
# CPU with model support (default)
docker compose up -d

# With CUDA (GPU encoding)
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

---

## Configuration

### Environment Variables

#### General

| Variable            | Default                           | Description                            |
| ------------------- | --------------------------------- | -------------------------------------- |
| `RUST_LOG`          | `info`                            | Log level                              |
| `HF_TOKEN`          | -                                 | HuggingFace token for private models   |
| `NEXT_PLAID_DATA`   | `~/.local/share/next-plaid`       | Host path for persistent index storage |
| `NEXT_PLAID_MODELS` | `~/.cache/huggingface/next-plaid` | Host path for downloaded model cache   |
| `MODELS_DIR`        | `/models`                         | Container path for models              |

#### Rate Limiting & Concurrency

| Variable                     | Default | Description                                      |
| ---------------------------- | ------- | ------------------------------------------------ |
| `RATE_LIMIT_PER_SECOND`      | `50`    | Max requests per second (sustained rate)         |
| `RATE_LIMIT_BURST_SIZE`      | `100`   | Burst size for rate limiting                     |
| `CONCURRENCY_LIMIT`          | `100`   | Max concurrent in-flight requests                |
| `MAX_QUEUED_TASKS_PER_INDEX` | `10`    | Max queued updates/deletes per index (semaphore) |
| `MAX_BATCH_DOCUMENTS`        | `300`   | Max documents to batch before processing         |
| `BATCH_CHANNEL_SIZE`         | `100`   | Buffer size for document batch queue             |
| `MAX_BATCH_TEXTS`            | `64`    | Max texts to batch for encoding                  |
| `ENCODE_BATCH_CHANNEL_SIZE`  | `256`   | Buffer size for encode batch queue               |

These variables can be set in your `docker-compose.yml` or via a `.env` file.

### Volume Mounts

| Container Path  | Default Host Path                 | Purpose                                        |
| --------------- | --------------------------------- | ---------------------------------------------- |
| `/data/indices` | `~/.local/share/next-plaid`       | Index storage (persistent)                     |
| `/models`       | `~/.cache/huggingface/next-plaid` | Model cache (auto-downloaded from HuggingFace) |

Models are downloaded once from HuggingFace and cached locally. On subsequent container starts, cached models are reused without re-downloading.

### Ports

| Port | Protocol | Description |
| ---- | -------- | ----------- |
| 8080 | HTTP     | API server  |

---

## Scaling

### Single Instance

For most use cases, a single container is sufficient:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
    reservations:
      memory: 4G
```

### Multiple Instances (Load Balancing)

For high throughput, run multiple instances behind a load balancer:

```yaml
services:
  next-plaid:
    image: ghcr.io/lightonai/next-plaid-api:latest
    deploy:
      replicas: 3
    volumes:
      - /shared/indices:/data/indices:ro # Read-only shared storage

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - next-plaid
```

!!! warning "Replicas and Index Updates"
    When running multiple replicas, **do not update the index** as NextPlaid does not currently handle replica synchronization. Each replica should either:

    - Point to a **read-only** shared index (as shown above)
    - Point to a **distinct index** if write operations are required

    Index updates on one replica will not propagate to others and may cause data inconsistencies.

---

## Health Checks

The container includes a health check:

```dockerfile
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=2 \
  CMD curl -f --max-time 5 http://localhost:8080/health || exit 1
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 15

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 15
```

---

## Logging

### Log Levels

Set via `RUST_LOG` environment variable:

| Level   | Description                   |
| ------- | ----------------------------- |
| `error` | Errors only                   |
| `warn`  | Warnings and errors           |
| `info`  | General information (default) |
| `debug` | Detailed debugging            |
| `trace` | Very verbose                  |

### JSON Logging

For production, configure structured logging:

```bash
docker run -e RUST_LOG=info,tower_http=debug \
  ghcr.io/lightonai/next-plaid-api:latest
```
