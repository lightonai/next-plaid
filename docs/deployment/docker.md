# Docker Deployment

Deploy NextPlaid using Docker containers.

## Quick Start

```bash
docker pull ghcr.io/lightonai/next-plaid-api:latest

docker run -d \
  --name next-plaid-api \
  -p 8080:8080 \
  -v ~/.local/share/next-plaid:/data/indices \
  ghcr.io/lightonai/next-plaid-api:latest
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

## Docker Compose

### CPU with Model Support (Default)

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
    command:
      [
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--index-dir",
        "/data/indices",
        "--model",
        "lightonai/GTE-ModernColBERT-v1-onnx",
        "--int8",
      ]
    healthcheck:
      test:
        ["CMD", "curl", "-f", "--max-time", "5", "http://localhost:8080/health"]
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

### CUDA (GPU Encoding)

Use as an overlay with `docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d`:

```yaml
# docker-compose.cuda.yml
services:
  next-plaid-api:
    build:
      target: runtime-cuda
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    command:
      [
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--index-dir",
        "/data/indices",
        "--model",
        "lightonai/GTE-ModernColBERT-v1-onnx",
        "--cuda",
      ]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

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

| Variable            | Default                          | Description                                 |
| ------------------- | -------------------------------- | ------------------------------------------- |
| `RUST_LOG`          | `info`                           | Log level                                   |
| `HF_TOKEN`          | -                                | HuggingFace token for private models        |
| `NEXT_PLAID_DATA`   | `~/.local/share/next-plaid`      | Host path for persistent index storage      |
| `NEXT_PLAID_MODELS` | `~/.cache/huggingface/next-plaid`| Host path for downloaded model cache        |

### Volume Mounts

| Container Path  | Default Host Path                   | Purpose                                        |
| --------------- | ----------------------------------- | ---------------------------------------------- |
| `/data/indices` | `~/.local/share/next-plaid`         | Index storage (persistent)                     |
| `/models`       | `~/.cache/huggingface/next-plaid`   | Model cache (auto-downloaded from HuggingFace) |

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
      memory: 8G
      cpus: "4"
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
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
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
