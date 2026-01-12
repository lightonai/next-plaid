# Production Deployment

Best practices for deploying NextPlaid in production.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
│                    (nginx/Traefik/ALB)                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  NextPlaid    │ │  NextPlaid    │ │  NextPlaid    │
│  Instance 1   │ │  Instance 2   │ │  Instance 3   │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │   Shared Storage    │
               │   (NFS/EFS/GCS)     │
               └─────────────────────┘
```

---

## Hardware Recommendations

### CPU

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Cores | 2 | 4-8 |
| Architecture | x86_64 | x86_64 with AVX2 |

BLAS-accelerated operations benefit from more cores.

### Memory

| Index Size | Minimum RAM | Recommended RAM |
|------------|-------------|-----------------|
| < 100K docs | 2 GB | 4 GB |
| 100K-1M docs | 4 GB | 8 GB |
| 1M-10M docs | 8 GB | 16 GB |
| > 10M docs | 16 GB+ | 32 GB+ |

### Storage

| Type | Use Case |
|------|----------|
| SSD | Recommended for all deployments |
| NVMe | Best for large indices with memory mapping |
| HDD | Not recommended (slow I/O) |

---

## Deployment Patterns

### Single Instance

For small to medium workloads:

```yaml
services:
  next-plaid:
    image: ghcr.io/lightonai/next-plaid-api:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
```

**Suitable for:**

- < 1M documents
- < 100 QPS
- Development/staging environments

### Replicated Instances

For high availability and throughput:

```yaml
services:
  next-plaid:
    image: ghcr.io/lightonai/next-plaid-api:latest
    deploy:
      replicas: 3
    volumes:
      - /shared/indices:/data/indices:ro
```

**Benefits:**

- High availability
- Load distribution
- Rolling updates

**Requirements:**

- Shared storage (NFS, EFS, GCS FUSE)
- Load balancer
- Read-only index access

### Sharded Deployment

For very large indices:

```yaml
# Shard 1: Documents 0-999999
shard-1:
  environment:
    - INDEX_DIR=/data/indices/shard1

# Shard 2: Documents 1000000-1999999
shard-2:
  environment:
    - INDEX_DIR=/data/indices/shard2
```

**Query flow:**

1. Query all shards in parallel
2. Merge results
3. Re-rank top-k

---

## Load Balancing

### nginx Configuration

```nginx
upstream next_plaid {
    least_conn;
    server next-plaid-1:8080;
    server next-plaid-2:8080;
    server next-plaid-3:8080;
}

server {
    listen 80;

    location / {
        proxy_pass http://next_plaid;
        proxy_connect_timeout 5s;
        proxy_read_timeout 60s;
    }

    location /health {
        proxy_pass http://next_plaid;
        proxy_connect_timeout 2s;
        proxy_read_timeout 5s;
    }
}
```

### AWS Application Load Balancer

- **Health check path**: `/health`
- **Health check interval**: 30 seconds
- **Healthy threshold**: 2
- **Unhealthy threshold**: 3
- **Timeout**: 10 seconds

---

## Monitoring

### Metrics to Track

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Request latency (p99) | Search response time | > 500ms |
| Error rate | 4xx/5xx responses | > 1% |
| Memory usage | Container memory | > 80% |
| CPU usage | Container CPU | > 80% |
| Index size | Documents indexed | Approaching max |

### Prometheus Metrics

NextPlaid exposes metrics at `/metrics` (when enabled):

```
# Request latency histogram
next_plaid_request_duration_seconds_bucket{endpoint="/search",le="0.1"} 1234

# Active requests
next_plaid_active_requests{endpoint="/search"} 5

# Index stats
next_plaid_index_documents{index="my_index"} 1000000
```

### Logging

Configure structured JSON logging:

```bash
RUST_LOG=info,tower_http=info
```

Log aggregation with:

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Loki + Grafana**
- **CloudWatch Logs**

---

## Security

### Network Security

1. **Run in private network**: No public internet exposure
2. **Use TLS**: Terminate at load balancer
3. **Firewall rules**: Allow only necessary ports

### Authentication

NextPlaid doesn't include built-in authentication. Options:

1. **Reverse proxy auth**: nginx with basic auth or OAuth
2. **API gateway**: Kong, AWS API Gateway
3. **Service mesh**: Istio, Linkerd

Example with nginx basic auth:

```nginx
location / {
    auth_basic "NextPlaid API";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://next_plaid;
}
```

### Container Security

- Run as non-root user (default)
- Use read-only filesystem where possible
- Scan images for vulnerabilities
- Use specific version tags (not `latest` in production)

---

## Backup and Recovery

### Index Backup

```bash
# Stop writes (or use consistent snapshot)
# Copy index directory
rsync -av /data/indices/ /backup/indices/

# Or use cloud storage
aws s3 sync /data/indices/ s3://my-bucket/indices/
```

### Recovery

```bash
# Restore from backup
rsync -av /backup/indices/ /data/indices/

# Restart service
docker compose restart
```

### Point-in-Time Recovery

For critical deployments:

1. Use versioned storage (S3 versioning)
2. Snapshot before major updates
3. Test recovery procedures regularly

---

## Updates and Rollbacks

### Rolling Update

```bash
# Pull new image
docker compose pull

# Rolling restart
docker compose up -d --no-deps next-plaid
```

### Blue-Green Deployment

1. Deploy new version to "green" environment
2. Run smoke tests
3. Switch load balancer to green
4. Keep blue running for quick rollback

### Rollback

```bash
# Rollback to previous version
docker compose down
docker compose up -d --pull never
```

---

## Performance Tuning

### Index Configuration

For large indices:

```python
IndexConfig(
    nbits=4,           # 4-bit for quality, 2-bit for speed
    batch_size=100000, # Larger batches for indexing
)
```

### Search Configuration

Balance latency vs. quality:

```python
# Production default
SearchParams(
    top_k=10,
    n_ivf_probe=8,
    n_full_scores=4096,
)

# High-quality retrieval
SearchParams(
    top_k=100,
    n_ivf_probe=16,
    n_full_scores=8192,
)
```

### BLAS Optimization

Ensure OpenBLAS uses optimal thread count:

```bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

---

## Checklist

Before going to production:

- [ ] Version-pinned Docker image
- [ ] Resource limits configured
- [ ] Health checks enabled
- [ ] Logging configured
- [ ] Monitoring in place
- [ ] Backup strategy defined
- [ ] Load testing completed
- [ ] Security review done
- [ ] Rollback procedure tested
