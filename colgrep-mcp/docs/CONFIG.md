# ColGREP MCP Configuration Guide

ColGREP MCP Server supports three backend options for storing your code index:

1. **Filesystem** (default, simplest)
2. **Local** (PostgreSQL + pgvector, self-hosted)
3. **Cloudflare** (D1 + R2 + Vectorize, cloud-native)

## Quick Start

### Default (Filesystem Backend)

No configuration needed! Just build and run:

```bash
cargo build --release -p colgrep-mcp
```

The index is stored in `.colgrep/` directory in your project root.

### Local PostgreSQL Backend

For self-hosting with PostgreSQL + pgvector:

1. **Install PostgreSQL with pgvector extension:**

```bash
# macOS (Homebrew)
brew install postgresql pgvector

# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
# Then install pgvector: https://github.com/pgvector/pgvector#installation

# Docker
docker run -d \
  --name colgrep-postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

2. **Create database:**

```bash
createdb colgrep
psql colgrep -c "CREATE EXTENSION vector;"
```

3. **Create configuration file:**

Create `~/.config/colgrep/mcp.toml` or `./colgrep-mcp.toml`:

```toml
backend = "local"

[local]
database_url = "postgresql://localhost:5432/colgrep"
max_connections = 10
vector_dimensions = 128  # ColBERT default

[general]
model = "lightonai/LateOn-Code-edge"  # ColBERT model (optional, overrides colgrep default)
auto_index = false
max_results = 15
context_lines = 6
```

4. **Build with local-db feature (default):**

```bash
cargo build --release -p colgrep-mcp --features local-db
```

### Cloudflare Backend (Coming Soon)

For cloud deployments using Cloudflare's serverless platform:

```toml
backend = "cloudflare"

[cloudflare]
account_id = "your-account-id"
api_token = "your-api-token"
vectorize_index = "colgrep-vectors"
d1_database = "colgrep-metadata"
r2_bucket = "colgrep-code"
```

Build with cloudflare feature:

```bash
cargo build --release -p colgrep-mcp --features cloudflare
```

See [CLOUDFLARE.md](./CLOUDFLARE.md) for complete setup instructions.

## Configuration File Locations

ColGREP MCP looks for configuration in this order:

1. `./colgrep-mcp.toml` (current directory)
2. `~/.config/colgrep/mcp.toml` (user config)
3. Environment variables
4. Default values

## Configuration Reference

### Backend Selection

```toml
backend = "local"  # Options: "filesystem", "local", "cloudflare"
```

### Local Backend Settings

```toml
[local]
# PostgreSQL connection string
database_url = "postgresql://user:password@localhost:5432/colgrep"

# Maximum number of database connections
max_connections = 10

# Vector dimensions (must match ColBERT model)
vector_dimensions = 128
```

**Environment variables:**
- `DATABASE_URL` - overrides `database_url`

### Cloudflare Backend Settings

```toml
[cloudflare]
# Cloudflare Account ID (required)
account_id = "abc123..."

# API token with permissions for Workers, D1, R2, Vectorize
api_token = "your-token"

# Resource names
vectorize_index = "colgrep-vectors"
d1_database = "colgrep-metadata"
r2_bucket = "colgrep-code"
```

**Environment variables:**
- `CLOUDFLARE_ACCOUNT_ID` - overrides `account_id`
- `CLOUDFLARE_API_TOKEN` - overrides `api_token`

### General Settings

```toml
[general]
# ColBERT model (HuggingFace ID or local path). Overrides colgrep default.
model = "lightonai/LateOn-Code-edge"

# Enable automatic incremental indexing on file changes
auto_index = false

# Maximum search results to return
max_results = 15

# Default context lines before/after matches
context_lines = 6
```

### Model Selection

Use `--list-models` to see available ColBERT models and which is default:

```bash
colgrep-mcp --list-models
```

Select a model via CLI:

```bash
colgrep-mcp --model lightonai/LateOn-Code
colgrep-mcp --http --model lightonai/LateOn-Code-edge
```

Or config file:

```toml
[general]
model = "lightonai/LateOn-Code-edge"
```

Precedence: `--model` CLI > `[general].model` in config > colgrep config > built-in default.

## Generate Example Config

```bash
colgrep-mcp --generate-config
```

Prints an example config to stdout. Redirect to a file for editing.

## Backend Comparison

| Feature | Filesystem | Local (PostgreSQL) | Cloudflare |
|---------|-----------|-------------------|------------|
| Setup | None | PostgreSQL + pgvector | Cloud account |
| Performance | Fast (local disk) | Fast (local DB) | Network latency |
| Scalability | Single machine | Team (shared DB) | Global |
| Cost | Free | Free (self-hosted) | Pay-as-you-go |
| Multi-user | No | Yes | Yes |
| Incremental Updates | Limited | Full support | Full support |
| Vector Search | CPU-based | Hardware-accelerated | Cloud-native |

## Feature Flags

Build with specific features:

```bash
# Default: local-db (PostgreSQL + pgvector)
cargo build --release -p colgrep-mcp

# Filesystem only (no databases)
cargo build --release -p colgrep-mcp --no-default-features

# Cloudflare backend
cargo build --release -p colgrep-mcp --features cloudflare

# Hardware acceleration (optional, pass through from colgrep)
cargo build --release -p colgrep-mcp --features cuda
cargo build --release -p colgrep-mcp --features coreml
```

## Database Schema (PostgreSQL)

The local backend creates these tables:

```sql
-- Projects: tracks indexed codebases
CREATE TABLE projects (
    id BIGSERIAL PRIMARY KEY,
    root_path TEXT UNIQUE,
    last_indexed TIMESTAMPTZ
);

-- Files: individual source files
CREATE TABLE files (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT REFERENCES projects(id),
    file_path TEXT,
    language TEXT
);

-- Code units: functions, classes, etc.
CREATE TABLE code_units (
    id BIGSERIAL PRIMARY KEY,
    file_id BIGINT REFERENCES files(id),
    line_number INTEGER,
    code TEXT,
    unit_type TEXT
);

-- Vectors: ColBERT multi-vector embeddings
CREATE TABLE vectors (
    id BIGSERIAL PRIMARY KEY,
    code_unit_id BIGINT REFERENCES code_units(id),
    vector_index INTEGER,
    embedding vector(128)  -- pgvector type
);

-- HNSW index for fast similarity search
CREATE INDEX ON vectors USING hnsw (embedding vector_cosine_ops);
```

## Troubleshooting

### PostgreSQL Connection Issues

```bash
# Test connection
psql "postgresql://localhost:5432/colgrep"

# Check if pgvector is installed
psql colgrep -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Performance Tuning (PostgreSQL)

For large codebases, tune PostgreSQL settings:

```sql
-- Increase work_mem for better index creation
ALTER SYSTEM SET work_mem = '256MB';

-- Tune HNSW parameters
CREATE INDEX ON vectors USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);  -- Higher = better quality, slower
```

### Reset Index

```bash
# Filesystem
rm -rf .colgrep

# PostgreSQL
psql colgrep -c "DROP TABLE IF EXISTS projects CASCADE;"
# Then restart colgrep-mcp to recreate schema
```

## Migration Guide

### From Filesystem to PostgreSQL

1. Export your filesystem index (if needed for backup)
2. Set up PostgreSQL with pgvector
3. Update configuration to use local backend
4. Restart MCP server - it will re-index automatically
5. Old `.colgrep/` directory can be deleted

### From PostgreSQL to Cloudflare

1. Set up Cloudflare resources (see CLOUDFLARE.md)
2. Update configuration to use cloudflare backend
3. Restart MCP server - it will migrate data automatically
4. PostgreSQL database can be retired

## Security

### Filesystem Backend
- Index stored locally in `.colgrep/`
- No network access required
- Suitable for sensitive codebases

### PostgreSQL Backend
- Use SSL connections in production:
  ```toml
  database_url = "postgresql://user:pass@host:5432/colgrep?sslmode=require"
  ```
- Restrict database access with firewall rules
- Use strong passwords or certificate auth

### Cloudflare Backend
- API tokens should have minimal required permissions
- Store tokens in environment variables, not config files
- Enable Cloudflare Access for additional security

## Support

- **Issues**: https://github.com/lightonai/next-plaid/issues
- **Discussions**: https://github.com/lightonai/next-plaid/discussions
- **Documentation**: https://github.com/lightonai/next-plaid/tree/main/colgrep-mcp
