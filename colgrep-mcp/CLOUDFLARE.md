# Cloudflare Integration Guide for ColGREP

This document outlines how to integrate ColGREP with Cloudflare's platform for distributed, edge-based code search.

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    User/Agent Query                   │
└────────────────────┬─────────────────────────────────┘
                     │
                     v
┌──────────────────────────────────────────────────────┐
│             Cloudflare Worker (MCP Server)           │
│  • Receives MCP requests                             │
│  • Generates embeddings (Workers AI)                 │
│  • Routes to appropriate storage                     │
└────────────────────┬─────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        v            v            v
┌─────────────┐ ┌─────────┐ ┌──────────┐
│  Vectorize  │ │   D1    │ │    R2    │
│   (Vectors) │ │(Metadata)│ │  (Code)  │
└─────────────┘ └─────────┘ └──────────┘
```

## Storage Strategy

### 1. Vectorize - Vector Embeddings

**Purpose**: Store ColBERT multi-vector embeddings for semantic search

**Data Structure**:
```typescript
interface VectorRecord {
  id: string;              // Unique code unit ID
  values: number[];        // Flattened ColBERT embeddings
  metadata: {
    file: string;          // Relative file path
    line: number;          // Line number
    unit_type: string;     // function, class, etc.
    language: string;      // Programming language
    r2_key: string;        // Reference to R2 object
    d1_id: string;         // Reference to D1 row
    score_boost?: number;  // Optional: boost frequently accessed code
  }
}
```

**Implementation**:
```typescript
// index.ts - Cloudflare Worker
import { Vectorize } from '@cloudflare/workers-types';

interface Env {
  VECTORIZE: VectorizeIndex;
  CODE_DB: D1Database;
  CODE_STORAGE: R2Bucket;
}

// Indexing: Insert vectors
async function indexCodeUnit(
  unit: CodeUnit,
  embeddings: number[][],  // ColBERT multi-vector
  env: Env
): Promise<void> {
  // Flatten multi-vector or use averaging strategy
  const flattenedEmbedding = flattenColBERTVectors(embeddings);

  await env.VECTORIZE.insert([{
    id: unit.id,
    values: flattenedEmbedding,
    metadata: {
      file: unit.file,
      line: unit.line,
      unit_type: unit.type,
      language: unit.language,
      r2_key: `code/${unit.project}/${unit.file}`,
      d1_id: unit.id
    }
  }]);
}

// Searching: Query vectors
async function searchCode(
  query: string,
  topK: number,
  env: Env
): Promise<SearchResult[]> {
  // Generate query embedding (using Workers AI)
  const queryEmbedding = await generateEmbedding(query, env);

  // Search Vectorize
  const results = await env.VECTORIZE.query(queryEmbedding, {
    topK,
    returnMetadata: true
  });

  // Fetch actual code from R2
  const enrichedResults = await Promise.all(
    results.matches.map(async (match) => ({
      ...match.metadata,
      code: await fetchCodeSnippet(match.metadata.r2_key, env),
      score: match.score
    }))
  );

  return enrichedResults;
}
```

**Multi-Vector Strategy**:

ColBERT produces multiple vectors per code unit. Options:

1. **Average Pooling** (Simplest):
```typescript
function flattenColBERTVectors(vectors: number[][]): number[] {
  const numVectors = vectors.length;
  const dim = vectors[0].length;
  const averaged = new Array(dim).fill(0);

  for (const vec of vectors) {
    for (let i = 0; i < dim; i++) {
      averaged[i] += vec[i] / numVectors;
    }
  }

  return averaged;
}
```

2. **Max Pooling** (Preserves strong signals):
```typescript
function flattenColBERTVectors(vectors: number[][]): number[] {
  const dim = vectors[0].length;
  const maxPooled = new Array(dim).fill(-Infinity);

  for (const vec of vectors) {
    for (let i = 0; i < dim; i++) {
      maxPooled[i] = Math.max(maxPooled[i], vec[i]);
    }
  }

  return maxPooled;
}
```

3. **Store Multiple** (Most accurate, more storage):
```typescript
// Store each vector separately with shared metadata
async function indexCodeUnitMultiVector(
  unit: CodeUnit,
  embeddings: number[][],
  env: Env
): Promise<void> {
  const vectors = embeddings.map((embedding, idx) => ({
    id: `${unit.id}_vec${idx}`,
    values: embedding,
    metadata: {
      ...unit,
      parent_id: unit.id,
      vector_index: idx
    }
  }));

  await env.VECTORIZE.insert(vectors);
}
```

### 2. D1 - Metadata Database

**Purpose**: Store file metadata, indexing state, enable SQL pre-filtering

**Schema**:
```sql
-- Projects table
CREATE TABLE projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  repository_url TEXT,
  default_branch TEXT DEFAULT 'main',
  indexed_at DATETIME,
  last_updated DATETIME,
  total_files INTEGER DEFAULT 0,
  total_units INTEGER DEFAULT 0
);

-- Files table
CREATE TABLE files (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  path TEXT NOT NULL,
  language TEXT,
  size_bytes INTEGER,
  content_hash TEXT,
  last_modified DATETIME,
  indexed_at DATETIME,
  r2_key TEXT,
  FOREIGN KEY (project_id) REFERENCES projects(id),
  UNIQUE(project_id, path)
);

CREATE INDEX idx_files_project ON files(project_id);
CREATE INDEX idx_files_language ON files(language);
CREATE INDEX idx_files_path ON files(path);

-- Code units table
CREATE TABLE code_units (
  id TEXT PRIMARY KEY,
  file_id TEXT NOT NULL,
  unit_type TEXT NOT NULL,  -- function, class, method, etc.
  name TEXT,
  line_start INTEGER,
  line_end INTEGER,
  vectorize_id TEXT,  -- Reference to Vectorize
  embedding_hash TEXT,  -- To detect changes
  indexed_at DATETIME,
  FOREIGN KEY (file_id) REFERENCES files(id)
);

CREATE INDEX idx_units_file ON code_units(file_id);
CREATE INDEX idx_units_type ON code_units(unit_type);
CREATE INDEX idx_units_name ON code_units(name);

-- Search history (optional, for analytics)
CREATE TABLE search_queries (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  query TEXT NOT NULL,
  results_count INTEGER,
  avg_score REAL,
  searched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  user_id TEXT,
  FOREIGN KEY (project_id) REFERENCES projects(id)
);
```

**Usage**:
```typescript
// Pre-filter before vector search
async function searchWithFilters(
  query: string,
  filters: {
    language?: string;
    filePattern?: string;
    excludeTests?: boolean;
  },
  env: Env
): Promise<SearchResult[]> {
  // Build SQL query for pre-filtering
  let sql = `
    SELECT cu.id, cu.vectorize_id
    FROM code_units cu
    JOIN files f ON cu.file_id = f.id
    WHERE 1=1
  `;
  const params: any[] = [];

  if (filters.language) {
    sql += ` AND f.language = ?`;
    params.push(filters.language);
  }

  if (filters.filePattern) {
    sql += ` AND f.path LIKE ?`;
    params.push(filters.filePattern);
  }

  if (filters.excludeTests) {
    sql += ` AND f.path NOT LIKE '%test%' AND f.path NOT LIKE '%spec%'`;
  }

  // Get filtered IDs
  const { results } = await env.CODE_DB.prepare(sql).bind(...params).all();
  const allowedIds = results.map(r => r.vectorize_id);

  // Now search Vectorize with filtered IDs
  const embedding = await generateEmbedding(query, env);
  const vectorResults = await env.VECTORIZE.query(embedding, {
    topK: 20,
    filter: { id: { $in: allowedIds } }
  });

  return vectorResults.matches;
}

// Track indexing progress
async function updateIndexingProgress(
  projectId: string,
  stats: IndexStats,
  env: Env
): Promise<void> {
  await env.CODE_DB.prepare(`
    UPDATE projects
    SET total_files = ?,
        total_units = ?,
        last_updated = CURRENT_TIMESTAMP
    WHERE id = ?
  `).bind(stats.files, stats.units, projectId).run();
}
```

### 3. R2 - Code Storage

**Purpose**: Store original code files and serve snippets in search results

**Structure**:
```
code/
├── {project_id}/
│   ├── {file_path}                    # Original files
│   └── snapshots/
│       └── {commit_hash}/
│           └── {file_path}            # Historical versions
└── indices/
    └── {project_id}/
        ├── vectors_{version}.bin      # Backup of vector index
        ├── metadata_{version}.json    # Backup of metadata
        └── state.json                 # Current index state
```

**Implementation**:
```typescript
// Store code file
async function storeCodeFile(
  projectId: string,
  filePath: string,
  content: string,
  env: Env
): Promise<string> {
  const key = `code/${projectId}/${filePath}`;
  await env.CODE_STORAGE.put(key, content, {
    httpMetadata: {
      contentType: 'text/plain; charset=utf-8',
    },
    customMetadata: {
      project: projectId,
      path: filePath,
      indexed_at: new Date().toISOString()
    }
  });
  return key;
}

// Fetch code snippet with context
async function fetchCodeSnippet(
  r2Key: string,
  lineStart: number,
  lineEnd: number,
  contextLines: number,
  env: Env
): Promise<string> {
  const object = await env.CODE_STORAGE.get(r2Key);
  if (!object) return '';

  const content = await object.text();
  const lines = content.split('\n');

  const start = Math.max(0, lineStart - contextLines - 1);
  const end = Math.min(lines.length, lineEnd + contextLines);

  return lines.slice(start, end).join('\n');
}

// Backup index to R2
async function backupIndex(
  projectId: string,
  indexData: IndexBackup,
  env: Env
): Promise<void> {
  const version = Date.now();
  await env.CODE_STORAGE.put(
    `indices/${projectId}/state_${version}.json`,
    JSON.stringify(indexData)
  );
}
```

## Workers AI Integration

**Generate Embeddings at the Edge**:

```typescript
import { Ai } from '@cloudflare/ai';

async function generateEmbedding(
  text: string,
  env: Env
): Promise<number[]> {
  const ai = new Ai(env.AI);

  // Use Workers AI embedding model
  const response = await ai.run('@cf/baai/bge-base-en-v1.5', {
    text: [text]
  });

  return response.data[0];
}

// Batch embeddings for indexing
async function batchGenerateEmbeddings(
  texts: string[],
  env: Env
): Promise<number[][]> {
  const ai = new Ai(env.AI);

  // Process in batches of 100
  const batchSize = 100;
  const embeddings: number[][] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const response = await ai.run('@cf/baai/bge-base-en-v1.5', {
      text: batch
    });
    embeddings.push(...response.data);
  }

  return embeddings;
}
```

**Note**: Workers AI models are different from ColBERT. For production, you'd want to:
1. Export ColBERT model to ONNX
2. Run it in a Worker with WebAssembly
3. Or use a compatible embedding model from Workers AI

## Complete MCP Server Worker

```typescript
// index.ts
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Handle MCP protocol requests
    const { method, params } = await request.json();

    switch (method) {
      case 'tools/list':
        return Response.json({
          tools: [
            {
              name: 'index_codebase',
              description: 'Index a codebase for semantic search',
              inputSchema: {
                type: 'object',
                properties: {
                  repository: { type: 'string' },
                  branch: { type: 'string', default: 'main' }
                },
                required: ['repository']
              }
            },
            {
              name: 'search',
              description: 'Search code semantically',
              inputSchema: {
                type: 'object',
                properties: {
                  query: { type: 'string' },
                  project: { type: 'string' },
                  topK: { type: 'number', default: 15 },
                  language: { type: 'string' },
                  include: { type: 'array', items: { type: 'string' } }
                },
                required: ['query']
              }
            }
          ]
        });

      case 'tools/call':
        return handleToolCall(params, env);

      default:
        return Response.json({ error: 'Unknown method' }, { status: 400 });
    }
  }
};

async function handleToolCall(params: any, env: Env): Promise<Response> {
  const { name, arguments: args } = params;

  switch (name) {
    case 'search':
      const results = await searchCode(
        args.query,
        args.topK || 15,
        {
          language: args.language,
          filePattern: args.include?.[0]
        },
        env
      );
      return Response.json({
        content: [{
          type: 'text',
          text: formatSearchResults(results)
        }]
      });

    // Other tool implementations...

    default:
      return Response.json({ error: 'Unknown tool' }, { status: 400 });
  }
}
```

## Deployment

```bash
# Install Wrangler
npm install -g wrangler

# Create resources
wrangler vectorize create colgrep-vectors --dimensions=768 --metric=cosine
wrangler d1 create colgrep-metadata
wrangler r2 bucket create colgrep-code

# Configure wrangler.toml
cat > wrangler.toml <<EOF
name = "colgrep-mcp"
main = "src/index.ts"
compatibility_date = "2024-01-01"

[[vectorize]]
binding = "VECTORIZE"
index_name = "colgrep-vectors"

[[d1_databases]]
binding = "CODE_DB"
database_name = "colgrep-metadata"
database_id = "your-database-id"

[[r2_buckets]]
binding = "CODE_STORAGE"
bucket_name = "colgrep-code"

[ai]
binding = "AI"
EOF

# Deploy
wrangler deploy
```

## Cost Estimation

### Vectorize
- **Free tier**: 30M queried dimensions/month, 5M stored dimensions
- **Paid**: $0.040 per 1M queried dimensions

### D1
- **Free tier**: 5M rows read/month, 100K rows written/month
- **Paid**: $0.001 per 1M rows read

### R2
- **Free tier**: 10 GB storage, 1M Class A ops, 10M Class B ops
- **No egress fees** (major advantage over S3)

### Workers AI
- **Free tier**: 10,000 neurons/day
- **Paid**: $0.011 per 1,000 neurons

**Example Project** (10K files, 100K code units):
- Vectorize: ~77M dimensions stored (768 dim × 100K) ≈ **$0.40/month** (after free tier)
- D1: Minimal (metadata only) ≈ **Free**
- R2: ~100MB code storage ≈ **Free**
- Workers AI: Depends on search volume

## Advantages of Cloudflare Architecture

1. **Global Distribution**: Vectors at the edge, sub-100ms searches worldwide
2. **No Egress Fees**: R2 has no bandwidth charges
3. **Integrated AI**: Workers AI for embeddings without external API
4. **Scalable**: Handles team-wide search, multi-project indexing
5. **Cost-Effective**: Generous free tiers, pay only for what you use

## Migration Path

**Phase 1**: Keep local indexing, add R2 backup
**Phase 2**: Move metadata to D1, use for pre-filtering
**Phase 3**: Migrate vectors to Vectorize for distributed search
**Phase 4**: Full Workers-based MCP server

---

**Next Steps**:
1. Prototype Vectorize integration with averaged ColBERT vectors
2. Benchmark search quality vs. local multi-vector search
3. Implement D1 schema and test pre-filtering
4. Deploy MCP server Worker with all integrations
