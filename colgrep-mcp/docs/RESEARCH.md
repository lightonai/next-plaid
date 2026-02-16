# Code Search Indexing Techniques - Research & Analysis

## Research on Top Codebase Indexing Techniques (2026)

### 1. Sourcegraph - Trigram-Based Search

**Architecture:**
- **Zoekt**: Fast trigram-based code search engine written in Go
- Sub-second queries across billions of lines of code
- Works well for variety of programming languages

**Key Components:**
- `gitserver`: Sharded service storing code, accessible to other services
- `repo-updater`: Singleton ensuring code freshness while respecting rate limits
- `zoekt`: Handles both repository indexing and indexed searches

**Indexing Strategy:**
- Indexes default branches by default (space optimization)
- Optional multi-branch indexing for cross-branch queries
- Balances common case optimization with resource usage

**Code Intelligence Layers:**
- Accurate navigation via SCIP-based semantic analysis
- Sourcegraph Cody uses: search + code graph (SCIP) + embeddings + relevance methods

**Sources:**
- [Zoekt Fast Trigram Based Code Search](https://github.com/sourcegraph/zoekt)
- [Sourcegraph Architecture](https://github.com/nmpowell/sourcegraph/blob/main/doc/dev/background-information/architecture/index.md)

---

### 2. GitHub Copilot - Multi-Strategy Semantic Search

**Embedding Models:**
- Proprietary transformer-based embeddings
- OpenAI embedding model specialized for code (text-embedding-ada-002 family)
- **October 2025**: New embedding model for improved context understanding

**Four Search Strategies:**

1. **Remote Code Search**
   - Hits GitHub `/embeddings/code_search` API
   - Semantically ranked chunks
   - Sub-second for multi-million-line codebases

2. **Local Embeddings Search**
   - High-dimensional vector conversion
   - SQLite-backed local index
   - Similarity search based on conceptual meaning

3. **Instant Indexing**
   - Triggers index build on-demand
   - Polls for up to 8 seconds
   - Small repos complete within window

4. **Hybrid Search**
   - Combines BM25 (keyword) with semantic understanding
   - Best of both worlds

**Architecture Components:**
- Vector embeddings transform queries + code into high-dimensional space
- Semantic similarity measured in vector space
- **November 2025**: Streamlined from 40+ tools to 13 core tools using embedding-guided selection

**Recent Features (2025-2026):**
- Instant contextually-aware assistance on repo open
- Auto-indexing when opening GitHub Copilot Chat
- Copilot CLI: Enhanced codebase search with semantic indexing
- Natural language questions with instant, context-aware answers

**Sources:**
- [GitHub New Embedding Model](https://www.infoq.com/news/2025/10/github-embedding-model/)
- [Copilot Indexing Magic](https://yasithrashan.medium.com/how-github-copilot-knows-your-code-inside-its-indexing-magic-aba59a0ce0e8)
- [GitHub Copilot Indexing Docs](https://docs.github.com/copilot/concepts/indexing-repositories-for-copilot-chat)

---

### 3. ColGREP - ColBERT Multi-Vector Retrieval

**Our Current Approach:**
- **ColBERT**: Multi-vector embeddings for each code unit
- Late interaction mechanism (MaxSim scoring)
- CPU-optimized with product quantization

**Advantages:**
- More nuanced than single-vector approaches
- Captures different aspects of code semantics
- Fast CPU inference

**Storage:**
- Local filesystem (`.colgrep/` directory)
- Memory-mapped indices for low RAM usage
- SQLite for metadata filtering

---

## Comparison Matrix

| Feature | Sourcegraph (Zoekt) | GitHub Copilot | ColGREP |
|---------|---------------------|----------------|---------|
| **Search Type** | Trigram keyword | Hybrid semantic | Semantic multi-vector |
| **Speed** | Sub-second | Sub-second | Fast (CPU) |
| **Scale** | Billions of LOC | Multi-million LOC | Project-level |
| **Accuracy** | Keyword-based | Embedding similarity | MaxSim multi-vector |
| **Offline** | Yes | Hybrid (local+remote) | Yes (fully local) |
| **Storage** | Custom index | SQLite + Remote | Filesystem + SQLite |
| **Languages** | Many (trigrams) | Many (embeddings) | 45+ (tree-sitter) |
| **Update** | Incremental | Auto-trigger | Manual/incremental |

---

## Best Practices for Code Search

### 1. Multi-Strategy Approach
**Lesson from GitHub**: Combine multiple search strategies
- Remote semantic search for scale
- Local embeddings for offline
- Instant indexing for new repos
- Keyword fallback for precision

### 2. Hybrid Search
**Lesson from Both**: Combine keyword + semantic
- BM25 for exact matches
- Embeddings for conceptual similarity
- Use keyword to filter, semantic to rank

### 3. Incremental Indexing
**Lesson from Sourcegraph**: Smart update strategies
- Index default branches by default
- Incremental updates vs full rebuilds
- Balance freshness with resources

### 4. Metadata Separation
**Lesson from GitHub**: Separate vectors from metadata
- Store embeddings in vector DB
- Keep source in object storage
- Use UUIDs/keys to link them

### 5. Code-Specific Parsing
**What we do well**: Tree-sitter for code structure
- Extract meaningful code units (functions, classes)
- Language-aware parsing
- Better than raw text chunking

---

## Cloudflare Storage Options for ColGREP

### Option 1: Vectorize (Primary Vector Storage)

**What it is:**
- Globally distributed vector database
- Built for Workers AI integration
- Designed for similarity search

**How to use with ColGREP:**
```javascript
// Store ColBERT embeddings
const vectors = codeUnits.map(unit => ({
  id: unit.id,
  values: unit.embeddings,  // Multi-vector from ColBERT
  metadata: {
    file: unit.file,
    line: unit.line,
    language: unit.language,
    r2_key: `code/${unit.id}` // Reference to R2
  }
}));

await vectorize.insert(vectors);
```

**Advantages:**
- Built for vector similarity search
- Global distribution
- Workers integration
- Metadata alongside vectors

**Considerations:**
- Multi-vector storage (ColBERT has multiple vectors per document)
- May need to flatten or aggregate embeddings

### Option 2: R2 (Code & Index Storage)

**What it is:**
- S3-compatible object storage
- No egress fees
- Global replication

**How to use with ColGREP:**
```javascript
// Store actual code files
await r2.put(`code/${fileId}`, codeContent);

// Store serialized indices
await r2.put(`indices/${projectId}/vectors.bin`, vectorIndex);
await r2.put(`indices/${projectId}/metadata.json`, metadata);
```

**Advantages:**
- Store large index files
- Keep code source separate from embeddings
- Cost-effective for large data

**Best for:**
- Storing original code files
- Backup of index files
- Serving code snippets in results

### Option 3: D1 (Metadata & State)

**What it is:**
- SQLite-based serverless database
- SQL queries at edge
- Relational data storage

**How to use with ColGREP:**
```sql
CREATE TABLE code_units (
  id TEXT PRIMARY KEY,
  file TEXT,
  line INTEGER,
  unit_type TEXT,
  language TEXT,
  vectorize_id TEXT,  -- Reference to Vectorize
  r2_key TEXT,        -- Reference to R2
  content_hash TEXT,
  indexed_at DATETIME
);

CREATE INDEX idx_file ON code_units(file);
CREATE INDEX idx_language ON code_units(language);
```

**Advantages:**
- SQL filtering before vector search
- Store file metadata, timestamps
- Track indexing state

**Best for:**
- Pre-filtering (by file, language, date)
- Managing index state
- Tracking what's indexed

---

## Recommended Hybrid Architecture

### Storage Distribution

```
┌─────────────────────────────────────────────┐
│                User Query                    │
└─────────────────┬───────────────────────────┘
                  │
                  v
┌─────────────────────────────────────────────┐
│           MCP Server (Edge Worker)          │
│  1. Parse query                             │
│  2. Generate embedding                      │
│  3. Pre-filter with D1 (optional)          │
└─────────────────┬───────────────────────────┘
                  │
                  v
┌─────────────────────────────────────────────┐
│          Vectorize (Vector Search)          │
│  - Store multi-vector embeddings            │
│  - Similarity search via MaxSim             │
│  - Return top-k results with metadata       │
└─────────────────┬───────────────────────────┘
                  │
                  v
┌─────────────────────────────────────────────┐
│          R2 (Code Retrieval)                │
│  - Fetch actual code snippets               │
│  - Serve with context lines                 │
└─────────────────────────────────────────────┘
```

### Implementation Plan

**Phase 1: Local-First (Current)**
- ✅ Store everything locally
- ✅ Fast for single-user dev
- ✅ No network dependency

**Phase 2: Hybrid Local + Cloud**
- Store indices in R2 for backup
- Use D1 for shared metadata
- Keep vectors local for speed

**Phase 3: Fully Distributed**
- Move embeddings to Vectorize
- Store code in R2
- Use D1 for all metadata
- Enable team-wide search

---

## LLM Query Generation Instructions

### Understanding ColGREP vs Traditional Search

Traditional search (grep, symbol search):
- **Requires exact keywords**: "getUserById"
- **Matches literal text**: func getUserById
- **Fails on**: different naming, paraphrasing

ColGREP semantic search:
- **Understands intent**: "function that fetches a user"
- **Matches meaning**: getUserById, fetchUser, loadUserData
- **Works with**: descriptions, purposes, patterns

### Query Generation Best Practices

#### 1. Be Descriptive, Not Prescriptive

❌ **Bad**: `getUserById`
✅ **Good**: `function that retrieves a user by their ID`

❌ **Bad**: `validateEmail`
✅ **Good**: `code that validates email addresses using regex`

**Why**: Semantic search understands descriptions better than exact names.

#### 2. Describe Behavior and Purpose

✅ **Good queries**:
- `function that retries failed HTTP requests with exponential backoff`
- `error handling for database connection timeouts`
- `middleware that checks if a user is authenticated`
- `utility that formats dates for display`

**Pattern**: `[what it is] that [what it does] [optionally: how/when/why]`

#### 3. Include Context When Useful

✅ **With context**:
- `React component that displays a list of users with pagination`
- `API endpoint that creates a new order and sends confirmation email`
- `helper function for parsing JSON with error recovery`

**When to add context**:
- Technology stack (React, Django, etc.)
- Domain concepts (user, order, payment)
- Related operations (with pagination, sends email)

#### 4. Use Natural Language

✅ **Natural**:
- `code that handles file uploads`
- `function for calculating shipping costs`
- `validation logic for user registration forms`

❌ **Too formal**:
- `file upload handler implementation`
- `shipping cost calculation algorithm`
- `user registration form validation procedure`

#### 5. Be Specific About Patterns

✅ **Specific patterns**:
- `async function that uses try-catch for error handling`
- `React hook that manages form state`
- `database query with JOIN operations`
- `recursive function for tree traversal`

#### 6. Combining Features

✅ **Multi-feature queries**:
- `function that fetches data from API and caches the result`
- `component that renders a table with sorting and filtering`
- `service that processes payments and updates inventory`

### Query Examples by Use Case

#### Finding Similar Code
```
Query: "function similar to calculateTotal that sums values with tax"
Finds: calculateSubtotal, computeGrandTotal, sumWithTax
```

#### Locating Patterns
```
Query: "error handling with logging and retry logic"
Finds: All error handlers that log AND retry
```

#### Cross-Cutting Concerns
```
Query: "logging of user actions for audit trail"
Finds: Audit logging across the codebase
```

#### API Usage Examples
```
Query: "uses axios to make POST requests with authentication"
Finds: API calls using axios library with auth
```

#### Refactoring Targets
```
Query: "code that directly modifies global state"
Finds: Anti-patterns for refactoring
```

### Query Templates

#### For Functions
- `function that [action] [object] [optional: method/constraint]`
- `[sync/async] function for [purpose]`
- `helper that [transforms/validates/processes] [data type]`

#### For Components (UI)
- `component that [displays/renders] [UI element] [optional: with features]`
- `[framework] component for [purpose]`

#### For Patterns
- `code that uses [pattern/library] for [purpose]`
- `implementation of [design pattern]`
- `[operation] with [error handling/logging/retry]`

#### For APIs
- `API endpoint that [HTTP method] [resource]`
- `route handler for [operation]`

### Advanced: Combining with Filters

Use file filters for targeted search:

```json
{
  "query": "authentication middleware",
  "include": ["*.ts", "*.js"],
  "exclude": ["*test*", "*spec*"]
}
```

```json
{
  "query": "database migration scripts",
  "include": ["migrations/**/*"]
}
```

### Common Pitfalls

❌ **Too vague**: `user code`
✅ **Better**: `function that creates a new user account`

❌ **Too specific**: `function named createUserAccountWithEmailVerification`
✅ **Better**: `function that creates user accounts with email verification`

❌ **Just keywords**: `http retry error`
✅ **Better**: `HTTP client that retries failed requests with error handling`

### Tips for LLMs Generating Queries

1. **Extract intent from user request**
   - User: "Find where we handle failed payments"
   - Query: "error handling for failed payment transactions"

2. **Add implicit context**
   - User: "auth check"
   - Query: "middleware that checks if user is authenticated"

3. **Expand abbreviations**
   - User: "db conn pool"
   - Query: "database connection pool configuration"

4. **Include related concepts**
   - User: "login"
   - Query: "authentication function that validates user credentials and creates session"

5. **Preserve technical terms**
   - User: "React hooks for state"
   - Query: "React hooks that manage component state"

---

## Summary

### Key Takeaways

1. **Multi-strategy works best**: Combine keyword, semantic, and hybrid approaches
2. **Metadata separation**: Store vectors, code, and metadata in appropriate stores
3. **Incremental is essential**: Don't rebuild entire index on every change
4. **Pre-filtering helps**: Use SQL/metadata filters before vector search
5. **Natural queries work**: LLMs should generate descriptive, natural language queries

### Next Steps for ColGREP

1. ✅ Implement full MCP protocol integration
2. ✅ Add progress notifications for indexing
3. ✅ Support incremental updates
4. 🔄 Evaluate Cloudflare storage for team features
5. 🔄 Add hybrid keyword + semantic search
6. 🔄 Implement query expansion/reformulation

---

**Sources:**
- [Sourcegraph Zoekt](https://github.com/sourcegraph/zoekt)
- [GitHub Copilot Indexing](https://docs.github.com/copilot/concepts/indexing-repositories-for-copilot-chat)
- [GitHub Embedding Model 2025](https://www.infoq.com/news/2025/10/github-embedding-model/)
- [Cloudflare Vectorize](https://developers.cloudflare.com/vectorize/)
- [Cloudflare Vector + Workers AI](https://blog.cloudflare.com/vectorize-vector-database-open-beta/)
