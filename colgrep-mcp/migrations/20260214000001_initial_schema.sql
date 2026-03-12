-- Initial schema for ColGREP MCP Server with pgvector
-- Stores code embeddings for semantic search

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Projects table - tracks indexed codebases
CREATE TABLE IF NOT EXISTS projects (
    id BIGSERIAL PRIMARY KEY,
    root_path TEXT NOT NULL UNIQUE,
    last_indexed TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Files table - tracks individual source files
CREATE TABLE IF NOT EXISTS files (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(project_id, file_path)
);

-- Code units table - tracks functions, classes, etc.
CREATE TABLE IF NOT EXISTS code_units (
    id BIGSERIAL PRIMARY KEY,
    file_id BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    code TEXT NOT NULL,
    unit_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Vectors table - stores ColBERT multi-vector embeddings
-- Each code unit has multiple vectors (ColBERT uses multiple tokens per document)
CREATE TABLE IF NOT EXISTS vectors (
    id BIGSERIAL PRIMARY KEY,
    code_unit_id BIGINT NOT NULL REFERENCES code_units(id) ON DELETE CASCADE,
    vector_index INTEGER NOT NULL,
    embedding vector(128) NOT NULL, -- ColBERT default dimension
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(code_unit_id, vector_index)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_files_project_id ON files(project_id);
CREATE INDEX IF NOT EXISTS idx_code_units_file_id ON code_units(file_id);
CREATE INDEX IF NOT EXISTS idx_vectors_code_unit_id ON vectors(code_unit_id);

-- Vector similarity index using HNSW (Hierarchical Navigable Small World)
-- This enables fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_vectors_embedding_hnsw ON vectors
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (can be faster for smaller datasets)
-- Uncomment if HNSW is too slow for your use case:
-- CREATE INDEX IF NOT EXISTS idx_vectors_embedding_ivfflat ON vectors
--     USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_files_updated_at BEFORE UPDATE ON files
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_code_units_updated_at BEFORE UPDATE ON code_units
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
