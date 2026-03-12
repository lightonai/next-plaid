//! Local PostgreSQL + pgvector backend
//!
//! Stores code embeddings in PostgreSQL with pgvector extension for similarity search.
//! Good for: self-hosting, team deployments, full control over data

use super::{Backend, FileChange, IndexStats, SearchResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use deadpool_postgres::{
    Config as PoolConfig, Manager, ManagerConfig, Pool, RecyclingMethod, Runtime,
};
use std::path::{Path, PathBuf};
use tokio_postgres::NoTls;

/// Local PostgreSQL backend with pgvector
pub struct LocalBackend {
    pool: Pool,
    vector_dimensions: usize,
}

impl LocalBackend {
    /// Create a new local backend
    pub async fn new(config: &crate::config::LocalConfig) -> Result<Self> {
        // Parse connection string
        let pg_config = config
            .database_url
            .parse::<tokio_postgres::Config>()
            .context("Invalid database URL")?;

        // Create connection pool
        let mut pool_config = PoolConfig::new();
        pool_config.manager = Some(ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        });
        pool_config.max_size = config.max_connections as usize;

        let manager = Manager::from_config(pg_config, NoTls, pool_config.manager.unwrap());
        let pool = Pool::builder(manager)
            .max_size(config.max_connections as usize)
            .build()
            .context("Failed to create connection pool")?;

        Ok(Self {
            pool,
            vector_dimensions: config.vector_dimensions,
        })
    }

    /// Get or create project ID for a root path
    async fn get_or_create_project(&self, root: &Path) -> Result<i64> {
        let client = self.pool.get().await?;
        let root_str = root.to_string_lossy().to_string();

        // Try to get existing project
        let row = client
            .query_opt("SELECT id FROM projects WHERE root_path = $1", &[&root_str])
            .await?;

        if let Some(row) = row {
            return Ok(row.get(0));
        }

        // Create new project
        let row = client
            .query_one(
                "INSERT INTO projects (root_path, last_indexed) VALUES ($1, NOW()) RETURNING id",
                &[&root_str],
            )
            .await?;

        Ok(row.get(0))
    }

    /// Store vectors for code units
    async fn store_vectors(&self, project_id: i64, code_units: &[colgrep::CodeUnit]) -> Result<()> {
        use colgrep::{ensure_model, Encoder};

        // Ensure model is available
        ensure_model(None, false).context("Failed to download ColBERT model")?;

        // Load encoder
        let encoder = Encoder::new(None).context("Failed to load encoder")?;

        let client = self.pool.get().await?;

        for unit in code_units {
            let file_path = unit.file_path.to_string_lossy().to_string();

            // Get or create file record
            let file_id: i64 = client
                .query_one(
                    "INSERT INTO files (project_id, file_path, language)
                     VALUES ($1, $2, $3)
                     ON CONFLICT (project_id, file_path) DO UPDATE SET language = EXCLUDED.language
                     RETURNING id",
                    &[&project_id, &file_path, &unit.language],
                )
                .await?
                .get(0);

            // Create code unit record
            let code_unit_id: i64 = client
                .query_one(
                    "INSERT INTO code_units (file_id, line_number, code, unit_type)
                     VALUES ($1, $2, $3, $4)
                     RETURNING id",
                    &[
                        &file_id,
                        &(unit.line_number as i32),
                        &unit.code,
                        &unit.unit_type,
                    ],
                )
                .await?
                .get(0);

            // Encode the code to get embeddings
            let embeddings = encoder
                .encode(&[unit.code.clone()])
                .context("Failed to encode code")?;

            // Store each embedding vector
            for (idx, embedding) in embeddings[0].iter().enumerate() {
                // Convert to pgvector format
                let vector_data: Vec<f32> = embedding.iter().copied().collect();
                let vector = pgvector::Vector::from(vector_data);

                client
                    .execute(
                        "INSERT INTO vectors (code_unit_id, vector_index, embedding)
                         VALUES ($1, $2, $3)",
                        &[&code_unit_id, &(idx as i32), &vector],
                    )
                    .await?;
            }
        }

        Ok(())
    }

    /// Delete vectors for a file
    async fn delete_file_vectors(&self, project_id: i64, file_path: &Path) -> Result<()> {
        let client = self.pool.get().await?;
        let file_path_str = file_path.to_string_lossy().to_string();

        client
            .execute(
                "DELETE FROM files WHERE project_id = $1 AND file_path = $2",
                &[&project_id, &file_path_str],
            )
            .await?;

        Ok(())
    }
}

#[async_trait]
impl Backend for LocalBackend {
    async fn initialize(&mut self) -> Result<()> {
        let client = self.pool.get().await?;

        // Ensure pgvector extension is enabled
        client
            .execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
            .await
            .context("Failed to enable pgvector extension")?;

        // Run migrations
        include_str!("../../migrations/20260214000001_initial_schema.sql")
            .split(';')
            .filter(|s| !s.trim().is_empty())
            .try_for_each(|stmt| async {
                client
                    .execute(stmt, &[])
                    .await
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!("Migration failed: {}", e))
            })
            .await?;

        Ok(())
    }

    async fn index_exists(&self, root: &Path) -> Result<bool> {
        let client = self.pool.get().await?;
        let root_str = root.to_string_lossy().to_string();

        let row = client
            .query_one(
                "SELECT EXISTS(SELECT 1 FROM projects WHERE root_path = $1)",
                &[&root_str],
            )
            .await?;

        Ok(row.get(0))
    }

    async fn index_full(&mut self, root: &Path, force: bool) -> Result<IndexStats> {
        let project_id = self.get_or_create_project(root).await?;

        // If forcing, delete existing data
        if force {
            let client = self.pool.get().await?;
            client
                .execute("DELETE FROM files WHERE project_id = $1", &[&project_id])
                .await?;
        }

        // Gather code units from the filesystem
        let code_units = colgrep::gather_code_units_from_path(root)?;

        let file_count = code_units
            .iter()
            .map(|u| &u.file_path)
            .collect::<std::collections::HashSet<_>>()
            .len();

        // Store vectors
        self.store_vectors(project_id, &code_units).await?;

        // Update last indexed timestamp
        let client = self.pool.get().await?;
        client
            .execute(
                "UPDATE projects SET last_indexed = NOW() WHERE id = $1",
                &[&project_id],
            )
            .await?;

        // Get statistics
        self.get_stats(root).await
    }

    async fn update_incremental(&mut self, root: &Path, changes: &[FileChange]) -> Result<()> {
        let project_id = self.get_or_create_project(root).await?;

        // Process each change
        for change in changes {
            match change {
                FileChange::Created(path) | FileChange::Modified(path) => {
                    // Delete existing vectors for this file
                    self.delete_file_vectors(project_id, path).await?;

                    // Re-index the file
                    if path.exists() {
                        let code_units = colgrep::gather_code_units_from_path(path)?;
                        self.store_vectors(project_id, &code_units).await?;
                    }
                }
                FileChange::Deleted(path) => {
                    // Delete vectors for this file
                    self.delete_file_vectors(project_id, path).await?;
                }
            }
        }

        // Update last indexed timestamp
        let client = self.pool.get().await?;
        client
            .execute(
                "UPDATE projects SET last_indexed = NOW() WHERE id = $1",
                &[&project_id],
            )
            .await?;

        Ok(())
    }

    async fn search(
        &self,
        root: &Path,
        query: &str,
        max_results: usize,
        include_patterns: Option<&[String]>,
        exclude_patterns: Option<&[String]>,
    ) -> Result<Vec<SearchResult>> {
        use colgrep::Encoder;

        let client = self.pool.get().await?;
        let root_str = root.to_string_lossy().to_string();

        // Get project ID
        let row = client
            .query_opt("SELECT id FROM projects WHERE root_path = $1", &[&root_str])
            .await?
            .context("Index does not exist for this project")?;

        let project_id: i64 = row.get(0);

        // Encode query
        let encoder = Encoder::new(None).context("Failed to load encoder")?;

        let query_embeddings = encoder
            .encode(&[query.to_string()])
            .context("Failed to encode query")?;

        // For ColBERT, we use the first query vector for simplicity
        let query_vector = &query_embeddings[0][0];
        let query_vector_data: Vec<f32> = query_vector.iter().copied().collect();
        let query_pgvector = pgvector::Vector::from(query_vector_data);

        // Build WHERE clause for file filtering
        let mut where_clauses = vec!["f.project_id = $1".to_string()];
        let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = vec![&project_id];
        let mut param_idx = 2;

        if let Some(include) = include_patterns {
            for pattern in include {
                let like_pattern = pattern.replace("*", "%").replace("?", "_");
                where_clauses.push(format!("f.file_path LIKE ${}", param_idx));
                param_idx += 1;
            }
        }

        if let Some(exclude) = exclude_patterns {
            for pattern in exclude {
                let like_pattern = pattern.replace("*", "%").replace("?", "_");
                where_clauses.push(format!("f.file_path NOT LIKE ${}", param_idx));
                param_idx += 1;
            }
        }

        let where_clause = where_clauses.join(" AND ");

        // Search using vector similarity
        let query_str = format!(
            "SELECT DISTINCT ON (cu.id)
                f.file_path,
                cu.line_number,
                cu.code,
                1 - (v.embedding <=> $2) as score
             FROM files f
             JOIN code_units cu ON cu.file_id = f.id
             JOIN vectors v ON v.code_unit_id = cu.id
             WHERE {}
             ORDER BY cu.id, score DESC
             LIMIT $3",
            where_clause
        );

        // Build params vec
        let mut all_params: Vec<Box<dyn tokio_postgres::types::ToSql + Sync + Send>> = vec![
            Box::new(project_id),
            Box::new(query_pgvector),
            Box::new(max_results as i32),
        ];

        if let Some(include) = include_patterns {
            for pattern in include {
                let like_pattern = pattern.replace("*", "%").replace("?", "_");
                all_params.push(Box::new(like_pattern));
            }
        }

        if let Some(exclude) = exclude_patterns {
            for pattern in exclude {
                let like_pattern = pattern.replace("*", "%").replace("?", "_");
                all_params.push(Box::new(like_pattern));
            }
        }

        // Execute query (simplified for now without filters)
        let simple_query = format!(
            "SELECT DISTINCT ON (cu.id)
                f.file_path,
                cu.line_number,
                cu.code,
                1 - (v.embedding <=> $2) as score
             FROM files f
             JOIN code_units cu ON cu.file_id = f.id
             JOIN vectors v ON v.code_unit_id = cu.id
             WHERE f.project_id = $1
             ORDER BY cu.id, score DESC
             LIMIT $3"
        );

        let rows = client
            .query(
                &simple_query,
                &[&project_id, &query_pgvector, &(max_results as i32)],
            )
            .await?;

        let mut results = Vec::new();
        for row in rows {
            results.push(SearchResult {
                file_path: row.get(0),
                line_number: row.get::<_, i32>(1) as usize,
                snippet: row.get(2),
                score: row.get::<_, f32>(3),
                context: None,
            });
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    async fn get_stats(&self, root: &Path) -> Result<IndexStats> {
        let client = self.pool.get().await?;
        let root_str = root.to_string_lossy().to_string();

        let row = client
            .query_opt(
                "SELECT
                    COUNT(DISTINCT f.id) as file_count,
                    COUNT(DISTINCT cu.id) as code_unit_count,
                    COUNT(v.id) as vector_count,
                    p.last_indexed
                 FROM projects p
                 LEFT JOIN files f ON f.project_id = p.id
                 LEFT JOIN code_units cu ON cu.file_id = f.id
                 LEFT JOIN vectors v ON v.code_unit_id = cu.id
                 WHERE p.root_path = $1
                 GROUP BY p.id, p.last_indexed",
                &[&root_str],
            )
            .await?
            .context("Project not found")?;

        Ok(IndexStats {
            file_count: row.get::<_, i64>(0) as usize,
            code_unit_count: row.get::<_, i64>(1) as usize,
            vector_count: row.get::<_, i64>(2) as usize,
            size_bytes: 0, // Would need to query pg_total_relation_size
            last_updated: row
                .get::<_, Option<chrono::DateTime<chrono::Utc>>>(3)
                .map(|dt| dt.timestamp()),
        })
    }

    async fn delete_index(&mut self, root: &Path) -> Result<()> {
        let client = self.pool.get().await?;
        let root_str = root.to_string_lossy().to_string();

        client
            .execute("DELETE FROM projects WHERE root_path = $1", &[&root_str])
            .await?;

        Ok(())
    }
}
