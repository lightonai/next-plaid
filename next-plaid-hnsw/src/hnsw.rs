//! Core HNSW (Hierarchical Navigable Small World) implementation.
//!
//! This module implements a memory-efficient HNSW index with:
//! - Memory-mapped file storage for low RAM usage
//! - Parallel search using rayon
//! - Configurable parameters with sensible defaults

use crate::error::{Error, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::{Mmap, MmapOptions};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Configuration for the HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node at each layer (default: 16).
    pub m: usize,
    /// Maximum number of connections at layer 0 (default: 2 * m).
    pub m0: usize,
    /// Size of dynamic candidate list during construction (default: 100).
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (default: 50).
    pub ef_search: usize,
    /// Normalization factor for level generation (default: 1 / ln(m)).
    pub ml: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 100,
            ef_search: 50,
            ml: 1.0 / (m as f32).ln(),
            seed: 42,
        }
    }
}

impl HnswConfig {
    /// Create a new configuration with custom M parameter.
    pub fn with_m(m: usize) -> Self {
        Self {
            m,
            m0: m * 2,
            ml: 1.0 / (m as f32).ln(),
            ..Default::default()
        }
    }

    /// Set the ef_construction parameter.
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set the ef_search parameter.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }
}

/// Metadata stored alongside the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswMetadata {
    /// Index configuration.
    pub config: HnswConfig,
    /// Dimension of vectors.
    pub dim: usize,
    /// Total number of vectors in the index.
    pub num_vectors: usize,
    /// Entry point node ID.
    pub entry_point: Option<usize>,
    /// Maximum level in the graph.
    pub max_level: usize,
}

/// Node representation in the HNSW graph.
#[derive(Debug, Clone, Default)]
struct Node {
    /// Neighbors at each level (level 0 to max_level).
    neighbors: Vec<Vec<usize>>,
}

impl Node {
    fn new(level: usize, m: usize, m0: usize) -> Self {
        let mut neighbors = Vec::with_capacity(level + 1);
        for l in 0..=level {
            let capacity = if l == 0 { m0 } else { m };
            neighbors.push(Vec::with_capacity(capacity));
        }
        Self { neighbors }
    }
}

/// Memory-mapped HNSW index.
pub struct HnswIndex {
    /// Directory containing index files.
    directory: PathBuf,
    /// Index metadata.
    metadata: HnswMetadata,
    /// Memory-mapped vector data.
    vectors_mmap: Option<Mmap>,
    /// In-memory vectors stored as contiguous f32 array for cache efficiency.
    vectors_flat: Vec<f32>,
    /// Dimension of vectors.
    vectors_dim: usize,
    /// Graph structure (nodes with their neighbors).
    nodes: Vec<RwLock<Node>>,
    /// RNG for level generation.
    rng: ChaCha8Rng,
    /// Start index of vectors in flat cache (for hybrid access during updates).
    new_vectors_start: usize,
    /// Number of vectors currently in flat cache.
    new_vectors_count: usize,
}

impl HnswIndex {
    /// Create a new empty HNSW index in the given directory.
    pub fn new<P: AsRef<Path>>(directory: P, dim: usize, config: HnswConfig) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory)?;

        let metadata = HnswMetadata {
            config: config.clone(),
            dim,
            num_vectors: 0,
            entry_point: None,
            max_level: 0,
        };

        let rng = ChaCha8Rng::seed_from_u64(config.seed);

        let index = Self {
            directory,
            metadata,
            vectors_mmap: None,
            vectors_flat: Vec::new(),
            vectors_dim: dim,
            nodes: Vec::new(),
            rng,
            new_vectors_start: 0,
            new_vectors_count: 0,
        };

        Ok(index)
    }

    /// Load an existing HNSW index from a directory.
    pub fn load<P: AsRef<Path>>(directory: P) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();

        if !directory.exists() {
            return Err(Error::DirectoryNotFound(
                directory.to_string_lossy().to_string(),
            ));
        }

        // Load metadata
        let metadata_path = directory.join("hnsw_metadata.json");
        let metadata_file = File::open(&metadata_path)
            .map_err(|_| Error::CorruptedIndex("hnsw_metadata.json not found".to_string()))?;
        let metadata: HnswMetadata = serde_json::from_reader(BufReader::new(metadata_file))?;

        // Load graph
        let graph_path = directory.join("hnsw_graph.bin");
        let nodes = Self::load_graph(&graph_path)?;

        // Memory-map vectors
        let vectors_path = directory.join("hnsw_vectors.bin");
        let vectors_mmap = if vectors_path.exists() && metadata.num_vectors > 0 {
            let file = File::open(&vectors_path)?;
            Some(unsafe { MmapOptions::new().map(&file)? })
        } else {
            None
        };

        let rng = ChaCha8Rng::seed_from_u64(metadata.config.seed);
        let dim = metadata.dim;

        Ok(Self {
            directory,
            metadata,
            vectors_mmap,
            vectors_flat: Vec::new(),
            vectors_dim: dim,
            nodes,
            rng,
            new_vectors_start: 0,
            new_vectors_count: 0,
        })
    }

    /// Update the index by adding new vectors.
    ///
    /// Vectors are indexed from 0 to n-1 where n is the total number of vectors.
    /// Returns the starting index of the newly added vectors.
    pub fn update(&mut self, vectors: &Array2<f32>) -> Result<usize> {
        let (num_new, dim) = vectors.dim();

        if num_new == 0 {
            return Ok(self.metadata.num_vectors);
        }

        // Check dimension
        if self.metadata.num_vectors > 0 && dim != self.metadata.dim {
            return Err(Error::DimensionMismatch {
                expected: self.metadata.dim,
                got: dim,
            });
        }

        // Update dimension if this is the first insertion
        if self.metadata.num_vectors == 0 {
            self.metadata.dim = dim;
            self.vectors_dim = dim;
        }

        let start_idx = self.metadata.num_vectors;

        // Memory-efficient hybrid approach:
        // - Don't load existing vectors into memory (they stay in mmap)
        // - Only keep new vectors in flat cache
        // - Use hybrid access methods during insertion

        // Clear any stale data and set up hybrid tracking
        self.vectors_flat.clear();
        self.new_vectors_start = start_idx;
        self.new_vectors_count = num_new;

        // Add new vectors to flat cache
        self.vectors_flat.reserve(num_new * dim);
        for row in vectors.axis_iter(Axis(0)) {
            self.vectors_flat.extend(row.iter());
        }

        // Insert each vector using hybrid access (new vectors from flat cache, existing from mmap)
        for i in 0..num_new {
            self.insert_node_hybrid(start_idx + i)?;
        }

        // Append vectors to the vectors file
        self.append_vectors(vectors)?;

        // Save updated metadata and graph
        self.save()?;

        // Clear cache and reset hybrid tracking
        self.vectors_flat.clear();
        self.vectors_flat.shrink_to_fit();
        self.new_vectors_start = 0;
        self.new_vectors_count = 0;
        self.reload_vectors_mmap()?;

        Ok(start_idx)
    }

    /// Search for the k nearest neighbors of query vectors.
    ///
    /// Returns (scores, indices) where:
    /// - scores: `Array2<f32>` of shape (num_queries, k) with similarity scores (higher is better)
    /// - indices: `Array2<i64>` of shape (num_queries, k) with vector indices (-1 for padding)
    pub fn search(&self, queries: &Array2<f32>, k: usize) -> Result<(Array2<f32>, Array2<i64>)> {
        self.search_with_filter(queries, k, None)
    }

    /// Search for the k nearest neighbors with a custom ef_search parameter.
    ///
    /// Higher ef_search values explore more candidates, giving better recall but slower search.
    /// This is useful when you need high accuracy (e.g., for outlier detection).
    ///
    /// # Arguments
    /// - `queries`: Query vectors of shape (num_queries, dim)
    /// - `k`: Number of nearest neighbors to return
    /// - `ef_search`: Size of dynamic candidate list during search (higher = better recall, slower)
    ///
    /// # Returns
    /// (scores, indices) where:
    /// - scores: `Array2<f32>` of shape (num_queries, k) with similarity scores (higher is better)
    /// - indices: `Array2<i64>` of shape (num_queries, k) with vector indices (-1 for padding)
    pub fn search_with_ef(
        &self,
        queries: &Array2<f32>,
        k: usize,
        ef_search: usize,
    ) -> Result<(Array2<f32>, Array2<i64>)> {
        if self.metadata.num_vectors == 0 {
            return Err(Error::EmptyIndex);
        }

        let (num_queries, dim) = queries.dim();

        if dim != self.metadata.dim {
            return Err(Error::DimensionMismatch {
                expected: self.metadata.dim,
                got: dim,
            });
        }

        // Parallel search for all queries with custom ef
        let results: Vec<Vec<(f32, usize)>> = queries
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|query| self.search_single_with_ef(query, k, ef_search))
            .collect();

        // Convert to output arrays
        let mut scores = Array2::from_elem((num_queries, k), f32::NEG_INFINITY);
        let mut indices = Array2::from_elem((num_queries, k), -1i64);

        for (i, neighbors) in results.iter().enumerate() {
            for (j, (score, idx)) in neighbors.iter().enumerate() {
                if j < k {
                    scores[[i, j]] = *score;
                    indices[[i, j]] = *idx as i64;
                }
            }
        }

        Ok((scores, indices))
    }

    /// Search for the k nearest neighbors of query vectors with an optional filter.
    ///
    /// The HNSW algorithm will explore all vectors during graph traversal, but only
    /// vectors whose IDs are in the filter set will be returned as results.
    ///
    /// # Arguments
    /// - `queries`: Query vectors of shape (num_queries, dim)
    /// - `k`: Number of nearest neighbors to return
    /// - `filter`: Optional set of vector IDs to include in results. If None, all vectors are considered.
    ///   This filter applies to ALL queries uniformly.
    ///
    /// # Returns
    /// (scores, indices) where:
    /// - scores: `Array2<f32>` of shape (num_queries, k) with similarity scores (higher is better)
    /// - indices: `Array2<i64>` of shape (num_queries, k) with vector indices (-1 for padding)
    pub fn search_with_filter(
        &self,
        queries: &Array2<f32>,
        k: usize,
        filter: Option<&HashSet<usize>>,
    ) -> Result<(Array2<f32>, Array2<i64>)> {
        if self.metadata.num_vectors == 0 {
            return Err(Error::EmptyIndex);
        }

        let (num_queries, dim) = queries.dim();

        if dim != self.metadata.dim {
            return Err(Error::DimensionMismatch {
                expected: self.metadata.dim,
                got: dim,
            });
        }

        // Parallel search for all queries
        let results: Vec<Vec<(f32, usize)>> = queries
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|query| self.search_single_filtered(query, k, filter))
            .collect();

        // Convert to output arrays
        let mut scores = Array2::from_elem((num_queries, k), f32::NEG_INFINITY);
        let mut indices = Array2::from_elem((num_queries, k), -1i64);

        for (i, neighbors) in results.iter().enumerate() {
            for (j, (score, idx)) in neighbors.iter().enumerate() {
                if j < k {
                    scores[[i, j]] = *score;
                    indices[[i, j]] = *idx as i64;
                }
            }
        }

        Ok((scores, indices))
    }

    /// Search for the k nearest neighbors of query vectors with per-query candidate lists.
    ///
    /// Each query has its own list of candidate vector IDs to consider for scoring.
    /// The HNSW algorithm will explore all vectors during graph traversal, but only
    /// vectors whose IDs are in the respective query's candidate list will be returned as results.
    ///
    /// # Arguments
    /// - `queries`: Query vectors of shape (num_queries, dim)
    /// - `k`: Number of nearest neighbors to return
    /// - `candidate_ids`: Slice of candidate ID slices, one per query. Must have length equal to num_queries.
    ///   Each inner slice contains the vector IDs that query is allowed to return.
    ///   Each query can have a different number of candidates.
    ///
    /// # Returns
    /// (scores, indices) where:
    /// - scores: `Array2<f32>` of shape (num_queries, k) with similarity scores (higher is better)
    /// - indices: `Array2<i64>` of shape (num_queries, k) with vector indices (-1 for padding)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Query 0 can only return from vectors [1, 5, 10, 20]
    /// // Query 1 can only return from vectors [100, 101, 102, 103, 104, 105, 106]
    /// // Query 2 can only return from vectors [50, 51]
    /// let candidate_ids: Vec<Vec<usize>> = vec![
    ///     vec![1, 5, 10, 20],
    ///     vec![100, 101, 102, 103, 104, 105, 106],
    ///     vec![50, 51],
    /// ];
    /// let candidate_refs: Vec<&[usize]> = candidate_ids.iter().map(|v| v.as_slice()).collect();
    /// let (scores, indices) = index.search_with_ids(&queries, k, &candidate_refs)?;
    /// ```
    ///
    /// # Errors
    /// Returns an error if the number of candidate lists doesn't match the number of queries.
    pub fn search_with_ids(
        &self,
        queries: &Array2<f32>,
        k: usize,
        candidate_ids: &[&[usize]],
    ) -> Result<(Array2<f32>, Array2<i64>)> {
        if self.metadata.num_vectors == 0 {
            return Err(Error::EmptyIndex);
        }

        let (num_queries, dim) = queries.dim();

        if dim != self.metadata.dim {
            return Err(Error::DimensionMismatch {
                expected: self.metadata.dim,
                got: dim,
            });
        }

        if candidate_ids.len() != num_queries {
            return Err(Error::InvalidArgument(format!(
                "Number of candidate lists ({}) must match number of queries ({})",
                candidate_ids.len(),
                num_queries
            )));
        }

        // Convert slices to HashSets for O(1) lookup during search
        let filters: Vec<HashSet<usize>> = candidate_ids
            .iter()
            .map(|ids| ids.iter().copied().collect())
            .collect();

        // Parallel search for all queries with their respective filters
        let results: Vec<Vec<(f32, usize)>> = queries
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(filters.par_iter())
            .map(|(query, filter)| {
                if filter.is_empty() {
                    Vec::new()
                } else {
                    self.search_single_filtered(query, k, Some(filter))
                }
            })
            .collect();

        // Convert to output arrays
        let mut scores = Array2::from_elem((num_queries, k), f32::NEG_INFINITY);
        let mut indices = Array2::from_elem((num_queries, k), -1i64);

        for (i, neighbors) in results.iter().enumerate() {
            for (j, (score, idx)) in neighbors.iter().enumerate() {
                if j < k {
                    scores[[i, j]] = *score;
                    indices[[i, j]] = *idx as i64;
                }
            }
        }

        Ok((scores, indices))
    }

    /// Get the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.metadata.num_vectors
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.metadata.num_vectors == 0
    }

    /// Get the dimension of vectors.
    pub fn dim(&self) -> usize {
        self.metadata.dim
    }

    /// Get the index configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.metadata.config
    }

    /// Get a single vector by ID.
    ///
    /// Returns the vector at the given index, or an error if the index is out of bounds
    /// or the vector cannot be read.
    pub fn get_vector(&self, id: usize) -> Result<Array1<f32>> {
        if id >= self.metadata.num_vectors {
            return Err(Error::InvalidArgument(format!(
                "Vector ID {} out of bounds (index has {} vectors)",
                id, self.metadata.num_vectors
            )));
        }

        self.get_vector_from_mmap(id).ok_or_else(|| {
            Error::CorruptedIndex(format!(
                "Failed to read vector {} from hnsw_vectors.bin",
                id
            ))
        })
    }

    /// Get the squared L2 norm of a vector by ID.
    ///
    /// Useful for converting inner product similarity to L2 distance:
    /// `L2² = ||a||² + ||b||² - 2 * inner_product(a, b)`
    pub fn get_vector_norm_sq(&self, id: usize) -> Result<f32> {
        let vec = self.get_vector(id)?;
        Ok(vec.dot(&vec))
    }

    /// Get all vectors as a 2D array.
    ///
    /// Used for exact brute-force search when the index is small.
    /// Returns vectors of shape (num_vectors, dim).
    ///
    /// **Warning**: This loads all vectors into memory. For large indices,
    /// use `get_vectors_range` to load in batches.
    pub fn get_all_vectors(&self) -> Result<Array2<f32>> {
        let n = self.metadata.num_vectors;
        let dim = self.metadata.dim;

        if n == 0 {
            return Ok(Array2::zeros((0, dim)));
        }

        let mut vectors = Array2::zeros((n, dim));

        for i in 0..n {
            if let Some(vec) = self.get_vector_from_mmap(i) {
                vectors.row_mut(i).assign(&vec);
            } else {
                return Err(Error::CorruptedIndex(format!(
                    "Failed to read vector {} from hnsw_vectors.bin",
                    i
                )));
            }
        }

        Ok(vectors)
    }

    /// Get a range of vectors as a 2D array.
    ///
    /// Loads vectors from index `start` (inclusive) to `end` (exclusive).
    /// Useful for loading vectors in batches to avoid memory issues with large indices.
    ///
    /// # Arguments
    /// - `start`: Starting index (inclusive)
    /// - `end`: Ending index (exclusive), clamped to num_vectors
    ///
    /// # Returns
    /// `Array2<f32>` of shape (end - start, dim)
    pub fn get_vectors_range(&self, start: usize, end: usize) -> Result<Array2<f32>> {
        let n = self.metadata.num_vectors;
        let dim = self.metadata.dim;

        let start = start.min(n);
        let end = end.min(n);

        if start >= end {
            return Ok(Array2::zeros((0, dim)));
        }

        let count = end - start;
        let mut vectors = Array2::zeros((count, dim));

        for i in 0..count {
            if let Some(vec) = self.get_vector_from_mmap(start + i) {
                vectors.row_mut(i).assign(&vec);
            } else {
                return Err(Error::CorruptedIndex(format!(
                    "Failed to read vector {} from hnsw_vectors.bin",
                    start + i
                )));
            }
        }

        Ok(vectors)
    }

    /// Save the index to disk.
    pub fn save(&self) -> Result<()> {
        // Save metadata
        let metadata_path = self.directory.join("hnsw_metadata.json");
        let metadata_file = File::create(&metadata_path)?;
        serde_json::to_writer_pretty(BufWriter::new(metadata_file), &self.metadata)?;

        // Save graph
        let graph_path = self.directory.join("hnsw_graph.bin");
        self.save_graph(&graph_path)?;

        Ok(())
    }

    // ============== Private Methods ==============

    // Legacy flat-only methods (kept for reference and potential future batch operations
    // where all vectors are already in memory). Currently unused since we use hybrid methods.

    /// Get vector from flat cache by index.
    #[allow(dead_code)]
    #[inline(always)]
    fn get_vector_flat(&self, id: usize) -> &[f32] {
        let start = id * self.vectors_dim;
        &self.vectors_flat[start..start + self.vectors_dim]
    }

    /// Compute dot product between query slice and vector in flat cache.
    #[allow(dead_code)]
    #[inline(always)]
    fn dot_product_flat(&self, query: &[f32], id: usize) -> f32 {
        let vec = self.get_vector_flat(id);
        query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum()
    }

    /// Get vector slice from flat cache if it's a new vector (within the current batch).
    /// Returns None if the vector is not in the flat cache (i.e., it's an existing vector in mmap).
    #[inline(always)]
    fn get_vector_from_flat_cache(&self, id: usize) -> Option<&[f32]> {
        if id >= self.new_vectors_start && id < self.new_vectors_start + self.new_vectors_count {
            let cache_idx = id - self.new_vectors_start;
            let start = cache_idx * self.vectors_dim;
            let end = start + self.vectors_dim;
            if end <= self.vectors_flat.len() {
                return Some(&self.vectors_flat[start..end]);
            }
        }
        None
    }

    /// Compute dot product using hybrid access (flat cache for new vectors, mmap for existing).
    /// This is used during incremental updates to avoid loading all existing vectors into memory.
    #[inline]
    fn dot_product_hybrid(&self, query: &[f32], id: usize) -> f32 {
        // Fast path: check flat cache first (new vectors in current batch)
        if let Some(vec) = self.get_vector_from_flat_cache(id) {
            return query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
        }
        // Slow path: read from mmap (existing vectors)
        if let Some(vec) = self.get_vector_from_mmap(id) {
            return query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
        }
        0.0 // Should not happen - vector not found
    }

    /// Get vector as owned Vec using hybrid access.
    /// Used when we need ownership of the vector data (e.g., for neighbor pruning).
    fn get_vector_hybrid_owned(&self, id: usize) -> Option<Vec<f32>> {
        // Try flat cache first (new vectors)
        if let Some(slice) = self.get_vector_from_flat_cache(id) {
            return Some(slice.to_vec());
        }
        // Fall back to mmap (existing vectors)
        self.get_vector_from_mmap(id).map(|arr| arr.to_vec())
    }

    /// Load vectors from mmap to flat cache.
    #[allow(dead_code)]
    fn load_vectors_to_flat(&mut self) -> Result<()> {
        let mmap = self
            .vectors_mmap
            .as_ref()
            .ok_or_else(|| Error::CorruptedIndex("hnsw_vectors.bin not loaded".to_string()))?;

        let dim = self.metadata.dim;
        let num_vectors = self.metadata.num_vectors;
        self.vectors_flat.clear();
        self.vectors_flat.reserve(num_vectors * dim);

        for id in 0..num_vectors {
            let offset = id * dim * std::mem::size_of::<f32>();
            let bytes = &mmap[offset..offset + dim * std::mem::size_of::<f32>()];
            let mut cursor = std::io::Cursor::new(bytes);

            for _ in 0..dim {
                self.vectors_flat.push(cursor.read_f32::<LittleEndian>()?);
            }
        }

        Ok(())
    }

    /// Append vectors to the vectors file.
    fn append_vectors(&self, vectors: &Array2<f32>) -> Result<()> {
        let vectors_path = self.directory.join("hnsw_vectors.bin");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&vectors_path)?;

        for row in vectors.axis_iter(Axis(0)) {
            for &val in row.iter() {
                file.write_f32::<LittleEndian>(val)?;
            }
        }

        file.flush()?;
        Ok(())
    }

    /// Reload the memory-mapped vectors file.
    fn reload_vectors_mmap(&mut self) -> Result<()> {
        let vectors_path = self.directory.join("hnsw_vectors.bin");
        if vectors_path.exists() {
            let file = File::open(&vectors_path)?;
            self.vectors_mmap = Some(unsafe { MmapOptions::new().map(&file)? });
        }
        Ok(())
    }

    /// Get a vector from mmap storage.
    fn get_vector_from_mmap(&self, id: usize) -> Option<Array1<f32>> {
        let mmap = self.vectors_mmap.as_ref()?;
        let dim = self.metadata.dim;
        let offset = id * dim * std::mem::size_of::<f32>();
        let end = offset + dim * std::mem::size_of::<f32>();

        if end > mmap.len() {
            return None;
        }

        let bytes = &mmap[offset..end];
        let mut vector = Array1::zeros(dim);
        let mut cursor = std::io::Cursor::new(bytes);

        for i in 0..dim {
            vector[i] = cursor.read_f32::<LittleEndian>().ok()?;
        }

        Some(vector)
    }

    /// Compute inner product (dot product) between two vectors.
    #[inline]
    fn inner_product(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
        a.dot(&b)
    }

    /// Generate a random level for a new node.
    fn random_level(&mut self) -> usize {
        let r: f32 = self.rng.gen();
        (-r.ln() * self.metadata.config.ml).floor() as usize
    }

    /// Insert a new node into the graph.
    #[allow(dead_code)]
    fn insert_node(&mut self, id: usize) -> Result<()> {
        let level = self.random_level();
        let m = self.metadata.config.m;
        let m0 = self.metadata.config.m0;
        let ef_construction = self.metadata.config.ef_construction;

        // Create new node
        let node = Node::new(level, m, m0);
        self.nodes.push(RwLock::new(node));

        // Get query vector as slice for fast access
        let query = self.get_vector_flat(id);

        // If this is the first node, set it as entry point
        if self.metadata.entry_point.is_none() {
            self.metadata.entry_point = Some(id);
            self.metadata.max_level = level;
            self.metadata.num_vectors += 1;
            return Ok(());
        }

        let entry_point = self.metadata.entry_point.unwrap();
        let mut curr_node = entry_point;

        // Greedy search from top level to level + 1
        for l in (level + 1..=self.metadata.max_level).rev() {
            curr_node = self.greedy_search_layer_flat(query, curr_node, l);
        }

        // Search and connect at each level from min(level, max_level) to 0
        let start_level = level.min(self.metadata.max_level);
        for l in (0..=start_level).rev() {
            // Search for nearest neighbors at this level
            let neighbors = self.search_layer_build_flat(query, curr_node, ef_construction, l);

            // Select the best M (or M0 for level 0) neighbors
            let max_connections = if l == 0 { m0 } else { m };
            let selected: Vec<usize> = neighbors
                .iter()
                .take(max_connections)
                .map(|(_, idx)| *idx)
                .collect();

            // Add connections to the new node
            {
                let mut new_node = self.nodes[id].write();
                if l < new_node.neighbors.len() {
                    new_node.neighbors[l] = selected.clone();
                }
            }

            // Add reverse connections and prune if necessary
            for &neighbor_id in &selected {
                let mut neighbor = self.nodes[neighbor_id].write();
                if l < neighbor.neighbors.len() {
                    if !neighbor.neighbors[l].contains(&id) {
                        neighbor.neighbors[l].push(id);
                    }

                    // Prune if we have too many connections
                    if neighbor.neighbors[l].len() > max_connections {
                        let neighbor_vec = self.get_vector_flat(neighbor_id);
                        let mut candidates: Vec<(f32, usize)> = neighbor.neighbors[l]
                            .iter()
                            .map(|&n| {
                                let v = self.get_vector_flat(n);
                                let dist: f32 =
                                    neighbor_vec.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                                (dist, n)
                            })
                            .collect();

                        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                        neighbor.neighbors[l] = candidates
                            .iter()
                            .take(max_connections)
                            .map(|(_, idx)| *idx)
                            .collect();
                    }
                }
            }

            // Update current node for next level
            if !neighbors.is_empty() {
                curr_node = neighbors[0].1;
            }
        }

        // Update entry point if new node has higher level
        if level > self.metadata.max_level {
            self.metadata.entry_point = Some(id);
            self.metadata.max_level = level;
        }

        self.metadata.num_vectors += 1;

        Ok(())
    }

    /// Greedy search to find closest node at a given layer (using flat vectors).
    #[allow(dead_code)]
    fn greedy_search_layer_flat(&self, query: &[f32], entry_point: usize, level: usize) -> usize {
        let mut curr_node = entry_point;
        let mut curr_dist = self.dot_product_flat(query, curr_node);

        loop {
            let node = self.nodes[curr_node].read();
            if level >= node.neighbors.len() {
                break;
            }

            let mut best_dist = curr_dist;
            let mut best_node = curr_node;

            for &neighbor in &node.neighbors[level] {
                let dist = self.dot_product_flat(query, neighbor);
                if dist > best_dist {
                    best_dist = dist;
                    best_node = neighbor;
                }
            }

            if best_node == curr_node {
                break;
            }
            curr_node = best_node;
            curr_dist = best_dist;
        }

        curr_node
    }

    /// Search for nearest neighbors at a specific layer during build (using flat vectors).
    #[allow(dead_code)]
    fn search_layer_build_flat(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(f32, usize)> {
        let entry_dist = self.dot_product_flat(query, entry_point);

        // Use a simple visited bitset for small indices, fallback to vec for large
        let num_nodes = self.nodes.len();
        let mut visited = vec![false; num_nodes];
        visited[entry_point] = true;

        // candidates: max-heap ordered by distance (we want to explore best first)
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        candidates.push((OrderedFloat(entry_dist), entry_point));

        // results: min-heap to track worst result for pruning
        let mut results: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
        results.push(Reverse((OrderedFloat(entry_dist), entry_point)));

        while let Some((OrderedFloat(cand_dist), cand_id)) = candidates.pop() {
            // Get the worst distance in results
            let worst_dist = results
                .peek()
                .map(|Reverse((d, _))| d.0)
                .unwrap_or(f32::NEG_INFINITY);

            // If candidate is worse than our worst result and we have enough results, stop
            if results.len() >= ef && cand_dist < worst_dist {
                break;
            }

            let node = self.nodes[cand_id].read();
            if level >= node.neighbors.len() {
                continue;
            }

            for &neighbor in &node.neighbors[level] {
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;

                let dist = self.dot_product_flat(query, neighbor);

                let worst = results
                    .peek()
                    .map(|Reverse((d, _))| d.0)
                    .unwrap_or(f32::NEG_INFINITY);

                if results.len() < ef || dist > worst {
                    candidates.push((OrderedFloat(dist), neighbor));
                    results.push(Reverse((OrderedFloat(dist), neighbor)));

                    // Keep only ef best results
                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Convert to sorted vector (highest distance first)
        let mut result_vec: Vec<(f32, usize)> = results
            .into_iter()
            .map(|Reverse((d, id))| (d.0, id))
            .collect();
        result_vec.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        result_vec
    }

    /// Greedy search to find closest node at a given layer (using hybrid vectors).
    /// Used during incremental updates when new vectors are in flat cache and existing in mmap.
    fn greedy_search_layer_hybrid(&self, query: &[f32], entry_point: usize, level: usize) -> usize {
        let mut curr_node = entry_point;
        let mut curr_dist = self.dot_product_hybrid(query, curr_node);

        loop {
            let node = self.nodes[curr_node].read();
            if level >= node.neighbors.len() {
                break;
            }

            let mut best_dist = curr_dist;
            let mut best_node = curr_node;

            for &neighbor in &node.neighbors[level] {
                let dist = self.dot_product_hybrid(query, neighbor);
                if dist > best_dist {
                    best_dist = dist;
                    best_node = neighbor;
                }
            }

            if best_node == curr_node {
                break;
            }
            curr_node = best_node;
            curr_dist = best_dist;
        }

        curr_node
    }

    /// Search for nearest neighbors at a specific layer during build (using hybrid vectors).
    /// Used during incremental updates when new vectors are in flat cache and existing in mmap.
    fn search_layer_build_hybrid(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(f32, usize)> {
        let entry_dist = self.dot_product_hybrid(query, entry_point);

        let num_nodes = self.nodes.len();
        let mut visited = vec![false; num_nodes];
        visited[entry_point] = true;

        let mut candidates: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        candidates.push((OrderedFloat(entry_dist), entry_point));

        let mut results: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
        results.push(Reverse((OrderedFloat(entry_dist), entry_point)));

        while let Some((OrderedFloat(cand_dist), cand_id)) = candidates.pop() {
            let worst_dist = results
                .peek()
                .map(|Reverse((d, _))| d.0)
                .unwrap_or(f32::NEG_INFINITY);

            if results.len() >= ef && cand_dist < worst_dist {
                break;
            }

            let node = self.nodes[cand_id].read();
            if level >= node.neighbors.len() {
                continue;
            }

            for &neighbor in &node.neighbors[level] {
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;

                let dist = self.dot_product_hybrid(query, neighbor);

                let worst = results
                    .peek()
                    .map(|Reverse((d, _))| d.0)
                    .unwrap_or(f32::NEG_INFINITY);

                if results.len() < ef || dist > worst {
                    candidates.push((OrderedFloat(dist), neighbor));
                    results.push(Reverse((OrderedFloat(dist), neighbor)));

                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<(f32, usize)> = results
            .into_iter()
            .map(|Reverse((d, id))| (d.0, id))
            .collect();
        result_vec.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        result_vec
    }

    /// Insert a new node into the graph using hybrid vector access.
    /// This is the memory-efficient version that uses mmap for existing vectors
    /// and flat cache only for the new vectors being inserted.
    fn insert_node_hybrid(&mut self, id: usize) -> Result<()> {
        let level = self.random_level();
        let m = self.metadata.config.m;
        let m0 = self.metadata.config.m0;
        let ef_construction = self.metadata.config.ef_construction;

        // Create new node
        let node = Node::new(level, m, m0);
        self.nodes.push(RwLock::new(node));

        // Get query vector - must clone since it's in flat cache and we need ownership
        // to avoid borrow checker issues during the rest of the method
        let query: Vec<f32> = self
            .get_vector_from_flat_cache(id)
            .expect("New vector should be in flat cache during hybrid insert")
            .to_vec();

        // If this is the first node, set it as entry point
        if self.metadata.entry_point.is_none() {
            self.metadata.entry_point = Some(id);
            self.metadata.max_level = level;
            self.metadata.num_vectors += 1;
            return Ok(());
        }

        let entry_point = self.metadata.entry_point.unwrap();
        let mut curr_node = entry_point;

        // Greedy search from top level to level + 1
        for l in (level + 1..=self.metadata.max_level).rev() {
            curr_node = self.greedy_search_layer_hybrid(&query, curr_node, l);
        }

        // Search and connect at each level from min(level, max_level) to 0
        let start_level = level.min(self.metadata.max_level);
        for l in (0..=start_level).rev() {
            // Search for nearest neighbors at this level
            let neighbors = self.search_layer_build_hybrid(&query, curr_node, ef_construction, l);

            // Select the best M (or M0 for level 0) neighbors
            let max_connections = if l == 0 { m0 } else { m };
            let selected: Vec<usize> = neighbors
                .iter()
                .take(max_connections)
                .map(|(_, idx)| *idx)
                .collect();

            // Add connections to the new node
            {
                let mut new_node = self.nodes[id].write();
                if l < new_node.neighbors.len() {
                    new_node.neighbors[l] = selected.clone();
                }
            }

            // Add reverse connections and prune if necessary
            for &neighbor_id in &selected {
                let mut neighbor = self.nodes[neighbor_id].write();
                if l < neighbor.neighbors.len() {
                    if !neighbor.neighbors[l].contains(&id) {
                        neighbor.neighbors[l].push(id);
                    }

                    // Prune if we have too many connections
                    if neighbor.neighbors[l].len() > max_connections {
                        // Get neighbor vector using hybrid access (may be in flat cache or mmap)
                        let neighbor_vec = self
                            .get_vector_hybrid_owned(neighbor_id)
                            .expect("Neighbor vector must exist");

                        let mut candidates: Vec<(f32, usize)> = neighbor.neighbors[l]
                            .iter()
                            .map(|&n| {
                                let dist = self.dot_product_hybrid(&neighbor_vec, n);
                                (dist, n)
                            })
                            .collect();

                        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                        neighbor.neighbors[l] = candidates
                            .iter()
                            .take(max_connections)
                            .map(|(_, idx)| *idx)
                            .collect();
                    }
                }
            }

            // Update current node for next level
            if !neighbors.is_empty() {
                curr_node = neighbors[0].1;
            }
        }

        // Update entry point if new node has higher level
        if level > self.metadata.max_level {
            self.metadata.entry_point = Some(id);
            self.metadata.max_level = level;
        }

        self.metadata.num_vectors += 1;

        Ok(())
    }

    /// Search for k nearest neighbors of a single query with optional filter (uses mmap).
    ///
    /// The filter only affects which results are returned - all vectors are still
    /// explored during graph traversal for proper HNSW search behavior.
    fn search_single_filtered(
        &self,
        query: ArrayView1<f32>,
        k: usize,
        filter: Option<&HashSet<usize>>,
    ) -> Vec<(f32, usize)> {
        let entry_point = match self.metadata.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let entry_vec = match self.get_vector_from_mmap(entry_point) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut curr_node = entry_point;
        let mut curr_dist = Self::inner_product(query, entry_vec.view());

        // Greedy search from top level to level 1
        for l in (1..=self.metadata.max_level).rev() {
            loop {
                let node = self.nodes[curr_node].read();
                if l >= node.neighbors.len() {
                    break;
                }

                let mut changed = false;
                for &neighbor in &node.neighbors[l] {
                    if let Some(neighbor_vec) = self.get_vector_from_mmap(neighbor) {
                        let dist = Self::inner_product(query, neighbor_vec.view());
                        if dist > curr_dist {
                            curr_dist = dist;
                            curr_node = neighbor;
                            changed = true;
                        }
                    }
                }

                if !changed {
                    break;
                }
            }
        }

        // Search at level 0 with ef_search
        // When filtering, we need a larger ef to ensure we find enough matching results
        let ef = if filter.is_some() {
            // Increase ef when filtering to explore more candidates
            (self.metadata.config.ef_search * 2).max(k * 4)
        } else {
            self.metadata.config.ef_search.max(k)
        };
        self.search_layer_query_filtered(query, curr_node, ef, k, filter)
    }

    /// Search for k nearest neighbors of a single query with custom ef_search (uses mmap).
    fn search_single_with_ef(
        &self,
        query: ArrayView1<f32>,
        k: usize,
        ef_search: usize,
    ) -> Vec<(f32, usize)> {
        let entry_point = match self.metadata.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let entry_vec = match self.get_vector_from_mmap(entry_point) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut curr_node = entry_point;
        let mut curr_dist = Self::inner_product(query, entry_vec.view());

        // Greedy search from top level to level 1
        for l in (1..=self.metadata.max_level).rev() {
            loop {
                let node = self.nodes[curr_node].read();
                if l >= node.neighbors.len() {
                    break;
                }

                let mut changed = false;
                for &neighbor in &node.neighbors[l] {
                    if let Some(neighbor_vec) = self.get_vector_from_mmap(neighbor) {
                        let dist = Self::inner_product(query, neighbor_vec.view());
                        if dist > curr_dist {
                            curr_dist = dist;
                            curr_node = neighbor;
                            changed = true;
                        }
                    }
                }

                if !changed {
                    break;
                }
            }
        }

        // Search at level 0 with custom ef_search
        let ef = ef_search.max(k);
        self.search_layer_query_filtered(query, curr_node, ef, k, None)
    }

    /// Search layer during query time with optional filter (uses mmap).
    ///
    /// The filter only affects which results are returned - all vectors are still
    /// explored (added to candidates) for proper graph traversal. This ensures
    /// the HNSW search can navigate through the graph effectively even when
    /// most vectors are filtered out.
    fn search_layer_query_filtered(
        &self,
        query: ArrayView1<f32>,
        entry_point: usize,
        ef: usize,
        k: usize,
        filter: Option<&HashSet<usize>>,
    ) -> Vec<(f32, usize)> {
        let entry_vec = match self.get_vector_from_mmap(entry_point) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let entry_dist = Self::inner_product(query, entry_vec.view());

        let num_nodes = self.nodes.len();
        let mut visited = vec![false; num_nodes];
        visited[entry_point] = true;

        // candidates: max-heap by distance (for graph exploration - includes ALL vectors)
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        candidates.push((OrderedFloat(entry_dist), entry_point));

        // filtered_results: only vectors that pass the filter
        let mut filtered_results: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> =
            BinaryHeap::new();

        // Check if entry point passes filter
        let entry_passes_filter = filter.is_none_or(|f| f.contains(&entry_point));
        if entry_passes_filter {
            filtered_results.push(Reverse((OrderedFloat(entry_dist), entry_point)));
        }

        // Track worst candidate distance for exploration cutoff
        let mut worst_candidate_dist = entry_dist;

        while let Some((OrderedFloat(cand_dist), cand_id)) = candidates.pop() {
            // Stop if candidate is worse than the worst we've seen and we've explored enough
            // We use a more lenient stopping condition to ensure thorough exploration
            if cand_dist < worst_candidate_dist * 0.5 && filtered_results.len() >= k {
                break;
            }

            let node = self.nodes[cand_id].read();
            if node.neighbors.is_empty() {
                continue;
            }

            for &neighbor in &node.neighbors[0] {
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;

                if let Some(neighbor_vec) = self.get_vector_from_mmap(neighbor) {
                    let dist = Self::inner_product(query, neighbor_vec.view());

                    // Always add to candidates for graph exploration
                    candidates.push((OrderedFloat(dist), neighbor));

                    // Update worst candidate distance
                    if dist > worst_candidate_dist {
                        worst_candidate_dist = dist;
                    }

                    // Only add to results if it passes the filter
                    let passes_filter = filter.is_none_or(|f| f.contains(&neighbor));
                    if passes_filter {
                        let worst_filtered = filtered_results
                            .peek()
                            .map(|Reverse((d, _))| d.0)
                            .unwrap_or(f32::NEG_INFINITY);

                        if filtered_results.len() < ef || dist > worst_filtered {
                            filtered_results.push(Reverse((OrderedFloat(dist), neighbor)));

                            // Keep only ef best filtered results
                            while filtered_results.len() > ef {
                                filtered_results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted vector (highest distance first)
        let mut result_vec: Vec<(f32, usize)> = filtered_results
            .into_iter()
            .map(|Reverse((d, id))| (d.0, id))
            .collect();
        result_vec.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        result_vec.truncate(k);
        result_vec
    }

    /// Save the graph structure to disk.
    fn save_graph(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write number of nodes
        writer.write_u64::<LittleEndian>(self.nodes.len() as u64)?;

        // Write each node
        for node in &self.nodes {
            let node = node.read();
            // Write number of levels
            writer.write_u32::<LittleEndian>(node.neighbors.len() as u32)?;

            for neighbors in &node.neighbors {
                // Write number of neighbors at this level
                writer.write_u32::<LittleEndian>(neighbors.len() as u32)?;
                for &neighbor in neighbors {
                    writer.write_u64::<LittleEndian>(neighbor as u64)?;
                }
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Load the graph structure from disk.
    fn load_graph(path: &Path) -> Result<Vec<RwLock<Node>>> {
        let file = File::open(path)
            .map_err(|_| Error::CorruptedIndex("hnsw_graph.bin not found".to_string()))?;
        let mut reader = BufReader::new(file);

        let num_nodes = reader.read_u64::<LittleEndian>()? as usize;
        let mut nodes = Vec::with_capacity(num_nodes);

        for _ in 0..num_nodes {
            let num_levels = reader.read_u32::<LittleEndian>()? as usize;
            let mut neighbors = Vec::with_capacity(num_levels);

            for _ in 0..num_levels {
                let num_neighbors = reader.read_u32::<LittleEndian>()? as usize;
                let mut level_neighbors = Vec::with_capacity(num_neighbors);

                for _ in 0..num_neighbors {
                    let neighbor = reader.read_u64::<LittleEndian>()? as usize;
                    level_neighbors.push(neighbor);
                }

                neighbors.push(level_neighbors);
            }

            nodes.push(RwLock::new(Node { neighbors }));
        }

        Ok(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use tempfile::tempdir;

    #[test]
    fn test_hnsw_basic() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        let mut index = HnswIndex::new(dir.path(), 128, config).unwrap();

        // Create some random vectors
        let vectors = Array2::from_shape_fn((100, 128), |(i, j)| ((i * 128 + j) as f32).sin());

        // Add vectors
        let start_idx = index.update(&vectors).unwrap();
        assert_eq!(start_idx, 0);
        assert_eq!(index.len(), 100);

        // Search
        let queries = vectors.slice(ndarray::s![0..5, ..]).to_owned();
        let (scores, indices) = index.search(&queries, 10).unwrap();

        assert_eq!(scores.dim(), (5, 10));
        assert_eq!(indices.dim(), (5, 10));

        // First result should be the query itself (or very close)
        for i in 0..5 {
            assert!(indices[[i, 0]] >= 0);
        }
    }

    #[test]
    fn test_hnsw_persistence() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        // Create and populate index
        {
            let mut index = HnswIndex::new(dir.path(), 64, config.clone()).unwrap();
            let vectors = Array2::from_shape_fn((50, 64), |(i, j)| ((i * 64 + j) as f32).cos());
            index.update(&vectors).unwrap();
        }

        // Load and verify
        let index = HnswIndex::load(dir.path()).unwrap();
        assert_eq!(index.len(), 50);
        assert_eq!(index.dim(), 64);

        // Search should work
        let queries = Array2::from_shape_fn((3, 64), |(i, j)| ((i * 64 + j) as f32).cos());
        let (_scores, indices) = index.search(&queries, 5).unwrap();
        assert_eq!(indices.dim(), (3, 5));
    }

    #[test]
    fn test_hnsw_incremental_update() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        let mut index = HnswIndex::new(dir.path(), 32, config).unwrap();

        // First batch
        let vectors1 = Array2::from_shape_fn((20, 32), |(i, j)| (i + j) as f32);
        let idx1 = index.update(&vectors1).unwrap();
        assert_eq!(idx1, 0);
        assert_eq!(index.len(), 20);

        // Second batch
        let vectors2 = Array2::from_shape_fn((30, 32), |(i, j)| (i + j + 100) as f32);
        let idx2 = index.update(&vectors2).unwrap();
        assert_eq!(idx2, 20);
        assert_eq!(index.len(), 50);
    }

    #[test]
    fn test_hnsw_filtered_search() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        let mut index = HnswIndex::new(dir.path(), 64, config).unwrap();

        // Create vectors where vectors 0-49 are similar to each other (group A)
        // and vectors 50-99 are similar to each other (group B)
        let vectors = Array2::from_shape_fn((100, 64), |(i, j)| {
            if i < 50 {
                ((i * 64 + j) as f32).sin()
            } else {
                ((i * 64 + j) as f32).cos()
            }
        });

        index.update(&vectors).unwrap();

        // Query using vector 0 (from group A)
        let query = vectors.slice(ndarray::s![0..1, ..]).to_owned();

        // Search without filter - should return results from both groups
        let (_scores_all, indices_all) = index.search(&query, 10).unwrap();

        // Create filter for only even-indexed vectors
        let filter: HashSet<usize> = (0..100).filter(|x| x % 2 == 0).collect();
        let (scores_filtered, indices_filtered) =
            index.search_with_filter(&query, 10, Some(&filter)).unwrap();

        // All filtered results should be even numbers
        for j in 0..10 {
            let idx = indices_filtered[[0, j]];
            if idx >= 0 {
                assert!(
                    idx % 2 == 0,
                    "Filtered result {} should be even, got {}",
                    j,
                    idx
                );
            }
        }

        // Verify we got results
        assert!(
            indices_filtered[[0, 0]] >= 0,
            "Should have at least one result"
        );
        assert!(scores_filtered[[0, 0]] > f32::NEG_INFINITY);

        // Search with empty filter should return no results (all -1)
        let empty_filter: HashSet<usize> = HashSet::new();
        let (_scores_empty, indices_empty) = index
            .search_with_filter(&query, 10, Some(&empty_filter))
            .unwrap();

        for j in 0..10 {
            assert_eq!(
                indices_empty[[0, j]],
                -1,
                "Empty filter should return -1 for all indices"
            );
        }

        // Search with None filter should behave like regular search
        let (_scores_none, indices_none) = index.search_with_filter(&query, 10, None).unwrap();
        assert_eq!(indices_none.dim(), indices_all.dim());
        // Results should be identical
        for j in 0..10 {
            assert_eq!(indices_none[[0, j]], indices_all[[0, j]]);
        }
    }

    #[test]
    fn test_hnsw_filtered_search_subset() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        let mut index = HnswIndex::new(dir.path(), 32, config).unwrap();

        // Create 50 vectors
        let vectors = Array2::from_shape_fn((50, 32), |(i, j)| ((i * 32 + j) as f32).sin());
        index.update(&vectors).unwrap();

        // Query with vector 10
        let query = vectors.slice(ndarray::s![10..11, ..]).to_owned();

        // Filter to only allow vectors 5, 10, 15, 20, 25
        let filter: HashSet<usize> = [5, 10, 15, 20, 25].into_iter().collect();
        let (_scores, indices) = index.search_with_filter(&query, 5, Some(&filter)).unwrap();

        // The top result should be 10 (the query itself)
        assert_eq!(indices[[0, 0]], 10, "Query vector should be top result");

        // All results should be from the filter set
        for j in 0..5 {
            let idx = indices[[0, j]];
            if idx >= 0 {
                assert!(
                    filter.contains(&(idx as usize)),
                    "Result {} with idx {} should be in filter set",
                    j,
                    idx
                );
            }
        }
    }

    #[test]
    fn test_hnsw_search_with_ids() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        let mut index = HnswIndex::new(dir.path(), 32, config).unwrap();

        // Create 100 vectors
        let vectors = Array2::from_shape_fn((100, 32), |(i, j)| ((i * 32 + j) as f32).sin());
        index.update(&vectors).unwrap();

        // Create 3 queries using vectors 10, 50, and 90
        let mut queries = Array2::zeros((3, 32));
        queries.row_mut(0).assign(&vectors.row(10));
        queries.row_mut(1).assign(&vectors.row(50));
        queries.row_mut(2).assign(&vectors.row(90));

        // Create different candidate lists for each query (different sizes!)
        let candidates1: Vec<usize> = (0..30).collect(); // Query 0: 30 candidates (vectors 0-29)
        let candidates2: Vec<usize> = (40..70).collect(); // Query 1: 30 candidates (vectors 40-69)
        let candidates3: Vec<usize> = (80..100).collect(); // Query 2: 20 candidates (vectors 80-99)

        let candidate_refs: Vec<&[usize]> = vec![&candidates1, &candidates2, &candidates3];

        let (_scores, indices) = index.search_with_ids(&queries, 5, &candidate_refs).unwrap();

        // Query 0 (vector 10) should return results from candidates1 (0-29)
        // Top result should be 10 itself
        assert_eq!(indices[[0, 0]], 10, "Query 0 top result should be 10");
        for j in 0..5 {
            let idx = indices[[0, j]];
            if idx >= 0 {
                assert!(
                    candidates1.contains(&(idx as usize)),
                    "Query 0 result {} should be in range 0-29, got {}",
                    j,
                    idx
                );
            }
        }

        // Query 1 (vector 50) should return results from candidates2 (40-69)
        // Top result should be 50 itself
        assert_eq!(indices[[1, 0]], 50, "Query 1 top result should be 50");
        for j in 0..5 {
            let idx = indices[[1, j]];
            if idx >= 0 {
                assert!(
                    candidates2.contains(&(idx as usize)),
                    "Query 1 result {} should be in range 40-69, got {}",
                    j,
                    idx
                );
            }
        }

        // Query 2 (vector 90) should return results from candidates3 (80-99)
        // Top result should be 90 itself
        assert_eq!(indices[[2, 0]], 90, "Query 2 top result should be 90");
        for j in 0..5 {
            let idx = indices[[2, j]];
            if idx >= 0 {
                assert!(
                    candidates3.contains(&(idx as usize)),
                    "Query 2 result {} should be in range 80-99, got {}",
                    j,
                    idx
                );
            }
        }
    }

    #[test]
    fn test_hnsw_search_with_ids_different_sizes() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        let mut index = HnswIndex::new(dir.path(), 32, config).unwrap();

        // Create 100 vectors
        let vectors = Array2::from_shape_fn((100, 32), |(i, j)| ((i * 32 + j) as f32).sin());
        index.update(&vectors).unwrap();

        // Create 3 queries
        let mut queries = Array2::zeros((3, 32));
        queries.row_mut(0).assign(&vectors.row(5));
        queries.row_mut(1).assign(&vectors.row(50));
        queries.row_mut(2).assign(&vectors.row(75));

        // Each query has a DIFFERENT number of candidates
        let candidates1: Vec<usize> = vec![5, 10, 15]; // Only 3 candidates
        let candidates2: Vec<usize> = (40..60).collect(); // 20 candidates
        let candidates3: Vec<usize> = vec![70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]; // 11 candidates

        let candidate_refs: Vec<&[usize]> = vec![&candidates1, &candidates2, &candidates3];

        let (_scores, indices) = index.search_with_ids(&queries, 5, &candidate_refs).unwrap();

        // Query 0 has only 3 candidates, so only 3 valid results
        assert_eq!(indices[[0, 0]], 5, "Query 0 top result should be 5");
        let valid_count_q0 = (0..5).filter(|&j| indices[[0, j]] >= 0).count();
        assert_eq!(
            valid_count_q0, 3,
            "Query 0 should have exactly 3 valid results"
        );

        // Query 1 top result should be 50
        assert_eq!(indices[[1, 0]], 50, "Query 1 top result should be 50");

        // Query 2 top result should be 75
        assert_eq!(indices[[2, 0]], 75, "Query 2 top result should be 75");
    }

    #[test]
    fn test_hnsw_search_with_ids_empty_candidates() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        let mut index = HnswIndex::new(dir.path(), 32, config).unwrap();

        let vectors = Array2::from_shape_fn((50, 32), |(i, j)| ((i * 32 + j) as f32).sin());
        index.update(&vectors).unwrap();

        // Create 2 queries
        let mut queries = Array2::zeros((2, 32));
        queries.row_mut(0).assign(&vectors.row(10));
        queries.row_mut(1).assign(&vectors.row(20));

        // First query has candidates, second has empty list
        let candidates1: Vec<usize> = vec![5, 10, 15, 20];
        let candidates2: Vec<usize> = vec![]; // Empty!

        let candidate_refs: Vec<&[usize]> = vec![&candidates1, &candidates2];

        let (_scores, indices) = index.search_with_ids(&queries, 5, &candidate_refs).unwrap();

        // Query 0 should have valid results
        assert_eq!(indices[[0, 0]], 10, "Query 0 top result should be 10");

        // Query 1 has no candidates, so all results should be -1
        for j in 0..5 {
            assert_eq!(
                indices[[1, j]],
                -1,
                "Query 1 result {} should be -1 (no candidates)",
                j
            );
        }
    }

    #[test]
    fn test_hnsw_search_with_ids_length_mismatch() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        let mut index = HnswIndex::new(dir.path(), 32, config).unwrap();

        let vectors = Array2::from_shape_fn((50, 32), |(i, j)| ((i * 32 + j) as f32).sin());
        index.update(&vectors).unwrap();

        // Create 3 queries but only 2 candidate lists
        let queries = vectors.slice(ndarray::s![0..3, ..]).to_owned();
        let candidates1: Vec<usize> = vec![0, 1, 2];
        let candidates2: Vec<usize> = vec![10, 11, 12];
        let candidate_refs: Vec<&[usize]> = vec![&candidates1, &candidates2];

        // Should return an error
        let result = index.search_with_ids(&queries, 5, &candidate_refs);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_vectors_range() {
        let dir = tempdir().unwrap();
        let config = HnswConfig::default();

        let mut index = HnswIndex::new(dir.path(), 32, config).unwrap();

        // Create 100 vectors with predictable values
        let vectors = Array2::from_shape_fn((100, 32), |(i, j)| (i * 32 + j) as f32);
        index.update(&vectors).unwrap();

        // Test getting a range of vectors
        let range = index.get_vectors_range(10, 20).unwrap();
        assert_eq!(range.dim(), (10, 32), "Should have 10 vectors");

        // Verify content matches original vectors
        for i in 0..10 {
            for j in 0..32 {
                assert_eq!(
                    range[[i, j]],
                    vectors[[10 + i, j]],
                    "Vector content should match at [{}, {}]",
                    i,
                    j
                );
            }
        }

        // Test edge cases
        let empty = index.get_vectors_range(50, 50).unwrap();
        assert_eq!(empty.dim(), (0, 32), "Empty range should return 0 vectors");

        let clamped = index.get_vectors_range(90, 200).unwrap();
        assert_eq!(clamped.dim(), (10, 32), "Should clamp to num_vectors");

        let out_of_bounds = index.get_vectors_range(100, 150).unwrap();
        assert_eq!(
            out_of_bounds.dim(),
            (0, 32),
            "Start beyond range returns empty"
        );
    }
}
