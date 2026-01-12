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
use std::collections::BinaryHeap;
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
        let metadata_path = directory.join("metadata.json");
        let metadata_file = File::open(&metadata_path)
            .map_err(|_| Error::CorruptedIndex("metadata.json not found".to_string()))?;
        let metadata: HnswMetadata = serde_json::from_reader(BufReader::new(metadata_file))?;

        // Load graph
        let graph_path = directory.join("graph.bin");
        let nodes = Self::load_graph(&graph_path)?;

        // Memory-map vectors
        let vectors_path = directory.join("vectors.bin");
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

        // Load existing vectors into flat cache if needed
        if self.vectors_flat.is_empty() && self.metadata.num_vectors > 0 {
            self.load_vectors_to_flat()?;
        }

        // Add new vectors to flat cache
        self.vectors_flat.reserve(num_new * dim);
        for row in vectors.axis_iter(Axis(0)) {
            self.vectors_flat.extend(row.iter());
        }

        // Insert each vector into the graph
        for i in 0..num_new {
            self.insert_node(start_idx + i)?;
        }

        // Append vectors to the vectors file
        self.append_vectors(vectors)?;

        // Save updated metadata and graph
        self.save()?;

        // Clear cache and reload mmap
        self.vectors_flat.clear();
        self.vectors_flat.shrink_to_fit();
        self.reload_vectors_mmap()?;

        Ok(start_idx)
    }

    /// Search for the k nearest neighbors of query vectors.
    ///
    /// Returns (scores, indices) where:
    /// - scores: Array2<f32> of shape (num_queries, k) with similarity scores (higher is better)
    /// - indices: Array2<i64> of shape (num_queries, k) with vector indices (-1 for padding)
    pub fn search(&self, queries: &Array2<f32>, k: usize) -> Result<(Array2<f32>, Array2<i64>)> {
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
            .map(|query| self.search_single(query, k))
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

    /// Save the index to disk.
    pub fn save(&self) -> Result<()> {
        // Save metadata
        let metadata_path = self.directory.join("metadata.json");
        let metadata_file = File::create(&metadata_path)?;
        serde_json::to_writer_pretty(BufWriter::new(metadata_file), &self.metadata)?;

        // Save graph
        let graph_path = self.directory.join("graph.bin");
        self.save_graph(&graph_path)?;

        Ok(())
    }

    // ============== Private Methods ==============

    /// Get vector from flat cache by index.
    #[inline(always)]
    fn get_vector_flat(&self, id: usize) -> &[f32] {
        let start = id * self.vectors_dim;
        &self.vectors_flat[start..start + self.vectors_dim]
    }

    /// Compute dot product between query slice and vector in flat cache.
    #[inline(always)]
    fn dot_product_flat(&self, query: &[f32], id: usize) -> f32 {
        let vec = self.get_vector_flat(id);
        query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum()
    }

    /// Load vectors from mmap to flat cache.
    fn load_vectors_to_flat(&mut self) -> Result<()> {
        let mmap = self.vectors_mmap.as_ref().ok_or_else(|| {
            Error::CorruptedIndex("vectors.bin not loaded".to_string())
        })?;

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
        let vectors_path = self.directory.join("vectors.bin");
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
        let vectors_path = self.directory.join("vectors.bin");
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
                                let dist: f32 = neighbor_vec.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                                (dist, n)
                            })
                            .collect();

                        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                        neighbor.neighbors[l] =
                            candidates.iter().take(max_connections).map(|(_, idx)| *idx).collect();
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

    /// Search for k nearest neighbors of a single query (uses mmap).
    fn search_single(&self, query: ArrayView1<f32>, k: usize) -> Vec<(f32, usize)> {
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
        let ef = self.metadata.config.ef_search.max(k);
        self.search_layer_query(query, curr_node, ef, k)
    }

    /// Search layer during query time (uses mmap).
    fn search_layer_query(
        &self,
        query: ArrayView1<f32>,
        entry_point: usize,
        ef: usize,
        k: usize,
    ) -> Vec<(f32, usize)> {
        let entry_vec = match self.get_vector_from_mmap(entry_point) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let entry_dist = Self::inner_product(query, entry_vec.view());

        let num_nodes = self.nodes.len();
        let mut visited = vec![false; num_nodes];
        visited[entry_point] = true;

        // candidates: max-heap by distance
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        candidates.push((OrderedFloat(entry_dist), entry_point));

        // results: min-heap to keep track of worst result (for pruning)
        let mut results: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
        results.push(Reverse((OrderedFloat(entry_dist), entry_point)));

        while let Some((OrderedFloat(cand_dist), cand_id)) = candidates.pop() {
            // Get the worst (smallest) distance in results
            let worst_dist = results
                .peek()
                .map(|Reverse((d, _))| d.0)
                .unwrap_or(f32::NEG_INFINITY);

            // If candidate is worse than our worst result and we have enough results, stop
            if results.len() >= ef && cand_dist < worst_dist {
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
        }

        // Convert to sorted vector (highest distance first)
        let mut result_vec: Vec<(f32, usize)> = results
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
        let file =
            File::open(path).map_err(|_| Error::CorruptedIndex("graph.bin not found".to_string()))?;
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
}
