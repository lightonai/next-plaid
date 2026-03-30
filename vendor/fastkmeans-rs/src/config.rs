/// Configuration for the FastKMeans algorithm
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters
    pub k: usize,

    /// Maximum number of iterations
    pub max_iters: usize,

    /// Convergence tolerance. When centroid shift is below this threshold,
    /// the algorithm stops early. Set to negative value to disable early stopping.
    pub tol: f64,

    /// Random seed for centroid initialization and subsampling
    pub seed: u64,

    /// Maximum points per centroid for subsampling.
    /// If n_samples > k * max_points_per_centroid, data will be subsampled.
    /// Set to None to disable subsampling.
    pub max_points_per_centroid: Option<usize>,

    /// Chunk size for data processing. Larger values use more memory but may be faster.
    pub chunk_size_data: usize,

    /// Chunk size for centroid processing. Larger values use more memory but may be faster.
    pub chunk_size_centroids: usize,

    /// Print verbose output during training
    pub verbose: bool,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 8,
            max_iters: 25,
            tol: 1e-8,
            seed: 0,
            max_points_per_centroid: Some(256),
            chunk_size_data: 51_200,
            chunk_size_centroids: 10_240,
            verbose: false,
        }
    }
}

impl KMeansConfig {
    /// Create a new configuration with the specified number of clusters
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ..Default::default()
        }
    }

    /// Set the maximum number of iterations
    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Set the convergence tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the maximum points per centroid (for subsampling)
    pub fn with_max_points_per_centroid(mut self, max_ppc: Option<usize>) -> Self {
        self.max_points_per_centroid = max_ppc;
        self
    }

    /// Set verbose mode
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the data chunk size
    pub fn with_chunk_size_data(mut self, chunk_size: usize) -> Self {
        self.chunk_size_data = chunk_size;
        self
    }

    /// Set the centroid chunk size
    pub fn with_chunk_size_centroids(mut self, chunk_size: usize) -> Self {
        self.chunk_size_centroids = chunk_size;
        self
    }
}
