//! SciFact benchmark for measuring index creation and search times.
//!
//! This example loads pre-dumped SciFact embeddings and measures:
//! - Index creation time (from embeddings + centroids)
//! - Search time (batch queries)
//!
//! Usage:
//!     cargo run --release --example scifact_benchmark --features npy,accelerate -- \
//!         --data-dir ./scifact_benchmark_data
//!
//! First, run the Python script to dump the data:
//!     cd benchmarks && uv run python dump_scifact.py

#[cfg(feature = "npy")]
use ndarray::{s, Array1, Array2, Axis};
#[cfg(feature = "npy")]
use ndarray_npy::ReadNpyExt;
#[cfg(feature = "npy")]
use next_plaid::{Index, IndexConfig, MmapIndex, ResidualCodec, SearchParameters};
#[cfg(feature = "npy")]
use std::fs::{self, File};
#[cfg(feature = "npy")]
use std::path::{Path, PathBuf};
#[cfg(feature = "npy")]
use std::time::Instant;

fn main() {
    #[cfg(not(feature = "npy"))]
    {
        eprintln!("This example requires the 'npy' feature");
        std::process::exit(1);
    }

    #[cfg(feature = "npy")]
    {
        if let Err(e) = run_benchmark() {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "npy")]
fn run_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut data_dir: Option<PathBuf> = None;
    let mut nbits: usize = 4;
    let mut top_k: usize = 100;
    let mut n_ivf_probe: usize = 8;
    let mut n_full_scores: usize = 8192;
    let mut use_mmap: bool = false;
    let mut profile: bool = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                i += 1;
                data_dir = Some(PathBuf::from(&args[i]));
            }
            "--nbits" => {
                i += 1;
                nbits = args[i].parse()?;
            }
            "--top-k" => {
                i += 1;
                top_k = args[i].parse()?;
            }
            "--n-ivf-probe" => {
                i += 1;
                n_ivf_probe = args[i].parse()?;
            }
            "--n-full-scores" => {
                i += 1;
                n_full_scores = args[i].parse()?;
            }
            "--mmap" => {
                use_mmap = true;
            }
            "--profile" => {
                profile = true;
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let data_dir = data_dir.ok_or("--data-dir is required")?;

    println!("======================================================================");
    println!("  SciFact Benchmark");
    println!("======================================================================");
    println!();
    println!("Configuration:");
    println!("  Data dir:      {}", data_dir.display());
    println!("  nbits:         {}", nbits);
    println!("  top_k:         {}", top_k);
    println!("  n_ivf_probe:   {}", n_ivf_probe);
    println!("  n_full_scores: {}", n_full_scores);
    println!("  use_mmap:      {}", use_mmap);
    println!();

    // Load embeddings
    println!("[1/4] Loading document embeddings...");
    let docs_dir = data_dir.join("documents");
    let embeddings = load_embeddings(&docs_dir)?;
    let num_docs = embeddings.len();
    let total_tokens: usize = embeddings.iter().map(|e| e.nrows()).sum();
    let avg_tokens = total_tokens as f64 / num_docs as f64;
    println!(
        "  Loaded {} documents ({} total tokens, {:.1} avg)",
        num_docs, total_tokens, avg_tokens
    );

    // Load centroids
    println!("[2/4] Loading centroids...");
    let centroids = load_centroids(&docs_dir)?;
    println!("  Centroids shape: {:?}", centroids.shape());

    // Load queries
    println!("[3/4] Loading queries...");
    let queries_dir = data_dir.join("queries");
    let queries = load_queries(&queries_dir)?;
    println!("  Loaded {} queries", queries.len());

    // Create index with timing
    println!("[4/4] Creating index...");
    let index_dir = tempfile::tempdir()?;
    let index_path = index_dir.path().join("index");

    let config = IndexConfig {
        nbits,
        batch_size: 50000,
        seed: Some(42),
        ..Default::default()
    };

    let (index, index_time) = if profile {
        run_profiled_index_creation(&embeddings, centroids, &index_path, &config)?
    } else {
        let start = Instant::now();
        let index = Index::create(
            &embeddings,
            centroids,
            index_path.to_str().unwrap(),
            &config,
        )?;
        (index, start.elapsed())
    };

    println!();
    println!("======================================================================");
    println!("  Results");
    println!("======================================================================");
    println!();
    println!("Index Creation:");
    println!("  Time:       {:.3}s", index_time.as_secs_f64());
    println!("  Documents:  {}", index.metadata.num_documents);
    println!("  Centroids:  {}", index.metadata.num_partitions);

    // Search with timing
    let params = SearchParameters {
        batch_size: 2000,
        n_full_scores,
        top_k,
        n_ivf_probe,
        ..Default::default()
    };

    let start = Instant::now();
    let results = if use_mmap {
        drop(index); // Free memory before mmap
        let mmap_index = MmapIndex::load(index_path.to_str().unwrap())?;
        mmap_index.search_batch(&queries, &params, true, None)?
    } else {
        index.search_batch(&queries, &params, true, None)?
    };
    let search_time = start.elapsed();

    println!();
    println!("Search:");
    println!("  Time:       {:.3}s", search_time.as_secs_f64());
    println!("  Queries:    {}", queries.len());
    println!(
        "  QPS:        {:.1}",
        queries.len() as f64 / search_time.as_secs_f64()
    );
    println!(
        "  Avg results: {:.1}",
        results.iter().map(|r| r.passage_ids.len()).sum::<usize>() as f64 / results.len() as f64
    );

    println!();
    println!("======================================================================");
    println!("  Summary");
    println!("======================================================================");
    println!();
    println!("  Index time:  {:.3}s", index_time.as_secs_f64());
    println!("  Search time: {:.3}s", search_time.as_secs_f64());
    println!(
        "  Total time:  {:.3}s",
        (index_time + search_time).as_secs_f64()
    );
    println!();

    Ok(())
}

#[cfg(feature = "npy")]
fn print_usage() {
    eprintln!(
        r#"SciFact Benchmark

Usage:
    scifact_benchmark --data-dir <path> [options]

Options:
    --data-dir <path>     Directory containing documents/ and queries/ subdirs (required)
    --nbits <n>           Quantization bits (default: 4)
    --top-k <n>           Results to return per query (default: 100)
    --n-ivf-probe <n>     IVF cells to probe (default: 8)
    --n-full-scores <n>   Candidates for exact scoring (default: 8192)
    --mmap                Use memory-mapped index for search
    --profile             Profile index creation breakdown

First, dump the data using Python:
    cd benchmarks && uv run python dump_scifact.py
"#
    );
}

#[cfg(feature = "npy")]
fn run_profiled_index_creation(
    embeddings: &[Array2<f32>],
    centroids: Array2<f32>,
    index_path: &Path,
    config: &IndexConfig,
) -> Result<(Index, std::time::Duration), Box<dyn std::error::Error>> {
    let overall_start = Instant::now();

    fs::create_dir_all(index_path)?;

    let num_documents = embeddings.len();
    let embedding_dim = centroids.ncols();
    let num_centroids = centroids.nrows();
    let total_embeddings: usize = embeddings.iter().map(|e| e.nrows()).sum();

    println!();
    println!("  Profiling index creation...");
    println!(
        "  Documents: {}, Tokens: {}, Centroids: {}",
        num_documents, total_embeddings, num_centroids
    );
    println!();

    // Step 1: Create HNSW index for centroids
    let start = Instant::now();
    let avg_residual = Array1::zeros(embedding_dim);
    let initial_codec = ResidualCodec::new(
        config.nbits,
        centroids.clone(),
        index_path,
        avg_residual.clone(),
        None,
        None,
    )?;
    let hnsw_time = start.elapsed();
    println!(
        "  [1] HNSW index creation:     {:>8.3}s",
        hnsw_time.as_secs_f64()
    );

    // Step 2: Sample and compute codes for heldout (codec training)
    let start = Instant::now();
    let sample_count = ((16.0 * (120.0 * num_documents as f64).sqrt()) as usize)
        .min(num_documents)
        .max(1);
    let heldout_size = (0.05 * total_embeddings as f64).min(50000.0) as usize;

    // Collect heldout embeddings
    let mut heldout_embeddings: Vec<f32> = Vec::with_capacity(heldout_size * embedding_dim);
    let mut collected = 0;
    for emb in embeddings.iter().rev().take(sample_count) {
        if collected >= heldout_size {
            break;
        }
        let take = (heldout_size - collected).min(emb.nrows());
        for row in emb.axis_iter(Axis(0)).take(take) {
            heldout_embeddings.extend(row.iter());
        }
        collected += take;
    }
    let heldout = Array2::from_shape_vec((collected, embedding_dim), heldout_embeddings)?;
    let sample_time = start.elapsed();
    println!(
        "  [2] Sample collection:       {:>8.3}s  ({} samples)",
        sample_time.as_secs_f64(),
        collected
    );

    // Step 3: Compress heldout into codes (HNSW search)
    let start = Instant::now();
    let heldout_codes = initial_codec.compress_into_codes(&heldout);
    let heldout_compress_time = start.elapsed();
    println!(
        "  [3] Heldout HNSW search:     {:>8.3}s  ({} searches, {:.0}/s)",
        heldout_compress_time.as_secs_f64(),
        collected,
        collected as f64 / heldout_compress_time.as_secs_f64()
    );

    // Step 4: Compute residuals and quantization params
    let start = Instant::now();
    let mut residuals = heldout.clone();
    for i in 0..heldout.nrows() {
        let centroid = initial_codec.centroids.row(heldout_codes[i]);
        for j in 0..embedding_dim {
            residuals[[i, j]] -= centroid[j];
        }
    }

    let n_options = 1 << config.nbits;
    let quantile_values: Vec<f64> = (1..n_options)
        .map(|i| i as f64 / n_options as f64)
        .collect();
    let weight_quantile_values: Vec<f64> = (0..n_options)
        .map(|i| (i as f64 + 0.5) / n_options as f64)
        .collect();
    let flat_residuals: Array1<f32> = residuals.iter().copied().collect();
    let bucket_cutoffs = Array1::from_vec(next_plaid::utils::quantiles(
        &flat_residuals,
        &quantile_values,
    ));
    let bucket_weights = Array1::from_vec(next_plaid::utils::quantiles(
        &flat_residuals,
        &weight_quantile_values,
    ));
    let quantile_time = start.elapsed();
    println!(
        "  [4] Quantization params:     {:>8.3}s",
        quantile_time.as_secs_f64()
    );

    // Step 5: Process all documents - compress into codes
    let start = Instant::now();
    let mut all_codes: Vec<usize> = Vec::with_capacity(total_embeddings);

    // Concatenate all embeddings for batch processing
    let mut batch_embeddings = Array2::<f32>::zeros((total_embeddings, embedding_dim));
    let mut offset = 0;
    for doc in embeddings {
        let n = doc.nrows();
        batch_embeddings
            .slice_mut(s![offset..offset + n, ..])
            .assign(doc);
        offset += n;
    }
    let concat_time = start.elapsed();

    let start = Instant::now();
    let batch_codes = initial_codec.compress_into_codes(&batch_embeddings);
    all_codes.extend(batch_codes.iter().copied());
    let compress_time = start.elapsed();
    println!(
        "  [5] All tokens HNSW search:  {:>8.3}s  ({} searches, {:.0}/s)",
        compress_time.as_secs_f64(),
        total_embeddings,
        total_embeddings as f64 / compress_time.as_secs_f64()
    );

    // Step 6: Compute residuals for all embeddings
    let start = Instant::now();
    {
        use rayon::prelude::*;
        let centroids_store = &initial_codec.centroids;
        batch_embeddings
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(batch_codes.as_slice().unwrap().par_iter())
            .for_each(|(mut row, &code)| {
                let centroid = centroids_store.row(code);
                row.iter_mut()
                    .zip(centroid.iter())
                    .for_each(|(r, c)| *r -= c);
            });
    }
    let residual_time = start.elapsed();
    println!(
        "  [6] Residual computation:    {:>8.3}s",
        residual_time.as_secs_f64()
    );

    // Step 7: Quantize residuals
    let start = Instant::now();
    let final_codec = ResidualCodec::new_with_store(
        config.nbits,
        initial_codec.centroids.clone(),
        avg_residual,
        Some(bucket_cutoffs),
        Some(bucket_weights),
    )?;
    let _batch_packed = final_codec.quantize_residuals(&batch_embeddings)?;
    let quantize_time = start.elapsed();
    println!(
        "  [7] Residual quantization:   {:>8.3}s",
        quantize_time.as_secs_f64()
    );

    let total_time = overall_start.elapsed();
    println!();
    println!(
        "  Total profiled time:         {:>8.3}s",
        total_time.as_secs_f64()
    );
    println!(
        "  (concat embeddings took:     {:>8.3}s)",
        concat_time.as_secs_f64()
    );

    // Now create the actual index using the normal path
    println!();
    println!("  Creating full index...");
    let start = Instant::now();
    let index = Index::create(embeddings, centroids, index_path.to_str().unwrap(), config)?;
    let full_time = start.elapsed();

    Ok((index, full_time))
}

#[cfg(feature = "npy")]
fn load_embeddings(data_dir: &Path) -> Result<Vec<Array2<f32>>, Box<dyn std::error::Error>> {
    let mut embeddings = Vec::new();

    let mut doc_files: Vec<_> = fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("doc_") && n.ends_with(".npy"))
                .unwrap_or(false)
        })
        .collect();

    doc_files.sort_by_key(|e| e.path());

    for entry in doc_files {
        let path = entry.path();
        let file = File::open(&path)?;
        let arr: Array2<f32> = Array2::read_npy(file)?;
        embeddings.push(arr);
    }

    Ok(embeddings)
}

#[cfg(feature = "npy")]
fn load_centroids(data_dir: &Path) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let centroids_path = data_dir.join("centroids.npy");
    let file = File::open(&centroids_path)?;
    let centroids: Array2<f32> = Array2::read_npy(file)?;
    Ok(centroids)
}

#[cfg(feature = "npy")]
fn load_queries(query_dir: &Path) -> Result<Vec<Array2<f32>>, Box<dyn std::error::Error>> {
    let mut queries = Vec::new();

    let mut query_files: Vec<_> = fs::read_dir(query_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("query_") && n.ends_with(".npy"))
                .unwrap_or(false)
        })
        .collect();

    query_files.sort_by_key(|e| e.path());

    for entry in query_files {
        let path = entry.path();
        let file = File::open(&path)?;
        let arr: Array2<f32> = Array2::read_npy(file)?;
        queries.push(arr);
    }

    Ok(queries)
}
