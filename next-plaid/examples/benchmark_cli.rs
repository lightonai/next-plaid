//! CLI tool for benchmarking next-plaid.
//!
//! This example provides a command-line interface for creating, updating, and searching
//! indexes, primarily used by the Python benchmark script.
//!
//! Usage:
//!     benchmark_cli create --data-dir <path> --index-dir <path> [--nbits <n>]
//!     benchmark_cli update --index-dir <path> --data-dir <path>
//!     benchmark_cli search --index-dir <path> --query-dir <path> [--top-k <n>]

#[cfg(feature = "npy")]
use std::fs::{self, File};
#[cfg(feature = "npy")]
use std::path::{Path, PathBuf};

#[cfg(feature = "npy")]
use next_plaid::{Index, IndexConfig, SearchParameters, UpdateConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "create" => run_create(&args[2..]),
        "update" => run_update(&args[2..]),
        "search" => run_search(&args[2..]),
        "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!(
        r#"Usage:
    benchmark_cli create --data-dir <path> --index-dir <path> [--nbits <n>]
    benchmark_cli update --index-dir <path> --data-dir <path>
    benchmark_cli search --index-dir <path> --query-dir <path> [options]

Create Options:
    --data-dir <path>     Directory containing doc_*.npy files and centroids.npy
    --index-dir <path>    Directory to write the index
    --nbits <n>           Number of bits for quantization (default: 4)

Update Options:
    --index-dir <path>    Directory containing the existing index
    --data-dir <path>     Directory containing doc_*.npy files to add

Search Options:
    --index-dir <path>    Directory containing the index
    --query-dir <path>    Directory containing query_*.npy files
    --top-k <n>           Number of results to return (default: 10)
    --n-ivf-probe <n>     Number of IVF cells to probe (default: 8)
    --n-full-scores <n>   Number of candidates for exact scoring (default: 4096)
    --mmap                Use memory-mapped index for lower RAM usage
"#
    );
}

#[cfg(feature = "npy")]
fn run_create(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut data_dir: Option<PathBuf> = None;
    let mut index_dir: Option<PathBuf> = None;
    let mut nbits: usize = 4;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                i += 1;
                data_dir = Some(PathBuf::from(&args[i]));
            }
            "--index-dir" => {
                i += 1;
                index_dir = Some(PathBuf::from(&args[i]));
            }
            "--nbits" => {
                i += 1;
                nbits = args[i].parse()?;
            }
            _ => {
                return Err(format!("Unknown option: {}", args[i]).into());
            }
        }
        i += 1;
    }

    let data_dir = data_dir.ok_or("--data-dir is required")?;
    let index_dir = index_dir.ok_or("--index-dir is required")?;

    // Load embeddings
    let embeddings = load_embeddings(&data_dir)?;
    eprintln!("Loaded {} documents", embeddings.len());

    // Load centroids
    let centroids = load_centroids(&data_dir)?;
    eprintln!("Loaded centroids: {:?}", centroids.shape());

    // Create config
    let config = IndexConfig {
        nbits,
        batch_size: 50000,
        seed: Some(42),
        ..Default::default()
    };

    // Create index
    let _index = Index::create(&embeddings, centroids, index_dir.to_str().unwrap(), &config)?;
    eprintln!("Index created at {:?}", index_dir);

    Ok(())
}

#[cfg(not(feature = "npy"))]
fn run_create(_args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    Err("Index creation requires 'npy' feature".into())
}

#[cfg(feature = "npy")]
fn run_update(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut index_dir: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--index-dir" => {
                i += 1;
                index_dir = Some(PathBuf::from(&args[i]));
            }
            "--data-dir" => {
                i += 1;
                data_dir = Some(PathBuf::from(&args[i]));
            }
            _ => {
                return Err(format!("Unknown option: {}", args[i]).into());
            }
        }
        i += 1;
    }

    let index_dir = index_dir.ok_or("--index-dir is required")?;
    let data_dir = data_dir.ok_or("--data-dir is required")?;

    // Load existing index
    let mut index = Index::load(index_dir.to_str().unwrap())?;
    eprintln!(
        "Loaded index with {} documents",
        index.metadata.num_documents
    );

    // Load new embeddings
    let embeddings = load_embeddings(&data_dir)?;
    eprintln!("Loaded {} new documents", embeddings.len());

    // Create update config
    let config = UpdateConfig::default();

    // Update index
    index.update_simple(&embeddings, Some(config.batch_size))?;
    eprintln!(
        "Index updated, now has {} documents",
        index.metadata.num_documents
    );

    Ok(())
}

#[cfg(not(feature = "npy"))]
fn run_update(_args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    Err("Index update requires 'npy' feature".into())
}

#[cfg(feature = "npy")]
fn run_search(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use next_plaid::MmapIndex;

    let mut index_dir: Option<PathBuf> = None;
    let mut query_dir: Option<PathBuf> = None;
    let mut top_k: usize = 10;
    let mut n_ivf_probe: usize = 8;
    let mut n_full_scores: usize = 4096;
    let mut use_mmap: bool = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--index-dir" => {
                i += 1;
                index_dir = Some(PathBuf::from(&args[i]));
            }
            "--query-dir" => {
                i += 1;
                query_dir = Some(PathBuf::from(&args[i]));
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
            _ => {
                return Err(format!("Unknown option: {}", args[i]).into());
            }
        }
        i += 1;
    }

    let index_dir = index_dir.ok_or("--index-dir is required")?;
    let query_dir = query_dir.ok_or("--query-dir is required")?;

    // Load queries
    let queries = load_queries(&query_dir)?;
    eprintln!("Loaded {} queries", queries.len());

    // Search parameters
    let params = SearchParameters {
        batch_size: 2000,
        n_full_scores,
        top_k,
        n_ivf_probe,
        ..Default::default()
    };

    // Run search with either regular or mmap index
    let results = if use_mmap {
        eprintln!("Using memory-mapped index...");
        let index = MmapIndex::load(index_dir.to_str().unwrap())?;
        eprintln!("Loaded mmap index with {} documents", index.num_documents());
        index.search_batch(&queries, &params, true, None)?
    } else {
        let index = Index::load(index_dir.to_str().unwrap())?;
        eprintln!(
            "Loaded index with {} documents",
            index.metadata.num_documents
        );
        index.search_batch(&queries, &params, false, None)?
    };

    // Output results as JSON
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "query_id": r.query_id,
                "passage_ids": r.passage_ids,
                "scores": r.scores,
            })
        })
        .collect();

    println!("{}", serde_json::to_string(&json_results)?);

    Ok(())
}

#[cfg(not(feature = "npy"))]
fn run_search(_args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    Err("Search requires 'npy' feature".into())
}

#[cfg(feature = "npy")]
fn load_embeddings(
    data_dir: &Path,
) -> Result<Vec<ndarray::Array2<f32>>, Box<dyn std::error::Error>> {
    use ndarray::Array2;
    use ndarray_npy::ReadNpyExt;

    let mut embeddings = Vec::new();

    // Read doc_*.npy files in order
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
fn load_centroids(data_dir: &Path) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
    use ndarray::Array2;
    use ndarray_npy::ReadNpyExt;

    let centroids_path = data_dir.join("centroids.npy");
    let file = File::open(&centroids_path)?;
    let centroids: Array2<f32> = Array2::read_npy(file)?;
    Ok(centroids)
}

#[cfg(feature = "npy")]
fn load_queries(query_dir: &Path) -> Result<Vec<ndarray::Array2<f32>>, Box<dyn std::error::Error>> {
    use ndarray::Array2;
    use ndarray_npy::ReadNpyExt;

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
