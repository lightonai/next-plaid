pub mod paths;
pub mod state;
pub mod worktree;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::thread;

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::gitignore::GitignoreBuilder;
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use next_plaid::{
    delete_from_index, encode_index_chunk, filtering, prepare_codec_artifacts,
    write_index_from_encoded_chunks, EncodedIndexChunk, IndexConfig, Metadata, MmapIndex,
    SearchParameters, UpdateConfig,
};
use next_plaid_onnx::{pool_document_embeddings, Colbert, ExecutionProvider};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "_cuda")]
use crate::acceleration::apply_acceleration_mode;
use crate::acceleration::{env_acceleration_mode_lossy, AccelerationMode};
use crate::embed::build_embedding_text;
use crate::parser::{build_call_graph, detect_language, extract_units, CodeUnit, Language};
use crate::signal::{is_interrupted, is_interrupted_outside_critical, CriticalSectionGuard};

use paths::{
    acquire_index_lock, get_index_dir_for_project, get_vector_index_path, try_acquire_index_lock,
    ProjectMetadata,
};
use state::{file_stat, hash_file, FileInfo, IndexState, INDEX_FORMAT_VERSION};

/// Maximum file size to index (512 KB)
/// Files larger than this are skipped to avoid:
/// - Slow parsing of generated/minified code
/// - Memory issues with very large files
/// - Indexing non-source files (binaries, data files)
const MAX_FILE_SIZE: u64 = 512 * 1024;

/// Number of documents to process before writing to the index.
/// Larger values reduce I/O overhead but use more memory.
const INDEX_CHUNK_SIZE: usize = 1024;

/// Marker file (in the index dir, next to state.json) that records an in-progress
/// resumable build. Its presence routes the next run back into `build_resumable` so an
/// interrupted initial build (e.g. an agent's command timeout) resumes from where it left
/// off instead of restarting from scratch.
const BUILDING_MARKER: &str = ".building";

/// Approximate number of code units to embed per resumable-build batch before committing
/// state to disk. Each committed batch survives an interruption, so smaller values checkpoint
/// more often (more resumable) at the cost of more index-append overhead.
const BUILD_CHECKPOINT_UNITS: usize = 4096;

/// Test-only counter of expensive `delete_from_index` invocations.
///
/// Issue #116: deleting many files used to call the full-index-rewrite primitive once per
/// file, making an incremental update O(changed_files × index_size). Batching collapses that
/// to a single call; this counter lets regression tests assert the batching holds.
#[cfg(test)]
static DELETE_FROM_INDEX_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Wrapper over [`next_plaid::delete_from_index`] that counts calls under `cfg(test)`.
/// Zero overhead in release builds.
fn delete_from_index_counted(ids: &[i64], index_path: &str) -> Result<usize> {
    #[cfg(test)]
    DELETE_FROM_INDEX_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(delete_from_index(ids, index_path)?)
}

/// Remove every index entry belonging to any of `files`, in a single index rewrite.
///
/// Both deletion primitives are full-index operations: `delete_from_index` rewrites every
/// chunk and rebuilds the IVF, and `filtering::delete` rewrites the whole metadata table.
/// Calling them once per file makes an incremental update O(changed_files × index_size) — the
/// cause of issue #116, where ~276 deleted files hung for minutes on a ~1.2 GB index.
/// Collecting every doc ID up front and deleting once collapses that to a single O(index_size)
/// rewrite, regardless of how many files changed.
///
/// The doc IDs for ALL files must be read before any deletion: both primitives renumber the
/// surviving documents, so interleaving reads and deletes would invalidate the IDs that haven't
/// been deleted yet. Returns the number of documents removed.
fn delete_files_from_index(index_path: &str, files: &[PathBuf]) -> Result<usize> {
    if files.is_empty() {
        return Ok(0);
    }

    // Gather the doc IDs for every file first (dedup in case a path appears twice), then
    // delete the whole set in one pass.
    let mut ids: Vec<i64> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();
    for file_path in files {
        let file_str = file_path.to_string_lossy().to_string();
        let file_ids =
            filtering::where_condition(index_path, "file = ?", &[serde_json::json!(file_str)])
                .unwrap_or_default();
        for id in file_ids {
            if seen.insert(id) {
                ids.push(id);
            }
        }
    }

    if ids.is_empty() {
        return Ok(0);
    }

    // Vector index (codes/residuals/IVF). Single batched rewrite (issue #116).
    delete_from_index_counted(&ids, index_path)?;

    // Metadata and FTS5 must be deleted together and kept aligned. `filtering::delete`
    // re-sequences the surviving `_subset_` IDs, so the FTS5 rows have to follow: a
    // tail-only delete keeps every survivor's ID (drop just the deleted FTS rows,
    // O(deleted)); any other delete shifts survivor IDs and needs a full FTS5 rebuild.
    // This mirrors `next_plaid::MmapIndex::delete_with_options` — the standalone delete
    // path used here previously updated the vector index and metadata but skipped FTS5,
    // leaving deleted documents' text discoverable via regex/keyword (`-e`) search and,
    // after re-sequencing, misaligned with the surviving rows. Detect the suffix case
    // before `filtering::delete` renumbers the table.
    let old_num_documents = filtering::count(index_path).map(|n| n as i64).unwrap_or(0);
    let mut valid: Vec<i64> = ids
        .iter()
        .copied()
        .filter(|&id| id >= 0 && id < old_num_documents)
        .collect();
    valid.sort_unstable();
    valid.dedup();
    let is_suffix_delete = {
        let suffix_start = old_num_documents - valid.len() as i64;
        valid.first().is_some_and(|&min| min >= suffix_start)
    };

    filtering::delete(index_path, &ids)?;
    if is_suffix_delete {
        next_plaid::text_search::delete(index_path, &valid)?;
    } else {
        // Survivor IDs shifted; FTS5 rowids no longer match METADATA.
        next_plaid::text_search::rebuild(index_path)?;
    }
    Ok(ids.len())
}

/// Decide whether a sibling worktree's index directory is a usable seed source, returning its
/// loaded [`IndexState`] when so (and `None` otherwise).
///
/// A source is usable only if it is a **complete**, current-version, non-dirty index:
/// - No `.building` marker. A resumable build writes `metadata.json` and checkpoints state with
///   `dirty = false` after each committed batch, so an interrupted build looks complete by every
///   other check while only holding a fraction of its documents. The marker is the sole signal
///   that the build hasn't finished; seeding from it would copy a partial store and treat it as
///   whole. See issue #115.
/// - `metadata.json` present and a filtering DB present (rules out absent/stale-format stores
///   that `index()` would discard anyway).
/// - State loads, is non-empty, has a compatible index format, and isn't dirty.
fn seed_source_state(src_dir: &Path) -> Option<IndexState> {
    if src_dir.join(BUILDING_MARKER).exists() {
        return None;
    }

    let src_vector = get_vector_index_path(src_dir);
    let src_vector_str = src_vector.to_str()?;
    if !src_vector.join("metadata.json").exists() || !filtering::exists(src_vector_str) {
        return None;
    }

    match IndexState::load(src_dir) {
        Ok(s)
            if !s.files.is_empty()
                && s.index_format_version == INDEX_FORMAT_VERSION
                && !s.dirty =>
        {
            Some(s)
        }
        _ => None,
    }
}

/// Threshold for switching to higher pool factor (fewer embeddings per doc).
/// When encoding more than this many units, use LARGE_BATCH_POOL_FACTOR.
const LARGE_BATCH_THRESHOLD: usize = 10_000;

/// Pool factor to use for large batches (> LARGE_BATCH_THRESHOLD units).
/// Higher value = fewer embeddings = faster indexing and smaller index.
const LARGE_BATCH_POOL_FACTOR: usize = 2;

const DEFAULT_ENCODE_BATCH_SIZE: usize = 64;

/// Threshold for forcing CPU encoding even when CUDA is available.
/// For small batches (< this many units), CPU is faster due to GPU initialization overhead.
#[cfg(feature = "_cuda")]
const SMALL_BATCH_CPU_THRESHOLD: usize = 300;
/// Bounded channel capacity between the pool and index stages.
/// Kept small (4 chunks) to limit memory: each chunk holds full embeddings
/// waiting to be written to disk. Back-pressure here slows encoding when
/// disk I/O falls behind.
const POOLED_EMBEDDING_QUEUE_CAPACITY: usize = 4;

/// Bounded channel capacity for upstream pipeline stages.
/// Tokenized chunks and raw token-level embeddings can be large on codebases with many
/// long functions. Keeping these queues bounded makes back-pressure propagate all the way
/// to parsing instead of letting encoding run far ahead of indexing.
const PIPELINE_QUEUE_CAPACITY: usize = 2;

/// Bounded channel capacity for chunks waiting on initial-create coding.
/// During first index creation the coding thread waits for k-means before compressing chunks,
/// so this queue must stay small or pooled embeddings pile up in memory.
const CODING_QUEUE_CAPACITY: usize = 2;

/// Default ONNX sessions for indexing on accelerator providers (CoreML/DirectML/MIGraphX)
/// when the user has not configured `settings --parallel`. On these providers every session
/// is a full model copy held in device memory, so default to a single session and let users
/// opt into more with `--parallel`. CPU indexing uses [`default_index_parallel_sessions`]
/// instead, which scales with cores because CPU encoding cannot run ahead of the bounded
/// pipeline queues (so extra sessions add throughput at modest, bounded memory cost).
const DEFAULT_INDEX_ACCEL_PARALLEL_SESSIONS: usize = 1;

/// Pick the default number of ONNX sessions for indexing based on the chosen execution
/// provider and the size of the encoding workload, when the user has not set
/// `settings --parallel`.
///
/// - **CPU**: scale with available cores, but cap by the workload via
///   [`IndexBuilder::capped_default_sessions`] — spinning up more sessions than there are
///   ~`DEFAULT_ENCODE_BATCH_SIZE`-unit chunks just wastes per-session graph build time. The
///   bounded pipeline queues keep memory flat regardless of session count.
/// - **Accelerators** (CoreML/CUDA/DirectML/MIGraphX): keep a single session by default,
///   since each one duplicates the model in device memory. Users can raise it explicitly.
fn default_index_parallel_sessions(
    execution_provider: ExecutionProvider,
    num_units: usize,
) -> usize {
    match execution_provider {
        ExecutionProvider::Cpu => IndexBuilder::capped_default_sessions(num_units),
        _ => DEFAULT_INDEX_ACCEL_PARALLEL_SESSIONS,
    }
}

/// Bounded channel capacity between the index and metadata stages.
/// Larger than the pooled queue because metadata rows are lightweight
/// (just unit refs + doc IDs, no embedding data).
const METADATA_QUEUE_CAPACITY: usize = 8;

fn compiled_accelerated_execution_provider() -> Option<ExecutionProvider> {
    if cfg!(feature = "_cuda") {
        Some(ExecutionProvider::Cuda)
    } else if cfg!(feature = "tensorrt") {
        Some(ExecutionProvider::TensorRT)
    } else if cfg!(feature = "coreml") {
        Some(ExecutionProvider::CoreML)
    } else if cfg!(feature = "directml") {
        Some(ExecutionProvider::DirectML)
    } else if cfg!(feature = "migraphx") {
        Some(ExecutionProvider::MIGraphX)
    } else {
        None
    }
}

fn execution_provider_for_acceleration_mode(mode: AccelerationMode) -> ExecutionProvider {
    match mode {
        AccelerationMode::ForceCpu => ExecutionProvider::Cpu,
        AccelerationMode::ForceGpu => {
            compiled_accelerated_execution_provider().unwrap_or(ExecutionProvider::Cuda)
        }
        AccelerationMode::Auto => {
            compiled_accelerated_execution_provider().unwrap_or(ExecutionProvider::Cpu)
        }
    }
}

#[cfg(all(
    not(feature = "_cuda"),
    any(feature = "coreml", feature = "directml", feature = "migraphx")
))]
fn indexing_execution_provider_for_acceleration_mode(mode: AccelerationMode) -> ExecutionProvider {
    match mode {
        AccelerationMode::ForceCpu => ExecutionProvider::Cpu,
        AccelerationMode::ForceGpu => {
            compiled_accelerated_execution_provider().unwrap_or(ExecutionProvider::Cuda)
        }
        AccelerationMode::Auto if cfg!(feature = "coreml") => {
            // CoreML retains substantial per-shape state during bulk document encoding.
            // Keep Auto indexing memory-bounded; users can still opt into CoreML with
            // --force-gpu, and search query encoding continues to prefer CoreML.
            ExecutionProvider::Cpu
        }
        AccelerationMode::Auto => {
            compiled_accelerated_execution_provider().unwrap_or(ExecutionProvider::Cpu)
        }
    }
}

/// Maximum documents accumulated on the update path before flushing to the
/// PLAID index. Each flush rewrites the full IVF and invalidates the merged
/// mmap files (O(index size) regardless of batch size), so bigger batches mean
/// fewer full-index rewrites; the cap bounds the embeddings held in memory.
const UPDATE_FLUSH_DOC_LIMIT: usize = 8192;

struct ParsedFileResult {
    path: PathBuf,
    units: Vec<CodeUnit>,
    file_info: Option<FileInfo>,
    skip_reason: Option<String>,
}

#[derive(Debug)]
pub struct UpdateStats {
    pub added: usize,
    pub changed: usize,
    pub deleted: usize,
    pub unchanged: usize,
    pub skipped: usize,
}

#[derive(Debug, Default)]
pub struct UpdatePlan {
    pub added: Vec<PathBuf>,
    pub changed: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
    pub unchanged: usize,
    /// Files whose content is unchanged (hash matches) but whose mtime/size on
    /// disk drifted from what state recorded — e.g. after `git checkout` or
    /// `touch`. They need no re-embedding, only a refreshed `FileInfo` persisted
    /// so the stat-only fast path applies next run instead of re-hashing forever.
    pub touched: Vec<(PathBuf, FileInfo)>,
}

// ============================================================================
// Pipeline data types
//
// Indexing runs as a multi-stage pipeline connected by channels:
//
//   [main thread]       [tokenize]    [encode]     [pool]       [index]      [metadata]
//   SortedUnit ──dedup──▶ Prepared ──▶ Tokenized ──▶ RawEncoded ──▶ Pooled ──▶ Indexed ──▶ DB
//        │                                                            │
//        └─ text built from CodeUnit                                  └─ written to PLAID index
//
// Each struct below represents the data flowing between two stages.
// The `original_to_unique` map tracks deduplication: identical code units
// share a single embedding, expanded back to the full set after pooling.
// ============================================================================

#[derive(Clone)]
struct SortedUnit {
    unit: Arc<CodeUnit>,
    text: Arc<str>,
}

/// After deduplication: unique texts to encode + a map from original positions
/// back to their unique index, so duplicates share a single GPU encoding pass.
struct PreparedChunk {
    units: Vec<Arc<CodeUnit>>,
    unique_texts: Vec<Arc<str>>,
    original_to_unique: Vec<usize>,
}

struct TokenizedChunk {
    units: Vec<Arc<CodeUnit>>,
    prepared_batches: Vec<next_plaid_onnx::PreparedDocumentBatch>,
    original_to_unique: Vec<usize>,
}

struct RawEncodedChunk {
    units: Vec<Arc<CodeUnit>>,
    raw_embeddings: Vec<ndarray::Array2<f32>>,
    original_to_unique: Vec<usize>,
}

/// After pooling: full per-document embeddings ready for PLAID index insertion.
/// Deduplication has been expanded — each unit has its own embedding copy.
struct PooledChunkForIndex {
    units: Vec<Arc<CodeUnit>>,
    embeddings: Vec<ndarray::Array2<f32>>,
}

/// After index insertion: the assigned document IDs for metadata DB storage.
struct IndexedChunkForMetadata {
    units: Vec<Arc<CodeUnit>>,
    doc_ids: Vec<i64>,
}

struct ChunkPipelineConfig<'a> {
    index_chunk_size: usize,
    pool_factor: Option<usize>,
    index_path: &'a str,
    config: IndexConfig,
    update_config: UpdateConfig,
    pb: Option<&'a ProgressBar>,
}

/// Embeddings waiting to be compressed and written to the PLAID index.
/// Sent to the coding thread which runs k-means + residual compression.
struct ChunkForCoding {
    embeddings: Vec<ndarray::Array2<f32>>,
}

/// Threshold for prompting user confirmation before indexing.
/// When encoding more than this many units, prompt the user unless auto_confirm is set.
pub const CONFIRMATION_THRESHOLD: usize = 30_000;

fn prepare_units_for_encoding(units: &[CodeUnit], sample_prefix_size: usize) -> Vec<SortedUnit> {
    let mut items: Vec<SortedUnit> = units
        .iter()
        .map(|unit| SortedUnit {
            unit: Arc::new(unit.clone()),
            text: Arc::<str>::from(build_embedding_text(unit)),
        })
        .collect();

    // Sort by file then line so units from the same file stay together,
    // then same-folder files are adjacent. This groups semantically related
    // code for better k-means centroids in the PLAID index.
    items.sort_unstable_by(|a, b| {
        a.unit
            .file
            .cmp(&b.unit.file)
            .then_with(|| a.unit.line.cmp(&b.unit.line))
    });

    // Take a strided sample for the k-means seed prefix so centroids are
    // representative of the whole codebase rather than just the first files.
    let sample_prefix_size = sample_prefix_size.min(items.len());
    if sample_prefix_size > 0 && sample_prefix_size < items.len() {
        let stride = items.len() / sample_prefix_size;
        let sampled_indices: std::collections::HashSet<usize> =
            (0..sample_prefix_size).map(|i| i * stride).collect();
        let (prefix, remainder): (Vec<_>, Vec<_>) = items
            .into_iter()
            .enumerate()
            .partition::<Vec<_>, _>(|(i, _)| sampled_indices.contains(i));
        let mut result: Vec<SortedUnit> = prefix.into_iter().map(|(_, item)| item).collect();
        result.extend(remainder.into_iter().map(|(_, item)| item));
        result
    } else {
        items
    }
}

/// Deduplicate code units with identical embedding text within a chunk.
/// Returns only unique texts for encoding, plus a mapping so each original
/// unit can retrieve its embedding after the GPU pass. On large codebases
/// this saves ~8% of encoding work (e.g. re-exported types, trait impls).
fn prepare_deduplicated_chunk(unit_chunk: &[SortedUnit]) -> PreparedChunk {
    let mut index_by_text: HashMap<&str, usize> = HashMap::new();
    let mut unique_texts: Vec<Arc<str>> = Vec::new();
    let mut original_to_unique: Vec<usize> = Vec::with_capacity(unit_chunk.len());

    for item in unit_chunk.iter() {
        if let Some(&unique_idx) = index_by_text.get(item.text.as_ref()) {
            original_to_unique.push(unique_idx);
        } else {
            let unique_idx = unique_texts.len();
            index_by_text.insert(item.text.as_ref(), unique_idx);
            unique_texts.push(Arc::clone(&item.text));
            original_to_unique.push(unique_idx);
        }
    }

    PreparedChunk {
        units: unit_chunk
            .iter()
            .map(|item| Arc::clone(&item.unit))
            .collect(),
        unique_texts,
        original_to_unique,
    }
}

fn run_encode_stage(
    receiver: mpsc::Receiver<TokenizedChunk>,
    sender: mpsc::SyncSender<RawEncodedChunk>,
    model: Colbert,
) -> Result<()> {
    while let Ok(chunk) = receiver.recv() {
        let raw_embeddings = model.encode_prepared_document_batches(chunk.prepared_batches)?;

        sender
            .send(RawEncodedChunk {
                units: chunk.units,
                raw_embeddings,
                original_to_unique: chunk.original_to_unique,
            })
            .context("Failed to send raw embeddings to pooling stage")?;
    }

    Ok(())
}

fn run_tokenize_stage(
    receiver: mpsc::Receiver<PreparedChunk>,
    sender: mpsc::SyncSender<TokenizedChunk>,
    model: Colbert,
) -> Result<()> {
    while let Ok(chunk) = receiver.recv() {
        let text_refs: Vec<&str> = chunk
            .unique_texts
            .iter()
            .map(|text| text.as_ref())
            .collect();
        let prepared_batches = model.tokenize_documents_in_batches(&text_refs)?;

        sender
            .send(TokenizedChunk {
                units: chunk.units,
                prepared_batches,
                original_to_unique: chunk.original_to_unique,
            })
            .context("Failed to send tokenized chunk to encode stage")?;
    }

    Ok(())
}

/// Pool raw token-level embeddings into document-level vectors, then expand
/// deduplicated embeddings back to the full set. After this stage every
/// original code unit has its own embedding copy ready for index insertion.
fn run_pool_stage(
    receiver: mpsc::Receiver<RawEncodedChunk>,
    sender: mpsc::SyncSender<PooledChunkForIndex>,
    pool_factor: Option<usize>,
) -> Result<()> {
    while let Ok(chunk) = receiver.recv() {
        let pooled_unique = pool_document_embeddings(chunk.raw_embeddings, pool_factor);
        // Expand: map each original unit back to its deduplicated embedding.
        let embeddings = chunk
            .original_to_unique
            .into_iter()
            .map(|unique_idx| pooled_unique[unique_idx].clone())
            .collect();

        sender
            .send(PooledChunkForIndex {
                units: chunk.units,
                embeddings,
            })
            .context("Failed to send pooled embeddings to index stage")?;
    }

    Ok(())
}

/// Write pooled embeddings to the PLAID index, assign doc IDs, and forward to metadata.
///
/// Two modes:
/// - **Initial create**: The first chunk seeds k-means centroids in a background thread.
///   While k-means runs, subsequent chunks are buffered in a coding channel. Once centroids
///   are ready, all chunks are compressed and written in one batch — producing a single
///   contiguous index file rather than incremental updates.
/// - **Update**: Each chunk is appended to the existing index via `update_or_create`.
fn run_index_stage(
    receiver: mpsc::Receiver<PooledChunkForIndex>,
    sender: mpsc::SyncSender<IndexedChunkForMetadata>,
    index_path: String,
    config: IndexConfig,
    update_config: UpdateConfig,
    initial_kmeans_sample_docs: usize,
) -> Result<()> {
    let initial_create = !Path::new(&index_path).join("metadata.json").exists();

    if initial_create {
        // Wait for the first chunk to use its embeddings as the k-means seed sample.
        let first_chunk = match receiver.recv() {
            Ok(chunk) => chunk,
            Err(_) => return Ok(()),
        };

        let mut next_doc_id = 0i64;
        let mut initial_create_config = config.clone();
        if initial_create_config.n_samples_kmeans.is_none() {
            initial_create_config.n_samples_kmeans = Some(initial_kmeans_sample_docs.max(1));
        }

        // Spawn k-means in a background thread so encoding can continue in parallel.
        // The coding thread blocks on the k-means result before compressing any chunks.
        let sample_embeddings = first_chunk.embeddings.clone();
        let kmeans_config = initial_create_config.clone();
        let kmeans_handle = thread::Builder::new()
            .name("colgrep-kmeans".to_string())
            .spawn(move || -> Result<_> {
                let centroids = next_plaid::compute_kmeans(
                    &sample_embeddings,
                    &next_plaid::kmeans::ComputeKmeansConfig {
                        kmeans_niters: kmeans_config.kmeans_niters,
                        max_points_per_centroid: kmeans_config.max_points_per_centroid,
                        seed: kmeans_config.seed.unwrap_or(42),
                        n_samples_kmeans: kmeans_config.n_samples_kmeans,
                        num_partitions: None,
                        force_cpu: kmeans_config.force_cpu,
                    },
                )?;
                Ok(prepare_codec_artifacts(
                    &sample_embeddings,
                    centroids,
                    &kmeans_config,
                )?)
            })
            .context("Failed to spawn kmeans stage thread")?;

        // Coding thread: waits for k-means centroids, then compresses every
        // chunk's embeddings into residual codes. All encoded chunks are collected
        // in memory and written to disk in a single atomic operation at the end,
        // which avoids a partially-written index if the process is interrupted.
        let (dedup_tx, dedup_rx) = mpsc::sync_channel::<ChunkForCoding>(CODING_QUEUE_CAPACITY);
        let coding_index_path = index_path.clone();
        let coding_config = initial_create_config.clone();
        let coding_force_cpu = update_config.force_cpu;
        let coding_handle = thread::Builder::new()
            .name("colgrep-coding".to_string())
            .spawn(move || -> Result<()> {
                let codec_artifacts = kmeans_handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("K-means stage thread panicked"))??;
                let mut encoded_chunks: Vec<EncodedIndexChunk> = Vec::new();
                while let Ok(chunk) = dedup_rx.recv() {
                    let encoded = encode_index_chunk(
                        &chunk.embeddings,
                        &codec_artifacts.codec,
                        coding_force_cpu,
                    )?;
                    encoded_chunks.push(encoded);
                }
                // Guard prevents Ctrl-C from interrupting the disk write mid-operation.
                let _guard = CriticalSectionGuard::new();
                write_index_from_encoded_chunks(
                    &encoded_chunks,
                    &codec_artifacts,
                    &coding_index_path,
                    &coding_config,
                )?;
                Ok(())
            })
            .context("Failed to spawn coding stage thread")?;

        let mut handle_chunk = |chunk: PooledChunkForIndex| -> Result<()> {
            let doc_count = chunk.embeddings.len();
            let doc_ids: Vec<i64> = (next_doc_id..next_doc_id + doc_count as i64).collect();
            next_doc_id += doc_count as i64;

            dedup_tx
                .send(ChunkForCoding {
                    embeddings: chunk.embeddings,
                })
                .context("Failed to send chunk to coding stage")?;

            sender
                .send(IndexedChunkForMetadata {
                    units: chunk.units,
                    doc_ids,
                })
                .context("Failed to send indexed chunk to metadata stage")
        };

        handle_chunk(first_chunk)?;
        while let Ok(chunk) = receiver.recv() {
            handle_chunk(chunk)?;
        }
        drop(dedup_tx);
        coding_handle
            .join()
            .map_err(|_| anyhow::anyhow!("Coding stage thread panicked"))??;
        return Ok(());
    }

    // Update path: accumulate pipeline chunks and flush in large batches.
    // Every MmapIndex::update_or_create call rewrites the full IVF and deletes
    // the merged mmap files no matter how few documents it adds, so its cost is
    // O(index size) per call. Flushing once per UPDATE_FLUSH_DOC_LIMIT documents
    // instead of once per pipeline chunk means a typical incremental update pays
    // that cost exactly once.
    let mut pending_units: Vec<Arc<CodeUnit>> = Vec::new();
    let mut pending_embeddings: Vec<ndarray::Array2<f32>> = Vec::new();

    while let Ok(chunk) = receiver.recv() {
        pending_units.extend(chunk.units);
        pending_embeddings.extend(chunk.embeddings);

        if pending_embeddings.len() >= UPDATE_FLUSH_DOC_LIMIT {
            flush_update_batch(
                &mut pending_units,
                &mut pending_embeddings,
                &sender,
                &index_path,
                &config,
                &update_config,
            )?;
        }
    }

    flush_update_batch(
        &mut pending_units,
        &mut pending_embeddings,
        &sender,
        &index_path,
        &config,
        &update_config,
    )?;

    Ok(())
}

/// Write one accumulated batch to the PLAID index and hand the assigned doc IDs
/// to the metadata stage. No-op when the batch is empty.
fn flush_update_batch(
    units: &mut Vec<Arc<CodeUnit>>,
    embeddings: &mut Vec<ndarray::Array2<f32>>,
    sender: &mpsc::SyncSender<IndexedChunkForMetadata>,
    index_path: &str,
    config: &IndexConfig,
    update_config: &UpdateConfig,
) -> Result<()> {
    if embeddings.is_empty() {
        return Ok(());
    }

    let _guard = CriticalSectionGuard::new();
    let (_, doc_ids) = MmapIndex::update_or_create(embeddings, index_path, config, update_config)?;
    embeddings.clear();

    sender
        .send(IndexedChunkForMetadata {
            units: std::mem::take(units),
            doc_ids,
        })
        .context("Failed to send indexed chunk to metadata stage")
}

/// Number of candidates exactly re-ranked with MaxSim after the cheap
/// centroid-score stage. Empirically on the semble 63-repo bench, bumping
/// from next-plaid's 4096 default to 8192 lifts mean NDCG@10 by ~0.006
/// (~+0.7%) at the cost of ~+280ms p50 query latency on GPU. Going beyond
/// 8192 (tested 16384) adds zero NDCG, so the gain saturates here — this
/// is a genuine PLAID-recall fix, not a knob-overfit.
const DEFAULT_N_FULL_SCORES: usize = 8192;

/// Build [`SearchParameters`] for next-plaid, allowing env-var overrides of
/// the three knobs that affect PLAID recall vs latency:
///
/// * `COLGREP_N_IVF_PROBE` (default = next-plaid 8) — how many IVF cells
///   to probe. Saturated empirically; left at the next-plaid default.
/// * `COLGREP_N_FULL_SCORES` (default 8192) — how many candidates get
///   exact MaxSim re-ranking after the cheap centroid stage.
/// * `COLGREP_CENTROID_SCORE_THRESHOLD` (default = next-plaid 0.4) —
///   minimum centroid max-score to be considered; -1 disables pruning.
///   Saturated empirically; left at the next-plaid default.
///
/// All three are search-time only, so they can be tuned on cached indices
/// without re-indexing.
fn search_params_from_env(top_k: usize) -> SearchParameters {
    let defaults = SearchParameters::default();
    let n_ivf_probe = std::env::var("COLGREP_N_IVF_PROBE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(defaults.n_ivf_probe);
    let n_full_scores = std::env::var("COLGREP_N_FULL_SCORES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_N_FULL_SCORES);
    let centroid_score_threshold = match std::env::var("COLGREP_CENTROID_SCORE_THRESHOLD") {
        Ok(s) => match s.parse::<f32>() {
            Ok(v) if v < 0.0 => None,
            Ok(v) => Some(v),
            Err(_) => defaults.centroid_score_threshold,
        },
        Err(_) => defaults.centroid_score_threshold,
    };
    SearchParameters {
        top_k,
        n_ivf_probe,
        n_full_scores,
        centroid_score_threshold,
        ..defaults
    }
}

fn run_metadata_stage(
    receiver: mpsc::Receiver<IndexedChunkForMetadata>,
    index_path: String,
    pb: Option<ProgressBar>,
) -> Result<()> {
    let mut filtering_exists = filtering::exists(&index_path);
    let mut completed_units = 0u64;

    while let Ok(chunk) = receiver.recv() {
        let metadata: Vec<serde_json::Value> = chunk
            .units
            .iter()
            .map(|unit| serde_json::to_value(unit.as_ref()).unwrap())
            .collect();
        let db_result = if filtering_exists {
            filtering::update(&index_path, &metadata, &chunk.doc_ids)
        } else {
            filtering::create(&index_path, &metadata, &chunk.doc_ids)
        };

        // If metadata insertion fails, remove the embeddings we just wrote
        // to keep the PLAID index and SQLite DB in sync.
        if let Err(e) = db_result {
            if let Err(rollback_err) = delete_from_index(&chunk.doc_ids, &index_path) {
                eprintln!("⚠️  Rollback failed: {}", rollback_err);
            }
            return Err(e.into());
        }

        if let Err(e) = next_plaid::text_search::index(
            &index_path,
            &metadata,
            &chunk.doc_ids,
            &next_plaid::FtsTokenizer::IdentifierAware,
        ) {
            eprintln!("⚠️  FTS indexing failed (non-fatal): {}", e);
        }

        filtering_exists = true;
        completed_units += chunk.units.len() as u64;
        if let Some(pb) = pb.as_ref() {
            pb.set_position(completed_units);
        }
    }

    Ok(())
}

/// Wire up and run the full indexing pipeline across dedicated threads.
///
/// The main thread feeds deduplicated chunks into the pipeline, which flows:
///   tokenize → encode (GPU/CPU) → pool → index (PLAID write) → metadata (SQLite)
///
/// Unbounded channels connect fast stages; bounded channels (`sync_channel`)
/// sit before the index and metadata stages to cap memory when disk I/O is slow.
/// Dropping `tokenize_tx` triggers cascading shutdown through the pipeline.
fn run_chunk_pipeline(
    model: Colbert,
    sorted_units: &[SortedUnit],
    pipeline: ChunkPipelineConfig<'_>,
) -> Result<bool> {
    let mut was_interrupted = false;
    let ChunkPipelineConfig {
        index_chunk_size,
        pool_factor,
        index_path,
        config,
        update_config,
        pb,
    } = pipeline;

    let (tokenize_tx, tokenize_rx) = mpsc::sync_channel::<PreparedChunk>(PIPELINE_QUEUE_CAPACITY);
    let (encode_tx, encode_rx) = mpsc::sync_channel::<TokenizedChunk>(PIPELINE_QUEUE_CAPACITY);
    let (pool_tx, pool_rx) = mpsc::sync_channel::<RawEncodedChunk>(PIPELINE_QUEUE_CAPACITY);
    let (index_tx, index_rx) =
        mpsc::sync_channel::<PooledChunkForIndex>(POOLED_EMBEDDING_QUEUE_CAPACITY);
    let (metadata_tx, metadata_rx) =
        mpsc::sync_channel::<IndexedChunkForMetadata>(METADATA_QUEUE_CAPACITY);

    let tokenize_model = model.clone();
    let tokenize_handle = thread::Builder::new()
        .name("colgrep-tokenize".to_string())
        .spawn(move || run_tokenize_stage(tokenize_rx, encode_tx, tokenize_model))
        .context("Failed to spawn tokenize stage thread")?;
    let encode_model = model.clone();
    let encode_handle = thread::Builder::new()
        .name("colgrep-encode".to_string())
        .spawn(move || run_encode_stage(encode_rx, pool_tx, encode_model))
        .context("Failed to spawn encode stage thread")?;
    let pool_handle = thread::Builder::new()
        .name("colgrep-pool".to_string())
        .spawn(move || run_pool_stage(pool_rx, index_tx, pool_factor))
        .context("Failed to spawn pool stage thread")?;
    let index_path_for_index = index_path.to_string();
    let index_handle = thread::Builder::new()
        .name("colgrep-index".to_string())
        .spawn(move || {
            run_index_stage(
                index_rx,
                metadata_tx,
                index_path_for_index,
                config,
                update_config,
                index_chunk_size,
            )
        })
        .context("Failed to spawn index stage thread")?;
    let index_path_for_metadata = index_path.to_string();
    let metadata_pb = pb.cloned();
    let metadata_handle = thread::Builder::new()
        .name("colgrep-metadata".to_string())
        .spawn(move || run_metadata_stage(metadata_rx, index_path_for_metadata, metadata_pb))
        .context("Failed to spawn metadata stage thread")?;

    for unit_chunk in sorted_units.chunks(index_chunk_size) {
        if is_interrupted_outside_critical() {
            was_interrupted = true;
            break;
        }

        let prepared = prepare_deduplicated_chunk(unit_chunk);

        tokenize_tx
            .send(prepared)
            .context("Failed to send prepared chunk to tokenize stage")?;
    }

    // Signal pipeline shutdown: dropping the sender closes the channel,
    // causing each stage's `recv()` loop to exit, which cascades through
    // all downstream stages.
    drop(tokenize_tx);

    tokenize_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Tokenize stage thread panicked"))??;
    encode_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Encode stage thread panicked"))??;
    pool_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Pool stage thread panicked"))??;
    index_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Index stage thread panicked"))??;
    metadata_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Metadata stage thread panicked"))??;

    Ok(was_interrupted)
}

fn parse_files_parallel(
    project_root: &Path,
    paths: &[PathBuf],
    pb: Option<&ProgressBar>,
) -> Vec<ParsedFileResult> {
    let progress = pb.cloned();

    paths
        .par_iter()
        .map(|path| {
            if is_interrupted() {
                return ParsedFileResult {
                    path: path.clone(),
                    units: Vec::new(),
                    file_info: None,
                    skip_reason: None,
                };
            }

            let full_path = project_root.join(path);
            let result = match detect_language(&full_path) {
                Some(lang) => match std::fs::read_to_string(&full_path) {
                    Ok(source) => {
                        let units = extract_units(path, &source, lang);
                        match hash_file(&full_path) {
                            Ok(content_hash) => match FileInfo::probe(&full_path, content_hash) {
                                Ok(file_info) => ParsedFileResult {
                                    path: path.clone(),
                                    units,
                                    file_info: Some(file_info),
                                    skip_reason: None,
                                },
                                Err(e) => ParsedFileResult {
                                    path: path.clone(),
                                    units: Vec::new(),
                                    file_info: None,
                                    skip_reason: Some(format!(
                                        "Skipping {} ({})",
                                        full_path.display(),
                                        e
                                    )),
                                },
                            },
                            Err(e) => ParsedFileResult {
                                path: path.clone(),
                                units: Vec::new(),
                                file_info: None,
                                skip_reason: Some(format!(
                                    "Skipping {} ({})",
                                    full_path.display(),
                                    e
                                )),
                            },
                        }
                    }
                    Err(e) => ParsedFileResult {
                        path: path.clone(),
                        units: Vec::new(),
                        file_info: None,
                        skip_reason: Some(format!("Skipping {} ({})", full_path.display(), e)),
                    },
                },
                None => ParsedFileResult {
                    path: path.clone(),
                    units: Vec::new(),
                    file_info: None,
                    skip_reason: None,
                },
            };

            if let Some(pb) = &progress {
                pb.inc(1);
            }

            result
        })
        .collect()
}

pub struct IndexBuilder {
    /// The model is lazily created only when needed for encoding
    model: Option<Colbert>,
    /// Builder parameters for lazy model creation
    model_path: PathBuf,
    quantized: bool,
    parallel_sessions: Option<usize>,
    batch_size: Option<usize>,
    project_root: PathBuf,
    index_dir: PathBuf,
    pool_factor: Option<usize>,
    encode_batch_size: Option<usize>,
    index_chunk_size: Option<usize>,
    dynamic_batch: bool,
    /// If true, skip user confirmation for large indexes
    auto_confirm: bool,
    /// Model id (e.g., "lightonai/LateOn-Code-edge"). Used both to scope the index
    /// directory (per-model indexes) and for "🤖 Model:" display.
    model_id: String,
}

impl IndexBuilder {
    pub fn new(project_root: &Path, model_id: &str, model_path: &Path) -> Result<Self> {
        Self::with_options(project_root, model_id, model_path, false, None, None, None)
    }

    pub fn with_quantized(
        project_root: &Path,
        model_id: &str,
        model_path: &Path,
        quantized: bool,
    ) -> Result<Self> {
        Self::with_options(
            project_root,
            model_id,
            model_path,
            quantized,
            None,
            None,
            None,
        )
    }

    pub fn with_options(
        project_root: &Path,
        model_id: &str,
        model_path: &Path,
        quantized: bool,
        pool_factor: Option<usize>,
        parallel_sessions: Option<usize>,
        batch_size: Option<usize>,
    ) -> Result<Self> {
        // Index directory is scoped to (path, model) so different models keep
        // separate indexes and switching models doesn't corrupt the existing one.
        let index_dir = get_index_dir_for_project(project_root, model_id)?;

        Ok(Self {
            model: None, // Lazily created when needed
            model_path: model_path.to_path_buf(),
            quantized,
            parallel_sessions,
            batch_size,
            project_root: project_root.to_path_buf(),
            index_dir,
            pool_factor,
            encode_batch_size: None,
            index_chunk_size: None,
            dynamic_batch: true,
            auto_confirm: false, // Prompt by default for large indexes
            model_id: model_id.to_string(),
        })
    }

    /// Set whether to automatically confirm indexing for large codebases (> 10K code units)
    pub fn set_auto_confirm(&mut self, auto_confirm: bool) {
        self.auto_confirm = auto_confirm;
    }

    pub fn set_encode_batch_size(&mut self, encode_batch_size: usize) {
        self.encode_batch_size = Some(encode_batch_size.max(1));
    }

    pub fn set_index_chunk_size(&mut self, index_chunk_size: usize) {
        self.index_chunk_size = Some(index_chunk_size.max(1));
    }

    pub fn set_dynamic_batch(&mut self, dynamic_batch: bool) {
        self.dynamic_batch = dynamic_batch;
    }

    /// Default session count for an encoding workload of `num_units` units.
    /// Each ONNX session pays a full graph parse/optimize/allocate on build, so
    /// spinning up the machine-wide default (up to 16) to encode a handful of
    /// changed units costs far more in session builds than it saves in
    /// parallelism. One session per DEFAULT_ENCODE_BATCH_SIZE units amortizes
    /// the build cost; explicit user configuration bypasses this cap.
    fn capped_default_sessions(num_units: usize) -> usize {
        let max_useful_sessions = num_units.div_ceil(DEFAULT_ENCODE_BATCH_SIZE).max(1);
        crate::config::get_default_cpu_parallel_sessions().min(max_useful_sessions)
    }

    /// Ensure the model is created for encoding.
    /// The model is lazily created on first use to avoid overhead when just scanning files
    /// or when checking for index updates that have no changes.
    ///
    /// # Arguments
    /// * `num_units` - Number of code units to encode. Used to decide whether to use GPU or CPU.
    ///   For small batches (< SMALL_BATCH_CPU_THRESHOLD), CPU is preferred even when CUDA is
    ///   available, as GPU initialization overhead outweighs the benefits for small workloads.
    fn ensure_model_created(&mut self, num_units: usize) -> Result<()> {
        if self.model.is_none() {
            #[cfg(feature = "_cuda")]
            let acceleration_mode = env_acceleration_mode_lossy();

            #[cfg(feature = "_cuda")]
            let (num_sessions, execution_provider) = {
                match acceleration_mode {
                    AccelerationMode::ForceCpu => {
                        apply_acceleration_mode(AccelerationMode::ForceCpu);
                        crate::onnx_runtime::ensure_onnx_runtime()
                            .context("Failed to initialize ONNX Runtime")?;

                        (
                            self.parallel_sessions.unwrap_or_else(|| {
                                default_index_parallel_sessions(ExecutionProvider::Cpu, num_units)
                            }),
                            ExecutionProvider::Cpu,
                        )
                    }
                    AccelerationMode::ForceGpu => {
                        apply_acceleration_mode(AccelerationMode::ForceGpu);
                        crate::onnx_runtime::ensure_onnx_runtime()
                            .context("Failed to initialize ONNX Runtime")?;

                        if !crate::onnx_runtime::is_cudnn_available() {
                            anyhow::bail!("FORCE_GPU is set, but cuDNN was not initialized");
                        }

                        if !next_plaid_onnx::is_cuda_available() {
                            anyhow::bail!(
                                "FORCE_GPU is set, but the CUDA execution provider was not initialized"
                            );
                        }

                        (
                            self.parallel_sessions
                                .unwrap_or(crate::config::DEFAULT_PARALLEL_SESSIONS_GPU),
                            ExecutionProvider::Cuda,
                        )
                    }
                    AccelerationMode::Auto => {
                        let force_cpu = num_units < SMALL_BATCH_CPU_THRESHOLD;
                        if force_cpu {
                            apply_acceleration_mode(AccelerationMode::ForceCpu);
                        } else {
                            apply_acceleration_mode(AccelerationMode::Auto);
                        }

                        crate::onnx_runtime::ensure_onnx_runtime()
                            .context("Failed to initialize ONNX Runtime")?;

                        let use_cuda = !force_cpu && {
                            crate::onnx_runtime::is_cudnn_available()
                                && next_plaid_onnx::is_cuda_available()
                        };

                        if use_cuda {
                            (
                                self.parallel_sessions
                                    .unwrap_or(crate::config::DEFAULT_PARALLEL_SESSIONS_GPU),
                                ExecutionProvider::Cuda,
                            )
                        } else {
                            (
                                self.parallel_sessions.unwrap_or_else(|| {
                                    default_index_parallel_sessions(
                                        ExecutionProvider::Cpu,
                                        num_units,
                                    )
                                }),
                                ExecutionProvider::Cpu,
                            )
                        }
                    }
                }
            };
            #[cfg(not(any(
                feature = "_cuda",
                feature = "directml",
                feature = "migraphx",
                feature = "coreml"
            )))]
            let (num_sessions, execution_provider) = {
                crate::onnx_runtime::ensure_onnx_runtime()
                    .context("Failed to initialize ONNX Runtime")?;

                (
                    self.parallel_sessions.unwrap_or_else(|| {
                        default_index_parallel_sessions(ExecutionProvider::Cpu, num_units)
                    }),
                    ExecutionProvider::Cpu,
                )
            };

            #[cfg(any(feature = "directml", feature = "migraphx", feature = "coreml"))]
            #[cfg(not(feature = "_cuda"))]
            let (num_sessions, execution_provider) = {
                let execution_provider = indexing_execution_provider_for_acceleration_mode(
                    env_acceleration_mode_lossy(),
                );

                crate::onnx_runtime::ensure_onnx_runtime()
                    .context("Failed to initialize ONNX Runtime")?;

                (
                    self.parallel_sessions.unwrap_or_else(|| {
                        default_index_parallel_sessions(execution_provider, num_units)
                    }),
                    execution_provider,
                )
            };

            // Print model info after ONNX runtime is initialized (and any potential re-exec)
            eprintln!("🤖 Model: {} ({})", self.model_id, execution_provider);
            eprintln!("📂 Building index...");

            // Use runtime default for batch size (respects cuDNN availability)
            let batch = self
                .batch_size
                .unwrap_or_else(crate::config::get_default_batch_size);

            // Suppress stderr during model loading to hide CoreML's harmless
            // "Context leak detected" warnings on macOS.
            // `with_suppressed_stderr` captures any panic message via a temporary
            // panic hook and prints it to the restored stderr before resuming,
            // so panics inside the suppressed region remain visible.
            let model = crate::stderr::with_suppressed_stderr(|| {
                Colbert::builder(&self.model_path)
                    .with_quantized(self.quantized)
                    .with_parallel(num_sessions)
                    .with_batch_size(batch)
                    .with_dynamic_batch(self.dynamic_batch)
                    .with_execution_provider(execution_provider)
                    .build()
            })
            .context("Failed to load ColBERT model")?;

            self.model = Some(model);
        }
        Ok(())
    }

    /// Get a reference to the model. Panics if model is not created.
    /// Call ensure_model_created() first.
    fn model(&self) -> &Colbert {
        self.model
            .as_ref()
            .expect("Model not created. Call ensure_model_created() first.")
    }

    /// Check if the current model is using GPU execution.
    #[cfg(feature = "_cuda")]
    fn is_using_gpu(&self) -> bool {
        self.model
            .as_ref()
            .is_some_and(|m| !matches!(m.requested_execution_provider, ExecutionProvider::Cpu))
    }

    /// Drop the current GPU model and rebuild with CPU execution.
    /// The ONNX Runtime is already initialized — only the model sessions are recreated.
    /// Uses `dynamic_batch(false)` because CPU encoding processes fixed-size batches
    /// sequentially — the token-budget bucketing of dynamic batch only helps GPU
    /// where plan reuse across similar shapes reduces kernel launch overhead.
    #[cfg(feature = "_cuda")]
    fn rebuild_model_for_cpu(&mut self) -> Result<()> {
        self.model = None;
        apply_acceleration_mode(AccelerationMode::ForceCpu);

        // CPU fallback rebuild: the original unit count isn't threaded here, so use the
        // uncapped (cores-based) CPU default — `usize::MAX` disables the workload cap.
        let num_sessions = self
            .parallel_sessions
            .unwrap_or_else(|| default_index_parallel_sessions(ExecutionProvider::Cpu, usize::MAX));
        let batch = crate::config::DEFAULT_BATCH_SIZE_CPU;

        let model = crate::stderr::with_suppressed_stderr(|| {
            Colbert::builder(&self.model_path)
                .with_quantized(self.quantized)
                .with_parallel(num_sessions)
                .with_batch_size(batch)
                .with_dynamic_batch(false)
                .with_execution_provider(ExecutionProvider::Cpu)
                .build()
        })
        .context("Failed to load ColBERT model for CPU fallback")?;

        self.model = Some(model);
        Ok(())
    }

    /// Run the encoding pipeline, falling back to CPU when GPU encoding fails.
    ///
    /// In auto mode, a GPU failure (e.g. OOM) triggers a transparent retry on CPU.
    /// With `--force-gpu`, failures produce a clear error instead of the raw ONNX message.
    fn run_encoding_pipeline(
        &mut self,
        sorted_units: &[SortedUnit],
        index_chunk_size: usize,
        pool_factor: Option<usize>,
        index_path: &str,
        pb: Option<&ProgressBar>,
    ) -> Result<bool> {
        let force_cpu = next_plaid::is_force_cpu();
        let config = IndexConfig {
            force_cpu,
            ..Default::default()
        };
        let update_config = UpdateConfig {
            force_cpu,
            ..Default::default()
        };

        let result = run_chunk_pipeline(
            self.model().clone(),
            sorted_units,
            ChunkPipelineConfig {
                index_chunk_size,
                pool_factor,
                index_path,
                config,
                update_config,
                pb,
            },
        );

        #[cfg(feature = "_cuda")]
        if let Err(gpu_err) = result {
            if self.is_using_gpu() {
                let accel = env_acceleration_mode_lossy();
                if accel == AccelerationMode::ForceGpu {
                    anyhow::bail!(
                        "GPU encoding failed with --force-gpu. \
                         Not enough GPU memory for batch size {batch} and document length. \
                         Try reducing the batch size or use auto mode to allow CPU fallback.\n\
                         \nCaused by: {gpu_err}",
                        batch = self
                            .batch_size
                            .unwrap_or(crate::config::DEFAULT_BATCH_SIZE_GPU),
                    );
                }

                eprintln!(
                    "\n⚠️  GPU encoding failed, falling back to CPU. \
                     This is usually caused by insufficient GPU memory for the batch size.\n"
                );

                self.rebuild_model_for_cpu()?;

                let force_cpu = next_plaid::is_force_cpu();
                let config = IndexConfig {
                    force_cpu,
                    ..Default::default()
                };
                let update_config = UpdateConfig {
                    force_cpu,
                    ..Default::default()
                };

                return run_chunk_pipeline(
                    self.model().clone(),
                    sorted_units,
                    ChunkPipelineConfig {
                        index_chunk_size,
                        pool_factor,
                        index_path,
                        config,
                        update_config,
                        pb,
                    },
                );
            }

            return Err(gpu_err);
        }

        result
    }

    /// Get the path to the index directory
    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    /// Compute the effective pool factor based on the number of units to encode.
    ///
    /// For large runs, forced pooling materially reduces downstream index cost.
    fn resolve_pool_factor(&self, num_units: usize) -> Option<usize> {
        if num_units > LARGE_BATCH_THRESHOLD {
            Some(LARGE_BATCH_POOL_FACTOR)
        } else {
            self.pool_factor
        }
    }

    /// Reconstruct IndexState from the filtering database.
    ///
    /// This is used when state.json is missing/empty but the index exists.
    /// Queries the filtering DB for all indexed file paths and rebuilds the state
    /// by computing hashes and mtimes for files that still exist on disk.
    ///
    /// Files that no longer exist are scheduled for deletion from the index.
    fn reconstruct_state_from_filtering_db(&self, index_path: &str) -> Result<IndexState> {
        // Get all metadata from filtering DB
        let all_metadata = filtering::get(index_path, None, &[], None)?;

        if all_metadata.is_empty() {
            anyhow::bail!("Filtering database is empty, cannot reconstruct state");
        }

        // Extract unique file paths from metadata
        let mut unique_files: HashSet<PathBuf> = HashSet::new();
        for meta in &all_metadata {
            if let Some(file_str) = meta.get("file").and_then(|v| v.as_str()) {
                unique_files.insert(PathBuf::from(file_str));
            }
        }

        if unique_files.is_empty() {
            anyhow::bail!("No file paths found in filtering database");
        }

        // Rebuild state by checking which files still exist
        let mut state = IndexState::default();

        for file_path in unique_files {
            let full_path = self.project_root.join(&file_path);

            // Only add files that still exist on disk
            if full_path.exists() {
                if let Ok(hash) = hash_file(&full_path) {
                    if let Ok(file_info) = FileInfo::probe(&full_path, hash) {
                        state.files.insert(file_path, file_info);
                    }
                }
            }
            // Files that don't exist will be detected as deleted in incremental_update
        }

        Ok(state)
    }

    /// Reconcile document count mismatch between filtering DB and vector index.
    ///
    /// This handles the case where the counts don't match, typically due to
    /// interrupted indexing operations.
    ///
    /// Strategy:
    /// - If filtering has MORE docs than vector index: delete orphan entries from filtering
    /// - If vector index has MORE docs than filtering: accept it (orphan embeddings don't affect search)
    fn reconcile_document_counts(
        &self,
        index_path: &str,
        filtering_count: usize,
        vector_count: usize,
    ) -> Result<()> {
        eprintln!(
            "⚠️  Index/DB desync: SQLite has {} entries, vector index has {} docs",
            filtering_count, vector_count
        );
        if filtering_count > vector_count {
            // Filtering DB has orphan entries (docs without embeddings).
            // The vector index uses sequential IDs starting from 0, so any
            // `_subset_` ID >= vector_count is an orphan. Push the filter into
            // SQL so we don't materialize every row's metadata just to find a
            // few stray IDs.
            let orphan_ids = filtering::where_condition(
                index_path,
                "_subset_ >= ?",
                &[serde_json::json!(vector_count as i64)],
            )?;

            if !orphan_ids.is_empty() {
                // Delete orphan entries from filtering DB
                filtering::delete(index_path, &orphan_ids)?;
            }
        }
        // If vector_count > filtering_count, the orphan embeddings don't affect search
        // results since filtering is used to select which docs to return.
        // We can safely proceed with incremental update.

        Ok(())
    }

    /// Automatically repair sync issues between the vector index and metadata DB.
    ///
    /// Handles two cases:
    /// 1. DB has more records than index: Delete extra DB records (IDs >= index count)
    /// 2. Index has more documents than DB: Delete extra documents from index (IDs >= DB count)
    ///
    /// Returns Ok(true) if repair was performed, Ok(false) if no repair needed.
    fn repair_index_db_sync(&self, index_dir: &Path) -> Result<bool> {
        let index_path = index_dir.to_str().unwrap();

        // Check if both exist
        if !index_dir.join("metadata.json").exists() {
            return Ok(false); // No index yet
        }
        if !filtering::exists(index_path) {
            return Ok(false); // No DB yet
        }

        let index_metadata =
            Metadata::load_from_path(index_dir).context("Failed to load index metadata")?;
        let db_count = filtering::count(index_path).context("Failed to get DB count")?;

        let index_count = index_metadata.num_documents;

        if index_count == db_count {
            return Ok(false); // Already in sync
        }

        eprintln!(
            "⚠️  Index/DB desync detected: index has {} docs, DB has {} records",
            index_count, db_count
        );

        if db_count > index_count {
            // DB has extra records - delete them
            let extra_ids: Vec<i64> = (index_count as i64..db_count as i64).collect();
            filtering::delete(index_path, &extra_ids)
                .context("Failed to delete extra DB records")?;
            eprintln!("🔧 Deleted {} orphan DB records", extra_ids.len());
        } else {
            // Index has extra documents - delete them
            let extra_ids: Vec<i64> = (db_count as i64..index_count as i64).collect();
            delete_from_index(&extra_ids, index_path)
                .context("Failed to delete extra index documents")?;
            eprintln!("🔧 Deleted {} orphan index documents", extra_ids.len());
        }

        // Verify repair succeeded
        let new_index_metadata = Metadata::load_from_path(index_dir)
            .context("Failed to reload index metadata after repair")?;
        let new_db_count =
            filtering::count(index_path).context("Failed to get DB count after repair")?;

        if new_index_metadata.num_documents != new_db_count {
            anyhow::bail!(
                "Repair failed: index still has {} documents but DB has {} records",
                new_index_metadata.num_documents,
                new_db_count
            );
        }

        Ok(true)
    }

    /// Single entry point for indexing.
    /// - Creates index if none exists
    /// - Updates incrementally if files changed
    /// - Full rebuild if `force = true`
    /// - Full rebuild if the on-disk index format is incompatible
    pub fn index(&mut self, languages: Option<&[Language]>, force: bool) -> Result<UpdateStats> {
        let _lock = acquire_index_lock(&self.index_dir)?;
        self.run_indexing(languages, force)
    }

    /// Shared indexing dispatch, run while the index lock is held by the caller.
    ///
    /// Routing:
    /// - forced or index-format change → atomic `full_rebuild` (protects a working index);
    /// - a fresh build, or resuming an interrupted resumable build → `build_resumable`
    ///   (checkpoints per batch so interruptions keep their progress);
    /// - corrupted filtering DB → atomic `full_rebuild`;
    /// - otherwise → `incremental_update`.
    fn run_indexing(&mut self, languages: Option<&[Language]>, force: bool) -> Result<UpdateStats> {
        // Clean up any leftover temp/old dirs from previous failed full rebuilds
        let _ = std::fs::remove_dir_all(self.index_dir.join("index.tmp"));
        let _ = std::fs::remove_dir_all(self.index_dir.join("index.old"));

        // Fresh git worktree? Seed from a sibling's index so we re-embed only the diff
        // instead of rebuilding from scratch. No-op for non-worktree projects.
        self.maybe_seed_from_worktree(force);

        let state = IndexState::load(&self.index_dir)?;
        let index_dir = get_vector_index_path(&self.index_dir);
        let index_path = index_dir.to_str().unwrap();
        let index_exists = index_dir.join("metadata.json").exists();
        let filtering_exists = filtering::exists(index_path);
        let building = self.index_dir.join(BUILDING_MARKER).exists();

        // Rebuild only when the on-disk index format is incompatible with this
        // binary. Gating on the CLI version instead meant every routine release
        // discarded every index and re-embedded the entire project.
        //
        // A missing/empty state.json also deserializes to format version 0, but
        // that case must stay on the cheap reconstruct-from-filtering-DB path
        // below rather than a full re-embed — hence the non-empty-files guard
        // (a genuine legacy-format state always has tracked files).
        let format_mismatch = index_exists
            && !state.files.is_empty()
            && state.index_format_version != INDEX_FORMAT_VERSION;

        // Forced or index-format change: clean atomic rebuild. Drop any in-progress
        // resumable-build marker so we don't try to resume an index we're discarding.
        if force || format_mismatch {
            let _ = std::fs::remove_file(self.index_dir.join(BUILDING_MARKER));
            return self.full_rebuild(languages);
        }

        // Fresh first build, or resume of an interrupted resumable build: build directly
        // into the real index dir and checkpoint state per batch, so an interrupted run
        // (e.g. an agent command timeout) keeps its progress instead of restarting.
        if building || !index_exists {
            return self.build_resumable(languages);
        }

        // metadata.json exists but the filtering DB is missing or unreadable → corrupted.
        if !filtering_exists || filtering::count(index_path).is_err() {
            eprintln!("⚠️  Filtering database corrupted, rebuilding index...");
            return self.full_rebuild(languages);
        }

        // State is out of sync with index (e.g., state.json was deleted but index exists)
        // Try to reconstruct state from filtering DB instead of full rebuild
        let state = if state.files.is_empty() {
            match self.reconstruct_state_from_filtering_db(index_path) {
                Ok(reconstructed) => {
                    eprintln!(
                        "📋 Reconstructed state from index ({} files)",
                        reconstructed.files.len()
                    );
                    reconstructed.save(&self.index_dir)?;
                    reconstructed
                }
                Err(_) => {
                    // Failed to reconstruct, fall back to full rebuild
                    return self.full_rebuild(languages);
                }
            }
        } else {
            state
        };

        // Check if metadata DB is in sync with vector index
        // If document counts don't match, try to reconcile instead of full rebuild
        if let Ok(metadata_count) = filtering::count(index_path) {
            if let Ok(index_metadata) = Metadata::load_from_path(&index_dir) {
                if metadata_count != index_metadata.num_documents {
                    // Try to reconcile the mismatch
                    match self.reconcile_document_counts(
                        index_path,
                        metadata_count,
                        index_metadata.num_documents,
                    ) {
                        Ok(()) => {
                            eprintln!(
                                "🔧 Reconciled index (filtering: {}, vector: {})",
                                metadata_count, index_metadata.num_documents
                            );
                        }
                        Err(_) => {
                            // Failed to reconcile, fall back to full rebuild
                            return self.full_rebuild(languages);
                        }
                    }
                }
            }
        }

        self.incremental_update(&state, languages)
    }

    /// Non-blocking version of `index()` for use during search.
    /// Returns `Ok(None)` immediately if another process holds the lock,
    /// allowing the caller to search the existing index without waiting.
    pub fn try_index(
        &mut self,
        languages: Option<&[Language]>,
        force: bool,
    ) -> Result<Option<UpdateStats>> {
        let Some(_lock) = try_acquire_index_lock(&self.index_dir)? else {
            return Ok(None);
        };

        self.run_indexing(languages, force).map(Some)
    }

    /// Index only specific files (for filtered search).
    /// Only indexes files that are not already in the index or have changed.
    /// Returns the number of files that were indexed.
    pub fn index_specific_files(&mut self, files: &[PathBuf]) -> Result<UpdateStats> {
        if files.is_empty() {
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged: 0,
                skipped: 0,
            });
        }

        let _lock = acquire_index_lock(&self.index_dir)?;
        let state = IndexState::load(&self.index_dir)?;
        let index_dir = get_vector_index_path(&self.index_dir);
        let index_path = index_dir.to_str().unwrap();

        // Build gitignore matcher to filter out gitignored files
        // This ensures index_specific_files respects .gitignore like scan_files does
        let gitignore = {
            let mut builder = GitignoreBuilder::new(&self.project_root);
            let gitignore_path = self.project_root.join(".gitignore");
            if gitignore_path.exists() {
                let _ = builder.add(&gitignore_path);
            }
            builder.build().ok()
        };

        // Load user-configured ignore/include overrides
        let config = crate::config::Config::load().unwrap_or_default();
        let extra_ignore = config.extra_ignore;
        let force_include = config.force_include;

        // Determine which files need indexing (new or changed)
        let mut files_added = Vec::new();
        let mut files_changed = Vec::new();
        let mut unchanged = 0;

        for path in files {
            // Security: skip files outside the project root (path traversal protection)
            if !is_within_project_root(&self.project_root, path) {
                continue;
            }

            let full_path = self.project_root.join(path);
            if !full_path.exists() {
                continue;
            }

            // Skip files in ignored directories (same filtering as scan_files)
            // Use the relative path so hidden-directory filtering doesn't reject
            // ancestor components of the project root itself.
            if should_ignore(path, &extra_ignore, &force_include) {
                continue;
            }

            // Skip gitignored files (same filtering as scan_files)
            // Use matched_path_or_any_parents to check if the file or any parent
            // directory is ignored (handles patterns like "/site" matching "site/...")
            if let Some(ref gi) = gitignore {
                if gi
                    .matched_path_or_any_parents(path, full_path.is_dir())
                    .is_ignore()
                {
                    continue;
                }
            }

            let hash = hash_file(&full_path)?;
            match state.files.get(path) {
                Some(info) if info.content_hash == hash => {
                    unchanged += 1;
                }
                Some(_) => {
                    // File exists in index but content changed
                    files_changed.push(path.clone());
                }
                None => {
                    // New file not in index
                    files_added.push(path.clone());
                }
            }
        }

        let files_to_index: Vec<PathBuf> = files_added
            .iter()
            .chain(files_changed.iter())
            .filter(|p| !state.ignored_files.contains(*p))
            .cloned()
            .collect();

        if files_to_index.is_empty() {
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged,
                skipped: 0,
            });
        }

        // Load or create state
        let mut new_state = state.clone();
        let mut new_units: Vec<CodeUnit> = Vec::new();

        // Progress bar for parsing
        let pb = ProgressBar::new(files_to_index.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb.set_message("Parsing files...");

        for parsed in parse_files_parallel(&self.project_root, &files_to_index, Some(&pb)) {
            if let Some(reason) = parsed.skip_reason {
                eprintln!("⚠️  {}", reason);
                new_state.ignored_files.insert(parsed.path);
                continue;
            }

            new_units.extend(parsed.units);
            if let Some(file_info) = parsed.file_info {
                new_state.files.insert(parsed.path, file_info);
            }
        }
        pb.finish_and_clear();

        if new_units.is_empty() {
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged,
                skipped: 0,
            });
        }

        // Build call graph
        build_call_graph(&mut new_units);

        // Ensure model is created before encoding (lazy initialization)
        self.ensure_model_created(new_units.len())?;

        let pb = ProgressBar::new(new_units.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb.set_message("Encoding...");

        // Create or update index
        std::fs::create_dir_all(index_path)?;

        let encode_batch_size = self.encode_batch_size.unwrap_or(DEFAULT_ENCODE_BATCH_SIZE);
        let index_chunk_size = self
            .index_chunk_size
            .unwrap_or(INDEX_CHUNK_SIZE)
            .max(encode_batch_size);

        // Compute effective pool factor based on batch size
        let pool_factor = self.resolve_pool_factor(new_units.len());

        // Delete changed files from index right before writing new data.
        // Deferred from earlier to minimize the window where data is missing
        // from the index (for concurrent readers and interrupt safety). Batched
        // into a single index rewrite — see delete_files_from_index / issue #116.
        delete_files_from_index(index_path, &files_changed)?;

        let sorted_units = prepare_units_for_encoding(&new_units, index_chunk_size);
        let was_interrupted = self.run_encoding_pipeline(
            &sorted_units,
            index_chunk_size,
            pool_factor,
            index_path,
            Some(&pb),
        )?;

        pb.finish_and_clear();

        if was_interrupted || is_interrupted() {
            // Don't save state — the index has partial data.
            anyhow::bail!("Indexing interrupted by user");
        }

        new_state.save(&self.index_dir)?;

        Ok(UpdateStats {
            added: files_added.len(),
            changed: files_changed.len(),
            deleted: 0,
            unchanged,
            skipped: 0,
        })
    }

    /// Scan files matching glob patterns (e.g., "*.py", "*.rs")
    /// Returns relative paths from project root
    pub fn scan_files_matching_patterns(&self, patterns: &[String]) -> Result<Vec<PathBuf>> {
        let (all_files, _skipped) = self.scan_files(None)?;

        if patterns.is_empty() {
            return Ok(all_files);
        }

        let filtered: Vec<PathBuf> = all_files
            .into_iter()
            .filter(|path| matches_glob_pattern(path, patterns))
            .collect();

        Ok(filtered)
    }

    /// Full rebuild (used when force=true or no index exists)
    /// If this project has no index yet but is a git worktree whose sibling has a usable
    /// index for the same model, seed this index from the sibling instead of rebuilding
    /// from scratch. After seeding, the normal incremental path re-embeds only the files
    /// that differ between the two branches.
    ///
    /// Best-effort: any failure (no git, no sibling index, copy error) is reported and the
    /// caller falls back to a full rebuild. Must be called while holding the index lock.
    fn maybe_seed_from_worktree(&self, force: bool) {
        // A forced rebuild or an already-populated index never seeds.
        if force
            || get_vector_index_path(&self.index_dir)
                .join("metadata.json")
                .exists()
        {
            return;
        }
        match self.try_seed_from_sibling_worktree() {
            Ok(true) | Ok(false) => {}
            Err(e) => eprintln!("⚠️  Worktree index seeding skipped ({e}); building from scratch"),
        }
    }

    /// Copy a usable sibling worktree's index into this project's index directory.
    /// Returns `Ok(true)` if an index was seeded, `Ok(false)` if no suitable sibling exists.
    fn try_seed_from_sibling_worktree(&self) -> Result<bool> {
        let candidates = worktree::seed_candidates(&self.project_root, &self.model_id)?;

        for candidate in candidates {
            let src_dir = &candidate.index_dir;
            // Validate the sibling holds a complete, format-compatible, non-dirty index that
            // isn't mid-build. Skip otherwise so we never seed from a half-built or stale store.
            let Some(src_state) = seed_source_state(src_dir) else {
                continue;
            };
            let src_vector = get_vector_index_path(src_dir);

            // Copy the vector/filtering store via a temp dir, then rename into place so an
            // interrupted copy never leaves a half-written index/ behind.
            let dest_vector = get_vector_index_path(&self.index_dir);
            let tmp = self.index_dir.join("index.tmp");
            if tmp.exists() {
                std::fs::remove_dir_all(&tmp)?;
            }
            worktree::copy_dir_all(&src_vector, &tmp)?;
            if dest_vector.exists() {
                std::fs::remove_dir_all(&dest_vector)?;
            }
            std::fs::rename(&tmp, &dest_vector)
                .context("Failed to move seeded index into place")?;

            // Persist state (save() restamps the version/format fields) and a
            // fresh project.json pointing at THIS worktree, not the source.
            src_state.save(&self.index_dir)?;
            ProjectMetadata::new(&self.project_root, &self.model_id).save(&self.index_dir)?;

            eprintln!(
                "📋 Seeded index from worktree {} ({} files); re-embedding only changed files",
                candidate.worktree_root.display(),
                src_state.files.len()
            );
            return Ok(true);
        }

        Ok(false)
    }

    /// Build (or resume building) the index directly into the real index directory,
    /// committing `state.json` after every batch of ~[`BUILD_CHECKPOINT_UNITS`] units.
    ///
    /// Unlike [`full_rebuild`], which encodes into a throwaway `index.tmp` and only persists
    /// on success, this commits incrementally — so an interrupted run (an agent's command
    /// timeout, Ctrl-C) keeps the batches it already finished. The presence of the
    /// [`BUILDING_MARKER`] file routes the next run back here to resume; it's removed once the
    /// build completes. Used for the first build and its resumes, never to rebuild over an
    /// already-complete index (that path stays on the atomic [`full_rebuild`]).
    fn build_resumable(&mut self, languages: Option<&[Language]>) -> Result<UpdateStats> {
        let index_dir = get_vector_index_path(&self.index_dir);
        std::fs::create_dir_all(&index_dir)?;
        let index_path = index_dir.to_str().unwrap().to_string();

        // Mark the build in progress so the next run resumes here even after some chunks
        // (and thus metadata.json) have been written.
        let marker = self.index_dir.join(BUILDING_MARKER);
        std::fs::write(&marker, "")?;

        // Resume: keep whatever a previous run already committed.
        let mut state = IndexState::load(&self.index_dir)?;

        // A prior run interrupted mid-batch may have left the vector index and filtering DB
        // slightly out of sync (a partially written chunk). Trim that before re-embedding so
        // we don't accumulate orphan documents across resumes.
        if index_dir.join("metadata.json").exists() && filtering::exists(&index_path) {
            let _ = self.repair_index_db_sync(&index_dir);
        }

        let (scanned, skipped) = self.scan_files(languages)?;
        let scanned_set: HashSet<PathBuf> = scanned.iter().cloned().collect();

        // Drop files that disappeared since a prior partial run.
        let stale: Vec<PathBuf> = state
            .files
            .keys()
            .filter(|p| !scanned_set.contains(*p))
            .cloned()
            .collect();
        delete_files_from_index(&index_path, &stale)?;
        for path in &stale {
            state.files.remove(path);
        }

        // Files still needing work: scanned, not already committed, not known-unparseable.
        let todo: Vec<PathBuf> = scanned
            .iter()
            .filter(|p| !state.files.contains_key(*p) && !state.ignored_files.contains(*p))
            .cloned()
            .collect();

        // A prior run already committed some files — resume from there rather than re-embedding
        // them. The committed files are excluded from `todo` above, so they are never recomputed.
        if !state.files.is_empty() {
            eprintln!(
                "📋 Resuming interrupted build: {} files already indexed, {} remaining",
                state.files.len(),
                todo.len()
            );
        }

        // Parse the remaining files (cheap relative to embedding) and build the call graph
        // over them so `called_by` is populated for this build's units.
        let pb = ProgressBar::new(todo.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb.set_message("Parsing files...");

        let mut all_units: Vec<CodeUnit> = Vec::new();
        let mut file_info: HashMap<PathBuf, FileInfo> = HashMap::new();
        for parsed in parse_files_parallel(&self.project_root, &todo, Some(&pb)) {
            if let Some(reason) = parsed.skip_reason {
                eprintln!("⚠️  {}", reason);
                state.ignored_files.insert(parsed.path);
                continue;
            }
            if let Some(fi) = parsed.file_info {
                file_info.insert(parsed.path.clone(), fi);
                all_units.extend(parsed.units);
            }
        }
        let parsing_interrupted = is_interrupted();
        pb.finish_and_clear();
        if parsing_interrupted {
            // Nothing was embedded this run; committed batches (if any) already persisted.
            state.dirty = false;
            state.save(&self.index_dir)?;
            anyhow::bail!("Indexing interrupted by user");
        }

        build_call_graph(&mut all_units);

        // Confirm once up front for very large codebases (auto-confirmed on non-TTY/agents).
        if !self.auto_confirm
            && all_units.len() > CONFIRMATION_THRESHOLD
            && !prompt_large_index_confirmation(all_units.len())
        {
            let _ = std::fs::remove_file(&marker);
            anyhow::bail!("Indexing cancelled by user");
        }

        // Regroup units per file so each batch contains whole files — that lets us checkpoint
        // `state` at file granularity after each committed batch.
        let mut units_by_file: HashMap<PathBuf, Vec<CodeUnit>> = HashMap::new();
        for unit in all_units {
            units_by_file
                .entry(unit.file.clone())
                .or_default()
                .push(unit);
        }

        let encode_batch_size = self.encode_batch_size.unwrap_or(DEFAULT_ENCODE_BATCH_SIZE);
        let index_chunk_size = self
            .index_chunk_size
            .unwrap_or(INDEX_CHUNK_SIZE)
            .max(encode_batch_size);

        let total_units: usize = units_by_file.values().map(|u| u.len()).sum();
        let already = state.files.len();
        let mut added = 0usize;

        // Files that parsed but yielded no code units (empty or import-only files) need no
        // embedding. Record them as done now — and persist — so they're not re-parsed on every
        // resume round (parity with full_rebuild, which also stores them). `file_info` only
        // holds this run's freshly-parsed files, none of which are in `state` yet.
        let mut recorded_empty = false;
        for (path, fi) in &file_info {
            if !units_by_file.contains_key(path) {
                state.files.insert(path.clone(), fi.clone());
                added += 1;
                recorded_empty = true;
            }
        }
        if recorded_empty {
            state.dirty = false;
            state.save(&self.index_dir)?;
        }

        // Encode in file-coherent batches of ~BUILD_CHECKPOINT_UNITS units, committing state
        // after each batch so interruptions keep finished work.
        let encode_pb = ProgressBar::new(total_units as u64);
        encode_pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        encode_pb.enable_steady_tick(std::time::Duration::from_millis(100));
        encode_pb.set_message("Encoding...");

        let mut batch_files: Vec<PathBuf> = Vec::new();
        let mut batch_units: Vec<CodeUnit> = Vec::new();
        let mut interrupted = false;

        // Deterministic order for stable checkpoints/resumes.
        let mut ordered: Vec<PathBuf> = units_by_file.keys().cloned().collect();
        ordered.sort();

        for file in ordered {
            let units = units_by_file.remove(&file).unwrap_or_default();
            batch_units.extend(units);
            batch_files.push(file);

            if batch_units.len() >= BUILD_CHECKPOINT_UNITS {
                if self.flush_build_batch(
                    &index_path,
                    &batch_files,
                    &batch_units,
                    index_chunk_size,
                    Some(&encode_pb),
                )? {
                    interrupted = true;
                    break;
                }
                for f in batch_files.drain(..) {
                    if let Some(fi) = file_info.get(&f) {
                        state.files.insert(f, fi.clone());
                        added += 1;
                    }
                }
                batch_units.clear();
                state.dirty = false;
                state.save(&self.index_dir)?; // checkpoint
            }
        }

        // Final partial batch.
        if !interrupted && !batch_units.is_empty() {
            if self.flush_build_batch(
                &index_path,
                &batch_files,
                &batch_units,
                index_chunk_size,
                Some(&encode_pb),
            )? {
                interrupted = true;
            } else {
                for f in batch_files.drain(..) {
                    if let Some(fi) = file_info.get(&f) {
                        state.files.insert(f, fi.clone());
                        added += 1;
                    }
                }
                state.dirty = false;
                state.save(&self.index_dir)?;
            }
        }

        encode_pb.finish_and_clear();

        if interrupted {
            // Persist the committed batches; keep the marker so the next run resumes.
            state.dirty = false;
            state.save(&self.index_dir)?;
            anyhow::bail!("Indexing interrupted by user");
        }

        // Build complete: drop the marker and persist final metadata.
        state.dirty = false;
        state.save(&self.index_dir)?;
        ProjectMetadata::new(&self.project_root, &self.model_id).save(&self.index_dir)?;
        let _ = std::fs::remove_file(&marker);

        Ok(UpdateStats {
            added,
            changed: 0,
            deleted: stale.len(),
            unchanged: already,
            skipped,
        })
    }

    /// Encode one resumable-build batch into the index. Each file's stale entries are deleted
    /// first so re-running after a mid-batch interruption is idempotent (no duplicate docs).
    /// Returns `Ok(true)` if encoding was interrupted.
    fn flush_build_batch(
        &mut self,
        index_path: &str,
        batch_files: &[PathBuf],
        batch_units: &[CodeUnit],
        index_chunk_size: usize,
        pb: Option<&ProgressBar>,
    ) -> Result<bool> {
        if batch_units.is_empty() {
            return Ok(false);
        }
        // Idempotent resume: clear any partial docs a prior interrupted run wrote for these
        // files, in a single batched index rewrite (issue #116).
        delete_files_from_index(index_path, batch_files)?;
        self.ensure_model_created(batch_units.len())?;
        let pool_factor = self.resolve_pool_factor(batch_units.len());
        let sorted_units = prepare_units_for_encoding(batch_units, index_chunk_size);
        let was_interrupted = self.run_encoding_pipeline(
            &sorted_units,
            index_chunk_size,
            pool_factor,
            index_path,
            pb,
        )?;
        Ok(was_interrupted || is_interrupted())
    }

    fn full_rebuild(&mut self, languages: Option<&[Language]>) -> Result<UpdateStats> {
        // A clean atomic rebuild supersedes any in-progress resumable build.
        let _ = std::fs::remove_file(self.index_dir.join(BUILDING_MARKER));

        let index_path = get_vector_index_path(&self.index_dir);
        let temp_path = self.index_dir.join("index.tmp");
        let old_path = self.index_dir.join("index.old");

        // Clean any leftover temp/old dirs from previous failed attempts
        if temp_path.exists() {
            std::fs::remove_dir_all(&temp_path)?;
        }
        if old_path.exists() {
            std::fs::remove_dir_all(&old_path)?;
        }

        let (files, skipped) = self.scan_files(languages)?;
        let mut state = IndexState::default();
        let mut all_units: Vec<CodeUnit> = Vec::new();

        // Progress bar for parsing files
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb.set_message("Parsing files...");

        // Extract units from all files
        for parsed in parse_files_parallel(&self.project_root, &files, Some(&pb)) {
            if let Some(reason) = parsed.skip_reason {
                eprintln!("⚠️  {}", reason);
                state.ignored_files.insert(parsed.path);
                continue;
            }

            all_units.extend(parsed.units);
            if let Some(file_info) = parsed.file_info {
                state.files.insert(parsed.path, file_info);
            }
        }
        let parsing_interrupted = is_interrupted();
        pb.finish_and_clear();

        if parsing_interrupted {
            eprintln!("⚠️  Indexing interrupted during parsing. Partial index not saved.");
            anyhow::bail!("Indexing interrupted by user");
        }

        // Build call graph to populate called_by
        build_call_graph(&mut all_units);

        // Prompt for confirmation if indexing a large codebase
        if !self.auto_confirm
            && all_units.len() > CONFIRMATION_THRESHOLD
            && !prompt_large_index_confirmation(all_units.len())
        {
            anyhow::bail!("Indexing cancelled by user");
        }

        let was_interrupted = if !all_units.is_empty() {
            // Ensure model is created before encoding (lazy initialization)
            self.ensure_model_created(all_units.len())?;

            #[cfg(feature = "_cuda")]
            if !crate::onnx_runtime::is_cudnn_available()
                && std::env::var("_COLGREP_CUDNN_NOTICE").is_err()
            {
                std::env::set_var("_COLGREP_CUDNN_NOTICE", "1");
                eprintln!("📂 cuDNN not found, encoding will use CPU.");
            }

            // Build new index in temp directory to avoid destroying the old one
            self.write_index_impl(&all_units, true, Some(&temp_path))?
        } else {
            false
        };

        if was_interrupted {
            // Clean up temp dir — the old index is untouched
            let _ = std::fs::remove_dir_all(&temp_path);
            anyhow::bail!("Indexing interrupted by user");
        }

        // Atomic swap: replace old index with newly built one
        if all_units.is_empty() {
            // No files to index — just remove the old index if it exists
            if index_path.exists() {
                std::fs::remove_dir_all(&index_path)?;
            }
        } else {
            if index_path.exists() {
                std::fs::rename(&index_path, &old_path)
                    .context("Failed to move old index aside")?;
            }
            if let Err(e) = std::fs::rename(&temp_path, &index_path) {
                // Try to restore old index
                if old_path.exists() && !index_path.exists() {
                    let _ = std::fs::rename(&old_path, &index_path);
                }
                return Err(anyhow::anyhow!(
                    "Failed to move new index into place: {}",
                    e
                ));
            }
            if old_path.exists() {
                let _ = std::fs::remove_dir_all(&old_path);
            }
        }

        // Save state and project metadata only on successful completion
        state.save(&self.index_dir)?;
        ProjectMetadata::new(&self.project_root, &self.model_id).save(&self.index_dir)?;

        Ok(UpdateStats {
            added: files.len(),
            changed: 0,
            deleted: 0,
            unchanged: 0,
            skipped,
        })
    }

    /// Incremental update (only re-index changed files)
    fn incremental_update(
        &mut self,
        old_state: &IndexState,
        languages: Option<&[Language]>,
    ) -> Result<UpdateStats> {
        let plan = self.compute_update_plan(old_state, languages)?;
        let index_dir = get_vector_index_path(&self.index_dir);
        let index_path = index_dir.to_str().unwrap();

        // Repair desync only if the previous run was interrupted mid-write.
        if old_state.dirty {
            if let Err(e) = self.repair_index_db_sync(&index_dir) {
                eprintln!("⚠️  Repair failed: {}, falling back to full rebuild", e);
                return self.full_rebuild(languages);
            }
        }

        // 0. Clean up orphaned entries (files in index but not on disk).
        // This handles directory deletion/rename and any inconsistencies, but it
        // queries every distinct file in the metadata DB and stats each one — too
        // expensive to run on every search. Deletions in the plan already cover the
        // common case; the periodic trigger is a safety net for true desyncs.
        let should_cleanup = !plan.deleted.is_empty()
            || (old_state.search_count > 0 && old_state.search_count.is_multiple_of(50));
        let orphaned_deleted = if should_cleanup {
            self.cleanup_orphaned_entries(index_path)?
        } else {
            0
        };

        // Nothing to do
        if plan.added.is_empty()
            && plan.changed.is_empty()
            && plan.deleted.is_empty()
            && orphaned_deleted == 0
        {
            // Nothing to re-embed. Still persist if either (a) some files' stat
            // drifted without a content change (refresh them so we don't re-hash
            // every run), or (b) the previous run left the index dirty — the
            // repair above already resynced the store, so clear the flag now.
            // Returning without persisting would force a re-hash or a costly
            // repair on every future run even though nothing is wrong (#115).
            if !plan.touched.is_empty() || old_state.dirty {
                let mut state = old_state.clone();
                for (path, info) in &plan.touched {
                    state.files.insert(path.clone(), info.clone());
                }
                state.dirty = false;
                state.save(&self.index_dir)?;
            }
            return Ok(UpdateStats {
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged: plan.unchanged,
                skipped: 0,
            });
        }

        let mut state = old_state.clone();
        // Refresh stat-drifted-but-unchanged files so the fast path applies next run.
        for (path, info) in &plan.touched {
            state.files.insert(path.clone(), info.clone());
        }

        if !plan.deleted.is_empty() || !plan.changed.is_empty() || !plan.added.is_empty() {
            state.dirty = true;
            state.save(&self.index_dir)?;
        }

        // 1. Delete chunks for deleted files (safe — not re-adding these). Batched into a
        //    single index rewrite — see delete_files_from_index / issue #116.
        delete_files_from_index(index_path, &plan.deleted)?;

        // Remove deleted files from state
        for path in &plan.deleted {
            state.files.remove(path);
        }

        // Also clean state of any files that no longer exist on disk
        // (handles directory deletion/rename and any state inconsistencies)
        let stale_paths: Vec<PathBuf> = state
            .files
            .keys()
            .filter(|p| !self.project_root.join(p).exists())
            .cloned()
            .collect();
        for path in stale_paths {
            state.files.remove(&path);
        }

        // 2. Index new/changed files (skip previously ignored files)
        let files_to_index: Vec<PathBuf> = plan
            .added
            .iter()
            .chain(plan.changed.iter())
            .filter(|p| !state.ignored_files.contains(*p))
            .cloned()
            .collect();

        let mut new_units: Vec<CodeUnit> = Vec::new();

        // Progress bar for parsing (only if there are files to index)
        let pb = if !files_to_index.is_empty() {
            let pb = ProgressBar::new(files_to_index.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                    .unwrap()
                    .progress_chars("█▓░"),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb.set_message("Parsing files...");
            Some(pb)
        } else {
            None
        };

        let mut skipped_files: Vec<PathBuf> = Vec::new();
        for parsed in parse_files_parallel(&self.project_root, &files_to_index, pb.as_ref()) {
            if let Some(reason) = parsed.skip_reason {
                eprintln!("⚠️  {}", reason);
                state.files.remove(&parsed.path);
                state.ignored_files.insert(parsed.path.clone());
                skipped_files.push(parsed.path);
                continue;
            }

            new_units.extend(parsed.units);
            if let Some(file_info) = parsed.file_info {
                state.files.insert(parsed.path, file_info);
            }
        }
        let parsing_interrupted = is_interrupted();
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }

        if parsing_interrupted {
            // Don't save state — the index may be inconsistent (changed files were
            // deleted from the index but not re-added). Next run will detect the
            // mismatch and re-index properly.
            anyhow::bail!("Indexing interrupted by user");
        }

        // Delete stale index entries for skipped files that were previously indexed
        // (e.g., files that became unreadable due to invalid UTF-8). Batched into one rewrite.
        let stale_skipped: Vec<PathBuf> = skipped_files
            .iter()
            .filter(|p| plan.changed.contains(p))
            .cloned()
            .collect();
        let _ = delete_files_from_index(index_path, &stale_skipped);

        // 3. Add new units to index
        let mut was_interrupted = false;
        if !new_units.is_empty() {
            // Build call graph for new units
            build_call_graph(&mut new_units);

            // Prompt for confirmation if indexing a large number of new units
            if !self.auto_confirm
                && new_units.len() > CONFIRMATION_THRESHOLD
                && !prompt_large_index_confirmation(new_units.len())
            {
                anyhow::bail!("Indexing cancelled by user");
            }

            // Ensure model is created before encoding (lazy initialization)
            self.ensure_model_created(new_units.len())?;

            // Progress bar for encoding
            let pb = ProgressBar::new(new_units.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("█▓░"),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb.set_message("Encoding...");

            let encode_batch_size = self.encode_batch_size.unwrap_or(DEFAULT_ENCODE_BATCH_SIZE);
            let index_chunk_size = self
                .index_chunk_size
                .unwrap_or(INDEX_CHUNK_SIZE)
                .max(encode_batch_size);

            // Compute effective pool factor based on batch size
            let pool_factor = self.resolve_pool_factor(new_units.len());

            // Delete changed files from index right before writing new data.
            // Deferred from earlier to minimize the window where data is missing
            // from the index (for concurrent readers and interrupt safety). Batched
            // into a single index rewrite — see delete_files_from_index / issue #116.
            delete_files_from_index(index_path, &plan.changed)?;

            let sorted_units = prepare_units_for_encoding(&new_units, index_chunk_size);
            let pipeline_interrupted = self.run_encoding_pipeline(
                &sorted_units,
                index_chunk_size,
                pool_factor,
                index_path,
                Some(&pb),
            )?;
            was_interrupted |= pipeline_interrupted;

            pb.finish_and_clear();
        }

        if was_interrupted || is_interrupted() {
            // Don't save state — the index has partial data. Next run will detect
            // the mismatch and re-index the missing files.
            anyhow::bail!("Indexing interrupted by user");
        }

        state.dirty = false;
        state.save(&self.index_dir)?;

        Ok(UpdateStats {
            added: plan.added.len(),
            changed: plan.changed.len(),
            deleted: plan.deleted.len(),
            unchanged: plan.unchanged,
            skipped: 0,
        })
    }

    fn scan_files(&self, languages: Option<&[Language]>) -> Result<(Vec<PathBuf>, usize)> {
        // Load user-configured ignore/include overrides from persistent config
        let config = crate::config::Config::load().unwrap_or_default();
        let extra_ignore = config.extra_ignore.clone();
        let force_include = config.force_include.clone();

        let project_root = self.project_root.clone();
        let walker = WalkBuilder::new(&self.project_root)
            .hidden(false) // Handle hidden files manually in should_ignore (with .github exception)
            .git_ignore(true)
            .follow_links(false) // Explicitly prevent symlink traversal outside project
            .filter_entry(move |entry| {
                // Only apply ignore rules to path components relative to the project root.
                // The project root itself is always trusted (the user explicitly chose it),
                // so hidden-directory filtering must not reject ancestor path components.
                match entry.path().strip_prefix(&project_root) {
                    Ok(rel) if rel.as_os_str().is_empty() => true, // root entry itself
                    Ok(rel) => !should_ignore(rel, &extra_ignore, &force_include),
                    Err(_) => !should_ignore(entry.path(), &extra_ignore, &force_include), // fallback (shouldn't happen)
                }
            })
            .build();

        let mut files = Vec::new();
        let mut skipped = 0;

        for entry in walker.filter_map(|e| e.ok()) {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }

            let path = entry.path();

            // Skip files that are too large
            if is_file_too_large(path) {
                skipped += 1;
                continue;
            }

            let lang = match detect_language(path) {
                Some(l) => l,
                None => continue,
            };

            if languages.map(|ls| ls.contains(&lang)).unwrap_or(true) {
                if let Ok(rel_path) = path.strip_prefix(&self.project_root) {
                    // Verify the file is truly within the project root (handles symlink escapes)
                    if is_within_project_root(&self.project_root, rel_path) {
                        files.push(rel_path.to_path_buf());
                    } else {
                        skipped += 1;
                    }
                }
            }
        }

        Ok((files, skipped))
    }
}

/// Check if a file exceeds the maximum size limit
fn is_file_too_large(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(meta) => meta.len() > MAX_FILE_SIZE,
        Err(_) => false, // If we can't read metadata, let it fail later
    }
}

/// Check if a path is within the project root directory.
/// This prevents path traversal attacks (e.g., ../../../etc/passwd).
/// The check is done by canonicalizing both paths and verifying
/// the resolved path starts with the project root.
fn is_within_project_root(project_root: &Path, relative_path: &Path) -> bool {
    // Check for obvious path traversal patterns first (fast path)
    let path_str = relative_path.to_string_lossy();
    if path_str.contains("..") {
        // Could be a traversal attempt - do full canonicalization check
        let full_path = project_root.join(relative_path);
        match full_path.canonicalize() {
            Ok(canonical) => {
                // Canonicalize project root as well for accurate comparison
                match project_root.canonicalize() {
                    Ok(canonical_root) => canonical.starts_with(&canonical_root),
                    Err(_) => false,
                }
            }
            Err(_) => false, // If canonicalization fails, reject the path
        }
    } else {
        // No ".." in path, but still verify the path doesn't escape via symlinks
        let full_path = project_root.join(relative_path);
        if !full_path.exists() {
            return true; // Non-existent paths will be skipped later anyway
        }
        match (full_path.canonicalize(), project_root.canonicalize()) {
            (Ok(canonical), Ok(canonical_root)) => canonical.starts_with(&canonical_root),
            _ => false,
        }
    }
}

/// Directories and patterns to always ignore (even without .gitignore)
const IGNORED_DIRS: &[&str] = &[
    // Version control
    ".git",
    ".svn",
    ".hg",
    // Dependencies
    "node_modules",
    "vendor",
    "third_party",
    "third-party",
    "external",
    // Build outputs.
    "target",
    "build",
    "dist",
    "out",
    "bin",
    "obj",
    // Python
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "env",
    ".tox",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    ".eggs",
    // JavaScript/TypeScript
    ".next",
    ".nuxt",
    ".output",
    ".cache",
    ".parcel-cache",
    ".turbo",
    // Rust
    "target",
    // Go
    "go.sum",
    // Java
    ".gradle",
    ".m2",
    // IDE/Editor
    ".idea",
    ".vscode",
    ".vs",
    "*.xcworkspace",
    "*.xcodeproj",
    // Test/Coverage
    "coverage",
    ".coverage",
    "htmlcov",
    ".nyc_output",
    // Misc
    "tmp",
    "temp",
    "logs",
    ".DS_Store",
];

/// Hidden directories that should be indexed (exceptions to hidden file filtering)
const ALLOWED_HIDDEN_DIRS: &[&str] = &[
    ".github",
    ".gitlab",
    ".circleci",
    ".buildkite",
    ".claude",
    ".claude-plugin",
];

/// Hidden files that should be indexed (exceptions to hidden file filtering)
const ALLOWED_HIDDEN_FILES: &[&str] = &[".gitlab-ci.yml", ".gitlab-ci.yaml", ".travis.yml"];

/// Check if a project root path contains an ignored directory in its path.
/// This is used to provide better error messages when indexing fails.
/// Returns Some(matched_pattern) if the path contains an ignored directory, None otherwise.
pub fn path_contains_ignored_dir(path: &Path) -> Option<&'static str> {
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            let name_str = name.to_string_lossy();
            for pattern in IGNORED_DIRS {
                // Only check exact matches (not suffix patterns like *.egg-info)
                if !pattern.starts_with('*') && name_str == *pattern {
                    return Some(pattern);
                }
            }
        }
    }
    None
}

/// Check if a path should be ignored, considering user-configured overrides.
///
/// `extra_ignore` - additional patterns to ignore (on top of IGNORED_DIRS)
/// `force_include` - patterns that override ignore rules (both default and extra)
///
/// Force-include patterns match against the full relative path as well as individual
/// components, supporting both directory names (e.g., ".vscode") and path patterns
/// (e.g., "vendor/internal").
fn should_ignore(path: &Path, extra_ignore: &[String], force_include: &[String]) -> bool {
    let path_str = path.to_string_lossy();

    // Check force-include against the full path first
    for pattern in force_include {
        if let Some(suffix) = pattern.strip_prefix('*') {
            if path_str.ends_with(suffix) {
                return false;
            }
        } else if path_str == *pattern || path_str.starts_with(&format!("{}/", pattern)) {
            return false;
        }
    }

    // Check each component of the path
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            let name_str = name.to_string_lossy();

            // Check force-include for this component
            let force_included = force_include.iter().any(|p| {
                if let Some(suffix) = p.strip_prefix('*') {
                    name_str.ends_with(suffix)
                } else {
                    name_str.as_ref() == p
                }
            });
            if force_included {
                continue; // Skip ignore checks for this component
            }

            // Skip hidden files/directories (starting with .) except allowed ones
            if name_str.starts_with('.')
                && !ALLOWED_HIDDEN_DIRS.contains(&name_str.as_ref())
                && !ALLOWED_HIDDEN_FILES.contains(&name_str.as_ref())
            {
                return true;
            }

            // Check default ignore patterns
            for pattern in IGNORED_DIRS {
                if let Some(suffix) = pattern.strip_prefix('*') {
                    if name_str.ends_with(suffix) {
                        return true;
                    }
                } else if name_str == *pattern {
                    return true;
                }
            }

            // Check user-configured extra ignore patterns
            for pattern in extra_ignore {
                if let Some(suffix) = pattern.strip_prefix('*') {
                    if name_str.ends_with(suffix) {
                        return true;
                    }
                } else if name_str.as_ref() == pattern {
                    return true;
                }
            }
        }
    }
    false
}

impl IndexBuilder {
    fn compute_update_plan(
        &self,
        state: &IndexState,
        languages: Option<&[Language]>,
    ) -> Result<UpdatePlan> {
        let (current_files, _skipped) = self.scan_files(languages)?;
        let current_set: HashSet<_> = current_files.iter().cloned().collect();

        let mut plan = UpdatePlan::default();

        for path in &current_files {
            // Skip files that previously failed to parse (e.g. invalid UTF-8)
            if state.ignored_files.contains(path) {
                continue;
            }
            let full_path = self.project_root.join(path);

            // Fast path: if mtime AND size are both unchanged, skip expensive
            // content hashing. Without this gate every search reads and hashes
            // every file in the repo before returning results. mtime is compared
            // at nanosecond precision and paired with size so a same-second edit
            // can't slip through.
            let current_stat = file_stat(&full_path).ok();
            if let (Some(info), Some((current_mtime, current_size))) =
                (state.files.get(path), current_stat)
            {
                if info.mtime == current_mtime && info.size == current_size {
                    plan.unchanged += 1;
                    continue;
                }
            }

            let hash = match hash_file(&full_path) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("⚠️  Skipping {} ({})", full_path.display(), e);
                    continue;
                }
            };

            match state.files.get(path) {
                Some(info) if info.content_hash == hash => {
                    // Content is identical but the stat drifted (e.g. `touch`,
                    // branch switch). Record the refreshed FileInfo so it gets
                    // persisted and the stat-only fast path applies next run.
                    if let Some((current_mtime, current_size)) = current_stat {
                        if info.mtime != current_mtime || info.size != current_size {
                            plan.touched.push((
                                path.clone(),
                                FileInfo {
                                    content_hash: hash,
                                    mtime: current_mtime,
                                    size: current_size,
                                },
                            ));
                        }
                    }
                    plan.unchanged += 1;
                }
                Some(_) => plan.changed.push(path.clone()),
                None => plan.added.push(path.clone()),
            }
        }

        for path in state.files.keys() {
            if !current_set.contains(path) {
                plan.deleted.push(path.clone());
            }
        }

        Ok(plan)
    }

    /// Clean up orphaned entries: files in index but not on disk
    /// This handles directory deletion/rename and any state inconsistencies
    fn cleanup_orphaned_entries(&self, index_path: &str) -> Result<usize> {
        // Pull only the distinct file paths from the metadata DB. The previous
        // implementation called `filtering::get` which streams every column of
        // every row (code text, embeddings metadata, etc.) on every search — a
        // tens-of-megabytes JSON deserialize on large indexes.
        let files = filtering::get_distinct_strings(index_path, "file").unwrap_or_default();

        // Collect every orphaned file, then remove them all in one batched index rewrite
        // rather than rewriting the whole index once per orphan (issue #116).
        let orphaned: Vec<PathBuf> = files
            .into_iter()
            .map(PathBuf::from)
            .filter(|rel| !self.project_root.join(rel).exists())
            .collect();

        delete_files_from_index(index_path, &orphaned)
    }

    #[allow(dead_code)]
    fn write_index(&mut self, units: &[CodeUnit]) -> Result<bool> {
        self.write_index_impl(units, false, None)
    }

    #[allow(dead_code)]
    fn write_index_with_progress(&mut self, units: &[CodeUnit]) -> Result<bool> {
        self.write_index_impl(units, true, None)
    }

    fn write_index_impl(
        &mut self,
        units: &[CodeUnit],
        show_progress: bool,
        target_index_path: Option<&Path>,
    ) -> Result<bool> {
        let index_dir = target_index_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| get_vector_index_path(&self.index_dir));
        let index_path = index_dir.to_str().unwrap();
        std::fs::create_dir_all(index_path)?;

        // Progress bar for encoding
        let pb = if show_progress {
            let pb = ProgressBar::new(units.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("█▓░"),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb.set_message("Encoding...");
            Some(pb)
        } else {
            None
        };

        let encode_batch_size = self.encode_batch_size.unwrap_or(DEFAULT_ENCODE_BATCH_SIZE);
        let index_chunk_size = self
            .index_chunk_size
            .unwrap_or(INDEX_CHUNK_SIZE)
            .max(encode_batch_size);

        // Compute effective pool factor based on batch size
        let pool_factor = self.resolve_pool_factor(units.len());

        let sorted_units = prepare_units_for_encoding(units, index_chunk_size);
        self.ensure_model_created(units.len())?;
        let was_interrupted = self.run_encoding_pipeline(
            &sorted_units,
            index_chunk_size,
            pool_factor,
            index_path,
            pb.as_ref(),
        )?;

        if let Some(pb) = pb {
            pb.finish_and_clear();
        }

        // Check if interrupted after all processing (including deferred interrupts)
        Ok(was_interrupted || is_interrupted())
    }

    /// Get index status (what would be updated)
    pub fn status(&self, languages: Option<&[Language]>) -> Result<UpdatePlan> {
        let state = IndexState::load(&self.index_dir)?;
        self.compute_update_plan(&state, languages)
    }
}

// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub unit: CodeUnit,
    pub score: f32,
}

/// Convert BRE (Basic Regular Expression) patterns to ERE (Extended Regular Expression).
///
/// This allows users to write grep-style patterns like "foo\|bar" which use BRE syntax,
/// and have them work correctly with Rust's regex crate which uses ERE syntax.
///
/// Conversions (applied only when safe):
/// - `\|` → `|` (alternation — always converted)
/// - `\+` → `+`, `\?` → `?` (quantifiers — only after a preceding atom)
/// - `\(` → `(`, `\)` → `)` (grouping — only when balanced as pairs)
/// - `\{` → `{`, `\}` → `}` (interval quantifiers — only when balanced and after an atom)
///
/// Conversions that would produce invalid ERE (unbalanced groups, leading quantifiers)
/// are skipped, keeping the original escape intact.
pub fn bre_to_ere(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let len = chars.len();

    // Phase 1: Find balanced \( ... \) and \{ ... \} pairs.
    // Only balanced pairs are safe to convert; unbalanced ones stay escaped
    // to avoid producing invalid ERE (e.g. `error\(4` staying `error\(4`
    // instead of becoming the invalid `error(4`).
    let mut convert = vec![false; len];

    fn mark_pairs(chars: &[char], convert: &mut [bool], open: char, close: char) {
        let len = chars.len();
        let mut stack: Vec<usize> = Vec::new();
        let mut i = 0;
        while i < len {
            if chars[i] == '\\' && i + 1 < len {
                match chars[i + 1] {
                    '\\' => {
                        i += 2;
                        continue;
                    }
                    c if c == open => {
                        stack.push(i);
                        i += 2;
                        continue;
                    }
                    c if c == close => {
                        if let Some(open_pos) = stack.pop() {
                            convert[open_pos] = true;
                            convert[i] = true;
                        }
                        i += 2;
                        continue;
                    }
                    _ => {
                        i += 2;
                        continue;
                    }
                }
            }
            i += 1;
        }
    }

    mark_pairs(&chars, &mut convert, '(', ')');
    mark_pairs(&chars, &mut convert, '{', '}');

    // Phase 2: Forward pass producing the ERE output.
    let mut result = String::with_capacity(pattern.len());
    let mut i = 0;
    let mut skip_close_brace = 0usize;

    while i < len {
        if chars[i] != '\\' || i + 1 >= len {
            result.push(chars[i]);
            i += 1;
            continue;
        }

        let next = chars[i + 1];
        match next {
            // Escaped backslash — keep both
            '\\' => {
                result.push('\\');
                result.push('\\');
                i += 2;
            }

            // Alternation — always safe
            '|' => {
                result.push('|');
                i += 2;
            }

            // `\+` and `\?` are a GNU BRE extension meaning quantifier; in
            // ERE (now the colgrep default) they are *literal*. The previous
            // code stripped the backslash unconditionally, so users typing
            // `\+` to match a literal `+` ended up running `+` as a
            // one-or-more quantifier against the preceding atom. Keep them
            // escaped so the regex engine sees a literal `+` / `?`.
            '+' | '?' => {
                result.push('\\');
                result.push(next);
                i += 2;
            }

            // Balanced grouping delimiters
            '(' | ')' if convert[i] => {
                result.push(next);
                i += 2;
            }

            // Balanced brace delimiters (interval quantifier)
            '{' if convert[i] => {
                if result.is_empty() {
                    skip_close_brace += 1;
                    result.push('\\');
                    result.push('{');
                } else {
                    result.push('{');
                }
                i += 2;
            }
            '}' if convert[i] => {
                if skip_close_brace > 0 {
                    skip_close_brace -= 1;
                    result.push('\\');
                    result.push('}');
                } else {
                    result.push('}');
                }
                i += 2;
            }

            // Everything else — keep escape as-is
            _ => {
                result.push('\\');
                result.push(next);
                i += 2;
            }
        }
    }
    result
}

/// Escape literal braces that are not valid regex quantifiers.
///
/// In regex, `{` and `}` are used for quantifiers like `{2}`, `{2,}`, `{2,4}`.
/// When users write patterns like `enum.*{` intending to match a literal brace,
/// the regex engine may try to parse `{` as a quantifier, causing issues.
///
/// This function converts non-quantifier braces to character class form `[{]` and `[}]`
/// which unambiguously matches literal braces.
///
/// Examples:
/// - `a{2,4}` → `a{2,4}` (valid quantifier, unchanged)
/// - `enum.*{` → `enum.*[{]` (literal brace)
/// - `\{[^}]*\}` → `[{][^}]*[}]` (literal braces for matching code blocks)
/// - `[{]` → `[{]` (already escaped, unchanged)
pub fn escape_literal_braces(pattern: &str) -> String {
    let mut result = String::with_capacity(pattern.len() + 10);
    let chars: Vec<char> = pattern.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut in_char_class = false;

    while i < len {
        let c = chars[i];

        // Track character class boundaries (but not escaped brackets)
        if c == '[' && (i == 0 || chars[i - 1] != '\\') {
            in_char_class = true;
            result.push(c);
            i += 1;
            continue;
        }
        if c == ']' && in_char_class && (i == 0 || chars[i - 1] != '\\') {
            in_char_class = false;
            result.push(c);
            i += 1;
            continue;
        }

        // Inside character class, braces are already literal
        if in_char_class {
            result.push(c);
            i += 1;
            continue;
        }

        // `\{` and `\}` are BRE-style escapes meaning "literal brace". The
        // caller (e.g. `bre_to_ere`) leaves them alone when unbalanced, so
        // by the time we see them here they are user-intended literals.
        // Convert directly to the unambiguous character-class form `[{]` /
        // `[}]` and skip both chars. Without this short-circuit the loop
        // pushes the `\` as-is and then mangles the `{` into `[{]`,
        // producing `\[{]` — a regex that matches the literal substring
        // `[{]`, not `{`.
        if c == '\\' && i + 1 < len {
            let next = chars[i + 1];
            if next == '{' || next == '}' {
                result.push('[');
                result.push(next);
                result.push(']');
                i += 2;
                continue;
            }
            // Other escape — keep both characters as-is.
            result.push('\\');
            result.push(next);
            i += 2;
            continue;
        }

        // Check for opening brace
        if c == '{' {
            // Look ahead to see if this is a valid quantifier: {n}, {n,}, {n,m}, {,m}
            if let Some(close_pos) = find_matching_brace(&chars, i) {
                let content: String = chars[i + 1..close_pos].iter().collect();
                if is_valid_quantifier(&content) {
                    // Valid quantifier - keep as-is
                    for ch in chars.iter().take(close_pos + 1).skip(i) {
                        result.push(*ch);
                    }
                    i = close_pos + 1;
                    continue;
                }
            }
            // Not a valid quantifier - escape the brace
            result.push_str("[{]");
            i += 1;
            continue;
        }

        // Check for closing brace (orphan, not part of quantifier)
        if c == '}' {
            // This is an orphan closing brace (quantifier closings are handled above)
            result.push_str("[}]");
            i += 1;
            continue;
        }

        result.push(c);
        i += 1;
    }

    result
}

/// Find the matching closing brace position, returns None if not found
fn find_matching_brace(chars: &[char], open_pos: usize) -> Option<usize> {
    for (i, ch) in chars.iter().enumerate().skip(open_pos + 1) {
        if *ch == '}' {
            return Some(i);
        }
        // Don't cross another opening brace
        if *ch == '{' {
            return None;
        }
    }
    None
}

/// Check if the content between braces is a valid regex quantifier
/// Valid forms: "n", "n,", "n,m", ",m" where n and m are non-negative integers
fn is_valid_quantifier(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }

    // Split by comma
    let parts: Vec<&str> = content.split(',').collect();

    match parts.len() {
        1 => {
            // {n} - must be a positive integer
            !parts[0].is_empty() && parts[0].chars().all(|c| c.is_ascii_digit())
        }
        2 => {
            // {n,} or {n,m} or {,m}
            let first_ok = parts[0].is_empty() || parts[0].chars().all(|c| c.is_ascii_digit());
            let second_ok = parts[1].is_empty() || parts[1].chars().all(|c| c.is_ascii_digit());
            // At least one part must have digits
            let has_digits = !parts[0].is_empty() || !parts[1].is_empty();
            first_ok && second_ok && has_digits
        }
        _ => false,
    }
}

/// Expand brace patterns like "*.{rs,md}" into ["*.rs", "*.md"]
/// Supports multiple brace groups: "{src,lib}/**/*.{rs,md}" expands to all combinations
fn expand_braces(pattern: &str) -> Vec<String> {
    // Find the first brace group
    let Some(start) = pattern.find('{') else {
        return vec![pattern.to_string()];
    };

    let Some(end) = pattern[start..].find('}') else {
        return vec![pattern.to_string()];
    };
    let end = start + end;

    // Extract prefix, alternatives, and suffix
    let prefix = &pattern[..start];
    let alternatives = &pattern[start + 1..end];
    let suffix = &pattern[end + 1..];

    // Split alternatives by comma (handle nested braces by counting)
    let mut results = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for c in alternatives.chars() {
        match c {
            '{' => {
                depth += 1;
                current.push(c);
            }
            '}' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                let expanded = format!("{}{}{}", prefix, current, suffix);
                // Recursively expand any remaining braces
                results.extend(expand_braces(&expanded));
                current.clear();
            }
            _ => current.push(c),
        }
    }

    // Don't forget the last alternative
    if !current.is_empty() || alternatives.ends_with(',') {
        let expanded = format!("{}{}{}", prefix, current, suffix);
        results.extend(expand_braces(&expanded));
    }

    results
}

/// Build a GlobSet from patterns for efficient matching
fn build_glob_set(patterns: &[String]) -> Option<GlobSet> {
    if patterns.is_empty() {
        return None;
    }

    // Expand brace patterns first
    let expanded_patterns: Vec<String> = patterns.iter().flat_map(|p| expand_braces(p)).collect();

    let mut builder = GlobSetBuilder::new();
    for pattern in &expanded_patterns {
        // Prepend **/ if pattern doesn't start with ** or /
        // This makes "*.rs" match files in any directory
        let normalized = if !pattern.starts_with("**/") && !pattern.starts_with('/') {
            format!("**/{}", pattern)
        } else {
            pattern.clone()
        };

        if let Ok(glob) = Glob::new(&normalized) {
            builder.add(glob);
        }
    }

    builder.build().ok()
}

/// Convert a glob pattern to a regex pattern
/// e.g., "*.test.ts" -> ".*\\.test\\.ts$"
/// e.g., "**/*.rs" -> ".*/.*\\.rs$"
fn glob_to_regex(pattern: &str) -> String {
    let mut regex = String::new();

    // If pattern doesn't start with ** or /, match anywhere in path
    if !pattern.starts_with("**/") && !pattern.starts_with('/') {
        regex.push_str("(^|.*/)")
    }

    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => {
                if chars.peek() == Some(&'*') {
                    chars.next(); // consume second *
                    if chars.peek() == Some(&'/') {
                        chars.next(); // consume /
                        regex.push_str("(.*/)?");
                    } else {
                        regex.push_str(".*");
                    }
                } else {
                    regex.push_str("[^/]*");
                }
            }
            '?' => regex.push('.'),
            '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
                regex.push('\\');
                regex.push(c);
            }
            _ => regex.push(c),
        }
    }

    regex.push('$');
    regex
}

/// Check if a string contains glob pattern metacharacters
fn is_glob_pattern(pattern: &str) -> bool {
    pattern.contains('*') || pattern.contains('?') || pattern.contains('[')
}

/// Convert a directory pattern (literal or glob) to a regex pattern
/// Supports both literal directory names and glob patterns:
/// - Literal: "vendor" -> "(^|/)vendor/" (matches any directory named vendor)
/// - Glob: "*/plugins" -> "^[^/]*/plugins/" (matches plugins under any single-level parent)
/// - Glob: "**/test_*" -> "(^|.*/)test_[^/]*/" (matches test_* directories at any depth)
fn dir_pattern_to_regex(pattern: &str) -> String {
    if is_glob_pattern(pattern) {
        // Handle as glob pattern - convert to regex for directory matching
        let mut regex = String::new();

        // Handle leading patterns
        let pattern = if let Some(stripped) = pattern.strip_prefix("**/") {
            // ** matches any depth including zero
            regex.push_str("(^|.*/)");
            stripped
        } else if let Some(stripped) = pattern.strip_prefix("*/") {
            // * matches exactly one directory level
            regex.push_str("^[^/]*/");
            stripped
        } else if let Some(stripped) = pattern.strip_prefix('/') {
            regex.push('^');
            stripped
        } else {
            // No leading slash - match at any position like literal directories
            regex.push_str("(^|/)");
            pattern
        };

        let mut chars = pattern.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '*' => {
                    if chars.peek() == Some(&'*') {
                        chars.next(); // consume second *
                        if chars.peek() == Some(&'/') {
                            chars.next(); // consume /
                            regex.push_str("(.*/)?");
                        } else {
                            regex.push_str(".*");
                        }
                    } else {
                        regex.push_str("[^/]*");
                    }
                }
                '?' => regex.push('.'),
                '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
                    regex.push('\\');
                    regex.push(c);
                }
                _ => regex.push(c),
            }
        }

        // Ensure pattern matches directories (with trailing slash)
        regex.push('/');
        regex
    } else {
        // Handle as literal directory name (current behavior)
        format!("(^|/){}/", regex::escape(pattern))
    }
}

/// Check if a file path matches any of the glob patterns
fn matches_glob_pattern(path: &Path, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }

    let Some(glob_set) = build_glob_set(patterns) else {
        return false;
    };

    glob_set.is_match(path)
}

pub struct Searcher {
    model: Colbert,
    index: MmapIndex,
    index_path: String,
}

impl Searcher {
    pub fn load(project_root: &Path, model_id: &str, model_path: &Path) -> Result<Self> {
        Self::load_with_quantized(project_root, model_id, model_path, false)
    }

    pub fn load_with_quantized(
        project_root: &Path,
        model_id: &str,
        model_path: &Path,
        quantized: bool,
    ) -> Result<Self> {
        let index_dir = get_index_dir_for_project(project_root, model_id)?;
        let vector_dir = get_vector_index_path(&index_dir);
        let index_path = vector_dir.to_str().unwrap().to_string();

        let acceleration_mode = env_acceleration_mode_lossy();
        let execution_provider = execution_provider_for_acceleration_mode(acceleration_mode);

        #[cfg(feature = "_cuda")]
        match acceleration_mode {
            AccelerationMode::ForceGpu => apply_acceleration_mode(AccelerationMode::ForceGpu),
            AccelerationMode::ForceCpu | AccelerationMode::Auto => {
                apply_acceleration_mode(AccelerationMode::ForceCpu)
            }
        }

        crate::onnx_runtime::ensure_onnx_runtime().context("Failed to initialize ONNX Runtime")?;

        #[cfg(feature = "_cuda")]
        if matches!(acceleration_mode, AccelerationMode::ForceGpu) {
            if !crate::onnx_runtime::is_cudnn_available() {
                anyhow::bail!("FORCE_GPU is set, but cuDNN was not initialized");
            }
            if !next_plaid_onnx::is_cuda_available() {
                anyhow::bail!(
                    "FORCE_GPU is set, but the CUDA execution provider was not initialized"
                );
            }
        }

        // Cap intra-op threads to avoid overhead on high-core-count systems
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(8)
            .min(crate::config::MAX_INTRA_OP_THREADS);

        // Suppress stderr during model loading to hide CoreML's harmless
        // "Context leak detected" warnings on macOS
        let model = crate::stderr::with_suppressed_stderr(|| {
            Colbert::builder(model_path)
                .with_quantized(quantized)
                .with_threads(num_threads)
                .with_execution_provider(execution_provider)
                .build()
        })
        .context("Failed to load ColBERT model")?;

        // Load index
        let index = MmapIndex::load(&index_path).context("Failed to load index")?;

        Ok(Self {
            model,
            index,
            index_path,
        })
    }

    /// Load a searcher from a specific index directory (for parent index use)
    pub fn load_from_index_dir(index_dir: &Path, model_path: &Path) -> Result<Self> {
        Self::load_from_index_dir_with_quantized(index_dir, model_path, false)
    }

    /// Load a searcher from a specific index directory with quantization option
    pub fn load_from_index_dir_with_quantized(
        index_dir: &Path,
        model_path: &Path,
        quantized: bool,
    ) -> Result<Self> {
        let vector_dir = get_vector_index_path(index_dir);
        let index_path = vector_dir.to_str().unwrap().to_string();

        let acceleration_mode = env_acceleration_mode_lossy();
        let execution_provider = execution_provider_for_acceleration_mode(acceleration_mode);

        #[cfg(feature = "_cuda")]
        match acceleration_mode {
            AccelerationMode::ForceGpu => apply_acceleration_mode(AccelerationMode::ForceGpu),
            AccelerationMode::ForceCpu | AccelerationMode::Auto => {
                apply_acceleration_mode(AccelerationMode::ForceCpu)
            }
        }

        crate::onnx_runtime::ensure_onnx_runtime().context("Failed to initialize ONNX Runtime")?;

        #[cfg(feature = "_cuda")]
        if matches!(acceleration_mode, AccelerationMode::ForceGpu) {
            if !crate::onnx_runtime::is_cudnn_available() {
                anyhow::bail!("FORCE_GPU is set, but cuDNN was not initialized");
            }
            if !next_plaid_onnx::is_cuda_available() {
                anyhow::bail!(
                    "FORCE_GPU is set, but the CUDA execution provider was not initialized"
                );
            }
        }

        // Cap intra-op threads to avoid overhead on high-core-count systems
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(8)
            .min(crate::config::MAX_INTRA_OP_THREADS);

        // Suppress stderr during model loading to hide CoreML's harmless
        // "Context leak detected" warnings on macOS
        let model = crate::stderr::with_suppressed_stderr(|| {
            Colbert::builder(model_path)
                .with_quantized(quantized)
                .with_threads(num_threads)
                .with_execution_provider(execution_provider)
                .build()
        })
        .context("Failed to load ColBERT model")?;

        let index = MmapIndex::load(&index_path).context("Failed to load index")?;

        Ok(Self {
            model,
            index,
            index_path,
        })
    }

    /// Filter results to files within a subdirectory prefix.
    /// Returns document IDs where file path starts with the given prefix.
    pub fn filter_by_path_prefix(&self, prefix: &Path) -> Result<Vec<i64>> {
        let prefix_str = prefix.to_string_lossy();
        // Use SQL LIKE with the prefix followed by %
        let like_pattern = format!("{}%", prefix_str);
        let subset = filtering::where_condition(
            &self.index_path,
            "file LIKE ?",
            &[serde_json::json!(like_pattern)],
        )
        .unwrap_or_default();

        Ok(subset)
    }

    /// Get document IDs matching the given file patterns using globset
    pub fn filter_by_file_patterns(&self, patterns: &[String]) -> Result<Vec<i64>> {
        if patterns.is_empty() {
            return Ok(vec![]);
        }

        // Build globset from patterns
        let Some(glob_set) = build_glob_set(patterns) else {
            return Ok(vec![]);
        };

        // Get all metadata from the index
        let all_metadata = filtering::get(&self.index_path, None, &[], None).unwrap_or_default();

        // Filter metadata by matching file paths against glob patterns
        let matching_ids: Vec<i64> = all_metadata
            .into_iter()
            .filter_map(|row| {
                let doc_id = row.get("_subset_")?.as_i64()?;
                let file = row.get("file")?.as_str()?;
                let path = Path::new(file);
                if glob_set.is_match(path) {
                    Some(doc_id)
                } else {
                    None
                }
            })
            .collect();

        Ok(matching_ids)
    }

    /// Get document IDs for code units that DON'T match exclude patterns (SQL-based)
    /// Uses REGEXP to filter out files matching any of the glob-like patterns
    pub fn filter_exclude_by_patterns(&self, patterns: &[String]) -> Result<Vec<i64>> {
        if patterns.is_empty() {
            // No exclusions - return all IDs
            return filtering::where_condition(&self.index_path, "1=1", &[])
                .map_err(|e| anyhow::anyhow!("{}", e));
        }

        // Convert glob patterns to regex patterns for SQL REGEXP
        // e.g., "*.test.ts" -> ".*\\.test\\.ts$"
        let regex_patterns: Vec<String> = patterns.iter().map(|p| glob_to_regex(p)).collect();

        // Build a combined regex: (pattern1|pattern2|...)
        let combined_regex = regex_patterns.join("|");

        // Use NOT REGEXP to exclude matching files
        let subset = filtering::where_condition_regexp(
            &self.index_path,
            "NOT (file REGEXP ?)",
            &[serde_json::json!(combined_regex)],
        )
        .unwrap_or_default();

        Ok(subset)
    }

    /// Get document IDs for code units NOT in excluded directories (SQL-based)
    /// Uses REGEXP to filter out files in any of the specified directories
    /// Supports both literal directory names and glob patterns:
    /// - Literal: "vendor", "node_modules", ".claude/plugins"
    /// - Glob: "*/plugins", "**/test_*", "**/*_generated"
    pub fn filter_exclude_by_dirs(&self, dirs: &[String]) -> Result<Vec<i64>> {
        if dirs.is_empty() {
            // No exclusions - return all IDs
            return filtering::where_condition(&self.index_path, "1=1", &[])
                .map_err(|e| anyhow::anyhow!("{}", e));
        }

        // Build regex to match paths containing any of the excluded directories
        // Supports both literal names and glob patterns
        // e.g., ["vendor", "*/plugins"] -> "(^|/)vendor/|(^|/)[^/]*/plugins/"
        let dir_patterns: Vec<String> = dirs.iter().map(|d| dir_pattern_to_regex(d)).collect();

        let combined_regex = dir_patterns.join("|");

        // Use NOT REGEXP to exclude files in these directories
        let subset = filtering::where_condition_regexp(
            &self.index_path,
            "NOT (file REGEXP ?)",
            &[serde_json::json!(combined_regex)],
        )
        .unwrap_or_default();

        Ok(subset)
    }

    /// Get document IDs for code units in the given files (exact match)
    pub fn filter_by_files(&self, files: &[String]) -> Result<Vec<i64>> {
        if files.is_empty() {
            return Ok(vec![]);
        }

        // Build SQL condition with OR for multiple exact file matches
        let mut conditions = Vec::new();
        let mut params = Vec::new();

        for file in files {
            conditions.push("file = ?");
            params.push(serde_json::json!(file));
        }

        let condition = conditions.join(" OR ");
        let subset =
            filtering::where_condition(&self.index_path, &condition, &params).unwrap_or_default();

        Ok(subset)
    }

    /// Get document IDs for code units containing the given text pattern
    ///
    /// Supports grep-compatible pattern matching options:
    /// - `extended_regexp`: Use extended regular expressions (ERE) - supports `|`, `+`, `?`, `()` etc.
    /// - `fixed_strings`: Treat pattern as literal string (no regex), takes precedence over extended_regexp
    /// - `word_regexp`: Match whole words only (add word boundaries)
    ///
    /// Pattern matching is always case-insensitive.
    /// Uses pure SQL queries with REGEXP support for efficient filtering.
    /// Automatically converts BRE (Basic Regular Expression) patterns to ERE.
    ///
    /// When not in fixed_strings mode, this function runs BOTH regex and literal searches,
    /// combining and deduplicating results. This handles cases where users search for code
    /// containing regex metacharacters (like parentheses) without escaping them.
    pub fn filter_by_text_pattern_with_options(
        &self,
        pattern: &str,
        extended_regexp: bool,
        fixed_strings: bool,
        word_regexp: bool,
        case_sensitive: bool,
    ) -> Result<Vec<i64>> {
        if pattern.is_empty() {
            return Ok(vec![]);
        }

        // When -F is explicitly set, only do literal matching
        if fixed_strings {
            let escaped = regex::escape(pattern);
            let regex_pattern = if word_regexp {
                format!(r"\b{}\b", escaped)
            } else {
                escaped
            };
            return filtering::where_condition_regexp(
                &self.index_path,
                "code REGEXP ?",
                &[serde_json::json!(regex_pattern)],
            )
            .map_err(|e| anyhow::anyhow!("{}", e));
        }

        // Build the regex pattern for regex-mode search.
        //
        // The SQLite-side REGEXP matches against the multi-line `code`
        // column. Grep treats `^`/`$` as line anchors, but the regex
        // engine's default anchors to start/end of the whole string, so
        // an unanchored search for `^use ` would only match chunks whose
        // first byte starts with `use ` — silently dropping hits on
        // every other line. Force multiline mode (`m`) so anchors behave
        // like grep, and case-insensitive mode (`i`) by default (matches
        // colgrep's historical behaviour). `--case-sensitive` drops the
        // `i` so the regex is matched exactly as typed.
        let flags = if case_sensitive { "(?m)" } else { "(?mi)" };
        let regex_pattern = if word_regexp {
            // Word boundaries without escaping (user wants regex + word match)
            let ere_pattern = escape_literal_braces(&bre_to_ere(pattern));
            format!(r"{}\b{}\b", flags, ere_pattern)
        } else if extended_regexp {
            // Extended regex (ERE) - convert BRE escapes to ERE, then escape literal braces
            format!("{}{}", flags, escape_literal_braces(&bre_to_ere(pattern)))
        } else {
            // Default: basic substring matching (escape for safety).
            // Inline flags still apply.
            format!("{}{}", flags, regex::escape(pattern))
        };

        // Build the fixed-string pattern for literal search
        let fixed_pattern = {
            let escaped = regex::escape(pattern);
            if word_regexp {
                format!(r"\b{}\b", escaped)
            } else {
                escaped
            }
        };

        // Run regex search first (may fail if pattern is invalid regex)
        let regex_results = filtering::where_condition_regexp(
            &self.index_path,
            "code REGEXP ?",
            &[serde_json::json!(regex_pattern)],
        );

        // If regex pattern equals fixed pattern, no need to run both searches
        if regex_pattern == fixed_pattern {
            return regex_results.map_err(|e| anyhow::anyhow!("{}", e));
        }

        // Run fixed-string search (always succeeds since pattern is escaped)
        let fixed_results = filtering::where_condition_regexp(
            &self.index_path,
            "code REGEXP ?",
            &[serde_json::json!(fixed_pattern)],
        )
        .unwrap_or_default();

        // Combine results: union of both searches, deduplicated via HashSet
        match regex_results {
            Ok(regex_ids) => {
                // Both succeeded - combine and deduplicate
                let mut combined: std::collections::HashSet<i64> = regex_ids.into_iter().collect();
                combined.extend(fixed_results);
                Ok(combined.into_iter().collect())
            }
            Err(_) => {
                // Regex failed (invalid pattern) - return fixed results only
                Ok(fixed_results)
            }
        }
    }

    /// Get metadata for specific document IDs
    pub fn get_metadata_for_ids(&self, ids: &[i64]) -> Result<Vec<serde_json::Value>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let metadata = filtering::get(&self.index_path, None, &[], Some(ids)).unwrap_or_default();
        Ok(metadata)
    }

    /// Encode a query once for reuse across multiple searches.
    pub fn encode_query(&self, query: &str) -> Result<ndarray::Array2<f32>> {
        let query_embeddings =
            crate::stderr::with_suppressed_stderr(|| self.model.encode_queries(&[query]))
                .context("Failed to encode query")?;
        Ok(query_embeddings.into_iter().next().unwrap())
    }

    /// Run FTS5 keyword search if the text index is available and the query
    /// remains non-empty after sanitization.
    ///
    /// Uses [`next_plaid::text_search::sanitize_fts5_query_or`] because the
    /// index is built with [`next_plaid::FtsTokenizer::IdentifierAware`]:
    /// each identifier
    /// in the corpus has been pre-split into its compound + camel/snake parts,
    /// so OR semantics let a natural-language query match documents that
    /// contain *any* relevant sub-part. BM25 still rewards documents that hit
    /// more query terms.
    pub fn fts5_search(
        &self,
        query: &str,
        top_k: usize,
        subset: Option<&[i64]>,
    ) -> Option<next_plaid::QueryResult> {
        let sanitized_query = next_plaid::text_search::sanitize_fts5_query_or(query);
        if sanitized_query.is_empty() {
            return None;
        }
        if let Some(sub) = subset {
            next_plaid::text_search::search_filtered(&self.index_path, &sanitized_query, top_k, sub)
                .ok()
        } else {
            next_plaid::text_search::search(&self.index_path, &sanitized_query, top_k).ok()
        }
    }

    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        subset: Option<&[i64]>,
    ) -> Result<Vec<SearchResult>> {
        let query_emb = self.encode_query(query)?;
        self.search_with_embedding(&query_emb, top_k, subset)
    }

    /// Semantic-only search with a pre-computed query embedding.
    pub fn search_with_embedding(
        &self,
        query_emb: &ndarray::Array2<f32>,
        top_k: usize,
        subset: Option<&[i64]>,
    ) -> Result<Vec<SearchResult>> {
        let params = search_params_from_env(top_k);
        let results = self
            .index
            .search(query_emb, &params, subset)
            .context("Search failed")?;

        let doc_ids: Vec<i64> = results.passage_ids.to_vec();
        let metadata = filtering::get(&self.index_path, None, &[], Some(&doc_ids))
            .context("Failed to retrieve metadata")?;

        let search_results: Vec<SearchResult> = metadata
            .into_iter()
            .zip(results.scores.iter())
            .filter_map(|(mut meta, &score)| {
                fix_sqlite_types(&mut meta);
                serde_json::from_value::<CodeUnit>(meta)
                    .ok()
                    .map(|unit| SearchResult { unit, score })
            })
            .collect();

        Ok(search_results)
    }

    /// Hybrid search: semantic retrieval fused with FTS5 keyword search via RRF.
    pub fn search_hybrid(
        &self,
        query: &str,
        top_k: usize,
        subset: Option<&[i64]>,
        alpha: f32,
    ) -> Result<Vec<SearchResult>> {
        let query_emb = self.encode_query(query)?;
        // Pass None so `search_hybrid_with_embedding` fetches FTS5 with its
        // own `fetch_k` (≥200), matching the semantic-side over-fetch.
        // Asymmetric pools (e.g. 200 semantic vs only 30 BM25) bias the
        // min-max fusion toward the side with more candidates.
        self.search_hybrid_with_embedding(&query_emb, query, top_k, subset, alpha, None)
    }

    /// Hybrid search using a pre-computed query embedding and optional cached
    /// FTS5 results.
    pub fn search_hybrid_with_embedding(
        &self,
        query_emb: &ndarray::Array2<f32>,
        query: &str,
        top_k: usize,
        subset: Option<&[i64]>,
        alpha: f32,
        fts5_results: Option<&next_plaid::QueryResult>,
    ) -> Result<Vec<SearchResult>> {
        // Over-fetch generously so that after path-noise penalty + boosts
        // + file collapse we still return exactly `top_k` distinct files
        // (and not a smaller, "approximate" set) — `-k` is a hard
        // contract from the user's perspective.
        //
        // The cost is small: each extra candidate is a single SQLite row
        // read plus a few constant-time score adjustments. We cap at
        // `num_documents()` so we never request more rows than the index
        // actually contains.
        let fetch_k = std::cmp::min(
            std::cmp::max(top_k * 20, 200),
            self.index.num_documents().max(top_k),
        );
        let params = search_params_from_env(fetch_k);
        let semantic = self
            .index
            .search(query_emb, &params, subset)
            .context("Semantic search failed")?;
        trace_log(
            query,
            "semantic",
            &semantic.passage_ids,
            &semantic.scores,
            20,
        );

        let owned_fts5;
        let keyword = match fts5_results {
            Some(fts5) => {
                if let Some(sub) = subset {
                    let sub_set: HashSet<i64> = sub.iter().copied().collect();
                    let mut filtered_ids = Vec::new();
                    let mut filtered_scores = Vec::new();
                    for (id, score) in fts5.passage_ids.iter().zip(fts5.scores.iter()) {
                        if sub_set.contains(id) {
                            filtered_ids.push(*id);
                            filtered_scores.push(*score);
                        }
                    }
                    owned_fts5 = next_plaid::QueryResult {
                        query_id: 0,
                        passage_ids: filtered_ids,
                        scores: filtered_scores,
                    };
                    Some(&owned_fts5)
                } else {
                    Some(fts5)
                }
            }
            None => {
                owned_fts5 =
                    self.fts5_search(query, fetch_k, subset)
                        .unwrap_or(next_plaid::QueryResult {
                            query_id: 0,
                            passage_ids: vec![],
                            scores: vec![],
                        });
                if owned_fts5.passage_ids.is_empty() {
                    None
                } else {
                    Some(&owned_fts5)
                }
            }
        };

        // Score-based min-max fusion: with the identifier-aware BM25 retriever,
        // FTS5 recall@200 is ~99.6%, so the relative-score combiner outperforms
        // pure rank-based RRF (rank-RRF caps both retrievers' contributions
        // even when one is much higher quality on a given query).
        //
        // Fuse into the larger `fetch_k` pool (not `top_k`) so the path-noise
        // reranker can pull a strong-but-buried implementation file above
        // tests / examples that happened to rank higher in the raw fusion.
        if let Some(kw) = keyword {
            trace_log(query, "bm25", &kw.passage_ids, &kw.scores, 20);
        }

        let (fused_ids, fused_scores) = if let Some(kw) = keyword {
            if kw.passage_ids.is_empty() {
                (semantic.passage_ids, semantic.scores)
            } else {
                next_plaid::text_search::fuse_relative_score(
                    &semantic.passage_ids,
                    &semantic.scores,
                    &kw.passage_ids,
                    &kw.scores,
                    alpha,
                    fetch_k,
                )
            }
        } else {
            (semantic.passage_ids, semantic.scores)
        };
        trace_log(query, "fused", &fused_ids, &fused_scores, 20);

        let metadata = filtering::get(&self.index_path, None, &[], Some(&fused_ids))
            .context("Failed to retrieve metadata")?;

        let apply_penalty = crate::ranking::should_apply_path_penalty(query);

        // Index returned metadata rows by `_subset_` id so we can pair each
        // row to its score safely. Using `Vec::zip` here was a bug: if any
        // id in `fused_ids` had no METADATA row (stale FTS5 reference),
        // every subsequent (meta, score) pair shifted by one — silently
        // attaching the wrong score to the wrong unit.
        let mut meta_by_id: std::collections::HashMap<i64, serde_json::Value> =
            std::collections::HashMap::with_capacity(metadata.len());
        for mut m in metadata {
            if let Some(id) = m.get("_subset_").and_then(|v| v.as_i64()) {
                fix_sqlite_types(&mut m);
                meta_by_id.insert(id, m);
            }
        }

        let mut search_results: Vec<SearchResult> = fused_ids
            .iter()
            .zip(fused_scores.iter())
            .filter_map(|(&id, &score)| {
                let meta = meta_by_id.remove(&id)?;
                serde_json::from_value::<CodeUnit>(meta).ok().map(|unit| {
                    let mut final_score = score;
                    if apply_penalty {
                        let file_str = unit.file.to_string_lossy();
                        final_score *= crate::ranking::file_path_penalty(&file_str);
                    }
                    SearchResult {
                        unit,
                        score: final_score,
                    }
                })
            })
            .collect();
        trace_log_results(query, "after_path_penalty", &search_results, 30);

        // Boost candidates whose file-path stem matches a query token —
        // queries like "interceptor manager" map almost surgically to
        // `InterceptorManager.js`, and the path tells us so cheaply.
        crate::ranking::apply_path_stem_boost(
            &mut search_results,
            query,
            |r| r.unit.file.to_str().unwrap_or(""),
            |r| r.score,
            |r, s| r.score = s,
        );
        trace_log_results(query, "after_path_stem_boost", &search_results, 30);

        // Boost units whose tree-sitter name matches a query token. Applied
        // before file-coherence so the symbol the user actually asked about
        // can lift its parent file above neighbours that merely reference it.
        crate::ranking::apply_definition_boost(
            &mut search_results,
            query,
            |r| r.unit.name.as_str(),
            |r| {
                matches!(
                    r.unit.unit_type,
                    crate::UnitType::Function
                        | crate::UnitType::Method
                        | crate::UnitType::Class
                        | crate::UnitType::Constant
                )
            },
            |r| r.score,
            |r, s| r.score = s,
        );
        trace_log_results(query, "after_definition_boost", &search_results, 30);

        // Boost files that appear in multiple candidate units: the file with
        // the most cumulative score in the pool gets `+0.2 * max_score` on
        // its top-scoring unit; others get a proportional share.
        crate::ranking::apply_file_coherence_boost(
            &mut search_results,
            |r| r.unit.file.to_str().unwrap_or(""),
            |r| r.score,
            |r, s| r.score = s,
        );
        trace_log_results(query, "after_coherence_boost", &search_results, 30);

        // Re-sort after the penalty + boosts adjust scores, then collapse
        // to one entry per file (merging start/end lines to cover every
        // matched unit from that file) before truncating to top_k.
        search_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        search_results = collapse_by_file(search_results, top_k);
        trace_log_results(query, "final", &search_results, 30);

        Ok(search_results)
    }

    pub fn num_documents(&self) -> usize {
        self.index.num_documents()
    }
}

/// Emit a single JSON line to stderr describing one stage of the hybrid
/// pipeline, when `COLGREP_TRACE` is truthy. No-op (essentially free) when
/// the env var is unset, so we can leave the call sites in release builds.
///
/// The shape is intentionally simple — one object per call:
///   {"stage":"fused","query":"...","ids":[...],"scores":[...]}
/// `diagnose_misses.py` picks these up via stderr and aggregates them per
/// query so the user can see which stage demotes the relevant file.
fn trace_log(query: &str, stage: &str, ids: &[i64], scores: &[f32], limit: usize) {
    if !trace_enabled() {
        return;
    }
    let n = ids.len().min(scores.len()).min(limit);
    // Build the JSON manually to avoid an extra dep and keep things fast.
    let mut s = String::with_capacity(64 + n * 32);
    s.push_str("{\"stage\":\"");
    s.push_str(stage);
    s.push_str("\",\"query\":");
    json_escape(&mut s, query);
    s.push_str(",\"ids\":[");
    for (i, id) in ids.iter().take(n).enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&id.to_string());
    }
    s.push_str("],\"scores\":[");
    for (i, sc) in scores.iter().take(n).enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{:.6}", sc));
    }
    s.push_str("]}");
    eprintln!("__COLGREP_TRACE__ {}", s);
}

/// Same as [`trace_log`] but accepts a slice of `SearchResult` so callers
/// can dump the post-rerank pool with file paths included.
fn trace_log_results(query: &str, stage: &str, results: &[SearchResult], limit: usize) {
    if !trace_enabled() {
        return;
    }
    let n = results.len().min(limit);
    let mut s = String::with_capacity(64 + n * 64);
    s.push_str("{\"stage\":\"");
    s.push_str(stage);
    s.push_str("\",\"query\":");
    json_escape(&mut s, query);
    s.push_str(",\"results\":[");
    for (i, r) in results.iter().take(n).enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"file\":");
        json_escape(&mut s, &r.unit.file.to_string_lossy());
        s.push_str(&format!(",\"score\":{:.6}}}", r.score));
    }
    s.push_str("]}");
    eprintln!("__COLGREP_TRACE__ {}", s);
}

fn trace_enabled() -> bool {
    matches!(
        std::env::var("COLGREP_TRACE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

fn json_escape(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Collapse search results so each file appears at most once.
///
/// Walks `results` (which the caller has already sorted by score descending)
/// and for every file keeps the first / highest-scoring unit as the leader,
/// then extends that leader's line range to cover every subsequent unit from
/// the same file: `line = min(line_i)`, `end_line = max(end_line_i)`.
///
/// Truncates to `top_k` unique files. The leader's other metadata (name,
/// signature, code, ...) is left untouched so consumers can still display
/// the top unit's structural information.
fn collapse_by_file(results: Vec<SearchResult>, top_k: usize) -> Vec<SearchResult> {
    let mut by_file: std::collections::HashMap<std::path::PathBuf, usize> =
        std::collections::HashMap::new();
    let mut out: Vec<SearchResult> = Vec::with_capacity(top_k.min(results.len()));
    for r in results {
        if let Some(&idx) = by_file.get(&r.unit.file) {
            // Merge: cover the full span of all candidates from this file.
            let leader = &mut out[idx];
            leader.unit.line = leader.unit.line.min(r.unit.line);
            leader.unit.end_line = leader.unit.end_line.max(r.unit.end_line);
        } else {
            if out.len() >= top_k {
                continue;
            }
            by_file.insert(r.unit.file.clone(), out.len());
            out.push(r);
        }
    }
    out
}

/// SQLite stores booleans as integers and arrays as JSON strings. Normalize
/// those back to the shapes the rest of colgrep expects.
fn fix_sqlite_types(meta: &mut serde_json::Value) {
    if let serde_json::Value::Object(ref mut obj) = meta {
        let keys: Vec<String> = obj.keys().cloned().collect();
        for key in keys {
            if key.starts_with("has_") || key.starts_with("is_") {
                if let Some(v) = obj.get(&key) {
                    if let Some(n) = v.as_i64() {
                        obj.insert(key, serde_json::Value::Bool(n != 0));
                    }
                }
                continue;
            }
            if let Some(serde_json::Value::String(s)) = obj.get(&key) {
                if s.starts_with('[') {
                    if let Ok(arr) = serde_json::from_str::<serde_json::Value>(s) {
                        if arr.is_array() {
                            obj.insert(key, arr);
                        }
                    }
                }
            }
        }
    }
}

/// Check if an index exists for the given project built with `model`.
pub fn index_exists(project_root: &Path, model: &str) -> bool {
    paths::index_exists(project_root, model)
}

/// Prompt the user for confirmation before indexing a large number of code units.
/// Returns true if the user confirms (y/Y/Enter), false otherwise.
fn prompt_large_index_confirmation(num_units: usize) -> bool {
    use std::io::{self, BufRead, Write};

    // Check if stdin is a TTY (interactive terminal)
    // If not (e.g., piped input or CI), auto-confirm to avoid blocking
    if !atty::is(atty::Stream::Stdin) {
        return true;
    }

    eprintln!(
        "\n⚠️  Large codebase detected: {} code units to index",
        num_units
    );
    eprintln!("   This may take a while. Use -y/--yes to skip this prompt in the future.\n");
    eprint!("   Proceed with indexing? [Y/n] ");
    io::stderr().flush().ok();

    let stdin = io::stdin();
    let mut line = String::new();
    if stdin.lock().read_line(&mut line).is_err() {
        return false;
    }

    let response = line.trim().to_lowercase();
    // Accept: empty (Enter), 'y', 'yes'
    response.is_empty() || response == "y" || response == "yes"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mtime fast path in `compute_update_plan` must skip content hashing
    /// for files whose stored mtime is unchanged. The stored hash is wrong on
    /// purpose: if the plan still reports the file as unchanged, hashing was
    /// skipped; once the mtime moves, the hash mismatch must be detected.
    /// Regression test — this gate was added in d0423d9 (4x query speedup on
    /// large repos) and silently lost in the d76cb4a rewrite.
    #[test]
    fn test_update_plan_skips_hashing_when_mtime_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        let file_path = temp.path().join("lib.py");
        std::fs::write(&file_path, "def f():\n    return 1\n").unwrap();

        let builder = test_builder(temp.path(), &temp.path().join("index"));

        let mut state = IndexState::default();
        // Bogus hash on purpose: a matching mtime+size must skip the hash entirely.
        state.files.insert(
            PathBuf::from("lib.py"),
            FileInfo::probe(&file_path, 0xDEAD_BEEF).unwrap(),
        );

        let plan = builder.compute_update_plan(&state, None).unwrap();
        assert_eq!(
            plan.unchanged, 1,
            "matching mtime+size must skip the content-hash comparison"
        );
        assert!(plan.changed.is_empty());

        // Move the mtime forward: the fast path no longer applies, so the
        // stale stored hash must now be detected as a change.
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&file_path)
            .unwrap();
        file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(120))
            .unwrap();
        drop(file);

        let plan = builder.compute_update_plan(&state, None).unwrap();
        assert_eq!(plan.changed, vec![PathBuf::from("lib.py")]);
        assert_eq!(plan.unchanged, 0);
    }

    /// When a file's mtime drifts but its content is unchanged (e.g. `touch`,
    /// `git checkout`), the plan must classify it as unchanged AND record a
    /// `touched` refresh so the stat-only fast path applies next run instead of
    /// re-hashing the file on every subsequent search.
    #[test]
    fn test_update_plan_refreshes_touched_files_with_unchanged_content() {
        let temp = tempfile::tempdir().unwrap();
        let file_path = temp.path().join("lib.py");
        std::fs::write(&file_path, "def f():\n    return 1\n").unwrap();

        let builder = test_builder(temp.path(), &temp.path().join("index"));

        // Stored state has the *correct* content hash and size, but a stale mtime.
        let real = FileInfo::probe(&file_path, hash_file(&file_path).unwrap()).unwrap();
        let mut state = IndexState::default();
        state.files.insert(
            PathBuf::from("lib.py"),
            FileInfo {
                mtime: real.mtime.saturating_sub(1),
                ..real.clone()
            },
        );

        let plan = builder.compute_update_plan(&state, None).unwrap();
        assert_eq!(plan.unchanged, 1);
        assert!(plan.changed.is_empty());
        assert_eq!(
            plan.touched.len(),
            1,
            "stale mtime must be queued for refresh"
        );
        assert_eq!(plan.touched[0].0, PathBuf::from("lib.py"));
        assert_eq!(plan.touched[0].1.mtime, real.mtime);
        assert_eq!(plan.touched[0].1.size, real.size);
    }

    /// Measures what the mtime fast path saves on the per-search update plan:
    /// the same synthetic tree planned once with matching stored mtimes
    /// (stat-only fast path) and once with mismatched mtimes but identical
    /// content (forces a read + xxh3 of every byte — what every search paid
    /// without the fast path). Hashing runs with a warm page cache, so the
    /// printed ratio understates the cold-cache real world.
    ///
    ///   cargo test -p colgrep --release measure_update_plan -- --ignored --nocapture
    #[test]
    #[ignore = "timing measurement; run manually in release mode"]
    fn measure_update_plan_mtime_fast_path() {
        const FILES: usize = 5_000;
        const FILE_BYTES: usize = 200 * 1024;

        let temp = tempfile::tempdir().unwrap();
        let line = "x = 1  # padding so the hasher does representative work\n";
        let blob = line.repeat(FILE_BYTES / line.len());

        let mut matching_mtimes = IndexState::default();
        let mut mismatched_mtimes = IndexState::default();
        for i in 0..FILES {
            let rel = PathBuf::from(format!("mod_{i:04}.py"));
            let full = temp.path().join(&rel);
            std::fs::write(&full, &blob).unwrap();
            let content_hash = hash_file(&full).unwrap();
            let (mtime, size) = file_stat(&full).unwrap();
            matching_mtimes.files.insert(
                rel.clone(),
                FileInfo {
                    content_hash,
                    mtime,
                    size,
                },
            );
            mismatched_mtimes.files.insert(
                rel,
                FileInfo {
                    content_hash,
                    mtime: mtime + 1,
                    size,
                },
            );
        }

        let builder = test_builder(temp.path(), &temp.path().join("index"));

        let started = std::time::Instant::now();
        let plan = builder
            .compute_update_plan(&mismatched_mtimes, None)
            .unwrap();
        let hashed_every_file = started.elapsed();
        assert_eq!(plan.unchanged, FILES);

        let started = std::time::Instant::now();
        let plan = builder.compute_update_plan(&matching_mtimes, None).unwrap();
        let stat_only = started.elapsed();
        assert_eq!(plan.unchanged, FILES);

        // The walk scales with file count and is paid either way; the
        // read+hash component the fast path removes scales with repo bytes.
        let removed = hashed_every_file.saturating_sub(stat_only);
        println!(
            "update plan, {} files x {} KiB ({} MiB total):\n  \
             full hashing (pre-fix per-search cost): {:?}\n  \
             mtime fast path:                        {:?}  ({:.1}x faster)\n  \
             read+hash component removed per search: {:?} (scales with repo bytes)",
            FILES,
            FILE_BYTES / 1024,
            FILES * FILE_BYTES / (1024 * 1024),
            hashed_every_file,
            stat_only,
            hashed_every_file.as_secs_f64() / stat_only.as_secs_f64().max(f64::EPSILON),
            removed,
        );
    }

    #[test]
    fn test_force_cpu_execution_provider_is_cpu() {
        assert_eq!(
            execution_provider_for_acceleration_mode(AccelerationMode::ForceCpu),
            ExecutionProvider::Cpu
        );
    }

    #[test]
    fn test_auto_execution_provider_uses_compiled_accelerator_or_cpu() {
        let expected = compiled_accelerated_execution_provider().unwrap_or(ExecutionProvider::Cpu);

        assert_eq!(
            execution_provider_for_acceleration_mode(AccelerationMode::Auto),
            expected
        );
    }

    #[test]
    fn test_accelerator_indexing_parallel_default_is_memory_bounded() {
        // Accelerator providers keep one session by default regardless of workload size:
        // each is a full model copy in device memory.
        assert_eq!(DEFAULT_INDEX_ACCEL_PARALLEL_SESSIONS, 1);
        assert_eq!(
            default_index_parallel_sessions(ExecutionProvider::CoreML, 1),
            1
        );
        assert_eq!(
            default_index_parallel_sessions(ExecutionProvider::CoreML, 1_000_000),
            1
        );
    }

    #[test]
    fn test_cpu_indexing_parallel_default_scales_with_cores_and_workload() {
        // A large workload uses the full cores-based default (the bounded pipeline queues
        // keep memory in check regardless of session count)...
        assert_eq!(
            default_index_parallel_sessions(ExecutionProvider::Cpu, usize::MAX),
            crate::config::get_default_cpu_parallel_sessions()
        );
        // ...while a tiny workload is capped so we don't pay per-session graph builds that
        // can never be used: one session per ~DEFAULT_ENCODE_BATCH_SIZE units, min 1.
        assert_eq!(
            default_index_parallel_sessions(ExecutionProvider::Cpu, 1),
            1
        );
        assert!(
            default_index_parallel_sessions(ExecutionProvider::Cpu, DEFAULT_ENCODE_BATCH_SIZE + 1)
                <= crate::config::get_default_cpu_parallel_sessions()
        );
    }

    #[test]
    #[cfg(all(feature = "coreml", not(feature = "_cuda")))]
    fn test_auto_indexing_provider_avoids_coreml() {
        assert_eq!(
            indexing_execution_provider_for_acceleration_mode(AccelerationMode::Auto),
            ExecutionProvider::Cpu
        );
    }

    #[test]
    fn test_glob_simple_extension() {
        let patterns = vec!["*.rs".to_string()];
        assert!(matches_glob_pattern(Path::new("src/main.rs"), &patterns));
        assert!(matches_glob_pattern(
            Path::new("nested/deep/file.rs"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("src/main.py"), &patterns));
    }

    #[test]
    fn test_glob_recursive_double_star() {
        let patterns = vec!["**/*.rs".to_string()];
        assert!(matches_glob_pattern(Path::new("src/main.rs"), &patterns));
        assert!(matches_glob_pattern(Path::new("a/b/c/d.rs"), &patterns));
        assert!(!matches_glob_pattern(Path::new("main.py"), &patterns));
    }

    #[test]
    fn test_glob_directory_pattern() {
        let patterns = vec!["src/**/*.rs".to_string()];
        assert!(matches_glob_pattern(Path::new("src/main.rs"), &patterns));
        assert!(matches_glob_pattern(
            Path::new("src/index/mod.rs"),
            &patterns
        ));
        // Matches anywhere src/ appears due to **/ prefix
        assert!(matches_glob_pattern(
            Path::new("project/src/main.rs"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("lib/main.rs"), &patterns));
    }

    #[test]
    fn test_glob_github_workflows() {
        let patterns = vec!["**/.github/**/*".to_string()];
        assert!(matches_glob_pattern(
            Path::new(".github/workflows/ci.yml"),
            &patterns
        ));
        assert!(matches_glob_pattern(
            Path::new("project/.github/actions/setup.yml"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("src/main.rs"), &patterns));
    }

    #[test]
    fn test_glob_multiple_patterns() {
        let patterns = vec!["*.rs".to_string(), "*.py".to_string()];
        assert!(matches_glob_pattern(Path::new("main.rs"), &patterns));
        assert!(matches_glob_pattern(Path::new("main.py"), &patterns));
        assert!(!matches_glob_pattern(Path::new("main.js"), &patterns));
    }

    #[test]
    fn test_glob_test_files() {
        let patterns = vec!["*_test.go".to_string()];
        assert!(matches_glob_pattern(
            Path::new("pkg/main_test.go"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("pkg/main.go"), &patterns));
    }

    #[test]
    fn test_glob_empty_patterns() {
        let patterns: Vec<String> = vec![];
        // Empty patterns should match everything
        assert!(matches_glob_pattern(Path::new("any/file.rs"), &patterns));
    }

    #[test]
    fn test_expand_braces_simple() {
        let expanded = expand_braces("*.{rs,md}");
        assert_eq!(expanded, vec!["*.rs", "*.md"]);
    }

    #[test]
    fn test_expand_braces_no_braces() {
        let expanded = expand_braces("*.rs");
        assert_eq!(expanded, vec!["*.rs"]);
    }

    #[test]
    fn test_expand_braces_with_path() {
        let expanded = expand_braces("src/**/*.{ts,tsx,js,jsx}");
        assert_eq!(
            expanded,
            vec!["src/**/*.ts", "src/**/*.tsx", "src/**/*.js", "src/**/*.jsx"]
        );
    }

    #[test]
    fn test_expand_braces_prefix() {
        let expanded = expand_braces("{src,lib}/**/*.rs");
        assert_eq!(expanded, vec!["src/**/*.rs", "lib/**/*.rs"]);
    }

    #[test]
    fn test_expand_braces_multiple_groups() {
        let expanded = expand_braces("{src,lib}/*.{rs,md}");
        assert_eq!(
            expanded,
            vec!["src/*.rs", "src/*.md", "lib/*.rs", "lib/*.md"]
        );
    }

    #[test]
    fn test_glob_brace_expansion() {
        // Test that brace expansion works with glob matching
        let patterns = vec!["*.{rs,py}".to_string()];
        assert!(matches_glob_pattern(Path::new("main.rs"), &patterns));
        assert!(matches_glob_pattern(Path::new("main.py"), &patterns));
        assert!(!matches_glob_pattern(Path::new("main.js"), &patterns));
    }

    #[test]
    fn test_glob_brace_expansion_with_directory() {
        let patterns = vec!["src/**/*.{ts,tsx}".to_string()];
        assert!(matches_glob_pattern(Path::new("src/app.ts"), &patterns));
        assert!(matches_glob_pattern(
            Path::new("src/components/Button.tsx"),
            &patterns
        ));
        assert!(!matches_glob_pattern(Path::new("src/main.js"), &patterns));
    }

    #[test]
    fn test_is_within_project_root_simple_path() {
        let temp_dir = std::env::temp_dir().join("plaid_test_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Simple relative path should be allowed
        assert!(is_within_project_root(&temp_dir, Path::new("src/main.rs")));
        assert!(is_within_project_root(&temp_dir, Path::new("file.txt")));
    }

    #[test]
    fn test_is_within_project_root_path_traversal() {
        let temp_dir = std::env::temp_dir().join("plaid_test_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Path traversal attempts should be rejected
        assert!(!is_within_project_root(
            &temp_dir,
            Path::new("../../../etc/passwd")
        ));
        assert!(!is_within_project_root(&temp_dir, Path::new("../sibling")));
        assert!(!is_within_project_root(
            &temp_dir,
            Path::new("foo/../../..")
        ));
    }

    #[test]
    fn test_is_within_project_root_hidden_traversal() {
        let temp_dir = std::env::temp_dir().join("plaid_test_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Hidden path traversal patterns
        assert!(!is_within_project_root(
            &temp_dir,
            Path::new("src/../../../etc/passwd")
        ));
        assert!(!is_within_project_root(
            &temp_dir,
            Path::new("./foo/../../../bar")
        ));
    }

    #[test]
    fn test_is_within_project_root_valid_dotdot_in_middle() {
        let temp_dir = std::env::temp_dir().join("plaid_test_project_dotdot");
        let sub_dir = temp_dir.join("src").join("subdir");
        let _ = std::fs::create_dir_all(&sub_dir);

        // Create a test file
        let test_file = temp_dir.join("src").join("main.rs");
        let _ = std::fs::write(&test_file, "fn main() {}");

        // Path that goes down then up but stays within project should be allowed
        // src/subdir/../main.rs resolves to src/main.rs
        assert!(is_within_project_root(
            &temp_dir,
            Path::new("src/subdir/../main.rs")
        ));

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_bre_to_ere_alternation() {
        // BRE alternation \| should become ERE |
        assert_eq!(bre_to_ere(r"foo\|bar"), "foo|bar");
        assert_eq!(bre_to_ere(r"a\|b\|c"), "a|b|c");
    }

    #[test]
    fn test_bre_to_ere_quantifiers() {
        // `\+` and `\?` are kept literal: in ERE (the colgrep default)
        // `+` and `?` *are* the quantifiers, and `\+` / `\?` mean literal
        // `+` / `?`. The previous behaviour stripped the backslash, so
        // `foo\+bar` (intended to match a literal `+`) silently ran as
        // a one-or-more quantifier against `foo`. Only the balanced
        // brace form `\{n,m\}` still converts to the ERE quantifier
        // `{n,m}` for BRE-compatibility.
        assert_eq!(bre_to_ere(r"a\+"), r"a\+");
        assert_eq!(bre_to_ere(r"a\?"), r"a\?");
        assert_eq!(bre_to_ere(r"a\{2,3\}"), "a{2,3}");
    }

    #[test]
    fn test_bre_to_ere_grouping() {
        // BRE grouping should become ERE
        assert_eq!(bre_to_ere(r"\(foo\)"), "(foo)");
        assert_eq!(bre_to_ere(r"\(a\|b\)"), "(a|b)");
    }

    #[test]
    fn test_bre_to_ere_escaped_backslash() {
        // Escaped backslash should be preserved
        assert_eq!(bre_to_ere(r"foo\\bar"), r"foo\\bar");
        assert_eq!(bre_to_ere(r"\\|"), r"\\|"); // escaped backslash + literal pipe
    }

    #[test]
    fn test_bre_to_ere_no_change() {
        // Patterns without BRE escapes should pass through unchanged
        assert_eq!(bre_to_ere("foo|bar"), "foo|bar");
        assert_eq!(bre_to_ere("a+b?"), "a+b?");
        assert_eq!(bre_to_ere(r"foo\.bar"), r"foo\.bar"); // escaped dot stays
    }

    #[test]
    fn test_bre_to_ere_mixed() {
        // Mixed BRE/ERE patterns (user's actual use case)
        assert_eq!(
            bre_to_ere(r"default.*25\|top_k.*25"),
            "default.*25|top_k.*25"
        );
    }

    #[test]
    fn test_bre_to_ere_trailing_backslash() {
        // Trailing backslash should be preserved
        assert_eq!(bre_to_ere(r"foo\"), r"foo\");
    }

    #[test]
    fn test_bre_to_ere_unbalanced_parens() {
        // Unbalanced \( or \) must stay escaped to avoid invalid ERE
        assert_eq!(bre_to_ere(r"error\(4"), r"error\(4");
        assert_eq!(bre_to_ere(r"foo\)"), r"foo\)");
        assert_eq!(bre_to_ere(r"a\(b\)c\(d"), "a(b)c\\(d");
    }

    #[test]
    fn test_bre_to_ere_leading_quantifiers() {
        // Leading quantifiers have no preceding atom — keep escaped
        assert_eq!(bre_to_ere(r"\+foo"), r"\+foo");
        assert_eq!(bre_to_ere(r"\?foo"), r"\?foo");
    }

    #[test]
    fn test_bre_to_ere_unbalanced_braces() {
        // Unbalanced \{ without \} must stay escaped
        assert_eq!(bre_to_ere(r"a\{2"), r"a\{2");
        // Leading \{...\} without preceding atom stays escaped
        assert_eq!(bre_to_ere(r"\{2\}"), r"\{2\}");
    }

    #[test]
    fn test_escape_literal_braces_quantifiers_unchanged() {
        // Valid quantifiers should remain unchanged
        assert_eq!(escape_literal_braces("a{2}"), "a{2}");
        assert_eq!(escape_literal_braces("a{2,}"), "a{2,}");
        assert_eq!(escape_literal_braces("a{2,4}"), "a{2,4}");
        assert_eq!(escape_literal_braces("a{,4}"), "a{,4}");
        assert_eq!(escape_literal_braces("Error[0-9]{2,4}"), "Error[0-9]{2,4}");
    }

    #[test]
    fn test_escape_literal_braces_literals_escaped() {
        // Literal braces should be converted to character class form
        assert_eq!(escape_literal_braces("enum.*{"), "enum.*[{]");
        assert_eq!(escape_literal_braces("struct {"), "struct [{]");
        assert_eq!(escape_literal_braces("}"), "[}]");
        assert_eq!(escape_literal_braces("{}"), "[{][}]");
    }

    #[test]
    fn test_escape_literal_braces_mixed() {
        // Mixed quantifiers and literal braces
        assert_eq!(
            escape_literal_braces("enum.*Error.*{[^}]*Error[0-9]{2,4}[^}]*}"),
            "enum.*Error.*[{][^}]*Error[0-9]{2,4}[^}]*[}]"
        );
    }

    #[test]
    fn test_escape_literal_braces_character_class_unchanged() {
        // Braces inside character classes should remain unchanged
        assert_eq!(escape_literal_braces("[{]"), "[{]");
        assert_eq!(escape_literal_braces("[}]"), "[}]");
        assert_eq!(escape_literal_braces("[{}]"), "[{}]");
        assert_eq!(escape_literal_braces("a[{]b"), "a[{]b");
    }

    #[test]
    fn test_escape_literal_braces_complex_pattern() {
        // The original failing pattern
        let pattern = r"enum\s+[A-Za-z0-9_]+Error\s*{[^}]*Error[0-9]{2,4}[^}]*}";
        let escaped = escape_literal_braces(pattern);
        assert_eq!(
            escaped,
            r"enum\s+[A-Za-z0-9_]+Error\s*[{][^}]*Error[0-9]{2,4}[^}]*[}]"
        );
    }

    #[test]
    fn test_combine_search_results_no_duplicates() {
        // Simulate the deduplication logic used in filter_by_text_pattern_with_options
        // when combining regex and fixed-string search results

        // Case 1: Overlapping results (same IDs from both searches)
        let regex_ids: Vec<i64> = vec![1, 2, 3, 4, 5];
        let fixed_ids: Vec<i64> = vec![3, 4, 5, 6, 7];

        let mut combined: std::collections::HashSet<i64> = regex_ids.into_iter().collect();
        combined.extend(fixed_ids);
        let result: Vec<i64> = combined.into_iter().collect();

        // Assert no duplicates
        let mut sorted = result.clone();
        sorted.sort();
        assert!(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "Combined results contain duplicates"
        );

        // Assert we have the union of both sets
        assert_eq!(sorted.len(), 7); // {1, 2, 3, 4, 5, 6, 7}

        // Case 2: Identical results (both searches return same IDs)
        let regex_ids: Vec<i64> = vec![10, 20, 30];
        let fixed_ids: Vec<i64> = vec![10, 20, 30];

        let mut combined: std::collections::HashSet<i64> = regex_ids.into_iter().collect();
        combined.extend(fixed_ids);
        let result: Vec<i64> = combined.into_iter().collect();

        let mut sorted = result.clone();
        sorted.sort();
        assert!(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "Identical results produced duplicates"
        );
        assert_eq!(sorted.len(), 3);

        // Case 3: Disjoint results (no overlap)
        let regex_ids: Vec<i64> = vec![1, 2, 3];
        let fixed_ids: Vec<i64> = vec![4, 5, 6];

        let mut combined: std::collections::HashSet<i64> = regex_ids.into_iter().collect();
        combined.extend(fixed_ids);
        let result: Vec<i64> = combined.into_iter().collect();

        let mut sorted = result.clone();
        sorted.sort();
        assert!(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "Disjoint results produced duplicates"
        );
        assert_eq!(sorted.len(), 6);
    }

    #[test]
    fn test_is_glob_pattern() {
        assert!(is_glob_pattern("*.rs"));
        assert!(is_glob_pattern("**/test"));
        assert!(is_glob_pattern("foo?bar"));
        assert!(is_glob_pattern("[abc]"));
        assert!(!is_glob_pattern("vendor"));
        assert!(!is_glob_pattern("node_modules"));
        assert!(!is_glob_pattern(".claude/plugins"));
    }

    #[test]
    fn test_dir_pattern_to_regex_literal() {
        // Literal directory names should work as before
        assert_eq!(dir_pattern_to_regex("vendor"), "(^|/)vendor/");
        assert_eq!(dir_pattern_to_regex("node_modules"), "(^|/)node_modules/");
        assert_eq!(
            dir_pattern_to_regex(".claude/plugins"),
            "(^|/)\\.claude/plugins/"
        );
    }

    #[test]
    fn test_dir_pattern_to_regex_glob() {
        // Test single wildcard prefix - matches exactly one level
        let pattern = dir_pattern_to_regex("*/plugins");
        assert_eq!(pattern, "^[^/]*/plugins/");

        // Test double wildcard - matches any depth
        let pattern = dir_pattern_to_regex("**/test_*");
        assert_eq!(pattern, "(^|.*/)test_[^/]*/");

        // Test wildcard at the end (no prefix slash)
        let pattern = dir_pattern_to_regex(".claude/*");
        assert_eq!(pattern, "(^|/)\\.claude/[^/]*/");
    }

    #[test]
    fn test_dir_pattern_to_regex_matching() {
        // Test that regex patterns match expected paths
        use regex::Regex;

        // Literal directory pattern - matches at any depth
        let pattern = dir_pattern_to_regex("vendor");
        let re = Regex::new(&pattern).unwrap();
        assert!(re.is_match("vendor/package.json"));
        assert!(re.is_match("src/vendor/lib.rs"));
        assert!(!re.is_match("vendorfile.txt"));

        // Glob pattern: */plugins - matches exactly one level deep
        let pattern = dir_pattern_to_regex("*/plugins");
        let re = Regex::new(&pattern).unwrap();
        assert!(re.is_match(".claude/plugins/tool.json"));
        assert!(re.is_match("foo/plugins/bar.txt"));
        assert!(!re.is_match("plugins/direct.txt")); // needs parent
        assert!(!re.is_match("a/b/plugins/deep.txt")); // too deep (two levels)

        // Glob pattern: **/test_* - matches at any depth
        let pattern = dir_pattern_to_regex("**/test_*");
        let re = Regex::new(&pattern).unwrap();
        assert!(re.is_match("test_utils/helper.rs"));
        assert!(re.is_match("src/test_integration/spec.rs"));
        assert!(re.is_match("a/b/c/test_foo/file.rs"));
        assert!(!re.is_match("src/testing/file.rs"));

        // Glob pattern with wildcard in middle - matches at any position
        let pattern = dir_pattern_to_regex(".claude/*");
        let re = Regex::new(&pattern).unwrap();
        assert!(re.is_match(".claude/plugins/file.json"));
        assert!(re.is_match("foo/.claude/bar/test.txt"));
        assert!(!re.is_match(".claude/file.json")); // .claude is not a parent dir
    }

    #[test]
    fn test_should_ignore_relative_hidden_subdir() {
        let empty: &[String] = &[];
        // Hidden subdirectories inside the project should be ignored
        assert!(should_ignore(Path::new(".hidden/foo.rs"), empty, empty));
        assert!(should_ignore(Path::new("src/.secret/bar.rs"), empty, empty));
        // But allowed hidden dirs are fine
        assert!(!should_ignore(
            Path::new(".github/workflows/ci.yml"),
            empty,
            empty
        ));
    }

    #[test]
    fn test_should_ignore_does_not_reject_dotprefixed_root_when_relative() {
        let empty: &[String] = &[];
        assert!(!should_ignore(Path::new("index.ts"), empty, empty));
        assert!(!should_ignore(Path::new("src/lib.rs"), empty, empty));
        assert!(!should_ignore(Path::new("package.json"), empty, empty));
    }

    #[test]
    fn test_should_ignore_absolute_dotprefixed_ancestors() {
        let empty: &[String] = &[];
        let path = Path::new("/home/user/.pi/agent/extensions/index.ts");
        assert!(should_ignore(path, empty, empty));
    }

    #[test]
    fn test_should_ignore_extra_ignore_patterns() {
        let empty: &[String] = &[];
        let extra = vec!["generated".to_string(), "*.pb.go".to_string()];

        // Extra ignore patterns should be respected
        assert!(should_ignore(
            Path::new("src/generated/types.rs"),
            &extra,
            empty
        ));
        assert!(should_ignore(Path::new("api/service.pb.go"), &extra, empty));
        // Normal files unaffected
        assert!(!should_ignore(Path::new("src/main.rs"), &extra, empty));
    }

    #[test]
    fn test_should_ignore_force_include_overrides_default() {
        let empty: &[String] = &[];
        let force = vec![".vscode".to_string()];

        // .vscode is normally ignored (hidden dir not in ALLOWED_HIDDEN_DIRS)
        assert!(should_ignore(
            Path::new(".vscode/settings.json"),
            empty,
            empty
        ));
        // But with force-include, it's allowed
        assert!(!should_ignore(
            Path::new(".vscode/settings.json"),
            empty,
            &force
        ));
    }

    #[test]
    fn test_should_ignore_force_include_overrides_ignored_dir() {
        let empty: &[String] = &[];
        let force = vec!["vendor".to_string()];

        // vendor is in IGNORED_DIRS by default
        assert!(should_ignore(Path::new("vendor/lib/util.go"), empty, empty));
        // But force-include overrides that
        assert!(!should_ignore(
            Path::new("vendor/lib/util.go"),
            empty,
            &force
        ));
    }

    #[test]
    fn test_should_ignore_force_include_overrides_extra_ignore() {
        let extra = vec!["generated".to_string()];
        let force = vec!["generated".to_string()];

        // Extra ignore says ignore "generated", but force-include overrides
        assert!(!should_ignore(
            Path::new("src/generated/types.rs"),
            &extra,
            &force
        ));
    }

    #[test]
    fn test_should_ignore_force_include_path_prefix() {
        let empty: &[String] = &[];
        let force = vec!["vendor/internal".to_string()];

        // Full path prefix match: vendor/internal/* is included
        assert!(!should_ignore(
            Path::new("vendor/internal/lib.go"),
            empty,
            &force
        ));
        // But vendor/external is still ignored (vendor is in IGNORED_DIRS)
        assert!(should_ignore(
            Path::new("vendor/external/lib.go"),
            empty,
            &force
        ));
    }

    #[test]
    fn test_should_ignore_force_include_suffix_pattern() {
        let empty: &[String] = &[];
        let force = vec!["*.egg-info".to_string()];

        // *.egg-info is in IGNORED_DIRS by default
        assert!(should_ignore(
            Path::new("mypackage.egg-info/PKG-INFO"),
            empty,
            empty
        ));
        // Force-include with suffix pattern overrides it
        assert!(!should_ignore(
            Path::new("mypackage.egg-info/PKG-INFO"),
            empty,
            &force
        ));
    }

    #[test]
    fn test_should_ignore_combined_extra_and_force() {
        let extra = vec!["snapshots".to_string()];
        let force = vec![".idea".to_string()];

        // snapshots is extra-ignored
        assert!(should_ignore(
            Path::new("tests/snapshots/test1.snap"),
            &extra,
            &force
        ));
        // .idea is normally a hidden+ignored dir, but force-included
        assert!(!should_ignore(
            Path::new(".idea/workspace.xml"),
            &extra,
            &force
        ));
        // Normal ignored dirs still work
        assert!(should_ignore(
            Path::new("node_modules/foo/bar.js"),
            &extra,
            &force
        ));
    }

    /// Build a minimal `IndexBuilder` pointing at the given (project, index) directories,
    /// without a model. Suitable for exercising index-maintenance paths that don't encode.
    fn test_builder(project_root: &Path, index_dir: &Path) -> IndexBuilder {
        IndexBuilder {
            model: None,
            model_path: PathBuf::from("/nonexistent-model"),
            quantized: false,
            parallel_sessions: None,
            batch_size: None,
            project_root: project_root.to_path_buf(),
            index_dir: index_dir.to_path_buf(),
            pool_factor: None,
            encode_batch_size: None,
            index_chunk_size: None,
            dynamic_batch: true,
            auto_confirm: true,
            model_id: "test-model".to_string(),
        }
    }

    /// Build a small vector index + filtering DB at `index_path`, distributing `n` documents
    /// evenly across `files` (each doc tagged with its file in the filtering DB). Model-free.
    fn build_fixture_index(index_path: &str, files: &[&str], docs_per_file: usize) {
        use ndarray::Array2;
        use next_plaid::IndexConfig;

        let n = files.len() * docs_per_file;
        let mut embeddings: Vec<Array2<f32>> = Vec::new();
        for i in 0..n {
            let mut doc = Array2::<f32>::zeros((5, 32));
            for j in 0..5 {
                for k in 0..32 {
                    doc[[j, k]] = (i as f32 * 0.1) + (j as f32 * 0.01) + (k as f32 * 0.001);
                }
            }
            for mut row in doc.rows_mut() {
                let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    row.iter_mut().for_each(|x| *x /= norm);
                }
            }
            embeddings.push(doc);
        }

        let config = IndexConfig {
            nbits: 2,
            batch_size: 50,
            seed: Some(42),
            kmeans_niters: 2,
            max_points_per_centroid: 256,
            n_samples_kmeans: None,
            start_from_scratch: 999,
            force_cpu: false,
            ..Default::default()
        };
        MmapIndex::create_with_kmeans(&embeddings, index_path, &config).unwrap();

        let metadata: Vec<serde_json::Value> = (0..n)
            .map(|i| {
                let f = files[i / docs_per_file];
                // Per-file unique, identifier-friendly token so FTS5 alignment after a
                // delete can be asserted (each surviving file's token must still match,
                // each deleted file's token must not).
                let token = format!("marker_{}", f.replace('.', "_"));
                serde_json::json!({ "file": f, "text": format!("def x(): # {token} body") })
            })
            .collect();
        let doc_ids: Vec<i64> = (0..n as i64).collect();
        filtering::create(index_path, &metadata, &doc_ids).unwrap();
        next_plaid::text_search::index(
            index_path,
            &metadata,
            &doc_ids,
            &next_plaid::FtsTokenizer::IdentifierAware,
        )
        .unwrap();
    }

    /// Token a `build_fixture_index` file is indexed under in FTS5.
    #[cfg(test)]
    fn fixture_token(file: &str) -> String {
        format!("marker_{}", file.replace('.', "_"))
    }

    /// Number of FTS5 hits for `token` (keyword path used by `-e`/hybrid search).
    #[cfg(test)]
    fn fts_hits(index_path: &str, token: &str) -> usize {
        next_plaid::text_search::search(index_path, token, 100)
            .map(|r| r.passage_ids.len())
            .unwrap_or(0)
    }

    /// Serializes every test that calls `delete_files_from_index`. The tests that assert
    /// the global `DELETE_FROM_INDEX_CALLS` counter read a before/after delta, which any
    /// concurrent deleting test would otherwise corrupt under parallel execution.
    #[cfg(test)]
    static DELETE_TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Issue #116: deleting many files must collapse into a *single* full-index rewrite, not one
    /// rewrite per file (which made incremental updates O(changed_files × index_size) and hung
    /// for minutes). Asserts both the call count and that exactly the right documents survive.
    #[test]
    fn test_delete_files_from_index_is_a_single_rewrite() {
        use std::sync::atomic::Ordering;
        let _serial = DELETE_TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());

        let tmp = tempfile::tempdir().unwrap();
        let index_path = tmp.path().to_str().unwrap();

        let files = ["a.rs", "b.rs", "c.rs", "d.rs"];
        let docs_per_file = 3;
        build_fixture_index(index_path, &files, docs_per_file);

        // Delete three of the four files in one batched call.
        let to_delete: Vec<PathBuf> = ["a.rs", "b.rs", "c.rs"].iter().map(PathBuf::from).collect();

        let before = DELETE_FROM_INDEX_CALLS.load(Ordering::Relaxed);
        let removed = delete_files_from_index(index_path, &to_delete).unwrap();
        let calls = DELETE_FROM_INDEX_CALLS.load(Ordering::Relaxed) - before;

        assert_eq!(removed, 9, "should remove 3 files × 3 docs");
        assert_eq!(
            calls, 1,
            "issue #116: deleting N files must be ONE index rewrite, not one per file"
        );

        // Only d.rs survives, with its 3 documents.
        let remaining = filtering::get_distinct_strings(index_path, "file").unwrap();
        assert_eq!(remaining, vec!["d.rs".to_string()]);
        let idx = MmapIndex::load(index_path).unwrap();
        assert_eq!(idx.metadata.num_documents, 3);
    }

    /// Deleting nothing (or only unknown files) must not rewrite the index at all.
    #[test]
    fn test_delete_files_from_index_noop_does_not_rewrite() {
        use std::sync::atomic::Ordering;
        let _serial = DELETE_TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());

        let tmp = tempfile::tempdir().unwrap();
        let index_path = tmp.path().to_str().unwrap();
        build_fixture_index(index_path, &["a.rs"], 2);

        let before = DELETE_FROM_INDEX_CALLS.load(Ordering::Relaxed);
        assert_eq!(delete_files_from_index(index_path, &[]).unwrap(), 0);
        assert_eq!(
            delete_files_from_index(index_path, &[PathBuf::from("missing.rs")]).unwrap(),
            0
        );
        assert_eq!(DELETE_FROM_INDEX_CALLS.load(Ordering::Relaxed) - before, 0);
    }

    /// Regression: `delete_files_from_index` must prune FTS5, not just the vector index
    /// and metadata. Deleting a MIDDLE file re-sequences survivor IDs (the rebuild path),
    /// so a stale FTS5 must not leave the deleted file's text searchable, and survivors'
    /// keyword hits must still map to exactly their own docs. Previously the standalone
    /// delete path skipped FTS5 entirely, so `-e`/keyword search kept finding deleted files.
    #[test]
    fn test_delete_files_from_index_prunes_fts5_middle() {
        let _serial = DELETE_TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let index_path = tmp.path().to_str().unwrap();
        let files = ["a.rs", "b.rs", "c.rs", "d.rs", "e.rs"];
        let dpf = 2;
        build_fixture_index(index_path, &files, dpf);

        // sanity: every file is keyword-searchable before deletion
        for f in files {
            assert_eq!(fts_hits(index_path, &fixture_token(f)), dpf, "pre {f}");
        }

        // delete a middle file -> survivor IDs shift -> FTS5 rebuild path
        let removed = delete_files_from_index(index_path, &[PathBuf::from("c.rs")]).unwrap();
        assert_eq!(removed, dpf);

        // deleted file's text must be gone from FTS5...
        assert_eq!(
            fts_hits(index_path, &fixture_token("c.rs")),
            0,
            "deleted file still keyword-searchable (stale FTS5)"
        );
        // ...and every survivor's keyword hits must still map to exactly its own docs
        // (no cross-contamination from misaligned FTS5 rowids).
        for f in ["a.rs", "b.rs", "d.rs", "e.rs"] {
            assert_eq!(
                fts_hits(index_path, &fixture_token(f)),
                dpf,
                "survivor {f} keyword hits misaligned after delete"
            );
        }
    }

    /// Regression: tail-of-ID-space delete takes the O(deleted) FTS5 suffix path; it must
    /// still drop the deleted file's rows and leave survivors aligned.
    #[test]
    fn test_delete_files_from_index_prunes_fts5_suffix() {
        let _serial = DELETE_TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let index_path = tmp.path().to_str().unwrap();
        let files = ["a.rs", "b.rs", "c.rs", "d.rs", "e.rs"];
        let dpf = 2;
        build_fixture_index(index_path, &files, dpf);

        // delete the LAST file -> its doc IDs are the tail -> FTS5 suffix-delete path
        let removed = delete_files_from_index(index_path, &[PathBuf::from("e.rs")]).unwrap();
        assert_eq!(removed, dpf);

        assert_eq!(
            fts_hits(index_path, &fixture_token("e.rs")),
            0,
            "tail-deleted still found"
        );
        for f in ["a.rs", "b.rs", "c.rs", "d.rs"] {
            assert_eq!(fts_hits(index_path, &fixture_token(f)), dpf, "survivor {f}");
        }
    }

    /// Issue #115 (bug 1): an index left dirty by an interrupted run must be cleaned once the
    /// repair has reconciled it, even when there are no file changes. Otherwise the dirty flag
    /// is never cleared and every future run pays for a needless repair — "permanently dirty".
    #[test]
    fn test_incremental_update_clears_dirty_with_no_changes() {
        let proj = tempfile::tempdir().unwrap();
        let idx = tempfile::tempdir().unwrap();

        // Persist a dirty state with no tracked files; the project tree is empty, so the update
        // plan is empty and we hit the "nothing to do" path.
        let dirty_state = IndexState {
            dirty: true,
            ..Default::default()
        };
        dirty_state.save(idx.path()).unwrap();
        assert!(IndexState::load(idx.path()).unwrap().dirty);

        let mut builder = test_builder(proj.path(), idx.path());
        builder.incremental_update(&dirty_state, None).unwrap();

        assert!(
            !IndexState::load(idx.path()).unwrap().dirty,
            "issue #115: dirty flag must be cleared after a no-op update, not left set forever"
        );
    }

    /// Issue #115 (bug 2): a sibling worktree whose resumable build is in progress / was
    /// interrupted (a `.building` marker is present) must be rejected as a seed source. Such an
    /// index passes every other completeness check — metadata.json, filtering DB, non-empty
    /// non-dirty current-version state — yet only holds a fraction of its documents.
    #[test]
    fn test_seed_source_rejects_in_progress_build() {
        let tmp = tempfile::tempdir().unwrap();
        let src_dir = tmp.path();
        let vector = get_vector_index_path(src_dir);
        std::fs::create_dir_all(&vector).unwrap();
        std::fs::write(vector.join("metadata.json"), "{}").unwrap();
        filtering::create(
            vector.to_str().unwrap(),
            &[serde_json::json!({ "file": "a.rs" })],
            &[0],
        )
        .unwrap();

        let mut state = IndexState::default();
        state.files.insert(
            PathBuf::from("a.rs"),
            FileInfo {
                content_hash: 1,
                mtime: 1,
                size: 0,
            },
        );
        state.save(src_dir).unwrap(); // save() stamps the current format version

        // Complete index, no marker → usable seed source.
        assert!(seed_source_state(src_dir).is_some());

        // Interrupted/in-progress build → rejected.
        std::fs::write(src_dir.join(BUILDING_MARKER), "").unwrap();
        assert!(
            seed_source_state(src_dir).is_none(),
            "issue #115: an in-progress (.building) sibling index must not be used as a seed"
        );
        std::fs::remove_file(src_dir.join(BUILDING_MARKER)).unwrap();

        // Sanity: the other rejection reasons still hold.
        // Incompatible index format (save() always stamps the current one, so
        // patch the persisted JSON directly).
        let state_path = paths::get_state_path(src_dir);
        let mut raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
        raw["index_format_version"] = serde_json::json!(INDEX_FORMAT_VERSION + 1);
        std::fs::write(&state_path, serde_json::to_string(&raw).unwrap()).unwrap();
        assert!(seed_source_state(src_dir).is_none(), "format mismatch");

        let mut dirty = IndexState::load(src_dir).unwrap();
        dirty.dirty = true;
        // Re-save with dirty set (save() preserves the flag, restamps version/format).
        dirty.save(src_dir).unwrap();
        assert!(seed_source_state(src_dir).is_none(), "dirty index");
    }

    /// A missing state.json must not be mistaken for an incompatible index format.
    /// A default-constructed state deserializes with `index_format_version == 0`,
    /// which differs from `INDEX_FORMAT_VERSION`; gating the rebuild on that alone
    /// would send a healthy index through a full re-embed instead of the cheap
    /// reconstruct-from-filtering-DB path.
    #[test]
    fn test_missing_state_reconstructs_instead_of_full_rebuild() {
        let proj = tempfile::tempdir().unwrap();
        let idx = tempfile::tempdir().unwrap();

        // A real (model-free) vector index + filtering DB referencing one project
        // file, but no state.json (deleted, or written by a pre-versioning build).
        std::fs::write(proj.path().join("a.rs"), "fn a() {}\n").unwrap();
        let vector = get_vector_index_path(idx.path());
        std::fs::create_dir_all(&vector).unwrap();
        build_fixture_index(vector.to_str().unwrap(), &["a.rs"], 2);

        let mut builder = test_builder(proj.path(), idx.path());
        // A full rebuild would need the builder's model (nonexistent) and fail;
        // the reconstruct path sees the on-disk file as unchanged.
        let stats = builder.run_indexing(None, false).unwrap();

        assert_eq!(
            stats.unchanged, 1,
            "file must be recognized, not re-embedded"
        );
        let state = IndexState::load(idx.path()).unwrap();
        assert_eq!(
            state.files.len(),
            1,
            "state must be reconstructed from the DB"
        );
        assert_eq!(state.index_format_version, INDEX_FORMAT_VERSION);
    }
}
