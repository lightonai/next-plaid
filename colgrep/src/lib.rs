//! colgrep: Semantic code search powered by ColBERT
//!
//! This crate provides semantic code search using:
//! - **next-plaid** - Multi-vector search (ColBERT/PLAID)
//! - **next-plaid-onnx** - ONNX-based ColBERT encoding
//! - **tree-sitter** - Multi-language code parsing

pub mod config;
pub mod embed;
pub mod index;
pub mod install;
pub mod model;
pub mod onnx_runtime;
pub mod parser;
pub mod signal;

pub use config::{Config, DEFAULT_BATCH_SIZE, DEFAULT_POOL_FACTOR};
pub use embed::build_embedding_text;
pub use index::paths::{
    acquire_index_lock, find_parent_index, get_colgrep_data_dir, get_index_dir_for_project,
    get_vector_index_path, ParentIndexInfo, ProjectMetadata,
};
pub use index::state::IndexState;
pub use index::{
    bre_to_ere, index_exists, IndexBuilder, SearchResult, Searcher, UpdatePlan, UpdateStats,
};
pub use model::{ensure_model, DEFAULT_MODEL};
pub use onnx_runtime::ensure_onnx_runtime;
pub use parser::{
    build_call_graph, detect_language, extract_units, is_text_format, CodeUnit, Language, UnitType,
};

// Install commands for AI coding tools
pub use install::{
    install_claude_code, install_codex, install_opencode, uninstall_claude_code, uninstall_codex,
    uninstall_opencode,
};

// Signal handling
pub use signal::{
    check_interrupted, is_interrupted, is_interrupted_outside_critical, setup_signal_handler,
    CriticalSectionGuard,
};
