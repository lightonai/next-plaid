//! next-plaid-cli: Semantic code search powered by ColBERT
//!
//! This crate provides semantic code search using:
//! - **next-plaid** - Multi-vector search (ColBERT/PLAID)
//! - **next-plaid-onnx** - ONNX-based ColBERT encoding
//! - **tree-sitter** - Multi-language code parsing

pub mod config;
pub mod embed;
pub mod index;
pub mod model;
pub mod onnx_runtime;
pub mod parser;

pub use config::Config;
pub use embed::build_embedding_text;
pub use index::paths::{
    find_parent_index, get_index_dir_for_project, get_plaid_data_dir, get_vector_index_path,
    ParentIndexInfo, ProjectMetadata,
};
pub use index::{index_exists, IndexBuilder, SearchResult, Searcher, UpdatePlan, UpdateStats};
pub use model::{ensure_model, DEFAULT_MODEL};
pub use onnx_runtime::ensure_onnx_runtime;
pub use parser::{build_call_graph, detect_language, extract_units, CodeUnit, Language, UnitType};
