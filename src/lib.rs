//! Lategrep: CPU-based PLAID implementation for multi-vector search
//!
//! This crate provides a pure Rust, CPU-only implementation of the PLAID algorithm
//! for efficient multi-vector search (late interaction retrieval).

pub mod codec;
pub mod error;
pub mod index;
pub mod search;
pub mod utils;

pub use codec::ResidualCodec;
pub use error::{Error, Result};
pub use index::{Index, IndexConfig, Metadata};
pub use search::{QueryResult, SearchParameters};
