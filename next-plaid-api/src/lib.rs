//! NextPlaid API Library
//!
//! This crate provides a REST API for the next-plaid multi-vector search engine.
//!
//! # Features
//!
//! - Index creation and management
//! - Document upload with metadata
//! - Multi-query batch search
//! - SQLite-based metadata filtering
//!
//! # Usage
//!
//! This crate is primarily used as a binary (`next-plaid-api`), but the modules
//! can be used programmatically to embed the API in other applications.

pub mod error;
pub mod handlers;
pub mod models;
pub mod state;

pub use error::{ApiError, ApiResult};
pub use state::{ApiConfig, AppState};
