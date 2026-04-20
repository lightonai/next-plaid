//! Request handlers for the next-plaid API.

pub mod documents;
pub mod encode;
pub mod metadata;
pub mod project_sync;
pub mod rerank;
pub mod search;

pub use documents::*;
pub use encode::*;
pub use metadata::*;
pub use project_sync::*;
pub use rerank::*;
pub use search::*;
