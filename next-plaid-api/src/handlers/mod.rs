//! Request handlers for the next-plaid API.

pub mod documents;
pub mod encode;
pub mod metadata;
pub mod rerank;
pub mod search;

pub use documents::*;
pub use encode::*;
pub use metadata::*;
pub use rerank::*;
pub use search::*;
