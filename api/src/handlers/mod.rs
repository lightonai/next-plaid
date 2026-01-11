//! Request handlers for the lategrep API.

pub mod documents;
pub mod encode;
pub mod metadata;
pub mod search;

pub use documents::*;
pub use encode::*;
pub use metadata::*;
pub use search::*;
