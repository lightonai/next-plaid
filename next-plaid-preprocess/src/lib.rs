//! Shared ColBERT preprocessing primitives for native and browser runtimes.

#[cfg(all(feature = "native", feature = "wasm"))]
compile_error!("features 'native' and 'wasm' are mutually exclusive");

#[cfg(not(any(feature = "native", feature = "wasm")))]
compile_error!("enable either the 'native' or 'wasm' feature");

mod config;
mod error;
mod prepare;

pub use config::ColbertConfig;
pub use error::{Error, Result};
pub use prepare::{
    build_skiplist, prepare_batch_from_tokenized_documents, prepare_batch_from_tokenizer_encodings,
    preprocess_texts, update_token_ids, PreparedDocumentBatch, TokenizedDocument,
};
