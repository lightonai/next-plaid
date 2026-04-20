use std::path::PathBuf;

use thiserror::Error as ThisError;

/// Result type for the shared preprocessing crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Stable error surface for shared preprocessing operations.
#[derive(Debug, ThisError)]
pub enum Error {
    /// The config JSON could not be parsed.
    #[error("failed to parse onnx_config.json")]
    JsonParse(#[from] serde_json::Error),

    /// A config file could not be read from disk.
    #[cfg(feature = "native")]
    #[error("failed to read config from {path}")]
    ConfigRead {
        /// The path that failed to read.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// A model directory did not contain the required config file.
    #[cfg(feature = "native")]
    #[error("onnx_config.json not found in {model_dir}")]
    ConfigNotFound {
        /// The model directory that was missing `onnx_config.json`.
        model_dir: PathBuf,
    },

    /// The provided config cannot support the requested preprocessing path.
    #[error("invalid config: {message}")]
    InvalidConfig {
        /// Human-readable explanation of the invalid setting.
        message: String,
    },

    /// The required prefix token could not be resolved from the tokenizer.
    #[error("prefix token '{prefix}' not found in tokenizer vocabulary")]
    MissingPrefixToken {
        /// The missing token string.
        prefix: String,
    },

    /// A tokenizer lookup failed for a required token.
    #[error("tokenizer lookup failed for '{token}'")]
    TokenizerLookupFailed {
        /// The token string that could not be resolved.
        token: String,
    },

    /// A provided encoding row had no usable tokens.
    #[error("encoding at row {row_index} was empty")]
    EmptyEncoding {
        /// Zero-based row index of the invalid encoding.
        row_index: usize,
    },

    /// A provided encoding row had incompatible field lengths.
    #[error(
        "encoding at row {row_index} has mismatched ids/type_ids lengths ({ids_len} vs {type_ids_len})"
    )]
    InvalidEncoding {
        /// Zero-based row index of the invalid encoding.
        row_index: usize,
        /// Number of token ids in the row.
        ids_len: usize,
        /// Number of token type ids in the row.
        type_ids_len: usize,
    },
}
