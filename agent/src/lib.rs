//! colgrep agent: a small model that localizes code by searching a repository with
//! colgrep and reading files through a read-only terminal.
//!
//! The default model is `lightonai/colgrep-default-minicpm5-2B`, trained with the
//! harness reproduced here (prompt, tool schemas, turn budget and observation format are
//! byte-identical to training). Every model-facing knob is configurable so another
//! checkpoint, prompt or template can be swapped in.

pub mod config;
pub mod engine;
pub mod llm;
pub mod metal;
pub mod model;
pub mod profile;
pub mod protocol;
pub mod pyjson;
#[cfg(feature = "local")]
pub mod runtime;
pub mod sandbox;
pub mod search;
pub mod session;
pub mod template;
pub mod toolcall;
