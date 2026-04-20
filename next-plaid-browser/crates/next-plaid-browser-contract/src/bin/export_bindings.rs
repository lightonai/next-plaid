//! Exports browser TypeScript bindings for the shared Rust contract.

use std::error::Error;

use next_plaid_browser_contract::{RuntimeRequest, RuntimeResponse, StorageRequest, StorageResponse};
use ts_rs::{Config, TS};

/// Writes the shared contract bindings into the configured TypeScript output directory.
fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::from_env().with_large_int("number");

    RuntimeRequest::export_all(&config)?;
    StorageRequest::export_all(&config)?;
    RuntimeResponse::export_all(&config)?;
    StorageResponse::export_all(&config)?;

    Ok(())
}
