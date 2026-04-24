//! Exports browser TypeScript bindings for the shared Rust contract.

use std::env;
use std::error::Error;
use std::fs;

use next_plaid_browser_contract::{
    RuntimeRequest, RuntimeResponse, StorageRequest, StorageResponse,
};
use ts_rs::{Config, TS};

/// Writes the shared contract bindings into the configured TypeScript output directory.
fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::from_env().with_large_int("number");

    RuntimeRequest::export_all(&config)?;
    StorageRequest::export_all(&config)?;
    RuntimeResponse::export_all(&config)?;
    StorageResponse::export_all(&config)?;

    trim_generated_trailing_whitespace()?;

    Ok(())
}

fn trim_generated_trailing_whitespace() -> Result<(), Box<dyn Error>> {
    let Some(output_dir) = env::var_os("TS_RS_EXPORT_DIR") else {
        return Ok(());
    };

    for entry in fs::read_dir(output_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("ts") {
            continue;
        }

        let contents = fs::read_to_string(&path)?;
        let mut normalized = String::with_capacity(contents.len());
        for line in contents.split_inclusive('\n') {
            if let Some(body) = line.strip_suffix('\n') {
                normalized.push_str(body.trim_end_matches([' ', '\t']));
                normalized.push('\n');
            } else {
                normalized.push_str(line.trim_end_matches([' ', '\t']));
            }
        }

        if normalized != contents {
            fs::write(path, normalized)?;
        }
    }

    Ok(())
}
