use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use xxhash_rust::xxh3::xxh3_64;

use super::paths::get_state_path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexState {
    /// CLI version that created/updated this index
    #[serde(default)]
    pub cli_version: String,
    pub files: HashMap<PathBuf, FileInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub content_hash: u64,
    pub mtime: u64,
}

impl IndexState {
    /// Load state from the given index directory
    pub fn load(index_dir: &Path) -> Result<Self> {
        let state_path = get_state_path(index_dir);
        if state_path.exists() {
            let content = fs::read_to_string(&state_path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(Self::default())
        }
    }

    /// Save state to the given index directory
    pub fn save(&self, index_dir: &Path) -> Result<()> {
        fs::create_dir_all(index_dir)?;

        // Update CLI version before saving
        let mut state = self.clone();
        state.cli_version = env!("CARGO_PKG_VERSION").to_string();

        let state_path = get_state_path(index_dir);
        let content = serde_json::to_string_pretty(&state)?;
        fs::write(&state_path, content)?;
        Ok(())
    }
}

/// Hash file content using xxHash for fast comparison
pub fn hash_file(path: &Path) -> Result<u64> {
    let content = fs::read(path)?;
    Ok(xxh3_64(&content))
}

/// Get file modification time as unix timestamp
pub fn get_mtime(path: &Path) -> Result<u64> {
    let metadata = fs::metadata(path)?;
    let mtime = metadata
        .modified()?
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();
    Ok(mtime)
}
