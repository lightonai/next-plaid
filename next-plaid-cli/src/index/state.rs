use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use xxhash_rust::xxh3::xxh3_64;

const INDEX_DIR: &str = ".plaid";
const STATE_FILE: &str = "state.json";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexState {
    pub files: HashMap<PathBuf, FileInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub content_hash: u64,
    pub mtime: u64,
}

impl IndexState {
    pub fn load(project_root: &Path) -> Result<Self> {
        let state_path = project_root.join(INDEX_DIR).join(STATE_FILE);
        if state_path.exists() {
            let content = fs::read_to_string(&state_path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, project_root: &Path) -> Result<()> {
        let index_dir = project_root.join(INDEX_DIR);
        fs::create_dir_all(&index_dir)?;

        let state_path = index_dir.join(STATE_FILE);
        let content = serde_json::to_string_pretty(self)?;
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
