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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_index_state_default() {
        let state = IndexState::default();
        assert!(state.cli_version.is_empty());
        assert!(state.files.is_empty());
    }

    #[test]
    fn test_file_info_serialization() {
        let info = FileInfo {
            content_hash: 12345678901234567890,
            mtime: 1700000000,
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("12345678901234567890"));
        assert!(json.contains("1700000000"));

        let deserialized: FileInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content_hash, 12345678901234567890);
        assert_eq!(deserialized.mtime, 1700000000);
    }

    #[test]
    fn test_index_state_serialization() {
        let mut files = HashMap::new();
        files.insert(
            PathBuf::from("src/main.rs"),
            FileInfo {
                content_hash: 123456,
                mtime: 1700000000,
            },
        );
        let state = IndexState {
            cli_version: "1.0.0".to_string(),
            files,
        };

        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("1.0.0"));
        assert!(json.contains("src/main.rs"));

        let deserialized: IndexState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.cli_version, "1.0.0");
        assert!(deserialized
            .files
            .contains_key(&PathBuf::from("src/main.rs")));
    }

    #[test]
    fn test_index_state_load_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let result = IndexState::load(temp_dir.path());
        assert!(result.is_ok());
        let state = result.unwrap();
        assert!(state.files.is_empty());
    }

    #[test]
    fn test_index_state_save_and_load() {
        let temp_dir = TempDir::new().unwrap();

        let mut state = IndexState::default();
        state.files.insert(
            PathBuf::from("test.rs"),
            FileInfo {
                content_hash: 999999,
                mtime: 1700000000,
            },
        );

        // Save
        state.save(temp_dir.path()).unwrap();

        // Load and verify
        let loaded = IndexState::load(temp_dir.path()).unwrap();
        assert!(loaded.files.contains_key(&PathBuf::from("test.rs")));
        let file_info = loaded.files.get(&PathBuf::from("test.rs")).unwrap();
        assert_eq!(file_info.content_hash, 999999);

        // CLI version should be set after saving
        assert!(!loaded.cli_version.is_empty());
    }

    #[test]
    fn test_hash_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // Create a file with known content
        let mut file = fs::File::create(&file_path).unwrap();
        file.write_all(b"Hello, World!").unwrap();

        let hash = hash_file(&file_path).unwrap();
        assert!(hash > 0);

        // Same content should produce same hash
        let hash2 = hash_file(&file_path).unwrap();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_hash_file_different_content() {
        let temp_dir = TempDir::new().unwrap();

        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");

        fs::write(&file1, "Content A").unwrap();
        fs::write(&file2, "Content B").unwrap();

        let hash1 = hash_file(&file1).unwrap();
        let hash2 = hash_file(&file2).unwrap();

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_get_mtime() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        fs::write(&file_path, "test content").unwrap();

        let mtime = get_mtime(&file_path).unwrap();
        // mtime should be a reasonable Unix timestamp (after year 2000)
        assert!(mtime > 946684800); // Jan 1, 2000
    }

    #[test]
    fn test_hash_file_nonexistent() {
        let result = hash_file(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_get_mtime_nonexistent() {
        let result = get_mtime(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }
}
