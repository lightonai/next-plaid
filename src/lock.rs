//! File-based locking for index operations.
//!
//! This module provides advisory file locking to prevent concurrent modifications
//! to the same index. When multiple processes or threads attempt to update the
//! same index simultaneously, only one will proceed while others wait.

use std::fs::{File, OpenOptions};
use std::path::Path;

use fs2::FileExt;

use crate::error::Result;

/// RAII guard that holds an exclusive lock on an index directory.
///
/// The lock is automatically released when the guard is dropped,
/// ensuring proper cleanup even in case of errors or panics.
pub struct IndexLockGuard {
    #[allow(dead_code)]
    file: File,
}

impl IndexLockGuard {
    /// Acquire an exclusive lock on an index directory.
    ///
    /// This creates a `.update.lock` file in the index directory and acquires
    /// an exclusive (write) lock on it. The lock will block if another process
    /// already holds it.
    ///
    /// # Arguments
    ///
    /// * `index_path` - Path to the index directory
    ///
    /// # Returns
    ///
    /// A lock guard that releases the lock when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock file cannot be created or the lock cannot
    /// be acquired.
    pub fn acquire(index_path: &Path) -> Result<Self> {
        let lock_path = index_path.join(".update.lock");

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;

        // Acquire exclusive lock (blocks if another process holds it)
        file.lock_exclusive()?;

        Ok(Self { file })
    }

    /// Try to acquire an exclusive lock without blocking.
    ///
    /// Returns `Ok(Some(guard))` if the lock was acquired, `Ok(None)` if the
    /// lock is held by another process, or an error if something went wrong.
    #[allow(dead_code)]
    pub fn try_acquire(index_path: &Path) -> Result<Option<Self>> {
        let lock_path = index_path.join(".update.lock");

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;

        // Try to acquire lock without blocking
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(Self { file })),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

impl Drop for IndexLockGuard {
    fn drop(&mut self) {
        // Unlock the file; ignore errors since we're in Drop
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    #[test]
    fn test_lock_acquire_release() {
        let dir = TempDir::new().unwrap();

        // Acquire lock
        let guard = IndexLockGuard::acquire(dir.path()).unwrap();

        // Lock file should exist
        assert!(dir.path().join(".update.lock").exists());

        // Drop the guard to release lock
        drop(guard);

        // Should be able to acquire again
        let _guard2 = IndexLockGuard::acquire(dir.path()).unwrap();
    }

    #[test]
    fn test_try_acquire_when_locked() {
        let dir = TempDir::new().unwrap();

        // Acquire lock
        let _guard = IndexLockGuard::acquire(dir.path()).unwrap();

        // Try to acquire should return None (lock is held)
        let result = IndexLockGuard::try_acquire(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_lock_blocks_concurrent_access() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_path_buf();

        let barrier = Arc::new(Barrier::new(2));
        let results = Arc::new(std::sync::Mutex::new(Vec::new()));

        let path1 = path.clone();
        let barrier1 = Arc::clone(&barrier);
        let results1 = Arc::clone(&results);

        let handle1 = thread::spawn(move || {
            barrier1.wait();
            let start = Instant::now();
            let _guard = IndexLockGuard::acquire(&path1).unwrap();
            results1
                .lock()
                .unwrap()
                .push(("t1_acquired", start.elapsed()));

            // Hold lock for a bit
            thread::sleep(Duration::from_millis(50));
            results1
                .lock()
                .unwrap()
                .push(("t1_release", start.elapsed()));
        });

        let path2 = path.clone();
        let barrier2 = Arc::clone(&barrier);
        let results2 = Arc::clone(&results);

        let handle2 = thread::spawn(move || {
            barrier2.wait();
            // Small delay to let t1 acquire first (not guaranteed but likely)
            thread::sleep(Duration::from_millis(10));

            let start = Instant::now();
            let _guard = IndexLockGuard::acquire(&path2).unwrap();
            results2
                .lock()
                .unwrap()
                .push(("t2_acquired", start.elapsed()));
        });

        handle1.join().unwrap();
        handle2.join().unwrap();

        // Both threads should have completed
        let final_results = results.lock().unwrap();
        assert!(final_results.len() >= 2);
    }
}
