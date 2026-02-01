//! Utilities for suppressing stderr output.
//!
//! This is used to suppress harmless but noisy warnings from CoreML
//! (e.g., "Context leak detected, msgtracer returned -1").

/// Guard that suppresses stderr while held.
/// Restores stderr on drop.
#[cfg(unix)]
pub struct SuppressStderr {
    original_fd: Option<std::os::fd::OwnedFd>,
}

#[cfg(unix)]
impl SuppressStderr {
    /// Create a new guard that suppresses stderr.
    /// Returns None if suppression fails (stderr will remain unchanged).
    pub fn new() -> Option<Self> {
        use std::fs::File;
        use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};

        // Save original stderr fd by duplicating it
        let stderr_fd = std::io::stderr().as_raw_fd();
        let original_fd = unsafe {
            let dup_fd = libc::dup(stderr_fd);
            if dup_fd < 0 {
                return None;
            }
            Some(OwnedFd::from_raw_fd(dup_fd))
        };

        // Open /dev/null and redirect stderr to it
        if let Ok(devnull) = File::open("/dev/null") {
            unsafe {
                libc::dup2(devnull.as_fd().as_raw_fd(), stderr_fd);
            }
        } else {
            return None;
        }

        Some(Self { original_fd })
    }
}

#[cfg(unix)]
impl Drop for SuppressStderr {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;

        if let Some(ref original) = self.original_fd {
            let stderr_fd = std::io::stderr().as_raw_fd();
            unsafe {
                libc::dup2(original.as_raw_fd(), stderr_fd);
            }
        }
    }
}

/// No-op implementation for non-Unix platforms.
#[cfg(not(unix))]
pub struct SuppressStderr;

#[cfg(not(unix))]
impl SuppressStderr {
    pub fn new() -> Option<Self> {
        Some(Self)
    }
}

/// Execute a closure with stderr suppressed.
/// If suppression fails, the closure runs with stderr unchanged.
pub fn with_suppressed_stderr<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let _guard = SuppressStderr::new();
    f()
}
