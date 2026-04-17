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
///
/// If the closure panics, the panic message and location are captured via a
/// temporary panic hook and printed to the (restored) stderr before the panic
/// is resumed. This prevents panics inside suppressed regions from becoming
/// silent, while still letting the panic propagate up the stack normally.
///
/// If suppression fails, the closure runs with stderr unchanged.
pub fn with_suppressed_stderr<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    use std::panic::{self, AssertUnwindSafe};
    use std::sync::{Arc, Mutex};

    // Buffer to capture panic info written during the suppressed region.
    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let captured_for_hook = Arc::clone(&captured);

    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let mut buf = format!("{}\n", info);
        // Include a backtrace when requested so RUST_BACKTRACE still works.
        let bt = std::backtrace::Backtrace::capture();
        if bt.status() == std::backtrace::BacktraceStatus::Captured {
            buf.push_str(&format!("stack backtrace:\n{}\n", bt));
        }
        if let Ok(mut slot) = captured_for_hook.lock() {
            *slot = Some(buf);
        }
    }));

    let guard = SuppressStderr::new();
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    drop(guard);

    // Restore the user's panic hook before doing anything else.
    panic::set_hook(prev_hook);

    match result {
        Ok(value) => value,
        Err(payload) => {
            if let Some(msg) = captured.lock().ok().and_then(|mut s| s.take()) {
                eprint!("{}", msg);
            }
            panic::resume_unwind(payload);
        }
    }
}
