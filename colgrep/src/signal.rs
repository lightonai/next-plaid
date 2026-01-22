//! Signal handling for graceful interruption during indexing
//!
//! This module provides a global flag that can be set by Ctrl+C (SIGINT) handlers
//! to allow long-running operations like indexing to exit gracefully.
//!
//! The module supports two modes:
//! - **Normal mode**: Interrupts are processed immediately (during encoding)
//! - **Critical section mode**: Interrupts are deferred until the critical section
//!   exits (during index writes to ensure data consistency)

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Global flag indicating whether an interrupt signal has been received.
/// This is set to `true` when Ctrl+C is pressed during indexing.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Counter for nested critical sections.
/// When > 0, we're in a critical section and interrupts should be deferred.
static CRITICAL_SECTION_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Check if an interrupt signal has been received.
#[inline]
pub fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

/// Check if an interrupt signal has been received, but only if not in a critical section.
/// Use this for encoding operations where we want immediate interruption.
#[inline]
pub fn is_interrupted_outside_critical() -> bool {
    if CRITICAL_SECTION_DEPTH.load(Ordering::Relaxed) > 0 {
        false
    } else {
        INTERRUPTED.load(Ordering::Relaxed)
    }
}

/// Reset the interrupt flag (useful for testing or reusing the indexer).
#[allow(dead_code)]
pub fn reset_interrupted() {
    INTERRUPTED.store(false, Ordering::Relaxed);
}

/// Enter a critical section where interrupts should be deferred.
/// Call this before starting index write operations.
/// Must be paired with `exit_critical_section()`.
#[inline]
pub fn enter_critical_section() {
    CRITICAL_SECTION_DEPTH.fetch_add(1, Ordering::SeqCst);
}

/// Exit a critical section.
/// After exiting all critical sections, deferred interrupts will be honored.
#[inline]
pub fn exit_critical_section() {
    CRITICAL_SECTION_DEPTH.fetch_sub(1, Ordering::SeqCst);
}

/// RAII guard for critical sections.
/// Automatically exits the critical section when dropped.
pub struct CriticalSectionGuard;

impl Default for CriticalSectionGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl CriticalSectionGuard {
    /// Enter a critical section and return a guard that exits it on drop.
    pub fn new() -> Self {
        enter_critical_section();
        CriticalSectionGuard
    }
}

impl Drop for CriticalSectionGuard {
    fn drop(&mut self) {
        exit_critical_section();
    }
}

/// Set up the Ctrl+C signal handler.
/// Should be called once at the start of indexing operations.
/// Returns an error if the handler cannot be set.
pub fn setup_signal_handler() -> Result<(), ctrlc::Error> {
    ctrlc::set_handler(move || {
        // Set the flag on first interrupt
        if !INTERRUPTED.swap(true, Ordering::SeqCst)
            && CRITICAL_SECTION_DEPTH.load(Ordering::Relaxed) > 0
        {
            eprintln!("\n⚠️  Interrupt received, finishing current write operation...");
        }
    })
}

/// Error type for interrupted operations.
#[derive(Debug, Clone)]
pub struct InterruptedError;

impl std::fmt::Display for InterruptedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Operation interrupted by user")
    }
}

impl std::error::Error for InterruptedError {}

/// Check for interruption and return an error if interrupted.
/// Use this in loops to bail out early.
#[inline]
pub fn check_interrupted() -> Result<(), InterruptedError> {
    if is_interrupted() {
        Err(InterruptedError)
    } else {
        Ok(())
    }
}
