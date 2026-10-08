//! Processes the agent started, so an interrupted run can stop them before exiting
//! (a `std::process::exit` skips the destructors that normally do it).

use std::sync::Mutex;

static CHILDREN: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// Remember a child process to stop if the run is interrupted.
pub fn register_child(pid: u32) {
    if let Ok(mut c) = CHILDREN.lock() {
        c.push(pid);
    }
}

/// Forget a child process (it was stopped normally).
pub fn unregister_child(pid: u32) {
    if let Ok(mut c) = CHILDREN.lock() {
        c.retain(|&p| p != pid);
    }
}

/// Kill every registered child process. Called from an interrupt handler: no
/// allocation-heavy work, and never blocks on a poisoned lock.
pub fn kill_children() {
    let Ok(children) = CHILDREN.try_lock() else {
        return;
    };
    for &pid in children.iter() {
        kill(pid);
    }
}

#[cfg(unix)]
fn kill(pid: u32) {
    // SAFETY: plain kill(2) on a pid this process spawned.
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
}

#[cfg(windows)]
fn kill(_pid: u32) {
    // The job object (see llm::server) kills the server when colgrep exits.
}
