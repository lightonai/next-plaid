//! Download progress bars, styled like colgrep's indexing bar.

use indicatif::{ProgressBar, ProgressStyle};

/// A byte-count bar for a download of `total` bytes (a spinner when the size is
/// unknown), hidden when `enabled` is false. indicatif also hides it when stderr is not
/// a terminal.
pub fn download_bar(total: Option<u64>, enabled: bool) -> ProgressBar {
    if !enabled {
        return ProgressBar::hidden();
    }
    let pb = match total {
        Some(n) if n > 0 => ProgressBar::new(n),
        _ => ProgressBar::new_spinner(),
    };
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▓░"),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

/// hf-hub download progress drawn with [`download_bar`].
pub struct HubProgress {
    enabled: bool,
    bar: Option<ProgressBar>,
}

impl HubProgress {
    pub fn new(enabled: bool) -> Self {
        Self { enabled, bar: None }
    }
}

impl hf_hub::api::Progress for HubProgress {
    fn init(&mut self, size: usize, _filename: &str) {
        self.bar = Some(download_bar(Some(size as u64), self.enabled));
    }

    fn update(&mut self, size: usize) {
        if let Some(bar) = &self.bar {
            bar.inc(size as u64);
        }
    }

    fn finish(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
    }
}
