use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use colgrep::abtest::sessions::{self, SessionSample};
use colgrep::abtest::stats_math::{fmt_change, fmt_thousands};
use colgrep::{get_colgrep_data_dir, get_vector_index_path, IndexState, ProjectMetadata};

/// Get the number of documents in an index by reading its metadata
fn get_index_document_count(vector_index_path: &Path) -> usize {
    let metadata_path = vector_index_path.join("metadata.json");
    if let Ok(content) = std::fs::read_to_string(&metadata_path) {
        if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(count) = metadata.get("num_documents").and_then(|v| v.as_u64()) {
                return count as usize;
            }
        }
    }
    0
}

/// One-line A/B summary. The full picture lives in `colgrep ab`; this is just
/// enough to notice that data is accumulating.
fn print_session_ab(samples: &[SessionSample]) {
    let Some(s) = sessions::summarize_sessions(samples) else {
        return;
    };
    println!(
        "  A/B: {} sessions ({} with colgrep, {} without)",
        s.n_treatment + s.n_control,
        s.n_treatment,
        s.n_control
    );
    match s.cost_ratio {
        Some(ratio) => println!(
            "    finding things costs {} per session with colgrep ({} vs {} tokens) — see `colgrep ab`",
            fmt_change(ratio),
            fmt_thousands(s.median_find_cost_treatment.round() as u64),
            fmt_thousands(s.median_find_cost_control.round() as u64),
        ),
        None => println!("    not enough data yet — see `colgrep ab`"),
    }
}

pub fn cmd_stats() -> Result<()> {
    let data_dir = get_colgrep_data_dir()?;
    if !data_dir.exists() {
        println!("No indexes found.");
        return Ok(());
    }

    let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    if index_dirs.is_empty() {
        println!("No indexes found.");
        return Ok(());
    }

    // Session samples are stored globally; group them per project path so each
    // project block below can pick up its own.
    let mut session_groups: HashMap<String, Vec<SessionSample>> = HashMap::new();
    for sample in sessions::load_session_samples() {
        session_groups
            .entry(sample.project_path.clone())
            .or_default()
            .push(sample);
    }

    let mut total_functions = 0usize;
    let mut total_searches = 0u64;

    for entry in &index_dirs {
        let index_path = entry.path();

        // Load project metadata (path + optional model)
        let (project_path, model) = match ProjectMetadata::load(&index_path) {
            Ok(m) => (m.project_path.display().to_string(), m.model),
            Err(_) => ("Unknown".to_string(), None),
        };

        // Load state for search count
        let state = IndexState::load(&index_path).unwrap_or_default();

        // Get function count from index metadata
        let vector_index_path = get_vector_index_path(&index_path);
        let num_functions = get_index_document_count(&vector_index_path);

        println!("Project: {}", project_path);
        match model {
            Some(m) => println!("  Model: {}", m),
            None => println!("  Model: (unknown — legacy index)"),
        }
        println!("  Functions indexed: {}", num_functions);
        println!("  Search count: {}", state.search_count);
        if let Some(samples) = session_groups.remove(&project_path) {
            print_session_ab(&samples);
        }
        println!();

        total_functions += num_functions;
        total_searches += state.search_count;
    }

    // Session samples for projects whose index was cleared or renamed.
    for (project_path, samples) in session_groups {
        println!("Project: {} (index no longer present)", project_path);
        print_session_ab(&samples);
        println!();
    }

    println!(
        "Total: {} indexes, {} functions, {} searches",
        index_dirs.len(),
        total_functions,
        total_searches
    );

    Ok(())
}

pub fn cmd_reset_stats() -> Result<()> {
    let data_dir = get_colgrep_data_dir()?;
    if !data_dir.exists() {
        println!("No indexes found.");
        return Ok(());
    }

    let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    if index_dirs.is_empty() {
        println!("No indexes found.");
        return Ok(());
    }

    let mut reset_count = 0;
    for entry in &index_dirs {
        let index_path = entry.path();
        if let Ok(mut state) = IndexState::load(&index_path) {
            state.reset_search_count();
            state.save(&index_path)?;
            reset_count += 1;
        }
    }

    // A/B session samples live outside the per-index directories.
    let _ = sessions::clear_session_samples();

    println!(
        "✅ Reset search statistics and A/B samples for {} index(es)",
        reset_count
    );
    Ok(())
}
