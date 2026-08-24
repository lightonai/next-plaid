//! `colgrep ab` — show what colgrep costs versus not having it.
//!
//! The report is written to be readable without knowing any statistics: two
//! costs, a plain-English comparison, and an honest sentence about whether
//! there is enough data yet. The p-value is available but demoted, because a
//! dashboard you can re-run at will invites reading a bare "significant" the
//! moment it appears — which is not what a fixed-sample p-value means.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use colored::Colorize;

use colgrep::abtest::sessions::{self, SessionAbSummary, SessionSample};
use colgrep::abtest::stats_math::{fmt_delta_short, fmt_multiple, fmt_p, fmt_range, fmt_thousands};

pub fn cmd_ab(filter_path: Option<&Path>, json: bool) -> Result<()> {
    let mut groups: HashMap<String, Vec<SessionSample>> = HashMap::new();
    for sample in sessions::load_session_samples() {
        groups
            .entry(sample.project_path.clone())
            .or_default()
            .push(sample);
    }

    if let Some(raw) = filter_path {
        let wanted = std::fs::canonicalize(raw)
            .unwrap_or_else(|_| raw.to_path_buf())
            .to_string_lossy()
            .into_owned();
        groups.retain(|path, _| *path == wanted);
        if groups.is_empty() {
            println!("No A/B data yet for {}", wanted);
            println!("Run `colgrep ab` to see every project with data.");
            return Ok(());
        }
    }

    let mut projects: Vec<(String, Vec<SessionSample>)> = groups.into_iter().collect();
    projects.sort_by_key(|(path, s)| (std::cmp::Reverse(s.len()), path.clone()));

    if json {
        println!("{}", serde_json::to_string_pretty(&to_json(&projects))?);
        return Ok(());
    }

    if projects.is_empty() {
        print_empty_state();
        return Ok(());
    }

    println!("{}", "Is colgrep saving you tokens?".bold());
    for (path, samples) in &projects {
        println!();
        print_project(path, samples);
    }
    println!();
    print_legend();
    println!();
    println!(
        "{}",
        "While this is on, some sessions run without colgrep so the two can be \
         compared — stop with `colgrep settings --no-ab-test`."
            .dimmed()
    );
    Ok(())
}

fn print_empty_state() {
    println!("No A/B data yet.");
    println!();
    println!("This measures whether an agent with colgrep spends fewer tokens finding");
    println!("code than one without. It is off by default, because measuring it means");
    println!("running some of your Claude Code sessions without colgrep.");
    println!();
    println!("  {:32} turn it on", "colgrep settings --ab-test".green());
    println!(
        "  {:32} once, so finished sessions get recorded",
        "colgrep --install-claude-code".green()
    );
    println!();
    println!("Then work normally for a few days and come back to `colgrep ab`.");
}

fn print_project(path: &str, samples: &[SessionSample]) {
    let Some(s) = sessions::summarize_sessions(samples) else {
        return;
    };
    println!("{}", path.cyan().bold());
    println!(
        "{}",
        format!(
            "{} sessions: {} with colgrep, {} without",
            s.n_treatment + s.n_control,
            s.n_treatment,
            s.n_control
        )
        .dimmed()
    );
    println!();

    print_verdict(&s);
    println!();
    print_table(&s);

    if let Some(explanation) = explain(&s) {
        println!();
        println!("  {}", explanation);
    }
    if s.contaminated_controls > 0 {
        println!(
            "  {}",
            format!(
                "⚠️  {} of the without-colgrep sessions used colgrep anyway, which blurs \
                 the comparison",
                s.contaminated_controls
            )
            .yellow()
        );
    }
}

/// The answer, first and in one line, because it is the only thing most
/// readers want. Everything below it is the supporting arithmetic.
fn print_verdict(s: &SessionAbSummary) {
    // Badge padded to a fixed width so the explanation lines below all start
    // in the same column regardless of which verdict fired.
    let say = |badge: colored::ColoredString, sentence: String, details: Vec<String>| {
        println!("  {:<11}{}", badge, sentence);
        for detail in details {
            println!("  {:<11}{}", "", detail.dimmed());
        }
    };

    match (s.cost_ratio, s.cost_ratio_ci) {
        // Interval spans no-change: no claim is defensible yet.
        (Some(ratio), Some((lo, hi))) if lo <= 1.0 && hi >= 1.0 => {
            let mut details = vec![format!(
                "looks {} so far, but the truth is {}",
                fmt_delta_short(ratio),
                fmt_range(lo, hi)
            )];
            let missing = 15usize.saturating_sub(s.n_treatment.min(s.n_control));
            if missing > 0 {
                details.push(format!(
                    "about {missing} more sessions per arm should settle it"
                ));
            }
            say(
                "TOO EARLY".yellow().bold(),
                "can't tell yet".to_string(),
                details,
            );
        }
        (Some(ratio), Some((lo, hi))) if hi < 1.0 => say(
            "WORTH IT".green().bold(),
            format!(
                "finding code costs {} with colgrep",
                fmt_delta_short(ratio).green().bold()
            ),
            vec![format!("the real saving is {}", fmt_range(lo, hi))],
        ),
        (Some(ratio), Some((lo, hi))) => say(
            "NOT HERE".red().bold(),
            format!(
                "finding code costs {} with colgrep",
                fmt_delta_short(ratio).red().bold()
            ),
            vec![format!("the real cost is {}", fmt_range(lo, hi))],
        ),
        _ => say(
            "TOO EARLY".yellow().bold(),
            "not enough sessions in each arm to compare".to_string(),
            Vec::new(),
        ),
    }
}

/// Side-by-side numbers, ordered so the reader can do the division by hand:
/// tokens used ÷ references found = tokens per reference. The first two rows are
/// therefore **totals over the arm's sessions**, not per-session medians —
/// mixing the two made the visible arithmetic wrong.
fn print_table(s: &SessionAbSummary) {
    let row = |label: &str, with: String, without: String, note: String| {
        let line = format!("  {:<28}{:>12}{:>13}", label, with, without);
        if note.is_empty() {
            println!("{}", line);
        } else {
            println!("{}   {}", line, note.dimmed());
        }
    };

    println!(
        "  {:<28}{:>12}{:>13}",
        "per session".dimmed(),
        "with colgrep".bold(),
        "without".bold()
    );

    // These two rows divide into the row below them. They are per-session
    // means, not totals: the division still comes out exact, and unlike
    // totals they stay honest when one arm ran more sessions than the other.
    row(
        "tokens used to find code",
        fmt_thousands(s.mean_find_cost_treatment.round() as u64),
        fmt_thousands(s.mean_find_cost_control.round() as u64),
        fmt_delta_short(safe_ratio(
            s.mean_find_cost_treatment,
            s.mean_find_cost_control,
        )),
    );
    row(
        "file:line references found",
        format!("{:.1}", s.mean_locations_treatment),
        format!("{:.1}", s.mean_locations_control),
        fmt_multiple(s.mean_locations_treatment, s.mean_locations_control),
    );
    println!("  {}", "─".repeat(66).dimmed());
    let key_note = s
        .cost_ratio
        .map(fmt_delta_short)
        .unwrap_or_else(|| "—".into());
    println!(
        "  {:<28}{:>12}{:>13}   {}",
        "tokens per reference found".bold(),
        fmt_thousands(s.cost_per_location_treatment.round() as u64).bold(),
        fmt_thousands(s.cost_per_location_control.round() as u64).bold(),
        if s.cost_ratio.is_some_and(|r| r < 1.0) {
            key_note.green().bold()
        } else {
            key_note.red().bold()
        }
    );

    // Per-session figures live in their own block so they are never mistaken
    // for inputs to the division above.
    println!();
    row(
        "searches run",
        format!("{:.1}", s.mean_search_calls_treatment),
        format!("{:.1}", s.mean_search_calls_control),
        String::new(),
    );
    row(
        "files opened afterwards",
        format!("{:.1}", s.mean_read_calls_treatment),
        format!("{:.1}", s.mean_read_calls_control),
        String::new(),
    );
    if let (Some(pt), Some(pc)) = (s.pct_of_project_treatment, s.pct_of_project_control) {
        row(
            "share of codebase read",
            format!("{:.2}%", pt),
            format!("{:.2}%", pc),
            String::new(),
        );
    }
    if let Some(p) = s.p_value {
        println!(
            "  {}",
            format!("(per-session cost, Mann-Whitney p={})", fmt_p(p)).dimmed()
        );
    }
}

/// Printed once, after every project, so the table needs no prior knowledge.
fn print_legend() {
    println!("{}", "What the rows mean".bold());
    let define = |term: &str, lines: &[&str]| {
        for (i, line) in lines.iter().enumerate() {
            if i == 0 {
                println!("  {:<28}{}", term, line);
            } else {
                println!("  {:<28}{}", "", line);
            }
        }
    };
    define(
        "tokens used to find code",
        &[
            "everything a search returned (colgrep, grep, rg, Grep/Glob),",
            "plus every file the agent opened afterwards to check a result",
        ],
    );
    define(
        "file:line references found",
        &[
            "distinct file:line spots the agent cited in its answers —",
            "a stand-in for how much of the answer it actually located",
        ],
    );
    define(
        "tokens per reference found",
        &[
            "the first row divided by the second — the number the verdict",
            "is based on. A tool that needs three searches to replace one",
            "grep costs more per reference even if each search is cheap.",
        ],
    );
}

/// One sentence naming *why* the headline came out the way it did. Without
/// this, a reader who sees similar token totals but a large per-place gap
/// reasonably suspects the report of contradicting itself.
fn explain(s: &SessionAbSummary) -> Option<String> {
    let ratio = s.cost_ratio?;
    let spend = safe_ratio(s.mean_find_cost_treatment, s.mean_find_cost_control);
    let spend_similar = (spend - 1.0).abs() < 0.15;
    let found_more = s.locations_treatment > s.locations_control;
    let cheaper = ratio < 1.0;

    let text = match (cheaper, spend_similar, found_more) {
        (true, true, true) => format!(
            "Both used about the same number of tokens looking, but colgrep found \
             {} references, so each one came cheaper.",
            fmt_multiple(s.locations_treatment as f64, s.locations_control as f64)
        ),
        (true, false, true) => "colgrep both spent less and found more.".to_string(),
        (true, false, false) => {
            "colgrep used fewer tokens, which more than covered finding slightly \
             fewer references."
                .to_string()
        }
        (false, true, false) => format!(
            "Both used about the same number of tokens looking, but colgrep found \
             {} references, so each one cost more.",
            fmt_multiple(s.locations_treatment as f64, s.locations_control as f64)
        ),
        (false, false, _) => format!(
            "colgrep used {} tokens looking, which the extra references it found \
             did not make up for.",
            fmt_delta_short(spend)
        ),
        _ => return None,
    };
    Some(text.dimmed().to_string())
}

fn safe_ratio(a: f64, b: f64) -> f64 {
    if b > 0.0 {
        a / b
    } else {
        f64::NAN
    }
}

/// Machine-readable form of exactly what the table shows, key for key, so a
/// dashboard built on this cannot print different numbers than the terminal.
fn to_json(projects: &[(String, Vec<SessionSample>)]) -> serde_json::Value {
    let rows: Vec<serde_json::Value> = projects
        .iter()
        .filter_map(|(path, samples)| {
            let s = sessions::summarize_sessions(samples)?;
            Some(serde_json::json!({
                "project_path": path,
                "sessions": { "with_colgrep": s.n_treatment, "without": s.n_control },
                // The headline, and the two rows that divide into it.
                "tokens_per_reference_found": {
                    "with_colgrep": s.cost_per_location_treatment,
                    "without": s.cost_per_location_control,
                    "ratio": s.cost_ratio,
                    "ratio_ci95": s.cost_ratio_ci.map(|(lo, hi)| vec![lo, hi]),
                },
                "per_session": {
                    "tokens_used_to_find_code": {
                        "with_colgrep": s.mean_find_cost_treatment,
                        "without": s.mean_find_cost_control,
                    },
                    "file_line_references_found": {
                        "with_colgrep": s.mean_locations_treatment,
                        "without": s.mean_locations_control,
                    },
                    "searches_run": {
                        "with_colgrep": s.mean_search_calls_treatment,
                        "without": s.mean_search_calls_control,
                    },
                    "files_opened_afterwards": {
                        "with_colgrep": s.mean_read_calls_treatment,
                        "without": s.mean_read_calls_control,
                    },
                    "share_of_codebase_read_pct": {
                        "with_colgrep": s.pct_of_project_treatment,
                        "without": s.pct_of_project_control,
                    },
                    "tokens_used_to_find_code_median": {
                        "with_colgrep": s.median_find_cost_treatment,
                        "without": s.median_find_cost_control,
                    },
                },
                "totals": {
                    "tokens_used_to_find_code": {
                        "with_colgrep": s.total_find_cost_treatment,
                        "without": s.total_find_cost_control,
                    },
                    "file_line_references_found": {
                        "with_colgrep": s.locations_treatment,
                        "without": s.locations_control,
                    },
                },
                "p_value": s.p_value,
                "contaminated_controls": s.contaminated_controls,
            }))
        })
        .collect();
    serde_json::json!({ "projects": rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_ratio_guards_a_zero_denominator() {
        assert_eq!(safe_ratio(10.0, 5.0), 2.0);
        assert!(safe_ratio(10.0, 0.0).is_nan());
    }
}
