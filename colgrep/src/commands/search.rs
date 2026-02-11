use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use colored::Colorize;

use colgrep::{
    acquire_index_lock, bre_to_ere, ensure_model, escape_literal_braces, find_parent_index,
    get_index_dir_for_project, get_vector_index_path, index_exists, is_text_format,
    path_contains_ignored_dir, Config, IndexBuilder, IndexState, Searcher, DEFAULT_MODEL,
};

use crate::display::{
    calc_display_ranges, find_representative_lines, group_results_by_file,
    print_highlighted_content, print_highlighted_ranges,
};
use crate::scoring::{compute_final_score, should_search_from_root};

/// Pre-compiled pattern matcher for efficient repeated matching.
/// Compiling regex is expensive (~microseconds), so we do it once and reuse.
enum PatternMatcher {
    /// Compiled regex for matching
    Regex(regex::Regex),
    /// Literal string for case-insensitive contains matching
    Literal(String),
}

impl PatternMatcher {
    /// Create a new pattern matcher based on the matching mode.
    /// - `extended_regexp`: Use ERE (extended regular expressions)
    /// - `fixed_strings`: Treat pattern as literal (overrides extended_regexp)
    /// - `word_regexp`: Add word boundaries
    fn new(pattern: &str, extended_regexp: bool, fixed_strings: bool, word_regexp: bool) -> Self {
        let effective_use_regex = extended_regexp && !fixed_strings;

        if effective_use_regex {
            let ere_pattern = escape_literal_braces(&bre_to_ere(pattern));
            let regex_pattern = if word_regexp {
                format!(r"\b{}\b", ere_pattern)
            } else {
                ere_pattern
            };
            match regex::RegexBuilder::new(&regex_pattern)
                .case_insensitive(true)
                .size_limit(10 * (1 << 20))
                .build()
            {
                Ok(re) => PatternMatcher::Regex(re),
                Err(_) => PatternMatcher::Literal(pattern.to_lowercase()),
            }
        } else if word_regexp {
            let escaped = regex::escape(pattern);
            let word_pattern = format!(r"\b{}\b", escaped);
            match regex::RegexBuilder::new(&word_pattern)
                .case_insensitive(true)
                .size_limit(10 * (1 << 20))
                .build()
            {
                Ok(re) => PatternMatcher::Regex(re),
                Err(_) => PatternMatcher::Literal(pattern.to_lowercase()),
            }
        } else {
            PatternMatcher::Literal(pattern.to_lowercase())
        }
    }

    /// Find matching line numbers within a code unit's content.
    /// Returns 1-indexed line numbers where matches were found.
    fn find_matches_in_unit(&self, unit: &colgrep::CodeUnit) -> Vec<usize> {
        match self {
            PatternMatcher::Regex(re) => {
                let matches: Vec<usize> = unit
                    .code
                    .lines()
                    .enumerate()
                    .filter_map(|(i, line)| {
                        if re.is_match(line) {
                            Some(unit.line + i)
                        } else {
                            None
                        }
                    })
                    .collect();

                // If regex matches nothing, fall back to literal match
                // (handles cases where user searches for regex metacharacters)
                if matches.is_empty() {
                    self.literal_fallback(unit)
                } else {
                    matches
                }
            }
            PatternMatcher::Literal(pattern_lower) => unit
                .code
                .lines()
                .enumerate()
                .filter_map(|(i, line)| {
                    if line.to_lowercase().contains(pattern_lower) {
                        Some(unit.line + i)
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    /// Literal fallback for when regex mode produces no matches.
    fn literal_fallback(&self, unit: &colgrep::CodeUnit) -> Vec<usize> {
        // Extract the original pattern from regex for fallback
        // This is a simplified fallback - just search the code directly
        let pattern_lower = match self {
            PatternMatcher::Regex(re) => re.as_str().to_lowercase(),
            PatternMatcher::Literal(p) => p.clone(),
        };

        unit.code
            .lines()
            .enumerate()
            .filter_map(|(i, line)| {
                if line.to_lowercase().contains(&pattern_lower) {
                    Some(unit.line + i)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Strip regex special characters from a pattern for use in semantic queries.
///
/// When combining a regex pattern with a semantic query, the regex metacharacters
/// (like `\s`, `\w`, `+`, `*`, etc.) have no semantic meaning and could negatively
/// affect the embedding quality. This function extracts only the meaningful text
/// content from a regex pattern.
///
/// Examples:
/// - `fn\s+\w+` -> `fn`
/// - `async\s+fn` -> `async fn`
/// - `Result<.*>` -> `Result`
/// - `foo|bar` -> `foo bar`
fn strip_regex_for_semantic(pattern: &str) -> String {
    let mut result = String::with_capacity(pattern.len());
    let mut chars = pattern.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            // Handle backslash escapes
            '\\' => {
                if let Some(&next) = chars.peek() {
                    match next {
                        // Common regex character classes - skip both backslash and class char
                        's' | 'S' | 'w' | 'W' | 'd' | 'D' | 'b' | 'B' | 'n' | 'r' | 't' => {
                            chars.next();
                            // Add a space to separate tokens (e.g., "fn\s+bar" -> "fn bar")
                            if !result.ends_with(' ') && !result.is_empty() {
                                result.push(' ');
                            }
                        }
                        // Escaped literal characters - keep the literal
                        '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '^' | '$'
                        | '|' | '\\' => {
                            chars.next();
                            result.push(next);
                        }
                        // Other escapes - skip the backslash, keep the char
                        _ => {
                            chars.next();
                            result.push(next);
                        }
                    }
                }
            }
            // Quantifiers and metacharacters - skip them
            '*' | '+' | '?' => {}
            // Anchors - skip them
            '^' | '$' => {}
            // Character class - skip entire [...] block
            #[allow(clippy::while_let_on_iterator)]
            '[' => {
                // Skip until we find the closing ]
                // Note: using while let because we need to call chars.next() inside for escaped chars
                let mut depth = 1;
                while let Some(inner) = chars.next() {
                    if inner == '\\' {
                        // Skip escaped char inside character class
                        chars.next();
                    } else if inner == '[' {
                        depth += 1;
                    } else if inner == ']' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
            }
            // Grouping - skip the parens but process contents
            '(' | ')' => {}
            // Alternation - convert to space
            '|' => {
                if !result.ends_with(' ') && !result.is_empty() {
                    result.push(' ');
                }
            }
            // Quantifier ranges {n,m} - skip entire block
            '{' => {
                for inner in chars.by_ref() {
                    if inner == '}' {
                        break;
                    }
                }
            }
            // Dot (any char) - skip
            '.' => {}
            // Regular characters - keep them
            _ => {
                result.push(c);
            }
        }
    }

    // Clean up multiple spaces and trim
    let cleaned: String = result.split_whitespace().collect::<Vec<_>>().join(" ");

    cleaned
}

/// Merge a semantic query with a sanitized regex pattern, removing duplicate tokens.
///
/// This prevents redundant tokens in the final query when the regex pattern
/// contains words that are already in the semantic query.
///
/// Example:
/// - query: "async function", pattern: "async fn" -> "async function fn"
/// - query: "error handling", pattern: "error" -> "error handling"
fn merge_query_with_pattern(query: &str, sanitized_pattern: &str) -> String {
    if sanitized_pattern.is_empty() {
        return query.to_string();
    }

    // Collect query tokens (lowercase for comparison)
    let query_tokens: std::collections::HashSet<String> =
        query.split_whitespace().map(|s| s.to_lowercase()).collect();

    // Filter pattern tokens to only include those not already in the query
    let new_tokens: Vec<&str> = sanitized_pattern
        .split_whitespace()
        .filter(|token| !query_tokens.contains(&token.to_lowercase()))
        .collect();

    if new_tokens.is_empty() {
        query.to_string()
    } else {
        format!("{} {}", query, new_tokens.join(" "))
    }
}

/// Resolve the model to use: CLI arg > saved config > default
pub fn resolve_model(cli_model: Option<&str>) -> String {
    if let Some(model) = cli_model {
        return model.to_string();
    }

    // Try to load from config
    if let Ok(config) = Config::load() {
        if let Some(model) = config.get_default_model() {
            return model.to_string();
        }
    }

    // Fall back to default
    DEFAULT_MODEL.to_string()
}

/// Resolve top_k: CLI arg > saved config > default
pub fn resolve_top_k(cli_k: Option<usize>, default: usize) -> usize {
    if let Some(k) = cli_k {
        return k;
    }

    // Try to load from config
    if let Ok(config) = Config::load() {
        if let Some(k) = config.get_default_k() {
            return k;
        }
    }

    default
}

/// Resolve context_lines (n): CLI arg > saved config > default
pub fn resolve_context_lines(cli_n: Option<usize>, default: usize) -> usize {
    if let Some(n) = cli_n {
        return n;
    }

    // Try to load from config
    if let Ok(config) = Config::load() {
        if let Some(n) = config.get_default_n() {
            return n;
        }
    }

    default
}

/// Resolve verbose: saved config > default (false)
pub fn resolve_verbose() -> bool {
    if let Ok(config) = Config::load() {
        return config.is_verbose();
    }
    false
}

/// Resolve pool_factor: --no-pool > --pool-factor > config > default (2)
pub fn resolve_pool_factor(cli_pool_factor: Option<usize>, no_pool: bool) -> Option<usize> {
    if no_pool {
        return Some(1); // Disable pooling
    }

    if let Some(factor) = cli_pool_factor {
        return Some(factor.max(1)); // Minimum is 1
    }

    // Try to load from config
    if let Ok(config) = Config::load() {
        return Some(config.get_pool_factor());
    }

    // Default pool factor
    Some(colgrep::DEFAULT_POOL_FACTOR)
}

#[allow(clippy::too_many_arguments)]
pub fn cmd_search(
    query: &str,
    paths: &[PathBuf],
    top_k: usize,
    cli_model: Option<&str>,
    json: bool,
    include_patterns: &[String],
    files_only: bool,
    show_content: bool,
    cli_context_lines: Option<usize>,
    text_pattern: Option<&str>,
    extended_regexp: bool,
    fixed_strings: bool,
    word_regexp: bool,
    exclude_patterns: &[String],
    exclude_dirs: &[String],
    code_only: bool,
    pool_factor: Option<usize>,
    auto_confirm: bool,
) -> Result<()> {
    // Resolve context_lines: CLI > config > default (20)
    let context_lines = resolve_context_lines(cli_context_lines, 20);
    // Collect results from all paths
    let mut all_results: Vec<colgrep::SearchResult> = Vec::new();
    let mut path_errors: Vec<String> = Vec::new();

    for path in paths {
        match search_single_path(
            query,
            path,
            top_k,
            cli_model,
            json,
            include_patterns,
            files_only,
            text_pattern,
            extended_regexp,
            fixed_strings,
            word_regexp,
            exclude_patterns,
            exclude_dirs,
            code_only,
            pool_factor,
            auto_confirm,
        ) {
            Ok(results) => all_results.extend(results),
            Err(e) => {
                let err_msg = format!("{}", e);
                // Check if this is a "path does not exist" error
                if err_msg.contains("Path does not exist:") {
                    // Store error message for later display, continue with other paths
                    path_errors.push(err_msg);
                } else {
                    // For other errors, fail immediately
                    return Err(e);
                }
            }
        }
    }

    // If ALL paths failed, return error with all messages
    if all_results.is_empty() && !path_errors.is_empty() {
        anyhow::bail!("{}", path_errors.join("\n\n"));
    }

    // Print warnings for failed paths (but we have some results)
    if !path_errors.is_empty() && !json && !files_only {
        for err in &path_errors {
            eprintln!("⚠️  {}\n", err);
        }
    }

    // Sort all results by score and take top_k
    all_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Filter out text/config files if --code-only is enabled, then take top_k
    let results: Vec<_> = if code_only {
        all_results
            .into_iter()
            .filter(|r| !is_text_format(r.unit.language))
            .take(top_k)
            .collect()
    } else {
        all_results.into_iter().take(top_k).collect()
    };

    // When -e is used without -F, automatically enable regex mode (ERE)
    let effective_extended_regexp = extended_regexp || (text_pattern.is_some() && !fixed_strings);

    // Output
    if files_only {
        // -l mode: show only unique filenames
        let mut seen_files = std::collections::HashSet::new();
        for result in &results {
            let file_str = result.unit.file.display().to_string();
            if seen_files.insert(file_str.clone()) {
                println!("{}", file_str);
            }
        }
    } else if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        if results.is_empty() {
            println!("No results found for: {}", query);
            return Ok(());
        }

        // Resolve verbose mode from config, but force verbose if -c or -n > 0 is used
        let verbose = if show_content || cli_context_lines.is_some_and(|n| n > 0) {
            true // Force verbose when user explicitly requests content display
        } else {
            resolve_verbose()
        };

        if !verbose {
            // Non-verbose (compact) mode: show filepath:lines (score: X.XX) ordered by score
            for result in &results {
                let file_path = result.unit.file.display();
                let start_line = result.unit.line;
                let end_line = result.unit.end_line;
                let score = result.score;
                println!(
                    "{}:{}-{} (score: {:.2})",
                    file_path, start_line, end_line, score
                );
            }
        } else {
            // Verbose mode: full content grouped by file with syntax highlighting

            // Pre-compile pattern matchers ONCE before the display loop.
            // This avoids expensive regex compilation on every result.
            let text_pattern_matcher = text_pattern.map(|p| {
                PatternMatcher::new(p, effective_extended_regexp, fixed_strings, word_regexp)
            });

            // For query matching (when no -e pattern), use literal matching
            let query_matcher = PatternMatcher::new(query, false, true, false);

            // Separate results into code files and documents/config files
            let (code_results, doc_results): (Vec<_>, Vec<_>) = results
                .iter()
                .partition(|r| !is_text_format(r.unit.language));

            let half_context = context_lines / 2;
            let has_text_pattern = text_pattern.is_some();

            // Calculate max line number across all results for consistent alignment
            let max_line_num = results.iter().map(|r| r.unit.end_line).max().unwrap_or(1);
            let line_num_width = max_line_num.to_string().len().max(4);

            // Display code results first, grouped by file
            if !code_results.is_empty() {
                let grouped = group_results_by_file(&code_results);
                for (file, file_results) in grouped {
                    // Print file header (file paths are absolute from search_single_path)
                    println!("file: {}", file.display().to_string().cyan());
                    for result in file_results {
                        let file_to_read = &result.unit.file;
                        if let Ok(content) = std::fs::read_to_string(file_to_read) {
                            let lines: Vec<&str> = content.lines().collect();
                            let end = result.unit.end_line.min(lines.len());
                            let max_lines = if show_content {
                                usize::MAX
                            } else {
                                context_lines
                            };

                            if has_text_pattern {
                                let file_matches = text_pattern_matcher
                                    .as_ref()
                                    .unwrap()
                                    .find_matches_in_unit(&result.unit);
                                let ranges = calc_display_ranges(
                                    &file_matches,
                                    result.unit.line,
                                    end,
                                    half_context,
                                    max_lines,
                                    true,
                                );
                                print_highlighted_ranges(
                                    file_to_read,
                                    &lines,
                                    &ranges,
                                    end,
                                    line_num_width,
                                );
                            } else {
                                let query_matches =
                                    query_matcher.find_matches_in_unit(&result.unit);
                                if !query_matches.is_empty() {
                                    let ranges = calc_display_ranges(
                                        &query_matches,
                                        result.unit.line,
                                        end,
                                        half_context,
                                        max_lines,
                                        true,
                                    );
                                    print_highlighted_ranges(
                                        file_to_read,
                                        &lines,
                                        &ranges,
                                        end,
                                        line_num_width,
                                    );
                                } else {
                                    // No exact match - find most representative line(s) based on token overlap
                                    let representative_lines = find_representative_lines(
                                        &result.unit.code,
                                        result.unit.line,
                                        query,
                                    );
                                    if !representative_lines.is_empty() {
                                        let ranges = calc_display_ranges(
                                            &representative_lines,
                                            result.unit.line,
                                            end,
                                            half_context,
                                            max_lines,
                                            true,
                                        );
                                        print_highlighted_ranges(
                                            file_to_read,
                                            &lines,
                                            &ranges,
                                            end,
                                            line_num_width,
                                        );
                                    } else {
                                        // Final fallback: show from beginning
                                        let start = result.unit.line.saturating_sub(1);
                                        if start < lines.len() {
                                            print_highlighted_content(
                                                file_to_read,
                                                &lines,
                                                start,
                                                max_lines,
                                                end,
                                                line_num_width,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    println!();
                }
            }

            // Display document/config results after, grouped by file
            if !doc_results.is_empty() {
                let grouped = group_results_by_file(&doc_results);
                for (file, file_results) in grouped {
                    println!("file: {}", file.display().to_string().cyan());
                    for result in file_results {
                        let file_to_read = &result.unit.file;
                        if let Ok(content) = std::fs::read_to_string(file_to_read) {
                            let lines: Vec<&str> = content.lines().collect();
                            let end = result.unit.end_line.min(lines.len());
                            let max_lines = if show_content { 250 } else { context_lines };

                            if has_text_pattern {
                                let file_matches = text_pattern_matcher
                                    .as_ref()
                                    .unwrap()
                                    .find_matches_in_unit(&result.unit);
                                let ranges = calc_display_ranges(
                                    &file_matches,
                                    result.unit.line,
                                    end,
                                    half_context,
                                    max_lines,
                                    true,
                                );
                                print_highlighted_ranges(
                                    file_to_read,
                                    &lines,
                                    &ranges,
                                    end,
                                    line_num_width,
                                );
                            } else {
                                let query_matches =
                                    query_matcher.find_matches_in_unit(&result.unit);
                                if !query_matches.is_empty() {
                                    let ranges = calc_display_ranges(
                                        &query_matches,
                                        result.unit.line,
                                        end,
                                        half_context,
                                        max_lines,
                                        true,
                                    );
                                    print_highlighted_ranges(
                                        file_to_read,
                                        &lines,
                                        &ranges,
                                        end,
                                        line_num_width,
                                    );
                                } else {
                                    // No exact match - find most representative line(s) based on token overlap
                                    let representative_lines = find_representative_lines(
                                        &result.unit.code,
                                        result.unit.line,
                                        query,
                                    );
                                    if !representative_lines.is_empty() {
                                        let ranges = calc_display_ranges(
                                            &representative_lines,
                                            result.unit.line,
                                            end,
                                            half_context,
                                            max_lines,
                                            true,
                                        );
                                        print_highlighted_ranges(
                                            file_to_read,
                                            &lines,
                                            &ranges,
                                            end,
                                            line_num_width,
                                        );
                                    } else {
                                        // Final fallback: show from beginning
                                        let start = result.unit.line.saturating_sub(1);
                                        if start < lines.len() {
                                            print_highlighted_content(
                                                file_to_read,
                                                &lines,
                                                start,
                                                max_lines,
                                                end,
                                                line_num_width,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}

/// Find the lowest existing parent directory and list its contents
fn find_existing_parent_and_list(path: &Path) -> String {
    let mut current = path.to_path_buf();

    // Walk up the path to find the first existing directory
    while !current.exists() {
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }

    // If we found an existing directory, list its contents
    if current.exists() && current.is_dir() {
        let mut entries: Vec<String> = Vec::new();
        if let Ok(dir_entries) = std::fs::read_dir(&current) {
            for entry in dir_entries.take(30).flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                if is_dir {
                    entries.push(format!("  {}/", name));
                } else {
                    entries.push(format!("  {}", name));
                }
            }
        }
        entries.sort();

        let suffix = if entries.len() >= 30 {
            "\n  ... (truncated)"
        } else {
            ""
        };

        format!(
            "Closest existing directory: {}\nContents:\n{}{}",
            current.display(),
            entries.join("\n"),
            suffix
        )
    } else {
        "Could not find any existing parent directory.".to_string()
    }
}

/// Search a single path and return results with absolute file paths
#[allow(clippy::too_many_arguments)]
fn search_single_path(
    query: &str,
    path: &PathBuf,
    top_k: usize,
    cli_model: Option<&str>,
    json: bool,
    include_patterns: &[String],
    files_only: bool,
    text_pattern: Option<&str>,
    extended_regexp: bool,
    fixed_strings: bool,
    word_regexp: bool,
    exclude_patterns: &[String],
    exclude_dirs: &[String],
    code_only: bool,
    pool_factor: Option<usize>,
    auto_confirm: bool,
) -> Result<Vec<colgrep::SearchResult>> {
    let path = match std::fs::canonicalize(path) {
        Ok(p) => p,
        Err(_) => {
            let help = find_existing_parent_and_list(path);
            anyhow::bail!("Path does not exist: {}\n\n{}", path.display(), help);
        }
    };

    // Check if path is a file (not a directory)
    // If so, we'll use the parent directory for indexing and filter to this specific file
    let (search_path, specific_file): (PathBuf, Option<PathBuf>) = if path.is_file() {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("File has no parent directory: {}", path.display()))?
            .to_path_buf();
        (parent, Some(path.clone()))
    } else {
        (path.clone(), None)
    };

    // When -e is used without -F, automatically enable regex mode (ERE)
    // This makes -e imply -E by default, with -F as the opt-out
    let effective_extended_regexp = extended_regexp || (text_pattern.is_some() && !fixed_strings);

    // Resolve model: CLI > config > default
    let model = resolve_model(cli_model);

    // Load config for settings
    let config = Config::load().unwrap_or_default();

    // Resolve quantized setting from config (default: false = use FP32)
    let quantized = !config.use_fp32();

    // Resolve parallel sessions and batch size from config
    let parallel_sessions = Some(config.get_parallel_sessions());
    let batch_size = Some(config.get_batch_size());

    // Check if index already exists (suppress model output if so)
    let has_existing_index =
        index_exists(&search_path) || find_parent_index(&search_path)?.is_some();

    // Ensure model is downloaded (quiet if we already have an index)
    let model_path = ensure_model(Some(&model), has_existing_index)?;

    // Check for parent index unless the resolved path is outside
    // the current directory (external project)
    let parent_info = {
        let is_external_project = std::env::current_dir()
            .map(|cwd| !search_path.starts_with(&cwd))
            .unwrap_or(false);

        if is_external_project {
            None
        } else {
            find_parent_index(&search_path)?
        }
    };

    // Determine effective project root and subdirectory filter
    let (effective_root, subdir_filter): (PathBuf, Option<PathBuf>) = match &parent_info {
        Some(info) => (
            info.project_path.clone(),
            Some(info.relative_subdir.clone()),
        ),
        None => (search_path.clone(), None),
    };

    // Check if --include patterns would escape the subdirectory
    // If so, search the full project (still within the same index - effective_root)
    // This does NOT escape to a different or parent index, it only removes the subdir restriction
    let subdir_filter = if let Some(ref subdir) = subdir_filter {
        if should_search_from_root(include_patterns, subdir, &effective_root) {
            if !json && !files_only {
                eprintln!("📂 Pattern escapes subdirectory, searching full project");
            }
            None // Skip subdir filter, search full index (still bounded by effective_root)
        } else {
            Some(subdir.clone())
        }
    } else {
        None
    };

    // Get files matching include patterns (for file-type filtering)
    // BUG FIX: Don't scan filesystem for --include patterns.
    // The filesystem scan finds files that aren't in the index, causing
    // filter_by_files() to return empty results. Instead, let the code
    // fall through to filter_by_file_patterns() which queries the index directly.
    let include_files: Option<Vec<String>> = None;

    // Auto-index: try incremental update without blocking on the lock.
    // If another process is indexing, skip the update and search the existing index.
    let mut index_locked = false;
    {
        let mut builder = IndexBuilder::with_options(
            &effective_root,
            &model_path,
            quantized,
            pool_factor,
            parallel_sessions,
            batch_size,
        )?;
        builder.set_auto_confirm(auto_confirm);
        builder.set_model_name(&model);

        // Try non-blocking index update
        match builder.try_index(None, false) {
            Ok(Some(stats)) => {
                let changes = stats.added + stats.changed + stats.deleted;
                if changes > 0 && !json && !files_only {
                    if let Some(ref info) = parent_info {
                        eprintln!(
                            "📂 Using index: {} (subdir: {}): indexed {} files\n",
                            info.project_path.display(),
                            info.relative_subdir.display(),
                            changes
                        );
                    } else {
                        eprintln!(
                            "📂 Using index: {}: indexed {} files\n",
                            effective_root.display(),
                            changes
                        );
                    }
                }
            }
            Ok(None) => {
                // Lock held by another process — search existing index
                index_locked = true;
                if !json && !files_only {
                    eprintln!(
                        "📂 Index is being updated by another process, searching existing index..."
                    );
                }
            }
            Err(e) => {
                let err_str = format!("{}", e);
                let err_debug = format!("{:?}", e);
                if err_str.contains("Indexing cancelled by user") {
                    return Err(e);
                }
                if err_str.contains("No data to merge")
                    || err_debug.contains("No data to merge")
                    || err_str.contains("Index load failed")
                {
                    // Index is corrupted - clear and rebuild
                    if !json && !files_only {
                        eprintln!("⚠️  Index corrupted, rebuilding...");
                    }

                    let index_dir = get_index_dir_for_project(&effective_root)?;
                    if index_dir.exists() {
                        let _lock = acquire_index_lock(&index_dir)?;
                        std::fs::remove_dir_all(&index_dir)?;
                    }

                    let mut new_builder = IndexBuilder::with_options(
                        &effective_root,
                        &model_path,
                        quantized,
                        pool_factor,
                        parallel_sessions,
                        batch_size,
                    )?;
                    new_builder.set_auto_confirm(auto_confirm);
                    new_builder.set_model_name(&model);
                    new_builder.index(None, false)?;
                } else {
                    return Err(e);
                }
            }
        }
    }

    // Verify index exists (at least partially)
    let index_dir = get_index_dir_for_project(&effective_root)?;
    let vector_index_path = get_vector_index_path(&index_dir);
    if !vector_index_path.join("metadata.json").exists() {
        if index_locked {
            // Index is being created for the first time by another process — nothing to search yet
            anyhow::bail!("colgrep index is currently being built, rely on grep for now.");
        }
        // Check if the path contains an ignored directory pattern
        if let Some(ignored_pattern) = path_contains_ignored_dir(&effective_root) {
            anyhow::bail!(
                "No files indexed. The path contains '{}' which is in the default ignore list.\n\
                 Ignored directories: tmp, temp, vendor, node_modules, target, build, dist, .git, etc.\n\
                 Try searching from a different directory or project root.",
                ignored_pattern
            );
        }
        anyhow::bail!("No index found. Index building may have failed (no indexable files found).");
    }

    // Load searcher (from parent index if applicable)
    // If loading fails while another process holds the lock, retry a few times in case
    // the failure is due to a transient mid-write state.
    // If loading fails without a concurrent updater, clear and rebuild the index.
    let load_searcher = || -> Result<Searcher> {
        match &parent_info {
            Some(info) => Searcher::load_from_index_dir_with_quantized(
                &info.index_dir,
                &model_path,
                quantized,
            ),
            None => Searcher::load_with_quantized(&effective_root, &model_path, quantized),
        }
    };

    let searcher = match load_searcher() {
        Ok(s) => s,
        Err(e) if index_locked => {
            // Another process is updating the index — the load failure is likely
            // due to a transient mid-write state. Retry a few times with short delays
            // rather than blocking on the lock (the updater may run for minutes).
            if !json && !files_only {
                eprintln!("⏳ Index load failed during update, retrying...");
            }
            const MAX_RETRIES: u32 = 3;
            const RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(500);
            let mut last_err = e;
            let mut loaded = None;
            for _ in 0..MAX_RETRIES {
                std::thread::sleep(RETRY_DELAY);
                match load_searcher() {
                    Ok(s) => {
                        loaded = Some(s);
                        break;
                    }
                    Err(e) => last_err = e,
                }
            }
            match loaded {
                Some(s) => s,
                None => {
                    return Err(last_err).with_context(|| {
                        "⏳ Index load failed while another process is updating. \
                         Rely on grep until the update completes."
                    });
                }
            }
        }
        Err(e)
            if {
                let err_debug = format!("{:?}", e);
                let err_display = format!("{}", e);
                err_debug.contains("No data to merge")
                    || err_display.contains("No data to merge")
                    || err_debug.contains("IndexLoad")
                    || err_display.contains("Index load failed")
            } =>
        {
            // Index is corrupted or empty (no concurrent updater) - clear and rebuild
            if !json && !files_only {
                eprintln!("⚠️  Index corrupted, rebuilding...");
            }

            let target_index_dir = match &parent_info {
                Some(info) => &info.index_dir,
                None => &index_dir,
            };
            if target_index_dir.exists() {
                let _lock = acquire_index_lock(target_index_dir)?;
                std::fs::remove_dir_all(target_index_dir)?;
            }

            let mut builder = IndexBuilder::with_options(
                &effective_root,
                &model_path,
                quantized,
                pool_factor,
                parallel_sessions,
                batch_size,
            )?;
            builder.set_auto_confirm(auto_confirm);
            builder.set_model_name(&model);
            builder.index(None, false)?;

            load_searcher()?
        }
        Err(e) => return Err(e),
    };

    // Build subset combining subdirectory filter, text pattern filter, and include patterns
    let subset = {
        let mut combined_ids: Option<Vec<i64>> = None;

        // Apply subdirectory filter first if using parent index
        if let Some(ref subdir) = subdir_filter {
            let subdir_ids = searcher.filter_by_path_prefix(subdir)?;
            if subdir_ids.is_empty() {
                if !json && !files_only {
                    eprintln!(
                        "No indexed code units in subdirectory: {}",
                        subdir.display()
                    );
                }
                return Ok(vec![]);
            }
            combined_ids = Some(subdir_ids);
        }

        // Apply text pattern filter: search indexed code directly (much faster than grep)
        if let Some(pattern) = text_pattern {
            // Use regex-based filtering with full grep flag support:
            // -e now implies ERE by default (no need for -E flag)
            // -F (fixed_strings): literal string matching, disables regex mode
            // -w (word_regexp): whole word matching with \b boundaries
            let pattern_ids = searcher.filter_by_text_pattern_with_options(
                pattern,
                effective_extended_regexp,
                fixed_strings,
                word_regexp,
            )?;

            if pattern_ids.is_empty() {
                if !json && !files_only {
                    eprintln!("No indexed code units contain pattern: {}", pattern);
                }
                return Ok(vec![]);
            }

            combined_ids = match combined_ids {
                Some(existing) => {
                    let existing_set: std::collections::HashSet<_> = existing.into_iter().collect();
                    Some(
                        pattern_ids
                            .into_iter()
                            .filter(|id| existing_set.contains(id))
                            .collect(),
                    )
                }
                None => Some(pattern_ids),
            };
        }

        // Apply include pattern filter (file type filtering)
        // Only use filesystem-scanned files if non-empty; otherwise fall back to index-based pattern matching
        if let Some(files) = include_files.as_ref().filter(|f| !f.is_empty()) {
            let file_ids = searcher.filter_by_files(files)?;
            combined_ids = match combined_ids {
                Some(existing) => {
                    let existing_set: std::collections::HashSet<_> = existing.into_iter().collect();
                    Some(
                        file_ids
                            .into_iter()
                            .filter(|id| existing_set.contains(id))
                            .collect(),
                    )
                }
                None => Some(file_ids),
            };
        } else if !include_patterns.is_empty() {
            let pattern_ids = searcher.filter_by_file_patterns(include_patterns)?;
            combined_ids = match combined_ids {
                Some(existing) => {
                    let existing_set: std::collections::HashSet<_> = existing.into_iter().collect();
                    Some(
                        pattern_ids
                            .into_iter()
                            .filter(|id| existing_set.contains(id))
                            .collect(),
                    )
                }
                None => Some(pattern_ids),
            };
        }

        // Apply specific file filter (when user passes a file path instead of directory)
        if let Some(ref file_path) = specific_file {
            // Convert absolute file path to relative path (relative to effective_root)
            let rel_path = file_path
                .strip_prefix(&effective_root)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();
            let file_ids = searcher.filter_by_files(std::slice::from_ref(&rel_path))?;
            if file_ids.is_empty() {
                if !json && !files_only {
                    eprintln!("No indexed code units in file: {}", file_path.display());
                }
                return Ok(vec![]);
            }
            combined_ids = match combined_ids {
                Some(existing) => {
                    let existing_set: std::collections::HashSet<_> = existing.into_iter().collect();
                    Some(
                        file_ids
                            .into_iter()
                            .filter(|id| existing_set.contains(id))
                            .collect(),
                    )
                }
                None => Some(file_ids),
            };
        }

        // Apply exclude pattern filter (SQL-based: returns IDs that DON'T match patterns)
        if !exclude_patterns.is_empty() {
            let included_ids = searcher.filter_exclude_by_patterns(exclude_patterns)?;
            let included_set: std::collections::HashSet<_> = included_ids.into_iter().collect();
            combined_ids = match combined_ids {
                Some(existing) => Some(
                    existing
                        .into_iter()
                        .filter(|id| included_set.contains(id))
                        .collect(),
                ),
                None => Some(included_set.into_iter().collect()),
            };
        }

        // Apply exclude-dir filter (SQL-based: returns IDs NOT in excluded directories)
        if !exclude_dirs.is_empty() {
            let included_ids = searcher.filter_exclude_by_dirs(exclude_dirs)?;
            let included_set: std::collections::HashSet<_> = included_ids.into_iter().collect();
            combined_ids = match combined_ids {
                Some(existing) => Some(
                    existing
                        .into_iter()
                        .filter(|id| included_set.contains(id))
                        .collect(),
                ),
                None => Some(included_set.into_iter().collect()),
            };
        }

        // Check if subset is empty after combining
        if let Some(ref ids) = combined_ids {
            if ids.is_empty() {
                if !json && !files_only {
                    eprintln!("No indexed code units match the specified filters");
                }
                return Ok(vec![]);
            }
        }

        combined_ids
    };

    // Search with optional filtering
    // Request more results to allow for re-ranking with query boost and test function demotion
    let search_top_k = if code_only { top_k * 4 } else { top_k * 3 };

    // When no -e flag is provided, run BOTH semantic search and hybrid search (query as text pattern)
    // This ensures exact matches are found even if the vector database doesn't rank them highly
    let results = if let Some(pattern) = &text_pattern {
        // -e flag provided: use existing hybrid search logic
        // Enhance semantic query with -e pattern (strip regex metacharacters and dedupe tokens)
        let sanitized_pattern = strip_regex_for_semantic(pattern);
        let enhanced_query = merge_query_with_pattern(query, &sanitized_pattern);
        searcher.search(&enhanced_query, search_top_k, subset.as_deref())?
    } else {
        // 1. Run pure semantic search
        let semantic_results = searcher.search(query, search_top_k, subset.as_deref())?;

        // 2. Run hybrid search: filter by query text, then semantic rank
        // Use fixed_strings mode to treat the query as a literal pattern
        let text_filtered_ids =
            searcher.filter_by_text_pattern_with_options(query, false, true, false)?;

        let hybrid_results = if !text_filtered_ids.is_empty() {
            // Intersect with existing subset if any
            let hybrid_subset: Vec<i64> = match &subset {
                Some(existing) => {
                    let existing_set: std::collections::HashSet<_> =
                        existing.iter().copied().collect();
                    text_filtered_ids
                        .into_iter()
                        .filter(|id| existing_set.contains(id))
                        .collect()
                }
                None => text_filtered_ids,
            };

            if !hybrid_subset.is_empty() {
                searcher.search(query, search_top_k, Some(&hybrid_subset))?
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // 3. Merge results: keep max score for each unique code unit (by file + line)
        let mut merged: HashMap<(PathBuf, usize), colgrep::SearchResult> = HashMap::new();

        for result in semantic_results {
            let key = (result.unit.file.clone(), result.unit.line);
            merged
                .entry(key)
                .and_modify(|existing| {
                    if result.score > existing.score {
                        *existing = result.clone();
                    }
                })
                .or_insert(result);
        }

        for result in hybrid_results {
            let key = (result.unit.file.clone(), result.unit.line);
            merged
                .entry(key)
                .and_modify(|existing| {
                    if result.score > existing.score {
                        *existing = result.clone();
                    }
                })
                .or_insert(result);
        }

        merged.into_values().collect::<Vec<_>>()
    };

    // Note: When -e is used, results are already filtered to units containing the pattern
    // via filter_by_text_pattern_with_options() above, which supports -E, -F, -w flags

    // Apply query boost and re-sort results
    let mut results: Vec<_> = results
        .into_iter()
        .map(|mut r| {
            r.score = compute_final_score(r.score, query, &r.unit, text_pattern);
            r
        })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Increment search count
    let index_dir = get_index_dir_for_project(&effective_root)?;
    if let Ok(mut state) = IndexState::load(&index_dir) {
        state.increment_search_count();
        let _ = state.save(&index_dir);
    }

    // Convert file paths to absolute for proper display when merging results from multiple paths
    let results: Vec<colgrep::SearchResult> = results
        .into_iter()
        .map(|mut r| {
            if !r.unit.file.is_absolute() {
                r.unit.file = effective_root.join(&r.unit.file);
            }
            r
        })
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test resolve_top_k function
    #[test]
    fn test_resolve_top_k_cli_provided() {
        // CLI value should take precedence
        assert_eq!(resolve_top_k(Some(30), 15), 30);
        assert_eq!(resolve_top_k(Some(1), 20), 1);
        assert_eq!(resolve_top_k(Some(100), 15), 100);
    }

    #[test]
    fn test_resolve_top_k_fallback_to_default() {
        // When CLI not provided and no config, should use default
        // Note: This test may be affected by actual config file
        let result = resolve_top_k(None, 15);
        // Should be either 25 (default) or whatever is in config
        assert!(result > 0);
    }

    // Test resolve_context_lines function
    #[test]
    fn test_resolve_context_lines_cli_provided() {
        // CLI value should take precedence
        assert_eq!(resolve_context_lines(Some(10), 20), 10);
        assert_eq!(resolve_context_lines(Some(0), 20), 0);
        assert_eq!(resolve_context_lines(Some(30), 20), 30);
    }

    #[test]
    fn test_resolve_context_lines_fallback_to_default() {
        // When CLI not provided and no config, should use default
        let result = resolve_context_lines(None, 20);
        // Should be either 20 (default) or whatever is in config
        assert!(result <= 100); // sanity check
    }

    // Test strip_regex_for_semantic function
    #[test]
    fn test_strip_regex_basic_patterns() {
        // Character classes should be stripped, leaving meaningful text
        assert_eq!(strip_regex_for_semantic(r"fn\s+\w+"), "fn");
        assert_eq!(strip_regex_for_semantic(r"async\s+fn"), "async fn");
        assert_eq!(strip_regex_for_semantic(r"\btest\b"), "test");
    }

    #[test]
    fn test_strip_regex_quantifiers() {
        // Quantifiers should be stripped
        assert_eq!(strip_regex_for_semantic("foo+"), "foo");
        assert_eq!(strip_regex_for_semantic("bar*"), "bar");
        assert_eq!(strip_regex_for_semantic("baz?"), "baz");
        assert_eq!(strip_regex_for_semantic("qux{2,5}"), "qux");
    }

    #[test]
    fn test_strip_regex_alternation() {
        // Alternation should become space-separated
        assert_eq!(strip_regex_for_semantic("foo|bar"), "foo bar");
        assert_eq!(strip_regex_for_semantic("a|b|c"), "a b c");
    }

    #[test]
    fn test_strip_regex_anchors() {
        // Anchors should be stripped
        assert_eq!(strip_regex_for_semantic("^start"), "start");
        assert_eq!(strip_regex_for_semantic("end$"), "end");
        assert_eq!(strip_regex_for_semantic("^both$"), "both");
    }

    #[test]
    fn test_strip_regex_character_classes() {
        // Character class brackets should be stripped entirely
        assert_eq!(strip_regex_for_semantic("[abc]"), "");
        assert_eq!(strip_regex_for_semantic("pre[abc]post"), "prepost");
        assert_eq!(strip_regex_for_semantic("[a-z]+"), "");
    }

    #[test]
    fn test_strip_regex_groups() {
        // Grouping parens should be stripped but contents kept
        assert_eq!(strip_regex_for_semantic("(foo)"), "foo");
        assert_eq!(strip_regex_for_semantic("(foo)(bar)"), "foobar");
    }

    #[test]
    fn test_strip_regex_escaped_literals() {
        // Escaped metacharacters should become literals
        assert_eq!(strip_regex_for_semantic(r"foo\.bar"), "foo.bar");
        assert_eq!(strip_regex_for_semantic(r"a\*b"), "a*b");
        assert_eq!(strip_regex_for_semantic(r"Result\<T\>"), "Result<T>");
    }

    #[test]
    fn test_strip_regex_dots() {
        // Dots (any char) should be stripped
        assert_eq!(strip_regex_for_semantic("a.b"), "ab");
        assert_eq!(strip_regex_for_semantic("Result<.*>"), "Result<>");
    }

    #[test]
    fn test_strip_regex_plain_text() {
        // Plain text should pass through unchanged
        assert_eq!(strip_regex_for_semantic("hello"), "hello");
        assert_eq!(strip_regex_for_semantic("hello world"), "hello world");
        assert_eq!(strip_regex_for_semantic("foo_bar"), "foo_bar");
    }

    #[test]
    fn test_strip_regex_complex_patterns() {
        // Complex real-world patterns
        assert_eq!(strip_regex_for_semantic(r"impl\s+\w+\s+for"), "impl for");
        assert_eq!(strip_regex_for_semantic(r"fn\s+test_\w+"), "fn test_");
        assert_eq!(
            strip_regex_for_semantic(r"pub\s+(async\s+)?fn"),
            "pub async fn"
        );
    }

    #[test]
    fn test_strip_regex_empty_result() {
        // Patterns that result in empty string
        assert_eq!(strip_regex_for_semantic(r"\s+"), "");
        assert_eq!(strip_regex_for_semantic(r"\w+"), "");
        assert_eq!(strip_regex_for_semantic(r".*"), "");
        assert_eq!(strip_regex_for_semantic(r"[a-z]+"), "");
    }

    // Test merge_query_with_pattern function
    #[test]
    fn test_merge_query_no_duplicates() {
        // No duplicates - all pattern tokens added
        assert_eq!(
            merge_query_with_pattern("error handling", "Result"),
            "error handling Result"
        );
        assert_eq!(
            merge_query_with_pattern("function", "async fn"),
            "function async fn"
        );
    }

    #[test]
    fn test_merge_query_with_duplicates() {
        // Duplicates should be removed (case-insensitive)
        assert_eq!(
            merge_query_with_pattern("async function", "async fn"),
            "async function fn"
        );
        assert_eq!(
            merge_query_with_pattern("error handling", "error"),
            "error handling"
        );
        assert_eq!(
            merge_query_with_pattern("Error Handling", "error handling"),
            "Error Handling"
        );
    }

    #[test]
    fn test_merge_query_all_duplicates() {
        // All pattern tokens are duplicates - just return query
        assert_eq!(merge_query_with_pattern("foo bar", "foo bar"), "foo bar");
        assert_eq!(merge_query_with_pattern("FOO BAR", "foo bar"), "FOO BAR");
    }

    #[test]
    fn test_merge_query_empty_pattern() {
        // Empty pattern - just return query
        assert_eq!(merge_query_with_pattern("query", ""), "query");
    }

    #[test]
    fn test_merge_query_partial_duplicates() {
        // Mix of duplicate and new tokens
        assert_eq!(
            merge_query_with_pattern("impl trait", "impl for"),
            "impl trait for"
        );
        assert_eq!(
            merge_query_with_pattern("pub fn", "pub async fn test"),
            "pub fn async test"
        );
    }
}
