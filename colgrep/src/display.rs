use std::collections::HashMap;
use std::path::{Path, PathBuf};

use colored::Colorize;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};

/// Maximum visible characters per line before truncation.
/// This is generous enough for normal code but prevents issues with
/// minified/obfuscated files that have extremely long lines.
pub const MAX_LINE_WIDTH: usize = 400;

/// Common programming keywords and stop words that should have reduced weight.
/// These appear frequently and are less discriminative for finding relevant code.
const STOP_WORDS: &[&str] = &[
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "and",
    "but",
    "or",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "not",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "that",
    "this",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "if",
    "then",
    "else",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "function",
    "method",
    "class",
    "struct",
    "enum",
    "type",
    "interface",
    "public",
    "private",
    "protected",
    "static",
    "const",
    "let",
    "var",
    "return",
    "true",
    "false",
    "null",
    "none",
    "nil",
    "void",
    "new",
    "delete",
    "get",
    "set",
    "add",
    "remove",
    "code",
    "logic",
    "implementation",
    "handle",
    "process",
];

/// Split an identifier (camelCase, PascalCase, snake_case, kebab-case) into components.
fn split_identifier(s: &str) -> Vec<String> {
    let mut components = Vec::new();
    let mut current = String::new();
    let mut prev_was_lower = false;

    for c in s.chars() {
        if c == '_' || c == '-' || c == '.' || c == '/' {
            // Separator - flush current component
            if !current.is_empty() {
                components.push(current.to_lowercase());
                current.clear();
            }
            prev_was_lower = false;
        } else if c.is_uppercase() && prev_was_lower {
            // camelCase boundary - flush and start new
            if !current.is_empty() {
                components.push(current.to_lowercase());
                current.clear();
            }
            current.push(c.to_ascii_lowercase());
            prev_was_lower = false;
        } else if c.is_alphanumeric() {
            current.push(c.to_ascii_lowercase());
            prev_was_lower = c.is_lowercase();
        } else {
            // Other character - flush
            if !current.is_empty() {
                components.push(current.to_lowercase());
                current.clear();
            }
            prev_was_lower = false;
        }
    }

    if !current.is_empty() {
        components.push(current);
    }

    components
}

/// Tokenize a query string into weighted tokens.
/// Returns (token, weight) pairs where weight is based on token length and rarity.
fn tokenize_query_weighted(query: &str) -> Vec<(String, f32)> {
    let components: Vec<String> = query
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .flat_map(split_identifier)
        .filter(|s| s.len() >= 2)
        .collect();

    components
        .into_iter()
        .map(|token| {
            // Weight by length (longer tokens are more specific)
            let length_weight = (token.len() as f32 / 4.0).clamp(0.5, 2.0);

            // Reduce weight for stop words
            let stop_word_factor = if STOP_WORDS.contains(&token.as_str()) {
                0.2
            } else {
                1.0
            };

            let weight = length_weight * stop_word_factor;
            (token, weight)
        })
        .filter(|(_, w)| *w > 0.1) // Filter out very low weight tokens
        .collect()
}

/// Check if a line contains a token, supporting:
/// - Exact substring match
/// - Identifier component match (camelCase, snake_case splitting)
///
/// Returns a score based on match quality (higher = better match)
fn token_match_score(line: &str, token: &str) -> f32 {
    let line_lower = line.to_lowercase();

    // Exact substring match (best)
    if line_lower.contains(token) {
        return 1.0;
    }

    // Split line into identifier components and check for matches
    let line_components: Vec<String> = line
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .flat_map(split_identifier)
        .collect();

    // Check for exact component match
    if line_components.iter().any(|comp| comp == token) {
        return 0.9;
    }

    // Check for prefix match (e.g., "auth" matches "authentication")
    if line_components
        .iter()
        .any(|comp| comp.starts_with(token) && comp.len() <= token.len() + 4)
    {
        return 0.7;
    }

    // Check for suffix match (e.g., "handler" matches "ErrorHandler")
    if line_components
        .iter()
        .any(|comp| comp.ends_with(token) && comp.len() <= token.len() + 4)
    {
        return 0.6;
    }

    // Check for substring in components (e.g., "error" in "handleerrors")
    if line_components.iter().any(|comp| comp.contains(token)) {
        return 0.5;
    }

    0.0
}

/// Score a window of lines (for context-aware matching).
/// Bonus for consecutive lines with matches.
fn score_line_window(lines: &[&str], tokens: &[(String, f32)]) -> f32 {
    let mut total_score = 0.0;
    let mut consecutive_match_count = 0;

    for line in lines {
        let mut line_score = 0.0;
        let mut matched_tokens = 0;

        for (token, weight) in tokens {
            let match_score = token_match_score(line, token);
            if match_score > 0.0 {
                line_score += match_score * weight;
                matched_tokens += 1;
            }
        }

        // Bonus for multiple tokens matching on same line
        if matched_tokens > 1 {
            line_score *= 1.0 + (matched_tokens as f32 - 1.0) * 0.3;
        }

        if line_score > 0.0 {
            consecutive_match_count += 1;
            total_score += line_score;
        } else {
            consecutive_match_count = 0;
        }
    }

    // Bonus for consecutive matching lines
    if consecutive_match_count > 1 {
        total_score *= 1.0 + (consecutive_match_count as f32 - 1.0) * 0.2;
    }

    total_score
}

/// Find the most representative line(s) in a code unit for a semantic query.
/// Uses a sliding window approach with weighted token matching.
/// Returns line numbers (1-indexed) of the most relevant lines.
pub fn find_representative_lines(code: &str, unit_start_line: usize, query: &str) -> Vec<usize> {
    let tokens = tokenize_query_weighted(query);
    if tokens.is_empty() {
        return vec![];
    }

    let lines: Vec<&str> = code.lines().collect();
    if lines.is_empty() {
        return vec![];
    }

    // Use a sliding window of 3 lines for context-aware scoring
    let window_size = 3.min(lines.len());
    let mut best_score = 0.0;
    let mut best_center_lines: Vec<usize> = vec![];

    for i in 0..=lines.len().saturating_sub(window_size) {
        let window = &lines[i..i + window_size];
        let score = score_line_window(window, &tokens);

        if score > best_score + 0.01 {
            // New best (with small epsilon for float comparison)
            best_score = score;
            // Center line of the window
            let center_idx = i + window_size / 2;
            best_center_lines = vec![unit_start_line + center_idx];
        } else if (score - best_score).abs() < 0.01 && score > 0.0 {
            // Tie - add this center line too
            let center_idx = i + window_size / 2;
            let line_num = unit_start_line + center_idx;
            if !best_center_lines.contains(&line_num) {
                best_center_lines.push(line_num);
            }
        }
    }

    // Also do single-line scoring for cases where one line has very high relevance
    let mut single_line_scores: Vec<(usize, f32)> = lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let score = tokens
                .iter()
                .map(|(token, weight)| token_match_score(line, token) * weight)
                .sum::<f32>();
            (unit_start_line + i, score)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();

    single_line_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // If best single line score is significantly higher than window score, prefer it
    if let Some((best_line, best_single_score)) = single_line_scores.first() {
        if *best_single_score > best_score * 0.8 && !best_center_lines.contains(best_line) {
            best_center_lines.insert(0, *best_line);
        }
    }

    // Return empty if score is too low (no meaningful match)
    if best_score < 0.3 {
        return vec![];
    }

    // Limit to top 3 representative lines
    best_center_lines.truncate(3);
    best_center_lines
}

/// Calculate merged display ranges for all matches within a code unit
/// Returns a vector of (start, end) ranges (0-indexed) that cover all matches with context
/// If include_signature is true, always includes the function signature (first line of unit)
pub fn calc_display_ranges(
    match_lines: &[usize],
    unit_start: usize,
    unit_end: usize,
    half_context: usize,
    max_lines: usize,
    include_signature: bool,
) -> Vec<(usize, usize)> {
    let signature_line = unit_start.saturating_sub(1); // 0-indexed first line of unit

    if match_lines.is_empty() {
        // No matches, show from beginning with max_lines limit
        let end = unit_end.min(signature_line + max_lines);
        return vec![(signature_line, end)];
    }

    // Filter matches within the unit range and sort
    let mut matches_in_range: Vec<usize> = match_lines
        .iter()
        .filter(|&&line| line >= unit_start && line <= unit_end)
        .copied()
        .collect();
    matches_in_range.sort();

    if matches_in_range.is_empty() {
        // No matches in range, show from beginning
        let end = unit_end.min(signature_line + max_lines);
        return vec![(signature_line, end)];
    }

    // Calculate ranges for each match (with context)
    // When not including signature, allow ranges to start before unit_start
    let min_start = if include_signature { signature_line } else { 0 };
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for &match_line in &matches_in_range {
        let start = match_line
            .saturating_sub(1)
            .saturating_sub(half_context)
            .max(min_start);
        let end = (match_line.saturating_sub(1) + half_context + 1).min(unit_end);
        ranges.push((start, end));
    }

    // Merge overlapping ranges
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                // Overlapping or adjacent, merge
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        } else {
            merged.push((start, end));
        }
    }

    // Ensure signature line is always included (only when include_signature is true)
    // If first range doesn't start at signature, prepend a signature-only range
    if include_signature {
        if let Some(first) = merged.first() {
            if first.0 > signature_line {
                // Add signature line as separate range (just the first line or two)
                let sig_end = (signature_line + 2).min(first.0); // Show 1-2 lines of signature
                merged.insert(0, (signature_line, sig_end));
            }
        }
    }

    merged
}

/// Truncate a string containing ANSI escape codes to a maximum visible width.
/// Returns the truncated string with "..." appended if truncation occurred.
pub fn truncate_ansi_string(s: &str, max_width: usize) -> String {
    let mut visible_count = 0;
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Start of ANSI escape sequence - copy it entirely
            result.push(c);
            // Copy until we hit 'm' (end of color code) or run out of chars
            while let Some(&next) = chars.peek() {
                result.push(chars.next().unwrap());
                if next == 'm' {
                    break;
                }
            }
        } else {
            // Regular visible character
            if visible_count >= max_width {
                // Truncation point reached
                result.push_str("\x1b[0m..."); // Reset color and add ellipsis
                return result;
            }
            result.push(c);
            visible_count += 1;
        }
    }

    result
}

/// Print content with syntax highlighting for multiple ranges
pub fn print_highlighted_ranges(
    file_path: &Path,
    lines: &[&str],
    ranges: &[(usize, usize)],
    unit_end: usize,
    line_num_width: usize,
) {
    let ps = SyntaxSet::load_defaults_newlines();
    let ts = ThemeSet::load_defaults();
    let theme = &ts.themes["base16-ocean.dark"];

    // Try to detect syntax from file extension
    let syntax = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| ps.find_syntax_by_extension(ext))
        .unwrap_or_else(|| ps.find_syntax_plain_text());

    for (range_idx, &(start, end)) in ranges.iter().enumerate() {
        let display_end = end.min(lines.len());
        let display_start = start.min(lines.len());

        if display_start >= lines.len() {
            continue;
        }

        // Reconstruct the content for highlighting
        let content_to_highlight: String = lines[display_start..display_end]
            .iter()
            .map(|l| format!("{}\n", l))
            .collect();

        let mut highlighter = HighlightLines::new(syntax, theme);

        for (i, line) in LinesWithEndings::from(&content_to_highlight).enumerate() {
            let line_num = display_start + i + 1;
            let ranges: Vec<(Style, &str)> = highlighter
                .highlight_line(line, &ps)
                .unwrap_or_else(|_| vec![(Style::default(), line)]);
            let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
            // Remove trailing newline for cleaner output
            let escaped = escaped.trim_end_matches('\n');
            // Truncate very long lines (e.g., minified JS)
            let escaped = truncate_ansi_string(escaped, MAX_LINE_WIDTH);
            println!(
                "{} {}\x1b[0m",
                format!("{:>width$}", line_num, width = line_num_width).dimmed(),
                escaped
            );
        }

        // Add separator between ranges, or "..." if more content follows
        if range_idx < ranges.len() - 1 || display_end < unit_end {
            println!("{}", "...".dimmed());
        }
    }
}

/// Print content with syntax highlighting (single range, legacy)
pub fn print_highlighted_content(
    file_path: &Path,
    lines: &[&str],
    start_line: usize,
    max_lines: usize,
    end_line: usize,
    line_num_width: usize,
) {
    let ps = SyntaxSet::load_defaults_newlines();
    let ts = ThemeSet::load_defaults();
    let theme = &ts.themes["base16-ocean.dark"];

    // Try to detect syntax from file extension
    let syntax = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| ps.find_syntax_by_extension(ext))
        .unwrap_or_else(|| ps.find_syntax_plain_text());

    let mut highlighter = HighlightLines::new(syntax, theme);

    let display_end = end_line.min(start_line.saturating_add(max_lines));
    let truncated = end_line > display_end;

    // Reconstruct the content for highlighting
    let content_to_highlight: String = lines[start_line..display_end]
        .iter()
        .map(|l| format!("{}\n", l))
        .collect();

    for (i, line) in LinesWithEndings::from(&content_to_highlight).enumerate() {
        let line_num = start_line + i + 1;
        let ranges: Vec<(Style, &str)> = highlighter
            .highlight_line(line, &ps)
            .unwrap_or_else(|_| vec![(Style::default(), line)]);
        let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
        // Remove trailing newline for cleaner output
        let escaped = escaped.trim_end_matches('\n');
        // Truncate very long lines (e.g., minified JS)
        let escaped = truncate_ansi_string(escaped, MAX_LINE_WIDTH);
        println!(
            "{} {}\x1b[0m",
            format!("{:>width$}", line_num, width = line_num_width).dimmed(),
            escaped
        );
    }

    if truncated {
        println!("{}", "...".dimmed());
    }
}

/// Group results by file, maintaining relevance order for files
/// Files are ordered by their most relevant result, and within each file,
/// results are sorted by line number (position in file)
pub fn group_results_by_file<'a>(
    results: &'a [&colgrep::SearchResult],
) -> Vec<(PathBuf, Vec<&'a colgrep::SearchResult>)> {
    let mut file_order: Vec<PathBuf> = Vec::new();
    let mut file_results: HashMap<PathBuf, Vec<&'a colgrep::SearchResult>> = HashMap::new();

    for result in results {
        let file = result.unit.file.clone();
        if !file_results.contains_key(&file) {
            file_order.push(file.clone());
        }
        file_results.entry(file).or_default().push(result);
    }

    file_order
        .into_iter()
        .filter_map(|file| {
            file_results.remove(&file).map(|mut results| {
                // Sort results by line number within each file
                results.sort_by_key(|r| r.unit.line);
                (file, results)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test calc_display_ranges function
    #[test]
    fn test_calc_display_ranges_no_matches() {
        let ranges = calc_display_ranges(&[], 10, 20, 3, 6, true);
        // Should show from beginning with max_lines limit
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (9, 15)); // signature_line=9, end=min(20, 9+6)=15
    }

    #[test]
    fn test_calc_display_ranges_single_match() {
        let match_lines = vec![15];
        let ranges = calc_display_ranges(&match_lines, 10, 25, 3, 10, true);
        // Should have signature and match context
        assert!(!ranges.is_empty());
    }

    #[test]
    fn test_calc_display_ranges_multiple_matches_merged() {
        // Two matches close enough to merge
        let match_lines = vec![12, 14];
        let ranges = calc_display_ranges(&match_lines, 10, 30, 3, 20, true);
        // Ranges should be merged since they're close together
        assert!(ranges.len() <= 2);
    }

    #[test]
    fn test_calc_display_ranges_matches_outside_unit() {
        // Matches outside the unit range should be filtered
        let match_lines = vec![5, 35]; // Both outside 10-25
        let ranges = calc_display_ranges(&match_lines, 10, 25, 3, 10, true);
        // Should fall back to showing from beginning
        assert!(!ranges.is_empty());
    }

    // Test split_identifier function
    #[test]
    fn test_split_identifier_snake_case() {
        let components = split_identifier("find_user_by_email");
        assert_eq!(components, vec!["find", "user", "by", "email"]);
    }

    #[test]
    fn test_split_identifier_camel_case() {
        let components = split_identifier("findUserByEmail");
        assert_eq!(components, vec!["find", "user", "by", "email"]);
    }

    #[test]
    fn test_split_identifier_pascal_case() {
        let components = split_identifier("ErrorHandler");
        assert_eq!(components, vec!["error", "handler"]);
    }

    #[test]
    fn test_split_identifier_mixed() {
        let components = split_identifier("HTTP_RequestHandler");
        assert_eq!(components, vec!["http", "request", "handler"]);
    }

    // Test tokenize_query_weighted function
    #[test]
    fn test_tokenize_query_weighted_basic() {
        let tokens = tokenize_query_weighted("error handling");
        let token_names: Vec<&str> = tokens.iter().map(|(t, _)| t.as_str()).collect();
        assert!(token_names.contains(&"error"));
        // "handling" might be split or kept depending on implementation
    }

    #[test]
    fn test_tokenize_query_weighted_stop_words() {
        let tokens = tokenize_query_weighted("the function that handles errors");
        // Stop words should have lower weight
        let the_weight = tokens.iter().find(|(t, _)| t == "the").map(|(_, w)| *w);
        let errors_weight = tokens.iter().find(|(t, _)| t == "errors").map(|(_, w)| *w);
        if let (Some(tw), Some(ew)) = (the_weight, errors_weight) {
            assert!(
                tw < ew,
                "Stop word 'the' should have lower weight than 'errors'"
            );
        }
    }

    // Test token_match_score function
    #[test]
    fn test_token_match_score_exact() {
        let score = token_match_score("let error = new Error();", "error");
        assert!(score > 0.8, "Exact match should score high");
    }

    #[test]
    fn test_token_match_score_camel_case() {
        let score = token_match_score("fn handleError() {", "error");
        assert!(score > 0.5, "Should match camelCase component");
    }

    #[test]
    fn test_token_match_score_snake_case() {
        let score = token_match_score("fn handle_error() {", "error");
        assert!(score > 0.5, "Should match snake_case component");
    }

    #[test]
    fn test_token_match_score_no_match() {
        let score = token_match_score("fn main() { println!(\"hello\"); }", "database");
        assert!(score < 0.1, "No match should score near zero");
    }

    // Test find_representative_lines function
    #[test]
    fn test_find_representative_lines_camel_case_match() {
        let code = "fn main() {\n    handleError();\n    logMessage();\n}";
        let lines = find_representative_lines(code, 1, "error handling");
        // Line 2 contains "error" in handleError
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_find_representative_lines_multiple_tokens() {
        let code = "fn fetch() {\n    let q = build();\n    let user = runDatabaseQuery(q);\n    return user;\n}";
        let lines = find_representative_lines(code, 1, "database query");
        // Line 3 contains both "database" and "query"
        assert!(lines.contains(&3));
    }

    #[test]
    fn test_find_representative_lines_no_match() {
        let code = "fn main() {\n    println!(\"hello\");\n}";
        let lines = find_representative_lines(code, 1, "authentication security");
        // No relevant tokens
        assert!(lines.is_empty());
    }

    #[test]
    fn test_find_representative_lines_window_context() {
        // Test that consecutive matching lines get boosted
        let code = "fn processAuth() {\n    validateToken();\n    checkPermissions();\n    return authorized;\n}";
        let lines = find_representative_lines(code, 1, "auth token validation permissions");
        // Should find lines in the auth-related section
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_find_representative_lines_prefix_match() {
        let code = "fn authenticate() {\n    let token = getAuthToken();\n    return validated;\n}";
        let lines = find_representative_lines(code, 1, "auth token");
        // "auth" should match "authenticate" and "getAuthToken"
        assert!(!lines.is_empty());
    }
}
