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
}
