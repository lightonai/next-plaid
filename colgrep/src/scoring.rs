use std::path::Path;

use colgrep::CodeUnit;

/// Compute boosted score based on literal query matches in code unit
pub fn compute_final_score(
    semantic_score: f32,
    query: &str,
    unit: &CodeUnit,
    text_pattern: Option<&str>,
) -> f32 {
    let mut score = semantic_score;
    let query_lower = query.to_lowercase();

    // Boost if query appears literally in name (strongest boost)
    if unit.name.to_lowercase().contains(&query_lower) {
        score += 3.0;
    }
    // Boost if query appears literally in signature
    if unit.signature.to_lowercase().contains(&query_lower) {
        score += 2.0;
    }
    // Boost if query appears in code preview (moderate boost)
    if unit.code.to_lowercase().contains(&query_lower) {
        score += 1.0;
    }

    // Decrement score for test functions unless query or pattern contains "test"
    let query_has_test = query_lower.contains("test");
    let pattern_has_test = text_pattern
        .map(|p| p.to_lowercase().contains("test"))
        .unwrap_or(false);
    if unit.name.to_lowercase().contains("test") && !query_has_test && !pattern_has_test {
        score -= 1.0;
    }

    score
}

/// Check if --include patterns would escape the subdirectory
/// A pattern escapes if it starts with `**/` followed by a specific directory name
/// that doesn't exist within the current subdirectory.
///
/// When this returns true, the caller should search the full project index
/// (still bounded by effective_root) rather than restricting to the subdirectory.
/// This does NOT cause the search to escape to a higher-level or different index.
pub fn should_search_from_root(
    include_patterns: &[String],
    subdir: &Path,
    effective_root: &Path,
) -> bool {
    for pattern in include_patterns {
        // Check for patterns like "**/.github/**/*" or "**/vendor/**"
        if let Some(rest) = pattern.strip_prefix("**/") {
            // Extract the first path component after "**/
            if let Some(dir_name) = rest.split('/').next() {
                // Skip if it's a wildcard pattern like "*.rs"
                if dir_name.contains('*') {
                    continue;
                }
                // Check if this directory exists in the current subdir
                let subdir_path = effective_root.join(subdir).join(dir_name);
                if !subdir_path.exists() {
                    // Directory doesn't exist in subdir, pattern escapes to root
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use colgrep::{Language, UnitType};

    use super::*;

    /// Helper to create a test CodeUnit with minimal required fields
    fn make_test_unit(name: &str, signature: &str, code: &str, file: &str) -> CodeUnit {
        let mut unit = CodeUnit::new(
            name.to_string(),
            PathBuf::from(file),
            1,
            10,
            Language::Rust,
            UnitType::Function,
            None,
        );
        unit.signature = signature.to_string();
        unit.code = code.to_string();
        unit
    }

    // Test compute_final_score function
    #[test]
    fn test_compute_final_score_no_boost() {
        let unit = make_test_unit(
            "other_function",
            "fn other_function()",
            "does something else",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // No matches, score should be unchanged
        assert_eq!(score, 5.0);
    }

    #[test]
    fn test_compute_final_score_name_boost() {
        let unit = make_test_unit(
            "search_query_handler",
            "fn search_query_handler()",
            "handles queries",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Name contains query, should have +3.0 boost, signature too +2.0
        assert!(score > 5.0);
        assert_eq!(score, 5.0 + 3.0 + 2.0); // name + signature
    }

    #[test]
    fn test_compute_final_score_signature_boost() {
        let unit = make_test_unit(
            "handler",
            "fn handler(search_query: &str)",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Signature contains query, should have +2.0 boost
        assert_eq!(score, 5.0 + 2.0);
    }

    #[test]
    fn test_compute_final_score_code_boost() {
        let unit = make_test_unit(
            "handler",
            "fn handler()",
            "processes search_query and returns",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Code preview contains query, should have +1.0 boost
        assert_eq!(score, 5.0 + 1.0);
    }

    #[test]
    fn test_compute_final_score_case_insensitive() {
        let unit = make_test_unit(
            "SEARCH_QUERY_HANDLER",
            "fn SEARCH_QUERY_HANDLER()",
            "handles queries",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Case insensitive match should work
        assert!(score > 5.0);
    }

    #[test]
    fn test_compute_final_score_all_boosts() {
        let unit = make_test_unit(
            "search_query",
            "fn search_query(search_query: T)",
            "search_query implementation",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // All three locations contain query
        assert_eq!(score, 5.0 + 3.0 + 2.0 + 1.0);
    }

    #[test]
    fn test_compute_final_score_test_function_decremented() {
        let unit = make_test_unit(
            "test_something",
            "fn test_something()",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        // Test function should be decremented by 1.0
        assert_eq!(score, 5.0 - 1.0);
    }

    #[test]
    fn test_compute_final_score_test_function_not_decremented_when_query_has_test() {
        let unit = make_test_unit(
            "test_something",
            "fn test_something()",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "test", &unit, None);
        // Query contains "test", so no decrement; name contains query so +3.0, signature +2.0
        assert_eq!(score, 5.0 + 3.0 + 2.0);
    }

    #[test]
    fn test_compute_final_score_test_function_not_decremented_when_pattern_has_test() {
        let unit = make_test_unit(
            "test_something",
            "fn test_something()",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, Some("test"));
        // Pattern contains "test", so no decrement
        assert_eq!(score, 5.0);
    }

    // Test should_search_from_root function
    #[test]
    fn test_should_search_from_root_no_patterns() {
        let patterns: Vec<String> = vec![];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project");
        assert!(!should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_wildcard_extension() {
        // Pattern like "**/*.rs" should NOT escape (it's a file extension, not a directory)
        let patterns = vec!["**/*.rs".to_string()];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project");
        assert!(!should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_no_star_star_prefix() {
        // Pattern like "src/**/*.py" should NOT escape (no **/ prefix)
        let patterns = vec!["src/**/*.py".to_string()];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project");
        assert!(!should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_simple_glob() {
        // Pattern like "*.json" should NOT escape (no **/ prefix)
        let patterns = vec!["*.json".to_string()];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project");
        assert!(!should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_escaping_pattern() {
        // Pattern like "**/.github/**/*" should escape if .github doesn't exist in subdir
        // Since /tmp/test_project/src/.github almost certainly doesn't exist, this should return true
        let patterns = vec!["**/.github/**/*".to_string()];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project_nonexistent");
        assert!(should_search_from_root(&patterns, &subdir, &root));
    }

    #[test]
    fn test_should_search_from_root_multiple_patterns_one_escapes() {
        // If ANY pattern escapes, should return true
        let patterns = vec![
            "**/*.rs".to_string(),         // doesn't escape (wildcard)
            "**/.github/**/*".to_string(), // escapes
        ];
        let subdir = PathBuf::from("src");
        let root = PathBuf::from("/tmp/test_project_nonexistent");
        assert!(should_search_from_root(&patterns, &subdir, &root));
    }
}
