use std::path::Path;

use colgrep::CodeUnit;

/// Compute final score with test function demotion
pub fn compute_final_score(
    semantic_score: f32,
    query: &str,
    unit: &CodeUnit,
    text_pattern: Option<&str>,
) -> f32 {
    let mut score = semantic_score;
    let query_lower = query.to_lowercase();

    // Demote test functions unless query or pattern contains "test"
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
    fn test_compute_final_score_no_adjustment() {
        let unit = make_test_unit(
            "other_function",
            "fn other_function()",
            "does something else",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        assert_eq!(score, 5.0);
    }

    #[test]
    fn test_compute_final_score_test_function_demoted() {
        let unit = make_test_unit(
            "test_something",
            "fn test_something()",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, None);
        assert_eq!(score, 5.0 - 1.0);
    }

    #[test]
    fn test_compute_final_score_test_function_not_demoted_when_query_has_test() {
        let unit = make_test_unit(
            "test_something",
            "fn test_something()",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "test", &unit, None);
        assert_eq!(score, 5.0);
    }

    #[test]
    fn test_compute_final_score_test_function_not_demoted_when_pattern_has_test() {
        let unit = make_test_unit(
            "test_something",
            "fn test_something()",
            "does something",
            "test.rs",
        );
        let score = compute_final_score(5.0, "search_query", &unit, Some("test"));
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
