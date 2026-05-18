use std::path::Path;

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

    use super::*;

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
