//! Odin parse fallback: a few constructs tree-sitter-odin does not know
//! (`%%=`, some `#directives`) can leave a whole file one ERROR node. Such a
//! file is parsed again one top-level declaration at a time.

/// Ranges of the file's top-level declarations: a column-0 `name ::` line
/// (or `package` / `import` / `foreign` / `when`), with the comment and
/// `@(attribute)` lines directly above it.
pub fn sections(source: &str) -> Vec<tree_sitter::Range> {
    let lines: Vec<&str> = source.split('\n').collect();
    let mut offsets = Vec::with_capacity(lines.len() + 1);
    let mut offset = 0;
    for line in &lines {
        offsets.push(offset);
        offset += line.len() + 1;
    }
    offsets.push(source.len());
    let opens = |line: &str| {
        if !line.starts_with(|c: char| c.is_alphabetic() || c == '_') {
            return false;
        }
        let word: String = line
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        matches!(word.as_str(), "package" | "import" | "foreign" | "when")
            || line[word.len()..].trim_start().starts_with("::")
            || line[word.len()..].trim_start().starts_with(", ")
    };
    let mut starts: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !opens(line) {
            continue;
        }
        let floor = starts.last().map_or(0, |s| s + 1);
        let mut top = i;
        while top > floor {
            let above = lines[top - 1].trim_start();
            if above.starts_with("//") || above.starts_with('@') || above.starts_with("#+") {
                top -= 1;
            } else {
                break;
            }
        }
        starts.push(top);
    }
    starts
        .iter()
        .enumerate()
        .map(|(k, &start)| {
            let end = starts.get(k + 1).copied().unwrap_or(lines.len());
            tree_sitter::Range {
                start_byte: offsets[start],
                end_byte: offsets[end].min(source.len()),
                start_point: tree_sitter::Point::new(start, 0),
                end_point: tree_sitter::Point::new(end, 0),
            }
        })
        .collect()
}
