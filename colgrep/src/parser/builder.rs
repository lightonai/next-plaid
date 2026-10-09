//! Shared helpers for the languages that have a dedicated extractor
//! (Erlang, F#, Clojure, Elm).
//!
//! Their grammars do not fit the node-kind driven walk in `mod.rs`: an Erlang
//! function is a run of sibling clauses, an Elm function is a type annotation
//! plus a sibling value declaration, a Clojure definition is a list whose head
//! symbol happens to be `defn`, and an F# `let` is a function, a constant or a
//! local binding depending on where it sits. Each of those modules decides
//! what a unit is; this module builds the `CodeUnit` from that decision.

use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::Node;

/// Source text of a node ("" if it is not valid UTF-8).
pub(super) fn text<'a>(node: Node, bytes: &'a [u8]) -> &'a str {
    node.utf8_text(bytes).unwrap_or("")
}

/// Collapse runs of whitespace (newlines included) into single spaces.
pub(super) fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Append `value` unless it is empty or already present.
pub(super) fn push_unique(target: &mut Vec<String>, value: impl Into<String>) {
    let value = value.into();
    if !value.is_empty() && !target.contains(&value) {
        target.push(value);
    }
}

/// Visit every node under `root` (root included) with an explicit stack, so
/// deeply nested code cannot overflow the call stack. `f` returns whether to
/// descend into the node's children.
pub(super) fn walk<'a>(root: Node<'a>, mut f: impl FnMut(Node<'a>) -> bool) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if f(node) {
            let children: Vec<_> = node.children(&mut node.walk()).collect();
            stack.extend(children.into_iter().rev());
        }
    }
}

/// Complexity / loop / branch / error-handling flags from node kinds.
pub(super) fn control_flow(
    root: Node,
    branches: &[&str],
    loops: &[&str],
    errors: &[&str],
) -> (usize, bool, bool, bool) {
    let (mut complexity, mut has_loops, mut has_branches, mut has_errors) =
        (1, false, false, false);
    walk(root, |n| {
        let kind = n.kind();
        if branches.contains(&kind) {
            complexity += 1;
            has_branches = true;
        } else if loops.contains(&kind) {
            complexity += 1;
            has_loops = true;
        } else if errors.contains(&kind) {
            has_errors = true;
        }
        true
    });
    (complexity, has_loops, has_branches, has_errors)
}

/// Build a unit covering `start_row..=end_row` (0-indexed rows). The code is
/// the full line range; the signature defaults to the trimmed `sig_row` line.
#[allow(clippy::too_many_arguments)]
pub(super) fn new_unit(
    path: &Path,
    lines: &[&str],
    lang: Language,
    name: String,
    unit_type: UnitType,
    start_row: usize,
    end_row: usize,
    sig_row: usize,
    parent: Option<&str>,
) -> CodeUnit {
    // tree-sitter can report an end row one past EOF for a construct left
    // unterminated at end-of-file; clamp so ranges stay inside the file.
    let last = lines.len().saturating_sub(1);
    let end_row = end_row.min(last);
    let start_row = start_row.min(end_row);
    let mut unit = CodeUnit::new(
        name,
        path.to_path_buf(),
        start_row + 1,
        end_row + 1,
        lang,
        unit_type,
        parent,
    );
    unit.signature = lines
        .get(sig_row)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    unit.code = lines[start_row..=end_row].join("\n");
    unit
}

/// Last row of `node` that holds something other than trailing comments.
/// Some grammars (F#) attach the comments that precede the next declaration
/// to the end of the previous one; a unit must not swallow its neighbour's
/// documentation.
pub(super) fn end_row_without_trailing(node: Node, trailing: &[&str]) -> usize {
    let mut end = node.end_position().row;
    let mut current = node;
    // Descend along last children while they are trailing trivia or wrappers
    // whose last child is trivia.
    loop {
        let children: Vec<_> = current.children(&mut current.walk()).collect();
        let Some(pos) = children.iter().rposition(|c| !trailing.contains(&c.kind())) else {
            break;
        };
        if pos + 1 == children.len() {
            // The last child is real content: look inside it.
            current = children[pos];
            continue;
        }
        let row = children[pos].end_position().row;
        end = end.min(row.max(node.start_position().row));
        break;
    }
    // A node that ends at column 0 ends on the previous line.
    if end == node.end_position().row && node.end_position().column == 0 && end > 0 {
        end -= 1;
    }
    end.max(node.start_position().row)
}

/// The first paragraph of a doc text (up to the first blank line).
pub(super) fn first_paragraph(doc: &str) -> &str {
    let mut end = doc.len();
    let mut offset = 0;
    for line in doc.split_inclusive('\n') {
        if line.trim().is_empty() && !doc[..offset].trim().is_empty() {
            end = offset;
            break;
        }
        offset += line.len();
    }
    doc[..end].trim()
}
