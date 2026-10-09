//! XML extraction (`.xml`, `.xsd`, `.xsl(t)`, `.xaml`, MSBuild projects,
//! `.plist`, ...), parsed with tree-sitter-xml.
//!
//! An XML file is one tree, so "one unit per top-level node" would make the
//! root element a single unit covering the whole file. Instead the tree is
//! cut by size: an element that fits in [`MAX_UNIT_LINES`] is one unit; a
//! larger one is split into runs of consecutive child elements that fit,
//! and a child that is itself too large is split the same way, recursively.
//! Units are named by their tag path with each element's identifying
//! attribute, e.g. `Project > Target Name="Build"` or
//! `manifest > application > activity android:name=".MainActivity"`, so a
//! query for the target, activity or dependency lands on the right lines.
//!
//! The ranges tile the file: a run of children also takes the comments and
//! text before it, the first run of an element starts at the element's
//! opening tag and the closing tag joins the last run, so no stray one-line
//! units are left for `</dependencies>`. Generated multi-megabyte files are
//! linear in the number of elements and come out as many small units, never
//! as one absurd one.
//!
//! XSLT `template`s and `function`s are [`UnitType::Function`]; everything
//! else is a [`UnitType::Section`].

use super::text::{chunk_ranges, fill_gaps_chunked};
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

/// An element up to this many lines is one unit; a larger one is split.
const MAX_UNIT_LINES: usize = 50;
/// ... and a run of sibling elements stops growing at this many characters.
const MAX_UNIT_CHARS: usize = 4000;
/// Attributes that identify an element among its siblings, in priority order.
const IDENTIFYING_ATTRIBUTES: &[&str] = &[
    "name",
    "Name",
    "x:Name",
    "x:Key",
    "android:name",
    "android:id",
    "id",
    "Id",
    "key",
    "Key",
    "Include",
    "Update",
    "Remove",
    "Project",
    "match",
    "ref",
    "type",
];
/// Child elements whose text identifies their parent (Maven coordinates,
/// Ant/Spring names, plist keys).
const IDENTIFYING_CHILDREN: &[&str] = &["artifactId", "name", "id", "key"];
/// Up to this many non-element lines (closing tags, a short comment) join
/// the neighbouring unit; a longer stretch becomes a unit of its own.
const SHORT_GAP: usize = 5;
/// Longest attribute value kept in a unit name.
const MAX_NAME_VALUE_CHARS: usize = 60;

pub fn extract_xml_units(path: &Path, source: &str) -> Vec<CodeUnit> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.iter().all(|l| l.trim().is_empty()) {
        return Vec::new();
    }

    let mut units = Vec::new();
    let trees = parse_top_level(source);
    let tops: Vec<Node> = trees.iter().filter_map(root_element).collect();

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document");
    if tops.is_empty() {
        // Not well-formed enough to have a root element: index by size.
        for (s, e) in chunk_ranges(&lines, 0, lines.len() - 1, MAX_UNIT_LINES, MAX_UNIT_CHARS) {
            units.push(make_unit(path, &lines, stem, s, e, UnitType::Section));
        }
    } else {
        let mut ctx = Context {
            path,
            lines: &lines,
            bytes: source.as_bytes(),
            units: &mut units,
            max_depth: super::max_recursion_depth(),
        };
        // The root's range takes the prolog before it and anything after.
        if tops.len() == 1 {
            ctx.split(tops[0], &[], 0, lines.len() - 1, 0);
        } else {
            ctx.split_children(&tops, &[], stem, 0, lines.len() - 1, 0);
        }
    }

    fill_gaps_chunked(
        &mut units,
        path,
        &lines,
        Language::Xml,
        MAX_UNIT_LINES,
        MAX_UNIT_CHARS,
    );
    units
}

/// XML fragments may have several top-level elements
/// (an XSLT module of bare templates, an entity-included chapter).
/// tree-sitter-xml keeps the first as the root and turns everything after it
/// into flat ERROR tokens, so each further top-level element is parsed again
/// from where the previous one ended, with the text before it blanked out
/// (newlines kept) so rows and byte offsets still match `source`.
const MAX_TOP_LEVEL_PARSES: usize = 200;
/// ... and at most this many bytes re-parsed in total; the rest of a large
/// fragment is covered by size-bounded raw chunks.
const MAX_REPARSED_BYTES: usize = 32 * 1024 * 1024;

fn parse_top_level(source: &str) -> Vec<Tree> {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_xml::LANGUAGE_XML.into())
        .is_err()
    {
        return Vec::new();
    }
    let mut trees: Vec<Tree> = Vec::new();
    let Some(first) = parser.parse(source, None) else {
        return trees;
    };
    let mut offset = match root_element(&first) {
        Some(root) => root.end_byte(),
        None => return trees,
    };
    trees.push(first);
    while trees.len() < MAX_TOP_LEVEL_PARSES
        && trees.len() * source.len() < MAX_REPARSED_BYTES
        && source[offset..].contains('<')
    {
        // One ASCII byte per source byte keeps every offset unchanged.
        let masked: String = source[..offset]
            .bytes()
            .map(|b| if b == b'\n' { '\n' } else { ' ' })
            .collect::<String>()
            + &source[offset..];
        let Some(tree) = parser.parse(&masked, None) else {
            break;
        };
        let Some(end) = root_element(&tree).map(|r| r.end_byte()) else {
            break;
        };
        if end <= offset {
            break;
        }
        offset = end;
        trees.push(tree);
    }
    trees
}

fn root_element(tree: &Tree) -> Option<Node<'_>> {
    let doc = tree.root_node();
    doc.child_by_field_name("root").or_else(|| {
        doc.children(&mut doc.walk())
            .find(|c| c.kind() == "element")
    })
}

fn make_unit(
    path: &Path,
    lines: &[&str],
    name: &str,
    start: usize,
    end: usize,
    unit_type: UnitType,
) -> CodeUnit {
    let mut unit = CodeUnit::new(
        name.to_string(),
        path.to_path_buf(),
        start + 1,
        end + 1,
        Language::Xml,
        unit_type,
        None,
    );
    unit.signature = lines[start..=end]
        .iter()
        .map(|l| l.trim())
        .find(|l| !l.is_empty() && !l.starts_with("<?") && !l.starts_with("<!--"))
        .or_else(|| {
            lines[start..=end]
                .iter()
                .map(|l| l.trim())
                .find(|l| !l.is_empty())
        })
        .unwrap_or_default()
        .to_string();
    unit.code = lines[start..=end].join("\n");
    unit
}

struct Context<'a, 'b> {
    path: &'a Path,
    lines: &'a [&'a str],
    bytes: &'a [u8],
    units: &'b mut Vec<CodeUnit>,
    max_depth: usize,
}

impl<'a> Context<'a, '_> {
    fn text(&self, node: Node) -> &'a str {
        node.utf8_text(self.bytes).unwrap_or("")
    }

    /// The start or empty-element tag of an element.
    fn tag(node: Node) -> Option<Node> {
        node.children(&mut node.walk())
            .find(|c| matches!(c.kind(), "STag" | "EmptyElemTag"))
    }

    fn tag_name(&self, element: Node) -> String {
        Self::tag(element)
            .and_then(|t| t.children(&mut t.walk()).find(|c| c.kind() == "Name"))
            .map(|n| self.text(n).to_string())
            .unwrap_or_else(|| "element".to_string())
    }

    fn child_elements<'n>(&self, element: Node<'n>) -> Vec<Node<'n>> {
        element
            .children(&mut element.walk())
            .filter(|c| c.kind() == "content")
            .flat_map(|content| {
                content
                    .children(&mut content.walk())
                    .filter(|c| c.kind() == "element")
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The value that identifies `element` among its siblings: an
    /// identifying attribute, or the text of an identifying child element.
    fn identity(&self, element: Node) -> Option<(String, String)> {
        if let Some(tag) = Self::tag(element) {
            let attrs: Vec<(String, String)> = tag
                .children(&mut tag.walk())
                .filter(|c| c.kind() == "Attribute")
                .filter_map(|a| {
                    let name = a.children(&mut a.walk()).find(|c| c.kind() == "Name")?;
                    let value = a.children(&mut a.walk()).find(|c| c.kind() == "AttValue")?;
                    let value = self.text(value).trim_matches(|c| c == '"' || c == '\'');
                    Some((self.text(name).to_string(), value.to_string()))
                })
                .collect();
            for wanted in IDENTIFYING_ATTRIBUTES {
                if let Some((n, v)) = attrs.iter().find(|(n, _)| n == wanted) {
                    if !v.trim().is_empty() {
                        return Some((n.clone(), shorten(v)));
                    }
                }
            }
        }
        for child in self.child_elements(element) {
            let tag = self.tag_name(child);
            if !IDENTIFYING_CHILDREN.contains(&tag.as_str()) {
                continue;
            }
            if !self.child_elements(child).is_empty() {
                continue;
            }
            let text: String = child
                .children(&mut child.walk())
                .filter(|c| c.kind() == "content")
                .map(|c| self.text(c))
                .collect();
            let text = text.trim();
            if !text.is_empty() && !text.contains('<') && text.len() <= MAX_NAME_VALUE_CHARS {
                return Some((tag, text.to_string()));
            }
        }
        None
    }

    /// `Target Name="Build"`, or the bare tag when nothing identifies it.
    fn describe(&self, element: Node) -> String {
        let tag = self.tag_name(element);
        match self.identity(element) {
            Some((attr, value)) => format!("{tag} {attr}=\"{value}\""),
            None => tag,
        }
    }

    fn unit_type_for(tag: &str) -> UnitType {
        let local = tag.rsplit(':').next().unwrap_or(tag);
        if tag.contains(':') && matches!(local, "template" | "function") {
            UnitType::Function
        } else {
            UnitType::Section
        }
    }

    fn push(&mut self, name: String, start: usize, end: usize, unit_type: UnitType) {
        if self.lines[start..=end].iter().all(|l| l.trim().is_empty()) {
            return;
        }
        self.units.push(make_unit(
            self.path, self.lines, &name, start, end, unit_type,
        ));
    }

    fn chars(&self, start: usize, end: usize) -> usize {
        self.lines[start..=end].iter().map(|l| l.len() + 1).sum()
    }

    /// Emit units for `element`, whose unit range is `start..=end` (the
    /// element itself plus any surrounding lines it is responsible for).
    fn split(&mut self, element: Node, path: &[String], start: usize, end: usize, depth: usize) {
        let desc = self.describe(element);
        let mut full: Vec<String> = path.to_vec();
        full.push(desc);
        let name = join_path(&full);
        let unit_type = Self::unit_type_for(&self.tag_name(element));
        let children = self.child_elements(element);

        // XSLT templates / functions are always units of their own, like
        // functions in code, even in a stylesheet small enough to fit.
        let has_callables = children
            .iter()
            .any(|c| Self::unit_type_for(&self.tag_name(*c)) == UnitType::Function);
        let fits = end + 1 - start <= MAX_UNIT_LINES && !has_callables;
        if fits || depth >= self.max_depth {
            self.push(name, start, end, unit_type);
            return;
        }
        if children.is_empty() {
            // A large leaf (embedded script, long text): cut by paragraphs.
            for (s, e) in chunk_ranges(self.lines, start, end, MAX_UNIT_LINES, MAX_UNIT_CHARS) {
                self.push(name.clone(), s, e, unit_type);
            }
            return;
        }

        self.split_children(&children, &full, &name, start, end, depth);
    }

    /// Tile `start..=end` with units for `children` (siblings under `path`):
    /// runs of small siblings are grouped, large ones are split recursively.
    fn split_children(
        &mut self,
        children: &[Node],
        full: &[String],
        name: &str,
        start: usize,
        end: usize,
        depth: usize,
    ) {
        let first_unit = self.units.len();
        let mut group_start = start;
        let mut group: Vec<Node> = Vec::new();
        let mut group_end: Option<usize> = None;
        for &child in children {
            let (cs, ce) = (
                child.start_position().row,
                child.end_position().row.min(end),
            );
            if cs < group_start && group_end.is_none() {
                // Shares a line with the previous (split) sibling.
                continue;
            }
            if let Some(ge) = group_end {
                if cs <= ge {
                    // Shares a line with the current run: lines can't split.
                    group.push(child);
                    group_end = Some(ge.max(ce));
                    continue;
                }
            }
            // A long stretch of comments / text before this child (a license
            // header, commented-out code) is its own unit, not a prefix.
            let gap_start = group_end.map_or(group_start, |ge| ge + 1);
            if cs > gap_start + SHORT_GAP {
                if let Some(ge) = group_end.take() {
                    self.flush(full, &group, group_start, ge);
                    group.clear();
                }
                self.chunk_text(name, gap_start, cs - 1);
                group_start = cs;
            }
            let big = ce + 1 - cs > MAX_UNIT_LINES
                || Self::unit_type_for(&self.tag_name(child)) == UnitType::Function;
            if big {
                if let Some(ge) = group_end.take() {
                    self.flush(full, &group, group_start, ge);
                    group.clear();
                    group_start = ge + 1;
                }
                self.split(child, full, group_start, ce, depth + 1);
                group_start = ce + 1;
                continue;
            }
            if let Some(ge) = group_end {
                if ce + 1 - group_start > MAX_UNIT_LINES
                    || self.chars(group_start, ce) > MAX_UNIT_CHARS
                {
                    self.flush(full, &group, group_start, ge);
                    group.clear();
                    group_start = ge + 1;
                }
            }
            group.push(child);
            group_end = Some(ce);
        }
        if let Some(ge) = group_end {
            self.flush(full, &group, group_start, ge);
            group_start = ge + 1;
        }
        self.tail(first_unit, name, group_start, end);
    }

    /// Lines after the last child (closing tags, trailing comments): a few
    /// join the last unit; a long tail is chunked on its own.
    fn tail(&mut self, first_unit: usize, name: &str, from: usize, end: usize) {
        if from > end || self.lines[from..=end].iter().all(|l| l.trim().is_empty()) {
            return;
        }
        if end + 1 - from <= SHORT_GAP {
            if let Some(last) = self.units[first_unit..].last_mut() {
                if last.end_line == from {
                    last.end_line = end + 1;
                    last.code = self.lines[last.line - 1..=end].join("\n");
                    return;
                }
            }
        }
        self.chunk_text(name, from, end);
    }

    /// Size-bounded units over lines that hold no element of their own.
    fn chunk_text(&mut self, name: &str, from: usize, end: usize) {
        for (s, e) in chunk_ranges(self.lines, from, end, MAX_UNIT_LINES, MAX_UNIT_CHARS) {
            self.push(name.to_string(), s, e, UnitType::Section);
        }
    }

    /// Emit one unit for a run of sibling elements under `path`.
    fn flush(&mut self, path: &[String], group: &[Node], start: usize, end: usize) {
        if group.is_empty() {
            return;
        }
        let (label, unit_type) = if group.len() == 1 {
            (
                self.describe(group[0]),
                Self::unit_type_for(&self.tag_name(group[0])),
            )
        } else {
            let mut tags: Vec<String> = Vec::new();
            for child in group {
                let tag = self.tag_name(*child);
                if !tags.contains(&tag) {
                    tags.push(tag);
                }
            }
            let mut label = if tags.len() > 4 {
                format!("{}, …", tags[..4].join(", "))
            } else {
                tags.join(", ")
            };
            let ids: Vec<String> = group
                .iter()
                .filter_map(|child| self.identity(*child).map(|(_, v)| v))
                .collect();
            match ids.len() {
                0 => {}
                1..=3 => label.push_str(&format!(" ({})", ids.join(", "))),
                n => label.push_str(&format!(" ({} … {})", ids[0], ids[n - 1])),
            }
            (label, UnitType::Section)
        };
        let mut full = path.to_vec();
        full.push(label);
        self.push(join_path(&full), start, end, unit_type);
    }
}

fn join_path(parts: &[String]) -> String {
    parts.join(" > ")
}

fn shorten(value: &str) -> String {
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if value.chars().count() <= MAX_NAME_VALUE_CHARS {
        value
    } else {
        let mut s: String = value.chars().take(MAX_NAME_VALUE_CHARS).collect();
        s.push('…');
        s
    }
}
