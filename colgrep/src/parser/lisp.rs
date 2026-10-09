//! S-expression languages: Scheme, Racket and Common Lisp.
//!
//! Their tree-sitter grammars only know lists, symbols and literals (the
//! Common Lisp one also recognises `defun`-style headers), so the generic
//! node-kind tables in `ast.rs` do not apply. A definition is instead
//! recognised by the head symbol of a list: `(define (f x) ...)`,
//! `(defun f (x) ...)`, `(struct point (x y))`, `(defclass shape () ...)`.
//!
//! Units produced:
//! - functions and macros (`define`, `define-syntax`, `defun`, `defmacro`,
//!   `defgeneric`, `defmethod`, ...), with parameters from the lambda list,
//!   docstrings (or the comment block right above), calls and local bindings;
//! - records, structs and classes (`define-record-type`, `struct`,
//!   `defclass`, `defstruct`, `define-condition`, Racket `class` expressions,
//!   whose `define/public` members become methods);
//! - modules and libraries (`define-library`, `library`, `module`,
//!   `module+`, `defpackage`, `defsystem`) as a unit for their header, with
//!   the definitions inside attached to them;
//! - top-level variables (`define x 1`, `defvar`, `defparameter`,
//!   `defconstant`) as constants;
//! - everything else as raw code.

use super::extract::{fill_raw_code_gaps, split_long_raw_code};
use super::language::get_tree_sitter_language;
use super::types::{CodeUnit, Language, UnitType};
use std::path::Path;
use tree_sitter::{Node, Parser};

/// A variable whose value spans more lines than this is read as a container
/// of the definitions inside it.
const LONG_VALUE_LINES: usize = 100;

/// Raw code blocks (runs of top-level expressions, e.g. test files) are cut
/// into pieces of about this many lines.
const RAW_CHUNK_LINES: usize = 60;

/// Main entry point for Scheme, Racket and Common Lisp files.
pub fn extract_lisp_units(path: &Path, source: &str, lang: Language) -> Vec<CodeUnit> {
    let mut parser = Parser::new();
    if parser
        .set_language(&get_tree_sitter_language(lang))
        .is_err()
    {
        return Vec::new();
    }
    // The Common Lisp grammar parses `(loop ...)` with a dedicated rule that
    // knows only part of the LOOP syntax (`for i of-type fixnum ...` and
    // other clauses throw it off), and its error recovery can then swallow
    // the whole file. Upper-casing the head turns every loop into a plain
    // list; the byte length is unchanged, so positions stay valid, and unit
    // code is cut from the original lines.
    let parsed_source = if lang == Language::CommonLisp {
        neutralize_loop_rule(source)
    } else {
        source.to_string()
    };
    let tree = match parser.parse(&parsed_source, None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let lines: Vec<&str> = source.lines().collect();
    let ctx = Ctx {
        path,
        lines: &lines,
        bytes: parsed_source.as_bytes(),
        lang,
        max_depth: super::max_recursion_depth(),
    };
    let root = tree.root_node();
    let file_imports = collect_file_imports(&ctx, root);

    let mut units = Vec::new();
    for child in root.named_children(&mut root.walk()) {
        visit_form(&ctx, child, None, &file_imports, &mut units, 0);
    }

    fill_raw_code_gaps(&mut units, path, &lines, lang, &file_imports);
    split_long_raw_code(&mut units, &lines, RAW_CHUNK_LINES);
    units
}

/// Replace the head of every `(loop` / `(cl:loop` form with `LOOP` (same
/// length), which the grammar's case-sensitive LOOP rule does not match.
pub(super) fn neutralize_loop_rule(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = bytes.to_vec();
    let mut i = 0;
    while let Some(off) = source[i..].find("loop") {
        let at = i + off;
        let end = at + 4;
        let before_ok =
            at > 0 && (bytes[at - 1] == b'(' || (at >= 4 && &bytes[at - 4..at] == b"(cl:"));
        let after_ok = end >= bytes.len() || bytes[end].is_ascii_whitespace() || bytes[end] == b')';
        if before_ok && after_ok {
            out[at..end].copy_from_slice(b"LOOP");
        }
        i = end;
    }
    // Only ASCII bytes were replaced by ASCII bytes.
    String::from_utf8(out).unwrap_or_else(|_| source.to_string())
}

struct Ctx<'a> {
    path: &'a Path,
    lines: &'a [&'a str],
    bytes: &'a [u8],
    lang: Language,
    max_depth: usize,
}

impl Ctx<'_> {
    fn text(&self, node: Node) -> &str {
        node.utf8_text(self.bytes).unwrap_or("")
    }

    fn is_cl(&self) -> bool {
        self.lang == Language::CommonLisp
    }
}

// ---------------------------------------------------------------------------
// Grammar normalisation
// ---------------------------------------------------------------------------

fn is_comment(kind: &str) -> bool {
    matches!(
        kind,
        "comment" | "block_comment" | "sexp_comment" | "dis_expr"
    )
}

fn is_list(kind: &str) -> bool {
    matches!(kind, "list" | "list_lit" | "defun")
}

fn is_symbol(kind: &str) -> bool {
    matches!(kind, "symbol" | "sym_lit" | "package_lit" | "defun_keyword")
}

fn is_string(kind: &str) -> bool {
    matches!(kind, "string" | "str_lit" | "here_string")
}

/// Quoted data: never code, so no calls or definitions inside.
fn is_quote(kind: &str) -> bool {
    matches!(kind, "quote" | "quoting_lit")
}

/// The elements of a list form, with comments dropped. Common Lisp's grammar
/// wraps `defun`/`defmacro`/`defgeneric`/`defmethod`/`lambda` forms in a
/// `defun` node whose header holds the keyword, name and lambda list; those
/// are flattened back into ordinary list elements.
fn elements(node: Node) -> Vec<Node> {
    match node.kind() {
        "list_lit" => {
            let mut cursor = node.walk();
            // `#:sym` is an anonymous `#` token plus a keyword, both under
            // the `value` field: keep named nodes only.
            let values: Vec<Node> = node
                .children_by_field_name("value", &mut cursor)
                .filter(|n| n.is_named())
                .collect();
            if values.is_empty() {
                if let Some(defun) = node
                    .named_children(&mut node.walk())
                    .find(|c| c.kind() == "defun")
                {
                    return elements(defun);
                }
                // `(loop ...)` and other special shapes: their parts
                return node
                    .named_children(&mut node.walk())
                    .filter(|c| !is_comment(c.kind()))
                    .collect();
            }
            values
        }
        "defun" => {
            let mut out = Vec::new();
            for child in node.named_children(&mut node.walk()) {
                if child.kind() == "defun_header" {
                    for field in ["keyword", "function_name", "specifier", "lambda_list"] {
                        if let Some(n) = child.child_by_field_name(field) {
                            out.push(n);
                        }
                    }
                    // `(defun ,name ...)` in a macro template has no fields
                    // beyond the keyword; keep the unquoted name too.
                    if child.child_by_field_name("function_name").is_none() {
                        for c in child.named_children(&mut child.walk()) {
                            if matches!(c.kind(), "unquoting_lit" | "unquote_splicing_lit") {
                                out.push(c);
                            }
                        }
                    }
                } else if !is_comment(child.kind()) {
                    out.push(child);
                }
            }
            out
        }
        _ => node
            .named_children(&mut node.walk())
            .filter(|c| !is_comment(c.kind()))
            .collect(),
    }
}

/// Normalised head symbol: lower-cased and without package prefix for Common
/// Lisp (`CL:DEFUN` -> `defun`), as written otherwise.
fn head_symbol(ctx: &Ctx, node: Node) -> Option<String> {
    if !is_symbol(node.kind()) {
        return None;
    }
    let text = ctx.text(node);
    Some(normalize_symbol(ctx, text))
}

fn normalize_symbol(ctx: &Ctx, text: &str) -> String {
    if ctx.is_cl() {
        let lower = text.to_lowercase();
        match lower.rfind(':') {
            Some(i) if i > 0 => lower[i + 1..].to_string(),
            _ => lower,
        }
    } else {
        text.to_string()
    }
}

/// Package prefix of a qualified Common Lisp symbol (`alexandria:when-let`).
fn symbol_package(ctx: &Ctx, node: Node) -> Option<String> {
    if !ctx.is_cl() || node.kind() != "package_lit" {
        return None;
    }
    let pkg = node.child_by_field_name("package")?;
    let name = ctx.text(pkg).trim_start_matches('#').to_lowercase();
    (!name.is_empty() && name != "cl" && name != "common-lisp").then_some(name)
}

/// Strip quotes from a string literal.
fn string_value(ctx: &Ctx, node: Node) -> String {
    let text = ctx.text(node);
    let text = text.strip_prefix('#').unwrap_or(text);
    text.trim_matches('"').trim().to_string()
}

/// Plain name of a symbol, keyword or string used as a package / library /
/// system designator: `:alexandria`, `#:alexandria`, `"alexandria"`.
fn designator(ctx: &Ctx, node: Node) -> Option<String> {
    let kind = node.kind();
    let text = if is_string(kind) {
        string_value(ctx, node)
    } else if !is_list(kind) && node.named_child_count() <= 2 {
        // Symbols, keywords and uninterned symbols (`#:name`).
        let t = ctx.text(node);
        let t = t.trim_start_matches("#:").trim_start_matches(':');
        if ctx.is_cl() {
            t.to_lowercase()
        } else {
            t.to_string()
        }
    } else {
        return None;
    };
    (!text.is_empty()).then_some(text)
}

// ---------------------------------------------------------------------------
// Definition forms
// ---------------------------------------------------------------------------

enum Form<'a> {
    /// A function, macro or method. `sig` is the signature list or name,
    /// `params` the lambda list, `body` the forms after it.
    Function {
        name: String,
        params: Vec<String>,
        return_type: Option<String>,
        body: Vec<Node<'a>>,
        parent: Option<String>,
        is_method: bool,
    },
    /// A record, struct or class declaration (no member functions).
    Record {
        name: String,
        extends: Option<String>,
        fields: Vec<String>,
        doc: Option<String>,
    },
    /// A Racket class expression bound to a name: its body holds methods.
    ClassExpr {
        name: String,
        extends: Option<String>,
        body: Vec<Node<'a>>,
    },
    /// A module, library or package: the unit covers the header, the body's
    /// definitions are attached to it.
    Container {
        name: String,
        body: Vec<Node<'a>>,
        doc: Option<String>,
    },
    Constant {
        name: String,
        doc: Option<String>,
    },
}

const LAMBDA_HEADS: &[&str] = &[
    "lambda",
    "λ",
    "named-lambda",
    "case-lambda",
    "lambda*",
    "opt-lambda",
    "match-lambda",
    "match-lambda*",
    "match-λ",
    "plambda",
];

/// Scheme/Racket heads that define a function from `(head (name . args) body)`
/// or `(head name (lambda ...))`.
fn is_scheme_function_head(head: &str) -> bool {
    matches!(
        head,
        "define"
            | "define*"
            | "define-public"
            | "define-inline"
            | "define-integrable"
            | "define-for-syntax"
            | "define/contract"
            | "define/match"
            | "define-macro"
            | "define-syntax"
            | "define-syntax*"
            | "define-syntaxes"
            | "define-syntax-rule"
            | "define-syntax-parser"
            | "define-syntax-parse-rule"
            | "define-simple-macro"
            | "define-generic"
            | "define-method"
            | "define-unit"
            | "defmacro"
            | "define*-public"
            | "define-syntax-public"
            | "define-syntax-class"
            | "define-splicing-syntax-class"
            | "define-memoized"
            | "define-values/invoke-unit"
            | "define-typed-syntax"
    )
}

/// Racket class member definitions.
fn is_racket_method_head(head: &str) -> bool {
    matches!(
        head,
        "define/public"
            | "define/private"
            | "define/override"
            | "define/augment"
            | "define/pubment"
            | "define/overment"
            | "define/augride"
            | "define/public-final"
            | "define/override-final"
            | "define/augment-final"
    )
}

/// Common Lisp heads parsed like `defun`: `(head name lambda-list body)`.
fn is_cl_function_head(head: &str) -> bool {
    matches!(
        head,
        "defun"
            | "defmacro"
            | "defgeneric"
            | "defmethod"
            | "define-compiler-macro"
            | "define-modify-macro"
            | "define-setf-expander"
            | "defsetf"
            | "deftype"
            | "define-method-combination"
            | "define-symbol-macro"
    )
}

fn is_cl_constant_head(head: &str) -> bool {
    matches!(
        head,
        "defvar"
            | "defparameter"
            | "defconstant"
            | "define-constant"
            | "defglobal"
            | "define-load-time-global"
            | "define-global-var"
            | "define-global-parameter"
    )
}

/// Classify a list form by its head symbol.
/// `class_owner` is the Racket class whose body `node` sits in: plain
/// `define`s there are methods.
fn classify<'a>(ctx: &Ctx, node: Node<'a>, class_owner: Option<&str>) -> Option<Form<'a>> {
    if !is_list(node.kind()) {
        return None;
    }
    let els = elements(node);
    let head = head_symbol(ctx, *els.first()?)?;
    if ctx.is_cl() {
        classify_cl(ctx, &head, &els)
    } else {
        classify_scheme(ctx, &head, &els, class_owner)
    }
}

fn classify_scheme<'a>(
    ctx: &Ctx,
    head: &str,
    els: &[Node<'a>],
    class_owner: Option<&str>,
) -> Option<Form<'a>> {
    let second = *els.get(1)?;
    match head {
        "define-library" | "library" => Some(Form::Container {
            name: library_name(ctx, second)?,
            body: els[2..].to_vec(),
            doc: None,
        }),
        "module" | "module*" | "module+" => {
            // `(module name lang body ...)` / `(module+ name body ...)`
            let name = designator(ctx, second)?;
            let skip = if head == "module+" { 2 } else { 3 };
            Some(Form::Container {
                name,
                body: els.get(skip..).map(<[Node]>::to_vec).unwrap_or_default(),
                doc: None,
            })
        }
        "define-record-type" | "define-record-type*" => {
            // R7RS / SRFI-9: (define-record-type <point> (make-point x y) point?
            //                  (x point-x) (y point-y set-point-y!))
            // R6RS: (define-record-type (point make-point point?)
            //         (parent shape) (fields x (mutable y)))
            let name = def_name(ctx, second)?;
            let mut fields = Vec::new();
            let mut extends = None;
            let clauses: Vec<(Option<String>, Vec<Node>)> = els[2..]
                .iter()
                .filter(|el| is_list(el.kind()))
                .map(|el| {
                    let sub = elements(*el);
                    (sub.first().and_then(|h| head_symbol(ctx, *h)), sub)
                })
                .collect();
            let r6rs = clauses
                .iter()
                .any(|(h, _)| matches!(h.as_deref(), Some("fields" | "parent")));
            if r6rs {
                for (head, sub) in &clauses {
                    match head.as_deref() {
                        Some("fields") => {
                            for f in sub.iter().skip(1) {
                                let field = if is_list(f.kind()) {
                                    elements(*f).get(1).and_then(|n| head_symbol_raw(ctx, *n))
                                } else {
                                    head_symbol_raw(ctx, *f)
                                };
                                if let Some(n) = field {
                                    push_unique(&mut fields, n);
                                }
                            }
                        }
                        Some("parent") => {
                            extends = sub.get(1).and_then(|n| head_symbol_raw(ctx, *n));
                        }
                        _ => {}
                    }
                }
            } else {
                for el in els.iter().skip(4) {
                    if let Some(n) = first_symbol(ctx, *el) {
                        push_unique(&mut fields, n);
                    }
                }
            }
            Some(Form::Record {
                name,
                extends,
                fields,
                doc: None,
            })
        }
        "struct" | "define-struct" | "struct/contract" | "define-struct/contract" => {
            // (struct name [super] (field ...) option ...)
            // (define-struct (name super) (field ...))
            let (name, mut extends) = if is_list(second.kind()) {
                let sub = elements(second);
                (
                    sub.first().and_then(|n| def_name(ctx, *n))?,
                    sub.get(1).and_then(|n| head_symbol(ctx, *n)),
                )
            } else {
                (head_symbol(ctx, second)?, None)
            };
            let mut field_list = els.get(2).copied();
            if let (Some(super_node), Some(fields)) = (els.get(2), els.get(3)) {
                if is_symbol(super_node.kind()) && is_list(fields.kind()) {
                    extends = head_symbol(ctx, *super_node);
                    field_list = Some(*fields);
                }
            }
            let fields = field_list
                .filter(|f| is_list(f.kind()))
                .map(|f| {
                    elements(f)
                        .into_iter()
                        .filter_map(|n| first_symbol(ctx, n))
                        .collect()
                })
                .unwrap_or_default();
            Some(Form::Record {
                name,
                extends,
                fields,
                doc: None,
            })
        }
        "define-class" => {
            // GOOPS: (define-class <name> (<super> ...) slot ...)
            let name = head_symbol(ctx, second)?;
            let extends = els
                .get(2)
                .filter(|s| is_list(s.kind()))
                .map(|s| symbols_joined(ctx, *s))
                .filter(|s| !s.is_empty());
            let fields = els
                .iter()
                .skip(3)
                .filter_map(|n| first_symbol(ctx, *n))
                .filter(|n| !n.starts_with('#'))
                .collect();
            Some(Form::Record {
                name,
                extends,
                fields,
                doc: None,
            })
        }
        "define-values" | "define-syntaxes" if is_list(second.kind()) => {
            let name = elements(second)
                .into_iter()
                .find_map(|n| head_symbol(ctx, n))?;
            Some(Form::Constant { name, doc: None })
        }
        _ => {
            let method = is_racket_method_head(head);
            let known = method || is_scheme_function_head(head);
            if !known
                && (!is_generic_scheme_definer(head) || form_rows(els) > GENERIC_DEFINER_MAX_LINES)
            {
                return None;
            }
            scheme_define(ctx, head, els, class_owner, method)
        }
    }
}

/// `(define (name . params) body)`, `(define name (lambda params body))`,
/// `(define name value)`, and the Racket class-expression binding.
fn scheme_define<'a>(
    ctx: &Ctx,
    head: &str,
    els: &[Node<'a>],
    class_owner: Option<&str>,
    method: bool,
) -> Option<Form<'a>> {
    let second = els[1];
    let is_method = method || (class_owner.is_some() && head == "define");
    let parent = class_owner;
    if is_list(second.kind()) {
        // (define (f a b) ...), curried (define ((f a) b) ...)
        let (name, params) = signature_name_and_params(ctx, second)?;
        let mut body_start = 2;
        let mut return_type = None;
        // Typed Racket: (define (f [x : Integer]) : Integer body)
        if els
            .get(2)
            .and_then(|n| head_symbol(ctx, *n))
            .is_some_and(|s| s == ":")
        {
            return_type = els.get(3).map(|n| ctx.text(*n).to_string());
            body_start = 4;
        }
        return Some(Form::Function {
            name,
            params,
            return_type,
            body: els
                .get(body_start..)
                .map(<[Node]>::to_vec)
                .unwrap_or_default(),
            parent: parent.map(str::to_string),
            is_method,
        });
    }
    let name = head_symbol(ctx, second)?;
    // Typed Racket: (define x : Integer 1)
    let value_idx = if els
        .get(2)
        .and_then(|n| head_symbol(ctx, *n))
        .is_some_and(|s| s == ":")
    {
        4
    } else {
        2
    };
    let value = els.get(value_idx).copied();
    if let Some(value) = value.filter(|v| is_list(v.kind())) {
        let vels = elements(value);
        let vhead = vels.first().and_then(|h| head_symbol(ctx, *h));
        if let Some(vhead) = vhead.as_deref() {
            if LAMBDA_HEADS.contains(&vhead) {
                return Some(Form::Function {
                    name,
                    params: lambda_params(ctx, vhead, &vels),
                    return_type: None,
                    body: vels.get(2..).map(<[Node]>::to_vec).unwrap_or_default(),
                    parent: parent.map(str::to_string),
                    is_method,
                });
            }
            if matches!(vhead, "class" | "class*" | "mixin") {
                let extends = vels
                    .get(1)
                    .map(|n| ctx.text(*n).to_string())
                    .filter(|_| vhead != "mixin");
                let skip = if vhead == "class" { 2 } else { 3 };
                return Some(Form::ClassExpr {
                    name,
                    extends,
                    body: vels.get(skip..).map(<[Node]>::to_vec).unwrap_or_default(),
                });
            }
        }
    }
    // Macros and syntax classes bound to a name are still definitions of
    // behaviour, not data.
    if head != "define" && head != "define-public" && is_scheme_function_head(head) {
        return Some(Form::Function {
            name,
            params: Vec::new(),
            return_type: None,
            body: els.get(2..).map(<[Node]>::to_vec).unwrap_or_default(),
            parent: parent.map(str::to_string),
            is_method,
        });
    }
    if is_method {
        // `(define/public x 1)`-style field
        return None;
    }
    Some(Form::Constant { name, doc: None })
}

fn classify_cl<'a>(ctx: &Ctx, head: &str, els: &[Node<'a>]) -> Option<Form<'a>> {
    let second = *els.get(1)?;
    if is_cl_function_head(head) {
        let name = def_name(ctx, second)?;
        // defmethod qualifiers: (defmethod area :around ((s circle)) ...)
        let mut idx = 2;
        if head == "defmethod" {
            while idx < 4
                && els
                    .get(idx)
                    .is_some_and(|n| matches!(n.kind(), "kwd_lit" | "sym_lit"))
            {
                idx += 1;
            }
        }
        let lambda_list = els.get(idx).copied().filter(|n| is_list(n.kind()));
        let params = lambda_list
            .map(|l| lambda_list_params(ctx, l))
            .unwrap_or_default();
        let parent = if head == "defmethod" {
            lambda_list.and_then(|l| method_specializer(ctx, l))
        } else {
            None
        };
        let body_start = if lambda_list.is_some() { idx + 1 } else { 2 };
        return Some(Form::Function {
            name,
            params,
            return_type: None,
            body: els
                .get(body_start..)
                .map(<[Node]>::to_vec)
                .unwrap_or_default(),
            is_method: parent.is_some(),
            parent,
        });
    }
    if is_cl_constant_head(head) {
        let name = head_symbol_raw(ctx, second)?;
        let doc = els
            .get(3)
            .filter(|n| is_string(n.kind()))
            .map(|n| string_value(ctx, *n));
        return Some(Form::Constant { name, doc });
    }
    match head {
        "defclass" | "define-condition" | "defclass*" | "define-class" => {
            let name = head_symbol_raw(ctx, second)?;
            let extends = els
                .get(2)
                .filter(|s| is_list(s.kind()))
                .map(|s| symbols_joined(ctx, *s))
                .filter(|s| !s.is_empty());
            let fields = els
                .get(3)
                .filter(|s| is_list(s.kind()))
                .map(|s| {
                    elements(*s)
                        .into_iter()
                        .filter_map(|n| first_symbol(ctx, n))
                        .collect()
                })
                .unwrap_or_default();
            Some(Form::Record {
                name,
                extends,
                fields,
                doc: option_documentation(ctx, els.get(4..).unwrap_or_default()),
            })
        }
        "defstruct" => {
            // (defstruct name slot ...) / (defstruct (name (:include parent) ...) "doc" slot ...)
            let (name, extends) = if is_list(second.kind()) {
                let sub = elements(second);
                let name = sub.first().and_then(|n| head_symbol_raw(ctx, *n))?;
                let extends = sub.iter().skip(1).find_map(|opt| {
                    let o = elements(*opt);
                    let key = o.first().map(|k| ctx.text(*k).to_lowercase());
                    (key.as_deref() == Some(":include"))
                        .then(|| o.get(1).and_then(|p| head_symbol_raw(ctx, *p)))
                        .flatten()
                });
                (name, extends)
            } else {
                (head_symbol_raw(ctx, second)?, None)
            };
            let mut rest = &els[2..];
            let mut doc = None;
            if let Some(first) = rest.first().filter(|n| is_string(n.kind())) {
                doc = Some(string_value(ctx, *first));
                rest = &rest[1..];
            }
            let fields = rest.iter().filter_map(|n| first_symbol(ctx, *n)).collect();
            Some(Form::Record {
                name,
                extends,
                fields,
                doc,
            })
        }
        "defpackage" | "define-package" => Some(Form::Container {
            name: designator(ctx, second)?,
            body: Vec::new(),
            doc: option_documentation(ctx, &els[2..]),
        }),
        "defsystem" => Some(Form::Container {
            name: designator(ctx, second)?,
            body: Vec::new(),
            doc: plist_string(ctx, &els[2..], ":description"),
        }),
        _ => {
            // Library-defined definers: deftest, define-test, defcfun, ...
            if !is_generic_cl_definer(head) || form_rows(els) > GENERIC_DEFINER_MAX_LINES {
                return None;
            }
            let name = def_name(ctx, second)?;
            let lambda_list = els.get(2).copied().filter(|n| is_list(n.kind()));
            match lambda_list {
                Some(l) => Some(Form::Function {
                    name,
                    params: lambda_list_params(ctx, l),
                    return_type: None,
                    body: els[3..].to_vec(),
                    parent: None,
                    is_method: false,
                }),
                None if els.len() > 3 => Some(Form::Function {
                    name,
                    params: Vec::new(),
                    return_type: None,
                    body: els[2..].to_vec(),
                    parent: None,
                    is_method: false,
                }),
                None => Some(Form::Constant { name, doc: None }),
            }
        }
    }
}

/// A form headed by a library-defined definer this long is usually a data
/// table (`(define-instruction-set :avx ...)` over hundreds of lines): it is
/// left to raw code, which is chunked, rather than made one giant unit.
const GENERIC_DEFINER_MAX_LINES: usize = 300;

fn form_rows(els: &[Node]) -> usize {
    match (els.first(), els.last()) {
        (Some(f), Some(l)) => l.end_position().row.saturating_sub(f.start_position().row),
        _ => 0,
    }
}

/// `define-foo`, `define/foo`: a library or user-defined definer.
fn is_generic_scheme_definer(head: &str) -> bool {
    (head.starts_with("define-") || head.starts_with("define/"))
        && !head.ends_with('?')
        && !head.ends_with('!')
}

/// `deftest`, `define-test`, `defcfun`, ...; not `default-*`, `defined-p`.
fn is_generic_cl_definer(head: &str) -> bool {
    head.len() > 3
        && head.starts_with("def")
        && !head.ends_with("-p")
        && !["default", "defer", "defined", "deflate"]
            .iter()
            .any(|p| head.starts_with(p))
}

/// Heads that certainly define a function, for definitions nested in a
/// function body (where a generic `def*` head is more likely a call).
fn is_known_function_definer(ctx: &Ctx, head: &str) -> bool {
    if ctx.is_cl() {
        is_cl_function_head(head)
    } else {
        head == "define" || is_scheme_function_head(head)
    }
}

/// A symbol's text as written (no case folding), for definition names.
fn head_symbol_raw(ctx: &Ctx, node: Node) -> Option<String> {
    is_symbol(node.kind()).then(|| ctx.text(node).to_string())
}

/// Name of a definition: a symbol, `(setf foo)` in Common Lisp, the first
/// symbol of a list otherwise (curried defines, `(name options...)`).
fn def_name(ctx: &Ctx, node: Node) -> Option<String> {
    let kind = node.kind();
    if is_symbol(kind) {
        return Some(ctx.text(node).to_string());
    }
    if is_string(kind) || matches!(kind, "kwd_lit" | "keyword") {
        return designator(ctx, node);
    }
    if is_list(kind) {
        let els = elements(node);
        if ctx.is_cl() && els.len() == 2 && head_symbol(ctx, els[0]).as_deref() == Some("setf") {
            return Some(format!("(setf {})", ctx.text(els[1])));
        }
        return els.into_iter().find_map(|n| {
            if is_list(n.kind()) {
                def_name(ctx, n)
            } else {
                head_symbol_raw(ctx, n)
            }
        });
    }
    None
}

/// The first symbol of a node: the symbol itself, or a list's head
/// (`(x 0)` -> `x`, `[x : Integer]` -> `x`).
fn first_symbol(ctx: &Ctx, node: Node) -> Option<String> {
    if is_symbol(node.kind()) {
        return Some(ctx.text(node).to_string());
    }
    if is_list(node.kind()) {
        let first = *elements(node).first()?;
        if is_list(first.kind()) {
            return first_symbol(ctx, first);
        }
        return head_symbol_raw(ctx, first);
    }
    None
}

fn symbols_joined(ctx: &Ctx, node: Node) -> String {
    elements(node)
        .into_iter()
        .filter_map(|n| head_symbol_raw(ctx, n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// R7RS / R6RS library names: `(srfi 1)` -> `srfi 1`.
fn library_name(ctx: &Ctx, node: Node) -> Option<String> {
    if !is_list(node.kind()) {
        return designator(ctx, node);
    }
    let parts: Vec<String> = elements(node)
        .into_iter()
        .filter(|n| !is_list(n.kind()))
        .map(|n| ctx.text(n).to_string())
        .collect();
    (!parts.is_empty()).then(|| parts.join(" "))
}

/// Name and parameters from a Scheme signature list, unwrapping curried
/// definitions: `((f a) b)` -> (`f`, [a, b]).
fn signature_name_and_params(ctx: &Ctx, sig: Node) -> Option<(String, Vec<String>)> {
    let els = elements(sig);
    let head = *els.first()?;
    let (name, mut params) = if is_list(head.kind()) {
        signature_name_and_params(ctx, head)?
    } else {
        (head_symbol_raw(ctx, head)?, Vec::new())
    };
    for p in scheme_params(ctx, &els[1..]) {
        push_unique(&mut params, p);
    }
    // The dotted rest argument: `(f a . rest)`
    if let Some(rest) = sig
        .named_children(&mut sig.walk())
        .find(|c| c.kind() == "dot")
    {
        if let Some(next) = rest
            .next_named_sibling()
            .and_then(|n| head_symbol_raw(ctx, n))
        {
            push_unique(&mut params, next);
        }
    }
    Some((name, params))
}

/// Parameter names from Scheme/Racket formals: symbols, `[x default]`,
/// `[x : Type]`, keyword arguments `#:key [k 1]`, `#!optional` markers.
fn scheme_params(ctx: &Ctx, formals: &[Node]) -> Vec<String> {
    let mut out = Vec::new();
    for f in formals {
        let kind = f.kind();
        if is_symbol(kind) {
            let t = ctx.text(*f);
            if t != "." && t != "..." && !t.starts_with("#!") && !t.starts_with('&') {
                push_unique(&mut out, t.to_string());
            }
        } else if is_list(kind) {
            if let Some(n) = first_symbol(ctx, *f) {
                push_unique(&mut out, n);
            }
        }
    }
    out
}

fn lambda_params(ctx: &Ctx, head: &str, els: &[Node]) -> Vec<String> {
    match head {
        "case-lambda" => {
            let mut out = Vec::new();
            for clause in els.iter().skip(1).filter(|c| is_list(c.kind())) {
                if let Some(formals) = elements(*clause).first() {
                    for p in formals_params(ctx, *formals) {
                        push_unique(&mut out, p);
                    }
                }
            }
            out
        }
        "match-lambda" | "match-lambda*" | "match-λ" => Vec::new(),
        "named-lambda" => els
            .get(1)
            .filter(|n| is_list(n.kind()))
            .map(|n| scheme_params(ctx, &elements(*n)[1..]))
            .unwrap_or_default(),
        _ => els
            .get(1)
            .map(|n| formals_params(ctx, *n))
            .unwrap_or_default(),
    }
}

/// `(a b)`, `(a . rest)` or a bare `args` symbol.
fn formals_params(ctx: &Ctx, formals: Node) -> Vec<String> {
    if is_symbol(formals.kind()) {
        return vec![ctx.text(formals).to_string()];
    }
    if !is_list(formals.kind()) {
        return Vec::new();
    }
    if ctx.is_cl() {
        return lambda_list_params(ctx, formals);
    }
    let mut out = scheme_params(ctx, &elements(formals));
    if let Some(rest) = formals
        .named_children(&mut formals.walk())
        .find(|c| c.kind() == "dot")
    {
        if let Some(next) = rest
            .next_named_sibling()
            .and_then(|n| head_symbol_raw(ctx, n))
        {
            push_unique(&mut out, next);
        }
    }
    out
}

/// Parameter names from a Common Lisp lambda list, skipping `&optional`,
/// `&key`, ... markers; `(var default)`, `((:key var) default)` and
/// specialised `(var class)` contribute `var`.
fn lambda_list_params(ctx: &Ctx, list: Node) -> Vec<String> {
    let mut out = Vec::new();
    for el in elements(list) {
        let kind = el.kind();
        if is_symbol(kind) {
            let t = ctx.text(el);
            if !t.starts_with('&') {
                push_unique(&mut out, t.to_string());
            }
        } else if is_list(kind) {
            let sub = elements(el);
            let Some(first) = sub.first() else { continue };
            let name = if is_list(first.kind()) {
                // ((:keyword var) default)
                elements(*first)
                    .into_iter()
                    .rev()
                    .find_map(|n| head_symbol_raw(ctx, n))
            } else {
                head_symbol_raw(ctx, *first)
            };
            if let Some(n) = name {
                push_unique(&mut out, n);
            }
        }
    }
    out
}

/// The class a `defmethod` specialises its first required parameter on:
/// `((s circle) radius)` -> `circle`. `t` and `(eql ...)` are not classes.
fn method_specializer(ctx: &Ctx, list: Node) -> Option<String> {
    for el in elements(list) {
        if is_symbol(el.kind()) {
            if ctx.text(el).starts_with('&') {
                return None;
            }
            continue;
        }
        if is_list(el.kind()) {
            let sub = elements(el);
            let class = sub.get(1).and_then(|n| head_symbol_raw(ctx, *n))?;
            return (!class.eq_ignore_ascii_case("t")).then_some(class);
        }
    }
    None
}

/// `(:documentation "...")` among defclass / defgeneric / defpackage options.
fn option_documentation(ctx: &Ctx, options: &[Node]) -> Option<String> {
    options.iter().find_map(|opt| {
        if !is_list(opt.kind()) {
            return None;
        }
        let o = elements(*opt);
        let key = o.first().map(|k| ctx.text(*k).to_lowercase())?;
        (key == ":documentation")
            .then(|| o.get(1).filter(|n| is_string(n.kind())))
            .flatten()
            .map(|n| string_value(ctx, *n))
    })
}

/// `:key "value"` in a property list (defsystem options).
fn plist_string(ctx: &Ctx, items: &[Node], key: &str) -> Option<String> {
    items.windows(2).find_map(|w| {
        (ctx.text(w[0]).eq_ignore_ascii_case(key) && is_string(w[1].kind()))
            .then(|| string_value(ctx, w[1]))
    })
}

fn push_unique(target: &mut Vec<String>, value: String) {
    if !value.is_empty() && !target.contains(&value) {
        target.push(value);
    }
}

// ---------------------------------------------------------------------------
// Unit construction
// ---------------------------------------------------------------------------

fn visit_form(
    ctx: &Ctx,
    node: Node,
    parent: Option<&str>,
    file_imports: &[String],
    units: &mut Vec<CodeUnit>,
    depth: usize,
) {
    if depth > ctx.max_depth {
        return;
    }
    // `#+sbcl (defun ...)`: the feature expression belongs to the unit.
    if node.kind() == "include_reader_macro" {
        if let Some(target) = node.child_by_field_name("target") {
            if let Some(form) = classify(ctx, target, None) {
                emit(ctx, form, node, target, parent, file_imports, units, depth);
            } else {
                scan_for_definitions(ctx, target, parent, file_imports, units, depth + 1, false);
            }
        }
        return;
    }
    match classify(ctx, node, None) {
        Some(form) => emit(ctx, form, node, node, parent, file_imports, units, depth),
        None => scan_for_definitions(ctx, node, parent, file_imports, units, depth + 1, false),
    }
}

/// Look for definitions nested in a form that is not one itself:
/// `(eval-when (...) (defun ...))`, `(let ((cache ...)) (defun ...))`,
/// `(begin (define ...))`, `(cond-expand (guile (define ...)))`, and internal
/// defines in a function body (`functions_only`).
fn scan_for_definitions(
    ctx: &Ctx,
    node: Node,
    parent: Option<&str>,
    file_imports: &[String],
    units: &mut Vec<CodeUnit>,
    depth: usize,
    functions_only: bool,
) {
    if depth > ctx.max_depth {
        return;
    }
    let kind = node.kind();
    if is_quote(kind) || is_comment(kind) || node.child_count() == 0 {
        return;
    }
    if kind == "quasiquote" || kind == "syn_quoting_lit" {
        // Macro templates: their definitions only exist after expansion.
        return;
    }
    if is_list(kind) {
        let known = !functions_only
            || elements(node)
                .first()
                .and_then(|h| head_symbol(ctx, *h))
                .is_some_and(|h| is_known_function_definer(ctx, &h));
        if known {
            if let Some(form) = classify(ctx, node, None) {
                if !functions_only || matches!(form, Form::Function { .. }) {
                    emit(ctx, form, node, node, parent, file_imports, units, depth);
                }
                return;
            }
        }
    }
    for child in node.named_children(&mut node.walk()) {
        if child.kind() == "include_reader_macro" && !functions_only {
            visit_form(ctx, child, parent, file_imports, units, depth + 1);
        } else {
            scan_for_definitions(
                ctx,
                child,
                parent,
                file_imports,
                units,
                depth + 1,
                functions_only,
            );
        }
    }
}

/// First line (0-indexed) of the comment block directly above `row`, if the
/// form starts its line. Blank lines end the block.
fn leading_comment_start(ctx: &Ctx, node: Node) -> usize {
    let row = node.start_position().row;
    let line = ctx.lines.get(row).copied().unwrap_or("");
    let col = node.start_position().column.min(line.len());
    if !line.get(..col).unwrap_or("").trim().is_empty() {
        return row;
    }
    let mut start = row;
    while start > 0 {
        let prev = ctx.lines[start - 1].trim_start();
        if prev.starts_with(';') {
            start -= 1;
        } else {
            break;
        }
    }
    start
}

/// Text of the `;` comment block between `from` and `to` (exclusive).
fn comment_text(ctx: &Ctx, from: usize, to: usize) -> Option<String> {
    let text: Vec<&str> = ctx.lines[from..to]
        .iter()
        .map(|l| l.trim().trim_start_matches(';').trim())
        .filter(|l| !l.is_empty() && !l.chars().all(|c| "-=*#;".contains(c)))
        .collect();
    (!text.is_empty()).then(|| text.join(" "))
}

#[allow(clippy::too_many_arguments)]
fn new_unit(
    ctx: &Ctx,
    span: Node,
    name: String,
    unit_type: UnitType,
    parent: Option<&str>,
    end_row: Option<usize>,
) -> (CodeUnit, usize) {
    let start_row = span.start_position().row;
    let code_start = leading_comment_start(ctx, span);
    let last = ctx.lines.len().saturating_sub(1);
    let end_row = end_row
        .unwrap_or_else(|| span.end_position().row)
        .min(last)
        .max(start_row.min(last));
    let mut unit = CodeUnit::new(
        name,
        ctx.path.to_path_buf(),
        code_start + 1,
        end_row + 1,
        ctx.lang,
        unit_type,
        parent,
    );
    unit.signature = ctx
        .lines
        .get(start_row)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let end = (end_row + 1).min(ctx.lines.len());
    unit.code = ctx.lines[code_start.min(end)..end].join("\n");
    (unit, code_start)
}

/// Docstring of a function body: a leading string followed by more forms
/// (Common Lisp, Guile, Emacs-style Scheme), or `(:documentation ...)` in a
/// `defgeneric`.
fn body_docstring(ctx: &Ctx, body: &[Node]) -> Option<String> {
    if let Some(first) = body.first() {
        if is_string(first.kind()) && body.len() > 1 {
            return Some(string_value(ctx, *first));
        }
    }
    if ctx.is_cl() {
        return option_documentation(ctx, body);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn emit(
    ctx: &Ctx,
    form: Form,
    span: Node,
    node: Node,
    parent: Option<&str>,
    file_imports: &[String],
    units: &mut Vec<CodeUnit>,
    depth: usize,
) {
    match form {
        Form::Function {
            name,
            params,
            return_type,
            body,
            parent: fn_parent,
            is_method,
        } => {
            let owner = fn_parent.as_deref().or(parent);
            let unit_type = if is_method && owner.is_some() {
                UnitType::Method
            } else {
                UnitType::Function
            };
            let (mut unit, code_start) = new_unit(ctx, span, name, unit_type, owner, None);
            let comment_doc = comment_text(ctx, code_start, span.start_position().row);
            unit.docstring = body_docstring(ctx, &body).or(comment_doc);
            unit.parameters = params;
            unit.return_type = return_type;
            let mut info = BodyInfo::default();
            for form in &body {
                analyze(ctx, *form, &mut info, 0);
            }
            info.finish(&mut unit, file_imports);
            units.push(unit);
            // Internal definitions: Scheme's internal `define`s, Lisp's
            // nested `defun`s. Each also becomes a unit, as Python's nested
            // functions do.
            for form in &body {
                scan_for_definitions(ctx, *form, None, file_imports, units, depth + 1, true);
            }
        }
        Form::Record {
            name,
            extends,
            fields,
            doc,
        } => {
            let (mut unit, code_start) = new_unit(ctx, span, name, UnitType::Class, None, None);
            unit.docstring =
                doc.or_else(|| comment_text(ctx, code_start, span.start_position().row));
            unit.extends = extends;
            unit.variables = fields;
            units.push(unit);
        }
        Form::ClassExpr {
            name,
            extends,
            body,
        } => {
            let (mut unit, code_start) =
                new_unit(ctx, span, name.clone(), UnitType::Class, None, None);
            unit.docstring = comment_text(ctx, code_start, span.start_position().row);
            unit.extends = extends;
            let mut info = BodyInfo::default();
            for form in &body {
                analyze(ctx, *form, &mut info, 0);
                // Fields: (init-field a [b 1]), (field [x 0]), (init ...)
                if is_list(form.kind()) {
                    let els = elements(*form);
                    let head = els.first().and_then(|h| head_symbol(ctx, *h));
                    if matches!(
                        head.as_deref(),
                        Some("field" | "init-field" | "init" | "inherit-field")
                    ) {
                        for f in els.iter().skip(1) {
                            if let Some(n) = first_symbol(ctx, *f) {
                                push_unique(&mut info.variables, n);
                            }
                        }
                    }
                }
            }
            info.finish(&mut unit, file_imports);
            units.push(unit);
            for member in body {
                if let Some(f @ Form::Function { .. }) = classify(ctx, member, Some(&name)) {
                    emit(
                        ctx,
                        f,
                        member,
                        member,
                        Some(&name),
                        file_imports,
                        units,
                        depth + 1,
                    );
                }
            }
        }
        Form::Container { name, body, doc } => {
            let index = units.len();
            for form in &body {
                visit_form(ctx, *form, Some(&name), file_imports, units, depth + 1);
            }
            // The unit covers the header (name, exports, imports) up to the
            // first definition inside; the definitions are units of their own.
            let first_inner = units[index..].iter().map(|u| u.line).min();
            let end_row = first_inner.map(|l| l.saturating_sub(2));
            let (mut unit, code_start) = new_unit(ctx, span, name, UnitType::Class, None, end_row);
            unit.docstring =
                doc.or_else(|| comment_text(ctx, code_start, span.start_position().row));
            let mut info = BodyInfo::default();
            analyze_imports_only(ctx, node, &mut info);
            unit.imports = info.imports;
            units.insert(index, unit);
        }
        Form::Constant { name, doc } => {
            // `(define $cp0 (let () (define ...) ... ))` over thousands of
            // lines is a module written as a closure: read it as a container
            // so the definitions inside become units.
            let rows = span.end_position().row - span.start_position().row + 1;
            if rows > LONG_VALUE_LINES && is_list(node.kind()) {
                let body = elements(node)
                    .get(2..)
                    .map(<[Node]>::to_vec)
                    .unwrap_or_default();
                let form = Form::Container { name, body, doc };
                emit(ctx, form, span, node, parent, file_imports, units, depth);
                return;
            }
            let (mut unit, code_start) = new_unit(ctx, span, name, UnitType::Constant, None, None);
            unit.docstring =
                doc.or_else(|| comment_text(ctx, code_start, span.start_position().row));
            unit.imports = file_imports.to_vec();
            units.push(unit);
        }
    }
}

/// Imports declared by a container form itself (library `import`, package
/// `:use`, system `:depends-on`).
fn analyze_imports_only(ctx: &Ctx, node: Node, info: &mut BodyInfo) {
    let mut imports = Vec::new();
    collect_imports_from_form(ctx, node, &mut imports, 0);
    info.imports = imports;
}

// ---------------------------------------------------------------------------
// Body analysis: calls, local bindings, control flow
// ---------------------------------------------------------------------------

#[derive(Default)]
struct BodyInfo {
    calls: Vec<String>,
    variables: Vec<String>,
    packages: Vec<String>,
    imports: Vec<String>,
    branches: usize,
    loops: usize,
    has_error_handling: bool,
}

impl BodyInfo {
    fn finish(self, unit: &mut CodeUnit, file_imports: &[String]) {
        unit.calls = self.calls;
        unit.variables = self.variables;
        unit.has_branches = self.branches > 0;
        unit.has_loops = self.loops > 0;
        unit.has_error_handling = self.has_error_handling;
        unit.complexity = 1 + self.branches + self.loops;
        // A file import is "used" by a unit when the unit references the
        // package (`alexandria:when-let`) or calls something named after it.
        let packages = self.packages;
        unit.imports = file_imports
            .iter()
            .filter(|imp| {
                let imp_l = imp.to_lowercase();
                let last = imp_l
                    .rsplit(['/', ' ', ':'])
                    .next()
                    .unwrap_or(&imp_l)
                    .trim_end_matches(".rkt")
                    .trim_end_matches(".scm")
                    .to_string();
                packages.iter().any(|p| *p == imp_l || *p == last)
            })
            .cloned()
            .collect();
    }
}

/// Forms that are syntax, not calls worth recording.
fn is_special_form(head: &str) -> bool {
    matches!(
        head,
        "define"
            | "define*"
            | "lambda"
            | "λ"
            | "let"
            | "let*"
            | "letrec"
            | "letrec*"
            | "let-values"
            | "let*-values"
            | "letrec-values"
            | "define-values"
            | "if"
            | "cond"
            | "case"
            | "when"
            | "unless"
            | "and"
            | "or"
            | "not"
            | "begin"
            | "begin0"
            | "progn"
            | "prog1"
            | "prog2"
            | "quote"
            | "quasiquote"
            | "unquote"
            | "set!"
            | "setf"
            | "setq"
            | "do"
            | "do*"
            | "loop"
            | "declare"
            | "the"
            | "function"
            | "block"
            | "return"
            | "return-from"
            | "else"
            | "otherwise"
            | "t"
            | "flet"
            | "labels"
            | "macrolet"
            | "symbol-macrolet"
            | "locally"
            | "multiple-value-bind"
            | "destructuring-bind"
            | "values"
            | "syntax"
            | "quasisyntax"
            | "syntax-rules"
            | "syntax-case"
            | "unsyntax"
            | "#%app"
            | "=>"
            | "..."
            | "_"
    )
}

/// `(let ((x 1) (y 2)) ...)`-shaped forms: index of the bindings list.
fn bindings_index(ctx: &Ctx, head: &str, els: &[Node]) -> Option<usize> {
    let named_let = matches!(head, "let" | "let*")
        && !ctx.is_cl()
        && els.get(1).is_some_and(|n| is_symbol(n.kind()));
    if named_let {
        return Some(2);
    }
    matches!(
        head,
        "let"
            | "let*"
            | "letrec"
            | "letrec*"
            | "let-values"
            | "let*-values"
            | "letrec-values"
            | "let-syntax"
            | "letrec-syntax"
            | "fluid-let"
            | "parameterize"
            | "parameterize*"
            | "with-syntax"
            | "syntax-parameterize"
            | "do"
            | "do*"
            | "let-values*"
            | "receive"
            | "with-slots"
            | "with-accessors"
            | "symbol-macrolet"
            | "let1"
    )
    .then_some(1)
}

fn is_loop_head(head: &str) -> bool {
    matches!(
        head,
        "do" | "do*"
            | "dolist"
            | "dotimes"
            | "loop"
            | "while"
            | "until"
            | "iterate"
            | "iter"
            | "do-symbols"
            | "do-external-symbols"
            | "do-all-symbols"
            | "for-each"
            | "vector-for-each"
            | "string-for-each"
            | "hash-table-walk"
            | "maphash"
    ) || head.starts_with("for/")
        || head.starts_with("for*/")
        || head == "for"
        || head == "for*"
}

fn is_branch_head(head: &str) -> bool {
    matches!(
        head,
        "if" | "cond"
            | "case"
            | "ecase"
            | "ccase"
            | "typecase"
            | "etypecase"
            | "ctypecase"
            | "when"
            | "unless"
            | "match"
            | "match*"
            | "case-lambda"
            | "if-let"
            | "when-let"
            | "when-let*"
            | "and-let*"
            | "switch"
            | "cond-expand"
    )
}

fn is_error_head(head: &str) -> bool {
    matches!(
        head,
        "handler-case"
            | "handler-bind"
            | "ignore-errors"
            | "unwind-protect"
            | "restart-case"
            | "restart-bind"
            | "with-handlers"
            | "with-handlers*"
            | "guard"
            | "dynamic-wind"
            | "catch"
            | "throw"
            | "error"
            | "cerror"
            | "raise"
            | "raise-argument-error"
            | "raise-user-error"
            | "signal"
            | "with-exception-handler"
            | "call-with-exception-handler"
            | "call/ec"
            | "condition-case"
            | "assert"
            | "check-type"
    )
}

/// Iteration forms whose first argument(s) bind loop variables: how many
/// binding specs follow the head.
fn iteration_specs(head: &str) -> Option<usize> {
    match head {
        "dolist" | "dotimes" | "do-symbols" | "do-external-symbols" | "do-all-symbols" => Some(1),
        "for/fold" | "for*/fold" | "for/foldr" | "for*/foldr" => Some(2),
        "for" | "for*" => Some(1),
        _ if head.starts_with("for/") || head.starts_with("for*/") => Some(1),
        _ => None,
    }
}

/// Clause forms whose clause heads are data or tests, not calls:
/// returns the index of the first clause.
fn clauses_index(head: &str) -> Option<usize> {
    match head {
        "cond" | "case-lambda" | "cond-expand" | "syntax-rules" => {
            if head == "syntax-rules" {
                Some(2)
            } else {
                Some(1)
            }
        }
        "case" | "ecase" | "ccase" | "typecase" | "etypecase" | "ctypecase" | "match"
        | "match*" | "syntax-case" | "syntax-parse" | "handler-case" | "restart-case"
        | "string-case" | "switch" => Some(2),
        "match-lambda" | "match-lambda*" | "match-λ" => Some(1),
        _ => None,
    }
}

fn analyze(ctx: &Ctx, node: Node, info: &mut BodyInfo, depth: usize) {
    if depth > ctx.max_depth {
        return;
    }
    let kind = node.kind();
    if is_quote(kind) || is_comment(kind) {
        return;
    }
    if kind == "var_quoting_lit" {
        // #'foo is a function reference
        if let Some(v) = node.child_by_field_name("value") {
            if let Some(h) = head_symbol(ctx, v) {
                push_unique(&mut info.calls, h);
            }
            if let Some(p) = symbol_package(ctx, v) {
                push_unique(&mut info.packages, p);
            }
        }
        return;
    }
    if kind == "package_lit" {
        if let Some(p) = symbol_package(ctx, node) {
            push_unique(&mut info.packages, p);
        }
        return;
    }
    if kind == "loop_macro" {
        info.loops += 1;
    }
    if !is_list(kind) {
        for child in node.named_children(&mut node.walk()) {
            analyze(ctx, child, info, depth + 1);
        }
        return;
    }
    let els = elements(node);
    let Some(first) = els.first().copied() else {
        return;
    };
    let Some(head) = head_symbol(ctx, first) else {
        for el in &els {
            analyze(ctx, *el, info, depth + 1);
        }
        return;
    };
    if let Some(p) = symbol_package(ctx, first) {
        push_unique(&mut info.packages, p);
    }
    if is_branch_head(&head) {
        info.branches += 1;
    }
    if is_loop_head(&head) {
        info.loops += 1;
    }
    if is_error_head(&head) {
        info.has_error_handling = true;
    }
    if !is_special_form(&head)
        && head.chars().any(char::is_alphabetic)
        && !LAMBDA_HEADS.contains(&head.as_str())
        && !head.starts_with("define")
        && !head.starts_with(':')
        && !head.starts_with("#:")
    {
        push_unique(&mut info.calls, head.clone());
    }

    let rest: &[Node] = &els[1..];
    // Lambda and nested definitions: skip the formals / signature.
    if LAMBDA_HEADS.contains(&head.as_str())
        || head.starts_with("define")
        || is_cl_function_head(&head)
        || is_cl_constant_head(&head)
    {
        for el in rest.iter().skip(1) {
            analyze(ctx, *el, info, depth + 1);
        }
        return;
    }
    if let Some(specs) = iteration_specs(&head) {
        // (dolist (x items) ...), (for/list ([x xs] #:when c) ...),
        // (for/fold ([acc 0]) ([x xs]) ...): the specs bind variables.
        info.loops += 1;
        let single = matches!(
            head.as_str(),
            "dolist" | "dotimes" | "do-symbols" | "do-external-symbols" | "do-all-symbols"
        );
        for spec in els.iter().skip(1).take(specs) {
            if !is_list(spec.kind()) {
                analyze(ctx, *spec, info, depth + 1);
                continue;
            }
            let bindings = if single { vec![*spec] } else { elements(*spec) };
            let mut after_keyword = false;
            for b in bindings {
                // `#:when (even? x)`: an expression, not a binding
                if !is_list(b.kind()) || after_keyword {
                    after_keyword = b.kind() == "keyword";
                    analyze(ctx, b, info, depth + 1);
                    continue;
                }
                let parts = elements(b);
                if let Some(var) = parts.first() {
                    if is_symbol(var.kind()) {
                        push_unique(&mut info.variables, ctx.text(*var).to_string());
                    } else if is_list(var.kind()) {
                        for v in elements(*var) {
                            if is_symbol(v.kind()) {
                                push_unique(&mut info.variables, ctx.text(v).to_string());
                            }
                        }
                    }
                }
                for v in parts.iter().skip(1) {
                    analyze(ctx, *v, info, depth + 1);
                }
            }
        }
        for el in els.iter().skip(1 + specs) {
            analyze(ctx, *el, info, depth + 1);
        }
        return;
    }
    if let Some(idx) = bindings_index(ctx, &head, &els) {
        if idx == 2 {
            // named let: the loop name is a local function
            info.loops += 1;
        }
        if let Some(bindings) = els.get(idx) {
            if is_list(bindings.kind()) {
                for b in elements(*bindings) {
                    if is_symbol(b.kind()) {
                        push_unique(&mut info.variables, ctx.text(b).to_string());
                    } else if is_list(b.kind()) {
                        let parts = elements(b);
                        if let Some(n) = parts.first() {
                            if is_symbol(n.kind()) {
                                push_unique(&mut info.variables, ctx.text(*n).to_string());
                            } else if is_list(n.kind()) {
                                // let-values: ((a b) expr)
                                for v in elements(*n) {
                                    if is_symbol(v.kind()) {
                                        push_unique(&mut info.variables, ctx.text(v).to_string());
                                    }
                                }
                            }
                        }
                        for v in parts.iter().skip(1) {
                            analyze(ctx, *v, info, depth + 1);
                        }
                    }
                }
            }
        }
        for el in els.iter().skip(idx + 1) {
            analyze(ctx, *el, info, depth + 1);
        }
        return;
    }
    if matches!(head.as_str(), "flet" | "labels" | "macrolet") {
        if let Some(bindings) = els.get(1) {
            for b in elements(*bindings) {
                for v in elements(b).iter().skip(2) {
                    analyze(ctx, *v, info, depth + 1);
                }
            }
        }
        for el in els.iter().skip(2) {
            analyze(ctx, *el, info, depth + 1);
        }
        return;
    }
    if matches!(
        head.as_str(),
        "multiple-value-bind" | "destructuring-bind" | "receive"
    ) {
        if let Some(vars) = els.get(1) {
            if is_list(vars.kind()) {
                for v in elements(*vars) {
                    if is_symbol(v.kind()) {
                        let t = ctx.text(v);
                        if !t.starts_with('&') {
                            push_unique(&mut info.variables, t.to_string());
                        }
                    }
                }
            }
        }
        for el in els.iter().skip(2) {
            analyze(ctx, *el, info, depth + 1);
        }
        return;
    }
    if let Some(idx) = clauses_index(&head) {
        for el in els.iter().take(idx).skip(1) {
            analyze(ctx, *el, info, depth + 1);
        }
        let data_head = !matches!(head.as_str(), "cond" | "cond-expand");
        for clause in els.iter().skip(idx) {
            if !is_list(clause.kind()) {
                analyze(ctx, *clause, info, depth + 1);
                continue;
            }
            let parts = elements(*clause);
            // handler-case / restart-case clauses: (type (var) body ...)
            let lambda_list = matches!(head.as_str(), "handler-case" | "restart-case");
            for (i, part) in parts.iter().enumerate() {
                if i == 0 && (data_head || !is_list(part.kind())) {
                    continue;
                }
                if i == 1 && lambda_list && is_list(part.kind()) {
                    continue;
                }
                analyze(ctx, *part, info, depth + 1);
            }
        }
        return;
    }
    for el in rest {
        analyze(ctx, *el, info, depth + 1);
    }
}

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

/// Libraries, modules, packages and files the file depends on.
fn collect_file_imports(ctx: &Ctx, root: Node) -> Vec<String> {
    let mut imports = Vec::new();
    for child in root.named_children(&mut root.walk()) {
        if child.kind() == "extension" {
            // #lang racket/base
            if let Some(lang_name) = child
                .named_children(&mut child.walk())
                .find(|c| c.kind() == "lang_name")
            {
                push_unique(&mut imports, ctx.text(lang_name).trim().to_string());
            }
            continue;
        }
        collect_imports_from_form(ctx, child, &mut imports, 0);
    }
    imports
}

fn collect_imports_from_form(ctx: &Ctx, node: Node, out: &mut Vec<String>, depth: usize) {
    if depth > 64 {
        return;
    }
    if node.kind() == "include_reader_macro" {
        if let Some(target) = node.child_by_field_name("target") {
            collect_imports_from_form(ctx, target, out, depth + 1);
        }
        return;
    }
    if !is_list(node.kind()) {
        return;
    }
    let els = elements(node);
    let Some(head) = els.first().and_then(|h| head_symbol(ctx, *h)) else {
        return;
    };
    match head.as_str() {
        "import"
        | "require"
        | "use-modules"
        | "require-extension"
        | "require-library"
        | "use"
        | "include"
        | "include-ci"
        | "include-library-declarations"
        | "load"
        | "quickload"
        | "load-system"
        | "require-system" => {
            for spec in &els[1..] {
                import_specs(ctx, *spec, out, depth + 1);
            }
        }
        "define-module" => {
            // (define-module (x y) #:use-module (ice-9 match) #:autoload (m) (syms))
            for w in els.windows(2) {
                let key = ctx.text(w[0]);
                if matches!(key, "#:use-module" | "#:autoload" | "#:re-export-module") {
                    import_specs(ctx, w[1], out, depth + 1);
                }
            }
        }
        "defpackage" | "define-package" => {
            for opt in &els[2..] {
                if !is_list(opt.kind()) {
                    continue;
                }
                let o = elements(*opt);
                let key = o.first().map(|k| ctx.text(*k).to_lowercase());
                match key.as_deref() {
                    Some(":use" | ":mix" | ":use-reexport" | ":reexport" | ":recycle") => {
                        for p in &o[1..] {
                            if let Some(name) = designator(ctx, *p) {
                                if name != "cl" && name != "common-lisp" {
                                    push_unique(out, name);
                                }
                            }
                        }
                    }
                    Some(":import-from" | ":shadowing-import-from") => {
                        if let Some(name) = o.get(1).and_then(|p| designator(ctx, *p)) {
                            push_unique(out, name);
                        }
                    }
                    Some(":local-nicknames") => {
                        for pair in &o[1..] {
                            if let Some(name) =
                                elements(*pair).get(1).and_then(|p| designator(ctx, *p))
                            {
                                push_unique(out, name);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        "defsystem" => {
            for w in els.windows(2) {
                if ctx.text(w[0]).eq_ignore_ascii_case(":depends-on") && is_list(w[1].kind()) {
                    for dep in elements(w[1]) {
                        import_specs(ctx, dep, out, depth + 1);
                    }
                }
            }
        }
        // Containers and top-level splices: imports inside them count.
        "define-library" | "library" | "module" | "module*" | "module+" | "begin" | "progn"
        | "eval-when" | "cond-expand" | "begin-for-syntax" => {
            for el in &els[1..] {
                collect_imports_from_form(ctx, *el, out, depth + 1);
                // cond-expand clauses: (feature (import ...))
                if head == "cond-expand" && is_list(el.kind()) {
                    for inner in elements(*el).iter().skip(1) {
                        collect_imports_from_form(ctx, *inner, out, depth + 1);
                    }
                }
            }
        }
        _ => {}
    }
}

/// One import spec: `racket/list`, `"util.rkt"`, `(srfi 1)`,
/// `(only (scheme base) car)`, `(prefix-in p: lib)`, `(for-syntax a b)`,
/// `:alexandria`, `(:version "x" "1.0")`.
fn import_specs(ctx: &Ctx, spec: Node, out: &mut Vec<String>, depth: usize) {
    if depth > 64 {
        return;
    }
    let kind = spec.kind();
    if is_quote(kind) || kind == "var_quoting_lit" {
        // '(:alexandria :cl-ppcre) / 'alexandria
        if let Some(v) = spec
            .child_by_field_name("value")
            .or_else(|| spec.named_child(spec.named_child_count().saturating_sub(1)))
        {
            if is_list(v.kind()) {
                for el in elements(v) {
                    import_specs(ctx, el, out, depth + 1);
                }
            } else {
                import_specs(ctx, v, out, depth + 1);
            }
        }
        return;
    }
    if !is_list(kind) {
        if let Some(name) = designator(ctx, spec) {
            if !name.starts_with('#') {
                push_unique(out, name);
            }
        }
        return;
    }
    let els = elements(spec);
    let Some(first) = els.first() else { return };
    if is_list(first.kind()) {
        // Guile: ((srfi srfi-1) #:select (fold))
        import_specs(ctx, *first, out, depth + 1);
        return;
    }
    let head = head_symbol(ctx, *first).or_else(|| designator(ctx, *first));
    match head.as_deref() {
        Some(
            "only" | "except" | "rename" | "prefix" | "only-in" | "except-in" | "rename-in"
            | "submod" | "file" | "lib" | "planet" | "library" | "only-meta-in",
        ) => {
            if let Some(inner) = els.get(1) {
                import_specs(ctx, *inner, out, depth + 1);
            }
        }
        Some("prefix-in" | "filtered-in") => {
            if let Some(inner) = els.get(2) {
                import_specs(ctx, *inner, out, depth + 1);
            }
        }
        Some(
            "for-syntax" | "for-template" | "for-label" | "for-meta" | "combine-in" | "relative-in"
            | "for-space",
        ) => {
            for inner in &els[1..] {
                import_specs(ctx, *inner, out, depth + 1);
            }
        }
        // ASDF: (:version "name" "1.0"), (:feature :sbcl "name")
        Some("version") | Some("feature") if ctx.is_cl() => {
            if let Some(inner) = els.iter().skip(1).find(|n| is_string(n.kind())) {
                import_specs(ctx, *inner, out, depth + 1);
            }
        }
        _ => {
            if let Some(name) = library_name(ctx, spec) {
                push_unique(out, name);
            }
        }
    }
}
