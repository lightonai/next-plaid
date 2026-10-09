//! Tests for LaTeX and BibTeX extraction (text-based, section units).

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const PAPER: &str = r#"\documentclass[11pt]{article}
\usepackage{amsmath,amssymb}
\usepackage[utf8]{inputenc}
\newcommand{\R}{\mathbb{R}}

% Softmax over the last axis.
\newcommand{\softmax}[1]{%
  \operatorname{softmax}\left(#1\right)%
}

\begin{document}
\title{Late Interaction}
\maketitle

\begin{abstract}
We study late interaction retrieval.
\end{abstract}

\section{Introduction}\label{sec:intro}
Dense retrieval compresses a document into one vector.

\subsection{Contributions}
We propose \emph{MaxSim} scoring.

\section{Method}
Each query token is matched to its most similar document token.
\end{document}
"#;

#[test]
fn test_section_embedding_text() {
    let units = assert_extractor_invariants(PAPER, Language::Latex, "paper.tex");
    let intro = get_unit_by_name(&units, "Introduction").unwrap();
    assert_eq!(intro.unit_type, UnitType::Section);
    let expected = r#"Section: Introduction
Signature: \section{Introduction}\label{sec:intro}
File: paper paper.tex
Code:
\section{Introduction}\label{sec:intro}
Dense retrieval compresses a document into one vector."#;
    assert_eq!(build_embedding_text(intro), expected);
}

#[test]
fn test_subsection_has_parent_section() {
    let units = parse(PAPER, Language::Latex, "paper.tex");
    let sub = get_unit_by_name(&units, "Contributions").unwrap();
    assert_eq!(sub.parent_class.as_deref(), Some("Introduction"));
    assert_eq!((sub.line, sub.end_line), (22, 23));
    let method = get_unit_by_name(&units, "Method").unwrap();
    assert_eq!(method.parent_class, None);
    assert!(method.code.contains("most similar document token"));
    let abs = get_unit_by_name(&units, "Abstract").unwrap();
    assert!(abs.code.contains("late interaction retrieval"));
}

#[test]
fn test_preamble_imports_and_definitions() {
    let units = parse(PAPER, Language::Latex, "paper.tex");
    let preamble = get_unit_by_name(&units, "preamble").unwrap();
    assert_eq!(
        preamble.imports,
        vec!["article", "amsmath", "amssymb", "inputenc"]
    );
    // One-line definitions stay in the preamble text.
    assert!(preamble.code.contains("\\newcommand{\\R}"));
    assert!(get_unit_by_name(&units, "\\R").is_none());

    // A multi-line definition is its own unit with its comment as doc.
    let softmax = get_unit_by_name(&units, "\\softmax").unwrap();
    assert_eq!(softmax.unit_type, UnitType::Function);
    assert_eq!((softmax.line, softmax.end_line), (6, 9));
    assert_eq!(
        softmax.docstring.as_deref(),
        Some("Softmax over the last axis.")
    );
    assert!(!preamble.code.contains("operatorname"));

    // Front matter after \begin{document} is named after the file.
    let front = get_unit_by_name(&units, "paper").unwrap();
    assert!(front.code.contains("\\maketitle"));
}

#[test]
fn test_starred_and_short_title_headings() {
    let source = r#"\chapter*{Acknowledgements}
Thanks.
\section[Short]{A \textbf{Long} Title~with $x$}
Body.
\paragraph{Training details.} We train for 10 epochs.
"#;
    let units = assert_extractor_invariants(source, Language::Latex, "thesis.tex");
    assert!(get_unit_by_name(&units, "Acknowledgements").is_some());
    let long = get_unit_by_name(&units, "A Long Title with $x$").unwrap();
    assert_eq!(long.parent_class.as_deref(), Some("Acknowledgements"));
    let para = get_unit_by_name(&units, "Training details.").unwrap();
    assert_eq!(para.parent_class.as_deref(), Some("A Long Title with $x$"));
}

/// A title made of a macro keeps the macro's name instead of vanishing.
#[test]
fn test_macro_title() {
    let source = "\\section{\\sys{}}\ntext\n\\subsection{Re-ranking with \\sys{}}\nmore\n";
    let units = assert_extractor_invariants(source, Language::Latex, "a.tex");
    assert!(get_unit_by_name(&units, "sys").is_some());
    let sub = get_unit_by_name(&units, "Re-ranking with sys").unwrap();
    assert_eq!(sub.parent_class.as_deref(), Some("sys"));
}

#[test]
fn test_commented_out_heading_is_not_a_section() {
    let source = "\\section{Real}\ntext\n% \\section{Old}\nmore 50\\% text\n";
    let units = assert_extractor_invariants(source, Language::Latex, "a.tex");
    assert!(get_unit_by_name(&units, "Old").is_none());
    let real = get_unit_by_name(&units, "Real").unwrap();
    assert_eq!(real.end_line, 4);
}

#[test]
fn test_style_file_definitions() {
    let source = r#"\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{mymacros}
\RequirePackage{xcolor}

% Draw a todo note in the margin.
\NewDocumentCommand{\todo}{O{red} m}{%
  \marginpar{\color{#1}#2}%
}

\def\@mymacro#1#2{%
  #1 and #2%
}

\newenvironment{keypoint}[1][Note]
  {\begin{quote}\textbf{#1:}}
  {\end{quote}}

\long\def\ignore#1{}
"#;
    let units = assert_extractor_invariants(source, Language::Latex, "mymacros.sty");
    let todo = get_unit_by_name(&units, "\\todo").unwrap();
    assert_eq!((todo.line, todo.end_line), (5, 8));
    assert_eq!(
        todo.docstring.as_deref(),
        Some("Draw a todo note in the margin.")
    );
    let mymacro = get_unit_by_name(&units, "\\@mymacro").unwrap();
    assert_eq!((mymacro.line, mymacro.end_line), (10, 12));
    let env = get_unit_by_name(&units, "keypoint").unwrap();
    assert_eq!((env.line, env.end_line), (14, 16));
    assert_eq!(env.unit_type, UnitType::Function);
    // Not a multi-line definition: stays in the surrounding text.
    assert!(get_unit_by_name(&units, "\\ignore").is_none());
    // No \begin{document}: loose text is named after the file.
    let head = get_unit_by_name(&units, "mymacros").unwrap();
    assert!(head.code.contains("\\ProvidesPackage"));
}

#[test]
fn test_beamer_frames() {
    let source = r#"\begin{document}
\begin{frame}{Motivation}
Why late interaction?
\end{frame}
\begin{frame}[fragile]
\frametitle{Results}
Numbers.
\end{frame}
\end{document}
"#;
    let units = assert_extractor_invariants(source, Language::Latex, "talk.tex");
    assert!(get_unit_by_name(&units, "Motivation").is_some());
    let results = get_unit_by_name(&units, "Results").unwrap();
    assert!(results.code.contains("Numbers."));
}

#[test]
fn test_long_section_is_chunked_at_paragraphs() {
    let mut source = String::from("\\section{Long}\n");
    for p in 0..40 {
        source.push_str(&format!(
            "Paragraph {p} line one.\nParagraph {p} line two.\n\n"
        ));
    }
    let units = assert_extractor_invariants(&source, Language::Latex, "long.tex");
    let chunks: Vec<_> = units.iter().filter(|u| u.name == "Long").collect();
    assert!(chunks.len() >= 2, "expected several chunks");
    for c in &chunks {
        assert!(c.end_line + 1 - c.line <= 60, "chunk too long");
        // Cut at a paragraph break: never ends mid-paragraph.
        assert!(c.code.trim_end().ends_with("line two."), "{:?}", c.code);
    }
}

#[test]
fn test_unbalanced_definition_does_not_swallow_file() {
    let source = "\\newcommand{\\broken}{\n  unclosed\n\\section{After}\ntext\n";
    let units = assert_extractor_invariants(source, Language::Latex, "broken.tex");
    // The definition never closes; it must not hide the section.
    assert!(get_unit_by_name(&units, "After").is_some());
}

#[test]
fn test_bibtex_entries() {
    let source = r#"@string{nips = "NeurIPS"}

@inproceedings{vaswani2017attention,
  title = {Attention Is All You Need},
  author = {Vaswani, Ashish and others},
  booktitle = nips,
  year = {2017}
}

@article{khattab2020colbert,
  title = "{ColBERT}: Efficient and Effective Passage Search",
  year = 2020,
}
"#;
    let units = assert_extractor_invariants(source, Language::Latex, "refs.bib");
    let attn = get_unit_by_name(&units, "vaswani2017attention").unwrap();
    assert_eq!((attn.line, attn.end_line), (3, 8));
    assert_eq!(attn.docstring.as_deref(), Some("Attention Is All You Need"));
    let colbert = get_unit_by_name(&units, "khattab2020colbert").unwrap();
    assert_eq!(
        colbert.docstring.as_deref(),
        Some("ColBERT: Efficient and Effective Passage Search")
    );
    let text = build_embedding_text(attn);
    assert!(text.starts_with(
        "Section: vaswani2017attention\nSignature: @inproceedings{vaswani2017attention,\nDescription: Attention Is All You Need\n"
    ));
}

/// `.cls` is also Visual Basic 6's class-module extension: such a file has
/// no LaTeX structure and is indexed as plain chunks, never dropped.
#[test]
fn test_cls_that_is_not_latex() {
    let source = "VERSION 1.0 CLASS\nBEGIN\n  MultiUse = -1\nEND\nPublic Function Add(a, b)\n  Add = a + b\nEnd Function\n";
    let units = assert_extractor_invariants(source, Language::Latex, "Calc.cls");
    assert!(!units.is_empty());
    assert!(units.iter().all(|u| u.unit_type != UnitType::Function));
}

#[test]
fn test_empty_file() {
    assert!(parse("", Language::Latex, "empty.tex").is_empty());
    assert!(parse("\n\n", Language::Latex, "empty.bib").is_empty());
}
