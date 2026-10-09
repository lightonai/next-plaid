//! Tests for Jupyter notebook (`.ipynb`) extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{CodeUnit, Language, UnitType};

/// An nbformat v4 notebook as Jupyter writes it (one-space indent, one source
/// line per JSON string), with an image output that must never be indexed.
const NOTEBOOK: &str = r##"{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Training a classifier\n",
    "\n",
    "We fit a logistic regression on the iris dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install scikit-learn\n",
    "%matplotlib inline\n",
    "import numpy as np\n",
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def train(X, y):\n",
    "    \"\"\"Fit a logistic regression model.\"\"\"\n",
    "    model = LogisticRegression()\n",
    "    model.fit(X, y)\n",
    "    return model\n",
    "\n",
    "clf = train(np.zeros((4, 2)), [0, 1, 0, 1])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"##;

fn notebook(file: &str, source: &str) -> Vec<CodeUnit> {
    parse(source, Language::Notebook, file)
}

/// Build a v4 notebook from `(cell_type, source lines)` and a kernel language.
fn build_notebook(language: &str, cells: &[(&str, &[&str])]) -> String {
    let cells: Vec<serde_json::Value> = cells
        .iter()
        .map(|(cell_type, lines)| {
            let n = lines.len();
            let source: Vec<String> = lines
                .iter()
                .enumerate()
                .map(|(i, l)| {
                    if i + 1 < n {
                        format!("{l}\n")
                    } else {
                        l.to_string()
                    }
                })
                .collect();
            serde_json::json!({
                "cell_type": cell_type,
                "metadata": {},
                "outputs": [],
                "source": source,
            })
        })
        .collect();
    let nb = serde_json::json!({
        "cells": cells,
        "metadata": {"language_info": {"name": language}},
        "nbformat": 4,
        "nbformat_minor": 5,
    });
    serde_json::to_string_pretty(&nb).unwrap()
}

/// The line of the `.ipynb` file holding the JSON string of `needle`.
fn file_line(source: &str, needle: &str) -> usize {
    source
        .lines()
        .position(|l| l.contains(needle))
        .map(|i| i + 1)
        .unwrap_or_else(|| panic!("{needle:?} not in notebook"))
}

#[test]
fn test_function_in_code_cell() {
    let units = notebook("train.ipynb", NOTEBOOK);

    let train = get_unit_by_name(&units, "train").unwrap();
    assert_eq!(train.unit_type, UnitType::Function);
    assert_eq!(train.language, Language::Python);
    // Lines are those of the cell's source strings in the .ipynb file.
    assert_eq!(
        (train.line, train.end_line),
        (
            file_line(NOTEBOOK, "def train(X, y)"),
            file_line(NOTEBOOK, "    return model")
        )
    );
    let text = build_embedding_text(train);
    let expected = r#"Function: train
Signature: def train(X, y):
Description: """Fit a logistic regression model.
Parameters: X, y
Calls: LogisticRegression, fit
Variables: model
File: train train.ipynb
Code:
def train(X, y):
    """Fit a logistic regression model."""
    model = LogisticRegression()
    model.fit(X, y)
    return model"#;
    assert_eq!(text, expected);
}

#[test]
fn test_markdown_cell_is_a_section() {
    let units = notebook("train.ipynb", NOTEBOOK);

    let section = get_unit_by_name(&units, "Training a classifier").unwrap();
    assert_eq!(section.unit_type, UnitType::Section);
    assert_eq!(section.language, Language::Markdown);
    assert_eq!(
        (section.line, section.end_line),
        (
            file_line(NOTEBOOK, "# Training a classifier"),
            file_line(NOTEBOOK, "We fit a logistic regression")
        )
    );
    assert_eq!(
        section.code,
        "# Training a classifier\n\nWe fit a logistic regression on the iris dataset."
    );
}

#[test]
fn test_statements_and_magics_become_raw_code_per_cell() {
    let units = notebook("train.ipynb", NOTEBOOK);

    // The import cell: magics and shell escapes stay in the unit's code but
    // do not break the parse.
    let imports_line = file_line(NOTEBOOK, "!pip install scikit-learn");
    let imports = units
        .iter()
        .find(|u| u.line == imports_line)
        .expect("import cell unit");
    assert_eq!(imports.unit_type, UnitType::RawCode);
    assert_eq!(imports.name, format!("raw_code_{imports_line}"));
    assert_eq!(
        imports.end_line,
        file_line(NOTEBOOK, "from sklearn.linear_model")
    );
    assert!(imports
        .code
        .starts_with("!pip install scikit-learn\n%matplotlib inline"));

    // The statement after the function is its own unit, inside its cell.
    let clf_line = file_line(NOTEBOOK, "clf = train(");
    let clf = units.iter().find(|u| u.line == clf_line).unwrap();
    assert_eq!(clf.unit_type, UnitType::RawCode);
    assert_eq!(clf.end_line, clf_line);
    assert_eq!(clf.code, "clf = train(np.zeros((4, 2)), [0, 1, 0, 1])");
}

#[test]
fn test_outputs_are_never_indexed() {
    let units = notebook("train.ipynb", NOTEBOOK);
    assert_eq!(
        units.len(),
        4,
        "{:?}",
        units.iter().map(|u| &u.name).collect::<Vec<_>>()
    );
    for unit in &units {
        assert!(!unit.code.contains("iVBORw0KGgo"), "{}", unit.name);
        assert!(!unit.code.contains("<Figure"), "{}", unit.name);
        assert!(!unit.code.contains("execution_count"), "{}", unit.name);
    }
}

/// Every unit's code lines line up with the file lines it claims, so `-e`
/// matches and result snippets land on the right line.
#[test]
fn test_unit_code_aligns_with_file_lines() {
    let units = notebook("train.ipynb", NOTEBOOK);
    let file_lines: Vec<&str> = NOTEBOOK.lines().collect();
    for unit in &units {
        for (i, code_line) in unit.code.lines().enumerate() {
            let json = file_lines[unit.line + i - 1].trim().trim_end_matches(',');
            let decoded: String = serde_json::from_str(json).unwrap();
            assert_eq!(decoded.trim_end_matches('\n'), code_line, "{}", unit.name);
        }
    }
}

#[test]
fn test_class_with_methods() {
    let source = build_notebook(
        "python",
        &[(
            "code",
            &[
                "class Trainer:",
                "    \"\"\"Runs the training loop.\"\"\"",
                "",
                "    def __init__(self, model):",
                "        self.model = model",
                "",
                "    def step(self, batch):",
                "        return self.model(batch)",
            ],
        )],
    );
    let units = notebook("loop.ipynb", &source);

    let class = get_unit_by_name(&units, "Trainer").unwrap();
    assert_eq!(class.unit_type, UnitType::Class);
    assert_eq!(class.line, file_line(&source, "class Trainer:"));
    let step = get_unit_by_name(&units, "step").unwrap();
    assert_eq!(step.unit_type, UnitType::Method);
    assert_eq!(step.parent_class.as_deref(), Some("Trainer"));
    assert_eq!(step.line, file_line(&source, "def step(self, batch)"));
    assert_eq!(
        step.end_line,
        file_line(&source, "return self.model(batch)")
    );
}

/// Imports from an early cell are attributed to functions in later cells.
#[test]
fn test_imports_span_cells() {
    let source = build_notebook(
        "python",
        &[
            ("code", &["import json"]),
            ("markdown", &["## Loading"]),
            ("code", &["def load(text):", "    return json.loads(text)"]),
        ],
    );
    let units = notebook("load.ipynb", &source);
    let load = get_unit_by_name(&units, "load").unwrap();
    // Same imports as when the cells are one Python file.
    let as_file = parse(
        "import json\n\ndef load(text):\n    return json.loads(text)",
        Language::Python,
        "load.py",
    );
    let expected = get_unit_by_name(&as_file, "load").unwrap();
    assert_eq!(load.imports, expected.imports);
    assert_eq!(load.imports, vec!["json"]);
    assert!(load.calls.contains(&"loads".to_string()));
    assert_eq!(
        get_unit_by_name(&units, "Loading").unwrap().unit_type,
        UnitType::Section
    );
}

#[test]
fn test_source_as_single_string() {
    let source = r##"{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": "# Title\nSome text"},
  {"cell_type": "code", "metadata": {}, "outputs": [],
   "source": "def add(a, b):\n    return a + b\n\nadd(1, 2)"}
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}"##;
    let units = notebook("single.ipynb", source);
    let add = get_unit_by_name(&units, "add").unwrap();
    let line = file_line(source, "def add(a, b)");
    // All of the cell sits on one line of the file.
    assert_eq!((add.line, add.end_line), (line, line));
    assert_eq!(add.parameters, vec!["a", "b"]);
    assert_eq!(add.code, "def add(a, b):\n    return a + b");
    let title = get_unit_by_name(&units, "Title").unwrap();
    assert_eq!(title.line, file_line(source, "# Title"));
    // No metadata at all: Python is assumed.
    assert_eq!(add.language, Language::Python);
}

/// A one-cell notebook splits exactly like the same code in a source file of
/// the kernel's language, shifted to the cell's lines.
fn assert_cell_matches_source_file(language: &str, lang: Language, file: &str, code: &[&str]) {
    let source = build_notebook(language, &[("code", code)]);
    let first_line = file_line(&source, code[0]);
    let shape = |units: Vec<CodeUnit>, offset: usize| {
        units
            .into_iter()
            .map(|u| {
                (
                    format!("{:?}", u.unit_type),
                    if u.unit_type == UnitType::RawCode {
                        String::new()
                    } else {
                        u.name
                    },
                    u.line + offset,
                    u.end_line + offset,
                    u.language,
                    u.parameters,
                    u.calls,
                    u.imports,
                    u.code,
                )
            })
            .collect::<Vec<_>>()
    };
    let mut in_notebook = shape(notebook("cell.ipynb", &source), 0);
    let mut in_file = shape(parse(&code.join("\n"), lang, file), first_line - 1);
    in_notebook.sort_by_key(|u| (u.2, u.3, u.0.clone()));
    in_file.sort_by_key(|u| (u.2, u.3, u.0.clone()));
    assert!(!in_notebook.is_empty());
    assert_eq!(in_notebook, in_file);
}

#[test]
fn test_python_cell_matches_py_file() {
    assert_cell_matches_source_file(
        "python",
        Language::Python,
        "cell.py",
        &[
            "import torch",
            "from torch import nn",
            "",
            "BATCH_SIZE = 64",
            "",
            "class MLP(nn.Module):",
            "    def __init__(self, dim):",
            "        super().__init__()",
            "        self.layer = nn.Linear(dim, 1)",
            "",
            "    def forward(self, x):",
            "        return torch.sigmoid(self.layer(x))",
            "",
            "model = MLP(8)",
        ],
    );
}

#[test]
fn test_r_notebook() {
    assert_cell_matches_source_file(
        "R",
        Language::R,
        "cell.R",
        &[
            "library(ggplot2)",
            "summarise_scores <- function(df, column) {",
            "  mean(df[[column]])",
            "}",
        ],
    );
    let source = build_notebook("R", &[("code", &["f <- function(x) {", "  x + 1", "}"])]);
    let units = notebook("analysis.ipynb", &source);
    let f = units
        .iter()
        .find(|u| u.unit_type == UnitType::Function)
        .unwrap();
    assert_eq!(f.language, Language::R);
    assert_eq!(
        (f.line, f.end_line),
        (
            file_line(&source, "f <- function"),
            file_line(&source, "x + 1") + 1
        )
    );
}

#[test]
fn test_julia_notebook() {
    assert_cell_matches_source_file(
        "julia",
        Language::Julia,
        "cell.jl",
        &[
            "using Flux",
            "function sigmoid(x)",
            "    1 / (1 + exp(-x))",
            "end",
            "loss(x, y) = Flux.mse(sigmoid.(x), y)",
        ],
    );
}

/// A kernel colgrep cannot parse still has its cells indexed, as raw code.
#[test]
fn test_unknown_kernel_language_is_raw_code() {
    let source = build_notebook(
        "matlab",
        &[(
            "code",
            &["function y = double_it(x)", "  y = 2 * x;", "end"],
        )],
    );
    let units = notebook("signal.ipynb", &source);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, UnitType::RawCode);
    assert_eq!(units[0].language, Language::Notebook);
    assert_eq!(units[0].line, file_line(&source, "function y = double_it"));
    assert!(units[0].code.contains("y = 2 * x;"));
}

#[test]
fn test_cell_magics() {
    let source = build_notebook(
        "python",
        &[
            (
                "code",
                &["%%time", "def slow():", "    return sum(range(10))"],
            ),
            ("code", &["%%bash", "build_all() {", "  make -j8", "}"]),
            ("code", &["%%html", "<div class=\"chart\"></div>"]),
            (
                "code",
                &["%%writefile helpers.py", "def helper():", "    return 1"],
            ),
        ],
    );
    let units = notebook("magics.ipynb", &source);

    let slow = get_unit_by_name(&units, "slow").unwrap();
    assert_eq!(slow.language, Language::Python);
    assert_eq!(slow.line, file_line(&source, "def slow()"));

    let build_all = get_unit_by_name(&units, "build_all").unwrap();
    assert_eq!(build_all.language, Language::Shell);
    assert_eq!(build_all.unit_type, UnitType::Function);

    let html = units
        .iter()
        .find(|u| u.code.contains("<div class"))
        .unwrap();
    assert_eq!(html.unit_type, UnitType::RawCode);
    assert_eq!(html.line, file_line(&source, "%%html"));

    assert!(get_unit_by_name(&units, "helper").is_some());

    // The magic lines themselves are covered.
    for magic in ["%%time", "%%bash", "%%html", "%%writefile"] {
        let line = file_line(&source, magic);
        assert!(
            units.iter().any(|u| u.line <= line && line <= u.end_line),
            "{magic} not covered"
        );
    }
}

/// `!cmd \` continues on the next line; the continuation is not Python.
#[test]
fn test_shell_escape_continuation() {
    let source = build_notebook(
        "python",
        &[(
            "code",
            &[
                "!accelerate launch train.py \\",
                "  --batch_size=8 \\",
                "  --lr=3e-4",
                "def after():",
                "    pass",
            ],
        )],
    );
    let units = notebook("launch.ipynb", &source);
    let after = get_unit_by_name(&units, "after").unwrap();
    assert_eq!(after.line, file_line(&source, "def after()"));
    let launch = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert_eq!(launch.line, file_line(&source, "!accelerate launch"));
    assert_eq!(launch.end_line, file_line(&source, "--lr=3e-4"));
}

#[test]
fn test_markdown_inline_images_are_stripped() {
    let source = build_notebook(
        "python",
        &[(
            "markdown",
            &["Architecture: ![net](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE=) done"],
        )],
    );
    let units = notebook("img.ipynb", &source);
    assert_eq!(units.len(), 1);
    assert_eq!(
        units[0].code,
        "Architecture: ![net](data:image/png;base64,) done"
    );
    assert!(units[0].name.starts_with("markdown_"));
}

#[test]
fn test_nbformat_v3() {
    let source = r#"{
 "metadata": {"name": "legacy"},
 "nbformat": 3,
 "nbformat_minor": 0,
 "worksheets": [
  {
   "cells": [
    {
     "cell_type": "heading",
     "level": 1,
     "metadata": {},
     "source": [
      "Legacy notebook"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "def legacy(x):\n",
      "    return x * 2"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {"output_type": "pyout", "text": ["42"]}
     ]
    }
   ],
   "metadata": {}
  }
 ]
}"#;
    let units = notebook("legacy.ipynb", source);
    let legacy = get_unit_by_name(&units, "legacy").unwrap();
    assert_eq!(legacy.language, Language::Python);
    assert_eq!(
        (legacy.line, legacy.end_line),
        (
            file_line(source, "def legacy(x)"),
            file_line(source, "return x * 2")
        )
    );
    assert!(get_unit_by_name(&units, "Legacy notebook").is_some());
    assert!(units.iter().all(|u| !u.code.contains("pyout")));
}

#[test]
fn test_invalid_or_empty_notebooks_do_not_panic() {
    for source in [
        "",
        "not json",
        "{\"cells\": [",
        "{\"cells\": 3}",
        "[1, 2, 3]",
        "{\"cells\": [{\"cell_type\": \"code\", \"source\": 7}, {\"cell_type\": 1}]}",
        "{\"cells\": [], \"metadata\": {}, \"nbformat\": 4, \"nbformat_minor\": 5}",
        "{\"cells\": [{\"cell_type\": \"code\", \"source\": [\"\", \"  \"]}]}",
        // Merge-conflict markers make the file invalid JSON.
        "{\n<<<<<<< HEAD\n \"cells\": []\n=======\n \"cells\": [1]\n>>>>>>> branch\n}",
    ] {
        assert!(notebook("broken.ipynb", source).is_empty(), "{source:?}");
    }
}

/// A notebook saved on one line (minified JSON) still yields its units.
#[test]
fn test_minified_notebook() {
    let pretty = build_notebook(
        "python",
        &[
            ("code", &["def f(x):", "    return x"]),
            ("markdown", &["# Notes"]),
        ],
    );
    let minified =
        serde_json::to_string(&serde_json::from_str::<serde_json::Value>(&pretty).unwrap())
            .unwrap();
    let units = notebook("min.ipynb", &minified);
    let f = get_unit_by_name(&units, "f").unwrap();
    assert_eq!((f.line, f.end_line), (1, 1));
    assert!(get_unit_by_name(&units, "Notes").is_some());
}

/// Malformed metadata does not lose the cells: Python is assumed.
#[test]
fn test_malformed_metadata_falls_back_to_python() {
    let source = r#"{"cells": [{"cell_type": "code", "source": ["def g():\n", "    pass"]}],
 "metadata": {"kernelspec": "oops", "language_info": 5}}"#;
    let units = notebook("meta.ipynb", source);
    assert_eq!(
        get_unit_by_name(&units, "g").unwrap().language,
        Language::Python
    );
}
