//! Tests for Bash / shell script code extraction.

use super::common::*;
use crate::parser::{Language, UnitType};

#[test]
fn test_function_keyword_form() {
    let source = r#"#!/bin/bash
set -euo pipefail

function deploy_app() {
    local env="$1"
    echo "deploying to ${env}"
}
"#;
    let units = assert_extractor_invariants(source, Language::Shell, "deploy.sh");
    let f = get_unit_by_name(&units, "deploy_app").expect("function unit");
    assert_eq!(f.unit_type, UnitType::Function);
    assert!(
        f.code.contains("local env") && f.code.contains("deploying to"),
        "function body captured: {:?}",
        f.code
    );
}

#[test]
fn test_posix_function_form() {
    // The `name() { ... }` form (no `function` keyword) must also chunk.
    let source = r#"rollback() {
    echo "rolling back"
}
"#;
    let units = assert_extractor_invariants(source, Language::Shell, "ops.sh");
    let f = get_unit_by_name(&units, "rollback").expect("posix-form function");
    assert_eq!(f.unit_type, UnitType::Function);
}

#[test]
fn test_doc_comment_attached_but_not_shebang() {
    let source = r#"#!/bin/bash

# Restarts the service gracefully.
# Retries up to 3 times.
restart_service() {
    systemctl restart "$1"
}
"#;
    let units = assert_extractor_invariants(source, Language::Shell, "svc.sh");
    let f = get_unit_by_name(&units, "restart_service").expect("function");
    assert!(
        f.code.contains("Restarts the service gracefully"),
        "leading # doc comment should be part of the unit: {:?}",
        f.code
    );
    assert!(
        !f.code.contains("#!/bin/bash"),
        "shebang must not be swallowed into the function unit: {:?}",
        f.code
    );
}

#[test]
fn test_top_level_commands_covered_as_raw_code() {
    // Scripts without functions (most cron/CI glue) must still be fully
    // indexed via RawCode gap-fill.
    let source = r#"#!/bin/sh
export PATH=/usr/local/bin:$PATH
curl -fsSL https://example.com/install.sh | sh
echo "done"
"#;
    let units = assert_extractor_invariants(source, Language::Shell, "install.sh");
    assert!(!units.is_empty());
    assert!(units
        .iter()
        .all(|u| matches!(u.unit_type, UnitType::RawCode)));
}

#[test]
fn test_mixed_functions_and_commands() {
    let source = r#"#!/bin/bash
VERSION="2.1"

build() {
    make -j"$(nproc)"
}

test_all() {
    make test
}

build
test_all
"#;
    let units = assert_extractor_invariants(source, Language::Shell, "ci.sh");
    assert!(get_unit_by_name(&units, "build").is_some());
    assert!(get_unit_by_name(&units, "test_all").is_some());
    // VERSION assignment and trailing invocations live in RawCode gaps.
    assert!(units
        .iter()
        .any(|u| matches!(u.unit_type, UnitType::RawCode) && u.code.contains("VERSION")));
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Shell, "empty.sh");
    assert!(units.is_empty());
}

#[test]
fn test_malformed_shell_doesnt_panic() {
    let _ = assert_extractor_invariants(
        "if [ -z \"$1\" ; then\n  echo unbalanced\nfi\ncase esac done {{{",
        Language::Shell,
        "broken.sh",
    );
}

// --- Stress / robustness (shared invariant harness in common.rs) ---

#[test]
fn stress_huge_file_1000_functions() {
    let mut source = String::from("#!/bin/bash\n\n");
    for i in 0..1000 {
        source.push_str(&format!("task_{i}() {{\n    echo \"step {i}\"\n}}\n\n"));
    }
    let units = assert_extractor_invariants(&source, Language::Shell, "huge.sh");
    let names: std::collections::HashSet<&str> = units
        .iter()
        .filter(|u| matches!(u.unit_type, UnitType::Function))
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(names.len(), 1000, "expected 1000 distinct functions");
    assert!(names.contains("task_0") && names.contains("task_999"));
}

#[test]
fn stress_heredoc_does_not_spawn_fake_functions() {
    // Function-looking text inside a heredoc is string content, not code.
    let source = r#"generate_script() {
    cat > /tmp/out.sh <<'EOF'
fake_function() {
    echo "I am data, not code"
}
EOF
}
"#;
    let units = assert_extractor_invariants(source, Language::Shell, "heredoc.sh");
    assert!(get_unit_by_name(&units, "generate_script").is_some());
    assert!(
        get_unit_by_name(&units, "fake_function").is_none(),
        "heredoc content must not become a function unit: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn stress_nested_functions_and_subshells() {
    // Bash allows nested function definitions; both levels become units.
    let source = r#"outer() {
    inner() {
        echo "nested"
    }
    result=$(echo "$(date "+%Y")" | awk '{ print $1 }')
    inner
}
"#;
    let units = assert_extractor_invariants(source, Language::Shell, "nested.sh");
    assert!(get_unit_by_name(&units, "outer").is_some());
    assert!(get_unit_by_name(&units, "inner").is_some());
}

#[test]
fn stress_deep_nesting_no_panic() {
    // 100 nested if-blocks: full AST walk stays below the recursion guard.
    let depth = 100;
    let mut source = String::new();
    for _ in 0..depth {
        source.push_str("if true; then\n");
    }
    source.push_str("  echo deep\n");
    for _ in 0..depth {
        source.push_str("fi\n");
    }
    let _ = assert_extractor_invariants(&source, Language::Shell, "deep.sh");
}

#[test]
fn stress_unicode_names_and_content() {
    let source = "# Déploiement complet ☕\ndéployer_café() {\n    echo \"déployé 🌍\"\n}\n";
    let units = assert_extractor_invariants(source, Language::Shell, "unicode.sh");
    assert!(get_unit_by_name(&units, "déployer_café").is_some());
}
