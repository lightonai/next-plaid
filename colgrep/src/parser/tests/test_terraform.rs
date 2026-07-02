//! Tests for Terraform / HCL code extraction.

use super::common::*;
use crate::parser::Language;

#[test]
fn test_resource_block() {
    let source = r#"resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "HelloWorld"
  }
}
"#;
    let units = parse(source, Language::Terraform, "main.tf");
    let unit = get_unit_by_name(&units, r#"resource "aws_instance" "web""#)
        .expect("resource block named by type + labels");
    assert_eq!(unit.language, Language::Terraform);
    // The whole block, including nested attributes, is folded into one unit.
    assert!(
        unit.code.contains("instance_type") && unit.code.contains("HelloWorld"),
        "resource code should include the whole block body: {:?}",
        unit.code
    );
}

#[test]
fn test_variable_and_output_blocks() {
    let source = r#"variable "region" {
  type    = string
  default = "us-east-1"
}

output "instance_ip" {
  value = aws_instance.web.private_ip
}
"#;
    let units = parse(source, Language::Terraform, "variables.tf");
    let var = get_unit_by_name(&units, r#"variable "region""#).expect("variable block");
    assert!(var.code.contains("us-east-1"), "code={:?}", var.code);
    let out = get_unit_by_name(&units, r#"output "instance_ip""#).expect("output block");
    assert!(out.code.contains("private_ip"), "code={:?}", out.code);
}

#[test]
fn test_module_block() {
    let source = r#"module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  cidr    = "10.0.0.0/16"
}
"#;
    let units = parse(source, Language::Terraform, "main.tf");
    let m = get_unit_by_name(&units, r#"module "vpc""#).expect("module block");
    assert!(
        m.code.contains("terraform-aws-modules/vpc/aws"),
        "code={:?}",
        m.code
    );
}

#[test]
fn test_provider_and_terraform_blocks() {
    let source = r#"terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}
"#;
    let units = parse(source, Language::Terraform, "providers.tf");
    // A label-less block is named by its type alone.
    let tf = get_unit_by_name(&units, "terraform").expect("terraform block");
    // Nested `required_providers` block is folded into the parent terraform
    // block because we don't recurse into HCL block bodies.
    assert!(
        tf.code.contains("required_providers") && tf.code.contains("hashicorp/aws"),
        "terraform block should fold in nested blocks: {:?}",
        tf.code
    );
    let provider = get_unit_by_name(&units, r#"provider "aws""#).expect("provider block");
    assert!(
        provider.code.contains("us-west-2"),
        "code={:?}",
        provider.code
    );
}

#[test]
fn test_data_and_locals_blocks() {
    let source = r#"data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
}

locals {
  common_tags = {
    Environment = "prod"
  }
}
"#;
    let units = parse(source, Language::Terraform, "main.tf");
    let data = get_unit_by_name(&units, r#"data "aws_ami" "ubuntu""#).expect("data block");
    assert!(data.code.contains("most_recent"), "code={:?}", data.code);
    let locals = get_unit_by_name(&units, "locals").expect("locals block");
    assert!(
        locals.code.contains("common_tags"),
        "code={:?}",
        locals.code
    );
}

#[test]
fn test_multiple_blocks_each_indexed() {
    let source = r#"resource "aws_s3_bucket" "a" {
  bucket = "bucket-a"
}

resource "aws_s3_bucket" "b" {
  bucket = "bucket-b"
}
"#;
    let units = parse(source, Language::Terraform, "main.tf");
    let names: Vec<&str> = units.iter().map(|u| u.name.as_str()).collect();
    assert!(
        names.contains(&r#"resource "aws_s3_bucket" "a""#),
        "expected bucket a in {:?}",
        names
    );
    assert!(
        names.contains(&r#"resource "aws_s3_bucket" "b""#),
        "expected bucket b in {:?}",
        names
    );
}

#[test]
fn test_empty_file_doesnt_panic() {
    let units = parse("", Language::Terraform, "empty.tf");
    assert!(units.is_empty());
}

#[test]
fn test_invalid_hcl_doesnt_panic() {
    // tree-sitter-hcl is lenient; malformed input must still return without
    // panicking. The unit set may be empty or partial.
    let _ = parse(
        "this is not valid hcl {{{ === ",
        Language::Terraform,
        "broken.tf",
    );
}

// ---------------------------------------------------------------------------
// Stress / robustness tests
//
// These push adversarial inputs through the extractor (via the shared
// assert_extractor_invariants harness in common.rs) to guard against panics,
// out-of-bounds line ranges, and coverage holes. The invariants asserted here
// (line bounds, non-empty block names, 100%-non-empty-line coverage) must hold
// for *any* input, so they double as a lightweight fuzz harness.
// ---------------------------------------------------------------------------

#[test]
fn stress_huge_file_2000_blocks() {
    // 2000 sibling blocks: each must become its own uniquely-named unit and the
    // whole file must stay fully covered.
    let mut source = String::new();
    for i in 0..2000 {
        source.push_str(&format!(
            "resource \"aws_s3_bucket\" \"bucket_{i}\" {{\n  bucket = \"b-{i}\"\n  acl    = \"private\"\n}}\n\n"
        ));
    }
    let units = assert_extractor_invariants(&source, Language::Terraform, "huge.tf");
    let block_names: std::collections::HashSet<&str> = units
        .iter()
        .filter(|u| matches!(u.unit_type, crate::parser::UnitType::Class))
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(
        block_names.len(),
        2000,
        "expected 2000 distinct block units, got {}",
        block_names.len()
    );
    assert!(block_names.contains(r#"resource "aws_s3_bucket" "bucket_0""#));
    assert!(block_names.contains(r#"resource "aws_s3_bucket" "bucket_1999""#));
}

#[test]
fn stress_deep_nesting_no_overflow() {
    // 200 levels of nested blocks. Because HCL blocks don't recurse into their
    // bodies, only the outermost block is a unit; the analysis layers still walk
    // the full depth but are bounded by the recursion-depth guard. Must return
    // (empty or a single folded unit) without panicking or overflowing.
    let depth = 200;
    let mut source = String::from("resource \"aws_deep\" \"x\" {\n");
    for i in 0..depth {
        source.push_str(&"  ".repeat(i + 1));
        source.push_str(&format!("dynamic \"level_{i}\" {{\n"));
    }
    for i in (1..=depth).rev() {
        source.push_str(&"  ".repeat(i));
        source.push_str("}\n");
    }
    source.push_str("}\n");
    // Just needs to not panic; invariants hold on whatever is returned.
    let _ = assert_extractor_invariants(&source, Language::Terraform, "deep.tf");
}

#[test]
fn stress_heredoc_does_not_spawn_fake_blocks() {
    // A heredoc string that contains block-looking text must NOT be parsed as
    // real blocks — it belongs to the enclosing `locals` block's body.
    let source = r#"locals {
  policy = <<-EOT
    {
      "Statement": [{ "Effect": "Allow" }]
    }
    resource "not_real" "should_not_parse" { foo = "bar" }
  EOT
  script = <<SCRIPT
for i in {1..10}; do echo "line $i {nested}"; done
SCRIPT
}
"#;
    let units = assert_extractor_invariants(source, Language::Terraform, "heredocs.tf");
    assert!(
        !units.iter().any(|u| u.name.contains("not_real")),
        "heredoc content must not become a block unit: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
    let locals = get_unit_by_name(&units, "locals").expect("locals block");
    assert!(
        locals.code.contains("not_real") && locals.code.contains("SCRIPT"),
        "heredoc text should live inside the locals block code"
    );
}

#[test]
fn stress_comments_ternary_and_for_expressions() {
    let source = r#"# hash comment
// slash comment
/* block
   comment */
variable "names" {
  type    = list(string)
  default = ["a", "b", "c"]  # inline comment
}

resource "aws_instance" "web" {
  count         = length(var.names)  // count
  instance_type = var.env == "prod" ? "t3.large" : "t3.micro"
  tags          = { for k, v in var.tags : k => v if v != "" }
}
"#;
    let units = assert_extractor_invariants(source, Language::Terraform, "exprs.tf");
    assert!(get_unit_by_name(&units, r#"variable "names""#).is_some());
    let web = get_unit_by_name(&units, r#"resource "aws_instance" "web""#).expect("web");
    assert!(web.code.contains("t3.large"), "ternary preserved in code");
}

#[test]
fn stress_unicode_identifiers_and_values() {
    let source = "variable \"région\" {\n  description = \"Déploiement — région 🌍\"\n  default     = \"eu-ouest-1\"\n}\n\nresource \"aws_instance\" \"café_serveur\" {\n  nom = \"société-café-☕\"\n}\n";
    let units = assert_extractor_invariants(source, Language::Terraform, "unicode.tf");
    assert!(
        get_unit_by_name(&units, r#"variable "région""#).is_some(),
        "unicode label captured in name: {:?}",
        units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>()
    );
    assert!(get_unit_by_name(&units, r#"resource "aws_instance" "café_serveur""#).is_some());
}

#[test]
fn stress_tfvars_pure_attributes_are_covered() {
    // A .tfvars file has no blocks — every line should still be covered as
    // RawCode, with no panic.
    let source = "region        = \"us-east-1\"\ninstance_type = \"t3.medium\"\ntags          = { Team = \"infra\" }\n";
    let units = assert_extractor_invariants(source, Language::Terraform, "prod.tfvars");
    assert!(!units.is_empty(), "tfvars attributes should yield raw code");
    assert!(units
        .iter()
        .all(|u| matches!(u.unit_type, crate::parser::UnitType::RawCode)));
}

#[test]
fn stress_malformed_unclosed_block() {
    // Unterminated block with a dangling nested block — must not panic.
    let source = "resource \"aws_thing\" \"unclosed\" {\n  name = \"x\"\n  nested {\n    x = 1\n";
    let _ = assert_extractor_invariants(source, Language::Terraform, "malformed.tf");
}

#[test]
fn stress_labels_labelless_and_identifier_labels() {
    let source = r#"terraform {}
locals {}
provider "aws" {}
variable "one" {}
data "aws_ami" "two" {}
weird "a" "b" "c" "d" { note = "repeated labels" }
identifier_label some_ident { key = "value" }
"#;
    let units = assert_extractor_invariants(source, Language::Terraform, "labels.tf");
    let names: Vec<&str> = units.iter().map(|u| u.name.as_str()).collect();
    for expected in [
        "terraform",
        "locals",
        r#"provider "aws""#,
        r#"variable "one""#,
        r#"data "aws_ami" "two""#,
    ] {
        assert!(
            names.contains(&expected),
            "missing {expected:?} in {names:?}"
        );
    }
    // Every label (string or identifier) before the brace is part of the name.
    assert!(
        names.contains(&r#"weird "a" "b" "c" "d""#),
        "repeated labels not all captured: {names:?}"
    );
    assert!(
        names.contains(&"identifier_label some_ident"),
        "identifier label not captured: {names:?}"
    );
}
