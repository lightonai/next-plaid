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
