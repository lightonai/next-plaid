//! Tests for Verilog / SystemVerilog code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const ALU: &str = r#"`include "prim_assert.sv"
`define ALU_WIDTH 32

// Arithmetic logic unit.
// Adds and shifts.
module ibex_alu #(
  parameter int unsigned Width = 32
) (
  input  logic              clk_i,
  input  ibex_pkg::alu_op_e operator_i,
  output logic [31:0]       result_o
);
  import ibex_pkg::*;
  logic [31:0] acc_q;

  always_ff @(posedge clk_i) acc_q <= result_o;

  always_comb begin : gen_result
    result_o = acc_q;
    if (operator_i == ALU_ADD) begin
      result_o = acc_q + 1;
    end
  end

  prim_fifo #(.Width(Width)) u_fifo (
    .clk_i (clk_i),
    .data_o(acc_q)
  );

  function automatic logic [31:0] reverse(input logic [31:0] x, input int n);
    logic [31:0] r;
    for (int i = 0; i < 32; i++) r[i] = x[31-i];
    return r;
  endfunction
endmodule
"#;

#[test]
fn test_module_embedding_text() {
    let units = assert_extractor_invariants(ALU, Language::Verilog, "rtl/ibex_alu.sv");

    let unit = get_unit_by_name(&units, "reverse").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Method: reverse
Signature: function automatic logic [31:0] reverse(input logic [31:0] x, input int n);
Class: ibex_alu
Parameters: x, n
Returns: logic [31:0]
Variables: r
File: rtl ibex alu ibex_alu.sv
Code:
  function automatic logic [31:0] reverse(input logic [31:0] x, input int n);
    logic [31:0] r;
    for (int i = 0; i < 32; i++) r[i] = x[31-i];
    return r;
  endfunction"#;
    assert_eq!(text, expected);
}

#[test]
fn test_module_ports_instances_and_imports() {
    let units = parse(ALU, Language::Verilog, "ibex_alu.sv");

    let alu = get_unit_by_name(&units, "ibex_alu").unwrap();
    assert_eq!(alu.unit_type, UnitType::Class);
    // The comment right above the module is its documentation and part of it.
    assert_eq!((alu.line, alu.end_line), (4, 35));
    assert_eq!(
        alu.docstring.as_deref(),
        Some("Arithmetic logic unit. Adds and shifts.")
    );
    assert_eq!(alu.signature, "module ibex_alu #(");
    // Ports are the module's parameters; generics stay in the code.
    assert_eq!(alu.parameters, vec!["clk_i", "operator_i", "result_o"]);
    // Instantiated submodules are calls.
    assert!(
        alu.calls.contains(&"prim_fifo".to_string()),
        "{:?}",
        alu.calls
    );
    assert!(alu.variables.contains(&"acc_q".to_string()));
    // `ibex_pkg::alu_op_e` and `import ibex_pkg::*` use the package.
    assert_eq!(alu.imports, vec!["ibex_pkg"]);

    let macro_unit = get_unit_by_name(&units, "ALU_WIDTH").unwrap();
    assert_eq!(macro_unit.unit_type, UnitType::Constant);
    assert_eq!((macro_unit.line, macro_unit.end_line), (2, 2));
}

/// A labelled multi-line always block is a unit; a one-line flop stays in
/// the module.
#[test]
fn test_procedural_blocks() {
    let units = parse(ALU, Language::Verilog, "ibex_alu.sv");

    let comb = get_unit_by_name(&units, "gen_result").unwrap();
    assert_eq!(comb.unit_type, UnitType::Method);
    assert_eq!(comb.parent_class.as_deref(), Some("ibex_alu"));
    assert_eq!((comb.line, comb.end_line), (18, 23));
    assert!(comb.has_branches);

    assert!(!units.iter().any(|u| u.name.starts_with("always_ff")));
}

#[test]
fn test_unlabelled_block_named_by_driven_signal() {
    let source = r#"module counter (input clk, input rst, output reg [7:0] count);
  always @(posedge clk) begin
    if (rst)
      count <= 0;
    else
      count <= count + 1;
  end

  initial begin
    $display("start");
    run_test("smoke");
  end
endmodule
"#;
    let units = assert_extractor_invariants(source, Language::Verilog, "counter.v");
    let always = get_unit_by_name(&units, "always count").unwrap();
    assert_eq!((always.line, always.end_line), (2, 7));
    let initial = get_unit_by_name(&units, "initial").unwrap();
    // System tasks are not calls; user tasks are.
    assert_eq!(initial.calls, vec!["run_test"]);
}

#[test]
fn test_verilog_1995_module_and_function() {
    let source = r#"module adder(a, b, sum);
  input [7:0] a, b;
  output [8:0] sum;

  function [8:0] add;
    input [7:0] x, y;
    begin
      add = x + y;
    end
  endfunction

  assign sum = add(a, b);
endmodule
"#;
    let units = assert_extractor_invariants(source, Language::Verilog, "adder.v");
    let adder = get_unit_by_name(&units, "adder").unwrap();
    assert_eq!(adder.parameters, vec!["a", "b", "sum"]);
    assert!(adder.calls.contains(&"add".to_string()));

    let add = get_unit_by_name(&units, "add").unwrap();
    assert_eq!(add.parameters, vec!["x", "y"]);
    assert_eq!(add.return_type.as_deref(), Some("[8:0]"));
}

#[test]
fn test_uvm_class() {
    let source = r#"import uvm_pkg::*;

// Drives sequence items onto the bus.
class bus_driver extends uvm_driver #(bus_item);
  `uvm_component_utils(bus_driver)
  int unsigned sent;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual task run_phase(uvm_phase phase);
    seq_item_port.get_next_item(req);
    drive_item(req);
    seq_item_port.item_done();
  endtask

  extern function void report_phase(uvm_phase phase);
endclass

function void bus_driver::report_phase(uvm_phase phase);
  `uvm_info("DRV", $sformatf("sent %0d", sent), UVM_LOW)
endfunction
"#;
    let units = assert_extractor_invariants(source, Language::Verilog, "bus_driver.sv");

    let class = get_unit_by_name(&units, "bus_driver").unwrap();
    assert_eq!(class.unit_type, UnitType::Class);
    assert_eq!(class.extends.as_deref(), Some("uvm_driver"));
    assert_eq!(
        class.docstring.as_deref(),
        Some("Drives sequence items onto the bus.")
    );
    assert!(class.variables.contains(&"sent".to_string()));

    let ctor = get_unit_by_name(&units, "new").unwrap();
    assert_eq!(ctor.parent_class.as_deref(), Some("bus_driver"));
    assert_eq!(ctor.parameters, vec!["name", "parent"]);

    let run = get_unit_by_name(&units, "run_phase").unwrap();
    assert_eq!(run.unit_type, UnitType::Method);
    assert_eq!(run.calls, vec!["drive_item", "get_next_item", "item_done"]);

    // Defined outside the class body, still a method of the class.
    let report = get_unit_by_name(&units, "report_phase").unwrap();
    assert_eq!(report.unit_type, UnitType::Method);
    assert_eq!(report.parent_class.as_deref(), Some("bus_driver"));
    assert_eq!((report.line, report.end_line), (21, 23));
    assert!(report.calls.contains(&"uvm_info".to_string()));
}

#[test]
fn test_package_and_interface() {
    let source = r#"package ibex_pkg;
  typedef enum logic [1:0] { ALU_ADD, ALU_SUB } alu_op_e;
  parameter int XLEN = 32;

  function automatic int clog2(int value);
    return $clog2(value);
  endfunction
endpackage

interface bus_if #(parameter int W = 8) (input logic clk);
  logic [W-1:0] data;
  logic valid;
  modport master (output data, output valid);
endinterface
"#;
    let units = assert_extractor_invariants(source, Language::Verilog, "pkg.sv");

    let pkg = get_unit_by_name(&units, "ibex_pkg").unwrap();
    assert_eq!(pkg.unit_type, UnitType::Class);
    assert_eq!((pkg.line, pkg.end_line), (1, 8));
    let clog2 = get_unit_by_name(&units, "clog2").unwrap();
    assert_eq!(clog2.parent_class.as_deref(), Some("ibex_pkg"));
    assert_eq!(clog2.return_type.as_deref(), Some("int"));
    // Package members are folded into the package, not top-level constants.
    assert!(get_unit_by_name(&units, "XLEN").is_none());

    let bus = get_unit_by_name(&units, "bus_if").unwrap();
    assert_eq!(bus.unit_type, UnitType::Class);
    assert_eq!(bus.parameters, vec!["clk"]);
    assert_eq!(bus.variables, vec!["data", "valid"]);
}

#[test]
fn test_header_file_constants() {
    let source = r#"`ifndef DEFS_SVH
`define DEFS_SVH
`define OPCODE_LOAD 7'h03
typedef logic [31:0] word_t;
localparam int unsigned NumRegs = 32;
`endif
"#;
    let units = assert_extractor_invariants(source, Language::Verilog, "defs.svh");
    for name in ["DEFS_SVH", "OPCODE_LOAD", "word_t", "NumRegs"] {
        let unit = get_unit_by_name(&units, name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(unit.unit_type, UnitType::Constant, "{name}");
        assert_eq!(unit.line, unit.end_line, "{name}");
    }
}

#[test]
fn test_includes_are_file_imports() {
    let source = r#"`include "uvm_macros.svh"
`include "dv/common_ifs.sv"
module top;
  import uvm_pkg::*;
  import dv_utils_pkg::*;
endmodule
"#;
    let units = parse(source, Language::Verilog, "top.sv");
    let raw = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert_eq!(
        raw.imports,
        vec!["common_ifs", "dv_utils_pkg", "uvm_macros", "uvm_pkg"]
    );
}

/// The license banner above the first module is separated by a blank line
/// and is neither its documentation nor part of it.
#[test]
fn test_license_banner_is_not_documentation() {
    let source = r#"// Copyright lowRISC contributors.
// SPDX-License-Identifier: Apache-2.0

/**
 * Register file
 */
module regfile (input logic clk_i);
endmodule
"#;
    let units = assert_extractor_invariants(source, Language::Verilog, "regfile.sv");
    let module = get_unit_by_name(&units, "regfile").unwrap();
    assert_eq!(module.docstring.as_deref(), Some("Register file"));
    assert_eq!((module.line, module.end_line), (4, 8));
}

/// Coq/Rocq also uses `.v`; a Coq proof is indexed as text, not as broken
/// Verilog.
#[test]
fn test_coq_source_in_dot_v_is_text() {
    let coq = r#"Require Import Arith.

Lemma add_comm : forall n m : nat, n + m = m + n.
Proof.
  intros. lia.
Qed.
"#;
    let units = parse(coq, Language::Verilog, "Add.v");
    assert!(!units.is_empty());
    assert!(units.iter().all(|u| u.language == Language::Text));

    let verilog = "module m;\nendmodule\n";
    let units = parse(verilog, Language::Verilog, "m.v");
    assert!(units.iter().all(|u| u.language == Language::Verilog));
    assert!(get_unit_by_name(&units, "m").is_some());
}
