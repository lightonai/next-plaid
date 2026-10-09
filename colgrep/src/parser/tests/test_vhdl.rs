//! Tests for VHDL code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const GPIO: &str = r#"library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- General purpose I/O controller.
entity gpio is
  generic (
    WIDTH : natural := 8
  );
  port (
    clk_i  : in  std_ulogic;
    rstn_i : in  std_ulogic;
    gpio_o : out std_ulogic_vector(WIDTH-1 downto 0)
  );
end gpio;

architecture rtl of gpio is
  signal dout : std_ulogic_vector(WIDTH-1 downto 0);
  constant RESET_VALUE : natural := 0;

  -- Even parity of a vector.
  function parity(v : std_ulogic_vector) return std_ulogic is
    variable p : std_ulogic := '0';
  begin
    for i in v'range loop
      p := p xor v(i);
    end loop;
    return p;
  end function parity;
begin
  bus_access: process(rstn_i, clk_i)
  begin
    if rstn_i = '0' then
      dout <= (others => '0');
    elsif rising_edge(clk_i) then
      dout <= std_ulogic_vector(to_unsigned(RESET_VALUE, WIDTH));
      report_status(dout(0), parity(dout));
    end if;
  end process bus_access;

  u_sync: entity work.synchronizer
    port map (clk_i => clk_i, d_i => dout(0));

  gpio_o <= dout;
end architecture rtl;
"#;

#[test]
fn test_function_embedding_text() {
    let units = assert_extractor_invariants(GPIO, Language::Vhdl, "rtl/gpio.vhd");

    let unit = get_unit_by_name(&units, "parity").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Method: parity
Signature: function parity(v : std_ulogic_vector) return std_ulogic is
Class: rtl of gpio
Description: Even parity of a vector.
Parameters: v
Returns: std_ulogic
Variables: p
File: rtl gpio gpio.vhd
Code:
  -- Even parity of a vector.
  function parity(v : std_ulogic_vector) return std_ulogic is
    variable p : std_ulogic := '0';
  begin
    for i in v'range loop
      p := p xor v(i);
    end loop;
    return p;
  end function parity;"#;
    assert_eq!(text, expected);
}

#[test]
fn test_entity_and_architecture() {
    let units = parse(GPIO, Language::Vhdl, "gpio.vhd");

    let entity = get_unit_by_name(&units, "gpio").unwrap();
    assert_eq!(entity.unit_type, UnitType::Class);
    assert_eq!((entity.line, entity.end_line), (5, 15));
    assert_eq!(
        entity.docstring.as_deref(),
        Some("General purpose I/O controller.")
    );
    // Ports, not generics, are the entity's parameters.
    assert_eq!(entity.parameters, vec!["clk_i", "rstn_i", "gpio_o"]);
    // A design unit sees the packages the file uses.
    assert_eq!(entity.imports, vec!["numeric_std", "std_logic_1164"]);

    // Architectures are qualified with their entity.
    let arch = get_unit_by_name(&units, "rtl of gpio").unwrap();
    assert_eq!(arch.unit_type, UnitType::Class);
    assert_eq!((arch.line, arch.end_line), (17, 45));
    assert_eq!(arch.variables, vec!["RESET_VALUE", "dout"]);
    // Entity instantiation is a call to the entity.
    assert!(arch.calls.contains(&"synchronizer".to_string()));
}

#[test]
fn test_process_calls_skip_indexing() {
    let units = parse(GPIO, Language::Vhdl, "gpio.vhd");

    let process = get_unit_by_name(&units, "bus_access").unwrap();
    assert_eq!(process.unit_type, UnitType::Method);
    assert_eq!(process.parent_class.as_deref(), Some("rtl of gpio"));
    assert_eq!((process.line, process.end_line), (31, 39));
    assert!(process.has_branches);
    // `dout(0)` indexes a signal; `std_ulogic_vector(...)` converts a type.
    assert_eq!(
        process.calls,
        vec!["parity", "report_status", "rising_edge", "to_unsigned"]
    );
}

#[test]
fn test_unlabelled_process_named_by_driven_signal() {
    let source = r#"architecture rtl of reg is
begin
  process(clk)
  begin
    if rising_edge(clk) then
      q <= d;
    end if;
  end process;
end architecture;
"#;
    let units = assert_extractor_invariants(source, Language::Vhdl, "reg.vhd");
    let process = get_unit_by_name(&units, "process q").unwrap();
    assert_eq!((process.line, process.end_line), (3, 8));
}

#[test]
fn test_package_and_body() {
    let source = r#"library ieee;
use ieee.std_logic_1164.all;

package util_pkg is
  constant XLEN : natural := 32;
  type state_t is (IDLE, BUSY);
  function to_int(x : std_ulogic) return integer;
  procedure pulse(signal s : out std_ulogic; constant n : in natural);
end package util_pkg;

package body util_pkg is
  function to_int(x : std_ulogic) return integer is
  begin
    if x = '1' then
      return 1;
    end if;
    return 0;
  end function;

  procedure pulse(signal s : out std_ulogic; constant n : in natural) is
  begin
    s <= '1';
  end procedure pulse;
end package body util_pkg;
"#;
    let units = assert_extractor_invariants(source, Language::Vhdl, "util_pkg.vhd");

    let pkg = get_unit_by_name(&units, "util_pkg").unwrap();
    assert_eq!(pkg.unit_type, UnitType::Class);
    assert_eq!((pkg.line, pkg.end_line), (4, 9));
    assert_eq!(pkg.variables, vec!["XLEN"]);

    let body = get_unit_by_name(&units, "util_pkg body").unwrap();
    assert_eq!((body.line, body.end_line), (11, 24));

    let to_int = get_unit_by_name(&units, "to_int").unwrap();
    assert_eq!(to_int.unit_type, UnitType::Method);
    assert_eq!(to_int.parent_class.as_deref(), Some("util_pkg body"));
    assert_eq!(to_int.parameters, vec!["x"]);
    assert_eq!(to_int.return_type.as_deref(), Some("integer"));

    let pulse = get_unit_by_name(&units, "pulse").unwrap();
    assert_eq!(pulse.parameters, vec!["s", "n"]);
    assert_eq!(pulse.return_type, None);
}

/// VUnit-style protected types are class-like: the body's methods are units.
#[test]
fn test_protected_type() {
    let source = r#"package log_pkg is
  type logger_t is protected
    procedure log(msg : string);
    impure function count return natural;
  end protected;
end package;

package body log_pkg is
  type logger_t is protected body
    variable n : natural := 0;

    procedure log(msg : string) is
    begin
      n := n + 1;
    end procedure;

    impure function count return natural is
    begin
      return n;
    end function;
  end protected body;
end package body;
"#;
    let units = assert_extractor_invariants(source, Language::Vhdl, "log_pkg.vhd");
    let body = get_unit_by_name(&units, "logger_t body").unwrap();
    assert_eq!(body.unit_type, UnitType::Class);
    let log = get_unit_by_name(&units, "log").unwrap();
    assert_eq!(log.parent_class.as_deref(), Some("logger_t body"));
    assert_eq!(log.parameters, vec!["msg"]);
    let count = get_unit_by_name(&units, "count").unwrap();
    assert_eq!(count.return_type.as_deref(), Some("natural"));
}

#[test]
fn test_vunit_testbench() {
    let source = r#"library vunit_lib;
context vunit_lib.vunit_context;

entity tb_fifo is
  generic (runner_cfg : string);
end entity;

architecture tb of tb_fifo is
begin
  main : process
  begin
    test_runner_setup(runner, runner_cfg);
    while test_suite loop
      if run("test_push_pop") then
        check_equal(pop, 1);
      end if;
    end loop;
    test_runner_cleanup(runner);
  end process;
end architecture;
"#;
    let units = assert_extractor_invariants(source, Language::Vhdl, "tb_fifo.vhd");
    let main = get_unit_by_name(&units, "main").unwrap();
    assert_eq!(
        main.calls,
        vec![
            "check_equal",
            "run",
            "test_runner_cleanup",
            "test_runner_setup"
        ]
    );
    let entity = get_unit_by_name(&units, "tb_fifo").unwrap();
    assert_eq!(entity.imports, vec!["vunit_context"]);
    // A generic-only entity has no ports.
    assert!(entity.parameters.is_empty());
}
