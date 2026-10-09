//! Tests for Solidity code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const TOKEN: &str = r#"// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "./IERC20.sol";
import {Context} from "../utils/Context.sol";
import "./SafeMath.sol";

/// @title A minimal token
/// @notice Tracks balances and moves them.
contract Token is Context, IERC20 {
    mapping(address => uint256) private _balances;
    address public owner;

    event Transfer(address indexed from, address indexed to, uint256 value);
    error Unauthorized(address caller);

    modifier onlyOwner() {
        if (_msgSender() != owner) revert Unauthorized(_msgSender());
        _;
    }

    constructor(address owner_) {
        owner = owner_;
    }

    /**
     * @dev Moves `value` tokens from the caller to `to`.
     */
    function transfer(address to, uint256 value) public virtual onlyOwner returns (bool) {
        address from = _msgSender();
        uint256 fromBalance = _balances[from];
        require(fromBalance >= value, "balance");
        _balances[from] = fromBalance - value;
        _balances[to] += value;
        emit Transfer(from, to, value);
        return true;
    }

    receive() external payable {}
}
"#;

#[test]
fn test_function_embedding_text() {
    let units = parse(TOKEN, Language::Solidity, "Token.sol");
    let unit = get_unit_by_name(&units, "transfer").unwrap();
    let expected = r#"Method: transfer
Signature: function transfer(address to, uint256 value) public virtual onlyOwner returns (bool) {
Class: Token
Description: Moves `value` tokens from the caller to `to`.
Parameters: to, value
Returns: bool
Calls: Transfer, _msgSender, onlyOwner, require
Variables: from, fromBalance
File: token Token.sol
Code:
    /**
     * @dev Moves `value` tokens from the caller to `to`.
     */
    function transfer(address to, uint256 value) public virtual onlyOwner returns (bool) {
        address from = _msgSender();
        uint256 fromBalance = _balances[from];
        require(fromBalance >= value, "balance");
        _balances[from] = fromBalance - value;
        _balances[to] += value;
        emit Transfer(from, to, value);
        return true;
    }"#;
    assert_eq!(build_embedding_text(unit), expected);
}

#[test]
fn test_contract_inheritance_and_natspec() {
    let units = parse(TOKEN, Language::Solidity, "Token.sol");
    let token = get_unit_by_name(&units, "Token").unwrap();
    assert_eq!(token.unit_type, UnitType::Class);
    assert_eq!(token.extends.as_deref(), Some("Context, IERC20"));
    assert_eq!(
        token.docstring.as_deref(),
        Some("A minimal token Tracks balances and moves them.")
    );
    assert_eq!((token.line, token.end_line), (8, 40));
}

#[test]
fn test_members() {
    let units = assert_extractor_invariants(TOKEN, Language::Solidity, "Token.sol");
    let members: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Method)
        .map(|u| u.name.as_str())
        .collect();
    assert_eq!(
        members,
        vec![
            "Transfer",
            "Unauthorized",
            "onlyOwner",
            "constructor",
            "transfer",
            "receive"
        ]
    );
    let event = get_unit_by_name(&units, "Transfer").unwrap();
    assert_eq!(event.parameters, vec!["from", "to", "value"]);
    let modifier = get_unit_by_name(&units, "onlyOwner").unwrap();
    assert!(modifier.calls.contains(&"Unauthorized".to_string()));
    assert_eq!(
        get_unit_by_name(&units, "constructor").unwrap().parameters,
        vec!["owner_"]
    );
}

#[test]
fn test_imports() {
    let units = parse(TOKEN, Language::Solidity, "Token.sol");
    let raw = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert_eq!(raw.imports, vec!["Context", "IERC20", "SafeMath"]);
}

#[test]
fn test_interface_library_and_free_definitions() {
    let source = r#"pragma solidity ^0.8.0;

uint256 constant MAX_FEE = 1_000_000;
type Currency is address;
error InvalidFee(uint24 fee);

interface IPool {
    function swap(address recipient, int256 amount) external returns (int256 amount0, int256 amount1);
    event Swap(address indexed sender, int256 amount0);
}

library FullMath {
    /// @notice Multiplies then divides.
    function mulDiv(uint256 a, uint256 b, uint256 denominator) internal pure returns (uint256 result) {
        result = (a * b) / denominator;
    }
}

struct Position {
    uint128 liquidity;
    uint256 feeGrowth;
}

enum Side { Buy, Sell }

function toCurrency(address a) pure returns (Currency) {
    return Currency.wrap(a);
}
"#;
    let units = assert_extractor_invariants(source, Language::Solidity, "Pool.sol");
    let kind = |n: &str| get_unit_by_name(&units, n).unwrap().unit_type;
    assert_eq!(kind("MAX_FEE"), UnitType::Constant);
    assert_eq!(kind("Currency"), UnitType::Constant);
    assert_eq!(kind("InvalidFee"), UnitType::Function);
    assert_eq!(kind("IPool"), UnitType::Class);
    assert_eq!(kind("FullMath"), UnitType::Class);
    assert_eq!(kind("Position"), UnitType::Class);
    assert_eq!(kind("Side"), UnitType::Class);
    assert_eq!(kind("toCurrency"), UnitType::Function);

    // An interface stays one unit: its signatures are not split out.
    assert!(get_unit_by_name(&units, "swap").is_none());

    let mul = get_unit_by_name(&units, "mulDiv").unwrap();
    assert_eq!(mul.unit_type, UnitType::Method);
    assert_eq!(mul.parent_class.as_deref(), Some("FullMath"));
    assert_eq!(mul.parameters, vec!["a", "b", "denominator"]);
    assert_eq!(mul.return_type.as_deref(), Some("uint256 result"));
    assert_eq!(mul.docstring.as_deref(), Some("Multiplies then divides."));

    let to = get_unit_by_name(&units, "toCurrency").unwrap();
    assert_eq!(to.calls, vec!["wrap"]);
    assert_eq!(to.return_type.as_deref(), Some("Currency"));
}

#[test]
fn test_calls_with_options_and_new() {
    let source = r#"contract Factory {
    function deploy(address payable to) external {
        Pool pool = new Pool();
        (bool ok, ) = to.call{value: 1 ether}("");
        if (!ok) revert();
    }
}
"#;
    let units = parse(source, Language::Solidity, "Factory.sol");
    let f = get_unit_by_name(&units, "deploy").unwrap();
    assert!(f.calls.contains(&"call".to_string()), "{:?}", f.calls);
    assert!(f.calls.contains(&"Pool".to_string()), "{:?}", f.calls);
    assert!(f.variables.contains(&"pool".to_string()));
    assert!(f.variables.contains(&"ok".to_string()));
}
