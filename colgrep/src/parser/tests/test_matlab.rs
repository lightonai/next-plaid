//! Tests for MATLAB code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const SMOOTH: &str = r#"function [out, n] = smooth_signal(x, window)
%SMOOTH_SIGNAL Smooth a signal with a moving average.
%   OUT = SMOOTH_SIGNAL(X, WINDOW) returns the smoothed signal.
if nargin < 2
    window = 5;
end
k = ones(window, 1) / window;
out = conv(x, k, 'same');
n = numel(out);
for i = 1:n
    out(i) = clip(out(i));
end
end

function y = clip(x)
% CLIP Clamp negative values to zero.
y = max(x, 0);
end
"#;

#[test]
fn test_function_embedding_text() {
    let units = parse(SMOOTH, Language::Matlab, "smooth_signal.m");

    let unit = get_unit_by_name(&units, "smooth_signal").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: smooth_signal
Signature: function [out, n] = smooth_signal(x, window)
Description: SMOOTH_SIGNAL Smooth a signal with a moving average. OUT = SMOOTH_SIGNAL(X, WINDOW) returns the smoothed signal.
Parameters: x, window
Returns: out, n
Calls: clip, conv, numel, ones
Variables: i, k, n, out, window
File: smooth signal smooth_signal.m
Code:
function [out, n] = smooth_signal(x, window)
%SMOOTH_SIGNAL Smooth a signal with a moving average.
%   OUT = SMOOTH_SIGNAL(X, WINDOW) returns the smoothed signal.
if nargin < 2
    window = 5;
end
k = ones(window, 1) / window;
out = conv(x, k, 'same');
n = numel(out);
for i = 1:n
    out(i) = clip(out(i));
end
end"#;
    assert_eq!(text, expected);
}

#[test]
fn test_local_function_and_call_graph() {
    let mut units = assert_extractor_invariants(SMOOTH, Language::Matlab, "smooth_signal.m");
    crate::parser::build_call_graph(&mut units);
    let clip = get_unit_by_name(&units, "clip").unwrap();
    assert_eq!(clip.unit_type, UnitType::Function);
    assert_eq!((clip.line, clip.end_line), (15, 18));
    assert_eq!(clip.parameters, vec!["x"]);
    assert_eq!(clip.return_type.as_deref(), Some("y"));
    assert_eq!(
        clip.docstring.as_deref(),
        Some("CLIP Clamp negative values to zero.")
    );
    assert_eq!(clip.called_by, vec!["smooth_signal"]);
}

/// `out(i)` indexes a variable; it is not a call.
#[test]
fn test_indexing_is_not_a_call() {
    let units = parse(SMOOTH, Language::Matlab, "smooth_signal.m");
    let unit = get_unit_by_name(&units, "smooth_signal").unwrap();
    assert!(!unit.calls.contains(&"out".to_string()));
    assert!(!unit.calls.contains(&"x".to_string()));
}

#[test]
fn test_classdef() {
    let source = r#"classdef BankAccount < handle & matlab.mixin.Copyable
    %BANKACCOUNT A bank account that notifies on low balance.
    properties (Constant)
        RATE = 0.05;
    end
    properties
        Balance double = 0
    end
    events
        InsufficientFunds
    end
    methods
        function obj = BankAccount(balance)
            % Construct an account with an opening balance.
            obj.Balance = balance;
        end
        function withdraw(obj, amount)
            if amount > obj.Balance
                notify(obj, 'InsufficientFunds');
                return
            end
            obj.Balance = obj.Balance - amount;
        end
    end
    methods (Static)
        function r = rate()
            r = BankAccount.RATE;
        end
    end
    methods (Abstract)
        report(obj)
    end
end
"#;
    let units = assert_extractor_invariants(source, Language::Matlab, "BankAccount.m");

    let class = get_unit_by_name(&units, "BankAccount").unwrap();
    assert_eq!(class.unit_type, UnitType::Class);
    assert_eq!(class.extends.as_deref(), Some("handle"));
    assert_eq!(
        class.docstring.as_deref(),
        Some("BANKACCOUNT A bank account that notifies on low balance.")
    );
    assert_eq!((class.line, class.end_line), (1, 33));

    let ctor = units
        .iter()
        .find(|u| u.name == "BankAccount" && u.unit_type == UnitType::Method)
        .unwrap();
    assert_eq!(ctor.parent_class.as_deref(), Some("BankAccount"));
    assert_eq!(ctor.return_type.as_deref(), Some("obj"));

    let withdraw = get_unit_by_name(&units, "withdraw").unwrap();
    assert_eq!(withdraw.unit_type, UnitType::Method);
    assert_eq!(withdraw.parameters, vec!["obj", "amount"]);
    assert_eq!(withdraw.calls, vec!["notify"]);
    assert!(withdraw.has_branches);

    let rate = get_unit_by_name(&units, "rate").unwrap();
    assert_eq!(rate.parent_class.as_deref(), Some("BankAccount"));
    assert!(rate.parameters.is_empty());
}

#[test]
fn test_script_with_nested_function_and_import() {
    let source = r#"% Plot a sine wave.
import matlab.io.*
x = linspace(0, 2*pi, 100);
y = sin(x);
hold on
plot(x, y);

function z = outer(a)
    z = inner(a);
    function q = inner(b)
        q = b * 2;
    end
end
"#;
    let units = assert_extractor_invariants(source, Language::Matlab, "script.m");

    let raw = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode)
        .unwrap();
    assert_eq!((raw.line, raw.end_line), (1, 6));
    assert_eq!(raw.imports, vec!["matlab.io"]);

    let outer = get_unit_by_name(&units, "outer").unwrap();
    assert_eq!(outer.calls, vec!["inner"]);
    // The nested function's variables belong to it, not to the outer one.
    assert_eq!(outer.variables, vec!["z"]);
    let inner = get_unit_by_name(&units, "inner").unwrap();
    assert_eq!((inner.line, inner.end_line), (10, 12));
}

/// Command syntax (`hold on`, `disp hello`) is a call.
#[test]
fn test_command_syntax_is_a_call() {
    let source = "function show(x)\nhold on\nplot(x);\nend\n";
    let units = parse(source, Language::Matlab, "show.m");
    let show = get_unit_by_name(&units, "show").unwrap();
    assert_eq!(show.calls, vec!["hold", "plot"]);
}
