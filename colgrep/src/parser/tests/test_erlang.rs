//! Tests for Erlang code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const KV: &str = r#"%%% @doc A tiny key-value store.
-module(kv_store).
-behaviour(gen_server).

-export([start_link/0, put/2, get/1]).
-import(lists, [foldl/3]).

-include_lib("kernel/include/logger.hrl").

-define(TIMEOUT, 5000).

-record(state, {table :: ets:tid(), count = 0 :: non_neg_integer()}).

-type key() :: atom() | binary().

%% @doc Stores a value.
-spec put(key(), term()) -> ok.
put(Key, Value) when is_atom(Key) ->
    gen_server:call(?MODULE, {put, Key, Value}, ?TIMEOUT);
put(Key, Value) ->
    put(binary_to_atom(Key), Value).

get(Key) ->
    case ets:lookup(kv, Key) of
        [{_, V}] -> {ok, V};
        [] -> {error, not_found}
    end.

handle_call({put, K, V}, _From, State = #state{count = C}) ->
    ets:insert(kv, {K, V}),
    Count = C + 1,
    F = fun ?MODULE:get/1,
    {reply, F(K), State#state{count = Count}}.
"#;

#[test]
fn test_function_clauses_form_one_unit() {
    let units = assert_extractor_invariants(KV, Language::Erlang, "kv_store.erl");

    let puts: Vec<_> = units.iter().filter(|u| u.name == "put").collect();
    assert_eq!(puts.len(), 1, "both clauses of put/2 are one unit");
    let put = puts[0];
    let text = build_embedding_text(put);
    let expected = r#"Function: put
Signature: put(Key, Value) when is_atom(Key) ->
Description: Stores a value.
Parameters: Key, Value
Returns: ok
Calls: binary_to_atom, gen_server:call, is_atom, put
Uses: gen_server
File: kv store kv_store.erl
Code:
%% @doc Stores a value.
-spec put(key(), term()) -> ok.
put(Key, Value) when is_atom(Key) ->
    gen_server:call(?MODULE, {put, Key, Value}, ?TIMEOUT);
put(Key, Value) ->
    put(binary_to_atom(Key), Value)."#;
    assert_eq!(text, expected);
    assert_eq!((put.line, put.end_line), (16, 21));
    assert!(put.has_branches);
}

#[test]
fn test_case_and_remote_calls() {
    let units = parse(KV, Language::Erlang, "kv_store.erl");
    let get = get_unit_by_name(&units, "get").unwrap();
    assert_eq!(get.unit_type, UnitType::Function);
    assert_eq!(get.parameters, vec!["Key"]);
    assert_eq!(get.calls, vec!["ets:lookup"]);
    assert_eq!(get.imports, vec!["ets"]);
    assert!(get.has_branches);
    assert!(get.docstring.is_none());
}

/// Patterns in the head are parameters; variables bound in the body are
/// variables; `?MODULE:f` and `fun ?MODULE:f/1` are local references.
#[test]
fn test_patterns_variables_and_fun_references() {
    let units = parse(KV, Language::Erlang, "kv_store.erl");
    let h = get_unit_by_name(&units, "handle_call").unwrap();
    assert_eq!(h.parameters, vec!["{put, K, V}", "_From", "State"]);
    assert_eq!(h.variables, vec!["Count", "F"]);
    assert!(h.calls.contains(&"ets:insert".to_string()), "{:?}", h.calls);
}

#[test]
fn test_records_types_and_macros() {
    let units = parse(KV, Language::Erlang, "kv_store.erl");

    let state = get_unit_by_name(&units, "state").unwrap();
    assert_eq!(state.unit_type, UnitType::Class);
    assert_eq!(state.variables, vec!["table", "count"]);

    let key = get_unit_by_name(&units, "key").unwrap();
    assert_eq!(key.unit_type, UnitType::Class);

    let timeout = get_unit_by_name(&units, "TIMEOUT").unwrap();
    assert_eq!(timeout.unit_type, UnitType::Constant);
    assert_eq!(timeout.code, "-define(TIMEOUT, 5000).");
}

#[test]
fn test_file_imports() {
    let units = parse(KV, Language::Erlang, "kv_store.erl");
    // Module attributes are raw code carrying the file's dependencies.
    let header = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.line == 1)
        .unwrap();
    assert_eq!(header.imports, vec!["gen_server", "lists", "logger"]);
}

/// OTP 27 `-doc` attributes document the function; only the first paragraph
/// is kept as the description.
#[test]
fn test_doc_attribute() {
    let source = r#"-module(m).

-doc """
Reverses a list.

## Examples
    1> m:rev([1,2]).
""".
-spec rev(list()) -> list().
rev(L) -> lists:reverse(L).
"#;
    let units = parse(source, Language::Erlang, "m.erl");
    let rev = get_unit_by_name(&units, "rev").unwrap();
    assert_eq!(rev.docstring.as_deref(), Some("Reverses a list."));
    assert_eq!(rev.return_type.as_deref(), Some("list()"));
    assert_eq!(rev.line, 3);
}

/// Same name, different arity: two functions. A banner comment separated by
/// a blank line is not documentation.
#[test]
fn test_arity_and_banner_comments() {
    let source = r#"%%====================================================================
%% API
%%====================================================================

seq(N) -> seq(1, N).
seq(A, B) when A =< B -> [A | seq(A + 1, B)];
seq(_, _) -> [].
"#;
    let units = assert_extractor_invariants(source, Language::Erlang, "s.erl");
    let seqs: Vec<_> = units.iter().filter(|u| u.name == "seq").collect();
    assert_eq!(seqs.len(), 2);
    assert_eq!((seqs[0].line, seqs[0].end_line), (5, 5));
    assert_eq!((seqs[1].line, seqs[1].end_line), (6, 7));
    assert!(seqs[0].docstring.is_none());
}

#[test]
fn test_callbacks_and_headers() {
    let source = r#"%% Called when the server starts.
-callback init(Args :: term()) -> {ok, State :: term()}.
"#;
    let units = parse(source, Language::Erlang, "my_behaviour.hrl");
    let init = get_unit_by_name(&units, "init").unwrap();
    assert_eq!(init.unit_type, UnitType::Function);
    assert_eq!(
        init.docstring.as_deref(),
        Some("Called when the server starts.")
    );
    assert_eq!(init.return_type.as_deref(), Some("{ok, State :: term()}"));
}

/// rebar.config / .app.src are files of terms: indexed as raw code.
#[test]
fn test_term_files() {
    let source = r#"{erl_opts, [debug_info]}.
{deps, [cowboy, jsx]}.
"#;
    let units = assert_extractor_invariants(source, Language::Erlang, "rebar.config");
    assert!(units.iter().all(|u| u.unit_type == UnitType::RawCode));
}

/// A construct the grammar does not know (sigils) must not hide the
/// functions around it.
#[test]
fn test_error_recovery_keeps_functions() {
    let source = r#"-module(m).
a() -> ~"sigil".
b(X) -> X + 1.
"#;
    let units = assert_extractor_invariants(source, Language::Erlang, "m.erl");
    assert!(
        get_unit_by_name(&units, "b").is_some(),
        "{:?}",
        units.iter().map(|u| &u.name).collect::<Vec<_>>()
    );
}
