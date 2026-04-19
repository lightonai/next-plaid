# Rust SQLite Wasm Spike

Date: 2026-04-19

## Goal

Answer one concrete question with a working proof:

- can the browser port move the SQLite keyword layer into Rust instead of
  keeping it in handwritten JavaScript?

## What was built

An isolated spike crate:

- `crates/next-plaid-browser-sqlite-spike`

It uses:

- `rusqlite`
- `sqlite-wasm-rs` transitively on `wasm32-unknown-unknown`
- `wasm-bindgen`
- `wasm-bindgen-test`

The spike does one minimal but meaningful workload:

1. open an in-memory SQLite database
2. create an FTS5 table
3. insert a few documents
4. run a BM25-ranked keyword query
5. run a subset-restricted keyword query
6. expose the result through a Wasm export

## What was verified

Host-side verification:

```bash
cargo test -p next-plaid-browser-sqlite-spike
```

Browser-target compile verification:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
CC_wasm32_unknown_unknown=/opt/homebrew/opt/llvm/bin/clang \
cargo check -p next-plaid-browser-sqlite-spike --target wasm32-unknown-unknown
```

Real browser verification:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
CC_wasm32_unknown_unknown=/opt/homebrew/opt/llvm/bin/clang \
wasm-pack test --headless --chrome
```

One-command reproduction:

```bash
./next-plaid-browser/scripts/prove_rust_sqlite_wasm.sh
```

## Result

The spike succeeded.

The real browser test passed with:

- SQLite opening inside browser-style Wasm
- FTS5 virtual table creation
- BM25 keyword ranking
- subset-constrained query execution

That is enough to say the Rust-side SQLite path is viable for the browser port.

## Important toolchain finding

The main blocker was not Rust itself. It was the C compiler used by
`sqlite-wasm-rs`.

On this machine:

- Apple/Homebrew Rust on PATH was not the right toolchain path for Wasm work
- Apple system `clang` could not compile C sources for
  `wasm32-unknown-unknown`
- Homebrew LLVM `clang` at `/opt/homebrew/opt/llvm/bin/clang` fixed the build

So this path is viable, but it has an explicit toolchain requirement that
should be documented and enforced in CI.

## Type binding finding

`wasm-pack` did generate a `.d.ts` file for the spike package, but the typing
surface is still coarse because the exported function is just:

- `sqlite_spike_probe_json(): string`

So this spike proves two separate things:

1. Rust SQLite in browser Wasm is viable.
2. Strong Rust-to-TypeScript contracts still need additional work on top.

For the typed contract layer, the next realistic options are:

- export structured `JsValue` payloads via `serde-wasm-bindgen`
- generate TypeScript types from the Rust contract crate
- use `ts-rs` or `wasm-bindgen` `typescript_custom_section` to keep Rust as the
  source of truth

## Architectural implication

This changes the recommendation for the browser runtime.

Instead of keeping the keyword engine in handwritten JavaScript, the better
path is now:

1. keep the Rust search/scoring kernel
2. port the current browser keyword engine into a Rust crate built around
   `rusqlite`
3. keep only a thin TypeScript worker bootstrap around the Wasm module
4. generate TypeScript contract types from Rust so drift is minimized

## What this spike did not verify

This spike did not yet prove:

- OPFS-backed persistence through `sqlite-wasm-vfs`
- IndexedDB-backed persistence through `sqlite-wasm-vfs`
- add / update / delete lifecycle operations for the browser FTS side
- the final Rust-to-TypeScript contract generation strategy

Those are the next focused slices.

## Recommendation

Proceed with a real port of:

- `runtime/keyword_engine.mjs`

into a Rust browser-side SQLite crate using `rusqlite`, while preserving:

- the current query behavior
- BM25 ranking semantics
- subset filtering
- future room for metadata filtering and mutation flows

At the same time, formalize the Wasm build environment:

- rustup-managed cargo/rustc
- `wasm32-unknown-unknown` target installed
- Homebrew LLVM `clang` wired in for `sqlite-wasm-rs`
