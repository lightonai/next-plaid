# S0 Tokenizer Wasm Compile Report

Written: 2026-04-20
Status: complete
Related plan:
- `2026-04-20-s0-tokenizer-wasm-compile-spike.md`

## Question answered

Can a browser-target Rust tokenizer path compile to Wasm, load in a real
browser context, and run one tokenize call with a practical artifact size?

## Short answer

**Yes, with caveats.**

The spike compiled, built a browser Wasm package, and passed headless Chrome
tests. The main caveats are:

1. the browser-safe feature set differs from the native crate's current
   `onig`-based configuration
2. local tooling required forcing rustup onto `PATH`
3. the unoptimized release artifact landed slightly above the initial 800 KB
   target, so Binaryen optimization remains part of the production build story

That makes this a **soft-pass**, not a clean pass.

## What was built

A throwaway crate was added at:

- `next-plaid-browser/crates/spike-tokenizer-wasm`

The spike exposes three minimal operations:

1. `init(tokenizer_json, config_json)`
2. `tokenize(text)`
3. `reset()`

The browser tests construct a small WordPiece tokenizer in Rust, serialize it
to JSON bytes, reload it through `Tokenizer::from_bytes`, and then prove that
the Wasm build can tokenize in headless Chrome.

## Dependency configuration tested

The spike used:

- `tokenizers = "=0.21.1"`
- `default-features = false`
- `features = ["unstable_wasm"]`

Important implication:

- this does **not** use `onig`
- this is browser-safe configuration, not the exact native dependency shape as
  it exists today

## Commands run

Host test:

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml -p spike-tokenizer-wasm
```

Browser build:

```bash
env PATH="$HOME/.cargo/bin:$PATH" \
  wasm-pack build /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/spike-tokenizer-wasm \
  --target web --release --out-dir pkg
```

Browser test:

```bash
env PATH="$HOME/.cargo/bin:$PATH" \
  wasm-pack test --chrome --headless \
  /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/spike-tokenizer-wasm
```

Size measurement:

```bash
WASM=/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/crates/spike-tokenizer-wasm/pkg/spike_tokenizer_wasm_bg.wasm
wc -c < "$WASM"
gzip -c "$WASM" | wc -c
```

Follow-up proof script after installing Binaryen and `wasm-bindgen`:

```bash
/Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/scripts/prove_tokenizer_wasm.sh
```

## Results

### Functional result

Host tests:

- passed

Headless Chrome Wasm tests:

- passed

Observed browser test cases:

1. round-trip tokenization of a small WordPiece fixture
2. unknown-token fallback to `[UNK]`

### Size result

Release Wasm artifact:

- raw bytes: `2,414,001`
- gzipped bytes: `807,033`

`wasm-opt -Oz` follow-up result:

- raw bytes: `1,999,938`
- gzipped bytes: `738,691`

Interpretation:

- this is slightly above the initial `<= 800 KB gzipped` target
- with Binaryen installed, the optimized artifact clears the target

## Tooling findings

### 1. `wasm-pack` initially used the wrong toolchain

Without forcing rustup to the front of `PATH`, `wasm-pack` picked up the
Homebrew Rust toolchain and failed to see the installed `wasm32` target.

Working invocation:

- `env PATH="$HOME/.cargo/bin:$PATH" ...`

### 2. `wasm-opt` is now installed and worth keeping in the release path

The first spike ran without Binaryen. A follow-up validation after installing
`wasm-opt` showed a meaningful size drop, from `807,033` gzipped to `738,691`
gzipped.

That means the production browser build should keep an explicit optimization
step rather than relying on the larger raw release artifact.

### 3. The browser-safe tokenizer feature set is real

The important positive result is that `tokenizers` with `unstable_wasm`
compiled and ran in a real browser test lane.

That removes the biggest architectural uncertainty from the shared-Rust
preprocessor direction.

## Recommendation

Proceed to the next stage with caveats.

### Recommended next step

Proceed to:

- SA shared-core extraction

### Caveats to carry forward

1. keep the browser tokenizer line browser-safe; do not assume the current
   native `onig` feature configuration carries over unchanged
2. keep `wasm-opt` in the toolchain and use the optimized artifact size for
   packaging decisions
3. keep the build-integrity harness in scope after extraction so the browser
   Wasm wrapper is continuously checked against the native Rust source

## Bottom line

The shared-Rust preprocessor direction is technically viable on the browser
target. The compile risk that could have invalidated the whole architecture did
not materialize.

The remaining issues are practical tuning and packaging questions, not
fundamental feasibility blockers.
