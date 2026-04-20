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
3. `wasm-opt` was unavailable locally, so the measured size is a release build
   without the extra Binaryen optimization step
4. the gzipped artifact landed slightly above the initial 800 KB target

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

Interpretation:

- this is slightly above the initial `<= 800 KB gzipped` target
- it is close enough to treat as a soft-pass, especially because no
  Binaryen/`wasm-opt` step was available locally

## Tooling findings

### 1. `wasm-pack` initially used the wrong toolchain

Without forcing rustup to the front of `PATH`, `wasm-pack` picked up the
Homebrew Rust toolchain and failed to see the installed `wasm32` target.

Working invocation:

- `env PATH="$HOME/.cargo/bin:$PATH" ...`

### 2. `wasm-opt` was not installed locally

`wasm-pack` attempted to run `wasm-opt` and failed because the binary was not
present.

For this spike, the crate was configured with:

- `[package.metadata.wasm-pack.profile.release]`
- `wasm-opt = false`

That allowed the release build to complete and gave a valid size number for the
unoptimized-by-Binaryen artifact.

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
2. treat `807,033` gzipped as a soft-pass against the `800 KB` target
3. if size sensitivity becomes acute, rerun the measurement with Binaryen
   available before making the final packaging call
4. keep the build-integrity harness in scope after extraction so the browser
   Wasm wrapper is continuously checked against the native Rust source

## Bottom line

The shared-Rust preprocessor direction is technically viable on the browser
target. The compile risk that could have invalidated the whole architecture did
not materialize.

The remaining issues are practical tuning and packaging questions, not
fundamental feasibility blockers.
