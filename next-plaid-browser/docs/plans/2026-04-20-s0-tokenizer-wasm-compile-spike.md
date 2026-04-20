# S0: Tokenizer Wasm Compile Spike

Written: 2026-04-20
Starting point: `741632b`
Status: executed, see companion report
Companion docs:
- `2026-04-20-rust-preprocessor-architecture-sketch.md`
- `2026-04-20-browser-embedding-parity-plan.md`
- `2026-04-20-s0-tokenizer-wasm-compile-report.md`

## Why this slice exists

The shared-Rust preprocessor direction is only worth pursuing if the intended
tokenizer / preprocessing stack actually compiles to browser Wasm with an
acceptable size and a usable runtime footprint.

That is the load-bearing constraint. It should be validated before any
extraction work starts.

## Goal

Answer one question empirically:

**Can the selected Rust tokenizer path compile to browser Wasm, load in a real
browser context, and run one tokenize call within an acceptable artifact size
budget?**

## Scope

### In scope

1. Create a throwaway spike crate for browser-target tokenizer compilation.
2. Configure the tokenizer dependency stack for the intended browser target.
3. Export one minimal query-tokenization function over wasm-bindgen.
4. Build the crate for the browser target in release mode.
5. Optimize the output artifact.
6. Load the artifact in a browser harness and run one tokenize call.
7. Record:
   - compile success or failure
   - required feature changes
   - blocked features
   - final raw and gzipped Wasm size
   - any browser runtime issues

### Out of scope

1. Extracting native preprocessing code into a shared crate.
2. Integrating the spike crate into the browser encoder worker.
3. Replacing the fixture tokenizer.
4. Full preprocessing parity.
5. ORT-Web integration changes.
6. Effect client refactors.

## Pass / fail criteria

### Pass

The spike passes if all of the following are true:

1. The tokenizer crate compiles to browser-target Wasm.
2. The artifact loads in the browser harness.
3. One exported tokenize call runs successfully.
4. No unsupported feature remains on the critical path.
5. The optimized artifact is within the initial budget:
   - target **<= 800 KB gzipped**

### Soft-pass

The spike can still be treated as usable with explicit follow-up if:

1. the build succeeds
2. the browser call succeeds
3. the artifact exceeds the size target but remains close enough to justify
   tuning

### Fail

The spike fails if any of the following happen:

1. the dependency stack does not compile to browser Wasm
2. the build requires native-only features that cannot be disabled cleanly
3. the browser runtime path does not execute successfully
4. the artifact size is far outside the initial budget with no obvious tuning
   path

## Known risk checklist

The spike should explicitly check and document these common browser-target
failure points:

1. regex backend compatibility
2. `onig` or other native dependencies
3. rayon / threading assumptions
4. `rand` / `getrandom` browser feature requirements
5. file I/O or mmap-dependent paths
6. normalizer behavior that depends on unsupported platform features
7. final binary size after optimization

## Proposed implementation shape

Recommended spike crate:

- `next-plaid-browser/crates/spike-tokenizer-wasm`

Recommended exported surface:

- `init(tokenizer_json_bytes, config_json_bytes)`
- `tokenize(text) -> token ids`

This should stay intentionally small. The spike is proving build viability, not
designing the final runtime interface.

## Build lane

Recommended build lane:

1. release browser-target Wasm build
2. binary optimization pass
3. gzip size measurement
4. browser smoke execution

The exact commands may change based on the toolchain we settle on, but the
recorded result should include the actual commands used.

## Deliverables

The slice should produce:

1. the throwaway spike crate
2. a short written report saved in the repo
3. recorded raw and gzipped Wasm sizes
4. recorded feature flags and disabled-path notes
5. a clear recommendation:
   - proceed with shared-Rust extraction
   - proceed only with caveats
   - do not proceed; use fallback browser tokenizer direction instead

## Follow-up decision tree

### If the spike passes

Proceed to:

- SA shared-core extraction

### If the spike soft-passes

Proceed only after writing down:

- which tuning work is needed
- whether the size tradeoff is acceptable

### If the spike fails

Do not start extraction.

Instead:

1. record the blocker
2. decide whether the blocker can be removed with a smaller dependency surface
3. if not, fall back to the browser-native tokenizer path

## Completion condition

This slice is complete when the repository contains a working spike artifact
and a written yes / no / caveated answer about whether the shared-Rust
preprocessor direction is technically viable on the browser target.
