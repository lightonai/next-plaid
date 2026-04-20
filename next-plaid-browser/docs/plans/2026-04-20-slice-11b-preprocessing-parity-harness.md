# Slice 11b: Native Preprocessing Fixture Export And Browser Parity Harness

Written: 2026-04-20
Starting point: `741632b`
Status: provisional follow-on plan
Companion docs:
- `2026-04-20-effect-tokenizer-and-parity-proposal.md`
- `2026-04-20-browser-embedding-parity-plan.md`
- `2026-04-20-browser-embedding-handoff.md`
- `2026-04-20-rust-preprocessor-architecture-sketch.md`
- `2026-04-20-s0-tokenizer-wasm-compile-spike.md`

## Status note

This document predates the shared-Rust preprocessor direction becoming the
preferred architecture.

Under the current direction:

1. the immediate next gate is the S0 tokenizer Wasm compile spike
2. preprocessing harness work remains useful, but only after the shared-Rust
   direction is proven viable
3. once extraction lands, this harness becomes primarily a native-vs-Wasm
   build-integrity check rather than a browser-reimplementation parity
   investigation

## Why this slice exists

The browser model-worker proof exists, but it does not yet prove the
quality-critical part of browser encoding:

- how text is normalized
- how tokens are produced
- where the prefix token lands
- how truncation reserves room for that prefix
- how query expansion fills the remaining query slots
- when `token_type_ids` are present versus omitted

That means the highest-information follow-on work after viability is proven is
not embedding numerics and not Effect client cleanup. The relevant next harness
work is:

**export native preprocessing fixtures and compare the browser path against
them exactly**

This slice is intentionally independent of:

- Slice 10 wire hardening
- future Effect worker-client cleanup
- future embedding parity work

It can land without reopening the Rust/Wasm search boundary.

## Goal

Produce a browser-readable fixture set from the native encoder path and use it
to prove that browser preprocessing matches native behavior exactly on a fixed
golden query set.

## Scope

### In scope

1. Native-side preprocessing fixture export.
2. A stable fixture transport format.
3. Browser-side fixture loading.
4. Browser preprocessing parity tests.
5. Evaluation of at least one browser tokenizer candidate against the fixture
   set.
6. A tokenizer decision memo or explicit fallback trigger.

### Out of scope

1. Effect worker-client refactor.
2. Numerical embedding parity.
3. WebGPU.
4. Cross-origin-isolated multithreading.
5. Search or scoring logic changes.
6. Browser document encoding.
7. A self-compiled Rust tokenizer wasm implementation.

## Locked decisions for this slice

1. **Fixture transport is JSON on disk plus browser `fetch`.**
   This keeps the fixtures diffable, browser-readable, and decoupled from TS
   build generation.

2. **Chromium is the active browser lane.**
   The preprocessing harness should run in the Chrome/Chromium lane first. The
   result should stay compatible with the existing browser harness direction.

3. **Preprocessing parity is exact, not approximate.**
   The comparison for this slice is structural equality, not tolerance-based
   similarity.

4. **Tokenizer fallback order is A -> B1 -> B2.**
   Where:
   - A = browser-native tokenizer candidate
   - B1 = existing browser-shipped wasm tokenizer package
   - B2 = self-compiled Rust tokenizer wasm path

5. **The current fixture tokenizer remains smoke-only.**
   It may continue to support tiny harness proofs, but it is not a candidate
   production tokenizer.

## Native behavior that the fixtures must capture

The fixture export has to preserve the actual native preprocessing contract.

That means each fixture case must be rich enough to validate:

1. raw input text
2. normalized text after lower-casing rules
3. tokenizer-produced ids before prefix insertion
4. real token length before padding
5. prefix token id
6. prefix insertion at slot 1
7. final token id sequence after prefix insertion
8. attention mask
9. token type ids when the model uses them
10. omission of token type ids when the model does not use them
11. truncation behavior with one slot reserved for the prefix token
12. query expansion behavior when enabled
13. final logical sequence length
14. the relevant config snapshot from `onnx_config.json`

## Proposed fixture shape

The exact field names can change during implementation, but the fixture needs
the following logical content:

- `case_id`
- `text`
- `normalized_text`
- `is_query`
- `config`
  - `query_prefix`
  - `query_length`
  - `do_query_expansion`
  - `uses_token_type_ids`
  - `mask_token_id`
  - `pad_token_id`
  - `do_lower_case`
- `prefix_token_id`
- `pre_prefix`
  - `token_ids`
  - `attention_mask`
  - `token_type_ids`
  - `real_length`
- `final`
  - `token_ids`
  - `attention_mask`
  - `token_type_ids`
  - `logical_length`
  - `layout`

This should be written as plain JSON so the browser harness can fetch it and so
reviewers can inspect it directly in diffs.

## Fixture set requirements

The slice should not ship with a single happy-path query. The fixture set must
cover the specific edge cases that could create retrieval drift.

Minimum required cases:

1. lower-case disabled, simple short query
2. lower-case enabled, mixed-case query
3. query shorter than `query_length`
4. query long enough to force truncation
5. query that exercises the reserved-prefix-slot behavior
6. model config with `uses_token_type_ids = true`
7. model config with `uses_token_type_ids = false`
8. query-expansion enabled
9. unknown token path
10. punctuation / whitespace normalization edge case that the chosen browser
    tokenizer candidate is likely to handle differently if it drifts

## Proposed file touch points

### Native side

The export tool should live close to the native preprocessing source of truth.

Recommended location:

- a dedicated exporter binary or example in `next-plaid-onnx`

Likely touch points:

- `next-plaid-onnx/src/lib.rs`
- `next-plaid-onnx/examples/...` or `next-plaid-onnx/src/bin/...`

### Browser side

Likely touch points:

- `next-plaid-browser/fixtures/preprocessing/`
- `next-plaid-browser/browser-src/model-worker/`
- `next-plaid-browser/browser-src/playwright-harness/`
- `next-plaid-browser/scripts/build_browser_harness.mjs`
- `next-plaid-browser/scripts/playwright_smoke.mjs`

## Browser implementation direction

This slice does not require a production tokenizer runtime to be selected up
front. It requires the browser harness to be able to run at least one real
candidate against the native fixtures.

Recommended browser-side structure:

1. Add a preprocessing test helper that:
   - loads one fixture file
   - runs one browser tokenizer/preprocessing candidate
   - compares every parity-relevant field

2. Keep the comparison layer separate from the worker transport layer.
   The harness should be able to compare preprocessing behavior without routing
   through the full worker API if that would make failures harder to inspect.

3. Use the browser harness to validate browser behavior, not just host-side
   TypeScript behavior.

## Tokenizer decision rule for the end of this slice

The slice is allowed to end in one of two states:

### State A: browser-native candidate accepted

Conditions:

- exact preprocessing parity passes on the fixture set
- bundle-size contribution is acceptable
- no unresolved edge-case mismatches remain

### State B: browser-native candidate rejected

Conditions:

- exact preprocessing parity fails
  or
- bundle-size contribution is outside the initial budget
  or
- unresolved edge-case mismatches remain

If State B happens, the slice must write down the explicit fallback trigger:

- escalate to B1 first
- escalate to B2 only if B1 also fails

## Bundle-size rubric

If multiple tokenizer candidates clear parity, use measured browser cost to
break the tie.

The initial rubric should include:

- gzipped bundle contribution
- cold-start impact
- implementation complexity

Starting budget:

- tokenizer-specific gzipped bundle contribution target of **<= 400 KB**

Anything above that should require explicit justification in the decision memo.

## Test plan

The implementation should verify at least:

1. native fixture export completes successfully
2. exported JSON fixtures are valid and browser-readable
3. browser preprocessing comparison passes on golden cases for the selected
   candidate
4. failures are legible enough to diagnose:
   - token id mismatch
   - attention mask mismatch
   - token type id mismatch
   - prefix placement mismatch
   - truncation mismatch
   - query expansion mismatch
5. the existing browser smoke lane still passes

Expected commands:

```bash
cd /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser
bun run typecheck
```

```bash
cargo test --manifest-path /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser/Cargo.toml
```

```bash
cd /Users/pooks/Dev/lighton-benchmark/next-plaid/next-plaid-browser
bun run smoke:chromium
```

And one native fixture export command to be added as part of this slice.

## Acceptance criteria

Slice 11b is complete when all of the following are true:

1. Native preprocessing fixtures exist on disk in a stable JSON format.
2. The browser harness reads those fixtures through `fetch`.
3. Prefix insertion, truncation, conditional token-type behavior, and
   query-expansion behavior are compared exactly on the fixture set.
4. At least one browser tokenizer candidate has been run against the fixture
   set.
5. A tokenizer decision memo exists, or an explicit fallback trigger exists.
6. The browser smoke lane still passes after the parity harness lands.

## Recommended implementation order

1. Add a native preprocessing fixture exporter.
2. Write the initial golden fixture set.
3. Add browser-side fixture loading.
4. Add exact preprocessing comparison helpers.
5. Run the first browser tokenizer candidate.
6. Record the tokenizer decision or fallback trigger.
7. Re-run the existing browser verification lane.

## Final recommendation

Treat this slice as a pure information-gathering and contract-proving slice.

Do not broaden the model-worker API here.
Do not start embedding numerics here.
Do not start performance tuning here.

Use this slice to prove the browser is preparing queries the same way native
does. That is the highest-value next fact to establish.
