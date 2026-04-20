# Browser Testing

This workspace uses a layered browser-testing strategy instead of trying to
force every concern through one tool.

## Test layers

### 1. Host-side Rust tests

Use normal `cargo test` for:

- kernel correctness
- native-vs-browser reference parity
- bundle manifest and loader validation

This is the fastest inner loop and should stay the main correctness gate.

### 2. Wasm parity tests in real browsers

Use `wasm-bindgen-test` through `wasm-pack test` for:

- Rust-authored tests that execute through the real Wasm/browser boundary
- query-time search parity checks in browser engines
- the worker-owned semantic search contract, because the current browser parity
  tests now load a named index and search it through the native-shaped runtime
  request path
- browser-run fusion fixtures for RRF and relative-score behavior

Recommended order of attention:

1. Chrome
2. Firefox
3. Safari

Chrome is the practical primary browser lane because it is usually the easiest
to automate in CI and the most convenient first gate for browser-side Wasm
correctness work.

The parity entrypoint in this repo is:

```bash
./scripts/test_browser_parity.sh chrome
```

To run all supported `wasm-pack` browser lanes from one command:

```bash
./scripts/test_browser_parity.sh all
```

Important constraint:

- the `chrome`, `firefox`, and `safari` `wasm-pack` lanes require those real
  browsers to exist on the machine

### 3. Playwright smoke tests

Use Playwright for browser-app behavior that goes beyond the Rust test harness:

- page boot
- Wasm module loading in a realistic page
- browser console failures
- future worker startup checks
- future OPFS / storage flows
- screenshots and interactive debugging

In this workspace, Playwright is the right smoke-test tool, not the main parity
framework.

The current smoke harness builds the Wasm crate for the web target, serves a
small harness page, boots a dedicated module worker, and verifies that the page
can load an index and execute the real worker-hosted Wasm request path
successfully for:

- stored-bundle semantic search after install and reload
- stored-bundle hybrid search after install and reload
- runtime health memory breakdown accounting
- semantic search
- keyword-only search
- hybrid fusion
- metadata-filtered subset search

Primary smoke command:

```bash
npm run smoke:chromium
```

Optional lanes:

```bash
npm run smoke:webkit
npm run smoke:firefox
npm run smoke:chrome
```

`chromium` is the recommended default because Playwright can provision it
directly. `chrome` is useful as a branded-browser regression lane when Google
Chrome is installed locally or in CI.

### 4. Safari verification

Real Safari still matters.

Use Safari WebDriver as the actual Safari verification lane on macOS when you
need confidence in Apple’s shipped engine rather than Playwright’s patched
WebKit build.

Practical rule:

- use Playwright WebKit for broad cross-browser smoke coverage
- use Safari WebDriver for real-Safari checks

## Best practices

- Keep Rust parity tests small and deterministic.
- Treat `wasm-bindgen-test` as the standard Wasm/browser harness.
- Treat Playwright as the standard browser-runtime smoke harness.
- Prefer Chromium as the first automated browser lane.
- Add branded Chrome only when you specifically want public-channel regression
  coverage.
- Keep Safari in the matrix, but do not make it the only fast local lane.
- Use headed runs for browser-debug sessions when headless automation is flaky.

## Commands

### Fast correctness loop

```bash
cargo test --manifest-path next-plaid-browser/Cargo.toml
```

### Browser parity

```bash
./scripts/test_browser_parity.sh chrome
./scripts/test_browser_parity.sh safari
./scripts/test_browser_parity.sh all
```

### Browser smoke

```bash
npm install
npm run smoke:chromium
```
