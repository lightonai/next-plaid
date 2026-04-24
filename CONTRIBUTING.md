# Contributing

Thanks for helping improve NextPlaid. This repository contains the native Rust crates, the API server, `colgrep`, and the private browser workspace under `next-plaid-browser/`.

## Ground Rules

- Keep changes focused and reviewable.
- Do not commit generated build output such as `target/`, browser harness bundles, wasm package output, or local model downloads.
- Keep user-facing docs polished. Internal handoff notes, agent scratch files, and planning memos do not belong in the upstream-facing tree.
- Preserve the browser package as private until the project intentionally moves to npm or crates.io publishing.

## Native Workspace

Run the standard Rust checks from the repository root:

```bash
cargo fmt --all --check
cargo test --workspace --locked
```

When changing release-facing crates, also run a publish dry run before asking for review:

```bash
cargo publish -p next-plaid --locked --dry-run
cargo publish -p next-plaid-onnx --locked --dry-run
cargo publish -p colgrep --locked --dry-run
```

## Browser Workspace

Run browser checks from `next-plaid-browser/`:

```bash
bun install --frozen-lockfile
bun run typecheck
bun run test
cargo clippy --manifest-path Cargo.toml --workspace --locked --all-targets -- -D warnings
cargo test --manifest-path Cargo.toml
BROWSER_HEADLESS=1 ./scripts/test_browser_parity.sh chrome
bun run build:harness
bun run smoke:chromium
bun run parity:preprocess
```

The browser workspace has its own Rust lockfile and Bun lockfile. Update `next-plaid-browser/Cargo.lock` only when browser Rust dependencies change, and update `next-plaid-browser/bun.lock` only when browser JavaScript dependencies change. Keep the root `Cargo.lock` scoped to the root Rust workspace.

## Security

Please do not open public issues for suspected vulnerabilities. Use the private reporting process in `SECURITY.md`.
