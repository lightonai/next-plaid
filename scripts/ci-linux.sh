#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo fmt --all -- --check
cargo clippy --all-targets --features openblas -- -D warnings
cargo test --features openblas
cargo doc --no-deps --features openblas
cargo bench --no-run --features openblas

pushd next-plaid-onnx >/dev/null
cargo test
cargo clippy --all-targets -- -D warnings
popd >/dev/null

pushd colgrep >/dev/null
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
popd >/dev/null
