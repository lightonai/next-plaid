#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/Cargo.toml"
PACKAGE="next-plaid-browser-sqlite-spike"
TARGET="wasm32-unknown-unknown"
source "$ROOT/scripts/wasm_env.sh"

resolve_wasm_toolchain
require_tool "${CARGO:-}" "cargo"
require_tool "${WASM_PACK:-}" "wasm-pack" "Install it with: cargo install wasm-pack --locked"
ensure_wasm_target "$TARGET"

"$CARGO" check \
  --manifest-path "$MANIFEST" \
  -p "$PACKAGE" \
  --target "$TARGET"

(
  cd "$ROOT/crates/$PACKAGE"
  "$WASM_PACK" test --headless --chrome
)

echo "Rust SQLite Wasm spike succeeded"
