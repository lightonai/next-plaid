#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/Cargo.toml"
PACKAGE="next-plaid-browser-sqlite-spike"
TARGET="wasm32-unknown-unknown"
RUSTUP_BIN="${HOME}/.cargo/bin/rustup"
CARGO_BIN="${HOME}/.cargo/bin/cargo"
WASM_PACK_BIN="${HOME}/.cargo/bin/wasm-pack"

if [[ -x "$RUSTUP_BIN" ]]; then
  RUSTUP="$RUSTUP_BIN"
  CARGO="$CARGO_BIN"
  WASM_PACK="$WASM_PACK_BIN"
else
  RUSTUP="$(command -v rustup || true)"
  CARGO="$(command -v cargo || true)"
  WASM_PACK="$(command -v wasm-pack || true)"
fi

if [[ -z "${RUSTUP:-}" || -z "${CARGO:-}" || -z "${WASM_PACK:-}" ]]; then
  echo "rustup, cargo, and wasm-pack are required"
  exit 1
fi

export PATH="$(dirname "$CARGO"):$PATH"

LLVM_CLANG="/opt/homebrew/opt/llvm/bin/clang"
if [[ -x "$LLVM_CLANG" ]]; then
  export CC_wasm32_unknown_unknown="$LLVM_CLANG"
fi

"$RUSTUP" target add "$TARGET" >/dev/null

"$CARGO" check \
  --manifest-path "$MANIFEST" \
  -p "$PACKAGE" \
  --target "$TARGET"

(
  cd "$ROOT/crates/$PACKAGE"
  "$WASM_PACK" test --headless --chrome
)

echo "Rust SQLite Wasm spike succeeded"
