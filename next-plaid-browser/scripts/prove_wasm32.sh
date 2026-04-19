#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/Cargo.toml"
TARGET="wasm32-unknown-unknown"
PACKAGE="next-plaid-browser-wasm"
RUSTUP_BIN="${HOME}/.cargo/bin/rustup"
CARGO_BIN="${HOME}/.cargo/bin/cargo"
RUSTC_BIN="${HOME}/.cargo/bin/rustc"

if [[ -x "$RUSTUP_BIN" ]]; then
  RUSTUP="$RUSTUP_BIN"
  CARGO="$CARGO_BIN"
  RUSTC="$RUSTC_BIN"
else
  RUSTUP="$(command -v rustup || true)"
  CARGO="$(command -v cargo || true)"
  RUSTC="$(command -v rustc || true)"
fi

if [[ -z "${CARGO:-}" ]]; then
  echo "cargo is not installed"
  exit 1
fi

if [[ -z "${RUSTC:-}" ]]; then
  echo "rustc is not installed"
  exit 1
fi

if [[ -z "${RUSTUP:-}" ]]; then
  cat <<'EOF'
rustup is not installed, so this machine cannot reliably bootstrap the wasm target.

Install rustup, then run:

  rustup target add wasm32-unknown-unknown
  ./next-plaid-browser/scripts/prove_wasm32.sh
EOF
  exit 2
fi

# Keep cargo and rustc on the same toolchain. This machine may also have a
# Homebrew Rust install on PATH, which can make `cargo` and `rustc` disagree
# about the installed targets.
export PATH="$(dirname "$CARGO"):$PATH"

LLVM_CLANG="/opt/homebrew/opt/llvm/bin/clang"
if [[ -x "$LLVM_CLANG" ]]; then
  export CC_wasm32_unknown_unknown="$LLVM_CLANG"
fi

"$RUSTUP" target add "$TARGET" >/dev/null

"$CARGO" build \
  --manifest-path "$MANIFEST" \
  -p "$PACKAGE" \
  --target "$TARGET" \
  --release

ARTIFACT="$ROOT/target/$TARGET/release/${PACKAGE//-/_}.wasm"

if [[ ! -f "$ARTIFACT" ]]; then
  echo "expected wasm artifact not found: $ARTIFACT"
  exit 3
fi

echo "wasm build succeeded"
echo "$ARTIFACT"
