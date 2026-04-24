#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/Cargo.toml"
TARGET="wasm32-unknown-unknown"
PACKAGE="next-plaid-browser-wasm"
source "$ROOT/scripts/wasm_env.sh"

resolve_wasm_toolchain
require_tool "${CARGO:-}" "cargo"
require_tool "${RUSTC:-}" "rustc"
ensure_wasm_target "$TARGET"

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
