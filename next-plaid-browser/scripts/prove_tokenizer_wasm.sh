#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="$ROOT/crates/spike-tokenizer-wasm"
TARGET="wasm32-unknown-unknown"
ARTIFACT="$CRATE_DIR/pkg/spike_tokenizer_wasm_bg.wasm"

source "$ROOT/scripts/wasm_env.sh"

resolve_wasm_toolchain
require_tool "${WASM_PACK:-}" "wasm-pack" "Install it with: cargo install wasm-pack --locked"
ensure_wasm_target "$TARGET"

"$WASM_PACK" build \
  "$CRATE_DIR" \
  --target web \
  --release \
  --out-dir pkg

if [[ ! -f "$ARTIFACT" ]]; then
  echo "expected wasm artifact not found: $ARTIFACT"
  exit 1
fi

raw_bytes="$(wc -c < "$ARTIFACT" | tr -d ' ')"
gzip_bytes="$(gzip -c -9 "$ARTIFACT" | wc -c | tr -d ' ')"

echo "Tokenizer wasm build succeeded"
echo "artifact: $ARTIFACT"
echo "raw_bytes: $raw_bytes"
echo "gzip_bytes: $gzip_bytes"

if [[ -n "${WASM_OPT:-}" ]]; then
  optimized_artifact="$(mktemp "${TMPDIR:-/tmp}/spike-tokenizer-wasm-opt.XXXXXX")"
  trap 'rm -f "$optimized_artifact"' EXIT

  "$WASM_OPT" -Oz "$ARTIFACT" -o "$optimized_artifact"

  optimized_raw_bytes="$(wc -c < "$optimized_artifact" | tr -d ' ')"
  optimized_gzip_bytes="$(gzip -c -9 "$optimized_artifact" | wc -c | tr -d ' ')"

  echo "optimized_raw_bytes: $optimized_raw_bytes"
  echo "optimized_gzip_bytes: $optimized_gzip_bytes"
else
  echo "wasm-opt not found; skipping optimized size report"
fi
