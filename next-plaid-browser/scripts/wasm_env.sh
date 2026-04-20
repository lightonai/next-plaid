#!/usr/bin/env bash
set -euo pipefail

resolve_wasm_toolchain() {
  local rustup_bin="${HOME}/.cargo/bin/rustup"
  local cargo_bin="${HOME}/.cargo/bin/cargo"
  local rustc_bin="${HOME}/.cargo/bin/rustc"
  local wasm_pack_bin="${HOME}/.cargo/bin/wasm-pack"
  local wasm_bindgen_bin="${HOME}/.cargo/bin/wasm-bindgen"

  if [[ -x "$rustup_bin" ]]; then
    export RUSTUP="$rustup_bin"
    export CARGO="$cargo_bin"
    export RUSTC="$rustc_bin"
  else
    export RUSTUP="${RUSTUP:-$(command -v rustup || true)}"
    export CARGO="${CARGO:-$(command -v cargo || true)}"
    export RUSTC="${RUSTC:-$(command -v rustc || true)}"
  fi

  if [[ -x "$wasm_pack_bin" ]]; then
    export WASM_PACK="$wasm_pack_bin"
  else
    export WASM_PACK="${WASM_PACK:-$(command -v wasm-pack || true)}"
  fi

  if [[ -x "$wasm_bindgen_bin" ]]; then
    export WASM_BINDGEN="$wasm_bindgen_bin"
  else
    export WASM_BINDGEN="${WASM_BINDGEN:-$(command -v wasm-bindgen || true)}"
  fi

  export WASM_OPT="${WASM_OPT:-$(command -v wasm-opt || true)}"

  if [[ -n "${CARGO:-}" ]]; then
    export PATH="$(dirname "$CARGO"):$PATH"
  fi

  local llvm_clang="/opt/homebrew/opt/llvm/bin/clang"
  if [[ -x "$llvm_clang" ]]; then
    export CC_wasm32_unknown_unknown="${CC_wasm32_unknown_unknown:-$llvm_clang}"
  fi
}

require_tool() {
  local value="$1"
  local name="$2"
  local hint="${3:-}"

  if [[ -n "$value" ]]; then
    return 0
  fi

  echo "$name is required"
  if [[ -n "$hint" ]]; then
    echo
    echo "$hint"
  fi
  exit 1
}

ensure_wasm_target() {
  local target="${1:-wasm32-unknown-unknown}"

  require_tool \
    "${RUSTUP:-}" \
    "rustup" \
    $'Install rustup, then run:\n\n  rustup target add wasm32-unknown-unknown'

  "$RUSTUP" target add "$target" >/dev/null
}
