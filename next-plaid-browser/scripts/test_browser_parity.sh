#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"
source "$workspace_root/scripts/wasm_env.sh"

resolve_wasm_toolchain
require_tool "${WASM_PACK:-}" "wasm-pack" "Install it with: cargo install wasm-pack --locked"
ensure_wasm_target

browser="${1:-chrome}"
headless="${BROWSER_HEADLESS:-1}"
shift || true

cd "$workspace_root"

args=()
if [[ "$headless" == "1" ]]; then
  args+=(--headless)
fi

case "$browser" in
  chrome|firefox|safari)
    args+=("--$browser")
    ;;
  all)
    args+=(--chrome --firefox --safari)
    ;;
  *)
    echo "usage: $0 [chrome|firefox|safari|all] [cargo-test-args...]" >&2
    exit 1
    ;;
esac

args+=("crates/next-plaid-browser-wasm")

if [[ "$#" -gt 0 ]]; then
  exec "$WASM_PACK" test "${args[@]}" -- "$@"
fi

exec "$WASM_PACK" test "${args[@]}"
