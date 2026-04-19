#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"

export PATH="$HOME/.cargo/bin:$PATH"

browser="${1:-chrome}"
headless="${BROWSER_HEADLESS:-1}"
shift || true

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack is required. Install it with: cargo install wasm-pack --locked" >&2
  exit 1
fi

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
  exec wasm-pack test "${args[@]}" -- "$@"
fi

exec wasm-pack test "${args[@]}"
