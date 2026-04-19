#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"

export PATH="$HOME/.cargo/bin:$PATH"

browser="${1:-safari}"
headless="${BROWSER_HEADLESS:-1}"

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack is required. Install it with: cargo install wasm-pack --locked" >&2
  exit 1
fi

cd "$workspace_root"

args=()
if [[ "$headless" == "1" ]]; then
  args+=(--headless)
fi
args+=("--$browser" "crates/next-plaid-browser-wasm")

exec wasm-pack test "${args[@]}"
