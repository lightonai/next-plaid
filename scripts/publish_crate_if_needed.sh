#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <crate-name>" >&2
  exit 2
fi

crate_name="$1"
tmp_output="$(mktemp)"
trap 'rm -f "$tmp_output"' EXIT

crate_version="$(cargo pkgid -p "$crate_name" | sed 's/.*[#@]//')"
crate_status="$(
  curl -sS -o /dev/null -w '%{http_code}' \
    "https://crates.io/api/v1/crates/${crate_name}/${crate_version}" || true
)"

if [[ "$crate_status" == "200" ]]; then
  echo "$crate_name $crate_version is already published; treating this as a successful no-op."
  exit 0
fi

cargo publish -p "$crate_name" --locked --dry-run

set +e
cargo publish -p "$crate_name" --locked >"$tmp_output" 2>&1
status="$?"
set -e

cat "$tmp_output"

if [[ "$status" -eq 0 ]]; then
  exit 0
fi

if grep -Eiq "already (uploaded|exists|published)|crate version .* is already uploaded" "$tmp_output"; then
  echo "$crate_name is already published; treating this as a successful no-op."
  exit 0
fi

exit "$status"
