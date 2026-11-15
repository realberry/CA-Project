#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$repo_root/data/day3/c4_results.first24MiB.jsonl.part"
temporary="${target}.tmp"
source_url="https://huggingface.co/datasets/allenai/analysis_mixtral/resolve/main/c4_results.jsonl"
expected_bytes=25165824
expected_sha256="4cb79ab27c967d008878981859f5883243c5890556fbaf4c85afc60f2797c6fe"

mkdir -p "$(dirname "$target")"
trap 'rm -f "$temporary"' EXIT

# The server normally honors the byte-range request. If it returns the complete
# file instead, head still retains only the deterministic 24 MiB prefix.
curl -L --fail --retry 3 --range 0-25165823 "$source_url" -o "$temporary"
actual_bytes="$(wc -c < "$temporary")"
if (( actual_bytes < expected_bytes )); then
  printf 'Download is too short: expected at least %d bytes, got %d.\n' \
    "$expected_bytes" "$actual_bytes" >&2
  exit 1
fi

head -c "$expected_bytes" "$temporary" > "$target"
printf '%s  %s\n' "$expected_sha256" "$target" | sha256sum --check --status
printf 'Trace ready: %s (%d bytes, SHA-256 verified)\n' "$target" "$expected_bytes"
