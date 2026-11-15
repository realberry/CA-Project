#!/usr/bin/env bash
set -euo pipefail

SIM_ROOT="${SIM_ROOT:-/home/verso/ca-assignment/Project/SSD-offloading/LLMSimulator}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-/home/verso/ca-assignment/Project/final-prefetch}"
CONFIG_ROOT="$SUBMISSION_ROOT/configs/final_day2_single_gpu_batch1"
RESULT_ROOT="$SIM_ROOT/results/final_day2/single_gpu_batch1"

mkdir -p "$RESULT_ROOT/raw"
backup="$(mktemp)"
cp "$SIM_ROOT/config.yaml" "$backup"
restore_config() {
  cp "$backup" "$SIM_ROOT/config.yaml"
  rm -f "$backup"
}
trap restore_config EXIT

run_case() {
  local name="$1"
  local config="$2"
  cp "$CONFIG_ROOT/$config" "$SIM_ROOT/config.yaml"
  # Keep the submitted, tested YAML intact while making its output location
  # portable across Linux usernames and simulator checkout locations.
  sed -i "s|^  output_directory:.*|  output_directory: $RESULT_ROOT/raw|" \
    "$SIM_ROOT/config.yaml"
  (
    cd "$SIM_ROOT"
    ./build/run > "$RESULT_ROOT/$name.log" 2>&1
  )
  printf '%s complete\n' "$name"
}

rm -f "$RESULT_ROOT/SUCCESS"
run_case baseline single_gpu_batch1_baseline.yaml
run_case offchip_compute single_gpu_batch1_offchip_compute.yaml
run_case offchip_load single_gpu_batch1_offchip_load.yaml

test "$(find "$RESULT_ROOT/raw" -maxdepth 1 -type f -name '*.csv' | wc -l)" -eq 3
touch "$RESULT_ROOT/SUCCESS"
printf 'All final single-GPU runs completed.\n'
