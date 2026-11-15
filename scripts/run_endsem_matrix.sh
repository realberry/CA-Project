#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_ROOT="${SIM_ROOT:-$HOME/ca-assignment/Project/SSD-offloading/LLMSimulator}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-$SCRIPT_ROOT}"
CONFIG_ROOT="$SUBMISSION_ROOT/configs/final_day2_single_gpu_batch1"
RESULT_ROOT="${RESULT_ROOT:-$SIM_ROOT/results/final_endsem/hardware_matrix}"
BATCH_SIZES=(${BATCH_SIZES:-1 4 8})
CONTEXT_LENGTHS=(${CONTEXT_LENGTHS:-512 1024 2048})

mkdir -p "$RESULT_ROOT"
backup="$(mktemp)"
cp "$SIM_ROOT/config.yaml" "$backup"
restore_config() {
  cp "$backup" "$SIM_ROOT/config.yaml"
  rm -f "$backup"
}
trap restore_config EXIT

run_case() {
  local batch="$1"
  local context="$2"
  local name="$3"
  local template="$4"
  local case_root="$RESULT_ROOT/batch${batch}_context${context}"
  local raw="$case_root/raw"
  mkdir -p "$raw"
  cp "$CONFIG_ROOT/$template" "$SIM_ROOT/config.yaml"
  sed -i "s/^  max_batch_size:.*/  max_batch_size: $batch/" "$SIM_ROOT/config.yaml"
  sed -i "s/^  input_len:.*/  input_len: $context/" "$SIM_ROOT/config.yaml"
  sed -i "s|^  output_directory:.*|  output_directory: $raw|" "$SIM_ROOT/config.yaml"
  printf '%s start batch=%s context=%s case=%s\n' \
    "$(date --iso-8601=seconds)" "$batch" "$context" "$name" \
    >> "$RESULT_ROOT/run.status"
  (
    cd "$SIM_ROOT"
    ./build/run > "$case_root/$name.log" 2>&1
  )
  printf '%s complete batch=%s context=%s case=%s\n' \
    "$(date --iso-8601=seconds)" "$batch" "$context" "$name" \
    >> "$RESULT_ROOT/run.status"
}

rm -f "$RESULT_ROOT/SUCCESS" "$RESULT_ROOT/run.status"
for batch in "${BATCH_SIZES[@]}"; do
  for context in "${CONTEXT_LENGTHS[@]}"; do
    run_case "$batch" "$context" baseline single_gpu_batch1_baseline.yaml
    run_case "$batch" "$context" offchip_compute single_gpu_batch1_offchip_compute.yaml
    run_case "$batch" "$context" offchip_load single_gpu_batch1_offchip_load.yaml
  done
done

expected=$((${#BATCH_SIZES[@]} * ${#CONTEXT_LENGTHS[@]} * 3))
actual="$(find "$RESULT_ROOT" -path '*/raw/*.csv' -type f | wc -l)"
test "$actual" -eq "$expected"
touch "$RESULT_ROOT/SUCCESS"
printf 'Completed %d end-semester hardware calibration runs.\n' "$expected"
