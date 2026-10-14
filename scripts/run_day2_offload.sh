#!/usr/bin/env bash
set -euo pipefail

root="/home/verso/ca-assignment/Project/SSD-offloading/LLMSimulator"
configs="$root/day2_configs"
results="$root/results/day2"
raw="$results/raw"

mkdir -p "$raw"
original_config="$(mktemp)"
cp "$root/config.yaml" "$original_config"
restore_config() {
  cp "$original_config" "$root/config.yaml"
  rm -f "$original_config"
}
trap restore_config EXIT
cd "$root"

run_case() {
  local case_name="$1"
  local config_name="$2"
  cp "$configs/$config_name" "$root/config.yaml"
  cp "$root/config.yaml" "$results/config.${case_name}.yaml"
  printf '%s\n' "$(date -Is) start $case_name" | tee "$results/${case_name}.status"
  "$root/build/run" > "$results/${case_name}.log" 2>&1
  printf '%s\n' "$(date -Is) complete $case_name" | tee -a "$results/${case_name}.status"
}

run_case baseline day2_baseline.yaml
run_case offchip_compute day2_offchip_compute.yaml
run_case offchip_load day2_offchip_load.yaml

touch "$results/SUCCESS"
