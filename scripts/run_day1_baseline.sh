#!/usr/bin/env bash
set -u

sim_root="${HOME}/ca-assignment/Project/LLMSimulator"
result_dir="${HOME}/ca-assignment/Project/day1"

mkdir -p "${result_dir}"
cp "${sim_root}/config.yaml" "${result_dir}/config.baseline.yaml"
git -C "${sim_root}" rev-parse HEAD > "${result_dir}/llmsimulator.commit"
git -C "${sim_root}" submodule status > "${result_dir}/submodules.txt"
g++ --version > "${result_dir}/compiler.txt"

date --iso-8601=seconds > "${result_dir}/baseline.started"
cd "${sim_root}/build" || exit 1
./run > "${result_dir}/baseline.log" 2>&1
exit_code=$?
date --iso-8601=seconds > "${result_dir}/baseline.finished"
printf '%s\n' "${exit_code}" > "${result_dir}/baseline.status"
exit "${exit_code}"
