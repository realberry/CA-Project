#!/usr/bin/env bash
set -u

sim_root="${HOME}/ca-assignment/Project/LLMSimulator"
project_root="${HOME}/ca-assignment/Project"
result_dir="${project_root}/day1"
config="${project_root}/duplex_smoke.yaml"

mkdir -p "${result_dir}"
cp "${config}" "${result_dir}/config.duplex-smoke.yaml"
date --iso-8601=seconds > "${result_dir}/duplex-smoke.started"
cd "${sim_root}/build" || exit 1
./run "${config}" > "${result_dir}/duplex-smoke.log" 2>&1
exit_code=$?
date --iso-8601=seconds > "${result_dir}/duplex-smoke.finished"
printf '%s\n' "${exit_code}" > "${result_dir}/duplex-smoke.status"
exit "${exit_code}"
