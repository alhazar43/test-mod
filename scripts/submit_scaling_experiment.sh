#!/bin/bash
set -euo pipefail

experiment_name="${EXPERIMENT_NAME:-gpu-scaling-$(date +%Y%m%d-%H%M%S)}"
partition="${PARTITION:-main-gpu}"
gpu_family="${GPU_FAMILY:-lovelace}"
seconds="${GPU_STRESS_SECONDS:-300}"
size="${GPU_STRESS_SIZE:-8192}"
dtype="${GPU_STRESS_DTYPE:-float16}"
monitor_interval="${GPU_MONITOR_INTERVAL:-5}"
log_every="${GPU_STRESS_LOG_EVERY:-100}"
results_root="${RESULTS_ROOT:-results}"

mkdir -p "${results_root}/${experiment_name}"
manifest="${results_root}/${experiment_name}/manifest.tsv"
printf "label\tjob_id\tgres\tseconds\tsize\tdtype\n" > "${manifest}"

submit_run() {
  local label="$1"
  local gres="$2"
  local job_id

  job_id="$(
    sbatch --parsable \
      --job-name "${label}" \
      --partition "${partition}" \
      --gres "${gres}" \
      --export "ALL,EXPERIMENT_NAME=${experiment_name},RESULTS_ROOT=${results_root},GPU_STRESS_SECONDS=${seconds},GPU_STRESS_SIZE=${size},GPU_STRESS_DTYPE=${dtype},GPU_STRESS_DEVICES=auto,GPU_MONITOR_INTERVAL=${monitor_interval},GPU_STRESS_LOG_EVERY=${log_every}" \
      jobs/utwente_gpu_stress.sbatch
  )"

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${job_id}" "${gres}" "${seconds}" "${size}" "${dtype}" >> "${manifest}"
  echo "submitted ${label}: ${job_id} (${gres})"
}

submit_run "gpu-scale-1" "gpu:${gpu_family}:1"
submit_run "gpu-scale-2" "gpu:${gpu_family}:2"

echo "experiment=${experiment_name}"
echo "manifest=${manifest}"
echo "watch with: squeue -u ${USER}"
