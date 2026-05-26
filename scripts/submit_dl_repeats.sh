#!/bin/bash
set -euo pipefail

experiment_name="${EXPERIMENT_NAME:-dl-rigorous-$(date +%Y%m%d-%H%M%S)}"
partition="${PARTITION:-main-gpu}"
repeats="${REPEATS:-5}"
seconds="${DL_SECONDS:-900}"
warmup_steps="${DL_WARMUP_STEPS:-20}"
measure_steps="${DL_MEASURE_STEPS:-200}"
batch_size="${DL_BATCH_SIZE:-8}"
seq_len="${DL_SEQ_LEN:-1024}"
d_model="${DL_D_MODEL:-1024}"
layers="${DL_LAYERS:-16}"
heads="${DL_HEADS:-16}"
dtype="${DL_DTYPE:-float16}"
results_root="${RESULTS_ROOT:-results}"

mkdir -p "${results_root}/${experiment_name}"
manifest="${results_root}/${experiment_name}/manifest.tsv"
printf "label\tjob_id\tgres\tconstraint\trepeat\tbatch_size\tseq_len\td_model\tlayers\twarmup_steps\tmeasure_steps\tdtype\n" > "${manifest}"

submit_run() {
  local label="$1"
  local gres="$2"
  local constraint="$3"
  local repeat="$4"
  local job_id

  job_id="$(
    sbatch --parsable \
      --job-name "${label}-r${repeat}" \
      --partition "${partition}" \
      --gres "${gres}" \
      --constraint "${constraint}" \
      --export "ALL,EXPERIMENT_NAME=${experiment_name},RESULTS_ROOT=${results_root},DL_SECONDS=${seconds},DL_WARMUP_STEPS=${warmup_steps},DL_MEASURE_STEPS=${measure_steps},DL_BATCH_SIZE=${batch_size},DL_SEQ_LEN=${seq_len},DL_D_MODEL=${d_model},DL_LAYERS=${layers},DL_HEADS=${heads},DL_DTYPE=${dtype},DL_DEVICES=auto" \
      jobs/utwente_dl_train.sbatch
  )"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${label}" "${job_id}" "${gres}" "${constraint}" "${repeat}" \
    "${batch_size}" "${seq_len}" "${d_model}" "${layers}" \
    "${warmup_steps}" "${measure_steps}" "${dtype}" >> "${manifest}"
  echo "submitted ${label} repeat ${repeat}: ${job_id}"
}

for repeat in $(seq 1 "${repeats}"); do
  submit_run "dl-a100" "gpu:ampere:1" "a100" "${repeat}"
  submit_run "dl-l40s" "gpu:lovelace:1" "l40s" "${repeat}"
done

echo "experiment=${experiment_name}"
echo "manifest=${manifest}"
echo "watch with: squeue -u ${USER}"
