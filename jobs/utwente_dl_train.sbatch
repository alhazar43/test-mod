#!/bin/bash
#SBATCH -J dl-train
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=dl_%j.out
#SBATCH --error=dl_%j.err

set -euo pipefail

echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "PYTHON_MODULE=${PYTHON_MODULE:-python/3.10.7}"
echo "CUDA_MODULE=${CUDA_MODULE:-nvidia/cuda-11.0}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME:-dl-manual}"

cd "${SLURM_SUBMIT_DIR}"

module load "${PYTHON_MODULE:-python/3.10.7}"
module load "${CUDA_MODULE:-nvidia/cuda-11.0}"
module list

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

result_dir="${RESULTS_ROOT:-results}/${EXPERIMENT_NAME:-dl-manual}/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "${result_dir}"

gpu_monitor_file="${result_dir}/nvidia_smi.csv"
summary_file="${result_dir}/summary.json"
env_file="${result_dir}/environment.txt"

{
  echo "Date: $(date)"
  echo "Node: $(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  echo "DL_SECONDS=${DL_SECONDS:-300}"
  echo "DL_BATCH_SIZE=${DL_BATCH_SIZE:-4}"
  echo "DL_SEQ_LEN=${DL_SEQ_LEN:-1024}"
  echo "DL_D_MODEL=${DL_D_MODEL:-1024}"
  echo "DL_LAYERS=${DL_LAYERS:-12}"
  echo "DL_HEADS=${DL_HEADS:-16}"
  echo "DL_DTYPE=${DL_DTYPE:-float16}"
  module list
} > "${env_file}" 2>&1

gpu_monitor_args=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  gpu_monitor_args=(-i "${CUDA_VISIBLE_DEVICES}")
fi

monitor_gpu() {
  nvidia-smi "${gpu_monitor_args[@]}" \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
    --format=csv,nounits
  while true; do
    sleep "${GPU_MONITOR_INTERVAL:-5}"
    nvidia-smi "${gpu_monitor_args[@]}" \
      --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
      --format=csv,nounits,noheader
  done
}

"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__); print('torch_cuda_runtime', torch.version.cuda); print('torch_cuda_available', torch.cuda.is_available())"
nvidia-smi

monitor_gpu > "${gpu_monitor_file}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

"${PYTHON_BIN}" scripts/dl_train_benchmark.py \
  --seconds "${DL_SECONDS:-300}" \
  --batch-size "${DL_BATCH_SIZE:-4}" \
  --seq-len "${DL_SEQ_LEN:-1024}" \
  --vocab-size "${DL_VOCAB_SIZE:-32768}" \
  --d-model "${DL_D_MODEL:-1024}" \
  --layers "${DL_LAYERS:-12}" \
  --heads "${DL_HEADS:-16}" \
  --mlp-ratio "${DL_MLP_RATIO:-4}" \
  --dtype "${DL_DTYPE:-float16}" \
  --devices "${DL_DEVICES:-auto}" \
  --log-every "${DL_LOG_EVERY:-10}" \
  --output-json "${summary_file}"

kill "${monitor_pid}" 2>/dev/null || true
trap - EXIT

echo "GPU monitor samples: ${gpu_monitor_file}"
tail -n 10 "${gpu_monitor_file}"
echo "Result directory: ${result_dir}"
echo "Summary JSON: ${summary_file}"
