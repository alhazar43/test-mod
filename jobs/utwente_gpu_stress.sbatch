#!/bin/bash
#SBATCH -J test-gpu
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=gpu_%j.out
#SBATCH --error=gpu_%j.err

set -euo pipefail

echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL:-5}"
echo "PYTHON_MODULE=${PYTHON_MODULE:-python/3.10.7}"
echo "CUDA_MODULE=${CUDA_MODULE:-nvidia/cuda-11.0}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME:-manual}"

cd "${SLURM_SUBMIT_DIR}"

module load "${PYTHON_MODULE:-python/3.10.7}"
module load "${CUDA_MODULE:-nvidia/cuda-11.0}"
module list

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "Python: $(${PYTHON_BIN} --version)"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__); print('torch_cuda_runtime', torch.version.cuda); print('torch_cuda_available', torch.cuda.is_available())"
nvidia-smi

result_dir="${RESULTS_ROOT:-results}/${EXPERIMENT_NAME:-manual}/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "${result_dir}"

gpu_monitor_file="${result_dir}/nvidia_smi.csv"
summary_file="${result_dir}/summary.json"
env_file="${result_dir}/environment.txt"

{
  echo "Date: $(date)"
  echo "Node: $(hostname)"
  echo "Submit dir: ${SLURM_SUBMIT_DIR}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
  echo "SLURM_GPUS=${SLURM_GPUS:-}"
  echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  echo "PYTHON_MODULE=${PYTHON_MODULE:-python/3.10.7}"
  echo "CUDA_MODULE=${CUDA_MODULE:-nvidia/cuda-11.0}"
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

monitor_gpu > "${gpu_monitor_file}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

"${PYTHON_BIN}" scripts/gpu_stress.py \
  --seconds "${GPU_STRESS_SECONDS:-600}" \
  --size "${GPU_STRESS_SIZE:-8192}" \
  --dtype "${GPU_STRESS_DTYPE:-float16}" \
  --devices "${GPU_STRESS_DEVICES:-auto}" \
  --log-every "${GPU_STRESS_LOG_EVERY:-100}" \
  --output-json "${summary_file}"

kill "${monitor_pid}" 2>/dev/null || true
trap - EXIT

echo "GPU monitor samples: ${gpu_monitor_file}"
tail -n 10 "${gpu_monitor_file}"
echo "Result directory: ${result_dir}"
echo "Summary JSON: ${summary_file}"
