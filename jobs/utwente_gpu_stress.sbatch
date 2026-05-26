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

cd "${SLURM_SUBMIT_DIR}"

# Adjust this to the CUDA/Python modules available on the cluster.
# Run `module avail cuda` and `module avail python` on the login node first.
# module load nvidia/cuda-12.1
# module load python/3.10

python3 -c "import torch; print('torch', torch.__version__)"
nvidia-smi

gpu_monitor_file="gpu_${SLURM_JOB_ID}_nvidia_smi.csv"
monitor_gpu() {
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
    --format=csv,nounits

  while true; do
    sleep "${GPU_MONITOR_INTERVAL:-5}"
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
      --format=csv,nounits,noheader
  done
}

monitor_gpu > "${gpu_monitor_file}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

python3 scripts/gpu_stress.py \
  --seconds "${GPU_STRESS_SECONDS:-600}" \
  --size "${GPU_STRESS_SIZE:-8192}" \
  --dtype "${GPU_STRESS_DTYPE:-float16}" \
  --log-every "${GPU_STRESS_LOG_EVERY:-10}"

kill "${monitor_pid}" 2>/dev/null || true
trap - EXIT

echo "GPU monitor samples: ${gpu_monitor_file}"
tail -n 10 "${gpu_monitor_file}"
