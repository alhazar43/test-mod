#!/bin/bash
#SBATCH -J gpu-probe
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=gpu_probe_%j.out
#SBATCH --error=gpu_probe_%j.err

set -euo pipefail

echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"

which nvidia-smi
nvidia-smi
