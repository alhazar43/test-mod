This repo is for testing purposes.

## UT HPC GPU Stress Job

This repo contains a small PyTorch workload for testing GPU allocation and load
on the University of Twente HPC Slurm cluster. It runs repeated CUDA matrix
multiplications, prints benchmark-style progress, and records `nvidia-smi`
samples from the allocated GPU.

Use this only through Slurm. Log in to a head node, then submit the job; do not
run GPU-heavy work directly on a login/head node.

## Files

- `scripts/gpu_stress.py`: configurable PyTorch GPU workload.
- `jobs/utwente_gpu_stress.sbatch`: Slurm submission script for `main-gpu`.
- `jobs/gpu_probe.sbatch`: minimal Slurm job that checks `nvidia-smi` on a GPU
  node.
- `requirements.txt`: CUDA-enabled PyTorch wheel dependency.

## Quick Start With `rsync`

Copy the repo from your local machine to the UT HPC login node:

```bash
rsync -av ./ <ut-username>@hpc-head1.ewi.utwente.nl:~/test-mod/
```

Connect to the login node:

```bash
ssh <ut-username>@hpc-head1.ewi.utwente.nl
```

Inspect available modules:

```bash
module avail
module avail python
module avail nvidia/cuda
```

UT HPC software is managed through Environment Modules. Load the Python and CUDA
modules before creating environments or running jobs. The Slurm scripts load
these by default:

```bash
module load python/3.10.7
module load nvidia/cuda-11.0
```

Do not run `apt install`, and do not try to install NVIDIA drivers or system CUDA
on `hpc-head1`. You normally do not have admin rights there, and the login node
is not the GPU node. It is expected that this fails on the login node:

```bash
nvidia-smi
```

Check `nvidia-smi` through Slurm instead:

```bash
cd ~/test-mod
sbatch jobs/gpu_probe.sbatch
tail -f gpu_probe_<jobid>.out
```

Create the Python environment once on the login node:

```bash
cd ~/test-mod
module load python/3.10.7
module load nvidia/cuda-11.0
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs PyTorch from the official CUDA 11.8 wheel index.
The official PyTorch installer documents this selector-based CUDA wheel workflow
for Linux pip installs: https://pytorch.org/get-started/locally/

Override the module versions at submit time if needed:

```bash
PYTHON_MODULE=python/3.10.7 CUDA_MODULE=nvidia/cuda-11.0 sbatch jobs/utwente_gpu_stress.sbatch
```

Submit the GPU job:

```bash
cd ~/test-mod
sbatch jobs/utwente_gpu_stress.sbatch
```

## Quick Start With Git

If this repo is pushed to GitHub, use this workflow on the HPC login node:

```bash
ssh <ut-username>@hpc-head1.ewi.utwente.nl
git clone https://github.com/alhazar43/test-mod.git ~/test-mod
cd ~/test-mod
module load python/3.10.7
module load nvidia/cuda-11.0
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sbatch jobs/utwente_gpu_stress.sbatch
```

For later updates:

```bash
cd ~/test-mod
git pull
sbatch jobs/utwente_gpu_stress.sbatch
```

## Monitoring

Check queue state:

```bash
squeue -u "$USER"
```

Watch the Slurm job output:

```bash
tail -f gpu_<jobid>.out
```

The Slurm output contains:

- assigned node and `CUDA_VISIBLE_DEVICES`
- GPU model, memory, compute capability, Torch version, and Torch CUDA runtime
- progress lines with approximate average TFLOPS
- final peak PyTorch memory usage
- the last samples from `nvidia-smi`

The job also writes a CSV-style monitor file:

```bash
gpu_<jobid>_nvidia_smi.csv
```

That file includes timestamped utilization, memory, power, and temperature
samples from the allocated GPU.

Inspect the monitor file:

```bash
tail -n 20 gpu_<jobid>_nvidia_smi.csv
```

Use the UT dashboard at http://hpc-status.ewi.utwente.nl/slurm/ to inspect your
running job. For your user ID:

```bash
squeue -u yuanw
squeue -j <jobid>
```

For completed jobs:

```bash
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList,AllocTRES
```

## Tuning

Tune the run without editing the Slurm file:

```bash
GPU_STRESS_SECONDS=1800 GPU_STRESS_SIZE=12288 sbatch jobs/utwente_gpu_stress.sbatch
```

Larger `GPU_STRESS_SIZE` values use more GPU memory and compute. If the job runs
out of memory, reduce the size.

## Requesting More GPUs

The UT wiki documents GPU requests as `gpu[:family]:amount`. The current default
job requests one GPU:

```bash
#SBATCH --gres=gpu:1
```

Your current dashboard allocation showed `gres/gpu:lovelace=1` on `hpc-node04`.
To ask Slurm for two Lovelace GPUs, override the Slurm request at submit time:

```bash
sbatch --gres=gpu:lovelace:2 jobs/utwente_gpu_stress.sbatch
```

Or request two GPUs without specifying family:

```bash
sbatch --gres=gpu:2 jobs/utwente_gpu_stress.sbatch
```

The Python workload uses all CUDA devices exposed by Slurm by default:

```bash
GPU_STRESS_DEVICES=auto sbatch --gres=gpu:lovelace:2 jobs/utwente_gpu_stress.sbatch
```

You can explicitly choose visible device indexes:

```bash
GPU_STRESS_DEVICES=0,1 sbatch --gres=gpu:lovelace:2 jobs/utwente_gpu_stress.sbatch
```

If you request two GPUs that should use NVLink, the UT wiki says to force socket
binding:

```bash
sbatch --gres=gpu:lovelace:2 --sockets-per-node=1 jobs/utwente_gpu_stress.sbatch
```

Queue time may increase when requesting more GPUs because Slurm must find one
node with enough free GPUs.

Control the diagnostics cadence:

```bash
GPU_MONITOR_INTERVAL=2 GPU_STRESS_LOG_EVERY=5 sbatch jobs/utwente_gpu_stress.sbatch
```

Useful variables:

- `GPU_STRESS_SECONDS`: workload duration in seconds. Default from Slurm script:
  `600`.
- `GPU_STRESS_SIZE`: square matrix size. Default: `8192`.
- `GPU_STRESS_DTYPE`: one of `float16`, `float32`, or `bfloat16`. Default:
  `float16`.
- `GPU_STRESS_DEVICES`: `auto`, `all`, or comma-separated visible CUDA indexes
  such as `0,1`. Default: `auto`.
- `GPU_STRESS_LOG_EVERY`: print Python progress every N matrix multiplications.
  Default: `10`.
- `GPU_MONITOR_INTERVAL`: write `nvidia-smi` samples every N seconds. Default:
  `5`.

## Interpreting Results

A healthy GPU run should show:

- non-empty `CUDA_VISIBLE_DEVICES`
- a real GPU name and total GPU memory
- high `utilization.gpu` values in `gpu_<jobid>_nvidia_smi.csv`
- nonzero power draw during the workload
- progress lines with `avg_tflops`
- final `peak_allocated` and `peak_reserved` memory values

If the job fails with CUDA unavailable, Slurm probably did not allocate a GPU or
the installed PyTorch wheel is CPU-only. The Slurm log prints
`torch_cuda_runtime` and `torch_cuda_available`; `torch_cuda_available` should be
`True` inside the GPU job. If the job fails with an out-of-memory error, lower
`GPU_STRESS_SIZE`.

## Repository Workflow

Using a remote Git repo is useful for this kind of work: keep code, Slurm files,
and small config files in Git, then pull the same revision on the HPC login node.
Do not commit generated logs, datasets, checkpoints, or benchmark output.

Typical workflow:

```bash
# local machine
git add README.md requirements.txt jobs/ scripts/ .gitignore
git commit -m "Add UT HPC GPU stress job"
git push

# HPC login node
git clone https://github.com/alhazar43/test-mod.git ~/test-mod
cd ~/test-mod
sbatch jobs/utwente_gpu_stress.sbatch
```
