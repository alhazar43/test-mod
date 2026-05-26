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

Inspect available Python modules:

```bash
module avail python
```

At the time this was written, the login node exposed `python/3.7.3`,
`python/3.9.9`, and `python/3.10.7`, but no CUDA module. That is fine for this
job: CUDA-enabled PyTorch wheels bundle the CUDA runtime, and the allocated GPU
compute node provides the NVIDIA driver.

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
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs PyTorch from the official CUDA 11.8 wheel index.
This gives PyTorch a CUDA runtime without needing a system CUDA module. The
official PyTorch installer documents this selector-based CUDA wheel workflow for
Linux pip installs: https://pytorch.org/get-started/locally/

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

## Tuning

Tune the run without editing the Slurm file:

```bash
GPU_STRESS_SECONDS=1800 GPU_STRESS_SIZE=12288 sbatch jobs/utwente_gpu_stress.sbatch
```

Larger `GPU_STRESS_SIZE` values use more GPU memory and compute. If the job runs
out of memory, reduce the size.

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
