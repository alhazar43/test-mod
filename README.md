This repo is a UT HPC GPU benchmark and Slurm adaptation template.

It has two purposes:

1. Benchmark available GPUs on the UT HPC cluster.
2. Provide a minimal pattern for adapting a local PyTorch training project to
   run under Slurm.

Do not run GPU-heavy work directly on `hpc-head1` or `hpc-head2`. Use the login
node to edit files, install Python packages, submit jobs, and inspect results.

## Files

- `jobs/gpu_probe.sbatch`: checks that a Slurm GPU allocation can see
  `nvidia-smi`.
- `jobs/utwente_gpu_stress.sbatch`: raw matrix-multiply GPU benchmark.
- `jobs/utwente_dl_train.sbatch`: synthetic transformer training benchmark.
- `scripts/gpu_stress.py`: raw tensor throughput workload.
- `scripts/dl_train_benchmark.py`: forward/backward/optimizer training workload.
- `scripts/submit_scaling_experiment.sh`: submits matched 1-GPU and 2-GPU
  matrix jobs.
- `scripts/submit_dl_repeats.sh`: submits repeated A100 vs L40S training jobs.
- `scripts/compare_results.py`: compares individual `summary.json` files.
- `scripts/summarize_results.py`: aggregates repeated runs with mean/std/SEM.
- `requirements.txt`: CUDA-enabled PyTorch wheel dependency.

## Setup On HPC

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
```

For later updates:

```bash
cd ~/test-mod
git pull
. .venv/bin/activate
```

## Probe GPU Access

`nvidia-smi` may not exist on the login node. Check it inside a Slurm GPU job:

```bash
sbatch jobs/gpu_probe.sbatch
tail -f gpu_probe_<jobid>.out
```

Check queue and completed jobs:

```bash
squeue -u yuanw
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList,AllocTRES
```

## Benchmark: Raw GPU Throughput

Single GPU:

```bash
EXPERIMENT_NAME=gpu-model-comparison GPU_STRESS_SECONDS=300 GPU_STRESS_SIZE=8192 \
  sbatch --job-name gpu-a100-1 --gres=gpu:ampere:1 --constraint=a100 \
  jobs/utwente_gpu_stress.sbatch
```

Two GPUs:

```bash
EXPERIMENT_NAME=gpu-scaling-l40 GPU_STRESS_SECONDS=300 GPU_STRESS_SIZE=8192 \
  sbatch --job-name gpu-l40-2 --gres=gpu:lovelace:2 --constraint=l40 \
  --sockets-per-node=1 jobs/utwente_gpu_stress.sbatch
```

Compare results:

```bash
python scripts/compare_results.py results/gpu-model-comparison --markdown
```

## Benchmark: Example Deep Learning Training

Smoke test:

```bash
EXPERIMENT_NAME=dl-smoke-a100 DL_SECONDS=120 \
  sbatch --job-name dl-a100-smoke --gres=gpu:ampere:1 --constraint=a100 \
  jobs/utwente_dl_train.sbatch
```

Single-run model comparison:

```bash
EXPERIMENT_NAME=dl-model-comparison DL_SECONDS=300 \
  sbatch --job-name dl-a100-1 --gres=gpu:ampere:1 --constraint=a100 \
  jobs/utwente_dl_train.sbatch

EXPERIMENT_NAME=dl-model-comparison DL_SECONDS=300 \
  sbatch --job-name dl-l40s-1 --gres=gpu:lovelace:1 --constraint=l40s \
  jobs/utwente_dl_train.sbatch
```

Rigorous repeated comparison:

```bash
EXPERIMENT_NAME=dl-rigorous-a100-l40s REPEATS=3 \
  DL_SECONDS=900 DL_WARMUP_STEPS=20 DL_MEASURE_STEPS=200 \
  DL_BATCH_SIZE=8 DL_LAYERS=16 \
  bash scripts/submit_dl_repeats.sh
```

Summarize repeated runs:

```bash
python scripts/summarize_results.py results/dl-rigorous-a100-l40s --markdown
```

## Result Workflow

Each benchmark writes curated artifacts under:

```text
results/<experiment-name>/<jobid>_<jobname>/
```

Important files:

- `summary.json`: machine-readable benchmark result.
- `nvidia_smi.csv`: sampled GPU utilization, memory, power, temperature.
- `environment.txt`: Slurm, module, CUDA, and job environment.

Commit only result artifacts you want to study locally:

```bash
git add results/<experiment-name>
git commit -m "Add benchmark results"
git push origin main
```

Then locally:

```bash
git pull
python3 scripts/summarize_results.py results/<experiment-name> --markdown
```

Raw Slurm stdout/stderr files are ignored by default.

## GPU Selection

Your `code` account can use `main-gpu`. Check available GPUs:

```bash
sinfo -p main-gpu -o "%P %N %G %f %t"
```

Useful constraints observed on `main-gpu`:

- `--gres=gpu:ampere:1 --constraint=a100`
- `--gres=gpu:ampere:1 --constraint=a40`
- `--gres=gpu:ampere:1 --constraint=a4500`
- `--gres=gpu:lovelace:1 --constraint=l40s`
- `--gres=gpu:lovelace:1 --constraint=l40`
- `--gres=gpu:turing:1 --constraint=rtx-6000`

Other partitions may show H200NVL or RTX6000Pro GPUs, but your `code` account is
not permitted on those partitions unless the HPC admins add access.

Your `guest-research` QOS allows at most `gres/gpu=2` at once. If a job is
pending with `QOSMaxGRESPerUser`, wait for current GPU jobs to finish or cancel
one.

## Adaptation Manual

Use this section when turning any local GPU workload into an HPC Slurm job. This
can be model training, evaluation, inference, simulation, rendering, or a custom
analysis script. Environment and path details are intentionally left to the
project owner.

### 1. Make One Runnable Command

Your project should have one command that starts the workload:

```bash
python train.py --config configs/train.yaml
```

Other valid examples:

```bash
python evaluate.py --checkpoint checkpoints/model.pt --split test
python infer.py --input data/input --output outputs/run
python simulate.py --config configs/experiment.yaml
```

Keep local paths, dataset paths, and output paths configurable through CLI args,
config files, or environment variables.

### 2. Copy A Slurm Template

Start from one of the job templates and replace the benchmark command.

For training-like workloads, `jobs/utwente_dl_train.sbatch` is usually the
closest template. Replace:

```bash
"${PYTHON_BIN}" scripts/dl_train_benchmark.py ...
```

with your workload command:

```bash
"${PYTHON_BIN}" train.py --config configs/train.yaml
```

For lighter probes or non-training commands, `jobs/gpu_probe.sbatch` can be used
as a minimal starting point.

Keep the useful Slurm wrapper pieces:

```bash
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

module load "${PYTHON_MODULE:-python/3.10.7}"
module load "${CUDA_MODULE:-nvidia/cuda-11.0}"
. .venv/bin/activate

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
module list
nvidia-smi
```

### 3. Write Results Predictably

Use one run directory per job:

```bash
RUN_DIR="${RESULTS_ROOT:-results}/${EXPERIMENT_NAME:-manual}/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
mkdir -p "${RUN_DIR}"
```

Recommended artifacts:

- `environment.txt`: modules, job ID, node, CUDA visibility.
- `metrics.json` or `summary.json`: final metrics.
- workload logs, checkpoints, predictions, or analysis outputs as needed.

### 4. Make GPU Use Explicit

Slurm allocating a GPU does not automatically make code use it. Your code must
move model and tensors to CUDA:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

For multiple GPUs, the code must explicitly support parallelism, for example
DDP, FSDP, DeepSpeed, model parallelism, or domain decomposition. Requesting
`--gres=gpu:2` alone does not make ordinary single-GPU code faster.

### 5. Add Checkpoint/Resume Where Relevant

HPC jobs can time out, fail, or be cancelled. Long jobs should periodically save
state and support resume when the workload supports it:

```bash
python train.py --config configs/train.yaml --resume <checkpoint>
```

### 6. Scale Only After A Smoke Test

Recommended progression:

1. CPU/import check if useful.
2. Short 1-GPU Slurm job.
3. Full 1-GPU job.
4. Different GPU types.
5. Multi-GPU only after the code supports it.

### 7. Compare Like With Like

For benchmarks, keep workload parameters fixed across GPUs:

- same model/config
- same batch size
- same precision
- same number of measured steps
- same warmup policy
- same dataset/input pipeline

Then vary only GPU model or GPU count.
