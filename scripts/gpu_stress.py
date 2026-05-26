#!/usr/bin/env python3
"""Run a configurable GPU scaling workload and write a machine-readable summary."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_module
import time
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sustained GPU load with repeated matrix multiplications."
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=int(os.environ.get("GPU_STRESS_SECONDS", "300")),
        help="How long each worker should run.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=int(os.environ.get("GPU_STRESS_SIZE", "8192")),
        help="Square matrix size. Increase for more GPU memory/load.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default=os.environ.get("GPU_STRESS_DTYPE", "float16"),
        help="Tensor dtype used for matrix multiplication.",
    )
    parser.add_argument(
        "--devices",
        default=os.environ.get("GPU_STRESS_DEVICES", "auto"),
        help="Comma-separated CUDA indexes, 'all', 'auto', or 'cpu'.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=int(os.environ.get("GPU_STRESS_LOG_EVERY", "100")),
        help="Print worker progress every N matrix multiplications.",
    )
    parser.add_argument(
        "--output-json",
        default=os.environ.get("GPU_STRESS_OUTPUT_JSON"),
        help="Optional path for a JSON run summary.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_device_specs(devices_arg: str) -> list[str]:
    if devices_arg == "cpu":
        return ["cpu"]

    if devices_arg in {"auto", "all"}:
        if torch.cuda.is_available():
            return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if devices_arg == "all":
            raise SystemExit("CUDA is not available. Did Slurm allocate a GPU?")
        return ["cpu"]

    if "," in devices_arg:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Did Slurm allocate a GPU?")
        return [f"cuda:{index.strip()}" for index in devices_arg.split(",") if index.strip()]

    if devices_arg.isdigit():
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Did Slurm allocate a GPU?")
        return [f"cuda:{devices_arg}"]

    if devices_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Did Slurm allocate a GPU?")
        return [devices_arg]

    raise SystemExit(f"Unsupported --devices value: {devices_arg}")


def device_metadata(device_spec: str) -> dict[str, Any]:
    device = torch.device(device_spec)
    if device.type != "cuda":
        return {"device": device_spec, "type": "cpu"}

    props = torch.cuda.get_device_properties(device)
    return {
        "device": device_spec,
        "type": "cuda",
        "logical_index": device.index,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gib": props.total_memory / 1024**3,
        "multiprocessors": props.multi_processor_count,
    }


def run_worker(
    worker_id: int,
    device_spec: str,
    seconds: int,
    size: int,
    dtype_name: str,
    log_every: int,
    result_queue: mp.Queue,
) -> None:
    device = torch.device(device_spec)
    dtype = dtype_from_name(dtype_name)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)
    c = torch.empty((size, size), device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.monotonic()
    deadline = start + seconds
    iterations = 0
    matmul_ops = 2 * size**3

    while time.monotonic() < deadline:
        if iterations % 2 == 0:
            torch.matmul(a, b, out=c)
        else:
            torch.matmul(b, a, out=c)
        iterations += 1

        if log_every > 0 and iterations % log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - start
            tflops = iterations * matmul_ops / elapsed / 1e12 if elapsed > 0 else 0.0
            print(
                f"worker={worker_id} device={device_spec} iter={iterations} "
                f"elapsed={elapsed:.1f}s avg_tflops={tflops:.2f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.monotonic() - start
    tflops = iterations * matmul_ops / elapsed / 1e12 if elapsed > 0 else 0.0
    result: dict[str, Any] = {
        "worker_id": worker_id,
        "device": device_spec,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "avg_tflops": tflops,
    }

    if device.type == "cuda":
        result.update(
            {
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            }
        )

    result_queue.put(result)


def write_summary(path: str | None, summary: dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    device_specs = resolve_device_specs(args.devices)
    bytes_per_element = torch.empty((), dtype=dtype_from_name(args.dtype)).element_size()
    matrix_gib = args.size * args.size * bytes_per_element / 1024**3
    working_set_gib = matrix_gib * 3 * len(device_specs)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"devices={','.join(device_specs)}", flush=True)
    print(
        f"seconds={args.seconds} size={args.size} dtype={args.dtype} "
        f"workers={len(device_specs)}",
        flush=True,
    )
    print(
        f"matrix_memory_per_tensor={matrix_gib:.2f}GiB "
        f"working_set_total={working_set_gib:.2f}GiB",
        flush=True,
    )

    metadata = [device_metadata(device_spec) for device_spec in device_specs]
    for item in metadata:
        print(f"device_metadata={json.dumps(item, sort_keys=True)}", flush=True)

    wall_start = time.monotonic()

    if len(device_specs) == 1 and device_specs[0] == "cpu":
        result_queue: mp.Queue = mp.Queue()
        run_worker(0, "cpu", args.seconds, args.size, args.dtype, args.log_every, result_queue)
        results = [result_queue.get()]
    else:
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=run_worker,
                args=(
                    index,
                    device_spec,
                    args.seconds,
                    args.size,
                    args.dtype,
                    args.log_every,
                    result_queue,
                ),
            )
            for index, device_spec in enumerate(device_specs)
        ]

        for process in processes:
            process.start()

        results = []
        while len(results) < len(processes):
            try:
                results.append(result_queue.get(timeout=5))
                continue
            except queue_module.Empty:
                failed = [process.exitcode for process in processes if process.exitcode]
                if failed:
                    for process in processes:
                        process.join(timeout=1)
                    raise SystemExit(f"Worker process failed with exit code {failed[0]}")

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise SystemExit(f"Worker process failed with exit code {process.exitcode}")

    wall_elapsed = time.monotonic() - wall_start
    results = sorted(results, key=lambda result: result["worker_id"])
    aggregate_tflops = sum(result["avg_tflops"] for result in results)

    summary = {
        "schema_version": 1,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seconds": args.seconds,
        "size": args.size,
        "dtype": args.dtype,
        "device_count": len(device_specs),
        "devices": metadata,
        "matrix_memory_per_tensor_gib": matrix_gib,
        "working_set_total_gib": working_set_gib,
        "wall_elapsed_seconds": wall_elapsed,
        "aggregate_tflops": aggregate_tflops,
        "workers": results,
    }

    print(
        f"done workers={len(device_specs)} wall_elapsed={wall_elapsed:.1f}s "
        f"aggregate_tflops={aggregate_tflops:.2f}",
        flush=True,
    )
    for result in results:
        print(
            f"worker={result['worker_id']} device={result['device']} "
            f"iterations={result['iterations']} avg_tflops={result['avg_tflops']:.2f}",
            flush=True,
        )

    write_summary(args.output_json, summary)
    if args.output_json:
        print(f"summary_json={args.output_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
