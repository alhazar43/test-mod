#!/usr/bin/env python3
"""Run a configurable CUDA matrix-multiply workload for Slurm smoke tests."""

from __future__ import annotations

import argparse
import os
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sustained GPU load with repeated matrix multiplications."
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=int(os.environ.get("GPU_STRESS_SECONDS", "300")),
        help="How long to run the workload.",
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
        help="Tensor dtype used for the matrix multiplication.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("GPU_STRESS_DEVICE", "cuda"),
        help="Torch device to use. Leave as cuda under Slurm GPU allocation.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=int(os.environ.get("GPU_STRESS_LOG_EVERY", "10")),
        help="Print progress every N matrix multiplications.",
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


def main() -> int:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Did Slurm allocate a GPU?")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        props = torch.cuda.get_device_properties(device)
        print(f"gpu={props.name}", flush=True)
        print(
            f"gpu_capability={props.major}.{props.minor} "
            f"total_memory={props.total_memory / 1024**3:.2f}GiB "
            f"multiprocessors={props.multi_processor_count}",
            flush=True,
        )
    print(
        f"seconds={args.seconds} size={args.size} dtype={args.dtype}",
        flush=True,
    )

    a = torch.randn((args.size, args.size), device=device, dtype=dtype)
    b = torch.randn((args.size, args.size), device=device, dtype=dtype)
    c = torch.empty((args.size, args.size), device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()

    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    matrix_gib = args.size * args.size * bytes_per_element / 1024**3
    working_set_gib = matrix_gib * 3
    print(
        f"matrix_memory={matrix_gib:.2f}GiB working_set={working_set_gib:.2f}GiB",
        flush=True,
    )

    start = time.monotonic()
    deadline = start + args.seconds
    iterations = 0
    matmul_ops = 2 * args.size**3

    while time.monotonic() < deadline:
        if iterations % 2 == 0:
            torch.matmul(a, b, out=c)
        else:
            torch.matmul(b, a, out=c)
        iterations += 1

        if args.log_every > 0 and iterations % args.log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                elapsed = time.monotonic() - start
                tflops = iterations * matmul_ops / elapsed / 1e12
                print(
                    f"iter={iterations} elapsed={elapsed:.1f}s "
                    f"avg_tflops={tflops:.2f} "
                    f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB",
                    flush=True,
                )
            else:
                print(
                    f"iter={iterations} elapsed={time.monotonic() - start:.1f}s",
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.monotonic() - start
    tflops = iterations * matmul_ops / elapsed / 1e12 if elapsed > 0 else 0.0
    print(
        f"done iterations={iterations} elapsed={elapsed:.1f}s avg_tflops={tflops:.2f}",
        flush=True,
    )
    if device.type == "cuda":
        print(
            f"peak_allocated={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GiB "
            f"peak_reserved={torch.cuda.max_memory_reserved(device) / 1024**3:.2f}GiB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
