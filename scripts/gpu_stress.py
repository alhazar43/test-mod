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
        help="Torch device to use for single-device mode.",
    )
    parser.add_argument(
        "--devices",
        default=os.environ.get("GPU_STRESS_DEVICES", "auto"),
        help=(
            "Comma-separated CUDA device indexes, 'all', or 'auto'. "
            "Auto uses all allocated CUDA devices, or CPU when CUDA is unavailable."
        ),
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


def resolve_devices(args: argparse.Namespace) -> list[torch.device]:
    if args.devices == "auto":
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]
        return [torch.device("cpu")]

    if args.devices == "all":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Did Slurm allocate a GPU?")
        return [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]

    if "," in args.devices:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Did Slurm allocate a GPU?")
        return [torch.device(f"cuda:{index.strip()}") for index in args.devices.split(",")]

    if args.devices.isdigit():
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Did Slurm allocate a GPU?")
        return [torch.device(f"cuda:{args.devices}")]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Did Slurm allocate a GPU?")
    return [torch.device(args.device)]


def main() -> int:
    args = parse_args()

    devices = resolve_devices(args)
    dtype = dtype_from_name(args.dtype)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"devices={','.join(str(device) for device in devices)}", flush=True)
    for device in devices:
        if device.type != "cuda":
            continue
        torch.cuda.reset_peak_memory_stats(device)
        props = torch.cuda.get_device_properties(device)
        print(f"gpu[{device.index}]={props.name}", flush=True)
        print(
            f"gpu_index={device.index} "
            f"gpu_capability={props.major}.{props.minor} "
            f"total_memory={props.total_memory / 1024**3:.2f}GiB "
            f"multiprocessors={props.multi_processor_count}",
            flush=True,
        )
    print(
        f"seconds={args.seconds} size={args.size} dtype={args.dtype}",
        flush=True,
    )

    tensors = []
    for device in devices:
        tensors.append(
            (
                device,
                torch.randn((args.size, args.size), device=device, dtype=dtype),
                torch.randn((args.size, args.size), device=device, dtype=dtype),
                torch.empty((args.size, args.size), device=device, dtype=dtype),
            )
        )

    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    matrix_gib = args.size * args.size * bytes_per_element / 1024**3
    working_set_gib = matrix_gib * 3 * len(devices)
    print(
        f"matrix_memory_per_tensor={matrix_gib:.2f}GiB "
        f"working_set_total={working_set_gib:.2f}GiB",
        flush=True,
    )

    start = time.monotonic()
    deadline = start + args.seconds
    iterations = 0
    matmul_ops_per_device = 2 * args.size**3

    while time.monotonic() < deadline:
        for _, a, b, c in tensors:
            if iterations % 2 == 0:
                torch.matmul(a, b, out=c)
            else:
                torch.matmul(b, a, out=c)
        iterations += 1

        if args.log_every > 0 and iterations % args.log_every == 0:
            cuda_devices = [device for device in devices if device.type == "cuda"]
            if cuda_devices:
                for device in cuda_devices:
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic() - start
                tflops = (
                    iterations * matmul_ops_per_device * len(devices) / elapsed / 1e12
                )
                memory = []
                for device in cuda_devices:
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    reserved = torch.cuda.memory_reserved(device) / 1024**3
                    memory.append(
                        f"cuda:{device.index}=alloc:{allocated:.2f}GiB,res:{reserved:.2f}GiB"
                    )
                print(
                    f"iter={iterations} elapsed={elapsed:.1f}s "
                    f"avg_tflops={tflops:.2f} "
                    f"memory=[{';'.join(memory)}]",
                    flush=True,
                )
            else:
                print(
                    f"iter={iterations} elapsed={time.monotonic() - start:.1f}s",
                    flush=True,
                )

    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    elapsed = time.monotonic() - start
    tflops = (
        iterations * matmul_ops_per_device * len(devices) / elapsed / 1e12
        if elapsed > 0
        else 0.0
    )
    print(
        f"done iterations={iterations} elapsed={elapsed:.1f}s avg_tflops={tflops:.2f}",
        flush=True,
    )
    for device in devices:
        if device.type != "cuda":
            continue
        print(
            f"cuda:{device.index} "
            f"peak_allocated={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GiB "
            f"peak_reserved={torch.cuda.max_memory_reserved(device) / 1024**3:.2f}GiB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
