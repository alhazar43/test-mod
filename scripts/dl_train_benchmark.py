#!/usr/bin/env python3
"""Synthetic transformer training benchmark for GPU comparison experiments."""

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
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_ratio * d_model)
        self.fc2 = nn.Linear(mlp_ratio * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        x = residual + self.proj(attn)

        residual = x
        x = self.norm2(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        return residual + x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, mlp_ratio) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic transformer training benchmark.")
    parser.add_argument("--seconds", type=int, default=int(os.environ.get("DL_SECONDS", "300")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DL_BATCH_SIZE", "4")))
    parser.add_argument("--seq-len", type=int, default=int(os.environ.get("DL_SEQ_LEN", "1024")))
    parser.add_argument("--vocab-size", type=int, default=int(os.environ.get("DL_VOCAB_SIZE", "32768")))
    parser.add_argument("--d-model", type=int, default=int(os.environ.get("DL_D_MODEL", "1024")))
    parser.add_argument("--layers", type=int, default=int(os.environ.get("DL_LAYERS", "12")))
    parser.add_argument("--heads", type=int, default=int(os.environ.get("DL_HEADS", "16")))
    parser.add_argument("--mlp-ratio", type=int, default=int(os.environ.get("DL_MLP_RATIO", "4")))
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=os.environ.get("DL_DTYPE", "float16"))
    parser.add_argument("--devices", default=os.environ.get("DL_DEVICES", "auto"))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("DL_LOG_EVERY", "10")))
    parser.add_argument("--output-json", default=os.environ.get("DL_OUTPUT_JSON"))
    return parser.parse_args()


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


def amp_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return None


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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
    args: argparse.Namespace,
    result_queue: mp.Queue,
) -> None:
    device = torch.device(device_spec)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(1234 + worker_id)
    model = TinyGPT(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.dtype == "float16"))
    autocast_dtype = amp_dtype(args.dtype)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.monotonic()
    deadline = start + args.seconds
    steps = 0
    total_tokens = 0
    last_loss = 0.0

    while time.monotonic() < deadline:
        tokens = torch.randint(
            0,
            args.vocab_size,
            (args.batch_size, args.seq_len + 1),
            device=device,
            dtype=torch.long,
        )
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=(device.type == "cuda" and autocast_dtype is not None),
        ):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        steps += 1
        total_tokens += args.batch_size * args.seq_len
        last_loss = float(loss.detach().cpu())

        if args.log_every > 0 and steps % args.log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - start
            print(
                f"worker={worker_id} device={device_spec} steps={steps} "
                f"elapsed={elapsed:.1f}s tokens_per_sec={total_tokens / elapsed:.1f} "
                f"loss={last_loss:.4f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.monotonic() - start
    result: dict[str, Any] = {
        "worker_id": worker_id,
        "device": device_spec,
        "steps": steps,
        "tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0.0,
        "last_loss": last_loss,
        "parameters": count_parameters(model),
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
    metadata = [device_metadata(device_spec) for device_spec in device_specs]

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"devices={','.join(device_specs)}", flush=True)
    print(
        f"seconds={args.seconds} batch_size={args.batch_size} seq_len={args.seq_len} "
        f"d_model={args.d_model} layers={args.layers} heads={args.heads} dtype={args.dtype}",
        flush=True,
    )
    for item in metadata:
        print(f"device_metadata={json.dumps(item, sort_keys=True)}", flush=True)

    wall_start = time.monotonic()
    if len(device_specs) == 1 and device_specs[0] == "cpu":
        result_queue: mp.Queue = mp.Queue()
        run_worker(0, "cpu", args, result_queue)
        results = [result_queue.get()]
    else:
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        processes = [
            context.Process(target=run_worker, args=(index, device_spec, args, result_queue))
            for index, device_spec in enumerate(device_specs)
        ]
        for process in processes:
            process.start()

        results = []
        while len(results) < len(processes):
            try:
                results.append(result_queue.get(timeout=10))
            except queue_module.Empty:
                failed = [process.exitcode for process in processes if process.exitcode]
                if failed:
                    raise SystemExit(f"Worker process failed with exit code {failed[0]}")

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise SystemExit(f"Worker process failed with exit code {process.exitcode}")

    wall_elapsed = time.monotonic() - wall_start
    results = sorted(results, key=lambda result: result["worker_id"])
    tokens_per_second = sum(result["tokens_per_second"] for result in results)
    parameters = results[0]["parameters"] if results else 0

    summary = {
        "schema_version": 1,
        "benchmark_type": "synthetic_transformer_training",
        "metric_name": "tokens_per_second",
        "metric_value": tokens_per_second,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seconds": args.seconds,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "dtype": args.dtype,
        "device_count": len(device_specs),
        "parameters": parameters,
        "devices": metadata,
        "wall_elapsed_seconds": wall_elapsed,
        "tokens_per_second": tokens_per_second,
        "workers": results,
    }
    print(
        f"done workers={len(device_specs)} wall_elapsed={wall_elapsed:.1f}s "
        f"tokens_per_second={tokens_per_second:.1f} parameters={parameters}",
        flush=True,
    )
    write_summary(args.output_json, summary)
    if args.output_json:
        print(f"summary_json={args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
