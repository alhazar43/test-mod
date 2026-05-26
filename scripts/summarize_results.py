#!/usr/bin/env python3
"""Aggregate repeated benchmark summaries by comparable condition."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated benchmark results.")
    parser.add_argument("paths", nargs="+", help="Result directories, summary JSON files, or globs.")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown table.")
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    summaries: list[Path] = []
    for pattern in patterns:
        matches = list(Path().glob(pattern)) if any(char in pattern for char in "*?[]") else [Path(pattern)]
        for match in matches:
            if match.is_dir():
                summaries.extend(match.rglob("summary.json"))
            elif match.suffix == ".json":
                summaries.append(match)
    return sorted(set(summaries))


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    data["_path"] = str(path)
    return data


def gpu_name(row: dict[str, Any]) -> str:
    names = sorted({device.get("name", device.get("device", "")) for device in row["devices"]})
    return "+".join(names)


def metric_name(row: dict[str, Any]) -> str:
    if "aggregate_tflops" in row:
        return "TFLOPS"
    return row.get("metric_name", "metric")


def metric_value(row: dict[str, Any]) -> float:
    if "aggregate_tflops" in row:
        return float(row["aggregate_tflops"])
    if "tokens_per_second" in row:
        return float(row["tokens_per_second"])
    return float(row["metric_value"])


def workload(row: dict[str, Any]) -> str:
    if row.get("benchmark_type") == "synthetic_transformer_training":
        return (
            f"transformer b{row['batch_size']} s{row['seq_len']} "
            f"d{row['d_model']} l{row['layers']} warm{row.get('warmup_steps', 0)} "
            f"meas{row.get('measure_steps', 0)}"
        )
    return f"matmul size{row['size']}"


def condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        workload(row),
        row["dtype"],
        row["device_count"],
        gpu_name(row),
        metric_name(row),
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[condition_key(row)].append(row)

    summaries = []
    for key, group in groups.items():
        values = [metric_value(row) for row in group]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        sem = std / math.sqrt(len(values)) if values else 0.0
        summaries.append(
            {
                "workload": key[0],
                "dtype": key[1],
                "gpus": key[2],
                "gpu_name": key[3],
                "metric": key[4],
                "n": len(values),
                "mean": mean,
                "std": std,
                "sem": sem,
                "min": min(values),
                "max": max(values),
                "jobs": ",".join(str(row.get("slurm_job_id") or "") for row in group),
            }
        )
    return sorted(summaries, key=lambda item: (item["workload"], item["gpu_name"], item["gpus"]))


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| workload | GPU | GPUs | dtype | metric | n | mean | std | sem | min | max | jobs |")
    print("| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        print(
            f"| {row['workload']} | {row['gpu_name']} | {row['gpus']} | {row['dtype']} "
            f"| {row['metric']} | {row['n']} | {row['mean']:.2f} | {row['std']:.2f} "
            f"| {row['sem']:.2f} | {row['min']:.2f} | {row['max']:.2f} | {row['jobs']} |"
        )


def print_text(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"workload={row['workload']!r} gpu={row['gpu_name']!r} gpus={row['gpus']} "
            f"dtype={row['dtype']} metric={row['metric']} n={row['n']} "
            f"mean={row['mean']:.2f} std={row['std']:.2f} sem={row['sem']:.2f} "
            f"min={row['min']:.2f} max={row['max']:.2f} jobs={row['jobs']}"
        )


def main() -> int:
    args = parse_args()
    paths = expand_paths(args.paths)
    if not paths:
        raise SystemExit("No summary.json files found.")

    rows = [load_summary(path) for path in paths]
    summary = summarize(rows)
    if args.markdown:
        print_markdown(summary)
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
