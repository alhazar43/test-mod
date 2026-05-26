#!/usr/bin/env python3
"""Compare committed GPU scaling experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GPU stress summaries.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result directories, summary JSON files, or glob patterns.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a Markdown table.",
    )
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


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| job | node | GPUs | dtype | size | TFLOPS | speedup | path |")
    print("| --- | --- | ---: | --- | ---: | ---: | ---: | --- |")
    baseline = min(rows, key=lambda row: row["device_count"])["aggregate_tflops"]
    for row in rows:
        speedup = row["aggregate_tflops"] / baseline if baseline else 0.0
        print(
            f"| {row.get('slurm_job_id') or ''} "
            f"| {row.get('slurm_job_nodelist') or ''} "
            f"| {row['device_count']} "
            f"| {row['dtype']} "
            f"| {row['size']} "
            f"| {row['aggregate_tflops']:.2f} "
            f"| {speedup:.2f}x "
            f"| `{row['_path']}` |"
        )


def print_text(rows: list[dict[str, Any]]) -> None:
    baseline = min(rows, key=lambda row: row["device_count"])["aggregate_tflops"]
    for row in rows:
        speedup = row["aggregate_tflops"] / baseline if baseline else 0.0
        print(
            f"job={row.get('slurm_job_id') or ''} "
            f"node={row.get('slurm_job_nodelist') or ''} "
            f"gpus={row['device_count']} "
            f"dtype={row['dtype']} "
            f"size={row['size']} "
            f"tflops={row['aggregate_tflops']:.2f} "
            f"speedup={speedup:.2f}x "
            f"path={row['_path']}"
        )


def main() -> int:
    args = parse_args()
    summaries = expand_paths(args.paths)
    if not summaries:
        raise SystemExit("No summary.json files found.")

    rows = sorted(
        (load_summary(path) for path in summaries),
        key=lambda row: (row["size"], row["dtype"], row["device_count"], row["_path"]),
    )

    if args.markdown:
        print_markdown(rows)
    else:
        print_text(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
