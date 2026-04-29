#!/usr/bin/env python3
"""Count unique hosts and flows for the Hypervision dataset.

Assumptions for Hypervision `.data` rows (space-separated, 8 fields):
  1: protocol
  2: source host id
  3: destination host id
  4: source port
  5: destination port
  6: timestamp or time-like value
  7-8: additional flow attributes

This script counts:
- flows: number of valid rows in `.data` files
- unique hosts: unique values seen in source/destination host columns

By default it scans `./datasets/data/*.data` from within the `reduction/` directory.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Sequence

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count unique hosts and flows in Hypervision .data files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["./datasets/data/*.data"],
        help=(
            "Input .data files and/or glob patterns. "
            "Default: ./datasets/data/*.data"
        ),
    )
    parser.add_argument(
        "--src-col",
        type=int,
        default=2,
        help="1-based source host column index (default: 2).",
    )
    parser.add_argument(
        "--dst-col",
        type=int,
        default=3,
        help="1-based destination host column index (default: 3).",
    )
    parser.add_argument(
        "--show-label-info",
        action="store_true",
        help="Also report corresponding .label file presence and size.",
    )
    return parser.parse_args()


def resolve_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matched = sorted(Path(p) for p in glob.glob(pattern))
        if matched:
            paths.extend(matched)
            continue

        candidate = Path(pattern)
        if candidate.exists():
            paths.append(candidate)

    result = sorted(set(paths))
    if not result:
        raise FileNotFoundError("No Hypervision .data files matched the provided inputs.")
    return result


def label_info(data_path: Path) -> str:
    label_path = data_path.with_suffix(".label")
    if not label_path.exists():
        return "missing"
    return f"present ({label_path.stat().st_size} bytes)"


def main() -> None:
    args = parse_args()

    if args.src_col < 1 or args.dst_col < 1:
        raise ValueError("Column indexes must be 1-based positive integers.")

    src_idx = args.src_col - 1
    dst_idx = args.dst_col - 1

    data_paths = resolve_paths(args.paths)

    total_flows = 0
    malformed_rows = 0
    unique_hosts = set()

    print("Hypervision field assumption: src host column =", args.src_col, ", dst host column =", args.dst_col)

    for data_path in tqdm(data_paths, desc="Processing .data files"):
        file_flows = 0
        with data_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in tqdm(handle, desc=f"  {data_path.name}", leave=False):
                row = line.strip()
                if not row:
                    continue

                parts = row.split()
                needed = max(src_idx, dst_idx)
                if len(parts) <= needed:
                    malformed_rows += 1
                    continue

                src = parts[src_idx]
                dst = parts[dst_idx]

                if src:
                    unique_hosts.add(src)
                if dst:
                    unique_hosts.add(dst)

                total_flows += 1
                file_flows += 1

        msg = f"{data_path.name}: flows={file_flows}"
        if args.show_label_info:
            msg += f", label={label_info(data_path)}"
        print(msg)

    print("\nCounts:")
    print(f"Files processed: {len(data_paths)}")
    print(f"Unique hosts: {len(unique_hosts)}")
    print(f"Flows (rows): {total_flows}")
    print(f"Malformed rows skipped: {malformed_rows}")


if __name__ == "__main__":
    main()
