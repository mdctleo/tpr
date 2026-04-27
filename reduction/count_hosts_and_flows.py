#!/usr/bin/env python3
"""Count unique hosts and flows from one or more CSV files.

Usage examples:
  python scripts/count_hosts_and_flows.py data1.csv data2.csv
  python scripts/count_hosts_and_flows.py "datasets/*.csv"
  python scripts/count_hosts_and_flows.py "datasets/*.csv" --host-cols src_addr dst_addr
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Iterable, List, Sequence

from tqdm import tqdm


DEFAULT_HOST_CANDIDATES = [
    "host",
    "hostname",
    "src_host",
    "dst_host",
    "source_host",
    "destination_host",
    "src_ip",
    "dest_ip",
    "dst_ip",
    "source_ip",
    "destination_ip",
    "source ip",
    "destination ip",
    "src_addr",
    "dst_addr",
    "local_ip",
    "remote_ip",
    "subjectname",
    "objectname",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count unique hosts and flow rows across CSV files."
    )
    parser.add_argument(
        "csvs",
        nargs="*",
        default=["./datasets/cic-ids-2017/TrafficLabelling /*.csv"],
        help="CSV files and/or glob patterns (for example datasets/*.csv). Defaults to ./datasets/cic-ids-2017/TrafficLabelling /*.csv",
    )
    parser.add_argument(
        "--host-cols",
        nargs="+",
        default=None,
        help=(
            "Optional host columns to use. If omitted, the script auto-detects "
            "common host columns."
        ),
    )
    return parser.parse_args()


def resolve_csv_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matched = sorted(Path(p) for p in glob.glob(pattern))
        if matched:
            paths.extend(matched)
            continue

        p = Path(pattern)
        if p.exists():
            paths.append(p)

    unique_sorted = sorted(set(paths))
    if not unique_sorted:
        raise FileNotFoundError("No CSV files matched the provided inputs.")
    return unique_sorted


def detect_host_columns(columns: Iterable[str], requested: Sequence[str] | None) -> List[str]:
    column_list = list(columns)
    if requested:
        return [col for col in requested if col in column_list]

    candidates = {c.lower() for c in DEFAULT_HOST_CANDIDATES}
    return [col for col in column_list if col.lower().strip() in candidates]


def main() -> None:
    args = parse_args()
    csv_paths = resolve_csv_paths(args.csvs)

    total_flows = 0
    unique_hosts = set()

    first_columns: List[str] | None = None
    host_columns: List[str] = []

    for idx, csv_path in enumerate(tqdm(csv_paths, desc="Processing files")):
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue

            if idx == 0:
                first_columns = list(reader.fieldnames)
                print("Columns:")
                print(", ".join(first_columns))
                host_columns = detect_host_columns(first_columns, args.host_cols)

            for row in tqdm(reader, desc=f"  {csv_path.name}", leave=False):
                total_flows += 1
                for host_col in host_columns:
                    value = row.get(host_col)
                    if value is not None:
                        value = value.strip()
                        if value:
                            unique_hosts.add(value)

    if first_columns is None:
        raise ValueError("No readable CSV header found in the provided files.")

    print("\nCounts:")
    print(f"Files processed: {len(csv_paths)}")
    print(f"Host columns used: {host_columns if host_columns else 'None found'}")
    print(f"Unique hosts: {len(unique_hosts)}")
    print(f"Flows (rows): {total_flows}")


if __name__ == "__main__":
    main()
