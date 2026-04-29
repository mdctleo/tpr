#!/usr/bin/env python3
# ******************************************************************************
# generate_reduced_csv.py
#
# Generate a reduced version of cic_20.csv by keeping only the FIRST timestamp
# for each unique (src, dst) pair. This creates a much smaller graph where each
# directed edge appears exactly once.
#
# Usage:
#   python generate_reduced_csv.py --input dataset/cic/cic_20.csv --output dataset/cic/cic_20_red.csv
#
# Date      Name       Description
# ========  =========  ========================================================
# 2024      Asawan     Reduced graph generation for PIKACHU
# ******************************************************************************

import argparse
import os
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Generate reduced CIC-IDS CSV')
    parser.add_argument('--input', type=str, default='dataset/cic/cic_20.csv',
                        help='Path to the original CIC-IDS CSV file')
    parser.add_argument('--output', type=str, default='dataset/cic/cic_20_red.csv',
                        help='Output path for reduced CSV')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Generating Reduced CIC-IDS CSV")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.input)
    print(f"Original records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Identify source and destination columns
    if 'src_computer' in df.columns:
        src_col, dst_col = 'src_computer', 'dst_computer'
    elif 'src' in df.columns:
        src_col, dst_col = 'src', 'dst'
    else:
        raise ValueError(f"Cannot find source/destination columns. Available: {list(df.columns)}")
    
    # Identify timestamp column
    if 'timestamp' in df.columns:
        ts_col = 'timestamp'
    elif 'time' in df.columns:
        ts_col = 'time'
    else:
        raise ValueError(f"Cannot find timestamp column. Available: {list(df.columns)}")
    
    print(f"Using columns: src={src_col}, dst={dst_col}, timestamp={ts_col}")
    
    # Sort by timestamp to ensure we keep the FIRST occurrence
    print("\nSorting by timestamp...")
    df_sorted = df.sort_values(by=ts_col)
    
    # Keep only the first occurrence of each (src, dst) pair
    print("Keeping first timestamp per (src, dst) pair...")
    df_reduced = df_sorted.drop_duplicates(subset=[src_col, dst_col], keep='first')
    
    # Sort back by original order (by index or timestamp)
    df_reduced = df_reduced.sort_values(by=ts_col)
    
    print(f"\nReduction statistics:")
    print(f"  - Original records: {len(df):,}")
    print(f"  - Reduced records:  {len(df_reduced):,}")
    print(f"  - Reduction ratio:  {len(df_reduced)/len(df)*100:.2f}%")
    print(f"  - Records removed:  {len(df) - len(df_reduced):,}")
    
    # Check unique (src, dst) pairs
    original_pairs = df.groupby([src_col, dst_col]).ngroups
    reduced_pairs = df_reduced.groupby([src_col, dst_col]).ngroups
    print(f"\nUnique (src, dst) pairs:")
    print(f"  - Original: {original_pairs:,}")
    print(f"  - Reduced:  {reduced_pairs:,}")
    print(f"  - Match:    {original_pairs == reduced_pairs}")
    
    # Check snapshot distribution
    print(f"\nSnapshot distribution in reduced data:")
    snapshot_counts = df_reduced['snapshot'].value_counts().sort_index()
    print(f"  - Total unique snapshots: {len(snapshot_counts)}")
    print(f"  - Snapshot range: {snapshot_counts.index.min()} to {snapshot_counts.index.max()}")
    
    # Check label distribution
    if 'label' in df_reduced.columns:
        print(f"\nLabel distribution in reduced data:")
        label_counts = df_reduced['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  - {label}: {count:,} ({count/len(df_reduced)*100:.2f}%)")
    
    # Save reduced CSV
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df_reduced.to_csv(args.output, index=False)
    
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved to: {args.output} ({file_size_mb:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("Done! You can now run main.py with --red flag to use this reduced dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()
