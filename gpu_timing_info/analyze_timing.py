#!/usr/bin/env python3
"""
GPU Timing Analysis Script

Analyzes training and inference timing for tainted batches:
1. KDE-enhanced runs (from batch_timing JSON files in artifacts_reduced)
2. Base runs (from batch_timing JSON files in artifacts, if available)

The goal is to compare the average time per tainted batch (batches containing
KDE-eligible edges) between KDE-enhanced configs and base configs.

Results are saved to gpu_timing_info/
"""

import json
import os
import re
from datetime import datetime
from collections import defaultdict
import statistics

# Base path for artifacts
BASE_PATH = "/scratch/asawan15/PIDSMaker/artifacts_base_and_kde_ts"

# Hash → Config mapping based on hash_to_run_mapping.txt
# The same hash appears in both artifacts/ and artifacts_reduced/ for different configs
# because the hash is derived from dataset-level parameters, not the config name.

# KDE-enhanced configs (in artifacts_reduced/)
KDE_CONFIGS = {
    # orthrus_edge_kde_ts
    ("2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7", "CLEARSCOPE_E3"): {
        "config": "orthrus_edge_kde_ts", "epoch": 0
    },
    ("970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727", "CADETS_E3"): {
        "config": "orthrus_edge_kde_ts", "epoch": 9
    },
    ("9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457", "THEIA_E3"): {
        "config": "orthrus_edge_kde_ts", "epoch": 11
    },
    # kairos_kde_ts
    ("293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e", "CLEARSCOPE_E3"): {
        "config": "kairos_kde_ts", "epoch": 3
    },
    ("133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269", "CADETS_E3"): {
        "config": "kairos_kde_ts", "epoch": 11
    },
    ("e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5", "THEIA_E3"): {
        "config": "kairos_kde_ts", "epoch": 9
    },
}

# Base configs (in artifacts/)
BASE_CONFIGS = {
    # orthrus_non_snooped_edge_ts
    ("293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e", "CLEARSCOPE_E3"): {
        "config": "orthrus_non_snooped_edge_ts", "epoch": 1
    },
    ("970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727", "CADETS_E3"): {
        "config": "orthrus_non_snooped_edge_ts", "epoch": 7
    },
    ("9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457", "THEIA_E3"): {
        "config": "orthrus_non_snooped_edge_ts", "epoch": 0
    },
    # kairos
    ("2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7", "CLEARSCOPE_E3"): {
        "config": "kairos", "epoch": 5
    },
    ("133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269", "CADETS_E3"): {
        "config": "kairos", "epoch": 3
    },
    ("e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5", "THEIA_E3"): {
        "config": "kairos", "epoch": 1
    },
}

# Mapping of KDE config to its base config counterpart
KDE_TO_BASE_CONFIG = {
    "orthrus_edge_kde_ts": "orthrus_non_snooped_edge_ts",
    "kairos_kde_ts": "kairos",
}


def parse_tainted_batches(batch_timing_dir, dataset):
    """
    Parse tainted batch timing data from batch_timing directory.
    
    Files:
    - tainted_batches_{dataset}.json: Training tainted batches (all epochs)
    - inference_tainted_batches_{dataset}_epoch{N}.json: Inference tainted batches per epoch
    
    Returns dict with:
    - training: {epoch: [list of batch timing dicts]}
    - inference: {epoch: [list of batch timing dicts]}
    """
    training_batches = defaultdict(list)
    inference_batches = defaultdict(list)
    
    # Parse training tainted batches
    training_file = os.path.join(batch_timing_dir, f"tainted_batches_{dataset}.json")
    if os.path.exists(training_file):
        with open(training_file, 'r') as f:
            data = json.load(f)
        
        for batch in data.get('batches', []):
            epoch = batch.get('epoch', 0)
            training_batches[epoch].append(batch)
    
    # Parse inference tainted batches for each epoch
    for filename in os.listdir(batch_timing_dir):
        if filename.startswith(f'inference_tainted_batches_{dataset}_epoch') and filename.endswith('.json'):
            match = re.search(r'epoch(\d+)', filename)
            if match:
                epoch = int(match.group(1))
                filepath = os.path.join(batch_timing_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                for batch in data.get('batches', []):
                    inference_batches[epoch].append(batch)
    
    return {
        'training': dict(training_batches),
        'inference': dict(inference_batches),
    }


def compute_batch_stats(batch_list):
    """
    Compute statistics for a list of batch timing dicts.
    
    Returns dict with:
    - count: number of batches
    - total_time_ms: sum of total_time_ms
    - avg_time_ms: average total_time_ms per batch
    - avg_forward_ms: average forward pass time
    - avg_backward_ms: average backward pass time (training only)
    - total_edges: total edges across all batches
    - avg_edges_per_batch: average edges per batch
    - kde_eligible_edges: total KDE-eligible edges
    """
    if not batch_list:
        return None
    
    count = len(batch_list)
    total_time_ms = sum(b.get('total_time_ms', 0) for b in batch_list)
    forward_times = [b.get('forward_time_ms', 0) for b in batch_list]
    backward_times = [b.get('backward_time_ms', 0) for b in batch_list if 'backward_time_ms' in b]
    total_edges = sum(b.get('total_edges', 0) for b in batch_list)
    kde_eligible = sum(b.get('kde_eligible_edges', 0) for b in batch_list)
    
    stats = {
        'count': count,
        'total_time_ms': total_time_ms,
        'avg_time_ms': total_time_ms / count,
        'avg_forward_ms': statistics.mean(forward_times) if forward_times else 0,
        'total_edges': total_edges,
        'avg_edges_per_batch': total_edges / count,
        'kde_eligible_edges': kde_eligible,
    }
    
    if backward_times:
        stats['avg_backward_ms'] = statistics.mean(backward_times)
    
    return stats


def main():
    output_dir = "/scratch/asawan15/PIDSMaker/gpu_timing_info"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("GPU TIMING ANALYSIS - Tainted Batches")
    print("=" * 80)
    
    # Collect timing data for all configs
    kde_timing = {}  # KDE-enhanced configs
    base_timing = {}  # Base configs (may not have batch_timing data)
    
    # ========================================================================
    # Parse KDE-enhanced runs (artifacts_reduced)
    # ========================================================================
    print("\n--- Parsing KDE-enhanced runs (artifacts_reduced) ---")
    
    kde_base_dir = os.path.join(BASE_PATH, "artifacts_reduced", "evaluation", "evaluation")
    
    for (hash_val, dataset), info in KDE_CONFIGS.items():
        config = info['config']
        epoch = info['epoch']
        
        batch_timing_dir = os.path.join(kde_base_dir, hash_val, dataset, "batch_timing")
        
        if not os.path.exists(batch_timing_dir):
            print(f"  [MISSING] {config}/{dataset}: {batch_timing_dir}")
            continue
        
        timing = parse_tainted_batches(batch_timing_dir, dataset)
        
        # Compute stats for training (all epochs combined)
        all_training_batches = []
        for batches in timing['training'].values():
            all_training_batches.extend(batches)
        
        # Compute stats for inference (specific epoch)
        inference_batches = timing['inference'].get(epoch, [])
        
        key = f"{config}_{dataset}"
        kde_timing[key] = {
            'config': config,
            'dataset': dataset,
            'epoch': epoch,
            'hash': hash_val,
            'training_all_epochs': compute_batch_stats(all_training_batches),
            'training_by_epoch': {e: compute_batch_stats(b) for e, b in timing['training'].items()},
            'inference_epoch': compute_batch_stats(inference_batches),
        }
        
        train_count = len(all_training_batches)
        inf_count = len(inference_batches)
        print(f"  [OK] {config}/{dataset}: {train_count} training batches, {inf_count} inference batches (epoch {epoch})")
    
    # ========================================================================
    # Parse Base runs (artifacts) - if they have batch_timing data
    # ========================================================================
    print("\n--- Parsing Base runs (artifacts) ---")
    
    base_base_dir = os.path.join(BASE_PATH, "artifacts", "evaluation", "evaluation")
    
    for (hash_val, dataset), info in BASE_CONFIGS.items():
        config = info['config']
        epoch = info['epoch']
        
        batch_timing_dir = os.path.join(base_base_dir, hash_val, dataset, "batch_timing")
        
        if not os.path.exists(batch_timing_dir):
            print(f"  [MISSING] {config}/{dataset}: batch_timing not available")
            continue
        
        timing = parse_tainted_batches(batch_timing_dir, dataset)
        
        # Compute stats
        all_training_batches = []
        for batches in timing['training'].values():
            all_training_batches.extend(batches)
        
        inference_batches = timing['inference'].get(epoch, [])
        
        key = f"{config}_{dataset}"
        base_timing[key] = {
            'config': config,
            'dataset': dataset,
            'epoch': epoch,
            'hash': hash_val,
            'training_all_epochs': compute_batch_stats(all_training_batches),
            'training_by_epoch': {e: compute_batch_stats(b) for e, b in timing['training'].items()},
            'inference_epoch': compute_batch_stats(inference_batches),
        }
        
        train_count = len(all_training_batches)
        inf_count = len(inference_batches)
        print(f"  [OK] {config}/{dataset}: {train_count} training batches, {inf_count} inference batches (epoch {epoch})")
    
    # ========================================================================
    # Generate Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("TAINTED BATCH TIMING RESULTS")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("GPU Timing Analysis - Tainted Batches")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Note: 'Tainted batches' are batches containing KDE-eligible edges")
    report_lines.append("      (edges with timestamps that could be processed using KDE).")
    report_lines.append("")
    
    # Summary table
    report_lines.append("=" * 100)
    report_lines.append("SUMMARY: Average Time per Tainted Batch")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"{'Config':<30} {'Dataset':<15} {'Phase':<12} {'Batches':<10} {'Avg ms/batch':<15} {'Total Time (s)':<15}")
    report_lines.append("-" * 100)
    
    for key in sorted(kde_timing.keys()):
        data = kde_timing[key]
        config = data['config']
        dataset = data['dataset']
        
        # Training
        if data['training_all_epochs']:
            stats = data['training_all_epochs']
            report_lines.append(
                f"{config:<30} {dataset:<15} {'Training':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
            )
        
        # Inference
        if data['inference_epoch']:
            stats = data['inference_epoch']
            report_lines.append(
                f"{config:<30} {dataset:<15} {'Inference':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
            )
    
    report_lines.append("")
    
    # If we have base timing data, add comparison
    if base_timing:
        report_lines.append("")
        report_lines.append("=" * 100)
        report_lines.append("BASE CONFIGS (non-KDE)")
        report_lines.append("=" * 100)
        report_lines.append("")
        report_lines.append(f"{'Config':<30} {'Dataset':<15} {'Phase':<12} {'Batches':<10} {'Avg ms/batch':<15} {'Total Time (s)':<15}")
        report_lines.append("-" * 100)
        
        for key in sorted(base_timing.keys()):
            data = base_timing[key]
            config = data['config']
            dataset = data['dataset']
            
            if data['training_all_epochs']:
                stats = data['training_all_epochs']
                report_lines.append(
                    f"{config:<30} {dataset:<15} {'Training':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
                )
            
            if data['inference_epoch']:
                stats = data['inference_epoch']
                report_lines.append(
                    f"{config:<30} {dataset:<15} {'Inference':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
                )
    
    # Detailed per-config breakdown
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("DETAILED BREAKDOWN BY CONFIG")
    report_lines.append("=" * 80)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        report_lines.append("")
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"KDE Config: {kde_config}")
        report_lines.append(f"Base Config: {base_config}")
        report_lines.append("=" * 80)
        
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            report_lines.append(f"\n--- {dataset} ---")
            
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            if kde_key in kde_timing:
                data = kde_timing[kde_key]
                report_lines.append(f"\n  KDE ({kde_config}), Epoch {data['epoch']}:")
                
                if data['training_all_epochs']:
                    stats = data['training_all_epochs']
                    report_lines.append(f"    Training: {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
                    report_lines.append(f"              {stats['total_edges']:,} total edges, {stats['kde_eligible_edges']:,} KDE-eligible")
                    report_lines.append(f"              Forward: {stats['avg_forward_ms']:.2f} ms avg")
                    if 'avg_backward_ms' in stats:
                        report_lines.append(f"              Backward: {stats['avg_backward_ms']:.2f} ms avg")
                
                if data['inference_epoch']:
                    stats = data['inference_epoch']
                    report_lines.append(f"    Inference (epoch {data['epoch']}): {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
                    report_lines.append(f"              {stats['total_edges']:,} total edges")
                else:
                    report_lines.append(f"    Inference (epoch {data['epoch']}): No tainted batches")
            
            if base_key in base_timing:
                data = base_timing[base_key]
                report_lines.append(f"\n  Base ({base_config}), Epoch {data['epoch']}:")
                
                if data['training_all_epochs']:
                    stats = data['training_all_epochs']
                    report_lines.append(f"    Training: {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
                
                if data['inference_epoch']:
                    stats = data['inference_epoch']
                    report_lines.append(f"    Inference (epoch {data['epoch']}): {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
    
    # Speedup comparison table
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("SPEEDUP COMPARISON: KDE vs Base Config")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("Note: Speedup > 1.0 means KDE is faster; < 1.0 means Base is faster")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Base Config':<30} {'KDE Config':<25} {'Base ms/batch':<15} {'KDE ms/batch':<15} {'Speedup':<10}")
    report_lines.append("-" * 130)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            if kde_key in kde_timing and base_key in base_timing:
                kde_data = kde_timing[kde_key]
                base_data = base_timing[base_key]
                
                if kde_data['training_all_epochs'] and base_data['training_all_epochs']:
                    kde_avg = kde_data['training_all_epochs']['avg_time_ms']
                    base_avg = base_data['training_all_epochs']['avg_time_ms']
                    speedup = base_avg / kde_avg if kde_avg > 0 else 0
                    
                    model = 'orthrus' if 'orthrus' in kde_config else 'kairos'
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {base_config:<30} {kde_config:<25} {base_avg:<15.2f} {kde_avg:<15.2f} {speedup:<10.2f}x"
                    )
    
    # Print and save report
    report_text = '\n'.join(report_lines)
    print(report_text)
    
    with open(os.path.join(output_dir, "tainted_batch_timing_report.txt"), 'w') as f:
        f.write(report_text)
    
    # Save detailed JSON
    detailed_results = {
        'kde_timing': kde_timing,
        'base_timing': base_timing,
    }
    
    with open(os.path.join(output_dir, "tainted_batch_timing_detailed.json"), 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to {output_dir}/")
    print("  - tainted_batch_timing_report.txt")
    print("  - tainted_batch_timing_detailed.json")


if __name__ == "__main__":
    main()
