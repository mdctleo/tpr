"""
Batch Tainting and Timing Instrumentation for KDE-based Training

This module provides utilities to:
1. Track which batches contain KDE-eligible edges (edges with >= min_occurrences timestamps)
2. Measure detailed timing information for tainted vs. non-tainted batches
3. Log comprehensive statistics about KDE usage during training and inference

Usage:
    from pidsmaker.utils.batch_timing import BatchTimingTracker, TaintedBatchLogger
    
    tracker = BatchTimingTracker(kde_eligible_edges, min_occurrences=10)
    
    for batch in batches:
        with tracker.time_batch(batch, phase='train'):
            output = model(batch)
    
    tracker.log_summary()
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Global state for batch timing across the training run
_global_timing_state = {
    'enabled': False,
    'tracker': None,
    'results': [],
}


@dataclass
class BatchTimingResult:
    """Result from timing a single batch."""
    batch_id: int
    phase: str  # 'train' or 'inference'
    epoch: int
    split: str  # 'train', 'val', 'test'
    
    # Batch composition
    total_edges: int
    kde_eligible_edges: int
    non_kde_edges: int
    taint_ratio: float  # kde_eligible / total
    
    # Timestamp counts
    total_timestamps: int = 0  # Total timestamps in the batch
    kde_eligible_timestamps: int = 0  # Timestamps in KDE-eligible edges
    
    # Timing (in milliseconds)
    forward_time_ms: float = 0.0
    backward_time_ms: Optional[float] = None  # Only for training
    total_time_ms: float = 0.0
    
    # Reduction info
    edges_that_could_be_reduced: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'batch_id': self.batch_id,
            'phase': self.phase,
            'epoch': self.epoch,
            'split': self.split,
            'total_edges': self.total_edges,
            'kde_eligible_edges': self.kde_eligible_edges,
            'non_kde_edges': self.non_kde_edges,
            'taint_ratio': self.taint_ratio,
            'total_timestamps': self.total_timestamps,
            'kde_eligible_timestamps': self.kde_eligible_timestamps,
            'forward_time_ms': self.forward_time_ms,
            'backward_time_ms': self.backward_time_ms,
            'total_time_ms': self.total_time_ms,
            'edges_that_could_be_reduced': self.edges_that_could_be_reduced,
        }


class BatchTimingTracker:
    """
    Tracks timing and taint information for batches during training/inference.
    
    A batch is "tainted" if it contains any KDE-eligible edges (edges with >= min_occurrences
    timestamps in the full dataset).
    
    Edge keys are 3-tuples: (src, dst, edge_type)
    """
    
    def __init__(
        self,
        kde_eligible_edges: Optional[Set[Tuple[int, int, int]]] = None,
        edge_occurrence_counts: Optional[Dict[Tuple[int, int, int], int]] = None,
        min_occurrences: int = 10,
        output_dir: str = "timing_results",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the batch timing tracker.
        
        Args:
            kde_eligible_edges: Set of (src, dst, edge_type) tuples that have KDE vectors
            edge_occurrence_counts: Dict mapping (src, dst, edge_type) to occurrence count
            min_occurrences: Threshold for KDE eligibility
            output_dir: Directory to save timing results
            device: CUDA device for timing (uses CUDA events if available)
        """
        self.kde_eligible_edges = kde_eligible_edges or set()
        self.edge_occurrence_counts = edge_occurrence_counts or {}
        self.min_occurrences = min_occurrences
        self.output_dir = output_dir
        self.device = device
        
        self.results: List[BatchTimingResult] = []
        self.batch_counter = 0
        self.current_epoch = 0
        self.current_split = 'train'
        
        # CUDA timing events
        self._use_cuda = device is not None and device.type == 'cuda'
        if self._use_cuda:
            self._starter = torch.cuda.Event(enable_timing=True)
            self._ender = torch.cuda.Event(enable_timing=True)
        
        # Summary statistics
        self._summary = defaultdict(lambda: defaultdict(list))
        
        # Timing state for start_batch/end_batch pattern
        self._batch_start_time = None
        self._batch_cuda_start = None
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"BatchTimingTracker initialized with {len(self.kde_eligible_edges)} KDE-eligible edges")
    
    def start_batch(self):
        """Start timing a batch. Call this before processing the batch."""
        if self._use_cuda:
            torch.cuda.synchronize()
            if self._batch_cuda_start is None:
                self._batch_cuda_start = torch.cuda.Event(enable_timing=True)
                self._batch_cuda_end = torch.cuda.Event(enable_timing=True)
            self._batch_cuda_start.record()
        self._batch_start_time = time.perf_counter()
    
    def end_batch(self, batch_idx: int, epoch: int, batch_edges: Set[Tuple[int, int, int]], phase: str = 'inference'):
        """
        End timing a batch and record results.
        
        Args:
            batch_idx: Index of the batch
            epoch: Current epoch (-1 for inference)
            batch_edges: Set of (src, dst, edge_type) tuples in the batch
            phase: Phase name (e.g., 'train', 'inference_val', 'inference_test')
        """
        # Calculate elapsed time
        if self._use_cuda:
            self._batch_cuda_end.record()
            torch.cuda.synchronize()
            forward_time_ms = self._batch_cuda_start.elapsed_time(self._batch_cuda_end)
        else:
            forward_time_ms = (time.perf_counter() - self._batch_start_time) * 1000
        
        # Analyze batch for KDE eligibility
        total_edges = len(batch_edges)
        kde_eligible = sum(1 for edge in batch_edges if edge in self.kde_eligible_edges)
        non_kde = total_edges - kde_eligible
        taint_ratio = kde_eligible / total_edges if total_edges > 0 else 0.0
        
        # Count edges that could be reduced (KDE-eligible with count > 1)
        edges_could_reduce = sum(
            1 for edge in batch_edges 
            if edge in self.kde_eligible_edges and self.edge_occurrence_counts.get(edge, 1) > 1
        )
        
        # Timestamp counts (each edge has 1 timestamp)
        total_timestamps = total_edges
        kde_eligible_timestamps = kde_eligible
        
        # Determine split from phase
        split = 'test'
        if 'val' in phase:
            split = 'val'
        elif 'train' in phase:
            split = 'train'
        
        # Create and store result
        result = BatchTimingResult(
            batch_id=batch_idx,
            phase=phase,
            epoch=epoch,
            split=split,
            total_edges=total_edges,
            kde_eligible_edges=kde_eligible,
            non_kde_edges=non_kde,
            taint_ratio=taint_ratio,
            forward_time_ms=forward_time_ms,
            total_time_ms=forward_time_ms,
            edges_that_could_be_reduced=edges_could_reduce,
            total_timestamps=total_timestamps,
            kde_eligible_timestamps=kde_eligible_timestamps,
        )
        
        self.results.append(result)
        self._update_summary(result)
        self.batch_counter += 1
    
    def set_epoch(self, epoch: int):
        """Set the current epoch for tracking."""
        self.current_epoch = epoch
    
    def set_split(self, split: str):
        """Set the current split (train/val/test) for tracking."""
        self.current_split = split
    
    def analyze_batch(self, batch) -> Dict[str, Any]:
        """
        Analyze a batch to determine KDE eligibility, taint ratio, and timestamp counts.
        
        Args:
            batch: Batch object with src, dst, edge_type attributes
            
        Returns:
            Dict with analysis results including timestamp counts
        """
        # Get source and destination node IDs
        if hasattr(batch, 'src') and hasattr(batch, 'dst'):
            src = batch.src.cpu().numpy() if isinstance(batch.src, torch.Tensor) else batch.src
            dst = batch.dst.cpu().numpy() if isinstance(batch.dst, torch.Tensor) else batch.dst
        elif hasattr(batch, 'edge_index'):
            edge_index = batch.edge_index.cpu().numpy() if isinstance(batch.edge_index, torch.Tensor) else batch.edge_index
            src, dst = edge_index[0], edge_index[1]
        else:
            logger.warning("Batch has no src/dst or edge_index attributes")
            return {'total_edges': 0, 'kde_eligible': 0, 'non_kde': 0, 'taint_ratio': 0.0,
                    'edges_could_reduce': 0, 'total_timestamps': 0, 'kde_eligible_timestamps': 0}
        
        # Get edge types
        if hasattr(batch, 'edge_type') and batch.edge_type is not None:
            edge_type = batch.edge_type
            if isinstance(edge_type, torch.Tensor):
                # edge_type is one-hot encoded, get the index
                if edge_type.ndim == 2:
                    edge_types = edge_type.max(dim=1).indices.cpu().numpy()
                else:
                    edge_types = edge_type.cpu().numpy()
            else:
                edge_types = edge_type
        else:
            # Default to edge type 0 if not available
            edge_types = np.zeros(len(src), dtype=np.int64)
        
        total_edges = len(src)
        kde_eligible = 0
        edges_could_reduce = 0
        
        # Track which edges are KDE-eligible for timestamp counting
        kde_eligible_mask = []
        
        for s, d, et in zip(src, dst, edge_types):
            edge_key = (int(s), int(d), int(et))
            is_kde = edge_key in self.kde_eligible_edges
            kde_eligible_mask.append(is_kde)
            if is_kde:
                kde_eligible += 1
            
            # Check if this edge could be reduced (KDE-eligible with count > 1)
            # Only KDE-eligible edges with multiple occurrences are actually reduced
            if is_kde:
                count = self.edge_occurrence_counts.get(edge_key, 1)
                if count > 1:
                    edges_could_reduce += 1
        
        non_kde = total_edges - kde_eligible
        taint_ratio = kde_eligible / total_edges if total_edges > 0 else 0.0
        
        # Count timestamps - each edge has one timestamp in the batch
        # total_timestamps = number of edges (each edge has a timestamp)
        total_timestamps = total_edges
        kde_eligible_timestamps = kde_eligible  # Each KDE-eligible edge has a timestamp
        
        return {
            'total_edges': total_edges,
            'kde_eligible': kde_eligible,
            'non_kde': non_kde,
            'taint_ratio': taint_ratio,
            'edges_could_reduce': edges_could_reduce,
            'total_timestamps': total_timestamps,
            'kde_eligible_timestamps': kde_eligible_timestamps,
        }
    
    def time_forward(self, batch, model_fn, phase: str = 'train') -> Tuple[Any, float]:
        """
        Time the forward pass of a model.
        
        Args:
            batch: Input batch
            model_fn: Callable that performs the forward pass
            phase: 'train' or 'inference'
            
        Returns:
            Tuple of (model output, forward time in ms)
        """
        analysis = self.analyze_batch(batch)
        
        if self._use_cuda:
            torch.cuda.synchronize()
            self._starter.record()
        
        start_time = time.perf_counter()
        output = model_fn(batch)
        
        if self._use_cuda:
            self._ender.record()
            torch.cuda.synchronize()
            forward_time_ms = self._starter.elapsed_time(self._ender)
        else:
            forward_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Create timing result
        result = BatchTimingResult(
            batch_id=self.batch_counter,
            phase=phase,
            epoch=self.current_epoch,
            split=self.current_split,
            total_edges=analysis['total_edges'],
            kde_eligible_edges=analysis['kde_eligible'],
            non_kde_edges=analysis['non_kde'],
            taint_ratio=analysis['taint_ratio'],
            forward_time_ms=forward_time_ms,
            total_time_ms=forward_time_ms,
            edges_that_could_be_reduced=analysis['edges_could_reduce'],
            total_timestamps=analysis['total_timestamps'],
            kde_eligible_timestamps=analysis['kde_eligible_timestamps'],
        )
        
        self.results.append(result)
        self._update_summary(result)
        self.batch_counter += 1
        
        return output, forward_time_ms
    
    def record_backward_time(self, backward_time_ms: float):
        """Record backward pass time for the most recent batch."""
        if self.results:
            self.results[-1].backward_time_ms = backward_time_ms
            self.results[-1].total_time_ms += backward_time_ms
    
    def _update_summary(self, result: BatchTimingResult):
        """Update running summary statistics."""
        key = (result.phase, result.split, result.epoch)
        
        # Categorize by taint level
        if result.taint_ratio == 0:
            taint_category = 'no_kde'
        elif result.taint_ratio < 0.5:
            taint_category = 'low_kde'
        else:
            taint_category = 'high_kde'
        
        self._summary[key][taint_category].append(result)
    
    def get_tainted_batches(self, min_taint_ratio: float = 0.0) -> List[BatchTimingResult]:
        """Get all batches with taint ratio above threshold."""
        return [r for r in self.results if r.taint_ratio > min_taint_ratio]
    
    def log_batch_detail(self, result: BatchTimingResult):
        """Log detailed information about a single batch."""
        is_tainted = result.kde_eligible_edges > 0
        taint_marker = "🔴 TAINTED" if is_tainted else "⚪ CLEAN"
        
        logger.info(
            f"{taint_marker} Batch #{result.batch_id} [{result.phase}/{result.split}] "
            f"Epoch {result.epoch}: "
            f"{result.total_edges} edges "
            f"(KDE: {result.kde_eligible_edges}, "
            f"ratio: {result.taint_ratio:.2%}) "
            f"| Forward: {result.forward_time_ms:.2f}ms "
            f"| Could reduce: {result.edges_that_could_be_reduced} "
            f"| Timestamps: {result.total_timestamps} total, {result.kde_eligible_timestamps} KDE-eligible"
        )
    
    def log_summary(self):
        """Log comprehensive summary of timing results."""
        logger.info("\n" + "=" * 80)
        logger.info("BATCH TIMING SUMMARY")
        logger.info("=" * 80)
        
        # Overall statistics
        total_batches = len(self.results)
        tainted_batches = len([r for r in self.results if r.kde_eligible_edges > 0])
        
        logger.info(f"Total batches processed: {total_batches}")
        logger.info(f"Tainted batches (with KDE edges): {tainted_batches} ({tainted_batches/total_batches*100:.1f}%)" if total_batches > 0 else "N/A")
        
        # Per-phase summary
        for phase in ['train', 'inference']:
            phase_results = [r for r in self.results if r.phase == phase]
            if not phase_results:
                continue
            
            logger.info(f"\n--- {phase.upper()} PHASE ---")
            
            # Tainted vs non-tainted comparison
            tainted = [r for r in phase_results if r.kde_eligible_edges > 0]
            non_tainted = [r for r in phase_results if r.kde_eligible_edges == 0]
            
            if tainted:
                avg_tainted_time = sum(r.forward_time_ms for r in tainted) / len(tainted)
                avg_tainted_edges = sum(r.total_edges for r in tainted) / len(tainted)
                avg_kde_edges = sum(r.kde_eligible_edges for r in tainted) / len(tainted)
                avg_reducible = sum(r.edges_that_could_be_reduced for r in tainted) / len(tainted)
                # Timestamp statistics
                total_ts = sum(r.total_timestamps for r in tainted)
                kde_ts = sum(r.kde_eligible_timestamps for r in tainted)
                avg_total_ts = sum(r.total_timestamps for r in tainted) / len(tainted)
                avg_kde_ts = sum(r.kde_eligible_timestamps for r in tainted) / len(tainted)
                
                logger.info(f"  Tainted batches: {len(tainted)}")
                logger.info(f"    Avg forward time: {avg_tainted_time:.2f}ms")
                logger.info(f"    Avg edges per batch: {avg_tainted_edges:.1f}")
                logger.info(f"    Avg KDE-eligible edges: {avg_kde_edges:.1f}")
                logger.info(f"    Avg reducible edges: {avg_reducible:.1f}")
                logger.info(f"    Total timestamps: {total_ts:,} (avg {avg_total_ts:.1f}/batch)")
                logger.info(f"    KDE-eligible timestamps: {kde_ts:,} (avg {avg_kde_ts:.1f}/batch, {kde_ts/total_ts*100:.1f}%)" if total_ts > 0 else "N/A")
            
            if non_tainted:
                avg_non_tainted_time = sum(r.forward_time_ms for r in non_tainted) / len(non_tainted)
                avg_non_tainted_edges = sum(r.total_edges for r in non_tainted) / len(non_tainted)
                # Timestamp statistics for non-tainted
                total_ts_nt = sum(r.total_timestamps for r in non_tainted)
                avg_total_ts_nt = sum(r.total_timestamps for r in non_tainted) / len(non_tainted)
                
                logger.info(f"  Non-tainted batches: {len(non_tainted)}")
                logger.info(f"    Avg forward time: {avg_non_tainted_time:.2f}ms")
                logger.info(f"    Avg edges per batch: {avg_non_tainted_edges:.1f}")
                logger.info(f"    Total timestamps: {total_ts_nt:,} (avg {avg_total_ts_nt:.1f}/batch)")
            
            # Potential speedup estimation
            if tainted and non_tainted:
                potential_reduction = sum(r.edges_that_could_be_reduced for r in tainted)
                total_edges = sum(r.total_edges for r in tainted)
                if total_edges > 0:
                    logger.info(f"  Potential edge reduction: {potential_reduction:,} / {total_edges:,} ({potential_reduction/total_edges*100:.1f}%)")
        
        logger.info("=" * 80)
    
    def save_results(self, filename: str = "batch_timing_results.json"):
        """Save all timing results to a JSON file."""
        output_path = os.path.join(self.output_dir, filename)
        
        # Compute aggregate timestamp statistics
        total_ts = sum(r.total_timestamps for r in self.results)
        kde_ts = sum(r.kde_eligible_timestamps for r in self.results)
        
        results_dict = {
            'config': {
                'min_occurrences': self.min_occurrences,
                'num_kde_eligible_edges': len(self.kde_eligible_edges),
            },
            'summary': {
                'total_batches': len(self.results),
                'tainted_batches': len([r for r in self.results if r.kde_eligible_edges > 0]),
                'total_timestamps': total_ts,
                'kde_eligible_timestamps': kde_ts,
                'kde_timestamp_ratio': kde_ts / total_ts if total_ts > 0 else 0.0,
            },
            'results': [r.to_dict() for r in self.results],
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved timing results to {output_path}")
        return output_path
    
    def save_detailed_tainted_report(self, filename: str = "tainted_batches_report.json"):
        """Save detailed report of tainted batches only."""
        tainted = self.get_tainted_batches(min_taint_ratio=0.0)
        tainted = [r for r in tainted if r.kde_eligible_edges > 0]
        
        output_path = os.path.join(self.output_dir, filename)
        
        report = {
            'total_tainted_batches': len(tainted),
            'batches': []
        }
        
        for r in tainted:
            report['batches'].append({
                'batch_number': r.batch_id,
                'phase': r.phase,
                'epoch': r.epoch,
                'split': r.split,
                'total_edges': r.total_edges,
                'kde_eligible_edges': r.kde_eligible_edges,
                'edges_reduced': r.edges_that_could_be_reduced,
                'taint_ratio': r.taint_ratio,
                'forward_time_ms': r.forward_time_ms,
                'backward_time_ms': r.backward_time_ms,
                'total_time_ms': r.total_time_ms,
            })
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved tainted batches report to {output_path}")
        return output_path


def load_kde_eligible_edges(kde_vectors_path: str, device: Optional[torch.device] = None) -> Tuple[Set[Tuple[int, int, int]], Dict[Tuple[int, int, int], int]]:
    """
    Load KDE-eligible edges from precomputed vectors .pt file.
    
    Args:
        kde_vectors_path: Path to the KDE vectors .pt file
        device: Device to load tensors to (default: cuda if available)
        
    Returns:
        Tuple of (set of (src, dst, edge_type) edge keys, dict of edge occurrence counts)
    """
    if not os.path.exists(kde_vectors_path):
        logger.warning(f"KDE vectors file not found: {kde_vectors_path}")
        return set(), {}
    
    # Use GPU if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load .pt file
        data = torch.load(kde_vectors_path, map_location=device)
        edge_vectors = data.get('edge_vectors', {})
        metadata = data.get('metadata', {})
        
        edge_keys = set(edge_vectors.keys())
        
        # Try to get occurrence counts from metadata if available
        # Parse from string format "src,dst,edge_type" back to tuple
        edge_counts_raw = metadata.get('edge_occurrence_counts', {})
        edge_counts = {}
        for edge_str, count in edge_counts_raw.items():
            if isinstance(edge_str, str):
                parts = edge_str.split(',')
                if len(parts) == 3:
                    src, dst, et = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
                    edge_counts[(src, dst, et)] = count
                elif len(parts) == 2:
                    # Legacy format
                    src, dst = int(parts[0].strip()), int(parts[1].strip())
                    edge_counts[(src, dst, 0)] = count
            elif isinstance(edge_str, tuple):
                edge_counts[edge_str] = count
        
        logger.info(f"Loaded {len(edge_keys)} KDE-eligible edges from {kde_vectors_path} (device: {device})")
        return edge_keys, edge_counts
        
    except Exception as e:
        logger.error(f"Failed to load KDE vectors: {e}")
        return set(), {}


def load_edge_occurrence_counts(stats_path: str) -> Dict[Tuple[int, int, int], int]:
    """
    Load edge occurrence counts from a stats file.
    
    Args:
        stats_path: Path to the edge stats JSON file
        
    Returns:
        Dict mapping (src, dst, edge_type) to occurrence count
    """
    if not os.path.exists(stats_path):
        logger.warning(f"Stats file not found: {stats_path}")
        return {}
    
    try:
        with open(stats_path, 'r') as f:
            data = json.load(f)
        
        counts = {}
        for edge_str, count in data.get('edge_counts', {}).items():
            # Parse edge string "src,dst,edge_type" format
            parts = edge_str.strip('()').split(',')
            if len(parts) == 3:
                src, dst, et = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
                counts[(src, dst, et)] = count
            elif len(parts) == 2:
                # Legacy format (src, dst) - use edge_type=0 as default
                src, dst = int(parts[0].strip()), int(parts[1].strip())
                counts[(src, dst, 0)] = count
        
        return counts
        
    except Exception as e:
        logger.error(f"Failed to load edge counts: {e}")
        return {}


# Convenience functions for global tracking

def init_global_tracker(
    dataset_name: str,
    kde_vectors_dir: str = "kde_vectors",
    output_dir: str = "timing_results",
    device: Optional[torch.device] = None,
    min_occurrences: int = 10,
) -> BatchTimingTracker:
    """
    Initialize the global batch timing tracker.
    
    Args:
        dataset_name: Name of the dataset
        kde_vectors_dir: Directory containing KDE vectors
        output_dir: Directory to save timing results
        device: CUDA device for timing
        min_occurrences: Threshold for KDE eligibility
        
    Returns:
        BatchTimingTracker instance
    """
    kde_vectors_path = os.path.join(kde_vectors_dir, f"{dataset_name}_kde_vectors.pt")
    kde_edges, edge_counts = load_kde_eligible_edges(kde_vectors_path, device=device)
    
    tracker = BatchTimingTracker(
        kde_eligible_edges=kde_edges,
        edge_occurrence_counts=edge_counts,
        min_occurrences=min_occurrences,
        output_dir=output_dir,
        device=device,
    )
    
    _global_timing_state['enabled'] = True
    _global_timing_state['tracker'] = tracker
    
    return tracker


def get_global_tracker() -> Optional[BatchTimingTracker]:
    """Get the global batch timing tracker."""
    return _global_timing_state.get('tracker')


def is_tracking_enabled() -> bool:
    """Check if global timing tracking is enabled."""
    return _global_timing_state.get('enabled', False)
