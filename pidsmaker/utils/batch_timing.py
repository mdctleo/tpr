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


def extract_edge_type_from_msg(msg: torch.Tensor, node_type_dim: int = 8, edge_type_dim: int = 16, return_tensor: bool = True) -> torch.Tensor:
    """
    Extract edge type indices from the msg tensor.
    
    The msg tensor structure is: [src_type, src_emb, edge_type, dst_type, dst_emb]
    where edge_type is one-hot encoded at position (node_type_dim + emb_dim).
    
    Args:
        msg: Message tensor of shape (N, msg_dim)
        node_type_dim: Number of node type dimensions (default 8 for DARPA E3)
        edge_type_dim: Number of edge type dimensions (default 16 for DARPA E3)
        return_tensor: If True, return a torch.Tensor (stays on same device as msg).
                       If False, return numpy array on CPU (legacy behavior).
        
    Returns:
        Tensor/Array of edge type indices (argmax of the one-hot edge type portion)
    """
    msg_dim = msg.shape[1]
    
    # Calculate emb_dim from msg structure:
    # msg_dim = node_type_dim + emb_dim + edge_type_dim + node_type_dim + emb_dim
    # msg_dim = 2 * node_type_dim + 2 * emb_dim + edge_type_dim
    emb_dim = (msg_dim - 2 * node_type_dim - edge_type_dim) // 2
    
    # Edge type starts at: node_type_dim + emb_dim
    edge_type_start = node_type_dim + emb_dim
    edge_type_end = edge_type_start + edge_type_dim
    
    # Extract edge type slice and get argmax (stays on same device as msg)
    edge_type_slice = msg[:, edge_type_start:edge_type_end]
    edge_types = edge_type_slice.argmax(dim=1)
    
    if return_tensor:
        return edge_types
    else:
        return edge_types.cpu().numpy()


def build_kde_edge_tensor(kde_eligible_edges: Set[Tuple[int, int, int]], device: torch.device = None) -> torch.Tensor:
    """
    Convert KDE eligible edges set to a tensor for GPU-accelerated lookups.
    
    Args:
        kde_eligible_edges: Set of (src, dst, edge_type) tuples
        device: Target device (default: cuda if available, else cpu)
        
    Returns:
        Tensor of shape (N, 3) containing [src, dst, edge_type] for each edge
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not kde_eligible_edges:
        return torch.empty((0, 3), dtype=torch.long, device=device)
    
    edge_list = list(kde_eligible_edges)
    edge_tensor = torch.tensor(edge_list, dtype=torch.long, device=device)
    return edge_tensor


def build_kde_edge_hash_tensor(kde_eligible_edges: Set[Tuple[int, int, int]], device: torch.device = None) -> torch.Tensor:
    """
    Build a sorted tensor of edge hashes for efficient GPU binary search lookup.
    
    Uses a hash function to combine (src, dst, edge_type) into a single int64.
    This enables O(N log M) lookup instead of O(N × M) for broadcasting.
    
    Args:
        kde_eligible_edges: Set of (src, dst, edge_type) tuples
        device: Target device (default: cuda if available, else cpu)
        
    Returns:
        Sorted tensor of edge hashes (int64)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not kde_eligible_edges:
        return torch.empty((0,), dtype=torch.long, device=device)
    
    # Hash function: combine src, dst, edge_type into a single int64
    # Using bit shifts: hash = (src << 40) | (dst << 16) | edge_type
    # This works for node IDs up to 2^24 (~16M) and edge types up to 2^16 (~65K)
    hashes = []
    for src, dst, et in kde_eligible_edges:
        h = (src << 40) | (dst << 16) | et
        hashes.append(h)
    
    hash_tensor = torch.tensor(hashes, dtype=torch.long, device=device)
    # Sort for binary search
    hash_tensor = hash_tensor.sort().values
    return hash_tensor


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
    
    GPU Optimization: KDE-eligible edges are stored as a tensor for vectorized lookups.
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
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.results: List[BatchTimingResult] = []
        self.batch_counter = 0
        self.current_epoch = 0
        self.current_split = 'train'
        
        # CUDA timing events
        self._use_cuda = self.device.type == 'cuda'
        if self._use_cuda:
            self._starter = torch.cuda.Event(enable_timing=True)
            self._ender = torch.cuda.Event(enable_timing=True)
        
        # Summary statistics
        self._summary = defaultdict(lambda: defaultdict(list))
        
        # Timing state for start_batch/end_batch pattern
        self._batch_start_time = None
        self._batch_cuda_start = None
        
        # GPU-accelerated KDE edge tensor for vectorized lookups
        self._kde_edge_tensor = build_kde_edge_tensor(self.kde_eligible_edges, self.device)
        
        # Hash-based lookup tensor for memory-efficient O(N log M) search
        self._kde_edge_hashes = build_kde_edge_hash_tensor(self.kde_eligible_edges, self.device)
        
        # Pre-compute edge occurrence counts tensor for edges that could be reduced
        self._edge_counts_tensor = None
        # Map from hash to count for efficient lookup
        self._edge_hash_to_count = {}
        if edge_occurrence_counts:
            counts = [edge_occurrence_counts.get(e, 1) for e in self.kde_eligible_edges]
            self._edge_counts_tensor = torch.tensor(counts, dtype=torch.long, device=self.device)
            # Build hash-to-count mapping for hash-based lookup
            for (src, dst, et), count in edge_occurrence_counts.items():
                h = (src << 40) | (dst << 16) | et
                self._edge_hash_to_count[h] = count
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"BatchTimingTracker initialized with {len(self.kde_eligible_edges)} KDE-eligible edges (device: {self.device})")
    
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
    
    def _find_kde_eligible_hash_based(self, src: torch.Tensor, dst: torch.Tensor, edge_types: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Memory-efficient hash-based method to find KDE-eligible edges using searchsorted.
        
        Uses O(N log M) time and O(N) memory instead of O(N × M) memory for broadcasting.
        
        Args:
            src: Source node tensor (N,) on GPU
            dst: Destination node tensor (N,) on GPU
            edge_types: Edge type tensor (N,) on GPU
            
        Returns:
            Tuple of (boolean mask of KDE-eligible edges, count of edges that could be reduced)
        """
        if len(self._kde_edge_hashes) == 0:
            return torch.zeros(len(src), dtype=torch.bool, device=src.device), 0
        
        # Compute hashes for batch edges: hash = (src << 40) | (dst << 16) | edge_type
        batch_hashes = (src.long() << 40) | (dst.long() << 16) | edge_types.long()
        
        # Use searchsorted for O(log M) lookup per edge
        indices = torch.searchsorted(self._kde_edge_hashes, batch_hashes)
        
        # Check if the found index contains the actual hash (searchsorted gives insertion point)
        # Clamp indices to valid range
        indices_clamped = indices.clamp(max=len(self._kde_edge_hashes) - 1)
        kde_eligible_mask = (self._kde_edge_hashes[indices_clamped] == batch_hashes)
        
        # Count edges that could be reduced (KDE-eligible with count > 1)
        edges_could_reduce = 0
        if self._edge_hash_to_count and kde_eligible_mask.any():
            # Get eligible edge hashes and check counts
            eligible_hashes = batch_hashes[kde_eligible_mask].cpu().numpy()
            for h in eligible_hashes:
                if self._edge_hash_to_count.get(int(h), 1) > 1:
                    edges_could_reduce += 1
        
        return kde_eligible_mask, edges_could_reduce
    
    def _find_kde_eligible_vectorized(self, src: torch.Tensor, dst: torch.Tensor, edge_types: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        GPU-accelerated method to find KDE-eligible edges.
        
        Automatically chooses between hash-based (memory-efficient) and broadcasting
        (faster for small batches) based on expected memory usage.
        
        Args:
            src: Source node tensor (N,) on GPU
            dst: Destination node tensor (N,) on GPU
            edge_types: Edge type tensor (N,) on GPU
            
        Returns:
            Tuple of (boolean mask of KDE-eligible edges, count of edges that could be reduced)
        """
        batch_size = len(src)
        kde_size = len(self._kde_edge_tensor)
        
        # Use hash-based for large memory footprint (N × M > 10M elements)
        # or always use hash-based as it's more memory-safe
        if batch_size * kde_size > 10_000_000 or kde_size > 5000:
            return self._find_kde_eligible_hash_based(src, dst, edge_types)
        
        if kde_size == 0:
            return torch.zeros(batch_size, dtype=torch.bool, device=src.device), 0
        
        # Stack batch edges into (N, 3) tensor: [src, dst, edge_type]
        batch_edges = torch.stack([src, dst, edge_types], dim=1)  # (N, 3)
        
        # Expand dimensions for broadcasting comparison:
        # batch_edges: (N, 1, 3), kde_edges: (1, M, 3) -> comparison: (N, M, 3)
        # Then check if all 3 values match (all() along dim=2) -> (N, M)
        # Then check if any KDE edge matches (any() along dim=1) -> (N,)
        batch_expanded = batch_edges.unsqueeze(1)  # (N, 1, 3)
        kde_expanded = self._kde_edge_tensor.unsqueeze(0)  # (1, M, 3)
        
        # Element-wise comparison and reduce
        matches = (batch_expanded == kde_expanded).all(dim=2)  # (N, M) - True where all 3 match
        kde_eligible_mask = matches.any(dim=1)  # (N,) - True if any KDE edge matches
        
        # Count edges that could be reduced (KDE-eligible with count > 1)
        edges_could_reduce = 0
        if self._edge_counts_tensor is not None and kde_eligible_mask.any():
            # For each KDE-eligible batch edge, find which KDE edge it matches
            # and check if that edge has count > 1
            # Cast bool to int for argmax (CUDA doesn't support argmax on bool)
            kde_indices = matches.int().argmax(dim=1)  # (N,) - index of matching KDE edge (0 if no match)
            kde_counts = self._edge_counts_tensor[kde_indices]  # (N,) - count for each match
            can_reduce = kde_eligible_mask & (kde_counts > 1)
            edges_could_reduce = can_reduce.sum().item()
        
        return kde_eligible_mask, edges_could_reduce

    def analyze_batch(self, batch) -> Dict[str, Any]:
        """
        Analyze a batch to determine KDE eligibility, taint ratio, and timestamp counts.
        
        Uses GPU-accelerated vectorized operations when possible.
        
        Args:
            batch: Batch object with original_edge_index, edge_type attributes
            
        Returns:
            Dict with analysis results including timestamp counts
        """
        # Determine device to use (prefer GPU if available)
        device = self.device
                
        # Get source and destination node IDs from ORIGINAL edge_index (global node IDs)
        # This is critical because KDE vectors use global node IDs, not reindexed local IDs
        # Keep tensors on their original device for GPU-accelerated processing
        if hasattr(batch, 'original_edge_index') and batch.original_edge_index is not None:
            edge_index = batch.original_edge_index
            if isinstance(edge_index, torch.Tensor):
                src, dst = edge_index[0].to(device), edge_index[1].to(device)
            else:
                src = torch.tensor(edge_index[0], dtype=torch.long, device=device)
                dst = torch.tensor(edge_index[1], dtype=torch.long, device=device)
        elif hasattr(batch, 'src') and hasattr(batch, 'dst'):
            if isinstance(batch.src, torch.Tensor):
                src, dst = batch.src.to(device), batch.dst.to(device)
            else:
                src = torch.tensor(batch.src, dtype=torch.long, device=device)
                dst = torch.tensor(batch.dst, dtype=torch.long, device=device)
        elif hasattr(batch, 'edge_index'):
            edge_index = batch.edge_index
            if isinstance(edge_index, torch.Tensor):
                src, dst = edge_index[0].to(device), edge_index[1].to(device)
            else:
                src = torch.tensor(edge_index[0], dtype=torch.long, device=device)
                dst = torch.tensor(edge_index[1], dtype=torch.long, device=device)
        else:
            logger.warning("Batch has no original_edge_index, src/dst or edge_index attributes")
            return {'total_edges': 0, 'kde_eligible': 0, 'non_kde': 0, 'taint_ratio': 0.0,
                    'edges_could_reduce': 0, 'total_timestamps': 0, 'kde_eligible_timestamps': 0}
        
        # Get edge types - try multiple sources in order of preference:
        # 1. msg tensor (RAW edge type embedded in the message for DARPA E3 datasets)
        #    IMPORTANT: Use msg FIRST because batch.edge_type may be triplet-encoded (remapped)
        #    but KDE vectors use RAW edge types from the original dataset
        # 2. edge_type attribute (may be triplet-encoded, fallback only)
        # 3. Default to 0
        if hasattr(batch, 'msg') and batch.msg is not None:
            # Extract edge types from msg tensor (for DARPA E3 datasets)
            # The msg tensor contains: [src_type, src_emb, edge_type, dst_type, dst_emb]
            try:
                msg = batch.msg
                if isinstance(msg, torch.Tensor):
                    # Use default dimensions for DARPA E3: node_type_dim=8, edge_type_dim=16
                    # return_tensor=True keeps it on GPU
                    edge_types = extract_edge_type_from_msg(msg, node_type_dim=8, edge_type_dim=16, return_tensor=True)
                    edge_types = edge_types.to(device)
                else:
                    edge_types = torch.zeros(len(src), dtype=torch.long, device=device)
            except Exception as e:
                logger.warning(f"Failed to extract edge types from msg tensor: {e}")
                edge_types = torch.zeros(len(src), dtype=torch.long, device=device)
        elif hasattr(batch, 'edge_type') and batch.edge_type is not None:
            # Fallback to edge_type attribute (may be triplet-encoded)
            edge_type = batch.edge_type
            if isinstance(edge_type, torch.Tensor):
                # edge_type is one-hot encoded, get the index
                if edge_type.ndim == 2:
                    edge_types = edge_type.max(dim=1).indices.to(device)
                else:
                    edge_types = edge_type.to(device)
            else:
                edge_types = torch.tensor(edge_type, dtype=torch.long, device=device)
        else:
            # Default to edge type 0 if not available
            edge_types = torch.zeros(len(src), dtype=torch.long, device=device)
        
        total_edges = len(src)
        
        # Use GPU-accelerated vectorized lookup
        kde_eligible_mask, edges_could_reduce = self._find_kde_eligible_vectorized(src, dst, edge_types)
        kde_eligible = kde_eligible_mask.sum().item()
        
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
