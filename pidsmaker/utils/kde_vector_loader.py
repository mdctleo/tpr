"""
RKHS Vector Loader for KAIROS-KDE

This module provides utilities to load precomputed RKHS vectors from disk
and make them available during training with O(1) lookup.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class RKHSVectorLoader:
    """
    Loads and manages precomputed RKHS vectors for efficient lookup during training.
    """
    
    def __init__(self, dataset_name: str, kde_vectors_dir: str = "kde_vectors"):
        """
        Initialize the RKHS vector loader.
        
        Args:
            dataset_name: Name of the dataset (e.g., CLEARSCOPE_E3)
            kde_vectors_dir: Directory containing precomputed vectors
        """
        self.dataset_name = dataset_name
        self.kde_vectors_dir = kde_vectors_dir
        self.edge_vectors: Dict[Tuple[int, int], torch.Tensor] = {}
        self.metadata: Dict = {}
        self.device = None
        
        self._load_vectors()
    
    def _load_vectors(self):
        """Load RKHS vectors from disk."""
        vector_file = os.path.join(self.kde_vectors_dir, f"{self.dataset_name}_kde_vectors.pt")
        
        if not os.path.exists(vector_file):
            logger.warning(f"RKHS vector file not found: {vector_file}")
            logger.warning("KDE time encoding will use fallback encoder for all edges")
            return
        
        logger.info(f"Loading precomputed RKHS vectors from {vector_file}...")
        
        try:
            data = torch.load(vector_file, map_location='cpu')
            self.edge_vectors = data['edge_vectors']
            self.metadata = data['metadata']
            
            logger.info(f"Loaded {len(self.edge_vectors)} RKHS vectors")
            logger.info(f"Metadata: {self.metadata}")
            
            # Compute memory usage
            total_size_mb = sum(v.numel() * v.element_size() for v in self.edge_vectors.values()) / (1024 * 1024)
            logger.info(f"RKHS vectors memory usage: {total_size_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to load RKHS vectors: {e}")
            self.edge_vectors = {}
            self.metadata = {}
    
    def get_vector(self, src: int, dst: int) -> Optional[torch.Tensor]:
        """
        Get RKHS vector for an edge.
        
        Args:
            src: Source node ID
            dst: Destination node ID
            
        Returns:
            RKHS vector if available, None otherwise
        """
        edge_key = (src, dst)
        vector = self.edge_vectors.get(edge_key)
        
        if vector is not None and self.device is not None:
            # Move to correct device if needed
            if vector.device != self.device:
                vector = vector.to(self.device)
                # Update cache with device-moved vector
                self.edge_vectors[edge_key] = vector
        
        return vector
    
    def get_vectors_batch(self, src_batch: torch.Tensor, dst_batch: torch.Tensor) -> torch.Tensor:
        """
        Get RKHS vectors for a batch of edges.
        
        Args:
            src_batch: Batch of source node IDs
            dst_batch: Batch of destination node IDs
            
        Returns:
            Tensor of shape (batch_size, rkhs_dim) with RKHS vectors.
            Returns None for edges without precomputed vectors.
        """
        batch_size = src_batch.shape[0]
        rkhs_dim = self.metadata.get('rkhs_dim', 20)
        device = src_batch.device
        
        # Initialize output tensor
        vectors = torch.zeros(batch_size, rkhs_dim, device=device, dtype=torch.float32)
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Convert to CPU for dictionary lookup
        src_cpu = src_batch.cpu().numpy()
        dst_cpu = dst_batch.cpu().numpy()
        
        # Lookup vectors
        for i in range(batch_size):
            edge_key = (int(src_cpu[i]), int(dst_cpu[i]))
            vector = self.edge_vectors.get(edge_key)
            
            if vector is not None:
                vectors[i] = vector.to(device)
                mask[i] = True
        
        return vectors, mask
    
    def set_device(self, device: torch.device):
        """Set the device for RKHS vectors."""
        self.device = device
        logger.info(f"RKHS vector device set to: {device}")
    
    def has_vector(self, src: int, dst: int) -> bool:
        """Check if RKHS vector exists for an edge."""
        return (src, dst) in self.edge_vectors
    
    def get_coverage_stats(self) -> Dict:
        """Get statistics about RKHS vector coverage."""
        return {
            'num_edges_with_vectors': len(self.edge_vectors),
            'rkhs_dim': self.metadata.get('rkhs_dim', 20),
            'min_occurrences': self.metadata.get('min_occurrences', 10),
            'dataset': self.dataset_name
        }
    
    def __len__(self) -> int:
        """Return number of edges with precomputed vectors."""
        return len(self.edge_vectors)
    
    def __contains__(self, edge_key: Tuple[int, int]) -> bool:
        """Check if edge has precomputed vector."""
        return edge_key in self.edge_vectors


# Global loader instance (singleton pattern)
_global_loader: Optional[RKHSVectorLoader] = None


def get_rkhs_loader(dataset_name: Optional[str] = None, kde_vectors_dir: str = "kde_vectors") -> RKHSVectorLoader:
    """
    Get or create the global RKHS vector loader.
    
    Args:
        dataset_name: Name of the dataset (required on first call)
        kde_vectors_dir: Directory containing precomputed vectors
        
    Returns:
        RKHSVectorLoader instance
    """
    global _global_loader
    
    if _global_loader is None:
        if dataset_name is None:
            raise ValueError("dataset_name must be provided on first call to get_rkhs_loader()")
        _global_loader = RKHSVectorLoader(dataset_name, kde_vectors_dir)
    
    return _global_loader


def reset_rkhs_loader():
    """Reset the global RKHS loader (useful for testing)."""
    global _global_loader
    _global_loader = None
