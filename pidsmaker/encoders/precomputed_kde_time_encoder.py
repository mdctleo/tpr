"""
Precomputed KDE Time Encoder for KAIROS-KDE

This encoder uses precomputed RKHS vectors loaded from disk instead of
computing KDE vectors during training. This is much more efficient for
large-scale datasets with 100K-200K edges.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from pidsmaker.utils.kde_vector_loader import get_rkhs_loader

logger = logging.getLogger(__name__)


class PrecomputedKDETimeEncoder(nn.Module):
    """
    Time encoder that uses precomputed RKHS vectors for frequent edges
    and falls back to standard time encoding for rare edges.
    """
    
    def __init__(
        self,
        out_channels: int,
        rkhs_dim: int = 20,
        dataset_name: Optional[str] = None,
        kde_vectors_dir: str = "kde_vectors",
        fallback_encoder: Optional[nn.Module] = None
    ):
        """
        Initialize the precomputed KDE time encoder.
        
        Args:
            out_channels: Output dimension (time encoding dimension)
            rkhs_dim: Dimension of RKHS vectors
            dataset_name: Name of the dataset for loading vectors
            kde_vectors_dir: Directory containing precomputed vectors
            fallback_encoder: Encoder to use for edges without RKHS vectors
        """
        super().__init__()
        self.out_channels = out_channels
        self.rkhs_dim = rkhs_dim
        self.dataset_name = dataset_name
        
        # RKHS projection layer (trainable)
        self.rkhs_projection = nn.Linear(rkhs_dim, out_channels)
        
        # Fallback encoder for rare edges
        if fallback_encoder is None:
            self.fallback_encoder = FallbackTimeEncoder(out_channels)
        else:
            self.fallback_encoder = fallback_encoder
        
        # Load precomputed RKHS vectors
        self.rkhs_loader = None
        if dataset_name is not None:
            try:
                self.rkhs_loader = get_rkhs_loader(dataset_name, kde_vectors_dir)
                logger.info(f"Loaded RKHS vectors for {len(self.rkhs_loader)} edges")
            except Exception as e:
                logger.warning(f"Failed to load RKHS vectors: {e}")
                logger.warning("Will use fallback encoder for all edges")
        
        # Mark as KDE encoder for compatibility
        self.is_kde = True
    
    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        time_diffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass using precomputed RKHS vectors.
        
        For edges with precomputed vectors:
            output = rkhs_projection(rkhs_vector)  # trainable projection
        For edges without precomputed vectors:
            output = fallback_encoder(t_diff)       # trainable fallback
        
        Args:
            src: Source node IDs [batch_size]
            dst: Destination node IDs [batch_size]
            time_diffs: Relative time differences [batch_size]
            
        Returns:
            Time encoding of shape (batch_size, out_channels)
        """
        batch_size = time_diffs.shape[0]
        device = time_diffs.device
        
        # Ensure fallback encoder is on correct device
        if next(self.fallback_encoder.parameters()).device != device:
            self.fallback_encoder = self.fallback_encoder.to(device)
        
        # If no RKHS vectors loaded, use fallback for all
        if self.rkhs_loader is None or len(self.rkhs_loader) == 0:
            return self.fallback_encoder(time_diffs)
        
        # Set device for RKHS loader
        if self.rkhs_loader.device != device:
            self.rkhs_loader.set_device(device)
        
        # Batch lookup: get RKHS vectors and mask for which edges have precomputed vectors
        rkhs_vectors, mask = self.rkhs_loader.get_vectors_batch(src, dst)
        
        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, device=device, dtype=torch.float32)
        
        # Project RKHS vectors for edges that have them (trainable)
        if mask.any():
            output[mask] = self.rkhs_projection(rkhs_vectors[mask])
        
        # Use fallback for edges without RKHS vectors (trainable)
        if (~mask).any():
            fallback_output = self.fallback_encoder(time_diffs[~mask])
            output[~mask] = fallback_output
        
        return output
    
    def forward_with_edges(self, src, dst, time_diffs):
        """Alias for forward() — kept for backward compatibility."""
        return self.forward(src, dst, time_diffs)
    
    def get_statistics(self):
        """Get statistics about RKHS vector usage."""
        if self.rkhs_loader is not None:
            return self.rkhs_loader.get_coverage_stats()
        return {'num_edges_with_vectors': 0}


class FallbackTimeEncoder(nn.Module):
    """
    Standard time encoder used as fallback for rare edges.
    Matches the original KAIROS TimeEncoder behavior.
    """
    
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(1, out_channels)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using cosine time encoding.
        
        Args:
            t: Time differences of shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Time encoding of shape (batch_size, out_channels)
        """
        return self.lin(t.view(-1, 1)).cos()
