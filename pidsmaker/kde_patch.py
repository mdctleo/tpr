"""
KDE Patching Module for Orthrus with Precomputed KDE Time Encodings

This module patches the TGN encoder to use precomputed KDE vectors for time encoding
instead of computing them on-the-fly. It loads precomputed RKHS vectors from disk
and integrates them into the training pipeline.
"""

import logging
import os
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)

# Global storage for KDE vectors and state
_kde_vectors_cache = {}
_kde_state = {
    'vectors_loaded': False,
    'dataset_name': None,
    'device': None,
    'stats': {
        'total_edges': 0,
        'edges_with_kde': 0,
        'fallback_count': 0,
        'kde_count': 0
    }
}


class PrecomputedKDETimeEncoder(nn.Module):
    """
    Time encoder that uses precomputed KDE vectors loaded from disk.
    Falls back to standard cosine encoding for edges without KDE vectors.
    """
    
    def __init__(
        self,
        out_channels: int,
        rkhs_dim: int = 20,
        dataset_name: Optional[str] = None,
        kde_file_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.out_channels = out_channels
        self.rkhs_dim = rkhs_dim
        self.dataset_name = dataset_name
        self.device = device or torch.device('cpu')
        
        # Projection layer to map RKHS vectors to time encoding dimension
        self.rkhs_projection = nn.Linear(rkhs_dim, out_channels)
        
        # Fallback encoder for edges without KDE vectors
        self.fallback_encoder = nn.Linear(1, out_channels)
        
        # Load precomputed KDE vectors
        self.kde_vectors = {}
        self.load_kde_vectors(kde_file_path)
        
        # Move to device
        self.to(self.device)
    
    def reset_parameters(self):
        """Reset parameters of learnable layers."""
        self.rkhs_projection.reset_parameters()
        self.fallback_encoder.reset_parameters()
        
    def load_kde_vectors(self, kde_file_path: Optional[str] = None):
        """Load precomputed KDE vectors from disk."""
        if kde_file_path and os.path.exists(kde_file_path):
            try:
                data = torch.load(kde_file_path, map_location='cpu')
                
                # Convert edge tuples to string keys for faster lookup
                for edge_tuple, vector in data.items():
                    if isinstance(edge_tuple, tuple) and len(edge_tuple) == 2:
                        src, dst = edge_tuple
                        key = f"{src}_{dst}"
                        self.kde_vectors[key] = vector.to(self.device)
                
                logger.info(f"Loaded {len(self.kde_vectors)} KDE vectors from {kde_file_path}")
                _kde_state['edges_with_kde'] = len(self.kde_vectors)
                _kde_state['vectors_loaded'] = True
            except Exception as e:
                logger.warning(f"Failed to load KDE vectors from {kde_file_path}: {e}")
        else:
            logger.warning(f"KDE vectors file not found: {kde_file_path}")
    
    def forward(self, src: torch.Tensor, dst: torch.Tensor, t_diff: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using precomputed KDE vectors when available.
        
        Args:
            src: Source node IDs (batch_size,)
            dst: Destination node IDs (batch_size,)
            t_diff: Time differences (batch_size,)
            
        Returns:
            Time encodings of shape (batch_size, out_channels)
        """
        batch_size = t_diff.shape[0]
        device = t_diff.device
        
        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, device=device)
        
        # Process each edge in the batch
        kde_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        kde_vectors_batch = []
        
        for i in range(batch_size):
            src_id = src[i].item() if torch.is_tensor(src[i]) else src[i]
            dst_id = dst[i].item() if torch.is_tensor(dst[i]) else dst[i]
            edge_key = f"{src_id}_{dst_id}"
            
            if edge_key in self.kde_vectors:
                kde_mask[i] = True
                kde_vectors_batch.append(self.kde_vectors[edge_key])
                _kde_state['stats']['kde_count'] += 1
            else:
                _kde_state['stats']['fallback_count'] += 1
        
        # Apply KDE projection for edges with vectors
        if kde_mask.any():
            kde_vectors_tensor = torch.stack(kde_vectors_batch).to(device)
            output[kde_mask] = self.rkhs_projection(kde_vectors_tensor)
        
        # Use fallback for edges without KDE vectors
        if (~kde_mask).any():
            fallback_input = t_diff[~kde_mask].view(-1, 1)
            output[~kde_mask] = self.fallback_encoder(fallback_input).cos()
        
        return output


def patch_for_kde_time_encoding(cfg) -> bool:
    """
    Patch the model to use precomputed KDE time encoding.
    
    Args:
        cfg: Configuration object
        
    Returns:
        True if patching was successful, False otherwise
    """
    try:
        # Extract KDE parameters from config object
        kde_params = getattr(cfg, 'kde_params', None)
        if kde_params is None:
            logger.warning("No kde_params found in config, skipping KDE patching")
            return False
        
        # Convert to dict if it's an object
        if not isinstance(kde_params, dict):
            kde_params = dict(kde_params)
        
        dataset_name = cfg.dataset.name if hasattr(cfg, 'dataset') else 'CLEARSCOPE_E3'
        rkhs_dim = kde_params.get('rkhs_dim', 20)
        time_dim = kde_params.get('time_dim', 50)
        
        # Set up KDE file path - use absolute path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kde_file_path = os.path.join(base_dir, "kde_vectors", f"{dataset_name}_kde_vectors.pt")
        
        logger.info(f"Patching for KDE time encoding:")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  RKHS dim: {rkhs_dim}")
        logger.info(f"  Time dim: {time_dim}")
        logger.info(f"  KDE file: {kde_file_path}")
        
        # Store configuration in global state
        _kde_state['dataset_name'] = dataset_name
        
        # Monkey-patch the TGN TimeEncoder creation
        import pidsmaker.tgn as tgn_module
        original_time_encoder = tgn_module.TimeEncoder
        
        class KDEPatchedTimeEncoder(nn.Module):
            """Wrapper that intercepts TimeEncoder creation and replaces with KDE version."""
            
            def __init__(self, out_channels: int):
                super().__init__()
                self.kde_encoder = PrecomputedKDETimeEncoder(
                    out_channels=out_channels,
                    rkhs_dim=rkhs_dim,
                    dataset_name=dataset_name,
                    kde_file_path=kde_file_path,
                    device=_kde_state.get('device', torch.device('cpu'))
                )
                self.out_channels = out_channels
            
            def reset_parameters(self):
                """Reset parameters."""
                if hasattr(self.kde_encoder, 'reset_parameters'):
                    self.kde_encoder.reset_parameters()
                
            def forward(self, t: torch.Tensor) -> torch.Tensor:
                # This will be overridden in the actual usage
                return self.kde_encoder.fallback_encoder(t.view(-1, 1)).cos()
        
        # Replace the TimeEncoder class
        tgn_module.TimeEncoder = KDEPatchedTimeEncoder
        
        # Also patch the TGN encoder to pass src/dst to time encoder
        _patch_tgn_encoder_forward()
        
        logger.info("Successfully patched for KDE time encoding")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch for KDE time encoding: {e}")
        import traceback
        traceback.print_exc()
        return False


def _patch_tgn_encoder_forward():
    """Patch TGN encoder forward method to pass src/dst to time encoder."""
    try:
        import pidsmaker.encoders.tgn_encoder as tgn_encoder_module
        
        # Store original forward method
        original_forward = tgn_encoder_module.TGNEncoder.forward
        
        def patched_forward(self, batch):
            """Modified forward that passes src/dst to time encoder."""
            # Get src, dst from batch
            src, dst = batch.src, batch.dst
            
            # Store in batch for time encoder access
            if hasattr(self, 'time_encoder') and hasattr(self.time_encoder, 'kde_encoder'):
                # Create a modified time encoding function
                original_time_enc_forward = self.time_encoder.forward
                
                def kde_time_forward(t_diff):
                    # Use the KDE encoder with src/dst information
                    return self.time_encoder.kde_encoder(src, dst, t_diff)
                
                # Temporarily replace the forward method
                self.time_encoder.forward = kde_time_forward
                
                # Call original forward
                result = original_forward(self, batch)
                
                # Restore original forward
                self.time_encoder.forward = original_time_enc_forward
                
                return result
            else:
                # No KDE encoder, use original forward
                return original_forward(self, batch)
        
        # Replace the forward method
        tgn_encoder_module.TGNEncoder.forward = patched_forward
        
        logger.info("Successfully patched TGN encoder forward method")
        
    except Exception as e:
        logger.warning(f"Failed to patch TGN encoder forward: {e}")


def reset_kde_state(model: nn.Module):
    """Reset KDE state for a fresh training run."""
    _kde_state['stats'] = {
        'total_edges': 0,
        'edges_with_kde': _kde_state.get('edges_with_kde', 0),
        'fallback_count': 0,
        'kde_count': 0
    }
    logger.info("KDE state reset")


def finalize_kde_training(model: nn.Module):
    """Finalize KDE training (no-op for precomputed vectors)."""
    logger.info("KDE training finalized (using precomputed vectors)")
    log_kde_debug_stats(model, -1, "final")


def log_kde_debug_stats(model: nn.Module, epoch: int, phase: str):
    """Log debug statistics about KDE usage."""
    stats = _kde_state['stats']
    total_uses = stats['kde_count'] + stats['fallback_count']
    
    if total_uses > 0:
        kde_ratio = stats['kde_count'] / total_uses * 100
        logger.info(f"[Epoch {epoch}] KDE Stats ({phase}):")
        logger.info(f"  - Edges with KDE vectors: {stats['edges_with_kde']}")
        logger.info(f"  - KDE uses: {stats['kde_count']} ({kde_ratio:.1f}%)")
        logger.info(f"  - Fallback uses: {stats['fallback_count']} ({100-kde_ratio:.1f}%)")
        logger.info(f"  - Vectors loaded: {_kde_state['vectors_loaded']}")