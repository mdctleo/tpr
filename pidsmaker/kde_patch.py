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

from pidsmaker.utils.kde_vector_loader import RKHSVectorLoader

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

    Uses RKHSVectorLoader for proper loading of vectors saved by kde_computation.py
    in the format {'edge_vectors': {(src,dst): tensor}, 'metadata': dict}.
    """

    def __init__(
        self,
        out_channels: int,
        rkhs_dim: int = 20,
        dataset_name: Optional[str] = None,
        kde_vectors_dir: str = "kde_vectors",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.out_channels = out_channels
        self.rkhs_dim = rkhs_dim
        self.dataset_name = dataset_name
        self._device = device or torch.device('cpu')

        # Projection layer to map RKHS vectors to time encoding dimension
        self.rkhs_projection = nn.Linear(rkhs_dim, out_channels)

        # Fallback encoder for edges without KDE vectors
        self.fallback_encoder = nn.Linear(1, out_channels)

        # Load precomputed KDE vectors via RKHSVectorLoader
        self.rkhs_loader: Optional[RKHSVectorLoader] = None
        if dataset_name is not None:
            try:
                self.rkhs_loader = RKHSVectorLoader(dataset_name, kde_vectors_dir)
                if len(self.rkhs_loader) > 0:
                    logger.info(f"Loaded {len(self.rkhs_loader)} KDE vectors for {dataset_name}")
                    _kde_state['stats']['edges_with_kde'] = len(self.rkhs_loader)
                    _kde_state['vectors_loaded'] = True
                else:
                    logger.warning(f"No KDE vectors found for {dataset_name} in {kde_vectors_dir}")
            except Exception as e:
                logger.warning(f"Failed to load KDE vectors: {e}")

        # Move to device
        self.to(self._device)

    def reset_parameters(self):
        """Reset parameters of learnable layers."""
        self.rkhs_projection.reset_parameters()
        self.fallback_encoder.reset_parameters()

    def forward(self, src: torch.Tensor, dst: torch.Tensor, t_diff: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using precomputed KDE vectors when available.

        For edges with precomputed vectors:
            output = rkhs_projection(rkhs_vector)  # trainable
        For edges without precomputed vectors:
            output = Linear(t_diff).cos()           # trainable fallback

        Args:
            src: Source node IDs (batch_size,)
            dst: Destination node IDs (batch_size,)
            t_diff: Time differences (batch_size,)

        Returns:
            Time encodings of shape (batch_size, out_channels)
        """
        batch_size = t_diff.shape[0]
        device = t_diff.device

        # If no RKHS vectors loaded, use fallback for all
        if self.rkhs_loader is None or len(self.rkhs_loader) == 0:
            return self.fallback_encoder(t_diff.view(-1, 1)).cos()

        # Ensure loader vectors are on the correct device
        if self.rkhs_loader.device != device:
            self.rkhs_loader.set_device(device)

        # Batch lookup: get RKHS vectors and mask for which edges have precomputed vectors
        rkhs_vectors, mask = self.rkhs_loader.get_vectors_batch(src, dst)

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, device=device, dtype=torch.float32)

        # Project RKHS vectors for edges that have them (trainable)
        if mask.any():
            output[mask] = self.rkhs_projection(rkhs_vectors[mask])
            _kde_state['stats']['kde_count'] += mask.sum().item()

        # Use fallback for edges without RKHS vectors (trainable)
        if (~mask).any():
            fallback_input = t_diff[~mask].view(-1, 1)
            output[~mask] = self.fallback_encoder(fallback_input).cos()
            _kde_state['stats']['fallback_count'] += (~mask).sum().item()

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
        
        # Set up KDE vectors directory (resolve relative paths against project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kde_vectors_dir = kde_params.get('kde_vectors_dir', 'kde_vectors')
        if not os.path.isabs(kde_vectors_dir):
            kde_vectors_dir = os.path.join(base_dir, kde_vectors_dir)
        
        logger.info(f"Patching for KDE time encoding:")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  RKHS dim: {rkhs_dim}")
        logger.info(f"  Time dim: {time_dim}")
        logger.info(f"  KDE vectors dir: {kde_vectors_dir}")
        
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
                    kde_vectors_dir=kde_vectors_dir,
                    device=_kde_state.get('device', torch.device('cpu'))
                )
                self.out_channels = out_channels
            
            def reset_parameters(self):
                """Reset parameters."""
                if hasattr(self.kde_encoder, 'reset_parameters'):
                    self.kde_encoder.reset_parameters()
                
            def forward(self, t: torch.Tensor) -> torch.Tensor:
                # Standard cosine encoding for TGNMemory internal use
                # (src/dst-aware KDE encoding is injected via _patch_tgn_encoder_forward)
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
        
        def patched_forward(self, batch, inference=False, **kwargs):
            """Modified forward that passes src/dst to time encoder."""
            # Store in batch for time encoder access
            if hasattr(self, 'time_encoder') and hasattr(self.time_encoder, 'kde_encoder'):
                # Derive src/dst from TGN edge index mapped to original node IDs.
                # batch.edge_index_tgn has local indices; n_id_tgn maps them back
                # to global node IDs, matching the shape of rel_t in the encoder.
                edge_index_tgn = batch.edge_index_tgn
                n_id_tgn = batch.n_id_tgn
                tgn_src = n_id_tgn[edge_index_tgn[0]]
                tgn_dst = n_id_tgn[edge_index_tgn[1]]

                # Create a modified time encoding function
                original_time_enc_forward = self.time_encoder.forward
                
                def kde_time_forward(t_diff):
                    # Use the KDE encoder with src/dst information
                    return self.time_encoder.kde_encoder(tgn_src, tgn_dst, t_diff)
                
                # Temporarily replace the forward method
                self.time_encoder.forward = kde_time_forward
                
                # Call original forward
                result = original_forward(self, batch, inference=inference, **kwargs)
                
                # Restore original forward
                self.time_encoder.forward = original_time_enc_forward
                
                return result
            else:
                # No KDE encoder, use original forward
                return original_forward(self, batch, inference=inference, **kwargs)
        
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