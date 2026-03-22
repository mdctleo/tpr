"""
Reduced Temporal Encoding Module for Graphs with Collapsed Edges

This module provides time encoding based on precomputed temporal summary features
stored directly in the reduced graphs (edge_temporal_features attribute).

The temporal features are: (first_ts_norm, last_ts_norm, count_norm)
- first_ts_norm: First timestamp, normalized to [0,1] relative to graph time range
- last_ts_norm: Last timestamp, normalized to [0,1] relative to graph time range  
- count_norm: Log-scaled count of occurrences, normalized

Unlike KDE encoding, this approach:
- Does NOT require external .pt files with precomputed vectors
- Works with ALL edges (not just those with >= min_occurrences)
- Features are stored directly in the graph data object
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Global state for reduced encoding
_reduced_state = {
    'enabled': False,
    'stats': {
        'total_edges': 0,
        'encoding_calls': 0,
    }
}


class ReducedTemporalEncoder(nn.Module):
    """
    Time encoder that uses precomputed temporal summary features from reduced graphs.
    
    Input: edge_temporal_features of shape (batch_size, 3)
           containing (first_ts_norm, last_ts_norm, count_norm)
    Output: Time encodings of shape (batch_size, out_channels)
    """
    
    def __init__(
        self,
        out_channels: int,
        feature_dim: int = 3,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.out_channels = out_channels
        self.feature_dim = feature_dim
        self._device = device or torch.device('cpu')
        
        # Projection layer: 3-dim features -> out_channels
        self.projection = nn.Linear(feature_dim, out_channels)
        
        # Move to device
        self.to(self._device)
        
        logger.info(f"ReducedTemporalEncoder initialized: {feature_dim} -> {out_channels}")
    
    def reset_parameters(self):
        """Reset parameters of learnable layers."""
        self.projection.reset_parameters()
    
    def forward(self, edge_temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using temporal summary features.
        
        Args:
            edge_temporal_features: Tensor of shape (batch_size, 3)
                                   containing (first_ts_norm, last_ts_norm, count_norm)
        
        Returns:
            Time encodings of shape (batch_size, out_channels)
        """
        _reduced_state['stats']['encoding_calls'] += 1
        _reduced_state['stats']['total_edges'] += edge_temporal_features.shape[0]
        
        # Project features to output dimension
        output = self.projection(edge_temporal_features)
        
        return output


def patch_for_reduced_time_encoding(cfg) -> bool:
    """
    Patch the model to use reduced temporal encoding from edge_temporal_features.
    
    This replaces the TGN TimeEncoder with ReducedTemporalEncoder that reads
    directly from batch.edge_temporal_features_tgn.
    
    Args:
        cfg: Configuration object with reduced_params
        
    Returns:
        True if patching was successful, False otherwise
    """
    try:
        # Extract reduced params from config
        reduced_params = getattr(cfg, 'reduced_params', None)
        if reduced_params is None:
            logger.warning("No reduced_params found in config, skipping reduced encoding patch")
            return False
        
        # Check if reduced graphs are enabled
        use_reduced = getattr(reduced_params, 'use_reduced_graphs', False)
        if not use_reduced:
            logger.info("use_reduced_graphs is False, skipping reduced encoding patch")
            return False
        
        feature_dim = getattr(reduced_params, 'feature_dim', 3)
        time_dim = getattr(reduced_params, 'time_dim', 50)
        
        logger.info(f"Patching for reduced temporal encoding:")
        logger.info(f"  Feature dim: {feature_dim}")
        logger.info(f"  Time dim: {time_dim}")
        
        # Mark reduced encoding as enabled
        _reduced_state['enabled'] = True
        
        # Monkey-patch the TGN TimeEncoder creation
        import pidsmaker.tgn as tgn_module
        
        class ReducedPatchedTimeEncoder(nn.Module):
            """Wrapper that creates ReducedTemporalEncoder instead of standard TimeEncoder."""
            
            def __init__(self, out_channels: int):
                super().__init__()
                self.reduced_encoder = ReducedTemporalEncoder(
                    out_channels=out_channels,
                    feature_dim=feature_dim,
                    device=torch.device('cpu')  # Will be moved to correct device later
                )
                self.out_channels = out_channels
                
                # Fallback for when edge_temporal_features not available
                # (e.g., during TGN memory internal operations)
                self.fallback_linear = nn.Linear(1, out_channels)
            
            def reset_parameters(self):
                """Reset parameters."""
                self.reduced_encoder.reset_parameters()
                self.fallback_linear.reset_parameters()
            
            def forward(self, t: torch.Tensor) -> torch.Tensor:
                """
                Standard forward for TGN internal use (memory module).
                Uses cosine encoding as fallback.
                """
                return self.fallback_linear(t.view(-1, 1)).cos()
        
        # Replace the TimeEncoder class
        tgn_module.TimeEncoder = ReducedPatchedTimeEncoder
        
        # Patch the TGN encoder to use edge_temporal_features
        _patch_tgn_encoder_for_reduced()
        
        logger.info("Successfully patched for reduced temporal encoding")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch for reduced temporal encoding: {e}")
        import traceback
        traceback.print_exc()
        return False


def _patch_tgn_encoder_for_reduced():
    """Patch TGN encoder forward method to use edge_temporal_features."""
    try:
        import pidsmaker.encoders.tgn_encoder as tgn_encoder_module
        
        # Store original forward method
        original_forward = tgn_encoder_module.TGNEncoder.forward
        
        def patched_forward(self, batch, inference=False, **kwargs):
            """Modified forward that uses edge_temporal_features for time encoding."""
            
            # Check if this encoder has the patched time_encoder with reduced_encoder
            if (hasattr(self, 'time_encoder') and 
                hasattr(self.time_encoder, 'reduced_encoder') and
                hasattr(batch, 'edge_temporal_features_tgn') and 
                batch.edge_temporal_features_tgn is not None):
                
                # Store original time_encoder forward
                original_time_enc_forward = self.time_encoder.forward
                
                def reduced_time_forward(t_diff):
                    """Use edge_temporal_features instead of t_diff."""
                    features = batch.edge_temporal_features_tgn
                    
                    # Check shape compatibility
                    if features.shape[0] != t_diff.shape[0]:
                        # Shape mismatch (e.g., TGN memory internal call)
                        # Use fallback
                        return self.time_encoder.fallback_linear(t_diff.view(-1, 1)).cos()
                    
                    # Use reduced encoder
                    return self.time_encoder.reduced_encoder(features)
                
                # Temporarily replace forward
                self.time_encoder.forward = reduced_time_forward
                
                # Call original forward
                result = original_forward(self, batch, inference=inference, **kwargs)
                
                # Restore original forward
                self.time_encoder.forward = original_time_enc_forward
                
                return result
            else:
                # No reduced encoding, use original
                return original_forward(self, batch, inference=inference, **kwargs)
        
        # Replace the forward method
        tgn_encoder_module.TGNEncoder.forward = patched_forward
        
        logger.info("Successfully patched TGN encoder for reduced temporal features")
        
    except Exception as e:
        logger.warning(f"Failed to patch TGN encoder for reduced: {e}")


def reset_reduced_state():
    """Reset reduced encoding state."""
    _reduced_state['stats'] = {
        'total_edges': 0,
        'encoding_calls': 0,
    }
    logger.info("Reduced encoding state reset")


def log_reduced_stats():
    """Log statistics about reduced encoding usage."""
    stats = _reduced_state['stats']
    logger.info(f"Reduced Encoding Stats:")
    logger.info(f"  - Enabled: {_reduced_state['enabled']}")
    logger.info(f"  - Encoding calls: {stats['encoding_calls']}")
    logger.info(f"  - Total edges encoded: {stats['total_edges']}")


def is_reduced_encoding_enabled() -> bool:
    """Check if reduced encoding is enabled."""
    return _reduced_state['enabled']
