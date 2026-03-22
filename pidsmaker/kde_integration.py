"""
Integration module for KDE-based time encoding in KAIROS.
This module provides factory functions and utilities to seamlessly integrate
KDE time encoding with minimal changes to existing code.
"""

import torch
import logging
from typing import Dict, Optional, Any

from pidsmaker.tgn import TimeEncoder as OriginalTimeEncoder
from pidsmaker.encoders.kde_time_encoder import KDETimeEncoder, FallbackTimeEncoder
from pidsmaker.encoders.tgn_kde_encoder import TGNKDEEncoder
from pidsmaker.encoders.tgn_encoder import TGNEncoder

logger = logging.getLogger(__name__)


def create_kde_time_encoder(config: Dict[str, Any]) -> KDETimeEncoder:
    """
    Create a KDE time encoder from configuration.
    
    Args:
        config: Configuration dictionary with KDE parameters
        
    Returns:
        Configured KDE time encoder
    """
    kde_params = config.get('kde_params', {})
    
    # Extract parameters with defaults
    out_channels = kde_params.get('time_dim', 100)
    rkhs_dim = kde_params.get('rkhs_dim', 100)
    min_occurrences = kde_params.get('min_occurrences', 10)
    bandwidth = kde_params.get('bandwidth', 'scott')
    n_quadrature_points = kde_params.get('n_quadrature_points', 50)
    
    # Create fallback encoder (original KAIROS time encoder)
    fallback_encoder = FallbackTimeEncoder(out_channels)
    
    logger.info(f"Creating KDE time encoder with parameters:")
    logger.info(f"  - RKHS dimension: {rkhs_dim}")
    logger.info(f"  - Min occurrences: {min_occurrences}")
    logger.info(f"  - Bandwidth: {bandwidth}")
    logger.info(f"  - Quadrature points: {n_quadrature_points}")
    
    kde_encoder = KDETimeEncoder(
        out_channels=out_channels,
        rkhs_dim=rkhs_dim,
        min_occurrences=min_occurrences,
        bandwidth=bandwidth,
        n_quadrature_points=n_quadrature_points,
        fallback_encoder=fallback_encoder
    )
    
    return kde_encoder


def create_tgn_kde_encoder(
    config: Dict[str, Any],
    node_features,
    edge_features,
    memory,
    gnn,
    min_dst_idx,
    neighbor_loader,
    device=None
) -> TGNKDEEncoder:
    """
    Create a TGN encoder with KDE-based time encoding.
    
    Args:
        config: Configuration dictionary
        node_features: Node feature dimensions
        edge_features: Edge feature list
        memory: TGN memory module
        gnn: GNN module
        min_dst_idx: Minimum destination index
        neighbor_loader: Neighbor loader for TGN
        device: Device for computation
        
    Returns:
        Configured TGN-KDE encoder
    """
    # Get encoder config
    encoder_config = config.get('encoder', {})
    tgn_config = encoder_config.get('tgn_kde', encoder_config.get('tgn', {}))
    
    # Create KDE time encoder if time_encoding is in edge features
    time_encoder = None
    if "time_encoding" in edge_features:
        time_encoder = create_kde_time_encoder(config)
        
    # Create TGN-KDE encoder
    encoder = TGNKDEEncoder(
        node_features=node_features,
        edge_features=edge_features,
        memory=memory,
        gnn=gnn,
        min_dst_idx=min_dst_idx,
        neighbor_loader=neighbor_loader,
        time_encoder=time_encoder,
        use_memory=tgn_config.get('use_memory', True),
        device=device,
        project_src_dst=tgn_config.get('project_src_dst', True),
        use_time_enc=False,  # Handled by KDE encoder
        use_time_order_encoding=tgn_config.get('use_time_order_encoding', False),
        gru=None  # Can be added if needed
    )
    
    return encoder


def patch_factory_for_kde():
    """
    Monkey-patch the factory module to support KDE encoders.
    This function should be called before using the factory.
    """
    import pidsmaker.factory as factory
    
    # Store original create_encoder function
    original_create_encoder = factory.create_encoder
    
    def create_encoder_with_kde(config, *args, **kwargs):
        """
        Extended encoder creation that supports KDE encoders.
        """
        encoder_methods = config.get('encoder', {}).get('used_methods', '')
        
        # Check if KDE encoder is requested
        if 'tgn_kde' in encoder_methods or config.get('used_method') == 'kde_enhanced':
            logger.info("Creating TGN encoder with KDE-based time encoding")
            
            # Extract necessary parameters from args/kwargs
            # This needs to match the signature of the original create_encoder
            node_features = kwargs.get('node_features', args[1] if len(args) > 1 else None)
            edge_features = kwargs.get('edge_features', args[2] if len(args) > 2 else None) 
            memory = kwargs.get('memory', args[3] if len(args) > 3 else None)
            gnn = kwargs.get('gnn', args[4] if len(args) > 4 else None)
            min_dst_idx = kwargs.get('min_dst_idx', args[5] if len(args) > 5 else 0)
            neighbor_loader = kwargs.get('neighbor_loader', args[6] if len(args) > 6 else None)
            device = kwargs.get('device', args[7] if len(args) > 7 else None)
            
            return create_tgn_kde_encoder(
                config=config,
                node_features=node_features,
                edge_features=edge_features,
                memory=memory,
                gnn=gnn,
                min_dst_idx=min_dst_idx,
                neighbor_loader=neighbor_loader,
                device=device
            )
        else:
            # Use original encoder creation
            return original_create_encoder(config, *args, **kwargs)
            
    # Replace the factory function
    factory.create_encoder = create_encoder_with_kde
    logger.info("Factory patched to support KDE encoders")


class KDETrainingHook:
    """
    Training hook to manage KDE encoder lifecycle.
    """
    
    def __init__(self, encoder: Optional[TGNKDEEncoder] = None):
        self.encoder = encoder
        
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        if self.encoder and hasattr(self.encoder, 'kde_time_encoder'):
            logger.debug(f"Epoch {epoch}: KDE encoder in training mode")
            
    def on_epoch_end(self, epoch: int):
        """Called at the end of each epoch."""
        if self.encoder and hasattr(self.encoder, 'kde_time_encoder'):
            stats = self.encoder.kde_time_encoder.get_statistics()
            logger.info(f"Epoch {epoch} - KDE Stats: {stats}")
            
    def on_training_end(self):
        """Called at the end of training."""
        if self.encoder and hasattr(self.encoder, 'finalize_training'):
            logger.info("Finalizing KDE encoder training")
            self.encoder.finalize_training()
            
    def on_training_start(self):
        """Called at the start of training."""
        if self.encoder and hasattr(self.encoder, 'reset_kde_encoder'):
            logger.info("Resetting KDE encoder for training")
            self.encoder.reset_kde_encoder()
