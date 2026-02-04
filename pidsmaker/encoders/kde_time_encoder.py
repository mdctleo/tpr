"""
KDE-based Time Encoder for KAIROS
Represents temporal patterns using Kernel Density Estimation in RKHS
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import logging
from scipy import integrate
from scipy.stats import gaussian_kde
from numpy.polynomial.hermite import hermgauss

logger = logging.getLogger(__name__)


class KDETimeEncoder(nn.Module):
    """
    Hybrid time encoder that uses KDE for frequent edges and fallback for rare edges.
    
    Args:
        out_channels: Output dimension for time encoding
        rkhs_dim: Dimension of RKHS vector representation
        min_occurrences: Minimum edge occurrences to use KDE (default: 10)
        bandwidth: KDE bandwidth parameter (default: 'scott')
        n_quadrature_points: Number of Gauss-Hermite quadrature points (default: 50)
        fallback_encoder: Fallback encoder for rare edges
    """
    
    def __init__(
        self,
        out_channels: int,
        rkhs_dim: int = 100,
        min_occurrences: int = 10,
        bandwidth: str = 'scott',
        n_quadrature_points: int = 50,
        fallback_encoder: Optional[nn.Module] = None
    ):
        super().__init__()
        self.out_channels = out_channels
        self.rkhs_dim = rkhs_dim
        self.min_occurrences = min_occurrences
        self.bandwidth = bandwidth
        self.n_quadrature_points = n_quadrature_points
        
        # Fallback encoder for rare edges (standard TimeEncoder)
        if fallback_encoder is None:
            self.fallback_encoder = FallbackTimeEncoder(out_channels)
        else:
            self.fallback_encoder = fallback_encoder
            
        # RKHS projection layer
        self.rkhs_projection = nn.Linear(rkhs_dim, out_channels)
        
        # Storage for edge statistics during training
        self.edge_timestamps = defaultdict(list)
        self.edge_kde_vectors = {}  # Precomputed RKHS vectors
        self.edge_counts = defaultdict(int)
        
        # Gauss-Hermite quadrature points and weights (will be moved to device later)
        quad_points, quad_weights = hermgauss(n_quadrature_points)
        self.register_buffer('quad_points', torch.tensor(quad_points, dtype=torch.float32))
        self.register_buffer('quad_weights', torch.tensor(quad_weights, dtype=torch.float32))
        
        # Training mode flag
        self.training_phase = True
        
        # Safety mechanism for memory management (heavily reduced for large graphs)
        self.max_total_timestamps = 180000  # Reduced from 3M to prevent segfault
        self.max_unique_edges = 18000  # Cap on unique edges to track
        self.total_timestamps_collected = 0
        self.collection_disabled = False
        self._gc_counter = 0  # Counter for periodic garbage collection
        
    def collect_timestamps(self, edge_ids: torch.Tensor, timestamps: torch.Tensor):
        """
        Collect timestamps for each edge during training.
        
        Args:
            edge_ids: Tensor of edge identifiers
            timestamps: Tensor of timestamps
        """
        if not self.training_phase or self.collection_disabled:
            return
        
        # Check if we've exceeded total timestamp limit
        if self.total_timestamps_collected >= self.max_total_timestamps:
            if not self.collection_disabled:
                logger.warning(f"Disabling timestamp collection after {self.total_timestamps_collected} timestamps to prevent memory issues")
                self.collection_disabled = True
            return
        
        # Limit memory usage by capping the number of timestamps per edge
        max_timestamps_per_edge = 100  # Further reduced to prevent segfault
        
        try:
            edge_ids_np = edge_ids.cpu().numpy()
            timestamps_np = timestamps.cpu().numpy()
            
            for edge_id, timestamp in zip(edge_ids_np, timestamps_np):
                edge_key = int(edge_id)
                
                # Limit timestamps per edge to prevent memory issues
                if len(self.edge_timestamps[edge_key]) < max_timestamps_per_edge:
                    self.edge_timestamps[edge_key].append(float(timestamp))
                    self.edge_counts[edge_key] += 1
                    self.total_timestamps_collected += 1
                else:
                    # Replace oldest timestamp with newest (sliding window)
                    self.edge_timestamps[edge_key].pop(0)
                    self.edge_timestamps[edge_key].append(float(timestamp))
                    # Don't increment total count for replacements
            
            # Periodic garbage collection to prevent memory buildup
            self._gc_counter += 1
            if self._gc_counter % 100 == 0:
                import gc
                gc.collect()
                    
        except Exception as e:
            logger.warning(f"Error collecting timestamps: {e}")
            # Continue training even if timestamp collection fails
    
    def compute_kde_function(self, timestamps: np.ndarray) -> gaussian_kde:
        """
        Compute KDE function from absolute timestamps.
        
        Args:
            timestamps: Array of absolute timestamps
            
        Returns:
            KDE function
        """
        if len(timestamps) < 2:
            # Not enough data for KDE, use single point with small variance
            timestamps = np.array([timestamps[0], timestamps[0] + 1e-6])
            
        kde = gaussian_kde(timestamps, bw_method=self.bandwidth)
        return kde
    
    def kde_to_rkhs_vector(self, kde: gaussian_kde) -> torch.Tensor:
        """
        Convert KDE function to RKHS vector representation using Gauss-Hermite quadrature.
        GPU-optimized version using PyTorch operations.
        
        Args:
            kde: KDE function
            
        Returns:
            RKHS vector representation (on same device as model)
        """
        device = next(self.parameters()).device
        
        # Get data statistics for scaling
        data_mean = float(np.mean(kde.dataset))
        data_std = float(np.std(kde.dataset)) if np.std(kde.dataset) > 0 else 1.0
        
        # Transform quadrature points to data scale (move to GPU)
        quad_points_gpu = self.quad_points.to(device)
        quad_weights_gpu = self.quad_weights.to(device)
        
        scaled_points = data_mean + np.sqrt(2) * data_std * quad_points_gpu
        
        # Evaluate KDE at quadrature points (still need CPU for scipy KDE)
        scaled_points_cpu = scaled_points.cpu().numpy()
        kde_values = kde(scaled_points_cpu)
        kde_values_gpu = torch.tensor(kde_values, dtype=torch.float32, device=device)
        
        # Create RKHS features using GPU operations
        rkhs_features = []
        
        # Feature 1: Weighted KDE values at quadrature points
        weighted_values = kde_values_gpu * quad_weights_gpu * np.sqrt(2 * data_std)
        rkhs_features.append(weighted_values[:self.rkhs_dim // 4])
        
        # Feature 2: Moments (mean, variance, skewness, kurtosis approximations)
        moments = []
        for p in range(1, 5):
            moment = torch.sum(weighted_values * (scaled_points ** p))
            moments.append(moment)
        moments_tensor = torch.stack(moments)
        
        # Feature 3: Fourier features
        n_fourier = self.rkhs_dim // 4
        frequencies = torch.linspace(0.1, 10, n_fourier, device=device)
        fourier_features = []
        for freq in frequencies:
            cos_feat = torch.sum(weighted_values * torch.cos(freq * scaled_points))
            sin_feat = torch.sum(weighted_values * torch.sin(freq * scaled_points))
            fourier_features.extend([cos_feat, sin_feat])
        fourier_features = torch.stack(fourier_features[:n_fourier])
        
        # Feature 4: Quantile features (computed on CPU, moved to GPU)
        quantiles_cpu = np.percentile(kde.dataset.flatten(), np.linspace(0, 100, self.rkhs_dim // 4))
        quantiles = torch.tensor(quantiles_cpu, dtype=torch.float32, device=device)
        
        # Combine all features
        moments_repeated = moments_tensor.repeat(self.rkhs_dim // 16)[:self.rkhs_dim // 4]
        
        all_features = torch.cat([
            weighted_values[:self.rkhs_dim // 4],
            moments_repeated,
            fourier_features,
            quantiles
        ])
        
        # Ensure correct dimension
        if all_features.shape[0] < self.rkhs_dim:
            padding = torch.zeros(self.rkhs_dim - all_features.shape[0], device=device)
            all_features = torch.cat([all_features, padding])
        else:
            all_features = all_features[:self.rkhs_dim]
            
        return all_features
    
    def build_kde_vectors(self):
        """
        Build KDE vectors for all frequent edges after training.
        """
        logger.info(f"Building KDE vectors for edges with >= {self.min_occurrences} occurrences")
        
        frequent_edges = 0
        rare_edges = 0
        failed_edges = 0
        
        device = next(self.parameters()).device
        
        # Process edges in batches to avoid memory issues
        edge_items = list(self.edge_timestamps.items())
        batch_size = 100  # Process 100 edges at a time
        
        for i in range(0, len(edge_items), batch_size):
            batch_items = edge_items[i:i + batch_size]
            
            for edge_key, timestamps in batch_items:
                try:
                    if len(timestamps) >= self.min_occurrences:
                        # Compute KDE for frequent edges using absolute timestamps
                        timestamps_array = np.array(timestamps)
                        kde = self.compute_kde_function(timestamps_array)
                        kde_vector = self.kde_to_rkhs_vector(kde)
                        # Ensure vector is on correct device
                        self.edge_kde_vectors[edge_key] = kde_vector.to(device)
                        frequent_edges += 1
                    else:
                        rare_edges += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to build KDE for edge {edge_key}: {e}")
                    failed_edges += 1
                    # Continue with other edges
            
            # Periodic cleanup to prevent memory buildup
            if i % (batch_size * 10) == 0:
                import gc
                gc.collect()
                
        logger.info(f"Built KDE vectors for {frequent_edges} frequent edges, "
                   f"{rare_edges} rare edges will use fallback encoder, "
                   f"{failed_edges} edges failed")
        
        # Clear timestamp storage to free memory
        self.edge_timestamps.clear()
        logger.info("Cleared timestamp storage to free memory")
        
    def forward(self, edge_ids: torch.Tensor, timestamps: torch.Tensor, time_diffs: torch.Tensor) -> torch.Tensor:
        """
        Encode time using hybrid KDE/fallback approach.
        
        Args:
            edge_ids: Tensor of edge identifiers [batch_size]
            timestamps: Tensor of absolute timestamps [batch_size] (used for KDE lookup context)
            time_diffs: Tensor of time differences [batch_size] (used for fallback encoder)
            
        Returns:
            Encoded time features [batch_size, out_channels]
        """
        try:
            batch_size = edge_ids.shape[0]
            device = edge_ids.device
            
            # Ensure fallback encoder exists and is on correct device
            if self.fallback_encoder is None:
                logger.warning("Fallback encoder is None, creating new one")
                self.fallback_encoder = FallbackTimeEncoder(self.out_channels).to(device)
            
            # Move fallback encoder to correct device if needed
            if next(self.fallback_encoder.parameters()).device != device:
                self.fallback_encoder = self.fallback_encoder.to(device)
            
            # Use fallback encoder only when KDE vectors are not available
            if len(self.edge_kde_vectors) == 0:
                # No KDE vectors built yet, use fallback encoder
                return self.fallback_encoder(time_diffs)
            
            # During inference, use KDE vectors where available
            output = torch.zeros(batch_size, self.out_channels, device=device, dtype=torch.float32)
            
            for i in range(batch_size):
                try:
                    edge_key = int(edge_ids[i].item())
                    
                    if edge_key in self.edge_kde_vectors:
                        # Use precomputed KDE vector for frequent edges
                        kde_vector = self.edge_kde_vectors[edge_key]
                        if kde_vector.device != device:
                            kde_vector = kde_vector.to(device)
                        output[i] = self.rkhs_projection(kde_vector)
                    else:
                        # Use fallback encoder for rare/new edges
                        time_tensor = time_diffs[i:i+1]  # Keep batch dimension
                        fallback_output = self.fallback_encoder(time_tensor)
                        output[i] = fallback_output.squeeze(0)
                        
                except Exception as e:
                    logger.warning(f"Error processing edge {i}: {e}, using fallback")
                    # Use fallback for this edge
                    time_tensor = time_diffs[i:i+1]
                    fallback_output = self.fallback_encoder(time_tensor)
                    output[i] = fallback_output.squeeze(0)
                    
            return output
            
        except Exception as e:
            logger.error(f"Critical error in KDE forward pass: {e}")
            # Emergency fallback: use fallback encoder for entire batch
            if self.fallback_encoder is not None:
                return self.fallback_encoder(time_diffs)
            else:
                # Last resort: return zeros
                return torch.zeros(time_diffs.shape[0], self.out_channels, 
                                 device=time_diffs.device, dtype=torch.float32)
    
    def set_training_phase(self, training: bool):
        """
        Set whether in training phase (collecting timestamps) or inference phase.
        """
        self.training_phase = training
        if not training:
            # Build KDE vectors when switching to inference
            self.build_kde_vectors()
            
    def get_statistics(self) -> Dict:
        """
        Get statistics about edge occurrences and KDE usage.
        """
        total_edges = len(self.edge_timestamps)
        frequent_edges = sum(1 for count in self.edge_counts.values() 
                           if count >= self.min_occurrences)
        rare_edges = total_edges - frequent_edges
        
        return {
            'total_edges': total_edges,
            'frequent_edges': frequent_edges,
            'rare_edges': rare_edges,
            'min_occurrences_threshold': self.min_occurrences,
            'kde_coverage': frequent_edges / total_edges if total_edges > 0 else 0
        }


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
        return self.lin(t.view(-1, 1)).cos()
