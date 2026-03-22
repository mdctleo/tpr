"""
TGN Encoder with KDE-based time encoding for KAIROS-KDE
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from .tgn_encoder import TGNEncoder
from .kde_time_encoder import KDETimeEncoder

logger = logging.getLogger(__name__)


class TGNKDEEncoder(TGNEncoder):
    """
    TGN Encoder that uses KDE-based time encoding for frequent edges.
    Inherits from TGNEncoder and overrides time encoding functionality.
    """
    
    def __init__(
        self,
        node_features,
        edge_features,
        memory,
        gnn,
        min_dst_idx,
        neighbor_loader,
        time_encoder: Optional[KDETimeEncoder] = None,
        use_memory: bool = True,
        device=None,
        project_src_dst: bool = True,
        use_time_enc: bool = False,
        use_time_order_encoding: bool = False,
        gru=None
    ):
        # Initialize parent class
        super().__init__(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            gnn=gnn,
            min_dst_idx=min_dst_idx,
            neighbor_loader=neighbor_loader,
            time_encoder=None,  # We'll override this
            use_memory=use_memory,
            device=device,
            project_src_dst=project_src_dst,
            use_time_enc=use_time_enc,
            use_time_order_encoding=use_time_order_encoding,
            gru=gru
        )
        
        # Replace time encoder with KDE version
        if time_encoder is not None and isinstance(time_encoder, KDETimeEncoder):
            self.kde_time_encoder = time_encoder
            self.time_encoder = time_encoder  # For compatibility
        else:
            # Fallback to creating a new KDE encoder if not provided
            logger.warning("KDE time encoder not provided, creating default")
            self.kde_time_encoder = KDETimeEncoder(
                out_channels=100,  # Default time dimension
                rkhs_dim=100,
                min_occurrences=10
            )
            self.time_encoder = self.kde_time_encoder
            
        # Storage for edge tracking
        self.edge_id_map = {}  # Maps (src, dst) to unique edge_id
        self.next_edge_id = 0
        
    def get_edge_id(self, src: int, dst: int) -> int:
        """
        Get unique edge ID for (src, dst) pair.
        """
        edge_key = (int(src), int(dst))
        if edge_key not in self.edge_id_map:
            self.edge_id_map[edge_key] = self.next_edge_id
            self.next_edge_id += 1
        return self.edge_id_map[edge_key]
    
    def forward(self, batch):
        """
        Forward pass with KDE-based time encoding.
        """
        # Get node and edge indices
        src, dst = batch.src_tgn, batch.dst_tgn
        edge_index = torch.stack([src, dst], dim=0)
        n_id = torch.cat([src, dst]).unique()
        
        # Create edge IDs for KDE lookup
        batch_size = src.shape[0]
        edge_ids = torch.zeros(batch_size, dtype=torch.long, device=src.device)
        
        for i in range(batch_size):
            edge_ids[i] = self.get_edge_id(src[i].item(), dst[i].item())
        
        # Collect timestamps during training
        if self.kde_time_encoder.training_phase:
            self.kde_time_encoder.collect_timestamps(edge_ids, batch.t_tgn)
        
        # Get node embeddings from memory
        h = None
        if self.use_memory:
            h, last_update = self.memory(n_id)
        else:
            last_update = self.memory.get_last_update(n_id)
            h = None
            
        # Prepare edge features
        edge_feats = []
        if "edge_type" in self.edge_features:
            edge_feats.append(batch.edge_type_tgn)
        if "msg" in self.edge_features:
            edge_feats.append(batch.msg_tgn)
            
        # Apply KDE-based time encoding
        if "time_encoding" in self.edge_features:
            curr_t = batch.t_tgn  # Absolute timestamps
            rel_t = last_update[edge_index[0]] - curr_t  # Relative time differences
            
            # Use KDE encoder with edge IDs, absolute timestamps, and time differences
            rel_t_enc = self.kde_time_encoder(edge_ids, curr_t, rel_t)
            edge_feats.append(rel_t_enc)
            
        edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
        
        # Apply time order encoding if needed
        if self.use_time_order_encoding:
            if len(edge_feats) > 0:
                edge_feats = self.gru(edge_feats)
                self.gru.detach_state()
                
        # Rest of forward pass (GNN, etc.)
        node_type = batch.node_type_tgn
        edge_type = batch.edge_type_tgn
        
        x_dict, edge_index_dict = None, None
        node_type_argmax = None
        
        if edge_index.shape[1] > 0:  # Ensure there are edges
            x_dict, edge_index_dict = self.to_hetero_batch(
                h, n_id, edge_index, edge_type, node_type
            )
            
            # Apply GNN
            if self.neighbor_loader is not None:
                # Get neighbors
                neighbors, nodes, e_id = self.neighbor_loader(n_id)
                out = self.gnn(
                    (h, h[neighbors].view(-1, self.gnn.in_channels)),
                    edge_index,
                    edge_attr=edge_feats,
                )
            else:
                out = self.gnn(h, n_id, batch.t_tgn, batch.msg_tgn, edge_index)
        else:
            # No edges case
            out = h
            
        # Project source and destination embeddings
        if self.project_src_dst:
            if out is not None:
                src_emb = out[edge_index[0]]
                dst_emb = out[edge_index[1]]
            else:
                src_emb = h[edge_index[0]] if h is not None else None
                dst_emb = h[edge_index[1]] if h is not None else None
        else:
            src_emb = None
            dst_emb = None
            
        return {
            'x_dict': x_dict,
            'edge_index_dict': edge_index_dict,
            'node_type_argmax': node_type_argmax,
            'src_emb': src_emb,
            'dst_emb': dst_emb,
            'edge_attr': edge_feats
        }
    
    def finalize_training(self):
        """
        Call after training to build KDE vectors.
        """
        logger.info("Finalizing KDE time encoder training")
        self.kde_time_encoder.set_training_phase(False)
        stats = self.kde_time_encoder.get_statistics()
        logger.info(f"KDE Statistics: {stats}")
        
    def reset_kde_encoder(self):
        """
        Reset the KDE encoder for new training.
        """
        self.kde_time_encoder.edge_timestamps.clear()
        self.kde_time_encoder.edge_kde_vectors.clear()
        self.kde_time_encoder.edge_counts.clear()
        self.kde_time_encoder.set_training_phase(True)
        self.edge_id_map.clear()
        self.next_edge_id = 0
