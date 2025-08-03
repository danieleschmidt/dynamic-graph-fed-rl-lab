"""Base classes for graph neural network models."""

import abc
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


class BaseGraphModel(nn.Module, abc.ABC):
    """Base class for all graph neural network models."""
    
    hidden_dim: int = 128
    dropout_rate: float = 0.1
    
    @abc.abstractmethod
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through the graph neural network.
        
        Args:
            node_features: Node feature matrix [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge feature matrix [num_edges, edge_dim]
            training: Whether in training mode
            
        Returns:
            Updated node embeddings [num_nodes, hidden_dim]
        """
        pass
    
    def encode_graph(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Encode entire graph into a single representation.
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge connectivity
            edge_features: Edge feature matrix
            training: Whether in training mode
            
        Returns:
            Graph-level embedding [hidden_dim]
        """
        node_embeddings = self(
            node_features, edge_index, edge_features, training
        )
        # Global mean pooling
        return jnp.mean(node_embeddings, axis=0)
    
    def compute_attention_weights(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute attention weights for interpretability.
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge connectivity
            
        Returns:
            Attention weights [num_edges]
        """
        # Default implementation returns uniform weights
        num_edges = edge_index.shape[1]
        return jnp.ones(num_edges) / num_edges


class GraphStateEncoder(nn.Module):
    """Encoder for graph-based environment states."""
    
    node_dim: int = 64
    edge_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 3
    
    def setup(self):
        self.node_encoder = nn.Dense(self.hidden_dim)
        self.edge_encoder = nn.Dense(self.hidden_dim) if self.edge_dim > 0 else None
        
        self.graph_layers = [
            GraphAttentionLayer(self.hidden_dim)
            for _ in range(self.num_layers)
        ]
        
        self.layer_norm = nn.LayerNorm()
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode graph state.
        
        Returns:
            node_embeddings: Updated node embeddings
            graph_embedding: Global graph representation
        """
        # Encode node features
        x = self.node_encoder(node_features)
        x = nn.activation.relu(x)
        
        # Encode edge features if provided
        edge_attr = None
        if edge_features is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_features)
            edge_attr = nn.activation.relu(edge_attr)
        
        # Apply graph layers
        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_attr, training)
            x = self.layer_norm(x)
        
        # Global pooling for graph-level representation
        graph_embedding = jnp.mean(x, axis=0)
        
        return x, graph_embedding


class GraphAttentionLayer(nn.Module):
    """Single graph attention layer."""
    
    hidden_dim: int
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    def setup(self):
        self.multihead_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        )
        self.feed_forward = nn.Sequential([
            nn.Dense(self.hidden_dim * 4),
            nn.activation.relu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(self.hidden_dim),
        ])
        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Apply graph attention layer."""
        # Create attention mask from edge connectivity
        num_nodes = node_features.shape[0]
        attention_mask = self._create_attention_mask(edge_index, num_nodes)
        
        # Self-attention
        residual = node_features
        x = self.layer_norm1(node_features)
        x = self.multihead_attn(
            inputs_q=x,
            inputs_kv=x,
            mask=attention_mask,
            deterministic=not training,
        )
        x = x + residual
        
        # Feed forward
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x + residual
        
        return x
    
    def _create_attention_mask(
        self, edge_index: jnp.ndarray, num_nodes: int
    ) -> jnp.ndarray:
        """Create attention mask from edge connectivity."""
        mask = jnp.zeros((num_nodes, num_nodes), dtype=bool)
        mask = mask.at[edge_index[0], edge_index[1]].set(True)
        mask = mask.at[jnp.arange(num_nodes), jnp.arange(num_nodes)].set(True)
        return mask


def create_graph_from_adjacency(
    adjacency_matrix: jnp.ndarray,
) -> Tuple[jnp.ndarray, int]:
    """Convert adjacency matrix to edge index format.
    
    Args:
        adjacency_matrix: Square adjacency matrix [num_nodes, num_nodes]
        
    Returns:
        edge_index: Edge connectivity [2, num_edges]
        num_edges: Number of edges
    """
    sources, targets = jnp.nonzero(adjacency_matrix, size=None)
    edge_index = jnp.stack([sources, targets], axis=0)
    num_edges = edge_index.shape[1]
    return edge_index, num_edges


def normalize_graph_features(
    node_features: jnp.ndarray,
    edge_features: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Normalize graph features for stable training.
    
    Args:
        node_features: Node feature matrix
        edge_features: Edge feature matrix
        
    Returns:
        Normalized node and edge features
    """
    # Normalize node features
    node_mean = jnp.mean(node_features, axis=0, keepdims=True)
    node_std = jnp.std(node_features, axis=0, keepdims=True) + 1e-8
    normalized_nodes = (node_features - node_mean) / node_std
    
    # Normalize edge features if provided
    normalized_edges = None
    if edge_features is not None:
        edge_mean = jnp.mean(edge_features, axis=0, keepdims=True)
        edge_std = jnp.std(edge_features, axis=0, keepdims=True) + 1e-8
        normalized_edges = (edge_features - edge_mean) / edge_std
    
    return normalized_nodes, normalized_edges