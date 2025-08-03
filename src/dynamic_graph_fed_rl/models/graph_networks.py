"""Graph neural network architectures for temporal processing."""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .base import BaseGraphModel, GraphAttentionLayer


class GraphAttentionNetwork(BaseGraphModel):
    """Multi-layer graph attention network."""
    
    num_layers: int = 3
    num_heads: int = 8
    
    def setup(self):
        self.layers = [
            GraphAttentionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]
        
        self.input_projection = nn.Dense(self.hidden_dim)
        self.output_projection = nn.Dense(self.hidden_dim)
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through graph attention network."""
        # Project input features
        x = self.input_projection(node_features)
        x = nn.activation.relu(x)
        
        # Apply attention layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_features, training)
        
        # Final projection
        x = self.output_projection(x)
        return x


class TemporalGraphConv(nn.Module):
    """Temporal graph convolution with memory."""
    
    hidden_dim: int = 128
    memory_dim: int = 64
    time_window: int = 10
    
    def setup(self):
        self.spatial_conv = nn.Dense(self.hidden_dim)
        self.temporal_conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.time_window,),
            padding='SAME',
        )
        self.memory_update = nn.GRUCell(features=self.memory_dim)
        self.output_projection = nn.Dense(self.hidden_dim)
    
    def __call__(
        self,
        node_features: jnp.ndarray,  # [time_steps, num_nodes, node_dim]
        edge_index: jnp.ndarray,
        memory_state: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply temporal graph convolution.
        
        Args:
            node_features: Temporal node features
            edge_index: Edge connectivity (assumed static)
            memory_state: Previous memory state
            training: Training mode flag
            
        Returns:
            Updated node embeddings and memory state
        """
        time_steps, num_nodes, node_dim = node_features.shape
        
        # Initialize memory if not provided
        if memory_state is None:
            memory_state = jnp.zeros((num_nodes, self.memory_dim))
        
        # Process each time step
        outputs = []
        current_memory = memory_state
        
        for t in range(time_steps):
            # Spatial convolution
            spatial_features = self.spatial_conv(node_features[t])
            spatial_features = nn.activation.relu(spatial_features)
            
            # Message passing (simplified)
            messages = self._message_passing(spatial_features, edge_index)
            
            # Update memory
            memory_input = jnp.concatenate([spatial_features, messages], axis=-1)
            memory_projection = nn.Dense(self.memory_dim)(memory_input)
            
            _, current_memory = self.memory_update(
                current_memory, memory_projection
            )
            
            # Combine spatial and temporal information
            combined = jnp.concatenate([spatial_features, current_memory], axis=-1)
            output = self.output_projection(combined)
            outputs.append(output)
        
        # Stack temporal outputs
        temporal_outputs = jnp.stack(outputs, axis=0)
        
        # Apply temporal convolution
        # Reshape for convolution: [num_nodes, time_steps, features]
        temporal_outputs = jnp.transpose(temporal_outputs, (1, 0, 2))
        temporal_outputs = self.temporal_conv(temporal_outputs[..., None])[..., 0]
        
        # Return final time step and memory
        final_output = temporal_outputs[:, -1, :]
        return final_output, current_memory
    
    def _message_passing(
        self, node_features: jnp.ndarray, edge_index: jnp.ndarray
    ) -> jnp.ndarray:
        """Simple message passing aggregation."""
        num_nodes = node_features.shape[0]
        
        # Aggregate messages from neighbors
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Use segment_sum for aggregation
        messages = jnp.zeros_like(node_features)
        messages = messages.at[target_nodes].add(node_features[source_nodes])
        
        return messages


class MultiScaleTemporalGNN(BaseGraphModel):
    """Multi-scale temporal graph neural network."""
    
    time_scales: List[int] = (1, 5, 20, 100)
    num_heads: int = 8
    
    def setup(self):
        # Temporal encoders for different scales
        self.temporal_encoders = [
            nn.GRU(features=self.hidden_dim)
            for _ in self.time_scales
        ]
        
        # Graph encoders for each scale
        self.graph_encoders = [
            GraphAttentionNetwork(
                hidden_dim=self.hidden_dim,
                num_layers=2,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
            )
            for _ in self.time_scales
        ]
        
        # Cross-scale attention
        self.scale_attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim * len(self.time_scales),
        )
        
        self.output_projection = nn.Dense(self.hidden_dim)
    
    def __call__(
        self,
        graph_sequence: List[Tuple[jnp.ndarray, jnp.ndarray]],
        training: bool = True,
    ) -> jnp.ndarray:
        """Process multi-scale temporal graph sequence.
        
        Args:
            graph_sequence: List of (node_features, edge_index) tuples
            training: Training mode flag
            
        Returns:
            Multi-scale graph representation
        """
        scale_representations = []
        
        for scale_idx, (temporal_enc, graph_enc) in enumerate(
            zip(self.temporal_encoders, self.graph_encoders)
        ):
            scale = self.time_scales[scale_idx]
            
            # Sample sequence at current scale
            sampled_sequence = graph_sequence[::scale]
            if len(sampled_sequence) == 0:
                sampled_sequence = [graph_sequence[-1]]
            
            # Encode graph features at each time step
            graph_features = []
            for node_features, edge_index in sampled_sequence:
                # Get node embeddings
                node_embeddings = graph_enc(
                    node_features, edge_index, None, training
                )
                # Global pooling
                graph_repr = jnp.mean(node_embeddings, axis=0)
                graph_features.append(graph_repr)
            
            # Stack for temporal encoding
            if len(graph_features) > 1:
                temporal_sequence = jnp.stack(graph_features)
                # Apply GRU
                carry = temporal_enc.initialize_carry(
                    jax.random.PRNGKey(0), (self.hidden_dim,)
                )
                carry, outputs = nn.scan(
                    temporal_enc,
                    variable_broadcast='params',
                    split_rngs={'params': False},
                )(carry, temporal_sequence)
                scale_repr = outputs[-1]  # Final output
            else:
                scale_repr = graph_features[0]
            
            scale_representations.append(scale_repr)
        
        # Fuse multi-scale representations
        fused = jnp.concatenate(scale_representations, axis=-1)
        
        # Apply cross-scale attention
        fused_expanded = fused[None, :]  # Add batch dimension
        attended, _ = self.scale_attention(
            inputs_q=fused_expanded,
            inputs_kv=fused_expanded,
            deterministic=not training,
        )
        attended = attended[0]  # Remove batch dimension
        
        # Final projection
        output = self.output_projection(attended)
        return output


class DynamicGraphEncoder(nn.Module):
    """Encoder for dynamic graphs with topology changes."""
    
    hidden_dim: int = 128
    max_nodes: int = 1000
    topology_embedding_dim: int = 64
    
    def setup(self):
        self.node_encoder = nn.Dense(self.hidden_dim)
        self.edge_encoder = nn.Dense(self.hidden_dim)
        
        # Topology change detector
        self.topology_detector = nn.Sequential([
            nn.Dense(self.topology_embedding_dim),
            nn.activation.tanh,
            nn.Dense(1),
            nn.activation.sigmoid,
        ])
        
        # Graph neural network
        self.gnn = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=3,
        )
        
        # Temporal integration
        self.temporal_encoder = nn.GRU(features=self.hidden_dim)
    
    def __call__(
        self,
        current_graph: Tuple[jnp.ndarray, jnp.ndarray],
        previous_embedding: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Encode dynamic graph with change detection.
        
        Args:
            current_graph: (node_features, edge_index)
            previous_embedding: Previous graph embedding
            training: Training mode flag
            
        Returns:
            graph_embedding: Current graph embedding
            updated_memory: Updated temporal memory
            change_probability: Topology change probability
        """
        node_features, edge_index = current_graph
        
        # Encode current graph
        current_embedding = self.gnn(
            node_features, edge_index, None, training
        )
        current_graph_embedding = jnp.mean(current_embedding, axis=0)
        
        # Detect topology changes
        if previous_embedding is not None:
            embedding_diff = current_graph_embedding - previous_embedding
            change_prob = self.topology_detector(embedding_diff)
            change_prob = float(change_prob)
        else:
            change_prob = 0.0
        
        # Update temporal memory
        if previous_embedding is not None:
            memory_input = jnp.stack([previous_embedding, current_graph_embedding])
            carry = self.temporal_encoder.initialize_carry(
                jax.random.PRNGKey(0), (self.hidden_dim,)
            )
            carry, outputs = nn.scan(
                self.temporal_encoder,
                variable_broadcast='params',
                split_rngs={'params': False},
            )(carry, memory_input)
            updated_memory = outputs[-1]
        else:
            updated_memory = current_graph_embedding
        
        return current_graph_embedding, updated_memory, change_prob


def create_temporal_graph_batch(
    graph_sequences: List[List[Tuple[jnp.ndarray, jnp.ndarray]]],
    max_time_steps: int,
    max_nodes: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create batched temporal graph data.
    
    Args:
        graph_sequences: List of graph sequences
        max_time_steps: Maximum sequence length
        max_nodes: Maximum number of nodes
        
    Returns:
        Batched node features, edge indices, and sequence masks
    """
    batch_size = len(graph_sequences)
    
    # Initialize tensors
    batched_nodes = jnp.zeros((batch_size, max_time_steps, max_nodes, 64))
    batched_edges = jnp.zeros((batch_size, max_time_steps, 2, max_nodes * 2))
    sequence_masks = jnp.zeros((batch_size, max_time_steps), dtype=bool)
    
    for batch_idx, sequence in enumerate(graph_sequences):
        seq_len = min(len(sequence), max_time_steps)
        sequence_masks = sequence_masks.at[batch_idx, :seq_len].set(True)
        
        for time_idx in range(seq_len):
            node_features, edge_index = sequence[time_idx]
            num_nodes = min(node_features.shape[0], max_nodes)
            num_edges = min(edge_index.shape[1], max_nodes * 2)
            
            # Pad node features
            batched_nodes = batched_nodes.at[
                batch_idx, time_idx, :num_nodes, :node_features.shape[1]
            ].set(node_features[:num_nodes])
            
            # Pad edge index
            batched_edges = batched_edges.at[
                batch_idx, time_idx, :, :num_edges
            ].set(edge_index[:, :num_edges])
    
    return batched_nodes, batched_edges, sequence_masks