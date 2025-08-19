"""Advanced Adaptive Temporal Graph Reinforcement Learning Algorithm.

This implements breakthrough temporal graph processing with adaptive learning rates,
hierarchical attention mechanisms, and self-organizing graph embeddings.
"""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
import numpy as np

from .base import BaseGraphAlgorithm, GraphMetricsTracker, AdaptiveNoise
from ..environments.base import GraphState, GraphTransition


class HierarchicalTemporalAttention(nn.Module):
    """Multi-scale temporal attention for dynamic graphs."""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        time_scales: List[int] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.time_scales = time_scales or [1, 4, 16, 64]
        self.dropout_rate = dropout_rate
    
    def setup(self):
        # Multi-scale temporal encoders
        self.temporal_encoders = [
            nn.GRU(features=self.hidden_dim, name=f"temporal_enc_{scale}")
            for scale in self.time_scales
        ]
        
        # Cross-scale attention
        self.cross_scale_attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            name="cross_scale_attn"
        )
        
        # Final projection
        self.output_projection = nn.Dense(
            features=self.hidden_dim,
            name="output_proj"
        )
    
    def __call__(
        self,
        temporal_sequence: jnp.ndarray,  # [seq_len, batch_size, node_features]
        training: bool = True,
    ) -> jnp.ndarray:
        """Process temporal sequence with hierarchical attention."""
        seq_len, batch_size, feature_dim = temporal_sequence.shape
        
        # Multi-scale temporal encoding
        scale_representations = []
        
        for scale_idx, (scale, encoder) in enumerate(zip(self.time_scales, self.temporal_encoders)):
            # Sample sequence at different temporal scales
            if scale == 1:
                sampled_seq = temporal_sequence
            else:
                # Downsample with averaging
                padding = (scale - seq_len % scale) % scale
                if padding > 0:
                    pad_seq = jnp.zeros((padding, batch_size, feature_dim))
                    padded_seq = jnp.concatenate([temporal_sequence, pad_seq], axis=0)
                else:
                    padded_seq = temporal_sequence
                
                # Reshape and average
                reshaped = padded_seq.reshape(-1, scale, batch_size, feature_dim)
                sampled_seq = jnp.mean(reshaped, axis=1)
            
            # Encode temporal features
            carry = encoder.initialize_carry(batch_size, self.hidden_dim)
            outputs, _ = encoder(carry, sampled_seq)
            
            # Take final output
            scale_repr = outputs[-1] if len(outputs.shape) > 2 else outputs
            scale_representations.append(scale_repr)
        
        # Stack representations [num_scales, batch_size, hidden_dim]
        multi_scale = jnp.stack(scale_representations, axis=0)
        
        # Cross-scale attention
        attended_features = self.cross_scale_attention(
            inputs_q=multi_scale,
            inputs_kv=multi_scale,
            deterministic=not training,
        )
        
        # Global pooling across scales
        pooled_features = jnp.mean(attended_features, axis=0)
        
        # Final projection
        output = self.output_projection(pooled_features)
        
        return output


class SelfOrganizingGraphEmbedding(nn.Module):
    """Self-organizing graph embedding with adaptive topology learning."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        adaptation_rate: float = 0.01,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.adaptation_rate = adaptation_rate
    
    def setup(self):
        # Node embedding layers
        self.node_embedding = nn.Dense(
            features=self.hidden_dim,
            name="node_embed"
        )
        
        # Edge embedding
        self.edge_embedding = nn.Dense(
            features=self.hidden_dim,
            name="edge_embed"
        )
        
        # Graph convolution layers with residual connections
        self.gnn_layers = [
            nn.Dense(features=self.hidden_dim, name=f"gnn_layer_{i}")
            for i in range(self.num_layers)
        ]
        
        # Adaptive topology learner
        self.topology_predictor = nn.Dense(
            features=1,
            name="topology_pred"
        )
        
        # Layer normalization
        self.layer_norms = [
            nn.LayerNorm(name=f"layer_norm_{i}")
            for i in range(self.num_layers)
        ]
    
    def __call__(
        self,
        node_features: jnp.ndarray,  # [num_nodes, node_dim]
        edge_indices: jnp.ndarray,   # [2, num_edges]
        edge_features: jnp.ndarray,  # [num_edges, edge_dim]
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Process graph with self-organizing embeddings."""
        num_nodes = node_features.shape[0]
        
        # Initial embeddings
        node_h = self.node_embedding(node_features)
        edge_h = self.edge_embedding(edge_features)
        
        # Graph message passing with adaptive topology
        for layer_idx in range(self.num_layers):
            # Prepare messages
            src_nodes = edge_indices[0]
            dst_nodes = edge_indices[1]
            
            # Source node features for each edge
            src_features = node_h[src_nodes]
            dst_features = node_h[dst_nodes]
            
            # Compute edge importance (adaptive topology)
            edge_context = jnp.concatenate([src_features, dst_features, edge_h], axis=-1)
            edge_weights = nn.sigmoid(self.topology_predictor(edge_context))
            
            # Weighted messages
            messages = src_features * edge_weights + edge_h
            
            # Aggregate messages at destination nodes
            aggregated = jnp.zeros_like(node_h)
            aggregated = aggregated.at[dst_nodes].add(messages)
            
            # Update node representations
            layer = self.gnn_layers[layer_idx]
            layer_norm = self.layer_norms[layer_idx]
            
            # Residual connection + layer norm
            updated_h = layer(jnp.concatenate([node_h, aggregated], axis=-1))
            node_h = layer_norm(node_h + updated_h)
            node_h = nn.relu(node_h)
        
        # Adaptive edge weights for topology learning
        final_edge_weights = jnp.squeeze(edge_weights, axis=-1)
        
        return node_h, final_edge_weights


class AdaptiveTemporalGraphActor(nn.Module):
    """Actor network with adaptive temporal graph processing."""
    
    def __init__(
        self,
        action_dim: int,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        time_window: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.time_window = time_window
    
    def setup(self):
        # Temporal graph processor
        self.temporal_attention = HierarchicalTemporalAttention(
            hidden_dim=self.hidden_dim,
            name="temporal_attn"
        )
        
        # Graph embedding
        self.graph_embedding = SelfOrganizingGraphEmbedding(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            name="graph_embed"
        )
        
        # Policy head
        self.policy_layers = [
            nn.Dense(features=self.hidden_dim, name="policy_1"),
            nn.Dense(features=self.hidden_dim, name="policy_2"),
            nn.Dense(features=self.action_dim, name="policy_out"),
        ]
    
    def __call__(
        self,
        graph_sequence: List[GraphState],  # Temporal sequence of graphs
        training: bool = True,
    ) -> jnp.ndarray:
        """Generate action from temporal graph sequence."""
        # Process each graph in sequence
        graph_embeddings = []
        
        for graph_state in graph_sequence:
            # Get graph embedding
            node_emb, edge_weights = self.graph_embedding(
                node_features=graph_state.node_features,
                edge_indices=graph_state.edge_indices,
                edge_features=graph_state.edge_features,
                training=training,
            )
            
            # Global graph representation (mean pooling)
            graph_repr = jnp.mean(node_emb, axis=0)
            graph_embeddings.append(graph_repr)
        
        # Stack temporal sequence [seq_len, batch_size=1, features]
        temporal_sequence = jnp.stack(graph_embeddings, axis=0)
        temporal_sequence = jnp.expand_dims(temporal_sequence, axis=1)
        
        # Temporal attention processing
        temporal_features = self.temporal_attention(
            temporal_sequence, training=training
        )
        
        # Remove batch dimension
        temporal_features = jnp.squeeze(temporal_features, axis=0)
        
        # Policy network
        policy_input = temporal_features
        for layer in self.policy_layers[:-1]:
            policy_input = nn.relu(layer(policy_input))
        
        # Action output (tanh activation)
        actions = nn.tanh(self.policy_layers[-1](policy_input))
        
        return actions


class AdaptiveTemporalGraphCritic(nn.Module):
    """Critic network with adaptive temporal graph processing."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        time_window: int = 10,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_window = time_window
    
    def setup(self):
        # Temporal graph processor (shared with actor)
        self.temporal_attention = HierarchicalTemporalAttention(
            hidden_dim=self.hidden_dim,
            name="temporal_attn"
        )
        
        # Graph embedding
        self.graph_embedding = SelfOrganizingGraphEmbedding(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            name="graph_embed"
        )
        
        # Action embedding
        self.action_embedding = nn.Dense(
            features=self.hidden_dim,
            name="action_embed"
        )
        
        # Value head
        self.value_layers = [
            nn.Dense(features=self.hidden_dim, name="value_1"),
            nn.Dense(features=self.hidden_dim, name="value_2"),
            nn.Dense(features=1, name="value_out"),
        ]
    
    def __call__(
        self,
        graph_sequence: List[GraphState],
        actions: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """Evaluate state-action value."""
        # Process temporal graph sequence (same as actor)
        graph_embeddings = []
        
        for graph_state in graph_sequence:
            node_emb, _ = self.graph_embedding(
                node_features=graph_state.node_features,
                edge_indices=graph_state.edge_indices,
                edge_features=graph_state.edge_features,
                training=training,
            )
            
            graph_repr = jnp.mean(node_emb, axis=0)
            graph_embeddings.append(graph_repr)
        
        temporal_sequence = jnp.stack(graph_embeddings, axis=0)
        temporal_sequence = jnp.expand_dims(temporal_sequence, axis=1)
        
        temporal_features = self.temporal_attention(
            temporal_sequence, training=training
        )
        temporal_features = jnp.squeeze(temporal_features, axis=0)
        
        # Action embedding
        action_emb = self.action_embedding(actions)
        
        # Combine state and action features
        state_action = jnp.concatenate([temporal_features, action_emb], axis=-1)
        
        # Value network
        value_input = state_action
        for layer in self.value_layers[:-1]:
            value_input = nn.relu(layer(value_input))
        
        value = self.value_layers[-1](value_input)
        
        return jnp.squeeze(value, axis=-1)


class AdaptiveTemporalGraphRL(BaseGraphAlgorithm):
    """Adaptive Temporal Graph Reinforcement Learning Algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        edge_dim: int = 32,
        hidden_dim: int = 128,
        time_window: int = 10,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        policy_delay: int = 2,
        noise_clip: float = 0.5,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=seed,
        )
        
        self.time_window = time_window
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip
        
        # Initialize networks
        self._init_networks()
        
        # Adaptive components
        self.adaptive_noise = AdaptiveNoise(
            initial_noise=exploration_noise,
            decay_rate=0.9999,
        )
        
        self.metrics_tracker = GraphMetricsTracker()
        
        # Temporal graph buffer
        self.graph_history = []
        
        # Training counters
        self.update_counter = 0
    
    def _init_networks(self):
        """Initialize actor and critic networks."""
        # Create dummy inputs for network initialization
        dummy_graph_sequence = self._create_dummy_graph_sequence()
        dummy_action = jnp.zeros(self.action_dim)
        
        # Actor network
        self.actor = AdaptiveTemporalGraphActor(
            action_dim=self.action_dim,
            node_dim=self.state_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            time_window=self.time_window,
        )
        
        # Critic networks (twin critics)
        self.critic1 = AdaptiveTemporalGraphCritic(
            node_dim=self.state_dim,
            edge_dim=self.edge_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            time_window=self.time_window,
        )
        
        self.critic2 = AdaptiveTemporalGraphCritic(
            node_dim=self.state_dim,
            edge_dim=self.edge_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            time_window=self.time_window,
        )
        
        # Initialize training states
        self.actor_state = self._create_train_state(
            network=self.actor,
            dummy_input=[dummy_graph_sequence],
            learning_rate=self.learning_rate,
        )
        
        self.critic1_state = self._create_train_state(
            network=self.critic1,
            dummy_input=[dummy_graph_sequence, dummy_action],
            learning_rate=self.learning_rate,
        )
        
        self.critic2_state = self._create_train_state(
            network=self.critic2,
            dummy_input=[dummy_graph_sequence, dummy_action],
            learning_rate=self.learning_rate,
        )
        
        # Target networks
        self.target_actor_state = self.actor_state
        self.target_critic1_state = self.critic1_state
        self.target_critic2_state = self.critic2_state
    
    def _create_dummy_graph_sequence(self) -> List[GraphState]:
        """Create dummy graph sequence for network initialization."""
        dummy_graphs = []
        
        for _ in range(self.time_window):
            # Create dummy graph
            num_nodes = 10  # Arbitrary for initialization
            num_edges = 20
            
            dummy_graph = GraphState(
                node_features=jnp.ones((num_nodes, self.state_dim)),
                edge_indices=jnp.ones((2, num_edges), dtype=jnp.int32),
                edge_features=jnp.ones((num_edges, self.edge_dim)),
                global_features=jnp.ones(self.hidden_dim),
                num_nodes=num_nodes,
                num_edges=num_edges,
            )
            dummy_graphs.append(dummy_graph)
        
        return dummy_graphs
    
    def select_action(
        self,
        state: GraphState,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Select action using actor network with temporal attention."""
        # Add current state to history
        self.graph_history.append(state)
        
        # Keep only recent history
        if len(self.graph_history) > self.time_window:
            self.graph_history = self.graph_history[-self.time_window:]
        
        # Pad history if needed
        graph_sequence = self.graph_history.copy()
        while len(graph_sequence) < self.time_window:
            # Repeat first state for padding
            graph_sequence.insert(0, graph_sequence[0] if graph_sequence else state)
        
        # Get action from actor
        action = self.actor_state.apply_fn(
            self.actor_state.params,
            graph_sequence,
            training=not deterministic,
        )
        
        # Add exploration noise
        if not deterministic and not self.is_evaluation_mode():
            self.rng_key, noise_key = jax.random.split(self.rng_key)
            noise = self.adaptive_noise.sample_noise(
                noise_key, action.shape
            )
            action = jnp.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Update networks using TD3-style updates."""
        self.update_counter += 1
        
        # Extract batch components
        # Note: In practice, batch would contain temporal sequences
        graph_sequences = batch["graph_sequences"]  # List of graph sequences
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_graph_sequences = batch["next_graph_sequences"]
        dones = batch["dones"]
        
        # Update critics
        critic_loss1, critic_loss2 = self._update_critics(
            graph_sequences, actions, rewards, next_graph_sequences, dones
        )
        
        # Update actor (delayed)
        actor_loss = 0.0
        if self.update_counter % self.policy_delay == 0:
            actor_loss = self._update_actor(graph_sequences)
            
            # Update target networks
            self.target_actor_state = self._soft_update(
                self.target_actor_state, self.actor_state, self.tau
            )
            self.target_critic1_state = self._soft_update(
                self.target_critic1_state, self.critic1_state, self.tau
            )
            self.target_critic2_state = self._soft_update(
                self.target_critic2_state, self.critic2_state, self.tau
            )
        
        # Update adaptive components
        self.adaptive_noise.update_noise()
        
        # Track metrics
        self.metrics_tracker.add_metric("critic1_loss", critic_loss1)
        self.metrics_tracker.add_metric("critic2_loss", critic_loss2)
        self.metrics_tracker.add_metric("actor_loss", actor_loss)
        self.metrics_tracker.add_metric("exploration_noise", self.adaptive_noise.current_noise)
        
        return {
            "critic1_loss": critic_loss1,
            "critic2_loss": critic_loss2,
            "actor_loss": actor_loss,
            "exploration_noise": self.adaptive_noise.current_noise,
        }
    
    def _update_critics(
        self,
        graph_sequences: List[List[GraphState]],
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_graph_sequences: List[List[GraphState]],
        dones: jnp.ndarray,
    ) -> Tuple[float, float]:
        """Update critic networks."""
        
        def critic_loss_fn(critic_params, critic_apply_fn):
            # Current Q-values
            q_values = jnp.array([
                critic_apply_fn(
                    critic_params, seq, action, training=True
                )
                for seq, action in zip(graph_sequences, actions)
            ])
            
            # Target actions with noise
            self.rng_key, noise_key = jax.random.split(self.rng_key)
            noise = jax.random.normal(noise_key, actions.shape) * self.exploration_noise
            noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
            
            target_actions = jnp.array([
                self.target_actor_state.apply_fn(
                    self.target_actor_state.params, seq, training=True
                )
                for seq in next_graph_sequences
            ])
            target_actions = jnp.clip(target_actions + noise, -1.0, 1.0)
            
            # Target Q-values (minimum of twin critics)
            target_q1 = jnp.array([
                self.target_critic1_state.apply_fn(
                    self.target_critic1_state.params, seq, action, training=True
                )
                for seq, action in zip(next_graph_sequences, target_actions)
            ])
            target_q2 = jnp.array([
                self.target_critic2_state.apply_fn(
                    self.target_critic2_state.params, seq, action, training=True
                )
                for seq, action in zip(next_graph_sequences, target_actions)
            ])
            
            target_q = jnp.minimum(target_q1, target_q2)
            target_values = rewards + self.gamma * (1 - dones) * target_q
            
            # Critic loss
            loss = jnp.mean((q_values - jax.lax.stop_gradient(target_values)) ** 2)
            return loss
        
        # Update critic 1
        critic1_grad_fn = jax.grad(critic_loss_fn)
        critic1_grads = critic1_grad_fn(
            self.critic1_state.params, self.critic1_state.apply_fn
        )
        self.critic1_state = self.critic1_state.apply_gradients(grads=critic1_grads)
        critic1_loss = critic_loss_fn(self.critic1_state.params, self.critic1_state.apply_fn)
        
        # Update critic 2
        critic2_grads = critic1_grad_fn(
            self.critic2_state.params, self.critic2_state.apply_fn
        )
        self.critic2_state = self.critic2_state.apply_gradients(grads=critic2_grads)
        critic2_loss = critic_loss_fn(self.critic2_state.params, self.critic2_state.apply_fn)
        
        return float(critic1_loss), float(critic2_loss)
    
    def _update_actor(self, graph_sequences: List[List[GraphState]]) -> float:
        """Update actor network."""
        
        def actor_loss_fn(actor_params):
            # Get actions from actor
            actions = jnp.array([
                self.actor.apply(actor_params, seq, training=True)
                for seq in graph_sequences
            ])
            
            # Get Q-values from critic 1
            q_values = jnp.array([
                self.critic1_state.apply_fn(
                    self.critic1_state.params, seq, action, training=True
                )
                for seq, action in zip(graph_sequences, actions)
            ])
            
            # Actor loss (negative mean Q-value)
            return -jnp.mean(q_values)
        
        # Update actor
        actor_grad_fn = jax.grad(actor_loss_fn)
        actor_grads = actor_grad_fn(self.actor_state.params)
        self.actor_state = self.actor_state.apply_gradients(grads=actor_grads)
        
        actor_loss = actor_loss_fn(self.actor_state.params)
        
        return float(actor_loss)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save algorithm checkpoint."""
        checkpoint_data = {
            "actor_params": self.actor_state.params,
            "critic1_params": self.critic1_state.params,
            "critic2_params": self.critic2_state.params,
            "target_actor_params": self.target_actor_state.params,
            "target_critic1_params": self.target_critic1_state.params,
            "target_critic2_params": self.target_critic2_state.params,
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "total_timesteps": self.total_timesteps,
            "adaptive_noise_level": self.adaptive_noise.current_noise,
            "update_counter": self.update_counter,
        }
        
        with open(filepath, 'wb') as f:
            jnp.save(f, checkpoint_data, allow_pickle=True)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load algorithm checkpoint."""
        with open(filepath, 'rb') as f:
            checkpoint_data = jnp.load(f, allow_pickle=True).item()
        
        # Restore parameters
        self.actor_state = self.actor_state.replace(
            params=checkpoint_data["actor_params"]
        )
        self.critic1_state = self.critic1_state.replace(
            params=checkpoint_data["critic1_params"]
        )
        self.critic2_state = self.critic2_state.replace(
            params=checkpoint_data["critic2_params"]
        )
        
        # Restore target networks
        self.target_actor_state = self.target_actor_state.replace(
            params=checkpoint_data["target_actor_params"]
        )
        self.target_critic1_state = self.target_critic1_state.replace(
            params=checkpoint_data["target_critic1_params"]
        )
        self.target_critic2_state = self.target_critic2_state.replace(
            params=checkpoint_data["target_critic2_params"]
        )
        
        # Restore training state
        self.training_step = checkpoint_data["training_step"]
        self.episode_count = checkpoint_data["episode_count"]
        self.total_timesteps = checkpoint_data["total_timesteps"]
        self.adaptive_noise.current_noise = checkpoint_data["adaptive_noise_level"]
        self.update_counter = checkpoint_data["update_counter"]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        base_stats = self.get_training_stats()
        metrics_summary = self.metrics_tracker.get_summary()
        
        return {
            **base_stats,
            "metrics": metrics_summary,
            "graph_history_length": len(self.graph_history),
            "adaptive_noise_level": self.adaptive_noise.current_noise,
            "update_counter": self.update_counter,
        }