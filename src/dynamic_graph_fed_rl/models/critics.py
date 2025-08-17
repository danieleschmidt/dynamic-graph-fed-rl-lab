"""Critic networks for graph-based value function approximation."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .base import BaseGraphModel
from .graph_networks import GraphAttentionNetwork


class GraphCritic(BaseGraphModel):
    """Graph-based critic network for Q-value approximation in TD3."""
    
    def setup(self):
        self.graph_encoder = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=3,
            dropout_rate=self.dropout_rate,
        )
        
        # Action processing
        self.action_encoder = nn.Sequential([
            nn.Dense(self.hidden_dim // 2),
            nn.activation.relu,
        ])
        
        # Combined Q-value head  
        self.q_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.hidden_dim // 2),
            nn.activation.relu,
            nn.Dense(1),
        ])
    
    def __call__(
        self,
        graph_state,
        actions: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through critic network for Q-value estimation."""
        # Handle both GraphState objects and direct tensor inputs
        if hasattr(graph_state, 'node_features'):
            node_features = graph_state.node_features
            edge_index = graph_state.edge_index
            edge_features = graph_state.edge_features
        else:
            # Backward compatibility 
            node_features = graph_state
            edge_index = None
            edge_features = None
        
        # Encode graph
        node_embeddings = self.graph_encoder(
            node_features, edge_index, edge_features, training
        )
        
        # Global pooling for state representation
        state_embedding = jnp.mean(node_embeddings, axis=0)
        
        # Encode actions
        action_embedding = self.action_encoder(actions)
        
        # Combine state and action representations
        combined_embedding = jnp.concatenate([state_embedding, action_embedding], axis=-1)
        
        # Estimate Q-value
        q_value = self.q_head(combined_embedding).squeeze(-1)
        
        return q_value
    
    def get_node_values(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Get value estimates for individual nodes.
        
        Returns:
            Node-level value estimates [num_nodes]
        """
        # Encode graph with full context
        node_embeddings = self.graph_encoder(
            node_features, edge_index, edge_features, training
        )
        
        # Get value for each node
        node_values = jax.vmap(self.value_head)(node_embeddings)
        node_values = node_values.squeeze(-1)
        
        return node_values


class DoubleGraphCritic(nn.Module):
    """Double critic network to reduce overestimation bias."""
    
    hidden_dim: int = 128
    dropout_rate: float = 0.1
    
    def setup(self):
        # Two separate critic networks
        self.critic1 = GraphCritic(
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        )
        
        self.critic2 = GraphCritic(
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        )
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through both critics.
        
        Returns:
            value1: First critic's value estimate
            value2: Second critic's value estimate
        """
        value1 = self.critic1(node_features, edge_index, edge_features, training)
        value2 = self.critic2(node_features, edge_index, edge_features, training)
        
        return value1, value2
    
    def get_min_value(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Get minimum value estimate for reduced overestimation."""
        value1, value2 = self(node_features, edge_index, edge_features, training)
        return jnp.minimum(value1, value2)


class ActionValueCritic(nn.Module):
    """Critic that estimates Q-values for state-action pairs."""
    
    action_dim: int
    hidden_dim: int = 128
    dropout_rate: float = 0.1
    
    def setup(self):
        # Graph encoder for states
        self.state_encoder = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=3,
            dropout_rate=self.dropout_rate,
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential([
            nn.Dense(self.hidden_dim // 2),
            nn.activation.relu,
            nn.Dense(self.hidden_dim // 2),
        ])
        
        # Combined Q-value head
        self.q_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.hidden_dim // 2),
            nn.activation.relu,
            nn.Dense(1),
        ])
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        actions: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through action-value critic.
        
        Args:
            node_features: Graph node features
            edge_index: Edge connectivity
            actions: Action vector
            edge_features: Edge features
            training: Training mode flag
            
        Returns:
            Q-value estimate
        """
        # Encode state
        node_embeddings = self.state_encoder(
            node_features, edge_index, edge_features, training
        )
        state_embedding = jnp.mean(node_embeddings, axis=0)
        
        # Encode action
        action_embedding = self.action_encoder(actions)
        
        # Combine state and action
        combined = jnp.concatenate([state_embedding, action_embedding], axis=-1)
        
        # Estimate Q-value
        q_value = self.q_head(combined).squeeze(-1)
        
        return q_value


class DistributionalGraphCritic(BaseGraphModel):
    \"\"\"Distributional critic for learning value distributions.\"\"\"
    
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    
    def setup(self):
        self.graph_encoder = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=3,
            dropout_rate=self.dropout_rate,
        )
        
        # Output distribution over value atoms
        self.distribution_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.hidden_dim // 2),
            nn.activation.relu,
            nn.Dense(self.num_atoms),
        ])
        
        # Create support for value distribution
        self.support = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        \"\"\"Forward pass through distributional critic.
        
        Returns:
            logits: Distribution logits over value atoms
            support: Value support (atom locations)
        \"\"\"
        # Encode graph
        node_embeddings = self.graph_encoder(
            node_features, edge_index, edge_features, training
        )
        graph_embedding = jnp.mean(node_embeddings, axis=0)
        
        # Get distribution logits
        logits = self.distribution_head(graph_embedding)
        
        return logits, self.support
    
    def get_expected_value(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        \"\"\"Get expected value from distribution.\"\"\"
        logits, support = self(node_features, edge_index, edge_features, training)
        
        # Convert logits to probabilities
        probs = nn.softmax(logits)
        
        # Compute expected value
        expected_value = jnp.sum(probs * support)
        
        return expected_value
    
    def compute_distributional_loss(
        self,
        logits: jnp.ndarray,
        target_distribution: jnp.ndarray,
    ) -> jnp.ndarray:
        \"\"\"Compute distributional Bellman loss.\"\"\"
        # Convert logits to log probabilities
        log_probs = nn.log_softmax(logits)
        
        # Cross-entropy loss with target distribution
        loss = -jnp.sum(target_distribution * log_probs)
        
        return loss


class AdvantageGraphCritic(nn.Module):
    \"\"\"Dueling network architecture for advantage estimation.\"\"\"
    
    hidden_dim: int = 128
    dropout_rate: float = 0.1
    
    def setup(self):
        # Shared graph encoder
        self.shared_encoder = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=2,
            dropout_rate=self.dropout_rate,
        )
        
        # Value stream
        self.value_stream = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(1),
        ])
        
        # Advantage stream  
        self.advantage_stream = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.hidden_dim),
        ])
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        \"\"\"Forward pass through dueling critic.
        
        Returns:
            value: State value estimate
            advantage: Node-level advantage estimates
        \"\"\"
        # Shared encoding
        node_embeddings = self.shared_encoder(
            node_features, edge_index, edge_features, training
        )
        
        # Value stream (global)
        graph_embedding = jnp.mean(node_embeddings, axis=0)
        value = self.value_stream(graph_embedding).squeeze(-1)
        
        # Advantage stream (per-node)
        advantages = jax.vmap(self.advantage_stream)(node_embeddings)
        advantages = advantages.squeeze(-1)
        
        # Normalize advantages (zero mean)
        advantages = advantages - jnp.mean(advantages)
        
        return value, advantages
    
    def get_q_values(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        \"\"\"Compute Q-values from value and advantage.\"\"\"
        value, advantages = self(node_features, edge_index, edge_features, training)
        
        # Q(s,a) = V(s) + A(s,a)
        q_values = value + advantages
        
        return q_values


class GraphVariationalCritic(nn.Module):
    \"\"\"Variational critic for uncertainty estimation.\"\"\"
    
    hidden_dim: int = 128
    latent_dim: int = 32
    dropout_rate: float = 0.1
    
    def setup(self):
        # Graph encoder
        self.graph_encoder = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=3,
            dropout_rate=self.dropout_rate,
        )
        
        # Variational encoder (q(z|s))
        self.encoder_mean = nn.Dense(self.latent_dim)
        self.encoder_logvar = nn.Dense(self.latent_dim)
        
        # Value decoder (p(v|s,z))
        self.value_decoder = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(1),
        ])
        
        # Prior parameters
        self.prior_mean = 0.0
        self.prior_logvar = 0.0
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        \"\"\"Forward pass through variational critic.
        
        Returns:
            value: Value estimate
            kl_divergence: KL divergence for regularization
            latent_sample: Sampled latent variable
        \"\"\"
        # Encode graph
        node_embeddings = self.graph_encoder(
            node_features, edge_index, edge_features, training
        )
        graph_embedding = jnp.mean(node_embeddings, axis=0)
        
        # Variational encoding
        z_mean = self.encoder_mean(graph_embedding)
        z_logvar = self.encoder_logvar(graph_embedding)
        
        # Sample latent variable
        if training:
            eps = jax.random.normal(rng_key, z_mean.shape)
            z_sample = z_mean + jnp.exp(0.5 * z_logvar) * eps
        else:
            z_sample = z_mean
        
        # Decode value
        decoder_input = jnp.concatenate([graph_embedding, z_sample], axis=-1)
        value = self.value_decoder(decoder_input).squeeze(-1)
        
        # Compute KL divergence
        kl_divergence = self._compute_kl_divergence(z_mean, z_logvar)
        
        return value, kl_divergence, z_sample
    
    def _compute_kl_divergence(
        self, z_mean: jnp.ndarray, z_logvar: jnp.ndarray
    ) -> jnp.ndarray:
        \"\"\"Compute KL divergence between posterior and prior.\"\"\"
        kl = 0.5 * jnp.sum(
            jnp.exp(z_logvar) + z_mean**2 - 1.0 - z_logvar
        )
        return kl
    
    def estimate_uncertainty(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        num_samples: int = 100,
        edge_features: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        \"\"\"Estimate value uncertainty via sampling.
        
        Returns:
            mean_value: Mean value estimate
            value_std: Standard deviation of value estimates
        \"\"\"
        values = []
        
        for i in range(num_samples):
            sample_key = jax.random.fold_in(rng_key, i)
            value, _, _ = self(
                node_features, edge_index, sample_key, edge_features, False
            )
            values.append(value)
        
        values = jnp.stack(values)
        mean_value = jnp.mean(values)
        value_std = jnp.std(values)
        
        return mean_value, value_std