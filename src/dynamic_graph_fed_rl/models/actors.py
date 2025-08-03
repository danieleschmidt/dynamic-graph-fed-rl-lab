"""Actor networks for graph-based reinforcement learning."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .base import BaseGraphModel
from .graph_networks import GraphAttentionNetwork


class GraphActor(BaseGraphModel):
    """Base actor network for graph environments."""
    
    action_dim: int
    max_action: float = 1.0
    
    def setup(self):
        self.graph_encoder = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=3,
            dropout_rate=self.dropout_rate,
        )
        
        self.policy_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.hidden_dim // 2),
            nn.activation.relu,
            nn.Dense(self.action_dim),
        ])
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through actor network."""
        # Encode graph
        node_embeddings = self.graph_encoder(
            node_features, edge_index, edge_features, training
        )
        
        # Global pooling for policy input
        graph_embedding = jnp.mean(node_embeddings, axis=0)
        
        # Generate action logits/values
        actions = self.policy_head(graph_embedding)
        
        return actions
    
    def get_local_actions(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        node_indices: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Get actions for specific nodes (local control).
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge connectivity
            node_indices: Indices of nodes to get actions for
            edge_features: Edge features
            training: Training mode flag
            
        Returns:
            Actions for specified nodes
        """
        # Encode entire graph for context
        node_embeddings = self.graph_encoder(
            node_features, edge_index, edge_features, training
        )
        
        # Get embeddings for specified nodes
        local_embeddings = node_embeddings[node_indices]
        
        # Generate actions for each local node
        local_actions = jax.vmap(self.policy_head)(local_embeddings)
        
        return local_actions


class ContinuousGraphActor(GraphActor):
    """Actor for continuous action spaces."""
    
    def setup(self):
        super().setup()
        
        # Separate heads for mean and log std
        self.mean_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.action_dim),
            nn.activation.tanh,  # Bounded actions
        ])
        
        self.log_std_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.action_dim),
        ])
        
        # Remove the original policy_head
        delattr(self, 'policy_head')
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning action mean and log std."""
        # Encode graph
        node_embeddings = self.graph_encoder(
            node_features, edge_index, edge_features, training
        )
        
        # Global pooling
        graph_embedding = jnp.mean(node_embeddings, axis=0)
        
        # Generate mean and log std
        action_mean = self.mean_head(graph_embedding) * self.max_action
        log_std = self.log_std_head(graph_embedding)
        
        # Clamp log std for stability
        log_std = jnp.clip(log_std, -20, 2)
        
        return action_mean, log_std
    
    def sample_action(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample action from policy distribution.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
        """
        action_mean, log_std = self(
            node_features, edge_index, edge_features, training
        )
        
        std = jnp.exp(log_std)
        
        # Sample from normal distribution
        noise = jax.random.normal(rng_key, action_mean.shape)
        action = action_mean + std * noise
        
        # Compute log probability
        log_prob = self._compute_log_prob(action, action_mean, log_std)
        
        # Apply action bounds
        action = jnp.clip(action, -self.max_action, self.max_action)
        
        return action, log_prob
    
    def _compute_log_prob(
        self,
        action: jnp.ndarray,
        mean: jnp.ndarray,
        log_std: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action under current policy."""
        std = jnp.exp(log_std)
        
        # Gaussian log probability
        log_prob = -0.5 * jnp.sum(
            jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1
        )
        
        return log_prob


class DiscreteGraphActor(GraphActor):
    """Actor for discrete action spaces."""
    
    temperature: float = 1.0
    
    def setup(self):
        super().setup()
        
        # Override policy head for discrete actions
        self.policy_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.hidden_dim // 2),
            nn.activation.relu,
            nn.Dense(self.action_dim),  # Raw logits
        ])
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass returning action logits."""
        # Encode graph
        node_embeddings = self.graph_encoder(
            node_features, edge_index, edge_features, training
        )
        
        # Global pooling
        graph_embedding = jnp.mean(node_embeddings, axis=0)
        
        # Generate action logits
        logits = self.policy_head(graph_embedding)
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        return logits
    
    def sample_action(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample action from categorical distribution.
        
        Returns:
            action: Sampled action index
            log_prob: Log probability of the action
        """
        logits = self(node_features, edge_index, edge_features, training)
        
        # Sample from categorical distribution
        action = jax.random.categorical(rng_key, logits)
        
        # Compute log probability
        log_probs = nn.log_softmax(logits)
        log_prob = log_probs[action]
        
        return action, log_prob
    
    def get_action_probabilities(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Get action probabilities for all actions."""
        logits = self(node_features, edge_index, edge_features, training)
        return nn.softmax(logits)


class HierarchicalGraphActor(nn.Module):
    """Hierarchical actor for multi-level control."""
    
    hidden_dim: int = 128
    high_level_action_dim: int = 10
    low_level_action_dim: int = 5
    dropout_rate: float = 0.1
    
    def setup(self):
        # High-level policy (global decisions)
        self.high_level_encoder = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=2,
            dropout_rate=self.dropout_rate,
        )
        
        self.high_level_policy = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.high_level_action_dim),
        ])
        
        # Low-level policy (local decisions)
        self.low_level_encoder = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=3,
            dropout_rate=self.dropout_rate,
        )
        
        self.low_level_policy = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.low_level_action_dim),
        ])
        
        # Goal conditioning for low-level policy
        self.goal_projector = nn.Dense(self.hidden_dim)
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through hierarchical actor.
        
        Returns:
            high_level_actions: Global policy actions
            low_level_actions: Local policy actions  
        """
        # High-level policy
        high_level_embeddings = self.high_level_encoder(
            node_features, edge_index, edge_features, training
        )
        global_embedding = jnp.mean(high_level_embeddings, axis=0)
        high_level_actions = self.high_level_policy(global_embedding)
        
        # Low-level policy with goal conditioning
        low_level_embeddings = self.low_level_encoder(
            node_features, edge_index, edge_features, training
        )
        
        # Project high-level action as goal
        goal_embedding = self.goal_projector(high_level_actions)
        
        # Condition low-level policy on goal
        conditioned_embeddings = low_level_embeddings + goal_embedding[None, :]
        
        # Generate low-level actions for each node
        low_level_actions = jax.vmap(self.low_level_policy)(conditioned_embeddings)
        
        return high_level_actions, low_level_actions


class GraphActorCritic(nn.Module):
    """Combined actor-critic network for graph environments."""
    
    action_dim: int
    hidden_dim: int = 128
    max_action: float = 1.0
    dropout_rate: float = 0.1
    
    def setup(self):
        # Shared graph encoder
        self.shared_encoder = GraphAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=3,
            dropout_rate=self.dropout_rate,
        )
        
        # Actor head
        self.actor_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(self.action_dim),
            nn.activation.tanh,
        ])
        
        # Critic head
        self.critic_head = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.activation.relu,
            nn.Dense(1),
        ])
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through actor-critic network.
        
        Returns:
            actions: Actor output
            values: Critic output
        """
        # Shared encoding
        node_embeddings = self.shared_encoder(
            node_features, edge_index, edge_features, training
        )
        graph_embedding = jnp.mean(node_embeddings, axis=0)
        
        # Actor output
        actions = self.actor_head(graph_embedding) * self.max_action
        
        # Critic output
        values = self.critic_head(graph_embedding).squeeze(-1)
        
        return actions, values