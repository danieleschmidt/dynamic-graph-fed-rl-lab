import secrets
"""Graph TD3 algorithm for federated reinforcement learning on dynamic graphs."""

import pickle
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from .base import BaseGraphAlgorithm, AdaptiveNoise, GraphMetricsTracker
from .buffers import GraphTemporalBuffer, collate_graph_transitions
from ..environments.base import GraphState, GraphTransition
from ..models.actors import GraphActor
from ..models.critics import GraphCritic


class GraphTD3(BaseGraphAlgorithm):
    """Twin Delayed Deep Deterministic Policy Gradient for graph environments."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        edge_dim: int = 32,
        hidden_dim: int = 128,
        gnn_type: str = "temporal_attention",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        buffer_size: int = 100000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
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
        
        self.gnn_type = gnn_type
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        # Initialize noise
        self.noise = AdaptiveNoise(
            initial_noise=exploration_noise,
            noise_type="gaussian"
        )
        
        # Initialize buffer
        self.buffer = GraphTemporalBuffer(
            capacity=buffer_size // 10,  # Store episodes, not individual transitions
            sequence_length=10,
        )
        
        # Metrics tracking
        self.metrics = GraphMetricsTracker(window_size=100)
        
        # Initialize networks
        self._initialize_networks()
        
        # Delayed policy update counter
        self._policy_update_counter = 0
    
    def _initialize_networks(self) -> None:
        """Initialize actor and critic networks."""
        # Actor network
        self.actor = GraphActor(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            gnn_type=self.gnn_type,
            edge_dim=self.edge_dim,
        )
        
        # Twin critics
        self.critic1 = GraphCritic(
            hidden_dim=self.hidden_dim,
            gnn_type=self.gnn_type,
            edge_dim=self.edge_dim,
        )
        
        self.critic2 = GraphCritic(
            hidden_dim=self.hidden_dim,
            gnn_type=self.gnn_type,
            edge_dim=self.edge_dim,
        )
        
        # Create dummy inputs for initialization
        dummy_graph = self._create_dummy_graph()
        dummy_action = jnp.zeros((1, self.action_dim))
        
        # Initialize training states
        self.actor_state = self._create_train_state(
            self.actor,
            (dummy_graph,),
            self.learning_rate,
        )
        
        self.critic1_state = self._create_train_state(
            self.critic1,
            (dummy_graph, dummy_action),
            self.learning_rate,
        )
        
        self.critic2_state = self._create_train_state(
            self.critic2,
            (dummy_graph, dummy_action),
            self.learning_rate,
        )
        
        # Target networks (copy of online networks)
        self.target_actor_state = self.actor_state
        self.target_critic1_state = self.critic1_state
        self.target_critic2_state = self.critic2_state
    
    def _create_dummy_graph(self) -> GraphState:
        """Create dummy graph for network initialization."""
        num_nodes = 10
        num_edges = 20
        
        # Random node features
        node_features = jax.random.normal(
            self.rng_key, (num_nodes, self.state_dim)
        )
        
        # Random edge index
        edge_index = jax.secrets.SystemRandom().randint(
            self.rng_key, (2, num_edges), 0, num_nodes
        )
        
        # Random edge features
        edge_features = jax.random.normal(
            self.rng_key, (num_edges, self.edge_dim)
        ) if self.edge_dim > 0 else None
        
        return GraphState(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=None,
            num_nodes=num_nodes,
            num_edges=num_edges,
            timestamp=0.0,
        )
    
    def select_action(
        self,
        state: GraphState,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Select action using current policy."""
        # Forward pass through actor
        action = self.actor_state.apply_fn(
            self.actor_state.params,
            state,
        )
        
        # Add exploration noise if not deterministic
        if not deterministic and not self.is_evaluation_mode():
            self.rng_key, noise_key = jax.random.split(self.rng_key)
            noise = self.noise.sample_noise(noise_key, action.shape)
            action = action + noise
            
            # Update noise schedule
            self.noise.update_noise()
        
        # Clip action to valid range
        action = jnp.clip(action, -1.0, 1.0)
        
        return action
    
    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Update TD3 parameters."""
        if not self.can_update():
            return {}
        
        # Sample batch from buffer
        transitions = self.buffer.sample_temporal(
            self.batch_size,
            lookback=5,
            respect_topology_changes=True,
        )
        
        if not transitions:
            return {}
        
        # Flatten transitions (use last transition from each sequence)
        flat_transitions = [seq[-1] for seq in transitions if seq]
        
        if len(flat_transitions) < self.batch_size // 2:
            return {}
        
        # Collate into batch
        batch = collate_graph_transitions(flat_transitions)
        
        # Update critics
        critic_losses = self._update_critics(batch)
        
        # Update actor (delayed)
        actor_loss = 0.0
        if self._policy_update_counter % self.policy_freq == 0:
            actor_loss = self._update_actor(batch)
            self._update_target_networks()
        
        self._policy_update_counter += 1
        self.training_step += 1
        
        # Track metrics
        metrics = {
            "critic1_loss": critic_losses[0],
            "critic2_loss": critic_losses[1],
            "actor_loss": actor_loss,
            "buffer_size": len(self.buffer),
            "noise_level": self.noise.current_noise,
        }
        
        for name, value in metrics.items():
            self.metrics.add_metric(name, value)
        
        return metrics
    
    def _update_critics(self, batch: Dict[str, jnp.ndarray]) -> Tuple[float, float]:
        """Update both critic networks."""
        
        def critic_loss_fn(critic_params, critic_apply_fn, batch):
            """Compute critic loss."""
            # Current Q-values
            current_q = critic_apply_fn(
                critic_params,
                self._batch_to_graph_state(batch, "states"),
                batch["actions"],
            )
            
            # Target Q-values
            with jax.lax.stop_gradient():
                # Target actions with noise
                target_actions = self.target_actor_state.apply_fn(
                    self.target_actor_state.params,
                    self._batch_to_graph_state(batch, "next_states"),
                )
                
                # Add clipped noise to target actions
                self.rng_key, noise_key = jax.random.split(self.rng_key)
                noise = jax.random.normal(noise_key, target_actions.shape) * self.policy_noise
                noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
                target_actions = jnp.clip(target_actions + noise, -1.0, 1.0)
                
                # Target Q-values (minimum of twin critics)
                target_q1 = self.target_critic1_state.apply_fn(
                    self.target_critic1_state.params,
                    self._batch_to_graph_state(batch, "next_states"),
                    target_actions,
                )
                
                target_q2 = self.target_critic2_state.apply_fn(
                    self.target_critic2_state.params,
                    self._batch_to_graph_state(batch, "next_states"),
                    target_actions,
                )
                
                target_q = jnp.minimum(target_q1, target_q2)
                target_q = batch["rewards"] + self.gamma * (1 - batch["dones"]) * target_q
            
            # MSE loss
            loss = jnp.mean((current_q - target_q) ** 2)
            return loss
        
        # Update critic 1
        grad_fn1 = jax.grad(critic_loss_fn)
        grads1 = grad_fn1(
            self.critic1_state.params,
            self.critic1_state.apply_fn,
            batch,
        )
        self.critic1_state = self.critic1_state.apply_gradients(grads=grads1)
        loss1 = critic_loss_fn(self.critic1_state.params, self.critic1_state.apply_fn, batch)
        
        # Update critic 2
        grad_fn2 = jax.grad(critic_loss_fn)
        grads2 = grad_fn2(
            self.critic2_state.params,
            self.critic2_state.apply_fn,
            batch,
        )
        self.critic2_state = self.critic2_state.apply_gradients(grads=grads2)
        loss2 = critic_loss_fn(self.critic2_state.params, self.critic2_state.apply_fn, batch)
        
        return float(loss1), float(loss2)
    
    def _update_actor(self, batch: Dict[str, jnp.ndarray]) -> float:
        """Update actor network."""
        
        def actor_loss_fn(actor_params):
            """Compute actor loss (deterministic policy gradient)."""
            # Actor actions
            actions = self.actor_state.apply_fn(
                actor_params,
                self._batch_to_graph_state(batch, "states"),
            )
            
            # Q-value from first critic
            q_value = self.critic1_state.apply_fn(
                self.critic1_state.params,
                self._batch_to_graph_state(batch, "states"),
                actions,
            )
            
            # Negative Q-value (maximize Q)
            loss = -jnp.mean(q_value)
            return loss
        
        # Update actor
        grad_fn = jax.grad(actor_loss_fn)
        grads = grad_fn(self.actor_state.params)
        self.actor_state = self.actor_state.apply_gradients(grads=grads)
        
        # Compute loss for metrics
        loss = actor_loss_fn(self.actor_state.params)
        return float(loss)
    
    def _update_target_networks(self) -> None:
        """Soft update of target networks."""
        # Update target actor
        self.target_actor_state = self._soft_update(
            self.target_actor_state,
            self.actor_state,
            self.tau,
        )
        
        # Update target critics
        self.target_critic1_state = self._soft_update(
            self.target_critic1_state,
            self.critic1_state,
            self.tau,
        )
        
        self.target_critic2_state = self._soft_update(
            self.target_critic2_state,
            self.critic2_state,
            self.tau,
        )
    
    def _batch_to_graph_state(self, batch: Dict[str, jnp.ndarray], key: str) -> GraphState:
        """Convert batch data to GraphState format."""
        states = batch[key]  # (batch_size, max_nodes, node_dim)
        edge_indices = batch[f"{key.replace('states', 'edge_indices')}"]  # (batch_size, 2, max_edges)
        
        # For simplicity, use first sample in batch
        # In practice, you'd need to handle variable graph sizes properly
        node_features = states[0]  # (max_nodes, node_dim)
        edge_index = edge_indices[0]  # (2, max_edges)
        
        # Find actual number of nodes and edges (non-zero)
        num_nodes = jnp.sum(jnp.any(node_features != 0, axis=1))
        num_edges = jnp.sum(jnp.any(edge_index != 0, axis=0))
        
        edge_features = batch.get("edge_features")
        if edge_features is not None:
            edge_features = edge_features[0]  # (max_edges, edge_dim)
        
        return GraphState(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=None,
            num_nodes=int(num_nodes),
            num_edges=int(num_edges),
            timestamp=0.0,
        )
    
    def get_q_values(self, state: GraphState, action: jnp.ndarray) -> Tuple[float, float]:
        """Get Q-values from both critics."""
        q1 = self.critic1_state.apply_fn(
            self.critic1_state.params,
            state,
            action,
        )
        
        q2 = self.critic2_state.apply_fn(
            self.critic2_state.params,
            state,
            action,
        )
        
        return float(q1), float(q2)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = super().get_training_stats()
        
        # Add TD3-specific stats
        stats.update({
            "policy_updates": self._policy_update_counter // self.policy_freq,
            "noise_level": self.noise.current_noise,
            "metrics_summary": self.metrics.get_summary(),
        })
        
        return stats
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save algorithm checkpoint."""
        checkpoint = {
            "actor_state": self.actor_state,
            "critic1_state": self.critic1_state,
            "critic2_state": self.critic2_state,
            "target_actor_state": self.target_actor_state,
            "target_critic1_state": self.target_critic1_state,
            "target_critic2_state": self.target_critic2_state,
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "total_timesteps": self.total_timesteps,
            "noise": self.noise,
            "policy_update_counter": self._policy_update_counter,
            "rng_key": self.rng_key,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "edge_dim": self.edge_dim,
                "hidden_dim": self.hidden_dim,
                "gnn_type": self.gnn_type,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "tau": self.tau,
                "policy_noise": self.policy_noise,
                "noise_clip": self.noise_clip,
                "policy_freq": self.policy_freq,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
            },
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load algorithm checkpoint."""
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)
        
        # Restore states
        self.actor_state = checkpoint["actor_state"]
        self.critic1_state = checkpoint["critic1_state"]
        self.critic2_state = checkpoint["critic2_state"]
        self.target_actor_state = checkpoint["target_actor_state"]
        self.target_critic1_state = checkpoint["target_critic1_state"]
        self.target_critic2_state = checkpoint["target_critic2_state"]
        
        # Restore counters
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.total_timesteps = checkpoint["total_timesteps"]
        self._policy_update_counter = checkpoint["policy_update_counter"]
        
        # Restore noise and RNG
        self.noise = checkpoint["noise"]
        self.rng_key = checkpoint["rng_key"]
        
        print(f"Loaded checkpoint from {filepath}")
        print(f"Training step: {self.training_step}")
        print(f"Episode count: {self.episode_count}")


class FederatedGraphTD3:
    """Federated wrapper for Graph TD3 algorithm."""
    
    def __init__(
        self,
        num_agents: int,
        agent_configs: List[Dict[str, Any]],
        communication_round: int = 100,
        aggregation_method: str = "fedavg",
    ):
        self.num_agents = num_agents
        self.communication_round = communication_round
        self.aggregation_method = aggregation_method
        
        # Initialize agents
        self.agents: List[GraphTD3] = []
        for i, config in enumerate(agent_configs):
            agent = GraphTD3(**config)
            self.agents.append(agent)
        
        # Global model for aggregation
        self.global_model_params = None
        self.round_count = 0
    
    def local_update(self, agent_id: int, num_steps: int) -> Dict[str, float]:
        """Perform local updates for specific agent."""
        agent = self.agents[agent_id]
        
        total_metrics = {}
        for _ in range(num_steps):
            if agent.can_update():
                metrics = agent.update({})
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = []
                    total_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {}
        for key, values in total_metrics.items():
            avg_metrics[key] = sum(values) / len(values) if values else 0.0
        
        return avg_metrics
    
    def aggregate_parameters(self) -> None:
        """Aggregate parameters from all agents."""
        if self.aggregation_method == "fedavg":
            self._federated_averaging()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        self.round_count += 1
    
    def _federated_averaging(self) -> None:
        """Perform federated averaging of agent parameters."""
        # Get parameters from all agents
        agent_params = []
        for agent in self.agents:
            params = {
                "actor": agent.actor_state.params,
                "critic1": agent.critic1_state.params,
                "critic2": agent.critic2_state.params,
            }
            agent_params.append(params)
        
        # Average parameters
        if not agent_params:
            return
        
        # Initialize averaged parameters
        avg_params = {}
        for network in ["actor", "critic1", "critic2"]:
            avg_params[network] = jax.tree_map(
                lambda *params: jnp.mean(jnp.stack(params), axis=0),
                *[p[network] for p in agent_params]
            )
        
        # Update all agents with averaged parameters
        for agent in self.agents:
            agent.actor_state = agent.actor_state.replace(params=avg_params["actor"])
            agent.critic1_state = agent.critic1_state.replace(params=avg_params["critic1"])
            agent.critic2_state = agent.critic2_state.replace(params=avg_params["critic2"])
            
            # Also update target networks
            agent.target_actor_state = agent.target_actor_state.replace(params=avg_params["actor"])
            agent.target_critic1_state = agent.target_critic1_state.replace(params=avg_params["critic1"])
            agent.target_critic2_state = agent.target_critic2_state.replace(params=avg_params["critic2"])
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics from all agents."""
        all_stats = []
        for agent in self.agents:
            stats = agent.get_training_stats()
            all_stats.append(stats)
        
        # Aggregate numeric metrics
        global_stats = {
            "round_count": self.round_count,
            "num_agents": self.num_agents,
            "avg_training_steps": sum(s["training_step"] for s in all_stats) / self.num_agents,
            "avg_episode_count": sum(s["episode_count"] for s in all_stats) / self.num_agents,
            "total_timesteps": sum(s["total_timesteps"] for s in all_stats),
        }
        
        return global_stats