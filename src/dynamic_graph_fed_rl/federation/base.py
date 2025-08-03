"""Base federated learning classes and protocols."""

import abc
import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


class BaseFederatedProtocol(abc.ABC):
    """Base class for federated learning protocols."""
    
    def __init__(
        self,
        num_agents: int,
        communication_round: int = 100,
        compression_ratio: float = 1.0,
        byzantine_tolerance: bool = False,
    ):
        self.num_agents = num_agents
        self.communication_round = communication_round
        self.compression_ratio = compression_ratio
        self.byzantine_tolerance = byzantine_tolerance
        
        # Protocol state
        self.round_count = 0
        self.communication_log = []
        
        # Agent states tracking
        self.agent_states = {}
        self.agent_metrics = {}
    
    @abc.abstractmethod
    async def aggregate_parameters(
        self,
        agent_parameters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate parameters from multiple agents."""
        pass
    
    @abc.abstractmethod
    def select_communication_partners(
        self,
        agent_id: int,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Select communication partners for an agent."""
        pass
    
    def compress_parameters(
        self,
        parameters: Dict[str, Any],
        compression_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compress parameters for communication efficiency."""
        if compression_ratio is None:
            compression_ratio = self.compression_ratio
        
        if compression_ratio >= 1.0:
            return parameters  # No compression
        
        compressed = {}
        for key, value in parameters.items():
            if isinstance(value, jnp.ndarray):
                # Gradient sparsification
                flat_params = value.flatten()
                k = int(len(flat_params) * compression_ratio)
                
                # Keep top-k by magnitude
                indices = jnp.argsort(jnp.abs(flat_params))[-k:]
                compressed_flat = jnp.zeros_like(flat_params)
                compressed_flat = compressed_flat.at[indices].set(flat_params[indices])
                
                compressed[key] = compressed_flat.reshape(value.shape)
            else:
                compressed[key] = value
        
        return compressed
    
    def detect_byzantine_agents(
        self,
        agent_parameters: List[Dict[str, Any]],
        threshold: float = 0.2,
    ) -> List[int]:
        """Detect potentially byzantine agents based on parameter deviation."""
        if not self.byzantine_tolerance or len(agent_parameters) < 3:
            return []
        
        byzantine_agents = []
        
        # Compute pairwise distances between agent parameters
        distances = []
        for i, params_i in enumerate(agent_parameters):
            agent_distances = []
            for j, params_j in enumerate(agent_parameters):
                if i != j:
                    distance = self._compute_parameter_distance(params_i, params_j)
                    agent_distances.append(distance)
            distances.append(jnp.mean(jnp.array(agent_distances)))
        
        # Identify outliers
        distances = jnp.array(distances)
        median_distance = jnp.median(distances)
        mad = jnp.median(jnp.abs(distances - median_distance))
        
        for i, distance in enumerate(distances):
            if distance > median_distance + threshold * mad:
                byzantine_agents.append(i)
        
        return byzantine_agents
    
    def _compute_parameter_distance(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any],
    ) -> float:
        """Compute L2 distance between parameter sets."""
        total_distance = 0.0
        total_elements = 0
        
        for key in params1.keys():
            if key in params2 and isinstance(params1[key], jnp.ndarray):
                diff = params1[key] - params2[key]
                total_distance += jnp.sum(diff ** 2)
                total_elements += diff.size
        
        if total_elements > 0:
            return float(jnp.sqrt(total_distance / total_elements))
        else:
            return 0.0
    
    def log_communication(
        self,
        sender_id: int,
        receiver_id: int,
        message_size: int,
        timestamp: float,
    ) -> None:
        """Log communication event."""
        event = {
            "round": self.round_count,
            "sender": sender_id,
            "receiver": receiver_id,
            "size": message_size,
            "timestamp": timestamp,
        }
        self.communication_log.append(event)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        if not self.communication_log:
            return {}
        
        total_messages = len(self.communication_log)
        total_bytes = sum(event["size"] for event in self.communication_log)
        avg_message_size = total_bytes / total_messages if total_messages > 0 else 0
        
        return {
            "total_messages": total_messages,
            "total_bytes": total_bytes,
            "avg_message_size": avg_message_size,
            "rounds": self.round_count,
            "messages_per_round": total_messages / self.round_count if self.round_count > 0 else 0,
        }


class FederatedOptimizer:
    """Optimizer for federated learning with various aggregation methods."""
    
    def __init__(
        self,
        algorithm: str = "fedavg",
        learning_rate: float = 1.0,
        momentum: float = 0.0,
        adaptive_lr: bool = True,
        convergence_threshold: float = 1e-6,
    ):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.adaptive_lr = adaptive_lr
        self.convergence_threshold = convergence_threshold
        
        # State tracking
        self.global_state = None
        self.momentum_state = None
        self.round_count = 0
        self.convergence_history = []
    
    def aggregate(
        self,
        agent_parameters: List[Dict[str, Any]],
        agent_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Aggregate parameters using specified algorithm."""
        if not agent_parameters:
            return {}
        
        if agent_weights is None:
            agent_weights = [1.0 / len(agent_parameters)] * len(agent_parameters)
        
        if self.algorithm == "fedavg":
            return self._federated_averaging(agent_parameters, agent_weights)
        elif self.algorithm == "fedprox":
            return self._federated_proximal(agent_parameters, agent_weights)
        elif self.algorithm == "scaffold":
            return self._scaffold_aggregation(agent_parameters, agent_weights)
        elif self.algorithm == "fedadam":
            return self._federated_adam(agent_parameters, agent_weights)
        else:
            raise ValueError(f"Unknown aggregation algorithm: {self.algorithm}")
    
    def _federated_averaging(
        self,
        agent_parameters: List[Dict[str, Any]],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Standard FedAvg aggregation."""
        if not agent_parameters:
            return {}
        
        # Initialize aggregated parameters
        aggregated = {}
        
        # Get parameter keys from first agent
        keys = list(agent_parameters[0].keys())
        
        for key in keys:
            # Skip non-tensor parameters
            if not isinstance(agent_parameters[0][key], jnp.ndarray):
                aggregated[key] = agent_parameters[0][key]
                continue
            
            # Weighted average of parameters
            weighted_sum = jnp.zeros_like(agent_parameters[0][key])
            
            for params, weight in zip(agent_parameters, weights):
                if key in params:
                    weighted_sum += weight * params[key]
            
            aggregated[key] = weighted_sum
        
        return aggregated
    
    def _federated_proximal(
        self,
        agent_parameters: List[Dict[str, Any]],
        weights: List[float],
        proximal_term: float = 0.01,
    ) -> Dict[str, Any]:
        """FedProx aggregation with proximal term."""
        # Start with standard averaging
        aggregated = self._federated_averaging(agent_parameters, weights)
        
        # Apply proximal regularization if we have previous global state
        if self.global_state is not None:
            for key in aggregated.keys():
                if key in self.global_state and isinstance(aggregated[key], jnp.ndarray):
                    # Proximal term: μ/2 * ||w - w_global||^2
                    proximal_update = proximal_term * (aggregated[key] - self.global_state[key])
                    aggregated[key] = aggregated[key] - self.learning_rate * proximal_update
        
        # Update global state
        self.global_state = aggregated.copy()
        
        return aggregated
    
    def _scaffold_aggregation(
        self,
        agent_parameters: List[Dict[str, Any]],
        weights: List[float],
    ) -> Dict[str, Any]:
        """SCAFFOLD aggregation with control variates."""
        # Simplified SCAFFOLD implementation
        # In practice, this would need control variates from agents
        aggregated = self._federated_averaging(agent_parameters, weights)
        
        # Update global state for convergence tracking
        if self.global_state is not None:
            self._update_convergence_metrics(aggregated)
        
        self.global_state = aggregated.copy()
        return aggregated
    
    def _federated_adam(
        self,
        agent_parameters: List[Dict[str, Any]],
        weights: List[float],
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> Dict[str, Any]:
        """FedAdam aggregation with adaptive moments."""
        # Compute pseudo-gradients
        if self.global_state is None:
            self.global_state = self._federated_averaging(agent_parameters, weights)
            self.momentum_state = {
                "m": {k: jnp.zeros_like(v) for k, v in self.global_state.items() 
                      if isinstance(v, jnp.ndarray)},
                "v": {k: jnp.zeros_like(v) for k, v in self.global_state.items() 
                      if isinstance(v, jnp.ndarray)},
            }
            return self.global_state
        
        # Compute pseudo-gradients from aggregated updates
        pseudo_gradients = {}
        aggregated_params = self._federated_averaging(agent_parameters, weights)
        
        for key in aggregated_params.keys():
            if isinstance(aggregated_params[key], jnp.ndarray):
                pseudo_gradients[key] = self.global_state[key] - aggregated_params[key]
        
        # Adam update
        self.round_count += 1
        
        for key in pseudo_gradients.keys():
            # Update biased first moment estimate
            self.momentum_state["m"][key] = (
                beta1 * self.momentum_state["m"][key] + 
                (1 - beta1) * pseudo_gradients[key]
            )
            
            # Update biased second moment estimate
            self.momentum_state["v"][key] = (
                beta2 * self.momentum_state["v"][key] + 
                (1 - beta2) * pseudo_gradients[key] ** 2
            )
            
            # Bias correction
            m_hat = self.momentum_state["m"][key] / (1 - beta1 ** self.round_count)
            v_hat = self.momentum_state["v"][key] / (1 - beta2 ** self.round_count)
            
            # Update parameters
            self.global_state[key] = self.global_state[key] - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        return self.global_state
    
    def _update_convergence_metrics(self, new_params: Dict[str, Any]) -> None:
        """Update convergence metrics."""
        if self.global_state is None:
            return
        
        # Compute parameter change magnitude
        total_change = 0.0
        total_params = 0
        
        for key in new_params.keys():
            if key in self.global_state and isinstance(new_params[key], jnp.ndarray):
                diff = new_params[key] - self.global_state[key]
                total_change += jnp.sum(diff ** 2)
                total_params += diff.size
        
        if total_params > 0:
            rms_change = float(jnp.sqrt(total_change / total_params))
            self.convergence_history.append(rms_change)
    
    def has_converged(self, window_size: int = 10) -> bool:
        """Check if optimization has converged."""
        if len(self.convergence_history) < window_size:
            return False
        
        recent_changes = self.convergence_history[-window_size:]
        avg_change = sum(recent_changes) / len(recent_changes)
        
        return avg_change < self.convergence_threshold
    
    def adapt_learning_rate(self, performance_metric: float) -> None:
        """Adapt learning rate based on performance."""
        if not self.adaptive_lr:
            return
        
        # Simple adaptive scheme
        if len(self.convergence_history) >= 2:
            recent_change = self.convergence_history[-1]
            prev_change = self.convergence_history[-2]
            
            if recent_change > prev_change:
                # Performance getting worse, reduce learning rate
                self.learning_rate *= 0.9
            elif recent_change < prev_change * 0.5:
                # Performance improving quickly, slightly increase learning rate
                self.learning_rate *= 1.01
        
        # Clamp learning rate
        self.learning_rate = jnp.clip(self.learning_rate, 1e-6, 1.0)


class FederatedActorCritic:
    """Wrapper for federated actor-critic algorithms."""
    
    def __init__(
        self,
        num_agents: int,
        communication: str = "async_gossip",
        buffer_type: str = "graph_temporal",
        aggregation_interval: int = 100,
        aggregation_method: str = "fedavg",
    ):
        self.num_agents = num_agents
        self.communication = communication
        self.buffer_type = buffer_type
        self.aggregation_interval = aggregation_interval
        self.aggregation_method = aggregation_method
        
        # Initialize federated optimizer
        self.optimizer = FederatedOptimizer(algorithm=aggregation_method)
        
        # Communication protocol
        if communication == "async_gossip":
            from .gossip import AsyncGossipProtocol
            self.protocol = AsyncGossipProtocol(num_agents)
        elif communication == "hierarchical":
            from .hierarchical import FederatedHierarchy
            self.protocol = FederatedHierarchy(num_agents)
        else:
            raise ValueError(f"Unknown communication protocol: {communication}")
        
        # Tracking
        self.global_round = 0
        self.agents = []
    
    def add_agent(self, agent) -> None:
        """Add an agent to the federated system."""
        if len(self.agents) >= self.num_agents:
            raise ValueError("Maximum number of agents reached")
        
        self.agents.append(agent)
    
    async def federated_round(self) -> Dict[str, Any]:
        """Execute one round of federated learning."""
        if len(self.agents) < self.num_agents:
            raise ValueError(f"Need {self.num_agents} agents, have {len(self.agents)}")
        
        # Collect parameters from all agents
        agent_parameters = []
        for agent in self.agents:
            if hasattr(agent, 'actor_state') and hasattr(agent, 'critic1_state'):
                params = {
                    "actor": agent.actor_state.params,
                    "critic1": agent.critic1_state.params,
                    "critic2": agent.critic2_state.params if hasattr(agent, 'critic2_state') else None,
                }
                agent_parameters.append(params)
        
        # Aggregate parameters
        if self.communication == "async_gossip":
            aggregated_params = await self.protocol.aggregate_parameters(agent_parameters)
        else:
            # Synchronous aggregation
            aggregated_params = self.optimizer.aggregate(agent_parameters)
        
        # Update all agents with aggregated parameters
        for agent in self.agents:
            if "actor" in aggregated_params:
                agent.actor_state = agent.actor_state.replace(params=aggregated_params["actor"])
                agent.target_actor_state = agent.target_actor_state.replace(params=aggregated_params["actor"])
            
            if "critic1" in aggregated_params:
                agent.critic1_state = agent.critic1_state.replace(params=aggregated_params["critic1"])
                agent.target_critic1_state = agent.target_critic1_state.replace(params=aggregated_params["critic1"])
            
            if "critic2" in aggregated_params and aggregated_params["critic2"] is not None:
                agent.critic2_state = agent.critic2_state.replace(params=aggregated_params["critic2"])
                agent.target_critic2_state = agent.target_critic2_state.replace(params=aggregated_params["critic2"])
        
        self.global_round += 1
        
        # Return aggregation metrics
        return {
            "global_round": self.global_round,
            "num_agents": len(self.agents),
            "communication_stats": self.protocol.get_communication_stats() if hasattr(self.protocol, 'get_communication_stats') else {},
            "convergence": self.optimizer.has_converged(),
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        agent_metrics = []
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'get_training_stats'):
                stats = agent.get_training_stats()
                stats["agent_id"] = i
                agent_metrics.append(stats)
        
        return {
            "global_round": self.global_round,
            "num_agents": len(self.agents),
            "agent_metrics": agent_metrics,
            "optimizer_stats": {
                "convergence_history": self.optimizer.convergence_history,
                "learning_rate": self.optimizer.learning_rate,
                "has_converged": self.optimizer.has_converged(),
            },
            "communication_stats": self.protocol.get_communication_stats() if hasattr(self.protocol, 'get_communication_stats') else {},
        }