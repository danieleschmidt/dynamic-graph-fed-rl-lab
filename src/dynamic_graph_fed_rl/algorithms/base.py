"""Base algorithm class for graph-based reinforcement learning."""

import abc
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from ..environments.base import GraphState, GraphTransition


class BaseGraphAlgorithm(abc.ABC):
    """Base class for graph-based RL algorithms."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        edge_dim: int = 32,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Initialize random key
        self.rng_key = jax.random.PRNGKey(seed)
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.total_timesteps = 0
        
        # Networks will be initialized by subclasses
        self.actor_state: Optional[TrainState] = None
        self.critic_state: Optional[TrainState] = None
        
        # Buffer will be initialized by subclasses
        self.buffer = None
    
    @abc.abstractmethod
    def select_action(
        self,
        state: GraphState,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Select action for given state."""
        pass
    
    @abc.abstractmethod
    def update(
        self,
        batch: Dict[str, jnp.ndarray],
    ) -> Dict[str, float]:
        """Update algorithm parameters."""
        pass
    
    @abc.abstractmethod
    def save_checkpoint(self, filepath: str) -> None:
        """Save algorithm checkpoint."""
        pass
    
    @abc.abstractmethod
    def load_checkpoint(self, filepath: str) -> None:
        """Load algorithm checkpoint."""
        pass
    
    def add_transition(self, transition: GraphTransition) -> None:
        """Add transition to replay buffer."""
        if self.buffer is not None:
            self.buffer.add(transition)
    
    def can_update(self) -> bool:
        """Check if algorithm can perform update."""
        if self.buffer is None:
            return False
        return len(self.buffer) >= self.batch_size
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "total_timesteps": self.total_timesteps,
        }
        
        if self.buffer is not None:
            stats["buffer_size"] = len(self.buffer)
        
        return stats
    
    def reset_episode(self) -> None:
        """Reset episode-specific tracking."""
        self.episode_count += 1
    
    def step(self) -> None:
        """Increment timestep counter."""
        self.total_timesteps += 1
    
    def _soft_update(
        self,
        target_state: TrainState,
        online_state: TrainState,
        tau: float,
    ) -> TrainState:
        """Soft update target network parameters."""
        new_target_params = jax.tree_map(
            lambda target, online: tau * online + (1 - tau) * target,
            target_state.params,
            online_state.params,
        )
        
        return target_state.replace(params=new_target_params)
    
    def _create_train_state(
        self,
        network: nn.Module,
        dummy_input: Any,
        learning_rate: float,
    ) -> TrainState:
        """Create training state for a network."""
        self.rng_key, init_key = jax.random.split(self.rng_key)
        
        # Initialize network parameters
        params = network.init(init_key, *dummy_input)
        
        # Create optimizer
        optimizer = optax.adam(learning_rate)
        
        # Create training state
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=optimizer,
        )
        
        return train_state
    
    def set_evaluation_mode(self, eval_mode: bool = True) -> None:
        """Set algorithm to evaluation mode."""
        self._evaluation_mode = eval_mode
    
    def is_evaluation_mode(self) -> bool:
        """Check if in evaluation mode."""
        return getattr(self, '_evaluation_mode', False)


class GraphLearningRateScheduler:
    """Learning rate scheduler for graph algorithms."""
    
    def __init__(
        self,
        initial_lr: float = 3e-4,
        decay_rate: float = 0.99,
        decay_steps: int = 1000,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
    ):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
    
    def __call__(self, step: int) -> float:
        """Compute learning rate for given step."""
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr * (step / self.warmup_steps)
        else:
            # Exponential decay
            decay_factor = self.decay_rate ** ((step - self.warmup_steps) / self.decay_steps)
            lr = self.initial_lr * decay_factor
        
        return max(lr, self.min_lr)


class AdaptiveNoise:
    """Adaptive noise for exploration in graph environments."""
    
    def __init__(
        self,
        initial_noise: float = 0.1,
        min_noise: float = 0.01,
        decay_rate: float = 0.9995,
        noise_type: str = "gaussian",
    ):
        self.initial_noise = initial_noise
        self.current_noise = initial_noise
        self.min_noise = min_noise
        self.decay_rate = decay_rate
        self.noise_type = noise_type
        
        self.step_count = 0
    
    def sample_noise(self, rng_key: jax.random.PRNGKey, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Sample noise with current noise level."""
        if self.noise_type == "gaussian":
            noise = jax.random.normal(rng_key, shape) * self.current_noise
        elif self.noise_type == "uniform":
            noise = jax.random.uniform(
                rng_key, shape, minval=-self.current_noise, maxval=self.current_noise
            )
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return noise
    
    def update_noise(self) -> None:
        """Update noise level (decay)."""
        self.current_noise = max(
            self.min_noise,
            self.current_noise * self.decay_rate
        )
        self.step_count += 1
    
    def reset_noise(self) -> None:
        """Reset noise to initial level."""
        self.current_noise = self.initial_noise
        self.step_count = 0


class GraphMetricsTracker:
    """Track training metrics for graph algorithms."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.step_count = 0
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Keep only recent values
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name] = self.metrics[name][-self.window_size:]
        
        self.step_count += 1
    
    def get_mean(self, name: str) -> float:
        """Get mean of recent metric values."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        return float(jnp.mean(jnp.array(self.metrics[name])))
    
    def get_std(self, name: str) -> float:
        """Get standard deviation of recent metric values."""
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return 0.0
        
        return float(jnp.std(jnp.array(self.metrics[name])))
    
    def get_latest(self, name: str) -> float:
        """Get latest metric value."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        return self.metrics[name][-1]
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        
        for name in self.metrics:
            if self.metrics[name]:
                summary[name] = {
                    "mean": self.get_mean(name),
                    "std": self.get_std(name),
                    "latest": self.get_latest(name),
                    "min": float(jnp.min(jnp.array(self.metrics[name]))),
                    "max": float(jnp.max(jnp.array(self.metrics[name]))),
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}
        self.step_count = 0


def create_graph_optimizer(
    learning_rate: Union[float, optax.Schedule],
    optimizer_type: str = "adam",
    gradient_clip_norm: Optional[float] = None,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Create optimizer for graph neural networks."""
    
    # Base optimizer
    if optimizer_type == "adam":
        optimizer = optax.adam(learning_rate)
    elif optimizer_type == "adamw":
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optax.sgd(learning_rate)
    elif optimizer_type == "rmsprop":
        optimizer = optax.rmsprop(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Add gradient clipping if specified
    if gradient_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(gradient_clip_norm),
            optimizer,
        )
    
    return optimizer