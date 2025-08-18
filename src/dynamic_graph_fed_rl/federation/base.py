"""Base classes for federated learning systems."""

import abc
from typing import Any, Dict, List, Optional

import jax.numpy as jnp


class BaseFederatedLearning(abc.ABC):
    """Base class for federated learning implementations."""
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agents = []
    
    @abc.abstractmethod
    def aggregate_parameters(self, agent_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate parameters from multiple agents."""
        pass
    
    @abc.abstractmethod
    def distribute_parameters(self, global_params: Dict[str, Any]) -> None:
        """Distribute global parameters to agents."""
        pass