"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from typing import Generator, Dict, Any


@pytest.fixture(scope="session")
def jax_config() -> Generator[None, None, None]:
    """Configure JAX for testing."""
    # Enable debugging
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
    
    # Use CPU for consistent testing
    jax.config.update("jax_platform_name", "cpu")
    
    yield
    
    # Reset after tests
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_debug_infs", False)


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def rng_key(random_seed: int) -> jax.random.PRNGKey:
    """JAX random key for testing."""
    return jax.random.PRNGKey(random_seed)


@pytest.fixture
def small_graph() -> Dict[str, Any]:
    """Small test graph for unit tests."""
    return {
        "nodes": jnp.array([[1.0, 0.5], [0.8, 0.3], [0.2, 0.9]]),  # 3 nodes, 2 features
        "edges": jnp.array([[0, 1], [1, 2], [2, 0]]),  # 3 edges
        "edge_features": jnp.array([[1.0], [0.5], [0.8]]),  # 3 edges, 1 feature
    }


@pytest.fixture
def medium_graph() -> Dict[str, Any]:
    """Medium test graph for integration tests."""
    num_nodes = 50
    num_edges = 100
    
    # Random node features
    nodes = jax.random.normal(jax.random.PRNGKey(42), (num_nodes, 4))
    
    # Random edges (ensure no self-loops)
    edges = []
    rng = np.random.RandomState(42)
    for _ in range(num_edges):
        src, dst = rng.choice(num_nodes, 2, replace=False)
        edges.append([src, dst])
    
    edges = jnp.array(edges)
    edge_features = jax.random.normal(jax.random.PRNGKey(43), (num_edges, 2))
    
    return {
        "nodes": nodes,
        "edges": edges, 
        "edge_features": edge_features,
    }


@pytest.fixture
def mock_environment_config() -> Dict[str, Any]:
    """Configuration for mock environments."""
    return {
        "num_agents": 5,
        "max_steps": 100,
        "reward_scale": 1.0,
        "observation_dim": 10,
        "action_dim": 2,
    }


@pytest.fixture
def federation_config() -> Dict[str, Any]:
    """Configuration for federated learning."""
    return {
        "num_agents": 10,
        "aggregation_interval": 50,
        "communication_rounds": 20,
        "gossip_probability": 0.3,
    }