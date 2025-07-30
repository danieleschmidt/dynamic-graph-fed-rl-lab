"""Pytest configuration and shared fixtures."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@pytest.fixture
def jax_key():
    \"\"\"Random key for JAX operations.\"\"\"
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_graph():
    \"\"\"Sample graph for testing.\"\"\"
    num_nodes = 10
    num_edges = 20
    
    return {
        'nodes': jnp.ones((num_nodes, 4)),  # 4D node features
        'edges': jnp.array([[0, 1, 2], [1, 2, 3]]),  # Edge indices
        'edge_attr': jnp.ones((3, 2)),  # 2D edge features
        'global_attr': jnp.array([1.0, 2.0]),
        'timestamp': 0.0,
    }


@pytest.fixture
def mock_config():
    \"\"\"Mock configuration for testing.\"\"\"
    return {
        'algorithm': 'GraphTD3',
        'federation': {
            'protocol': 'async_gossip',
            'num_agents': 4,
            'aggregation_interval': 100,
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 3e-4,
            'buffer_size': 10000,
        },
        'environment': {
            'scenario': 'traffic_network',
            'num_nodes': 50,
            'max_steps': 200,
        }
    }