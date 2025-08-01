"""Graph-related test fixtures."""

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple
import pytest


def create_random_graph(
    num_nodes: int = 10,
    num_edges: int = 20, 
    node_features: int = 4,
    edge_features: int = 2,
    seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """Create a random graph for testing."""
    rng = np.random.RandomState(seed)
    
    # Node features
    nodes = jnp.array(rng.normal(0, 1, (num_nodes, node_features)))
    
    # Random edges (no self-loops, no duplicates)
    edge_set = set()
    while len(edge_set) < num_edges:
        src, dst = rng.choice(num_nodes, 2, replace=False)
        edge_set.add((min(src, dst), max(src, dst)))
    
    edges = jnp.array(list(edge_set))
    edge_attrs = jnp.array(rng.normal(0, 1, (num_edges, edge_features)))
    
    return {
        "nodes": nodes,
        "edges": edges,
        "edge_features": edge_attrs,
        "num_nodes": num_nodes,
        "num_edges": num_edges
    }


def create_temporal_graph_sequence(
    base_graph: Dict[str, jnp.ndarray],
    sequence_length: int = 10,
    topology_change_prob: float = 0.1,
    feature_noise_std: float = 0.1,
    seed: int = 42
) -> List[Dict[str, jnp.ndarray]]:
    """Create a sequence of temporal graphs."""
    rng = np.random.RandomState(seed)
    sequence = []
    
    current_graph = {k: v for k, v in base_graph.items()}
    
    for t in range(sequence_length):
        # Add feature noise
        noisy_nodes = current_graph["nodes"] + rng.normal(
            0, feature_noise_std, current_graph["nodes"].shape
        )
        
        noisy_edges = current_graph["edge_features"] + rng.normal(
            0, feature_noise_std, current_graph["edge_features"].shape
        )
        
        # Potential topology changes
        edges = current_graph["edges"]
        if rng.random() < topology_change_prob:
            # Random edge addition/removal
            if rng.random() < 0.5 and len(edges) > 1:
                # Remove random edge
                remove_idx = rng.choice(len(edges))
                edges = jnp.delete(edges, remove_idx, axis=0)
                noisy_edges = jnp.delete(noisy_edges, remove_idx, axis=0)
            else:
                # Add random edge
                num_nodes = current_graph["num_nodes"]
                new_src, new_dst = rng.choice(num_nodes, 2, replace=False)
                new_edge = jnp.array([[new_src, new_dst]])
                new_edge_feat = rng.normal(0, 1, (1, noisy_edges.shape[1]))
                
                edges = jnp.concatenate([edges, new_edge], axis=0)
                noisy_edges = jnp.concatenate([noisy_edges, new_edge_feat], axis=0)
        
        graph_snapshot = {
            "nodes": noisy_nodes,
            "edges": edges,
            "edge_features": noisy_edges,
            "num_nodes": current_graph["num_nodes"],
            "num_edges": len(edges),
            "timestamp": t
        }
        
        sequence.append(graph_snapshot)
        current_graph = graph_snapshot
    
    return sequence


def create_hierarchical_graph(
    levels: int = 3,
    nodes_per_level: List[int] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Create a hierarchical graph structure."""
    if nodes_per_level is None:
        nodes_per_level = [10, 5, 2][:levels]
    
    rng = np.random.RandomState(seed)
    
    total_nodes = sum(nodes_per_level)
    nodes = jnp.array(rng.normal(0, 1, (total_nodes, 4)))
    
    edges = []
    edge_features = []
    
    # Within-level connections
    node_offset = 0
    for level_nodes in nodes_per_level:
        level_indices = list(range(node_offset, node_offset + level_nodes))
        
        # Create some intra-level edges
        for i in range(len(level_indices) - 1):
            edges.append([level_indices[i], level_indices[i + 1]])
            edge_features.append(rng.normal(0, 1, 2))
        
        node_offset += level_nodes
    
    # Between-level connections (hierarchical)
    node_offset = 0
    for level in range(levels - 1):
        current_level_start = node_offset
        current_level_end = node_offset + nodes_per_level[level]
        next_level_start = current_level_end
        next_level_end = next_level_start + nodes_per_level[level + 1]
        
        # Connect each node in next level to some nodes in current level
        for next_node in range(next_level_start, next_level_end):
            num_connections = rng.randint(1, min(3, nodes_per_level[level]))
            connected_nodes = rng.choice(
                range(current_level_start, current_level_end),
                num_connections,
                replace=False
            )
            
            for curr_node in connected_nodes:
                edges.append([curr_node, next_node])
                edge_features.append(rng.normal(0, 1, 2))
        
        node_offset += nodes_per_level[level]
    
    return {
        "nodes": nodes,
        "edges": jnp.array(edges),
        "edge_features": jnp.array(edge_features),
        "levels": levels,
        "nodes_per_level": nodes_per_level,
        "num_nodes": total_nodes,
        "num_edges": len(edges)
    }


@pytest.fixture
def traffic_network_fixture() -> Dict[str, Any]:
    """Traffic network graph fixture."""
    # Create a grid-like traffic network
    grid_size = 5
    num_nodes = grid_size * grid_size
    
    # Node features: [traffic_density, signal_phase, capacity, demand]
    rng = np.random.RandomState(42)
    nodes = jnp.array([
        [
            rng.uniform(0, 1),      # traffic density
            rng.choice([0, 1]),     # signal phase (red/green)
            rng.uniform(50, 200),   # capacity
            rng.uniform(10, 100)    # demand
        ]
        for _ in range(num_nodes)
    ])
    
    # Grid edges (bidirectional)
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            
            # Right neighbor
            if j < grid_size - 1:
                edges.append([node_id, node_id + 1])
                edges.append([node_id + 1, node_id])
            
            # Bottom neighbor
            if i < grid_size - 1:
                edges.append([node_id, node_id + grid_size])
                edges.append([node_id + grid_size, node_id])
    
    # Edge features: [capacity, current_flow, travel_time]
    edge_features = jnp.array([
        [
            rng.uniform(20, 100),   # capacity
            rng.uniform(0, 50),     # current flow
            rng.uniform(1, 10)      # travel time
        ]
        for _ in range(len(edges))
    ])
    
    return {
        "nodes": nodes,
        "edges": jnp.array(edges),
        "edge_features": edge_features,
        "grid_size": grid_size,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "scenario": "traffic_network"
    }


@pytest.fixture
def power_grid_fixture() -> Dict[str, Any]:
    """Power grid graph fixture."""
    # IEEE 14-bus simplified system
    num_buses = 14
    
    # Bus features: [voltage, frequency, generation, load]
    rng = np.random.RandomState(123)
    nodes = jnp.array([
        [
            rng.uniform(0.95, 1.05),  # voltage (p.u.)
            rng.uniform(59.8, 60.2),  # frequency (Hz)
            rng.uniform(0, 200),      # generation (MW)
            rng.uniform(10, 100)      # load (MW)
        ]
        for _ in range(num_buses)
    ])
    
    # Transmission lines (based on IEEE 14-bus topology)
    transmission_lines = [
        (0, 1), (0, 4), (1, 2), (1, 3), (1, 4),
        (2, 3), (3, 6), (4, 5), (4, 7), (4, 9),
        (5, 10), (6, 11), (6, 12), (7, 8), (8, 13),
        (9, 10), (9, 13), (10, 11), (11, 12), (12, 13)
    ]
    
    edges = jnp.array(transmission_lines)
    
    # Edge features: [resistance, reactance, capacity, power_flow]
    edge_features = jnp.array([
        [
            rng.uniform(0.01, 0.1),   # resistance (p.u.)
            rng.uniform(0.05, 0.5),   # reactance (p.u.)
            rng.uniform(100, 500),    # capacity (MVA)
            rng.uniform(-50, 200)     # power flow (MW)
        ]
        for _ in range(len(transmission_lines))
    ])
    
    return {
        "nodes": nodes,
        "edges": edges,
        "edge_features": edge_features,
        "num_nodes": num_buses,
        "num_edges": len(transmission_lines),
        "scenario": "power_grid"
    }