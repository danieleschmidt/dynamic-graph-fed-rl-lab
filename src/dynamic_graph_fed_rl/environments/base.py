import secrets
"""Base classes for dynamic graph environments."""

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import jax.numpy as jnp
import networkx as nx
import numpy as np


@dataclass
class GraphState:
    """State representation for graph environments."""
    
    node_features: jnp.ndarray  # [num_nodes, node_dim]
    edge_index: jnp.ndarray     # [2, num_edges]
    edge_features: Optional[jnp.ndarray] = None  # [num_edges, edge_dim]
    global_features: Optional[jnp.ndarray] = None  # [global_dim]
    adjacency_matrix: Optional[jnp.ndarray] = None  # [num_nodes, num_nodes]
    timestamp: float = 0.0
    
    def __post_init__(self):
        """Validate state dimensions."""
        if self.edge_features is not None:
            assert self.edge_features.shape[0] == self.edge_index.shape[1], \
                "Edge features must match number of edges"
        
        if self.adjacency_matrix is not None:
            num_nodes = self.node_features.shape[0]
            assert self.adjacency_matrix.shape == (num_nodes, num_nodes), \
                "Adjacency matrix must be square with size num_nodes"
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.node_features.shape[0]
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.edge_index.shape[1]
    
    @property
    def node_dim(self) -> int:
        """Dimension of node features."""
        return self.node_features.shape[1]
    
    @property
    def edge_dim(self) -> int:
        """Dimension of edge features."""
        if self.edge_features is None:
            return 0
        return self.edge_features.shape[1]
    
    def get_subgraph(self, node_indices: List[int], radius: int = 1) -> 'GraphState':
        """Extract subgraph around specified nodes.
        
        Args:
            node_indices: Central nodes for subgraph
            radius: Number of hops to include
            
        Returns:
            Subgraph state
        """
        # Convert to NetworkX for easier subgraph extraction
        G = self.to_networkx()
        
        # Find all nodes within radius
        subgraph_nodes = set(node_indices)
        for _ in range(radius):
            new_nodes = set()
            for node in subgraph_nodes:
                new_nodes.update(G.neighbors(node))
            subgraph_nodes.update(new_nodes)
        
        subgraph_nodes = sorted(list(subgraph_nodes))
        
        # Create node mapping
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}
        
        # Extract subgraph features
        sub_node_features = self.node_features[subgraph_nodes]
        
        # Extract relevant edges
        mask = jnp.isin(self.edge_index[0], jnp.array(subgraph_nodes)) & \
               jnp.isin(self.edge_index[1], jnp.array(subgraph_nodes))
        
        sub_edge_index = self.edge_index[:, mask]
        
        # Remap edge indices
        for i, old_idx in enumerate(sub_edge_index[0]):
            sub_edge_index = sub_edge_index.at[0, i].set(node_mapping[old_idx])
        for i, old_idx in enumerate(sub_edge_index[1]):
            sub_edge_index = sub_edge_index.at[1, i].set(node_mapping[old_idx])
        
        # Extract edge features if available
        sub_edge_features = None
        if self.edge_features is not None:
            sub_edge_features = self.edge_features[mask]
        
        return GraphState(
            node_features=sub_node_features,
            edge_index=sub_edge_index,
            edge_features=sub_edge_features,
            global_features=self.global_features,
            timestamp=self.timestamp,
        )
    
    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph."""
        G = nx.Graph()
        
        # Add nodes with features
        for i in range(self.num_nodes):
            node_attrs = {f"feature_{j}": float(self.node_features[i, j]) 
                         for j in range(self.node_dim)}
            G.add_node(i, **node_attrs)
        
        # Add edges
        edges = [(int(self.edge_index[0, i]), int(self.edge_index[1, i])) 
                 for i in range(self.num_edges)]
        G.add_edges_from(edges)
        
        return G
    
    def compute_graph_metrics(self) -> Dict[str, float]:
        """Compute basic graph topology metrics."""
        G = self.to_networkx()
        
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": nx.density(G),
            "avg_clustering": nx.average_clustering(G),
            "avg_degree": 2 * self.num_edges / self.num_nodes if self.num_nodes > 0 else 0,
            "is_connected": nx.is_connected(G),
        }


@dataclass
class GraphTransition:
    """Transition data for graph environments."""
    
    state: GraphState
    action: Union[jnp.ndarray, Dict[int, jnp.ndarray]]
    reward: Union[float, Dict[int, float]]
    next_state: GraphState
    done: bool
    info: Dict[str, Any]
    topology_changed: bool = False
    
    def __post_init__(self):
        """Validate transition data."""
        if isinstance(self.action, dict) and isinstance(self.reward, dict):
            assert set(self.action.keys()) == set(self.reward.keys()), \
                "Action and reward agent sets must match"


class BaseGraphEnvironment(gym.Env, abc.ABC):
    """Base class for dynamic graph environments."""
    
    def __init__(
        self,
        max_nodes: int = 1000,
        max_edges: int = 5000,
        node_feature_dim: int = 64,
        edge_feature_dim: int = 32,
        global_feature_dim: int = 16,
        time_varying_topology: bool = True,
        edge_failure_rate: float = 0.01,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.time_varying_topology = time_varying_topology
        self.edge_failure_rate = edge_failure_rate
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Environment state
        self.current_state: Optional[GraphState] = None
        self.step_count = 0
        self.episode_count = 0
        
        # Graph topology tracking
        self.base_topology: Optional[nx.Graph] = None
        self.topology_history: List[nx.Graph] = []
        
        # Define observation and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
    
    @abc.abstractmethod
    def _create_observation_space(self) -> gym.Space:
        """Create observation space for the environment."""
        pass
    
    @abc.abstractmethod
    def _create_action_space(self) -> gym.Space:
        """Create action space for the environment."""
        pass
    
    @abc.abstractmethod
    def _initialize_graph(self) -> GraphState:
        """Initialize the graph topology and features."""
        pass
    
    @abc.abstractmethod
    def _update_dynamics(
        self, 
        state: GraphState, 
        actions: Union[jnp.ndarray, Dict[int, jnp.ndarray]]
    ) -> GraphState:
        """Update environment dynamics based on actions."""
        pass
    
    @abc.abstractmethod
    def _compute_reward(
        self,
        state: GraphState,
        actions: Union[jnp.ndarray, Dict[int, jnp.ndarray]],
        next_state: GraphState,
    ) -> Union[float, Dict[int, float]]:
        """Compute reward for the transition."""
        pass
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GraphState, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_count += 1
        self.topology_history = []
        
        # Initialize graph
        self.current_state = self._initialize_graph()
        self.base_topology = self.current_state.to_networkx()
        self.topology_history.append(self.base_topology.copy())
        
        info = {
            "episode": self.episode_count,
            "topology_metrics": self.current_state.compute_graph_metrics(),
        }
        
        return self.current_state, info
    
    def step(
        self, 
        actions: Union[jnp.ndarray, Dict[int, jnp.ndarray]]
    ) -> Tuple[GraphState, Union[float, Dict[int, float]], bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if self.current_state is None:
            raise RuntimeError("Environment must be reset before stepping")
        
        # Update topology if enabled
        topology_changed = False
        if self.time_varying_topology:
            topology_changed = self._maybe_update_topology()
        
        # Apply actions and update dynamics
        next_state = self._update_dynamics(self.current_state, actions)
        
        # Compute rewards
        rewards = self._compute_reward(self.current_state, actions, next_state)
        
        # Check termination conditions
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        terminated = self._check_termination(next_state)
        
        # Update state
        self.current_state = next_state
        
        # Collect info
        info = {
            "step": self.step_count,
            "topology_changed": topology_changed,
            "topology_metrics": next_state.compute_graph_metrics(),
            "episode": self.episode_count,
        }
        
        return next_state, rewards, terminated, truncated, info
    
    def _maybe_update_topology(self) -> bool:
        """Randomly update graph topology."""
        if not self.time_varying_topology:
            return False
        
        # Simple edge failure/recovery model
        current_graph = self.current_state.to_networkx()
        topology_changed = False
        
        # Edge failures
        edges_to_remove = []
        for edge in current_graph.edges():
            if np.secrets.SystemRandom().random() < self.edge_failure_rate:
                edges_to_remove.append(edge)
                topology_changed = True
        
        current_graph.remove_edges_from(edges_to_remove)
        
        # Edge recoveries (restore some failed edges)
        missing_edges = set(self.base_topology.edges()) - set(current_graph.edges())
        for edge in missing_edges:
            if np.secrets.SystemRandom().random() < self.edge_failure_rate * 0.1:  # Lower recovery rate
                current_graph.add_edge(*edge)
                topology_changed = True
        
        if topology_changed:
            # Update current state with new topology
            self.current_state = self._update_state_topology(self.current_state, current_graph)
            self.topology_history.append(current_graph.copy())
        
        return topology_changed
    
    def _update_state_topology(self, state: GraphState, new_graph: nx.Graph) -> GraphState:
        """Update state with new graph topology."""
        # Convert NetworkX back to edge index format
        edges = list(new_graph.edges())
        if edges:
            edge_index = jnp.array([[edge[0] for edge in edges], 
                                  [edge[1] for edge in edges]])
        else:
            edge_index = jnp.zeros((2, 0), dtype=int)
        
        # Update edge features to match new topology
        new_edge_features = None
        if state.edge_features is not None:
            num_new_edges = edge_index.shape[1]
            edge_dim = state.edge_features.shape[1]
            new_edge_features = jnp.zeros((num_new_edges, edge_dim))
        
        return GraphState(
            node_features=state.node_features,
            edge_index=edge_index,
            edge_features=new_edge_features,
            global_features=state.global_features,
            timestamp=state.timestamp + 1,
        )
    
    def _check_termination(self, state: GraphState) -> bool:
        """Check if episode should terminate."""
        # Default: terminate if graph becomes disconnected
        if state.num_edges == 0:
            return True
        
        G = state.to_networkx()
        if not nx.is_connected(G):
            return True
        
        return False
    
    def get_local_view(self, agent_id: int, radius: int = 2) -> GraphState:
        """Get local view of the graph for an agent."""
        if self.current_state is None:
            raise RuntimeError("Environment not initialized")
        
        # For base implementation, agent_id corresponds to node index
        return self.current_state.get_subgraph([agent_id], radius)
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if self.current_state is None:
            return None
        
        if mode == "human":
            try:
                import matplotlib.pyplot as plt
                
                G = self.current_state.to_networkx()
                pos = nx.spring_layout(G)
                
                plt.figure(figsize=(10, 8))
                nx.d# SECURITY WARNING: Potential SQL injection - use parameterized queries
raw(G, pos, with_labels=True, node_color='lightblue', 
                       node_size=500, font_size=8)
                plt.title(f"Graph Environment (Step {self.step_count})")
                plt.axis('off')
                plt.show()
                
            except ImportError:
                print("Matplotlib not available for rendering")
        
        return None
    
    def close(self):
        """Clean up environment resources."""
        self.current_state = None
        self.topology_history = []
    
    def get_topology_stability(self) -> float:
        """Compute topology stability metric."""
        if len(self.topology_history) < 2:
            return 1.0
        
        # Compare last two topologies
        recent_graph = self.topology_history[-1]
        previous_graph = self.topology_history[-2]
        
        # Jaccard similarity of edge sets
        recent_edges = set(recent_graph.edges())
        previous_edges = set(previous_graph.edges())
        
        intersection = len(recent_edges & previous_edges)
        union = len(recent_edges | previous_edges)
        
        return intersection / union if union > 0 else 1.0
    
    def get_environment_metrics(self) -> Dict[str, Any]:
        """Get comprehensive environment metrics."""
        if self.current_state is None:
            return {}
        
        metrics = self.current_state.compute_graph_metrics()
        metrics.update({
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "topology_stability": self.get_topology_stability(),
            "num_topology_changes": len(self.topology_history) - 1,
        })
        
        return metrics