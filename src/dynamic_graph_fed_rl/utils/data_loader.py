"""Data loading and preprocessing utilities for graph datasets."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from ..environments.base import GraphState


class GraphDataset:
    """Dataset class for loading graph sequences."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        sequence_length: int = 10,
        overlap: int = 5,
        normalize_features: bool = True,
        cache_size: int = 1000,
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.normalize_features = normalize_features
        self.cache_size = cache_size
        
        # Data storage
        self.graph_sequences = []
        self.metadata = {}
        self.feature_stats = {}
        self.cache = {}
        
        # Load data
        self._load_dataset()
        
        if self.normalize_features:
            self._compute_feature_statistics()
    
    def _load_dataset(self) -> None:
        """Load graph dataset from files."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")
        
        if self.data_path.is_file():
            # Single file
            self._load_single_file(self.data_path)
        else:
            # Directory with multiple files
            self._load_directory(self.data_path)
        
        print(f"Loaded {len(self.graph_sequences)} graph sequences")
    
    def _load_single_file(self, file_path: Path) -> None:
        """Load graphs from a single file."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif suffix == '.npz':
            data = dict(np.load(file_path))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Parse data structure
        if isinstance(data, dict):
            if 'sequences' in data:
                self.graph_sequences = data['sequences']
            elif 'graphs' in data:
                # Convert single graphs to sequences
                graphs = data['graphs']
                self.graph_sequences = [graphs[i:i+self.sequence_length] 
                                       for i in range(0, len(graphs) - self.sequence_length + 1, 
                                                     self.sequence_length - self.overlap)]
            
            self.metadata = data.get('metadata', {})
        elif isinstance(data, list):
            # List of graphs
            self.graph_sequences = [data[i:i+self.sequence_length] 
                                   for i in range(0, len(data) - self.sequence_length + 1,
                                                 self.sequence_length - self.overlap)]
    
    def _load_directory(self, dir_path: Path) -> None:
        """Load graphs from directory of files."""
        graph_files = list(dir_path.glob("*.pkl")) + list(dir_path.glob("*.json"))
        graph_files.sort()
        
        all_graphs = []
        for file_path in graph_files:
            try:
                self._load_single_file(file_path)
                # Flatten any sequences loaded
                for seq in self.graph_sequences:
                    all_graphs.extend(seq)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        # Create sequences from all graphs
        if all_graphs:
            self.graph_sequences = [all_graphs[i:i+self.sequence_length] 
                                   for i in range(0, len(all_graphs) - self.sequence_length + 1,
                                                 self.sequence_length - self.overlap)]
    
    def _compute_feature_statistics(self) -> None:
        """Compute statistics for feature normalization."""
        all_node_features = []
        all_edge_features = []
        all_global_features = []
        
        for sequence in self.graph_sequences:
            for graph in sequence:
                if isinstance(graph, GraphState):
                    if graph.node_features is not None:
                        all_node_features.append(graph.node_features)
                    if graph.edge_features is not None:
                        all_edge_features.append(graph.edge_features)
                    if graph.global_features is not None:
                        all_global_features.append(graph.global_features)
                elif isinstance(graph, dict):
                    if 'node_features' in graph:
                        all_node_features.append(jnp.array(graph['node_features']))
                    if 'edge_features' in graph:
                        all_edge_features.append(jnp.array(graph['edge_features']))
                    if 'global_features' in graph:
                        all_global_features.append(jnp.array(graph['global_features']))
        
        # Compute statistics
        if all_node_features:
            all_node_features = jnp.concatenate(all_node_features, axis=0)
            self.feature_stats['node_mean'] = jnp.mean(all_node_features, axis=0)
            self.feature_stats['node_std'] = jnp.std(all_node_features, axis=0) + 1e-8
        
        if all_edge_features:
            all_edge_features = jnp.concatenate(all_edge_features, axis=0)
            self.feature_stats['edge_mean'] = jnp.mean(all_edge_features, axis=0)
            self.feature_stats['edge_std'] = jnp.std(all_edge_features, axis=0) + 1e-8
        
        if all_global_features:
            all_global_features = jnp.stack(all_global_features, axis=0)
            self.feature_stats['global_mean'] = jnp.mean(all_global_features, axis=0)
            self.feature_stats['global_std'] = jnp.std(all_global_features, axis=0) + 1e-8
    
    def normalize_graph(self, graph: GraphState) -> GraphState:
        """Normalize graph features using computed statistics."""
        if not self.normalize_features or not self.feature_stats:
            return graph
        
        normalized_graph = GraphState(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_features=graph.edge_features,
            global_features=graph.global_features,
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            timestamp=graph.timestamp,
        )
        
        # Normalize node features
        if (graph.node_features is not None and 
            'node_mean' in self.feature_stats):
            normalized_graph.node_features = (
                (graph.node_features - self.feature_stats['node_mean']) / 
                self.feature_stats['node_std']
            )
        
        # Normalize edge features
        if (graph.edge_features is not None and 
            'edge_mean' in self.feature_stats):
            normalized_graph.edge_features = (
                (graph.edge_features - self.feature_stats['edge_mean']) / 
                self.feature_stats['edge_std']
            )
        
        # Normalize global features
        if (graph.global_features is not None and 
            'global_mean' in self.feature_stats):
            normalized_graph.global_features = (
                (graph.global_features - self.feature_stats['global_mean']) / 
                self.feature_stats['global_std']
            )
        
        return normalized_graph
    
    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.graph_sequences)
    
    def __getitem__(self, idx: int) -> List[GraphState]:
        """Get sequence by index."""
        if idx >= len(self.graph_sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.graph_sequences)}")
        
        # Check cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Load and process sequence
        sequence = self.graph_sequences[idx]
        processed_sequence = []
        
        for graph_data in sequence:
            if isinstance(graph_data, GraphState):
                graph = graph_data
            elif isinstance(graph_data, dict):
                graph = self._dict_to_graph_state(graph_data)
            else:
                raise ValueError(f"Unsupported graph data type: {type(graph_data)}")
            
            # Normalize if enabled
            if self.normalize_features:
                graph = self.normalize_graph(graph)
            
            processed_sequence.append(graph)
        
        # Cache if under limit
        if len(self.cache) < self.cache_size:
            self.cache[idx] = processed_sequence
        
        return processed_sequence
    
    def _dict_to_graph_state(self, graph_dict: Dict[str, Any]) -> GraphState:
        """Convert dictionary to GraphState object."""
        return GraphState(
            node_features=jnp.array(graph_dict.get('node_features', [])),
            edge_index=jnp.array(graph_dict.get('edge_index', [[], []])),
            edge_features=jnp.array(graph_dict.get('edge_features', [])) if 'edge_features' in graph_dict else None,
            global_features=jnp.array(graph_dict.get('global_features', [])) if 'global_features' in graph_dict else None,
            num_nodes=graph_dict.get('num_nodes', 0),
            num_edges=graph_dict.get('num_edges', 0),
            timestamp=graph_dict.get('timestamp', 0.0),
        )
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> List[List[GraphState]]:
        """Get a batch of sequences."""
        if shuffle:
            indices = np.random.choice(len(self.graph_sequences), size=batch_size, replace=True)
        else:
            indices = np.arange(min(batch_size, len(self.graph_sequences)))
        
        return [self[idx] for idx in indices]
    
    def get_data_iterator(self, batch_size: int, shuffle: bool = True) -> Iterator[List[List[GraphState]]]:
        """Get iterator over batches."""
        while True:
            yield self.get_batch(batch_size, shuffle)


class GraphDataGenerator:
    """Generate synthetic graph datasets for testing and development."""
    
    def __init__(
        self,
        num_graphs: int = 1000,
        min_nodes: int = 10,
        max_nodes: int = 100,
        node_feature_dim: int = 8,
        edge_feature_dim: int = 4,
        global_feature_dim: int = 3,
        topology_types: List[str] = None,
        dynamics_types: List[str] = None,
        seed: int = 42,
    ):
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.topology_types = topology_types or ['random', 'scale_free', 'small_world']
        self.dynamics_types = dynamics_types or ['random_walk', 'diffusion', 'epidemic']
        
        self.rng = np.random.RandomState(seed)
    
    def generate_graph_sequence(
        self,
        sequence_length: int = 10,
        topology_changes: bool = True,
    ) -> List[GraphState]:
        """Generate a sequence of related graphs."""
        # Initial graph
        num_nodes = self.rng.randint(self.min_nodes, self.max_nodes + 1)
        topology_type = self.rng.choice(self.topology_types)
        dynamics_type = self.rng.choice(self.dynamics_types)
        
        # Create base topology
        G = self._create_topology(num_nodes, topology_type)
        
        sequence = []
        for t in range(sequence_length):
            # Evolve topology if enabled
            if topology_changes and t > 0 and self.rng.random() < 0.1:
                G = self._evolve_topology(G)
            
            # Generate features
            graph_state = self._create_graph_state(G, dynamics_type, t)
            sequence.append(graph_state)
        
        return sequence
    
    def _create_topology(self, num_nodes: int, topology_type: str) -> nx.Graph:
        """Create graph topology."""
        if topology_type == 'random':
            p = 2.0 / num_nodes  # Expected degree ≈ 2
            G = nx.erdos_renyi_graph(num_nodes, p, seed=self.rng.randint(10000))
        elif topology_type == 'scale_free':
            m = max(1, num_nodes // 10)
            G = nx.barabasi_albert_graph(num_nodes, m, seed=self.rng.randint(10000))
        elif topology_type == 'small_world':
            k = max(2, num_nodes // 5)
            if k >= num_nodes:
                k = num_nodes - 1
            if k % 2 == 1:
                k -= 1  # k must be even for watts_strogatz_graph
            p = 0.3
            G = nx.watts_strogatz_graph(num_nodes, k, p, seed=self.rng.randint(10000))
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")
        
        # Ensure connectivity
        if not nx.is_connected(G):
            # Add edges to make connected
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = self.rng.choice(list(components[i]))
                node2 = self.rng.choice(list(components[i + 1]))
                G.add_edge(node1, node2)
        
        return G
    
    def _evolve_topology(self, G: nx.Graph) -> nx.Graph:
        """Evolve graph topology slightly."""
        H = G.copy()
        num_nodes = len(H.nodes())
        
        # Random edge additions/removals
        if self.rng.random() < 0.3 and len(H.edges()) > num_nodes - 1:
            # Remove random edge (but keep connected)
            edge = self.rng.choice(list(H.edges()))
            H.remove_edge(*edge)
            
            # Check if still connected
            if not nx.is_connected(H):
                H.add_edge(*edge)  # Restore edge
        
        if self.rng.random() < 0.3:
            # Add random edge
            possible_edges = [(u, v) for u in range(num_nodes) 
                             for v in range(u + 1, num_nodes) 
                             if not H.has_edge(u, v)]
            if possible_edges:
                edge = self.rng.choice(possible_edges)
                H.add_edge(*edge)
        
        return H
    
    def _create_graph_state(
        self,
        G: nx.Graph,
        dynamics_type: str,
        timestep: int,
    ) -> GraphState:
        """Create GraphState with synthetic features."""
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        
        # Node features
        if dynamics_type == 'random_walk':
            node_features = self._random_walk_features(num_nodes, timestep)
        elif dynamics_type == 'diffusion':
            node_features = self._diffusion_features(G, timestep)
        elif dynamics_type == 'epidemic':
            node_features = self._epidemic_features(G, timestep)
        else:
            node_features = self.rng.randn(num_nodes, self.node_feature_dim)
        
        # Edge features
        edges = list(G.edges())
        if edges:
            edge_features = self.rng.randn(num_edges, self.edge_feature_dim)
            edge_index = jnp.array([[e[0] for e in edges], [e[1] for e in edges]])
        else:
            edge_features = jnp.zeros((0, self.edge_feature_dim))
            edge_index = jnp.zeros((2, 0), dtype=int)
        
        # Global features
        global_features = jnp.array([
            timestep / 10.0,  # Normalized time
            len(G.edges()) / (num_nodes * (num_nodes - 1) / 2),  # Density
            nx.average_clustering(G) if num_nodes > 2 else 0.0,  # Clustering
        ])
        
        return GraphState(
            node_features=jnp.array(node_features),
            edge_index=edge_index,
            edge_features=jnp.array(edge_features),
            global_features=global_features,
            num_nodes=num_nodes,
            num_edges=num_edges,
            timestamp=float(timestep),
        )
    
    def _random_walk_features(self, num_nodes: int, timestep: int) -> np.ndarray:
        """Generate random walk-based features."""
        features = np.zeros((num_nodes, self.node_feature_dim))
        
        for i in range(num_nodes):
            # Position in feature space evolves as random walk
            base_pos = self.rng.randn(self.node_feature_dim // 2)
            noise = self.rng.randn(self.node_feature_dim // 2) * 0.1
            
            features[i, :self.node_feature_dim // 2] = base_pos + noise * timestep
            features[i, self.node_feature_dim // 2:] = self.rng.randn(self.node_feature_dim // 2)
        
        return features
    
    def _diffusion_features(self, G: nx.Graph, timestep: int) -> np.ndarray:
        """Generate diffusion-based features."""
        num_nodes = len(G.nodes())
        features = np.zeros((num_nodes, self.node_feature_dim))
        
        # Simulate diffusion process
        A = nx.adjacency_matrix(G).toarray()
        D = np.diag(np.sum(A, axis=1))
        L = D - A  # Laplacian matrix
        
        # Initial condition
        if not hasattr(self, '_diffusion_state'):
            self._diffusion_state = self.rng.randn(num_nodes, self.node_feature_dim // 2)
        
        # Diffusion step
        if timestep > 0:
            self._diffusion_state = self._diffusion_state - 0.1 * L @ self._diffusion_state
        
        features[:, :self.node_feature_dim // 2] = self._diffusion_state
        features[:, self.node_feature_dim // 2:] = self.rng.randn(num_nodes, self.node_feature_dim // 2)
        
        return features
    
    def _epidemic_features(self, G: nx.Graph, timestep: int) -> np.ndarray:
        """Generate epidemic spreading features."""
        num_nodes = len(G.nodes())
        features = np.zeros((num_nodes, self.node_feature_dim))
        
        # SIR model states
        if not hasattr(self, '_epidemic_state'):
            self._epidemic_state = np.zeros(num_nodes)  # 0: S, 1: I, 2: R
            # Initial infection
            patient_zero = self.rng.randint(num_nodes)
            self._epidemic_state[patient_zero] = 1
        
        # Epidemic dynamics
        if timestep > 0:
            new_state = self._epidemic_state.copy()
            for node in G.nodes():
                if self._epidemic_state[node] == 1:  # Infected
                    # Recovery
                    if self.rng.random() < 0.1:
                        new_state[node] = 2
                    # Spread to neighbors
                    for neighbor in G.neighbors(node):
                        if self._epidemic_state[neighbor] == 0 and self.rng.random() < 0.2:
                            new_state[neighbor] = 1
            self._epidemic_state = new_state
        
        # Features based on epidemic state
        for i in range(num_nodes):
            features[i, 0] = self._epidemic_state[i]
            features[i, 1] = sum(1 for neighbor in G.neighbors(i) if self._epidemic_state[neighbor] == 1)
            features[i, 2:] = self.rng.randn(self.node_feature_dim - 2)
        
        return features
    
    def generate_dataset(self, output_path: Union[str, Path]) -> None:
        """Generate and save complete dataset."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sequences = []
        for i in range(self.num_graphs):
            sequence = self.generate_graph_sequence()
            sequences.append(sequence)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.num_graphs} sequences")
        
        # Save dataset
        dataset = {
            'sequences': sequences,
            'metadata': {
                'num_sequences': len(sequences),
                'sequence_length': len(sequences[0]) if sequences else 0,
                'node_feature_dim': self.node_feature_dim,
                'edge_feature_dim': self.edge_feature_dim,
                'global_feature_dim': self.global_feature_dim,
                'topology_types': self.topology_types,
                'dynamics_types': self.dynamics_types,
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {output_path}")


def load_real_graph_dataset(
    dataset_name: str,
    data_dir: Union[str, Path] = "data/",
) -> GraphDataset:
    """Load real-world graph dataset."""
    data_dir = Path(data_dir)
    
    if dataset_name == "traffic_los_angeles":
        return _load_traffic_dataset(data_dir / "traffic_la")
    elif dataset_name == "power_grid_texas":
        return _load_power_grid_dataset(data_dir / "power_texas")
    elif dataset_name == "social_network":
        return _load_social_network_dataset(data_dir / "social")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_traffic_dataset(data_path: Path) -> GraphDataset:
    """Load traffic network dataset."""
    # Placeholder for real traffic data loading
    print(f"Loading traffic dataset from {data_path}")
    
    # For now, generate synthetic traffic data
    generator = GraphDataGenerator(
        num_graphs=500,
        min_nodes=20,
        max_nodes=200,
        node_feature_dim=8,
        edge_feature_dim=4,
        topology_types=['small_world'],  # Traffic networks are small-world
        dynamics_types=['diffusion'],    # Traffic flow is diffusion-like
    )
    
    # Create temporary dataset
    temp_path = data_path / "synthetic_traffic.pkl"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    generator.generate_dataset(temp_path)
    
    return GraphDataset(temp_path)


def _load_power_grid_dataset(data_path: Path) -> GraphDataset:
    """Load power grid dataset."""
    print(f"Loading power grid dataset from {data_path}")
    
    # Generate synthetic power grid data
    generator = GraphDataGenerator(
        num_graphs=300,
        min_nodes=50,
        max_nodes=500,
        node_feature_dim=6,
        edge_feature_dim=3,
        topology_types=['scale_free'],   # Power grids are scale-free
        dynamics_types=['diffusion'],
    )
    
    temp_path = data_path / "synthetic_power.pkl"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    generator.generate_dataset(temp_path)
    
    return GraphDataset(temp_path)


def _load_social_network_dataset(data_path: Path) -> GraphDataset:
    """Load social network dataset."""
    print(f"Loading social network dataset from {data_path}")
    
    # Generate synthetic social network data
    generator = GraphDataGenerator(
        num_graphs=200,
        min_nodes=100,
        max_nodes=1000,
        node_feature_dim=10,
        edge_feature_dim=2,
        topology_types=['scale_free', 'small_world'],
        dynamics_types=['epidemic', 'diffusion'],
    )
    
    temp_path = data_path / "synthetic_social.pkl"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    generator.generate_dataset(temp_path)
    
    return GraphDataset(temp_path)