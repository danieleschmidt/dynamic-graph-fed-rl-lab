"""Traffic network environment for dynamic graph federated RL."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import jax.numpy as jnp
import networkx as nx
import numpy as np

from .base import BaseGraphEnvironment, GraphState


@dataclass
class IntersectionNode:
    """Traffic intersection node representation."""
    
    node_id: int
    position: Tuple[float, float]
    traffic_signal_state: int  # 0: red, 1: yellow, 2: green
    queue_length: float
    flow_capacity: float
    signal_timing: List[float]  # [red_time, yellow_time, green_time]
    
    def update_signal(self, action: int) -> None:
        """Update traffic signal based on action."""
        if action in [0, 1, 2]:  # Valid signal states
            self.traffic_signal_state = action
    
    def compute_delay(self) -> float:
        """Compute traffic delay at intersection."""
        # Simple delay model based on queue length and signal state
        base_delay = self.queue_length / max(self.flow_capacity, 1e-6)
        
        # Penalty for red light
        if self.traffic_signal_state == 0:  # Red
            base_delay *= 2.0
        elif self.traffic_signal_state == 1:  # Yellow
            base_delay *= 1.5
        
        return base_delay


@dataclass
class TrafficState(GraphState):
    """Extended graph state for traffic networks."""
    
    intersections: List[IntersectionNode]
    traffic_flow: jnp.ndarray  # [num_edges] - vehicles per minute
    congestion_level: jnp.ndarray  # [num_edges] - 0.0 to 1.0
    incident_locations: List[int]  # Node indices with incidents
    weather_condition: float  # 0.0 (clear) to 1.0 (severe)
    time_of_day: float  # 0.0 to 24.0 hours
    
    def get_total_delay(self) -> float:
        """Compute total network delay."""
        return sum(intersection.compute_delay() for intersection in self.intersections)
    
    def get_average_congestion(self) -> float:
        """Compute average network congestion."""
        return float(jnp.mean(self.congestion_level))
    
    def get_throughput(self) -> float:
        """Compute network throughput (vehicles/minute)."""
        return float(jnp.sum(self.traffic_flow))


class TrafficNetworkEnv(BaseGraphEnvironment):
    """Dynamic traffic network environment."""
    
    def __init__(
        self,
        num_intersections: int = 100,
        scenario: str = "city_grid",
        rush_hour_dynamics: bool = True,
        incident_probability: float = 0.001,
        weather_enabled: bool = True,
        signal_coordination: bool = False,
        **kwargs
    ):
        self.num_intersections = num_intersections
        self.scenario = scenario
        self.rush_hour_dynamics = rush_hour_dynamics
        self.incident_probability = incident_probability
        self.weather_enabled = weather_enabled
        self.signal_coordination = signal_coordination
        
        # Traffic-specific parameters
        self.max_flow_capacity = 60.0  # vehicles per minute
        self.base_flow_rate = 20.0
        self.congestion_threshold = 0.8
        
        super().__init__(
            max_nodes=num_intersections,
            node_feature_dim=8,  # [signal_state, queue_length, capacity, delay, x, y, incident, weather_impact]
            edge_feature_dim=4,  # [flow, congestion, capacity, travel_time]
            global_feature_dim=3,  # [time_of_day, weather, total_incidents]
            **kwargs
        )
    
    def _create_observation_space(self) -> gym.Space:
        """Create observation space for traffic network."""
        return gym.spaces.Dict({
            "node_features": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_nodes, self.node_feature_dim),
                dtype=np.float32
            ),
            "edge_index": gym.spaces.Box(
                low=0, high=self.max_nodes-1,
                shape=(2, self.max_edges),
                dtype=np.int32
            ),
            "edge_features": gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.max_edges, self.edge_feature_dim),
                dtype=np.float32
            ),
            "global_features": gym.spaces.Box(
                low=0.0, high=24.0,
                shape=(self.global_feature_dim,),
                dtype=np.float32
            ),
        })
    
    def _create_action_space(self) -> gym.Space:
        """Create action space for traffic signals."""
        # Each intersection can set signal to red(0), yellow(1), or green(2)
        return gym.spaces.MultiDiscrete([3] * self.num_intersections)
    
    def _initialize_graph(self) -> TrafficState:
        """Initialize traffic network topology and state."""
        # Create base topology based on scenario
        if self.scenario == "city_grid":
            G = self._create_grid_topology()
        elif self.scenario == "highway_network":
            G = self._create_highway_topology()
        elif self.scenario == "random_network":
            G = self._create_random_topology()
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        
        self.base_topology = G
        
        # Initialize intersections
        intersections = []
        for i in range(self.num_intersections):
            if i in G.nodes():
                pos = G.nodes[i].get('pos', (np.random.random(), np.random.random()))
                intersection = IntersectionNode(
                    node_id=i,
                    position=pos,
                    traffic_signal_state=np.random.randint(0, 3),
                    queue_length=np.random.uniform(0, 10),
                    flow_capacity=np.random.uniform(30, self.max_flow_capacity),
                    signal_timing=[30.0, 5.0, 25.0],  # Default timing
                )
                intersections.append(intersection)
        
        # Create node features
        node_features = self._compute_node_features(intersections, G)
        
        # Create edge features
        edges = list(G.edges())
        edge_index = jnp.array([[e[0] for e in edges], [e[1] for e in edges]]) if edges else jnp.zeros((2, 0), dtype=int)
        edge_features = self._compute_edge_features(edges, G)
        
        # Initialize traffic state
        traffic_flow = jnp.array([self.base_flow_rate + np.random.normal(0, 5) 
                                 for _ in range(len(edges))])
        traffic_flow = jnp.clip(traffic_flow, 0, 100)
        
        congestion_level = traffic_flow / self.max_flow_capacity
        congestion_level = jnp.clip(congestion_level, 0, 1)
        
        # Global features: [time_of_day, weather, total_incidents]
        global_features = jnp.array([
            8.0,  # Start at 8 AM
            0.0 if not self.weather_enabled else np.random.uniform(0, 0.3),
            0.0,  # No initial incidents
        ])
        
        return TrafficState(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=global_features,
            intersections=intersections,
            traffic_flow=traffic_flow,
            congestion_level=congestion_level,
            incident_locations=[],
            weather_condition=float(global_features[1]),
            time_of_day=float(global_features[0]),
            timestamp=0.0,
        )
    
    def _create_grid_topology(self) -> nx.Graph:
        """Create grid-based city topology."""
        grid_size = int(np.sqrt(self.num_intersections))
        G = nx.grid_2d_graph(grid_size, grid_size)
        
        # Convert to integer node labels and add positions
        H = nx.Graph()
        for i, (x, y) in enumerate(G.nodes()):
            if i >= self.num_intersections:
                break
            H.add_node(i, pos=(float(x), float(y)))
        
        # Add edges based on grid structure
        for (x1, y1), (x2, y2) in G.edges():
            node1 = x1 * grid_size + y1
            node2 = x2 * grid_size + y2
            if node1 < self.num_intersections and node2 < self.num_intersections:
                H.add_edge(node1, node2)
        
        return H
    
    def _create_highway_topology(self) -> nx.Graph:
        """Create highway network topology."""
        G = nx.Graph()
        
        # Main highway (linear)
        main_highway_nodes = min(self.num_intersections // 3, 50)
        for i in range(main_highway_nodes - 1):
            G.add_edge(i, i + 1)
            G.nodes[i]['pos'] = (float(i), 0.0)
        
        # Secondary roads (branches)
        current_node = main_highway_nodes
        for i in range(0, main_highway_nodes, 5):  # Branch every 5 nodes
            if current_node >= self.num_intersections:
                break
            
            # Create branch
            branch_length = np.random.randint(3, 8)
            G.add_edge(i, current_node)
            G.nodes[current_node]['pos'] = (float(i), 1.0)
            
            for j in range(1, branch_length):
                if current_node + j >= self.num_intersections:
                    break
                G.add_edge(current_node + j - 1, current_node + j)
                G.nodes[current_node + j]['pos'] = (float(i), float(j + 1))
            
            current_node += branch_length
        
        return G
    
    def _create_random_topology(self) -> nx.Graph:
        """Create random network topology."""
        # Erdős–Rényi random graph
        p = 4.0 / self.num_intersections  # Expected degree ≈ 4
        G = nx.erdos_renyi_graph(self.num_intersections, p)
        
        # Add random positions
        for i in range(self.num_intersections):
            G.nodes[i]['pos'] = (np.random.uniform(0, 10), np.random.uniform(0, 10))
        
        return G
    
    def _compute_node_features(self, intersections: List[IntersectionNode], G: nx.Graph) -> jnp.ndarray:
        """Compute node feature matrix."""
        features = []
        
        for i in range(self.num_intersections):
            if i < len(intersections):
                intersection = intersections[i]
                incident_flag = 1.0 if i in getattr(self, '_current_incidents', []) else 0.0
                weather_impact = getattr(self, '_current_weather', 0.0)
                
                node_feat = [
                    float(intersection.traffic_signal_state),
                    intersection.queue_length,
                    intersection.flow_capacity,
                    intersection.compute_delay(),
                    intersection.position[0],
                    intersection.position[1],
                    incident_flag,
                    weather_impact,
                ]
            else:
                # Padding for unused nodes
                node_feat = [0.0] * self.node_feature_dim
            
            features.append(node_feat)
        
        return jnp.array(features)
    
    def _compute_edge_features(self, edges: List[Tuple[int, int]], G: nx.Graph) -> jnp.ndarray:
        """Compute edge feature matrix."""
        if not edges:
            return jnp.zeros((0, self.edge_feature_dim))
        
        features = []
        for edge in edges:
            # Distance between intersections
            if 'pos' in G.nodes[edge[0]] and 'pos' in G.nodes[edge[1]]:
                pos1 = G.nodes[edge[0]]['pos']
                pos2 = G.nodes[edge[1]]['pos']
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            else:
                distance = 1.0
            
            # Initial edge features: [flow, congestion, capacity, travel_time]
            edge_feat = [
                self.base_flow_rate,  # Initial flow
                0.3,  # Initial congestion
                self.max_flow_capacity,  # Capacity
                distance * 2.0,  # Travel time (proportional to distance)
            ]
            features.append(edge_feat)
        
        return jnp.array(features)
    
    def _update_dynamics(
        self, 
        state: TrafficState, 
        actions: jnp.ndarray
    ) -> TrafficState:
        """Update traffic network dynamics."""
        # Update time
        new_time = state.time_of_day + 0.1  # 6-minute timesteps
        if new_time >= 24.0:
            new_time -= 24.0
        
        # Update weather
        new_weather = state.weather_condition
        if self.weather_enabled:
            # Simple weather dynamics
            new_weather += np.random.normal(0, 0.05)
            new_weather = np.clip(new_weather, 0, 1)
        
        # Update traffic signals
        new_intersections = []
        for i, intersection in enumerate(state.intersections):
            new_intersection = IntersectionNode(
                node_id=intersection.node_id,
                position=intersection.position,
                traffic_signal_state=int(actions[i]) if i < len(actions) else intersection.traffic_signal_state,
                queue_length=intersection.queue_length,
                flow_capacity=intersection.flow_capacity,
                signal_timing=intersection.signal_timing,
            )
            new_intersections.append(new_intersection)
        
        # Update traffic flow based on signals and time of day
        new_traffic_flow = self._compute_traffic_flow(state, new_intersections, new_time)
        
        # Update queue lengths based on flow and signals
        self._update_queue_lengths(new_intersections, new_traffic_flow, state)
        
        # Update congestion
        edge_capacities = state.edge_features[:, 2] if state.edge_features is not None else jnp.ones(len(new_traffic_flow)) * self.max_flow_capacity
        new_congestion = new_traffic_flow / jnp.maximum(edge_capacities, 1e-6)
        new_congestion = jnp.clip(new_congestion, 0, 1)
        
        # Random incidents
        new_incidents = list(state.incident_locations)
        if np.random.random() < self.incident_probability:
            incident_node = np.random.randint(0, self.num_intersections)
            if incident_node not in new_incidents:
                new_incidents.append(incident_node)
        
        # Resolve some incidents
        if new_incidents and np.random.random() < 0.1:  # 10% chance to resolve
            resolved_incident = np.random.choice(new_incidents)
            new_incidents.remove(resolved_incident)
        
        # Update node features
        self._current_incidents = new_incidents
        self._current_weather = new_weather
        new_node_features = self._compute_node_features(new_intersections, self.base_topology)
        
        # Update edge features
        if state.edge_features is not None:
            new_edge_features = state.edge_features.at[:, 0].set(new_traffic_flow)
            new_edge_features = new_edge_features.at[:, 1].set(new_congestion)
            # Travel time increases with congestion
            base_travel_times = state.edge_features[:, 3]
            congestion_multiplier = 1.0 + 2.0 * new_congestion
            new_travel_times = base_travel_times * congestion_multiplier
            new_edge_features = new_edge_features.at[:, 3].set(new_travel_times)
        else:
            new_edge_features = None
        
        # Update global features
        new_global_features = jnp.array([new_time, new_weather, float(len(new_incidents))])
        
        return TrafficState(
            node_features=new_node_features,
            edge_index=state.edge_index,
            edge_features=new_edge_features,
            global_features=new_global_features,
            intersections=new_intersections,
            traffic_flow=new_traffic_flow,
            congestion_level=new_congestion,
            incident_locations=new_incidents,
            weather_condition=new_weather,
            time_of_day=new_time,
            timestamp=state.timestamp + 1,
        )
    
    def _compute_traffic_flow(
        self, 
        state: TrafficState, 
        intersections: List[IntersectionNode],
        time_of_day: float
    ) -> jnp.ndarray:
        """Compute traffic flow on edges."""
        # Rush hour pattern
        if self.rush_hour_dynamics:
            if 7.0 <= time_of_day <= 9.0 or 17.0 <= time_of_day <= 19.0:
                flow_multiplier = 2.0
            elif 22.0 <= time_of_day or time_of_day <= 6.0:
                flow_multiplier = 0.5
            else:
                flow_multiplier = 1.0
        else:
            flow_multiplier = 1.0
        
        # Weather impact
        weather_impact = 1.0 - 0.3 * state.weather_condition
        
        # Base flow with rush hour and weather effects
        base_flow = self.base_flow_rate * flow_multiplier * weather_impact
        
        # Add random variations
        flow_variations = np.random.normal(0, base_flow * 0.2, size=len(state.traffic_flow))
        new_flow = state.traffic_flow + flow_variations
        
        # Traffic signal impact on flow
        for i, edge_idx in enumerate(state.edge_index.T):
            source_node, target_node = int(edge_idx[0]), int(edge_idx[1])
            
            # Flow reduced if source intersection has red light
            if source_node < len(intersections):
                source_signal = intersections[source_node].traffic_signal_state
                if source_signal == 0:  # Red
                    new_flow = new_flow.at[i].multiply(0.3)
                elif source_signal == 1:  # Yellow
                    new_flow = new_flow.at[i].multiply(0.7)
            
            # Additional reduction for incidents
            if source_node in state.incident_locations or target_node in state.incident_locations:
                new_flow = new_flow.at[i].multiply(0.2)
        
        # Ensure non-negative flow
        new_flow = jnp.maximum(new_flow, 0)
        
        return new_flow
    
    def _update_queue_lengths(
        self,
        intersections: List[IntersectionNode],
        traffic_flow: jnp.ndarray,
        state: TrafficState,
    ) -> None:
        """Update queue lengths at intersections."""
        for intersection in intersections:
            node_id = intersection.node_id
            
            # Find incoming flow
            incoming_flow = 0.0
            for i, edge_idx in enumerate(state.edge_index.T):
                if int(edge_idx[1]) == node_id and i < len(traffic_flow):
                    incoming_flow += float(traffic_flow[i])
            
            # Find outgoing capacity
            outgoing_capacity = intersection.flow_capacity
            if intersection.traffic_signal_state == 0:  # Red
                outgoing_capacity *= 0.1
            elif intersection.traffic_signal_state == 1:  # Yellow
                outgoing_capacity *= 0.5
            
            # Update queue: inflow - outflow
            queue_change = (incoming_flow - outgoing_capacity) * 0.1  # Time step factor
            intersection.queue_length = max(0, intersection.queue_length + queue_change)
    
    def _compute_reward(
        self,
        state: TrafficState,
        actions: jnp.ndarray,
        next_state: TrafficState,
    ) -> float:
        """Compute reward for traffic network optimization."""
        # Minimize total delay
        delay_reward = -(next_state.get_total_delay() / len(next_state.intersections))
        
        # Minimize congestion
        congestion_penalty = -next_state.get_average_congestion() * 10
        
        # Maximize throughput
        throughput_reward = next_state.get_throughput() * 0.1
        
        # Penalty for incidents
        incident_penalty = -len(next_state.incident_locations) * 5
        
        # Reward for signal coordination (if enabled)
        coordination_reward = 0.0
        if self.signal_coordination:
            coordination_reward = self._compute_coordination_reward(next_state)
        
        total_reward = (
            delay_reward + 
            congestion_penalty + 
            throughput_reward + 
            incident_penalty + 
            coordination_reward
        )
        
        return float(total_reward)
    
    def _compute_coordination_reward(self, state: TrafficState) -> float:
        """Reward for coordinated signal timing."""
        if len(state.intersections) < 2:
            return 0.0
        
        # Simple coordination: reward adjacent green signals
        coordination_score = 0.0
        
        for i, edge_idx in enumerate(state.edge_index.T):
            source_node, target_node = int(edge_idx[0]), int(edge_idx[1])
            
            if (source_node < len(state.intersections) and 
                target_node < len(state.intersections)):
                
                source_signal = state.intersections[source_node].traffic_signal_state
                target_signal = state.intersections[target_node].traffic_signal_state
                
                # Reward if both are green (creates green wave)
                if source_signal == 2 and target_signal == 2:
                    coordination_score += 1.0
        
        return coordination_score * 0.5
    
    def get_multi_agent_actions(self, actions: Dict[int, int]) -> jnp.ndarray:
        """Convert multi-agent action dictionary to array."""
        action_array = jnp.zeros(self.num_intersections, dtype=int)
        
        for agent_id, action in actions.items():
            if 0 <= agent_id < self.num_intersections:
                action_array = action_array.at[agent_id].set(action)
        
        return action_array
    
    def get_agent_rewards(
        self,
        state: TrafficState,
        actions: Dict[int, int],
        next_state: TrafficState,
    ) -> Dict[int, float]:
        """Compute individual agent rewards."""
        global_reward = self._compute_reward(state, self.get_multi_agent_actions(actions), next_state)
        
        # Individual local rewards
        agent_rewards = {}
        for agent_id in actions.keys():
            if agent_id < len(next_state.intersections):
                intersection = next_state.intersections[agent_id]
                
                # Local delay reduction
                local_delay = intersection.compute_delay()
                local_reward = -local_delay
                
                # Local congestion
                local_congestion = 0.0
                edge_count = 0
                for i, edge_idx in enumerate(next_state.edge_index.T):
                    if int(edge_idx[0]) == agent_id or int(edge_idx[1]) == agent_id:
                        local_congestion += float(next_state.congestion_level[i])
                        edge_count += 1
                
                if edge_count > 0:
                    local_reward -= (local_congestion / edge_count) * 5
                
                # Combine global and local rewards
                agent_rewards[agent_id] = 0.7 * global_reward + 0.3 * local_reward
            else:
                agent_rewards[agent_id] = 0.0
        
        return agent_rewards