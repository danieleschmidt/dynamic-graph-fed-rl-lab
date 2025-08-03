"""Asynchronous gossip protocol for federated learning."""

import asyncio
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import networkx as nx

from .base import BaseFederatedProtocol


class AsyncGossipProtocol(BaseFederatedProtocol):
    """Asynchronous gossip protocol for decentralized parameter sharing."""
    
    def __init__(
        self,
        num_agents: int,
        topology: str = "random",
        fanout: int = 3,
        communication_round: int = 100,
        compression_ratio: float = 0.1,
        byzantine_tolerance: bool = True,
        gossip_interval: float = 1.0,
        max_concurrent_exchanges: int = 5,
    ):
        super().__init__(
            num_agents=num_agents,
            communication_round=communication_round,
            compression_ratio=compression_ratio,
            byzantine_tolerance=byzantine_tolerance,
        )
        
        self.topology = topology
        self.fanout = fanout
        self.gossip_interval = gossip_interval
        self.max_concurrent_exchanges = max_concurrent_exchanges
        
        # Build communication topology
        self.graph = self._build_topology()
        
        # Agent state tracking
        self.agent_versions = {i: 0 for i in range(num_agents)}
        self.parameter_cache = {}
        self.exchange_locks = {}
        self.active_exchanges = set()
        
        # Gossip statistics
        self.gossip_stats = {
            "total_exchanges": 0,
            "successful_exchanges": 0,
            "failed_exchanges": 0,
            "avg_exchange_time": 0.0,
            "bytes_transferred": 0,
        }
    
    def _build_topology(self) -> nx.Graph:
        """Build communication topology graph."""
        if self.topology == "random":
            return self._random_topology()
        elif self.topology == "ring":
            return self._ring_topology()
        elif self.topology == "small_world":
            return self._small_world_topology()
        elif self.topology == "scale_free":
            return self._scale_free_topology()
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
    
    def _random_topology(self) -> nx.Graph:
        """Create random graph topology."""
        G = nx.Graph()
        G.add_nodes_from(range(self.num_agents))
        
        # Add random edges ensuring connectivity
        for i in range(self.num_agents):
            # Ensure each node has at least fanout neighbors
            neighbors = random.sample(
                [j for j in range(self.num_agents) if j != i],
                min(self.fanout, self.num_agents - 1)
            )
            for neighbor in neighbors:
                G.add_edge(i, neighbor)
        
        return G
    
    def _ring_topology(self) -> nx.Graph:
        """Create ring topology."""
        G = nx.cycle_graph(self.num_agents)
        
        # Add shortcuts for better connectivity
        for i in range(self.num_agents):
            # Add shortcuts to nodes fanout//2 away
            for offset in range(1, min(self.fanout // 2 + 1, self.num_agents // 2)):
                neighbor = (i + offset) % self.num_agents
                G.add_edge(i, neighbor)
        
        return G
    
    def _small_world_topology(self) -> nx.Graph:
        """Create small-world topology."""
        # Watts-Strogatz small-world graph
        k = min(self.fanout, self.num_agents // 2)
        p = 0.3  # Rewiring probability
        return nx.watts_strogatz_graph(self.num_agents, k, p)
    
    def _scale_free_topology(self) -> nx.Graph:
        """Create scale-free topology."""
        # Barabási–Albert preferential attachment
        m = min(self.fanout // 2, self.num_agents // 4)
        return nx.barabasi_albert_graph(self.num_agents, m)
    
    def select_communication_partners(
        self,
        agent_id: int,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Select communication partners based on topology."""
        if exclude_ids is None:
            exclude_ids = []
        
        # Get neighbors from topology
        neighbors = list(self.graph.neighbors(agent_id))
        
        # Remove excluded agents
        available_neighbors = [n for n in neighbors if n not in exclude_ids]
        
        # Select subset of neighbors for this round
        k = min(self.fanout, len(available_neighbors))
        if k > 0:
            return random.sample(available_neighbors, k)
        else:
            return []
    
    async def agent_gossip_loop(
        self,
        agent_id: int,
        get_agent_params_fn: callable,
        update_agent_params_fn: callable,
    ) -> None:
        """Main gossip loop for an agent."""
        while True:
            try:
                # Wait for gossip interval
                await asyncio.sleep(self.gossip_interval + random.uniform(-0.1, 0.1))
                
                # Check if we can start new exchanges
                if len(self.active_exchanges) >= self.max_concurrent_exchanges:
                    continue
                
                # Select communication partners
                partners = self.select_communication_partners(
                    agent_id,
                    exclude_ids=list(self.active_exchanges)
                )
                
                if not partners:
                    continue
                
                # Start exchange with one random partner
                partner_id = random.choice(partners)
                
                # Create exchange task
                exchange_task = asyncio.create_task(
                    self._exchange_parameters(
                        agent_id,
                        partner_id,
                        get_agent_params_fn,
                        update_agent_params_fn,
                    )
                )
                
                # Don't await - let it run asynchronously
                
            except Exception as e:
                print(f"Error in gossip loop for agent {agent_id}: {e}")
                await asyncio.sleep(1.0)
    
    async def _exchange_parameters(
        self,
        agent_id: int,
        partner_id: int,
        get_agent_params_fn: callable,
        update_agent_params_fn: callable,
    ) -> bool:
        """Exchange parameters between two agents."""
        exchange_id = f"{agent_id}-{partner_id}"
        
        # Check if exchange is already active
        if exchange_id in self.active_exchanges or f"{partner_id}-{agent_id}" in self.active_exchanges:
            return False
        
        # Mark exchange as active
        self.active_exchanges.add(exchange_id)
        
        try:
            start_time = time.time()
            
            # Get current parameters
            local_params = get_agent_params_fn(agent_id)
            partner_params = get_agent_params_fn(partner_id)
            
            # Check parameter versions to avoid unnecessary exchanges
            local_version = self.agent_versions[agent_id]
            partner_version = self.agent_versions[partner_id]
            
            if local_version == partner_version:
                # No new information to exchange
                return True
            
            # Compress parameters
            compressed_local = self.compress_parameters(local_params)
            compressed_partner = self.compress_parameters(partner_params)
            
            # Simulate network delay
            await asyncio.sleep(random.uniform(0.01, 0.05))
            
            # Aggregate parameters (simple averaging)
            aggregated_params = self._aggregate_two_agents(
                compressed_local, compressed_partner
            )
            
            # Update both agents
            update_agent_params_fn(agent_id, aggregated_params)
            update_agent_params_fn(partner_id, aggregated_params)
            
            # Update versions
            new_version = max(local_version, partner_version) + 1
            self.agent_versions[agent_id] = new_version
            self.agent_versions[partner_id] = new_version
            
            # Update statistics
            exchange_time = time.time() - start_time
            self.gossip_stats["successful_exchanges"] += 1
            self.gossip_stats["total_exchanges"] += 1
            self.gossip_stats["avg_exchange_time"] = (
                (self.gossip_stats["avg_exchange_time"] * (self.gossip_stats["successful_exchanges"] - 1) + exchange_time) /
                self.gossip_stats["successful_exchanges"]
            )
            
            # Estimate bytes transferred (simplified)
            bytes_transferred = sum(
                param.nbytes for param in compressed_local.values() 
                if isinstance(param, jnp.ndarray)
            ) * 2  # Bidirectional
            self.gossip_stats["bytes_transferred"] += bytes_transferred
            
            # Log communication
            self.log_communication(agent_id, partner_id, bytes_transferred, time.time())
            
            return True
            
        except Exception as e:
            self.gossip_stats["failed_exchanges"] += 1
            self.gossip_stats["total_exchanges"] += 1
            print(f"Exchange failed between agents {agent_id} and {partner_id}: {e}")
            return False
            
        finally:
            # Remove from active exchanges
            self.active_exchanges.discard(exchange_id)
    
    def _aggregate_two_agents(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate parameters from two agents."""
        aggregated = {}
        
        for key in params1.keys():
            if key in params2:
                if isinstance(params1[key], jnp.ndarray):
                    # Simple averaging
                    aggregated[key] = 0.5 * (params1[key] + params2[key])
                else:
                    # Keep first agent's value for non-tensors
                    aggregated[key] = params1[key]
            else:
                aggregated[key] = params1[key]
        
        # Add any keys only in params2
        for key in params2.keys():
            if key not in aggregated:
                aggregated[key] = params2[key]
        
        return aggregated
    
    async def aggregate_parameters(
        self,
        agent_parameters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate parameters from all agents (for synchronous fallback)."""
        if not agent_parameters:
            return {}
        
        # Use weighted averaging
        weights = [1.0 / len(agent_parameters)] * len(agent_parameters)
        
        aggregated = {}
        keys = list(agent_parameters[0].keys())
        
        for key in keys:
            if not isinstance(agent_parameters[0][key], jnp.ndarray):
                aggregated[key] = agent_parameters[0][key]
                continue
            
            # Weighted average
            weighted_sum = jnp.zeros_like(agent_parameters[0][key])
            for params, weight in zip(agent_parameters, weights):
                if key in params:
                    weighted_sum += weight * params[key]
            
            aggregated[key] = weighted_sum
        
        return aggregated
    
    def get_topology_metrics(self) -> Dict[str, Any]:
        """Get communication topology metrics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.num_agents,
            "clustering_coefficient": nx.average_clustering(self.graph),
            "avg_shortest_path": nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else float('inf'),
            "diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else float('inf'),
            "is_connected": nx.is_connected(self.graph),
        }
    
    def get_gossip_metrics(self) -> Dict[str, Any]:
        """Get gossip protocol specific metrics."""
        success_rate = (
            self.gossip_stats["successful_exchanges"] / self.gossip_stats["total_exchanges"]
            if self.gossip_stats["total_exchanges"] > 0 else 0.0
        )
        
        return {
            **self.gossip_stats,
            "success_rate": success_rate,
            "active_exchanges": len(self.active_exchanges),
            "agent_versions": self.agent_versions.copy(),
            "topology_metrics": self.get_topology_metrics(),
        }
    
    def visualize_topology(self, save_path: Optional[str] = None) -> None:
        """Visualize communication topology."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 8))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(self.graph, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph, pos, 
                node_color='lightblue',
                node_size=500,
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                self.graph, pos,
                alpha=0.5,
                edge_color='gray'
            )
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, pos)
            
            plt.title(f"Communication Topology ({self.topology})")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("matplotlib not available for topology visualization")
    
    def update_topology(self, new_topology: str = None) -> None:
        """Update communication topology (for dynamic environments)."""
        if new_topology:
            self.topology = new_topology
        
        # Rebuild graph
        self.graph = self._build_topology()
        
        # Reset versions to trigger new exchanges
        for agent_id in range(self.num_agents):
            self.agent_versions[agent_id] += 1


class AdaptiveGossipProtocol(AsyncGossipProtocol):
    """Adaptive gossip protocol that adjusts parameters based on performance."""
    
    def __init__(
        self,
        num_agents: int,
        adaptation_interval: int = 50,
        performance_window: int = 10,
        **kwargs,
    ):
        super().__init__(num_agents, **kwargs)
        
        self.adaptation_interval = adaptation_interval
        self.performance_window = performance_window
        self.performance_history = []
        self.adaptation_count = 0
    
    def adapt_parameters(self, performance_metric: float) -> None:
        """Adapt gossip parameters based on performance."""
        self.performance_history.append(performance_metric)
        
        # Keep only recent performance
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)
        
        # Check if it's time to adapt
        if len(self.performance_history) >= self.performance_window:
            recent_avg = sum(self.performance_history) / len(self.performance_history)
            
            if len(self.performance_history) >= 2:
                prev_avg = sum(self.performance_history[:-1]) / (len(self.performance_history) - 1)
                
                if recent_avg < prev_avg:  # Performance getting worse
                    # Increase communication frequency
                    self.gossip_interval = max(0.1, self.gossip_interval * 0.9)
                    self.fanout = min(self.num_agents - 1, self.fanout + 1)
                elif recent_avg > prev_avg * 1.1:  # Performance improving significantly
                    # Decrease communication frequency to save bandwidth
                    self.gossip_interval = min(5.0, self.gossip_interval * 1.1)
                    self.fanout = max(1, self.fanout - 1)
        
        self.adaptation_count += 1
    
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get adaptation-specific metrics."""
        base_metrics = self.get_gossip_metrics()
        
        adaptation_metrics = {
            "adaptation_count": self.adaptation_count,
            "current_gossip_interval": self.gossip_interval,
            "current_fanout": self.fanout,
            "performance_history": self.performance_history.copy(),
        }
        
        return {**base_metrics, **adaptation_metrics}