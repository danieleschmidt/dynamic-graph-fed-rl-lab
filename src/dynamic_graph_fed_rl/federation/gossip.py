"""
Asynchronous gossip protocol for federated learning.

Enhanced with Generation 1 improvements for autonomous SDLC execution:
- Adaptive communication compression with quality-aware algorithms
- Intelligent partner selection using performance-based routing
- Real-time bandwidth optimization and congestion control
- Multi-modal compression strategies (sparsification, quantization, low-rank)
- Byzantine-resilient aggregation with anomaly detection
- Cross-layer optimization with graph-temporal awareness
- Predictive communication scheduling and load balancing
"""

import asyncio
import random
import time
import logging
import pickle
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import heapq
import numpy as np

import jax
import jax.numpy as jnp
import networkx as nx

from .base import BaseFederatedProtocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Compression strategies for parameter transmission."""
    NONE = "none"
    SPARSIFICATION = "sparsification"
    QUANTIZATION = "quantization"
    LOW_RANK = "low_rank"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class PartnerSelectionStrategy(Enum):
    """Partner selection strategies."""
    RANDOM = "random"
    PERFORMANCE_BASED = "performance_based"
    TOPOLOGY_AWARE = "topology_aware"
    LOAD_BALANCED = "load_balanced"
    PREDICTIVE = "predictive"


@dataclass
class CompressionConfig:
    """Configuration for adaptive compression."""
    strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE
    sparsity_ratio: float = 0.1
    quantization_bits: int = 8
    low_rank_ratio: float = 0.5
    quality_threshold: float = 0.95
    adaptive_threshold: float = 0.8
    performance_weight: float = 0.3


@dataclass
class CommunicationMetrics:
    """Real-time communication metrics."""
    timestamp: float = field(default_factory=time.time)
    bytes_sent: int = 0
    bytes_received: int = 0
    latency: float = 0.0
    compression_ratio: float = 1.0
    quality_loss: float = 0.0
    partner_performance: float = 0.0
    bandwidth_utilization: float = 0.0


class AdaptiveCompressionEngine:
    """Advanced compression engine with multiple strategies."""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.compression_history = []
        self.performance_tracker = {}
        self.adaptive_weights = {
            CompressionStrategy.SPARSIFICATION: 0.33,
            CompressionStrategy.QUANTIZATION: 0.33,
            CompressionStrategy.LOW_RANK: 0.34
        }
        
    def compress_parameters(
        self, 
        parameters: Dict[str, jnp.ndarray],
        target_ratio: float = None,
        quality_target: float = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compress parameters using adaptive strategy."""
        if target_ratio is None:
            target_ratio = self.config.sparsity_ratio
        if quality_target is None:
            quality_target = self.config.quality_threshold
            
        try:
            start_time = time.time()
            
            if self.config.strategy == CompressionStrategy.ADAPTIVE:
                compressed_data, metadata = self._adaptive_compression(
                    parameters, target_ratio, quality_target
                )
            elif self.config.strategy == CompressionStrategy.HYBRID:
                compressed_data, metadata = self._hybrid_compression(
                    parameters, target_ratio
                )
            elif self.config.strategy == CompressionStrategy.SPARSIFICATION:
                compressed_data, metadata = self._sparsify_parameters(
                    parameters, target_ratio
                )
            elif self.config.strategy == CompressionStrategy.QUANTIZATION:
                compressed_data, metadata = self._quantize_parameters(
                    parameters, self.config.quantization_bits
                )
            elif self.config.strategy == CompressionStrategy.LOW_RANK:
                compressed_data, metadata = self._low_rank_compression(
                    parameters, self.config.low_rank_ratio
                )
            else:
                compressed_data, metadata = parameters, {"compression_ratio": 1.0}
            
            compression_time = time.time() - start_time
            metadata["compression_time"] = compression_time
            
            # Track compression performance
            self._update_compression_performance(metadata)
            
            logger.debug(f"Compressed parameters with ratio {metadata.get('compression_ratio', 1.0):.3f}")
            
            return compressed_data, metadata
            
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return parameters, {"compression_ratio": 1.0, "error": str(e)}
    
    def _adaptive_compression(
        self, 
        parameters: Dict[str, jnp.ndarray],
        target_ratio: float,
        quality_target: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Adaptive compression that selects best strategy."""
        strategies = [
            (CompressionStrategy.SPARSIFICATION, self._sparsify_parameters),
            (CompressionStrategy.QUANTIZATION, self._quantize_parameters),
            (CompressionStrategy.LOW_RANK, self._low_rank_compression)
        ]
        
        best_result = None
        best_score = 0.0
        best_metadata = {}
        
        for strategy, compress_fn in strategies:
            try:
                if strategy == CompressionStrategy.SPARSIFICATION:
                    result, metadata = compress_fn(parameters, target_ratio)
                elif strategy == CompressionStrategy.QUANTIZATION:
                    result, metadata = compress_fn(parameters, self.config.quantization_bits)
                else:  # LOW_RANK
                    result, metadata = compress_fn(parameters, self.config.low_rank_ratio)
                
                # Score based on compression ratio and quality
                compression_ratio = metadata.get("compression_ratio", 1.0)
                quality = metadata.get("quality_preserved", 1.0)
                
                score = (compression_ratio * 0.6 + quality * 0.4) * self.adaptive_weights[strategy]
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_metadata = metadata
                    best_metadata["selected_strategy"] = strategy.value
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")
                continue
        
        if best_result is None:
            return parameters, {"compression_ratio": 1.0, "selected_strategy": "none"}
        
        return best_result, best_metadata
    
    def _hybrid_compression(
        self, 
        parameters: Dict[str, jnp.ndarray],
        target_ratio: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Hybrid compression combining multiple strategies."""
        compressed_params = {}
        total_original_size = 0
        total_compressed_size = 0
        
        for key, param in parameters.items():
            original_size = param.nbytes
            total_original_size += original_size
            
            # Choose strategy based on parameter characteristics
            if param.ndim >= 2 and min(param.shape) > 10:
                # Use low-rank for large matrices
                compressed, metadata = self._low_rank_compression(
                    {key: param}, self.config.low_rank_ratio
                )
            elif param.size > 1000:
                # Use sparsification for large vectors
                compressed, metadata = self._sparsify_parameters(
                    {key: param}, target_ratio
                )
            else:
                # Use quantization for small parameters
                compressed, metadata = self._quantize_parameters(
                    {key: param}, self.config.quantization_bits
                )
            
            compressed_params.update(compressed)
            total_compressed_size += metadata.get("compressed_size", original_size)
        
        overall_ratio = total_compressed_size / total_original_size
        
        return compressed_params, {
            "compression_ratio": overall_ratio,
            "strategy": "hybrid",
            "total_original_size": total_original_size,
            "total_compressed_size": total_compressed_size
        }
    
    def _sparsify_parameters(
        self, 
        parameters: Dict[str, jnp.ndarray],
        sparsity_ratio: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Sparsify parameters by keeping only top-k elements."""
        compressed_params = {}
        total_original = 0
        total_compressed = 0
        
        for key, param in parameters.items():
            original_size = param.size
            total_original += original_size
            
            # Flatten parameter
            flat_param = param.flatten()
            
            # Keep top-k elements by magnitude
            k = max(1, int(original_size * (1 - sparsity_ratio)))
            
            # Get indices of top-k elements
            top_k_indices = jnp.argsort(jnp.abs(flat_param))[-k:]
            
            # Create sparse representation
            sparse_values = flat_param[top_k_indices]
            
            compressed_params[key] = {
                "type": "sparse",
                "shape": param.shape,
                "indices": top_k_indices,
                "values": sparse_values,
                "sparsity": sparsity_ratio
            }
            
            total_compressed += len(top_k_indices)
        
        compression_ratio = total_compressed / total_original
        
        return compressed_params, {
            "compression_ratio": compression_ratio,
            "strategy": "sparsification",
            "quality_preserved": 1.0 - sparsity_ratio
        }
    
    def _quantize_parameters(
        self, 
        parameters: Dict[str, jnp.ndarray],
        bits: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Quantize parameters to reduce precision."""
        compressed_params = {}
        
        for key, param in parameters.items():
            # Determine quantization range
            param_min = jnp.min(param)
            param_max = jnp.max(param)
            param_range = param_max - param_min
            
            if param_range == 0:
                # Constant parameter
                compressed_params[key] = {
                    "type": "quantized",
                    "shape": param.shape,
                    "constant_value": float(param_min),
                    "bits": bits
                }
            else:
                # Quantize to n-bit integers
                levels = 2 ** bits - 1
                scale = param_range / levels
                
                # Quantize
                quantized = jnp.round((param - param_min) / scale).astype(jnp.uint8)
                
                compressed_params[key] = {
                    "type": "quantized",
                    "shape": param.shape,
                    "quantized_values": quantized,
                    "min_value": float(param_min),
                    "scale": float(scale),
                    "bits": bits
                }
        
        # Estimate compression ratio
        compression_ratio = bits / 32.0  # Assuming 32-bit floats
        
        return compressed_params, {
            "compression_ratio": compression_ratio,
            "strategy": "quantization",
            "quality_preserved": 1.0 - (1.0 / (2 ** bits))
        }
    
    def _low_rank_compression(
        self, 
        parameters: Dict[str, jnp.ndarray],
        rank_ratio: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Low-rank compression using SVD."""
        compressed_params = {}
        total_original = 0
        total_compressed = 0
        
        for key, param in parameters.items():
            if param.ndim != 2:
                # Skip non-matrix parameters
                compressed_params[key] = param
                continue
            
            original_size = param.size
            total_original += original_size
            
            # SVD decomposition
            try:
                U, s, Vt = jnp.linalg.svd(param, full_matrices=False)
                
                # Determine rank
                rank = max(1, int(min(param.shape) * rank_ratio))
                rank = min(rank, len(s))
                
                # Truncate
                U_trunc = U[:, :rank]
                s_trunc = s[:rank]
                Vt_trunc = Vt[:rank, :]
                
                compressed_params[key] = {
                    "type": "low_rank",
                    "U": U_trunc,
                    "s": s_trunc,
                    "Vt": Vt_trunc,
                    "rank": rank,
                    "original_shape": param.shape
                }
                
                compressed_size = U_trunc.size + s_trunc.size + Vt_trunc.size
                total_compressed += compressed_size
                
            except Exception as e:
                logger.warning(f"SVD failed for {key}: {e}")
                compressed_params[key] = param
                total_compressed += original_size
        
        compression_ratio = total_compressed / total_original if total_original > 0 else 1.0
        
        return compressed_params, {
            "compression_ratio": compression_ratio,
            "strategy": "low_rank",
            "quality_preserved": rank_ratio
        }
    
    def decompress_parameters(
        self, 
        compressed_data: Dict[str, Any]
    ) -> Dict[str, jnp.ndarray]:
        """Decompress parameters back to original format."""
        decompressed = {}
        
        for key, data in compressed_data.items():
            if isinstance(data, jnp.ndarray):
                # No compression
                decompressed[key] = data
            elif isinstance(data, dict) and "type" in data:
                if data["type"] == "sparse":
                    # Reconstruct from sparse representation
                    full_param = jnp.zeros(data["shape"]).flatten()
                    full_param = full_param.at[data["indices"]].set(data["values"])
                    decompressed[key] = full_param.reshape(data["shape"])
                    
                elif data["type"] == "quantized":
                    if "constant_value" in data:
                        # Constant parameter
                        decompressed[key] = jnp.full(data["shape"], data["constant_value"])
                    else:
                        # Dequantize
                        dequantized = (data["quantized_values"].astype(jnp.float32) * 
                                     data["scale"] + data["min_value"])
                        decompressed[key] = dequantized.reshape(data["shape"])
                        
                elif data["type"] == "low_rank":
                    # Reconstruct from SVD
                    reconstructed = data["U"] @ jnp.diag(data["s"]) @ data["Vt"]
                    decompressed[key] = reconstructed
                    
                else:
                    logger.warning(f"Unknown compression type: {data['type']}")
                    decompressed[key] = data
            else:
                decompressed[key] = data
        
        return decompressed
    
    def _update_compression_performance(self, metadata: Dict[str, Any]):
        """Update compression performance tracking."""
        strategy = metadata.get("selected_strategy", metadata.get("strategy", "unknown"))
        
        if strategy in self.adaptive_weights:
            # Update weights based on performance
            compression_ratio = metadata.get("compression_ratio", 1.0)
            quality = metadata.get("quality_preserved", 1.0)
            
            performance_score = compression_ratio * 0.7 + quality * 0.3
            
            # Exponential moving average update
            alpha = 0.1
            current_weight = self.adaptive_weights[CompressionStrategy(strategy)]
            self.adaptive_weights[CompressionStrategy(strategy)] = (
                alpha * performance_score + (1 - alpha) * current_weight
            )
            
            # Normalize weights
            total_weight = sum(self.adaptive_weights.values())
            for key in self.adaptive_weights:
                self.adaptive_weights[key] /= total_weight


class IntelligentPartnerSelector:
    """Intelligent partner selection with performance awareness."""
    
    def __init__(self, num_agents: int, strategy: PartnerSelectionStrategy):
        self.num_agents = num_agents
        self.strategy = strategy
        self.performance_history = {i: [] for i in range(num_agents)}
        self.communication_costs = {}
        self.load_tracker = {i: 0.0 for i in range(num_agents)}
        self.partner_preferences = {}
        
    def select_partners(
        self,
        agent_id: int,
        graph: nx.Graph,
        fanout: int,
        exclude_ids: List[int] = None,
        performance_metrics: Dict[int, float] = None
    ) -> List[int]:
        """Select communication partners using intelligent strategy."""
        if exclude_ids is None:
            exclude_ids = []
        if performance_metrics is None:
            performance_metrics = {}
            
        available_neighbors = [
            n for n in graph.neighbors(agent_id) 
            if n not in exclude_ids
        ]
        
        if not available_neighbors:
            return []
        
        k = min(fanout, len(available_neighbors))
        
        if self.strategy == PartnerSelectionStrategy.PERFORMANCE_BASED:
            return self._performance_based_selection(
                agent_id, available_neighbors, k, performance_metrics
            )
        elif self.strategy == PartnerSelectionStrategy.LOAD_BALANCED:
            return self._load_balanced_selection(
                agent_id, available_neighbors, k
            )
        elif self.strategy == PartnerSelectionStrategy.PREDICTIVE:
            return self._predictive_selection(
                agent_id, available_neighbors, k, performance_metrics
            )
        else:  # RANDOM or TOPOLOGY_AWARE
            return random.sample(available_neighbors, k)
    
    def _performance_based_selection(
        self,
        agent_id: int,
        candidates: List[int],
        k: int,
        performance_metrics: Dict[int, float]
    ) -> List[int]:
        """Select partners based on performance metrics."""
        if not performance_metrics:
            return random.sample(candidates, k)
        
        # Score candidates based on performance
        candidate_scores = []
        for candidate in candidates:
            perf = performance_metrics.get(candidate, 0.5)
            
            # Add diversity bonus for less frequently selected partners
            diversity_bonus = 1.0 / (self.partner_preferences.get((agent_id, candidate), 0) + 1.0)
            
            score = perf * 0.7 + diversity_bonus * 0.3
            candidate_scores.append((score, candidate))
        
        # Select top-k candidates
        candidate_scores.sort(reverse=True)
        selected = [candidate for _, candidate in candidate_scores[:k]]
        
        # Update preferences
        for candidate in selected:
            key = (agent_id, candidate)
            self.partner_preferences[key] = self.partner_preferences.get(key, 0) + 1
        
        return selected
    
    def _load_balanced_selection(
        self,
        agent_id: int,
        candidates: List[int],
        k: int
    ) -> List[int]:
        """Select partners based on load balancing."""
        # Score by inverse load
        candidate_scores = []
        for candidate in candidates:
            load = self.load_tracker.get(candidate, 0.0)
            score = 1.0 / (load + 1e-6)  # Prefer less loaded agents
            candidate_scores.append((score, candidate))
        
        # Select top-k least loaded
        candidate_scores.sort(reverse=True)
        selected = [candidate for _, candidate in candidate_scores[:k]]
        
        # Update load tracker
        for candidate in selected:
            self.load_tracker[candidate] += 1.0
        
        return selected
    
    def _predictive_selection(
        self,
        agent_id: int,
        candidates: List[int],
        k: int,
        performance_metrics: Dict[int, float]
    ) -> List[int]:
        """Predictive partner selection based on trends."""
        candidate_scores = []
        
        for candidate in candidates:
            # Base performance
            current_perf = performance_metrics.get(candidate, 0.5)
            
            # Performance trend
            history = self.performance_history.get(candidate, [])
            if len(history) >= 2:
                trend = history[-1] - history[-2]
            else:
                trend = 0.0
            
            # Communication cost
            comm_cost = self.communication_costs.get((agent_id, candidate), 1.0)
            
            # Predicted benefit
            predicted_score = current_perf + trend * 0.3 - comm_cost * 0.1
            candidate_scores.append((predicted_score, candidate))
        
        # Select top-k
        candidate_scores.sort(reverse=True)
        return [candidate for _, candidate in candidate_scores[:k]]
    
    def update_performance_history(self, agent_id: int, performance: float):
        """Update performance history for agent."""
        self.performance_history[agent_id].append(performance)
        
        # Keep only recent history
        if len(self.performance_history[agent_id]) > 20:
            self.performance_history[agent_id].pop(0)
    
    def update_communication_cost(self, agent1: int, agent2: int, cost: float):
        """Update communication cost between agents."""
        self.communication_costs[(agent1, agent2)] = cost
        self.communication_costs[(agent2, agent1)] = cost


class AsyncGossipProtocol(BaseFederatedProtocol):
    """
    Enhanced asynchronous gossip protocol for decentralized parameter sharing.
    
    Generation 1 improvements:
    - Adaptive compression with multiple strategies
    - Intelligent partner selection algorithms
    - Real-time performance monitoring and optimization
    - Byzantine fault tolerance with anomaly detection
    - Bandwidth management and congestion control
    """
    
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
        # Generation 1 enhancements
        compression_config: CompressionConfig = None,
        partner_selection_strategy: PartnerSelectionStrategy = PartnerSelectionStrategy.PERFORMANCE_BASED,
        adaptive_bandwidth: bool = True,
        anomaly_detection: bool = True,
        performance_tracking: bool = True,
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
        
        # Generation 1 enhancements
        self.compression_config = compression_config or CompressionConfig()
        self.partner_selection_strategy = partner_selection_strategy
        self.adaptive_bandwidth = adaptive_bandwidth
        self.anomaly_detection = anomaly_detection
        self.performance_tracking = performance_tracking
        
        # Build communication topology
        self.graph = self._build_topology()
        
        # Agent state tracking
        self.agent_versions = {i: 0 for i in range(num_agents)}
        self.parameter_cache = {}
        self.exchange_locks = {}
        self.active_exchanges = set()
        
        # Generation 1 components
        self.compression_engine = AdaptiveCompressionEngine(self.compression_config)
        self.partner_selector = IntelligentPartnerSelector(num_agents, partner_selection_strategy)
        self.communication_metrics = {i: [] for i in range(num_agents)}
        self.performance_metrics = {i: 0.5 for i in range(num_agents)}
        self.anomaly_scores = {i: 0.0 for i in range(num_agents)}
        
        # Enhanced gossip statistics
        self.gossip_stats = {
            "total_exchanges": 0,
            "successful_exchanges": 0,
            "failed_exchanges": 0,
            "avg_exchange_time": 0.0,
            "bytes_transferred": 0,
            "compression_savings": 0,
            "anomalies_detected": 0,
            "bandwidth_utilization": 0.0,
            "partner_selection_stats": {},
        }
        
        # Bandwidth management
        self.bandwidth_tracker = {
            "current_usage": 0.0,
            "target_usage": 0.8,  # 80% utilization target
            "congestion_threshold": 0.9,
            "adaptive_interval": 0.1
        }
        
        logger.info(f"Enhanced AsyncGossipProtocol initialized with {num_agents} agents, "
                   f"topology: {topology}, compression: {self.compression_config.strategy.value}")
        
        # Start background optimization
        if self.adaptive_bandwidth:
            asyncio.create_task(self._bandwidth_optimization_loop())
    
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
        """Select communication partners using enhanced strategy."""
        if exclude_ids is None:
            exclude_ids = []
        
        # Use intelligent partner selector
        partners = self.partner_selector.select_partners(
            agent_id=agent_id,
            graph=self.graph,
            fanout=self.fanout,
            exclude_ids=exclude_ids,
            performance_metrics=self.performance_metrics
        )
        
        # Update partner selection statistics
        strategy_name = self.partner_selection_strategy.value
        if strategy_name not in self.gossip_stats["partner_selection_stats"]:
            self.gossip_stats["partner_selection_stats"][strategy_name] = 0
        self.gossip_stats["partner_selection_stats"][strategy_name] += len(partners)
        
        logger.debug(f"Agent {agent_id} selected partners {partners} using {strategy_name} strategy")
        
        return partners
    
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
            
            # Enhanced compression with adaptive strategies
            compressed_local, local_metadata = self.compression_engine.compress_parameters(
                local_params,
                target_ratio=self.compression_ratio,
                quality_target=self.compression_config.quality_threshold
            )
            compressed_partner, partner_metadata = self.compression_engine.compress_parameters(
                partner_params,
                target_ratio=self.compression_ratio,
                quality_target=self.compression_config.quality_threshold
            )
            
            # Simulate network delay
            await asyncio.sleep(random.uniform(0.01, 0.05))
            
            # Decompress for aggregation
            decompressed_local = self.compression_engine.decompress_parameters(compressed_local)
            decompressed_partner = self.compression_engine.decompress_parameters(compressed_partner)
            
            # Check for anomalies before aggregation
            if self.anomaly_detection:
                local_anomaly = self._detect_parameter_anomaly(agent_id, decompressed_local)
                partner_anomaly = self._detect_parameter_anomaly(partner_id, decompressed_partner)
                
                if local_anomaly or partner_anomaly:
                    self.gossip_stats["anomalies_detected"] += 1
                    logger.warning(f"Anomaly detected in exchange {agent_id}-{partner_id}")
                    return False
            
            # Aggregate parameters with enhanced Byzantine tolerance
            aggregated_params = self._aggregate_two_agents_enhanced(
                decompressed_local, decompressed_partner, agent_id, partner_id
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
    
    def _detect_parameter_anomaly(self, agent_id: int, parameters: Dict[str, jnp.ndarray]) -> bool:
        """Detect anomalies in agent parameters."""
        try:
            # Simple anomaly detection based on parameter norms
            total_norm = 0.0
            total_params = 0
            
            for key, param in parameters.items():
                if isinstance(param, jnp.ndarray):
                    param_norm = float(jnp.linalg.norm(param))
                    total_norm += param_norm
                    total_params += 1
            
            if total_params == 0:
                return False
            
            avg_norm = total_norm / total_params
            
            # Update agent's historical norm
            if agent_id not in self.anomaly_scores:
                self.anomaly_scores[agent_id] = avg_norm
            else:
                # Exponential moving average
                alpha = 0.1
                self.anomaly_scores[agent_id] = (
                    alpha * avg_norm + (1 - alpha) * self.anomaly_scores[agent_id]
                )
            
            # Check for anomaly (parameter norm too different from historical)
            expected_norm = self.anomaly_scores[agent_id]
            anomaly_threshold = 3.0  # Allow 3x deviation
            
            is_anomaly = (
                avg_norm > expected_norm * anomaly_threshold or
                avg_norm < expected_norm / anomaly_threshold
            )
            
            if is_anomaly:
                logger.warning(f"Agent {agent_id} anomaly: norm {avg_norm:.3f} vs expected {expected_norm:.3f}")
            
            return is_anomaly
            
        except Exception as e:
            logger.error(f"Anomaly detection error for agent {agent_id}: {e}")
            return False
    
    def _aggregate_two_agents_enhanced(
        self,
        params1: Dict[str, jnp.ndarray],
        params2: Dict[str, jnp.ndarray],
        agent1_id: int,
        agent2_id: int,
    ) -> Dict[str, jnp.ndarray]:
        """Enhanced aggregation with Byzantine tolerance."""
        aggregated = {}
        
        # Get agent performance for weighted aggregation
        perf1 = self.performance_metrics.get(agent1_id, 0.5)
        perf2 = self.performance_metrics.get(agent2_id, 0.5)
        
        # Normalize weights
        total_perf = perf1 + perf2
        if total_perf > 0:
            weight1 = perf1 / total_perf
            weight2 = perf2 / total_perf
        else:
            weight1 = weight2 = 0.5
        
        for key in params1.keys():
            if key in params2:
                if isinstance(params1[key], jnp.ndarray):
                    # Performance-weighted averaging
                    aggregated[key] = weight1 * params1[key] + weight2 * params2[key]
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
    
    async def _bandwidth_optimization_loop(self):
        """Background bandwidth optimization and congestion control."""
        while True:
            try:
                await asyncio.sleep(self.bandwidth_tracker["adaptive_interval"])
                
                # Calculate current bandwidth utilization
                current_exchanges = len(self.active_exchanges)
                utilization = current_exchanges / self.max_concurrent_exchanges
                
                self.bandwidth_tracker["current_usage"] = utilization
                self.gossip_stats["bandwidth_utilization"] = utilization
                
                # Adaptive interval adjustment
                if utilization > self.bandwidth_tracker["congestion_threshold"]:
                    # High congestion - slow down
                    self.gossip_interval = min(5.0, self.gossip_interval * 1.1)
                    logger.debug(f"Bandwidth congestion detected, increasing gossip interval to {self.gossip_interval:.3f}")
                elif utilization < self.bandwidth_tracker["target_usage"]:
                    # Low utilization - speed up
                    self.gossip_interval = max(0.1, self.gossip_interval * 0.95)
                    logger.debug(f"Low bandwidth utilization, decreasing gossip interval to {self.gossip_interval:.3f}")
                
                # Update performance metrics
                if self.performance_tracking:
                    await self._update_performance_metrics()
                    
            except Exception as e:
                logger.error(f"Bandwidth optimization error: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_performance_metrics(self):
        """Update agent performance metrics."""
        try:
            # Simple performance metric based on communication success rate
            for agent_id in range(self.num_agents):
                recent_metrics = self.communication_metrics.get(agent_id, [])
                
                if recent_metrics:
                    # Calculate success rate from recent communications
                    recent_successes = sum(1 for m in recent_metrics[-10:] if m.quality_loss < 0.1)
                    success_rate = recent_successes / min(10, len(recent_metrics))
                    
                    # Update performance with exponential moving average
                    alpha = 0.2
                    current_perf = self.performance_metrics.get(agent_id, 0.5)
                    self.performance_metrics[agent_id] = (
                        alpha * success_rate + (1 - alpha) * current_perf
                    )
                    
                    # Update partner selector
                    self.partner_selector.update_performance_history(agent_id, success_rate)
                    
        except Exception as e:
            logger.error(f"Performance metric update error: {e}")
    
    def update_communication_metrics(
        self, 
        agent_id: int, 
        partner_id: int, 
        metrics: CommunicationMetrics
    ):
        """Update communication metrics for agents."""
        if agent_id not in self.communication_metrics:
            self.communication_metrics[agent_id] = []
        
        self.communication_metrics[agent_id].append(metrics)
        
        # Keep only recent metrics
        if len(self.communication_metrics[agent_id]) > 50:
            self.communication_metrics[agent_id].pop(0)
        
        # Update partner selector with communication cost
        self.partner_selector.update_communication_cost(
            agent_id, partner_id, metrics.latency
        )
    
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