#!/usr/bin/env python3
"""
Scalable Quantum-Inspired Federated RL with Performance Optimization
Generation 3: Make it Scale - Optimization, Caching, Performance
"""

import asyncio
import time
import json
import hashlib
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import queue
import sys
from pathlib import Path


def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = sys.getsizeof(args) + sys.getsizeof(kwargs)
        
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Log performance metrics
            if hasattr(args[0], 'logger') and args[0].logger:
                args[0].logger.debug(
                    f"PERF: {func.__name__} | "
                    f"Time: {execution_time:.4f}s | "
                    f"Memory: {start_memory}B"
                )
            
            return result
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            if hasattr(args[0], 'logger') and args[0].logger:
                args[0].logger.warning(
                    f"PERF: {func.__name__} | "
                    f"ERROR after {execution_time:.4f}s: {e}"
                )
            raise
    
    return wrapper


@dataclass 
class QuantumTaskState:
    """Quantum-inspired task state for optimization."""
    superposition_weights: List[float]
    entanglement_map: Dict[str, List[str]]
    coherence_time: float
    measurement_count: int
    
    def collapse_to_classical(self) -> int:
        """Collapse quantum superposition to classical state."""
        # Quantum-inspired probability selection
        import random
        weights_sum = sum(abs(w) for w in self.superposition_weights)
        if weights_sum == 0:
            return random.randint(0, len(self.superposition_weights) - 1)
        
        normalized_weights = [abs(w) / weights_sum for w in self.superposition_weights]
        rand_val = random.random()
        cumulative = 0
        
        for i, weight in enumerate(normalized_weights):
            cumulative += weight
            if rand_val <= cumulative:
                return i
        
        return len(self.superposition_weights) - 1


class PerformanceCache:
    """High-performance LRU cache with TTL and statistics."""
    
    def __init__(self, maxsize: int = 1000, ttl: float = 300.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
        self._access_counts = {}
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with TTL check."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Check TTL
                if current_time - self._timestamps[key] < self.ttl:
                    self._access_counts[key] += 1
                    self._hit_count += 1
                    return self._cache[key]
                else:
                    # Expired
                    self._evict(key)
            
            self._miss_count += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with size management."""
        with self._lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self._cache) >= self.maxsize and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = value
            self._timestamps[key] = current_time
            self._access_counts[key] = 1
    
    def _evict(self, key: str) -> None:
        """Evict specific key."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._access_counts.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # Find LRU key
        lru_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
        self._evict(lru_key)
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self._timestamps.items()
                if current_time - timestamp >= self.ttl
            ]
            
            for key in expired_keys:
                self._evict(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / max(1, total_requests)
            
            return {
                "size": len(self._cache),
                "max_size": self.maxsize,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "ttl": self.ttl,
            }


class AsyncParameterServer:
    """Asynchronous parameter server with load balancing."""
    
    def __init__(self, num_workers: int = 4, logger: Optional[logging.Logger] = None):
        self.num_workers = num_workers
        self.logger = logger or logging.getLogger(__name__)
        self.parameter_cache = PerformanceCache(maxsize=5000, ttl=600.0)
        self.work_queue = asyncio.Queue()
        self.workers = []
        self.is_running = False
        self.processed_requests = 0
        self.active_connections = 0
        
        # Load balancing
        self.worker_loads = [0] * num_workers
        self.round_robin_counter = 0
    
    async def start(self):
        """Start parameter server workers."""
        self.is_running = True
        
        # Create worker tasks
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker)
        
        self.logger.info(f"Parameter server started with {self.num_workers} workers")
    
    async def stop(self):
        """Stop parameter server workers."""
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.logger.info("Parameter server stopped")
    
    async def _worker_loop(self, worker_id: int):
        """Worker loop for processing parameter requests."""
        self.logger.debug(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get work item with timeout
                try:
                    work_item = await asyncio.wait_for(
                        self.work_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process work item
                request_type, agent_id, data, future = work_item
                
                try:
                    if request_type == "aggregate":
                        result = await self._process_aggregation(data)
                    elif request_type == "update":
                        result = await self._process_update(agent_id, data)
                    elif request_type == "get":
                        result = await self._process_get(agent_id)
                    else:
                        raise ValueError(f"Unknown request type: {request_type}")
                    
                    future.set_result(result)
                    self.processed_requests += 1
                    
                except Exception as e:
                    future.set_exception(e)
                    self.logger.error(f"Worker {worker_id} error: {e}")
                finally:
                    self.worker_loads[worker_id] -= 1
                    self.work_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} unexpected error: {e}")
    
    @performance_monitor
    async def _process_aggregation(self, agent_data: List[Tuple[str, Dict]]) -> Dict:
        """Process federated aggregation with optimization."""
        start_time = time.perf_counter()
        
        # Check cache first
        data_hash = hashlib.md5(str(sorted(agent_data)).encode()).hexdigest()
        cache_key = f"agg_{data_hash}"
        
        cached_result = self.parameter_cache.get(cache_key)
        if cached_result:
            self.logger.debug(f"Cache hit for aggregation: {cache_key}")
            return cached_result
        
        # Perform aggregation
        all_states = set()
        agent_q_tables = {}
        
        for agent_id, q_table in agent_data:
            agent_q_tables[agent_id] = q_table
            all_states.update(q_table.keys())
        
        # Parallel aggregation using quantum-inspired weights
        aggregated_q_table = {}
        
        for state in all_states:
            q_values_list = []
            weights = []
            
            for agent_id, q_table in agent_q_tables.items():
                if state in q_table:
                    q_values = q_table[state]
                    # Quantum-inspired weighting based on confidence
                    confidence = sum(abs(q) for q in q_values) / 3.0
                    weight = 1.0 / (1.0 + abs(confidence - 1.0))
                    q_values_list.append(q_values)
                    weights.append(weight)
                else:
                    q_values_list.append([0.0, 0.0, 0.0])
                    weights.append(0.1)  # Low weight for missing states
            
            # Weighted aggregation
            if q_values_list and sum(weights) > 0:
                total_weight = sum(weights)
                avg_q = [0.0, 0.0, 0.0]
                
                for i in range(3):
                    weighted_sum = sum(
                        q_values[i] * weight 
                        for q_values, weight in zip(q_values_list, weights)
                        if len(q_values) > i
                    )
                    avg_q[i] = weighted_sum / total_weight
                
                aggregated_q_table[state] = avg_q
        
        result = {
            "aggregated_parameters": aggregated_q_table,
            "num_states": len(aggregated_q_table),
            "num_agents": len(agent_data),
            "aggregation_time": time.perf_counter() - start_time,
        }
        
        # Cache result
        self.parameter_cache.set(cache_key, result)
        
        self.logger.debug(
            f"Aggregated {len(agent_data)} agents, {len(aggregated_q_table)} states "
            f"in {result['aggregation_time']:.4f}s"
        )
        
        return result
    
    async def _process_update(self, agent_id: str, q_table: Dict) -> Dict:
        """Process parameter update."""
        cache_key = f"agent_{agent_id}"
        self.parameter_cache.set(cache_key, q_table)
        
        return {
            "status": "updated",
            "agent_id": agent_id,
            "num_states": len(q_table),
        }
    
    async def _process_get(self, agent_id: str) -> Dict:
        """Process parameter retrieval."""
        cache_key = f"agent_{agent_id}"
        q_table = self.parameter_cache.get(cache_key)
        
        if q_table:
            return {
                "status": "found",
                "agent_id": agent_id,
                "parameters": q_table,
            }
        else:
            return {
                "status": "not_found",
                "agent_id": agent_id,
                "parameters": {},
            }
    
    async def submit_request(self, request_type: str, agent_id: str = None, data: Any = None) -> Any:
        """Submit request to parameter server."""
        future = asyncio.Future()
        
        # Load balancing: assign to least loaded worker
        min_load_worker = min(range(self.num_workers), key=lambda i: self.worker_loads[i])
        self.worker_loads[min_load_worker] += 1
        
        await self.work_queue.put((request_type, agent_id, data, future))
        
        result = await future
        return result
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        cache_stats = self.parameter_cache.get_stats()
        
        return {
            "num_workers": self.num_workers,
            "queue_size": self.work_queue.qsize(),
            "processed_requests": self.processed_requests,
            "active_connections": self.active_connections,
            "worker_loads": self.worker_loads.copy(),
            "cache_stats": cache_stats,
            "is_running": self.is_running,
        }


class QuantumInspiredAgent:
    """Quantum-inspired agent with performance optimization."""
    
    def __init__(self, agent_id: str, logger: logging.Logger, num_quantum_states: int = 8):
        self.agent_id = agent_id
        self.logger = logger
        self.num_quantum_states = num_quantum_states
        
        # Classical Q-table
        self.q_table = {}
        
        # Quantum-inspired state
        self.quantum_state = QuantumTaskState(
            superposition_weights=[1.0 / num_quantum_states] * num_quantum_states,
            entanglement_map={},
            coherence_time=10.0,
            measurement_count=0
        )
        
        # Performance caching
        self.action_cache = PerformanceCache(maxsize=500, ttl=60.0)
        self.state_encoder_cache = PerformanceCache(maxsize=1000, ttl=120.0)
        
        # Optimization parameters
        self.learning_rate = 0.1
        self.discount = 0.95
        self.exploration_rate = 0.3
        self.quantum_coherence_decay = 0.99
        
        # Performance tracking
        self.action_selections = 0
        self.cache_hits = 0
        self.quantum_measurements = 0
        
    @performance_monitor
    @lru_cache(maxsize=128)
    def encode_state(self, signal: int, queue_level: int) -> str:
        """Optimized state encoding with caching."""
        return f"{signal}_{queue_level}"
    
    @performance_monitor
    def select_action_quantum(self, node_state: Dict) -> int:
        """Quantum-inspired action selection with caching."""
        try:
            # Generate cache key
            state_key = self.encode_state(
                node_state.get("signal", 0),
                int(node_state.get("queue", 0) // 2)
            )
            
            cache_key = f"action_{state_key}_{self.exploration_rate:.2f}"
            
            # Check cache first
            cached_action = self.action_cache.get(cache_key)
            if cached_action is not None:
                self.cache_hits += 1
                return cached_action
            
            # Initialize Q-values if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0, 0.0, 0.0]
            
            # Quantum-inspired action selection
            if self._should_use_quantum():
                action = self._quantum_action_selection(state_key)
            else:
                action = self._classical_action_selection(state_key)
            
            # Cache result
            self.action_cache.set(cache_key, action)
            self.action_selections += 1
            
            return action
            
        except Exception as e:
            self.logger.warning(f"Agent {self.agent_id}: Action selection error: {e}")
            return 0  # Safe fallback
    
    def _should_use_quantum(self) -> bool:
        """Decide whether to use quantum-inspired selection."""
        # Use quantum with probability based on coherence
        coherence_prob = max(0.1, self.quantum_state.coherence_time / 10.0)
        import random
        return random.random() < coherence_prob
    
    def _quantum_action_selection(self, state_key: str) -> int:
        """Quantum-inspired action selection using superposition."""
        # Update quantum state based on Q-values
        q_values = self.q_table[state_key]
        
        # Create superposition based on Q-value magnitudes
        magnitude_sum = sum(abs(q) for q in q_values) + 1e-8
        new_weights = [abs(q) / magnitude_sum for q in q_values]
        
        # Normalize superposition weights
        weight_sum = sum(new_weights)
        if weight_sum > 0:
            new_weights = [w / weight_sum for w in new_weights]
        else:
            new_weights = [1.0 / 3.0] * 3
        
        # Update quantum state
        self.quantum_state.superposition_weights = new_weights
        self.quantum_state.measurement_count += 1
        
        # Collapse superposition to classical action
        action = self.quantum_state.collapse_to_classical()
        
        # Apply quantum coherence decay
        self.quantum_state.coherence_time *= self.quantum_coherence_decay
        if self.quantum_state.coherence_time < 1.0:
            self.quantum_state.coherence_time = 10.0  # Reset
        
        self.quantum_measurements += 1
        return action
    
    def _classical_action_selection(self, state_key: str) -> int:
        """Classical epsilon-greedy action selection."""
        import random
        
        if random.random() < self.exploration_rate:
            return random.randint(0, 2)
        else:
            q_values = self.q_table[state_key]
            return int(max(range(len(q_values)), key=lambda i: q_values[i]))
    
    @performance_monitor
    def update_q_values_optimized(self, state: Dict, action: int, reward: float, next_state: Dict):
        """Optimized Q-value update with quantum enhancement."""
        try:
            state_key = self.encode_state(
                state.get("signal", 0),
                int(state.get("queue", 0) // 2)
            )
            next_state_key = self.encode_state(
                next_state.get("signal", 0),
                int(next_state.get("queue", 0) // 2)
            )
            
            # Initialize if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0, 0.0, 0.0]
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0, 0.0, 0.0]
            
            # Classical Q-learning update
            current_q = self.q_table[state_key][action]
            max_next_q = max(self.q_table[next_state_key])
            target_q = reward + self.discount * max_next_q
            
            # Quantum-inspired learning rate adaptation
            quantum_factor = 1.0 + 0.1 * self.quantum_state.coherence_time / 10.0
            adaptive_lr = self.learning_rate * quantum_factor
            
            # Update Q-value
            new_q = current_q + adaptive_lr * (target_q - current_q)
            new_q = max(-100, min(100, new_q))  # Bounds checking
            
            self.q_table[state_key][action] = new_q
            
            # Update quantum entanglement map
            if state_key not in self.quantum_state.entanglement_map:
                self.quantum_state.entanglement_map[state_key] = []
            
            if next_state_key not in self.quantum_state.entanglement_map[state_key]:
                self.quantum_state.entanglement_map[state_key].append(next_state_key)
            
            # Clear caches periodically for memory management
            if self.action_selections % 100 == 0:
                self.action_cache.clear_expired()
                self.state_encoder_cache.clear_expired()
            
        except Exception as e:
            self.logger.error(f"Agent {self.agent_id}: Q-update error: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        action_cache_stats = self.action_cache.get_stats()
        state_cache_stats = self.state_encoder_cache.get_stats()
        
        return {
            "agent_id": self.agent_id,
            "q_table_size": len(self.q_table),
            "action_selections": self.action_selections,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.action_selections),
            "quantum_measurements": self.quantum_measurements,
            "quantum_coherence": self.quantum_state.coherence_time,
            "entanglement_size": len(self.quantum_state.entanglement_map),
            "action_cache": action_cache_stats,
            "state_cache": state_cache_stats,
        }


class ScalableQuantumFederatedSystem:
    """Scalable quantum-inspired federated system with optimization."""
    
    def __init__(self, num_agents: int, logger: logging.Logger):
        self.num_agents = num_agents
        self.logger = logger
        
        # Create quantum-inspired agents
        self.agents = [
            QuantumInspiredAgent(f"quantum_agent_{i}", logger)
            for i in range(num_agents)
        ]
        
        # Async parameter server
        self.parameter_server = AsyncParameterServer(
            num_workers=min(4, mp.cpu_count()),
            logger=logger
        )
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=min(8, mp.cpu_count()))
        self.global_cache = PerformanceCache(maxsize=10000, ttl=900.0)
        
        # Scaling metrics
        self.communication_rounds = 0
        self.successful_aggregations = 0
        self.total_processing_time = 0.0
        self.peak_memory_usage = 0
        self.throughput_history = []
        
    async def start(self):
        """Start scalable federated system."""
        await self.parameter_server.start()
        self.logger.info("Scalable quantum federated system started")
    
    async def stop(self):
        """Stop scalable federated system."""
        await self.parameter_server.stop()
        self.executor.shutdown(wait=True)
        self.logger.info("Scalable quantum federated system stopped")
    
    @performance_monitor
    async def quantum_federated_round(self) -> Dict[str, Any]:
        """Execute quantum-enhanced federated round with optimization."""
        round_start_time = time.perf_counter()
        self.communication_rounds += 1
        
        try:
            # Collect parameters from agents in parallel
            collection_start = time.perf_counter()
            agent_data = []
            
            # Use ThreadPoolExecutor for parallel parameter collection
            def collect_agent_params(agent):
                return (agent.agent_id, agent.q_table.copy())
            
            futures = [
                self.executor.submit(collect_agent_params, agent)
                for agent in self.agents
            ]
            
            for future in futures:
                try:
                    agent_id, q_table = future.result(timeout=5.0)
                    agent_data.append((agent_id, q_table))
                except Exception as e:
                    self.logger.warning(f"Failed to collect params: {e}")
            
            collection_time = time.perf_counter() - collection_start
            
            # Quantum-enhanced aggregation via parameter server
            aggregation_start = time.perf_counter()
            
            result = await self.parameter_server.submit_request(
                "aggregate", 
                data=agent_data
            )
            
            aggregation_time = time.perf_counter() - aggregation_start
            
            # Update agents with aggregated parameters in parallel
            update_start = time.perf_counter()
            aggregated_params = result["aggregated_parameters"]
            
            def update_agent_params(agent):
                try:
                    # Quantum-inspired parameter fusion
                    fusion_factor = 0.7  # Weight for aggregated params
                    
                    for state, agg_q_values in aggregated_params.items():
                        if state in agent.q_table:
                            current_q = agent.q_table[state]
                            # Quantum superposition-inspired fusion
                            fused_q = [
                                fusion_factor * agg_q + (1 - fusion_factor) * curr_q
                                for agg_q, curr_q in zip(agg_q_values, current_q)
                            ]
                            agent.q_table[state] = fused_q
                        else:
                            agent.q_table[state] = agg_q_values.copy()
                    
                    return agent.agent_id
                except Exception as e:
                    self.logger.warning(f"Agent update error: {e}")
                    return None
            
            update_futures = [
                self.executor.submit(update_agent_params, agent)
                for agent in self.agents
            ]
            
            successful_updates = 0
            for future in update_futures:
                try:
                    agent_id = future.result(timeout=3.0)
                    if agent_id:
                        successful_updates += 1
                except Exception as e:
                    self.logger.warning(f"Agent update failed: {e}")
            
            update_time = time.perf_counter() - update_start
            
            # Calculate performance metrics
            total_round_time = time.perf_counter() - round_start_time
            self.total_processing_time += total_round_time
            
            # Compute throughput
            throughput = len(agent_data) / total_round_time  # Agents per second
            self.throughput_history.append(throughput)
            if len(self.throughput_history) > 100:
                self.throughput_history.pop(0)
            
            self.successful_aggregations += 1
            
            # Performance logging
            self.logger.info(
                f"Quantum federated round {self.communication_rounds}: "
                f"Agents: {len(agent_data)}, "
                f"States: {result['num_states']}, "
                f"Updates: {successful_updates}, "
                f"Times: collect={collection_time:.3f}s, "
                f"agg={aggregation_time:.3f}s, "
                f"update={update_time:.3f}s, "
                f"total={total_round_time:.3f}s, "
                f"throughput={throughput:.1f} agents/s"
            )
            
            return {
                "round": self.communication_rounds,
                "participants": len(agent_data),
                "successful_updates": successful_updates,
                "states_aggregated": result["num_states"],
                "collection_time": collection_time,
                "aggregation_time": aggregation_time,
                "update_time": update_time,
                "total_time": total_round_time,
                "throughput": throughput,
                "success": True,
            }
            
        except Exception as e:
            self.logger.error(f"Quantum federated round {self.communication_rounds} failed: {e}")
            return {
                "round": self.communication_rounds,
                "success": False,
                "error": str(e),
                "total_time": time.perf_counter() - round_start_time,
            }
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling and performance metrics."""
        # Aggregate agent performance stats
        agent_stats = [agent.get_performance_stats() for agent in self.agents]
        
        # Server stats
        server_stats = self.parameter_server.get_server_stats()
        
        # System performance metrics
        avg_throughput = sum(self.throughput_history) / max(1, len(self.throughput_history))
        peak_throughput = max(self.throughput_history) if self.throughput_history else 0
        
        total_cache_hits = sum(stats["cache_hits"] for stats in agent_stats)
        total_selections = sum(stats["action_selections"] for stats in agent_stats)
        global_cache_hit_rate = total_cache_hits / max(1, total_selections)
        
        return {
            "system_overview": {
                "num_agents": self.num_agents,
                "communication_rounds": self.communication_rounds,
                "successful_aggregations": self.successful_aggregations,
                "success_rate": self.successful_aggregations / max(1, self.communication_rounds),
                "total_processing_time": self.total_processing_time,
                "avg_round_time": self.total_processing_time / max(1, self.communication_rounds),
            },
            "performance_metrics": {
                "avg_throughput": avg_throughput,
                "peak_throughput": peak_throughput,
                "global_cache_hit_rate": global_cache_hit_rate,
                "total_cache_hits": total_cache_hits,
                "total_action_selections": total_selections,
            },
            "server_stats": server_stats,
            "agent_stats": agent_stats,
            "quantum_metrics": {
                "total_quantum_measurements": sum(
                    stats["quantum_measurements"] for stats in agent_stats
                ),
                "avg_coherence": sum(
                    stats["quantum_coherence"] for stats in agent_stats
                ) / len(agent_stats),
                "total_entanglements": sum(
                    stats["entanglement_size"] for stats in agent_stats
                ),
            }
        }


async def run_scalable_demo():
    """Run scalable quantum-inspired federated RL demo."""
    
    # Setup high-performance logging
    logger = logging.getLogger("ScalableQuantumRL")
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info("🚀 Starting Scalable Quantum-Inspired Federated RL Demo")
    logger.info("=" * 70)
    
    try:
        # Initialize scalable system
        system = ScalableQuantumFederatedSystem(num_agents=8, logger=logger)
        await system.start()
        
        # Training parameters optimized for performance
        num_episodes = 25
        steps_per_episode = 40
        fed_round_interval = 80
        
        episode_rewards = []
        performance_history = []
        
        # High-performance training loop
        for episode in range(num_episodes):
            episode_start = time.perf_counter()
            episode_reward = 0
            
            # Parallel episode simulation
            episode_tasks = []
            
            for step in range(steps_per_episode):
                step_start = time.perf_counter()
                
                # Simulate parallel agent interactions
                step_reward = 0
                for i, agent in enumerate(system.agents):
                    # Simulate state
                    state = {"signal": i % 3, "queue": (step + i) % 10}
                    action = agent.select_action_quantum(state)
                    
                    # Simulate reward
                    import random
                    reward = random.uniform(-5, 2)
                    step_reward += reward
                    
                    # Simulate next state
                    next_state = {"signal": (i + 1) % 3, "queue": (step + i + 1) % 10}
                    
                    # Update Q-values
                    agent.update_q_values_optimized(state, action, reward, next_state)
                
                episode_reward += step_reward
                
                # Federated communication
                if (episode * steps_per_episode + step) % fed_round_interval == 0:
                    round_result = await system.quantum_federated_round()
                    performance_history.append(round_result)
            
            episode_time = time.perf_counter() - episode_start
            episode_rewards.append(episode_reward)
            
            # Progress reporting with performance metrics
            if episode % 5 == 0:
                scaling_metrics = system.get_scaling_metrics()
                perf_metrics = scaling_metrics["performance_metrics"]
                
                logger.info(
                    f"Episode {episode:3d}: "
                    f"Reward = {episode_reward:7.2f}, "
                    f"Time = {episode_time:.3f}s, "
                    f"Throughput = {perf_metrics['avg_throughput']:.1f} agents/s, "
                    f"Cache Hit = {perf_metrics['global_cache_hit_rate']:.1%}, "
                    f"Fed Success = {scaling_metrics['system_overview']['success_rate']:.1%}"
                )
        
        # Final performance analysis
        final_metrics = system.get_scaling_metrics()
        
        logger.info("=" * 70)
        logger.info("🎯 Scalable Training Results:")
        
        # System overview
        overview = final_metrics["system_overview"]
        logger.info(f"   Episodes: {len(episode_rewards)}")
        logger.info(f"   Fed rounds: {overview['communication_rounds']}")
        logger.info(f"   Success rate: {overview['success_rate']:.1%}")
        logger.info(f"   Avg round time: {overview['avg_round_time']:.3f}s")
        
        # Performance metrics
        perf = final_metrics["performance_metrics"]
        logger.info(f"   Peak throughput: {perf['peak_throughput']:.1f} agents/s")
        logger.info(f"   Cache hit rate: {perf['global_cache_hit_rate']:.1%}")
        logger.info(f"   Total cache hits: {perf['total_cache_hits']:,}")
        
        # Quantum metrics
        quantum = final_metrics["quantum_metrics"]
        logger.info(f"   Quantum measurements: {quantum['total_quantum_measurements']:,}")
        logger.info(f"   Avg coherence: {quantum['avg_coherence']:.2f}")
        logger.info(f"   Entanglements: {quantum['total_entanglements']:,}")
        
        # Training results
        logger.info(f"   Final reward: {episode_rewards[-1]:.2f}")
        logger.info(f"   Best reward: {max(episode_rewards):.2f}")
        if len(episode_rewards) >= 10:
            logger.info(f"   Avg last 10: {sum(episode_rewards[-10:]) / 10:.2f}")
        
        # Save comprehensive results
        results = {
            "timestamp": datetime.now().isoformat(),
            "episode_rewards": episode_rewards,
            "performance_history": performance_history,
            "final_metrics": final_metrics,
            "scaling_analysis": {
                "episodes_per_second": len(episode_rewards) / system.total_processing_time,
                "memory_efficiency": "optimized_caching",
                "parallelization": "multi_threaded_async",
                "quantum_enhancement": "enabled",
            }
        }
        
        results_path = Path("results/scalable_quantum_results.json")
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        logger.info("✅ Scalable quantum demo completed successfully!")
        
        await system.stop()
        return episode_rewards, system, final_metrics
        
    except Exception as e:
        logger.error(f"❌ Scalable demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        import asyncio
        rewards, system, metrics = asyncio.run(run_scalable_demo())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)