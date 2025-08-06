"""
Auto-scaling and load balancing for quantum task planner.

Implements intelligent scaling mechanisms:
- Adaptive resource scaling based on workload patterns
- Load balancing across quantum computation nodes
- Circuit breaker patterns for fault tolerance
- Auto-scaling triggers and policies
- Horizontal and vertical scaling coordination
"""

import asyncio
import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable, Union, Tuple
from collections import defaultdict, deque
from enum import Enum
import statistics
import numpy as np

from .core import QuantumTask, TaskState
from .exceptions import QuantumPlannerError
from .concurrency import ConcurrencyManager
from .performance import PerformanceManager


class ScalingDirection(Enum):
    """Scaling direction indicators."""
    UP = "up"
    DOWN = "down" 
    STABLE = "stable"


class ScalingType(Enum):
    """Types of scaling operations."""
    HORIZONTAL = "horizontal"  # More instances
    VERTICAL = "vertical"      # More resources per instance
    HYBRID = "hybrid"          # Both horizontal and vertical


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_utilization: float
    memory_utilization: float
    queue_length: int
    throughput: float
    latency_p95: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingDecision:
    """Represents a scaling decision."""
    direction: ScalingDirection
    scaling_type: ScalingType
    magnitude: float  # 0.0-1.0 indicating strength of scaling
    target_instances: int
    target_resources: Dict[str, float]
    reasoning: str
    confidence: float  # 0.0-1.0 confidence in decision


class LoadBalancer:
    """
    Advanced load balancer for quantum task distribution.
    
    Implements multiple load balancing strategies with circuit breaker protection.
    """
    
    def __init__(self, circuit_breaker_threshold: float = 0.5):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        # Load balancing strategies
        self.strategies = {
            "round_robin": self._round_robin,
            "least_connections": self._least_connections,
            "weighted_random": self._weighted_random,
            "quantum_aware": self._quantum_aware,
        }
        
        self.current_strategy = "quantum_aware"
        self.round_robin_index = 0
        
        # Circuit breaker state
        self.circuit_states: Dict[str, str] = {}  # "closed", "open", "half_open"
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_failure_time: Dict[str, float] = {}
        
        # Performance tracking
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.success_rates: Dict[str, float] = defaultdict(lambda: 1.0)
    
    def register_node(
        self, 
        node_id: str, 
        capacity: Dict[str, float],
        endpoint: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a computation node."""
        self.nodes[node_id] = {
            "capacity": capacity,
            "endpoint": endpoint,
            "metadata": metadata or {},
            "current_load": defaultdict(float),
            "health_score": 1.0,
            "last_heartbeat": time.time()
        }
        
        self.circuit_states[node_id] = "closed"
        self.failure_counts[node_id] = 0
    
    def unregister_node(self, node_id: str):
        """Unregister a computation node."""
        self.nodes.pop(node_id, None)
        self.circuit_states.pop(node_id, None)
        self.failure_counts.pop(node_id, None)
        self.request_counts.pop(node_id, None)
        self.response_times.pop(node_id, None)
        self.success_rates.pop(node_id, None)
    
    def select_node(
        self, 
        task_requirements: Dict[str, float],
        quantum_properties: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Select best node for task execution."""
        available_nodes = self._get_available_nodes(task_requirements)
        
        if not available_nodes:
            return None
        
        # Apply current load balancing strategy
        strategy_func = self.strategies.get(self.current_strategy, self._round_robin)
        selected_node = strategy_func(available_nodes, task_requirements, quantum_properties)
        
        return selected_node
    
    def _get_available_nodes(self, task_requirements: Dict[str, float]) -> List[str]:
        """Get nodes that can handle the task requirements."""
        available = []
        
        for node_id, node_info in self.nodes.items():
            # Check circuit breaker
            if self.circuit_states[node_id] == "open":
                # Check if enough time has passed to try half-open
                if time.time() - self.last_failure_time.get(node_id, 0) > 30.0:
                    self.circuit_states[node_id] = "half_open"
                else:
                    continue
            
            # Check resource capacity
            can_handle = True
            for resource, required in task_requirements.items():
                capacity = node_info["capacity"].get(resource, 0.0)
                current_load = node_info["current_load"].get(resource, 0.0)
                
                if capacity - current_load < required:
                    can_handle = False
                    break
            
            if can_handle:
                available.append(node_id)
        
        return available
    
    def _round_robin(
        self, 
        available_nodes: List[str],
        task_requirements: Dict[str, float],
        quantum_properties: Optional[Dict[str, Any]]
    ) -> str:
        """Round robin load balancing."""
        if not available_nodes:
            return None
        
        # Simple round robin
        selected = available_nodes[self.round_robin_index % len(available_nodes)]
        self.round_robin_index += 1
        
        return selected
    
    def _least_connections(
        self, 
        available_nodes: List[str],
        task_requirements: Dict[str, float],
        quantum_properties: Optional[Dict[str, Any]]
    ) -> str:
        """Least connections load balancing."""
        if not available_nodes:
            return None
        
        # Select node with least active connections
        min_connections = float('inf')
        best_node = available_nodes[0]
        
        for node_id in available_nodes:
            connections = self.request_counts.get(node_id, 0)
            if connections < min_connections:
                min_connections = connections
                best_node = node_id
        
        return best_node
    
    def _weighted_random(
        self, 
        available_nodes: List[str],
        task_requirements: Dict[str, float],
        quantum_properties: Optional[Dict[str, Any]]
    ) -> str:
        """Weighted random load balancing based on node health."""
        if not available_nodes:
            return None
        
        # Calculate weights based on health scores
        weights = []
        for node_id in available_nodes:
            health_score = self.nodes[node_id]["health_score"]
            success_rate = self.success_rates.get(node_id, 1.0)
            weight = health_score * success_rate
            weights.append(weight)
        
        # Weighted random selection
        if sum(weights) > 0:
            import random
            selected_idx = random.choices(range(len(available_nodes)), weights=weights)[0]
            return available_nodes[selected_idx]
        else:
            return available_nodes[0]
    
    def _quantum_aware(
        self, 
        available_nodes: List[str],
        task_requirements: Dict[str, float],
        quantum_properties: Optional[Dict[str, Any]]
    ) -> str:
        """Quantum-aware load balancing considering coherence and entanglement."""
        if not available_nodes:
            return None
        
        best_node = None
        best_score = float('-inf')
        
        for node_id in available_nodes:
            node_info = self.nodes[node_id]
            
            # Base score from resource efficiency
            resource_efficiency = self._calculate_resource_efficiency(node_id, task_requirements)
            
            # Health and performance factors
            health_score = node_info["health_score"]
            success_rate = self.success_rates.get(node_id, 1.0)
            
            # Response time factor (lower is better)
            avg_response_time = statistics.mean(self.response_times.get(node_id, [0.1]))
            response_factor = 1.0 / max(avg_response_time, 0.01)
            
            # Quantum coherence bonus
            coherence_bonus = 0.0
            if quantum_properties:
                # Prefer nodes that maintain quantum coherence
                coherence = quantum_properties.get("coherence", 0.5)
                coherence_bonus = coherence * 0.2
            
            # Combined score
            score = (
                0.4 * resource_efficiency +
                0.3 * health_score +
                0.2 * success_rate +
                0.1 * min(response_factor, 2.0) +
                coherence_bonus
            )
            
            if score > best_score:
                best_score = score
                best_node = node_id
        
        return best_node or available_nodes[0]
    
    def _calculate_resource_efficiency(
        self, 
        node_id: str, 
        task_requirements: Dict[str, float]
    ) -> float:
        """Calculate resource utilization efficiency for node."""
        node_info = self.nodes[node_id]
        
        total_efficiency = 0.0
        resource_count = 0
        
        for resource, required in task_requirements.items():
            if required > 0:
                capacity = node_info["capacity"].get(resource, 0.0)
                current_load = node_info["current_load"].get(resource, 0.0)
                
                if capacity > 0:
                    # Efficiency: how well this fits available capacity
                    available = capacity - current_load
                    efficiency = min(1.0, available / required) if required > 0 else 1.0
                    
                    # Prefer moderate utilization (not too empty, not too full)
                    utilization_after = (current_load + required) / capacity
                    utilization_preference = 1.0 - abs(0.7 - utilization_after)  # Prefer 70% utilization
                    
                    combined_efficiency = 0.7 * efficiency + 0.3 * utilization_preference
                    total_efficiency += combined_efficiency
                    resource_count += 1
        
        return total_efficiency / max(resource_count, 1)
    
    def record_request_start(self, node_id: str, task_id: str):
        """Record start of request to node."""
        if node_id in self.nodes:
            self.request_counts[node_id] += 1
    
    def record_request_end(
        self, 
        node_id: str, 
        task_id: str, 
        success: bool, 
        response_time: float
    ):
        """Record end of request and update metrics."""
        if node_id not in self.nodes:
            return
        
        # Update request count
        self.request_counts[node_id] = max(0, self.request_counts[node_id] - 1)
        
        # Update response time
        self.response_times[node_id].append(response_time)
        
        # Update success rate
        old_rate = self.success_rates[node_id]
        alpha = 0.1  # Learning rate
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * old_rate
        self.success_rates[node_id] = new_rate
        
        # Update circuit breaker
        if success:
            self.failure_counts[node_id] = max(0, self.failure_counts[node_id] - 1)
            if self.circuit_states[node_id] == "half_open":
                self.circuit_states[node_id] = "closed"
        else:
            self.failure_counts[node_id] += 1
            self.last_failure_time[node_id] = time.time()
            
            # Check if circuit should open
            if self.failure_counts[node_id] >= 5:  # Threshold
                self.circuit_states[node_id] = "open"
    
    def update_node_load(self, node_id: str, resource_loads: Dict[str, float]):
        """Update current resource load for node."""
        if node_id in self.nodes:
            self.nodes[node_id]["current_load"] = resource_loads
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across nodes."""
        distribution = {}
        
        for node_id, node_info in self.nodes.items():
            total_requests = sum(self.request_counts.values())
            node_requests = self.request_counts.get(node_id, 0)
            
            distribution[node_id] = {
                "request_percentage": (node_requests / max(total_requests, 1)) * 100,
                "active_requests": node_requests,
                "success_rate": self.success_rates.get(node_id, 1.0),
                "avg_response_time": statistics.mean(
                    self.response_times.get(node_id, [0.0])
                ) if self.response_times.get(node_id) else 0.0,
                "circuit_state": self.circuit_states.get(node_id, "closed"),
                "health_score": node_info["health_score"]
            }
        
        return distribution


class AutoScaler:
    """
    Intelligent auto-scaler for quantum computation resources.
    
    Monitors system metrics and automatically adjusts resource allocation.
    """
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 20,
        scale_up_threshold: float = 0.75,
        scale_down_threshold: float = 0.25,
        cooldown_period: float = 300.0  # 5 minutes
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        # Current state
        self.current_instances = min_instances
        self.current_resources: Dict[str, float] = {"cpu": 1.0, "memory": 1.0}
        
        # Scaling history
        self.scaling_history: deque = deque(maxlen=100)
        self.last_scaling_time = 0.0
        
        # Metrics collection
        self.metrics_history: deque = deque(maxlen=1000)
        self.prediction_model = None
        
        # Scaling policies
        self.scaling_policies: List[Callable] = [
            self._resource_based_policy,
            self._queue_based_policy,
            self._latency_based_policy,
            self._predictive_policy
        ]
        
        # Auto-tuning parameters
        self.auto_tune_enabled = True
        self.policy_weights: Dict[str, float] = {
            "resource_based": 0.4,
            "queue_based": 0.3,
            "latency_based": 0.2,
            "predictive": 0.1
        }
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics sample for scaling decisions."""
        self.metrics_history.append(metrics)
        
        # Auto-tune policy weights based on recent performance
        if self.auto_tune_enabled and len(self.metrics_history) > 50:
            self._auto_tune_policies()
    
    def should_scale(self) -> Optional[ScalingDecision]:
        """Determine if scaling is needed and return decision."""
        if not self.metrics_history:
            return None
        
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.cooldown_period:
            return None
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
        
        # Apply scaling policies
        policy_decisions = []
        
        for policy in self.scaling_policies:
            decision = policy(recent_metrics)
            if decision:
                policy_decisions.append(decision)
        
        # Combine policy decisions
        final_decision = self._combine_decisions(policy_decisions)
        
        return final_decision
    
    def _resource_based_policy(self, metrics: List[ScalingMetrics]) -> Optional[ScalingDecision]:
        """Resource utilization based scaling policy."""
        if not metrics:
            return None
        
        # Average resource utilization
        avg_cpu = statistics.mean(m.cpu_utilization for m in metrics)
        avg_memory = statistics.mean(m.memory_utilization for m in metrics)
        max_utilization = max(avg_cpu, avg_memory)
        
        if max_utilization > self.scale_up_threshold:
            # Scale up
            magnitude = min(1.0, (max_utilization - self.scale_up_threshold) / 0.25)
            target_instances = min(
                self.max_instances,
                math.ceil(self.current_instances * (1 + magnitude * 0.5))
            )
            
            return ScalingDecision(
                direction=ScalingDirection.UP,
                scaling_type=ScalingType.HORIZONTAL,
                magnitude=magnitude,
                target_instances=target_instances,
                target_resources=dict(self.current_resources),
                reasoning=f"High resource utilization: CPU={avg_cpu:.2f}, Memory={avg_memory:.2f}",
                confidence=magnitude
            )
        
        elif max_utilization < self.scale_down_threshold and self.current_instances > self.min_instances:
            # Scale down
            magnitude = min(1.0, (self.scale_down_threshold - max_utilization) / 0.25)
            target_instances = max(
                self.min_instances,
                math.floor(self.current_instances * (1 - magnitude * 0.3))
            )
            
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                scaling_type=ScalingType.HORIZONTAL,
                magnitude=magnitude,
                target_instances=target_instances,
                target_resources=dict(self.current_resources),
                reasoning=f"Low resource utilization: CPU={avg_cpu:.2f}, Memory={avg_memory:.2f}",
                confidence=magnitude
            )
        
        return None
    
    def _queue_based_policy(self, metrics: List[ScalingMetrics]) -> Optional[ScalingDecision]:
        """Queue length based scaling policy."""
        if not metrics:
            return None
        
        avg_queue_length = statistics.mean(m.queue_length for m in metrics)
        
        # Dynamic threshold based on current instances
        queue_threshold_per_instance = 10
        scale_up_queue_threshold = queue_threshold_per_instance * self.current_instances
        scale_down_queue_threshold = queue_threshold_per_instance * self.current_instances * 0.2
        
        if avg_queue_length > scale_up_queue_threshold:
            # Scale up based on queue pressure
            magnitude = min(1.0, avg_queue_length / scale_up_queue_threshold - 1.0)
            target_instances = min(
                self.max_instances,
                math.ceil(avg_queue_length / queue_threshold_per_instance)
            )
            
            return ScalingDecision(
                direction=ScalingDirection.UP,
                scaling_type=ScalingType.HORIZONTAL,
                magnitude=magnitude,
                target_instances=target_instances,
                target_resources=dict(self.current_resources),
                reasoning=f"High queue length: {avg_queue_length:.1f} (threshold: {scale_up_queue_threshold})",
                confidence=0.8
            )
        
        elif avg_queue_length < scale_down_queue_threshold and self.current_instances > self.min_instances:
            # Scale down due to low queue
            magnitude = min(0.5, (scale_down_queue_threshold - avg_queue_length) / scale_down_queue_threshold)
            target_instances = max(
                self.min_instances,
                max(1, math.ceil(avg_queue_length / queue_threshold_per_instance))
            )
            
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                scaling_type=ScalingType.HORIZONTAL,
                magnitude=magnitude,
                target_instances=target_instances,
                target_resources=dict(self.current_resources),
                reasoning=f"Low queue length: {avg_queue_length:.1f}",
                confidence=0.6
            )
        
        return None
    
    def _latency_based_policy(self, metrics: List[ScalingMetrics]) -> Optional[ScalingDecision]:
        """Latency based scaling policy."""
        if not metrics:
            return None
        
        avg_latency = statistics.mean(m.latency_p95 for m in metrics)
        latency_threshold = 1.0  # 1 second P95 latency threshold
        
        if avg_latency > latency_threshold:
            # High latency - scale up
            magnitude = min(1.0, (avg_latency - latency_threshold) / latency_threshold)
            
            # Prefer vertical scaling for latency issues
            if magnitude > 0.7:
                # Significant latency issue - scale resources
                new_resources = {
                    resource: min(4.0, value * (1 + magnitude * 0.5))
                    for resource, value in self.current_resources.items()
                }
                
                return ScalingDecision(
                    direction=ScalingDirection.UP,
                    scaling_type=ScalingType.VERTICAL,
                    magnitude=magnitude,
                    target_instances=self.current_instances,
                    target_resources=new_resources,
                    reasoning=f"High latency: {avg_latency:.2f}s (threshold: {latency_threshold}s)",
                    confidence=0.7
                )
            else:
                # Moderate latency - add instances
                target_instances = min(
                    self.max_instances,
                    math.ceil(self.current_instances * (1 + magnitude * 0.3))
                )
                
                return ScalingDecision(
                    direction=ScalingDirection.UP,
                    scaling_type=ScalingType.HORIZONTAL,
                    magnitude=magnitude,
                    target_instances=target_instances,
                    target_resources=dict(self.current_resources),
                    reasoning=f"Moderate latency increase: {avg_latency:.2f}s",
                    confidence=0.6
                )
        
        return None
    
    def _predictive_policy(self, metrics: List[ScalingMetrics]) -> Optional[ScalingDecision]:
        """Predictive scaling based on trend analysis."""
        if len(metrics) < 5:
            return None
        
        # Simple trend analysis
        recent_cpu = [m.cpu_utilization for m in metrics[-5:]]
        recent_throughput = [m.throughput for m in metrics[-5:]]
        
        # Calculate trends
        cpu_trend = self._calculate_trend(recent_cpu)
        throughput_trend = self._calculate_trend(recent_throughput)
        
        # Predict future state
        current_cpu = recent_cpu[-1]
        predicted_cpu = current_cpu + cpu_trend * 3  # 3 periods ahead
        
        # Predictive scaling decision
        if cpu_trend > 0.05 and predicted_cpu > self.scale_up_threshold:
            magnitude = min(1.0, cpu_trend * 5)  # Scale trend sensitivity
            target_instances = min(
                self.max_instances,
                math.ceil(self.current_instances * (1 + magnitude * 0.2))
            )
            
            return ScalingDecision(
                direction=ScalingDirection.UP,
                scaling_type=ScalingType.HORIZONTAL,
                magnitude=magnitude,
                target_instances=target_instances,
                target_resources=dict(self.current_resources),
                reasoning=f"Predictive scaling: CPU trend {cpu_trend:.3f}, predicted {predicted_cpu:.2f}",
                confidence=0.4  # Lower confidence for predictions
            )
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend."""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        y = values
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def _combine_decisions(self, decisions: List[ScalingDecision]) -> Optional[ScalingDecision]:
        """Combine multiple scaling decisions into final decision."""
        if not decisions:
            return None
        
        # Weight decisions by confidence and policy weight
        weighted_decisions = []
        total_weight = 0.0
        
        for i, decision in enumerate(decisions):
            policy_name = list(self.policy_weights.keys())[min(i, len(self.policy_weights) - 1)]
            policy_weight = self.policy_weights.get(policy_name, 0.1)
            
            weight = decision.confidence * policy_weight
            weighted_decisions.append((decision, weight))
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Find dominant direction
        up_weight = sum(w for d, w in weighted_decisions if d.direction == ScalingDirection.UP)
        down_weight = sum(w for d, w in weighted_decisions if d.direction == ScalingDirection.DOWN)
        
        if up_weight > down_weight and up_weight > 0.3:
            # Scale up
            up_decisions = [d for d, w in weighted_decisions if d.direction == ScalingDirection.UP]
            
            # Combine target instances (max)
            target_instances = max(d.target_instances for d in up_decisions)
            
            # Combine target resources (max)
            target_resources = {}
            for resource in self.current_resources:
                max_resource = max(
                    d.target_resources.get(resource, self.current_resources[resource])
                    for d in up_decisions
                )
                target_resources[resource] = max_resource
            
            # Combined reasoning
            reasons = [d.reasoning for d in up_decisions]
            combined_reasoning = "; ".join(reasons)
            
            return ScalingDecision(
                direction=ScalingDirection.UP,
                scaling_type=ScalingType.HYBRID,  # Combine both types
                magnitude=up_weight / total_weight,
                target_instances=target_instances,
                target_resources=target_resources,
                reasoning=combined_reasoning,
                confidence=up_weight / total_weight
            )
        
        elif down_weight > 0.3:
            # Scale down
            down_decisions = [d for d, w in weighted_decisions if d.direction == ScalingDirection.DOWN]
            
            target_instances = min(d.target_instances for d in down_decisions)
            
            target_resources = {}
            for resource in self.current_resources:
                min_resource = min(
                    d.target_resources.get(resource, self.current_resources[resource])
                    for d in down_decisions
                )
                target_resources[resource] = min_resource
            
            reasons = [d.reasoning for d in down_decisions]
            combined_reasoning = "; ".join(reasons)
            
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                scaling_type=ScalingType.HYBRID,
                magnitude=down_weight / total_weight,
                target_instances=target_instances,
                target_resources=target_resources,
                reasoning=combined_reasoning,
                confidence=down_weight / total_weight
            )
        
        return None
    
    def _auto_tune_policies(self):
        """Auto-tune policy weights based on recent performance."""
        if len(self.metrics_history) < 50 or len(self.scaling_history) < 5:
            return
        
        # Analyze recent scaling effectiveness
        recent_scalings = list(self.scaling_history)[-5:]
        
        # Simple effectiveness metric: did scaling improve performance?
        effectiveness_scores = {}
        
        for scaling in recent_scalings:
            # Find metrics before and after scaling
            scaling_time = scaling.get("timestamp", 0)
            
            before_metrics = [
                m for m in self.metrics_history
                if scaling_time - 300 < m.timestamp < scaling_time  # 5 min before
            ]
            after_metrics = [
                m for m in self.metrics_history
                if scaling_time < m.timestamp < scaling_time + 300  # 5 min after
            ]
            
            if before_metrics and after_metrics:
                before_perf = statistics.mean(
                    m.throughput / max(m.latency_p95, 0.1) for m in before_metrics
                )
                after_perf = statistics.mean(
                    m.throughput / max(m.latency_p95, 0.1) for m in after_metrics
                )
                
                effectiveness = (after_perf - before_perf) / max(before_perf, 0.1)
                policy_type = scaling.get("policy", "unknown")
                
                if policy_type not in effectiveness_scores:
                    effectiveness_scores[policy_type] = []
                effectiveness_scores[policy_type].append(effectiveness)
        
        # Update policy weights based on effectiveness
        for policy, scores in effectiveness_scores.items():
            if policy in self.policy_weights and scores:
                avg_effectiveness = statistics.mean(scores)
                
                # Adjust weight based on effectiveness
                adjustment = avg_effectiveness * 0.1  # Small adjustment
                new_weight = max(0.05, min(0.8, self.policy_weights[policy] + adjustment))
                self.policy_weights[policy] = new_weight
        
        # Normalize weights
        total_weight = sum(self.policy_weights.values())
        if total_weight > 0:
            for policy in self.policy_weights:
                self.policy_weights[policy] /= total_weight
    
    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        try:
            # Record scaling action
            scaling_record = {
                "timestamp": time.time(),
                "decision": decision,
                "previous_instances": self.current_instances,
                "previous_resources": dict(self.current_resources)
            }
            
            # Update current state
            self.current_instances = decision.target_instances
            self.current_resources = dict(decision.target_resources)
            self.last_scaling_time = time.time()
            
            # Record in history
            self.scaling_history.append(scaling_record)
            
            return True
            
        except Exception as e:
            # Revert on failure
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        recent_decisions = list(self.scaling_history)[-10:] if self.scaling_history else []
        
        return {
            "current_instances": self.current_instances,
            "current_resources": dict(self.current_resources),
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "policy_weights": dict(self.policy_weights),
            "recent_scalings": len(recent_decisions),
            "last_scaling_time": self.last_scaling_time,
            "cooldown_remaining": max(0, self.cooldown_period - (time.time() - self.last_scaling_time)),
            "auto_tune_enabled": self.auto_tune_enabled
        }


class ScalingManager:
    """
    Main scaling manager coordinating load balancing and auto-scaling.
    
    Provides unified interface for all scaling operations.
    """
    
    def __init__(
        self,
        load_balancer: Optional[LoadBalancer] = None,
        auto_scaler: Optional[AutoScaler] = None,
        concurrency_manager: Optional[ConcurrencyManager] = None
    ):
        self.load_balancer = load_balancer or LoadBalancer()
        self.auto_scaler = auto_scaler or AutoScaler()
        self.concurrency_manager = concurrency_manager
        
        # Scaling coordination
        self.scaling_enabled = True
        self.monitoring_interval = 30.0  # 30 seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.scaling_effectiveness: deque = deque(maxlen=100)
        
    async def start_monitoring(self):
        """Start automatic scaling monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop automatic scaling monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
    
    async def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while True:
            try:
                if self.scaling_enabled:
                    # Collect current metrics
                    metrics = await self._collect_metrics()
                    
                    if metrics:
                        # Add metrics to auto-scaler
                        self.auto_scaler.add_metrics(metrics)
                        
                        # Check for scaling decisions
                        decision = self.auto_scaler.should_scale()
                        
                        if decision and decision.confidence > 0.3:
                            # Execute scaling decision
                            success = await self._execute_scaling_decision(decision)
                            
                            # Track effectiveness
                            self.scaling_effectiveness.append({
                                "decision": decision,
                                "success": success,
                                "timestamp": time.time()
                            })
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Scaling monitoring error: {e}")
                await asyncio.sleep(60)  # Longer pause on error
    
    async def _collect_metrics(self) -> Optional[ScalingMetrics]:
        """Collect current system metrics."""
        try:
            # Get metrics from concurrency manager if available
            if self.concurrency_manager:
                concurrency_metrics = self.concurrency_manager.get_concurrency_metrics()
                
                # Calculate derived metrics
                cpu_util = concurrency_metrics.resource_utilization.get("cpu", 0.0)
                memory_util = concurrency_metrics.resource_utilization.get("memory", 0.0)
                queue_length = sum(concurrency_metrics.queue_sizes.values())
                throughput = concurrency_metrics.task_throughput
                
                # Estimate latency (simplified)
                latency_p95 = 1.0 / max(throughput, 0.1) if throughput > 0 else 1.0
                
                # Calculate error rate (simplified)
                error_rate = 0.05 if concurrency_metrics.deadlock_detections > 0 else 0.01
                
                return ScalingMetrics(
                    cpu_utilization=cpu_util,
                    memory_utilization=memory_util,
                    queue_length=int(queue_length),
                    throughput=throughput,
                    latency_p95=latency_p95,
                    error_rate=error_rate
                )
        
        except Exception as e:
            print(f"Metrics collection error: {e}")
        
        return None
    
    async def _execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        try:
            print(f"Executing scaling decision: {decision.direction.value} "
                  f"({decision.scaling_type.value}) - {decision.reasoning}")
            
            # Execute the scaling
            success = self.auto_scaler.execute_scaling(decision)
            
            if success:
                # Update load balancer if needed
                if decision.direction == ScalingDirection.UP:
                    # Could add new nodes to load balancer here
                    pass
                elif decision.direction == ScalingDirection.DOWN:
                    # Could remove nodes from load balancer here
                    pass
            
            return success
            
        except Exception as e:
            print(f"Scaling execution error: {e}")
            return False
    
    def register_computation_node(
        self,
        node_id: str,
        capacity: Dict[str, float],
        endpoint: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register new computation node with load balancer."""
        self.load_balancer.register_node(node_id, capacity, endpoint, metadata)
    
    def get_node_for_task(
        self,
        task_requirements: Dict[str, float],
        quantum_properties: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Get best node for task execution."""
        return self.load_balancer.select_node(task_requirements, quantum_properties)
    
    def record_task_execution(
        self,
        node_id: str,
        task_id: str,
        success: bool,
        execution_time: float
    ):
        """Record task execution results."""
        self.load_balancer.record_request_end(node_id, task_id, success, execution_time)
    
    def get_scaling_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        return {
            "timestamp": time.time(),
            "scaling_enabled": self.scaling_enabled,
            "monitoring_active": self.monitoring_task is not None,
            "auto_scaler_status": self.auto_scaler.get_scaling_status(),
            "load_distribution": self.load_balancer.get_load_distribution(),
            "scaling_effectiveness": len(self.scaling_effectiveness),
            "recent_scaling_success_rate": (
                sum(1 for e in self.scaling_effectiveness if e["success"]) /
                max(len(self.scaling_effectiveness), 1)
            )
        }
    
    def enable_scaling(self):
        """Enable automatic scaling."""
        self.scaling_enabled = True
    
    def disable_scaling(self):
        """Disable automatic scaling."""
        self.scaling_enabled = False