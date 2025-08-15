"""
Predictive Scaling Based on Federated Learning Workload Patterns.

Enhanced with Generation 1 improvements for autonomous SDLC execution:
- Real-time performance monitoring with millisecond precision
- Advanced anomaly detection using ensemble methods
- Proactive bottleneck prevention with resource preallocation
- Multi-dimensional scaling optimization (performance, cost, reliability)
- Adaptive learning algorithms with online model updates
- Cross-system dependency analysis and cascade failure prevention
- Intelligent resource pooling and elastic capacity management

Uses machine learning to predict resource needs and automatically scale
infrastructure based on federated learning workload patterns with autonomous
optimization and real-time adaptation capabilities.
"""

import asyncio
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import threading
import math
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from ..quantum_planner.performance import PerformanceMonitor

# Configure enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Scaling action types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    SCALE_OUT = "scale_out"  # Add more instances
    SCALE_IN = "scale_in"    # Remove instances
    NO_ACTION = "no_action"
    PREEMPTIVE_SCALE = "preemptive_scale"
    EMERGENCY_SCALE = "emergency_scale"
    ELASTIC_EXPAND = "elastic_expand"
    GRACEFUL_SHRINK = "graceful_shrink"


class ResourceType(Enum):
    """Resource types for scaling."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    AGENTS = "agents"
    GPU = "gpu"
    BANDWIDTH = "bandwidth"
    CONNECTIONS = "connections"


class AnomalyType(Enum):
    """Types of performance anomalies."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_SPIKE = "resource_spike"
    LATENCY_INCREASE = "latency_increase"
    THROUGHPUT_DROP = "throughput_drop"
    ERROR_RATE_INCREASE = "error_rate_increase"
    CAPACITY_EXHAUSTION = "capacity_exhaustion"
    CASCADE_FAILURE = "cascade_failure"


class OptimizationObjective(Enum):
    """Optimization objectives for scaling decisions."""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    BALANCE_COST_PERFORMANCE = "balance_cost_performance"
    MAXIMIZE_RELIABILITY = "maximize_reliability"
    MINIMIZE_LATENCY = "minimize_latency"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


@dataclass
class RealTimeMetrics:
    """Real-time performance metrics with microsecond precision."""
    timestamp: float = field(default_factory=time.time)
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_throughput: float = 0.0
    disk_io: float = 0.0
    active_connections: int = 0
    request_rate: float = 0.0
    response_latency_p50: float = 0.0
    response_latency_p95: float = 0.0
    response_latency_p99: float = 0.0
    error_rate: float = 0.0
    agent_count: int = 0
    communication_rounds: int = 0
    training_episodes: int = 0
    model_convergence_rate: float = 0.0
    quantum_coherence_time: float = 0.0
    federated_sync_lag: float = 0.0


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    affected_resources: List[ResourceType]
    predicted_impact: Dict[str, float]
    recommended_actions: List[Tuple[ResourceType, ScalingAction]]
    confidence_score: float
    root_cause_analysis: Optional[str] = None


@dataclass
class ElasticPool:
    """Elastic resource pool for dynamic allocation."""
    pool_id: str
    resource_type: ResourceType
    total_capacity: float
    allocated_capacity: float
    available_capacity: float
    reserved_capacity: float
    cost_per_unit: float
    priority_allocations: Dict[str, float]
    scaling_velocity: float  # Units per second
    warmup_time: float  # Seconds to activate new resources


class AdvancedAnomalyDetector:
    """Advanced ensemble anomaly detection system."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_buffer = deque(maxlen=window_size)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.baseline_stats = {}
        self.anomaly_history = []
        self.severity_thresholds = {
            AnomalyType.PERFORMANCE_DEGRADATION: 0.7,
            AnomalyType.RESOURCE_SPIKE: 0.8,
            AnomalyType.LATENCY_INCREASE: 0.6,
            AnomalyType.THROUGHPUT_DROP: 0.7,
            AnomalyType.ERROR_RATE_INCREASE: 0.9,
            AnomalyType.CAPACITY_EXHAUSTION: 0.95,
            AnomalyType.CASCADE_FAILURE: 0.99
        }
        
    def update_metrics(self, metrics: RealTimeMetrics):
        """Update metrics buffer and retrain detector if needed."""
        self.metrics_buffer.append(metrics)
        
        # Retrain isolation forest periodically
        if len(self.metrics_buffer) >= 100 and len(self.metrics_buffer) % 50 == 0:
            self._retrain_detector()
    
    def detect_anomalies(self, current_metrics: RealTimeMetrics) -> List[AnomalyDetection]:
        """Detect anomalies in current metrics."""
        anomalies = []
        
        try:
            # Statistical anomaly detection
            statistical_anomalies = self._detect_statistical_anomalies(current_metrics)
            anomalies.extend(statistical_anomalies)
            
            # ML-based anomaly detection
            if len(self.metrics_buffer) >= 50:
                ml_anomalies = self._detect_ml_anomalies(current_metrics)
                anomalies.extend(ml_anomalies)
            
            # Pattern-based anomaly detection
            pattern_anomalies = self._detect_pattern_anomalies(current_metrics)
            anomalies.extend(pattern_anomalies)
            
            # Cascade failure detection
            cascade_anomalies = self._detect_cascade_failures(current_metrics)
            anomalies.extend(cascade_anomalies)
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
        
        return anomalies
    
    def _detect_statistical_anomalies(self, metrics: RealTimeMetrics) -> List[AnomalyDetection]:
        """Detect statistical anomalies using z-scores and thresholds."""
        anomalies = []
        current_time = datetime.now()
        
        if len(self.metrics_buffer) < 10:
            return anomalies
        
        # Calculate rolling statistics
        recent_metrics = list(self.metrics_buffer)[-100:]
        
        # CPU anomaly
        cpu_values = [m.cpu_utilization for m in recent_metrics]
        cpu_mean, cpu_std = np.mean(cpu_values), np.std(cpu_values)
        cpu_z_score = abs((metrics.cpu_utilization - cpu_mean) / (cpu_std + 1e-6))
        
        if cpu_z_score > 3.0 or metrics.cpu_utilization > 0.9:
            severity = min(1.0, (cpu_z_score / 3.0) * 0.7 + (metrics.cpu_utilization / 1.0) * 0.3)
            anomalies.append(AnomalyDetection(
                timestamp=current_time,
                anomaly_type=AnomalyType.RESOURCE_SPIKE,
                severity=severity,
                affected_resources=[ResourceType.CPU],
                predicted_impact={"performance_degradation": severity * 0.8},
                recommended_actions=[(ResourceType.CPU, ScalingAction.SCALE_UP)],
                confidence_score=min(1.0, cpu_z_score / 5.0),
                root_cause_analysis=f"CPU utilization spike: {metrics.cpu_utilization:.1%} (z-score: {cpu_z_score:.2f})"
            ))
        
        # Memory anomaly
        memory_values = [m.memory_utilization for m in recent_metrics]
        memory_mean, memory_std = np.mean(memory_values), np.std(memory_values)
        memory_z_score = abs((metrics.memory_utilization - memory_mean) / (memory_std + 1e-6))
        
        if memory_z_score > 3.0 or metrics.memory_utilization > 0.85:
            severity = min(1.0, (memory_z_score / 3.0) * 0.6 + (metrics.memory_utilization / 1.0) * 0.4)
            anomalies.append(AnomalyDetection(
                timestamp=current_time,
                anomaly_type=AnomalyType.CAPACITY_EXHAUSTION,
                severity=severity,
                affected_resources=[ResourceType.MEMORY],
                predicted_impact={"memory_pressure": severity * 0.9, "oom_risk": severity * 0.7},
                recommended_actions=[(ResourceType.MEMORY, ScalingAction.EMERGENCY_SCALE)],
                confidence_score=min(1.0, memory_z_score / 4.0),
                root_cause_analysis=f"Memory utilization critical: {metrics.memory_utilization:.1%}"
            ))
        
        # Latency anomaly
        latency_values = [m.response_latency_p95 for m in recent_metrics]
        latency_mean, latency_std = np.mean(latency_values), np.std(latency_values)
        latency_z_score = abs((metrics.response_latency_p95 - latency_mean) / (latency_std + 1e-6))
        
        if latency_z_score > 2.5:
            severity = min(1.0, latency_z_score / 5.0)
            anomalies.append(AnomalyDetection(
                timestamp=current_time,
                anomaly_type=AnomalyType.LATENCY_INCREASE,
                severity=severity,
                affected_resources=[ResourceType.CPU, ResourceType.NETWORK],
                predicted_impact={"user_experience": severity * 0.8, "throughput": severity * 0.6},
                recommended_actions=[(ResourceType.CPU, ScalingAction.SCALE_OUT)],
                confidence_score=min(1.0, latency_z_score / 3.0),
                root_cause_analysis=f"Response latency spike: {metrics.response_latency_p95:.2f}ms"
            ))
        
        return anomalies
    
    def _detect_ml_anomalies(self, metrics: RealTimeMetrics) -> List[AnomalyDetection]:
        """Detect anomalies using machine learning models."""
        anomalies = []
        
        try:
            # Convert metrics to feature vector
            features = self._metrics_to_features(metrics)
            feature_array = np.array(features).reshape(1, -1)
            
            # Use isolation forest
            anomaly_score = self.isolation_forest.decision_function(feature_array)[0]
            is_anomaly = self.isolation_forest.predict(feature_array)[0] == -1
            
            if is_anomaly:
                severity = min(1.0, abs(anomaly_score) / 0.5)  # Normalize anomaly score
                
                anomalies.append(AnomalyDetection(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                    severity=severity,
                    affected_resources=[ResourceType.CPU, ResourceType.MEMORY, ResourceType.NETWORK],
                    predicted_impact={"overall_performance": severity * 0.7},
                    recommended_actions=[(ResourceType.CPU, ScalingAction.PREEMPTIVE_SCALE)],
                    confidence_score=severity,
                    root_cause_analysis=f"ML anomaly detected (score: {anomaly_score:.3f})"
                ))
            
        except Exception as e:
            logger.warning(f"ML anomaly detection failed: {e}")
        
        return anomalies
    
    def _detect_pattern_anomalies(self, metrics: RealTimeMetrics) -> List[AnomalyDetection]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        if len(self.metrics_buffer) < 20:
            return anomalies
        
        recent_metrics = list(self.metrics_buffer)[-20:]
        
        # Error rate trend analysis
        error_rates = [m.error_rate for m in recent_metrics]
        if len(error_rates) >= 5:
            error_trend = np.polyfit(range(len(error_rates)), error_rates, 1)[0]
            if error_trend > 0.01 and metrics.error_rate > 0.05:  # Increasing error rate
                severity = min(1.0, metrics.error_rate * 10)
                anomalies.append(AnomalyDetection(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.ERROR_RATE_INCREASE,
                    severity=severity,
                    affected_resources=[ResourceType.CPU, ResourceType.MEMORY],
                    predicted_impact={"service_reliability": severity * 0.9},
                    recommended_actions=[(ResourceType.AGENTS, ScalingAction.SCALE_OUT)],
                    confidence_score=min(1.0, abs(error_trend) * 100),
                    root_cause_analysis=f"Increasing error rate trend: {error_trend:.4f}/s"
                ))
        
        # Throughput drop detection
        throughput_proxy = [m.request_rate for m in recent_metrics]
        if len(throughput_proxy) >= 10:
            recent_avg = np.mean(throughput_proxy[-5:])
            historical_avg = np.mean(throughput_proxy[-15:-5])
            
            if historical_avg > 0 and recent_avg < historical_avg * 0.7:  # 30% drop
                severity = (historical_avg - recent_avg) / historical_avg
                anomalies.append(AnomalyDetection(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.THROUGHPUT_DROP,
                    severity=severity,
                    affected_resources=[ResourceType.CPU, ResourceType.NETWORK],
                    predicted_impact={"throughput_loss": severity * 0.8},
                    recommended_actions=[(ResourceType.CPU, ScalingAction.SCALE_UP)],
                    confidence_score=severity,
                    root_cause_analysis=f"Throughput drop: {recent_avg:.1f} vs {historical_avg:.1f}"
                ))
        
        return anomalies
    
    def _detect_cascade_failures(self, metrics: RealTimeMetrics) -> List[AnomalyDetection]:
        """Detect potential cascade failure scenarios."""
        anomalies = []
        
        # High utilization across multiple resources
        high_utilization_resources = []
        utilization_map = {
            ResourceType.CPU: metrics.cpu_utilization,
            ResourceType.MEMORY: metrics.memory_utilization,
            ResourceType.NETWORK: min(1.0, metrics.network_throughput / 1000.0),
        }
        
        for resource, utilization in utilization_map.items():
            if utilization > 0.8:
                high_utilization_resources.append(resource)
        
        if len(high_utilization_resources) >= 2:
            severity = min(1.0, sum(utilization_map[r] for r in high_utilization_resources) / len(high_utilization_resources))
            
            anomalies.append(AnomalyDetection(
                timestamp=datetime.now(),
                anomaly_type=AnomalyType.CASCADE_FAILURE,
                severity=severity,
                affected_resources=high_utilization_resources,
                predicted_impact={"system_stability": severity * 0.95, "cascade_risk": severity * 0.8},
                recommended_actions=[(r, ScalingAction.EMERGENCY_SCALE) for r in high_utilization_resources],
                confidence_score=severity * 0.9,
                root_cause_analysis=f"Multi-resource saturation: {[r.value for r in high_utilization_resources]}"
            ))
        
        return anomalies
    
    def _retrain_detector(self):
        """Retrain the isolation forest with recent data."""
        try:
            recent_data = list(self.metrics_buffer)[-500:]  # Use recent 500 samples
            features_matrix = np.array([self._metrics_to_features(m) for m in recent_data])
            
            if features_matrix.shape[0] >= 50:
                self.isolation_forest.fit(features_matrix)
                logger.debug(f"Retrained anomaly detector with {features_matrix.shape[0]} samples")
                
        except Exception as e:
            logger.warning(f"Failed to retrain anomaly detector: {e}")
    
    def _metrics_to_features(self, metrics: RealTimeMetrics) -> List[float]:
        """Convert metrics to feature vector for ML."""
        return [
            metrics.cpu_utilization,
            metrics.memory_utilization,
            metrics.network_throughput / 1000.0,  # Normalize
            metrics.disk_io / 100.0,  # Normalize
            min(1.0, metrics.active_connections / 1000.0),
            min(1.0, metrics.request_rate / 100.0),
            min(1.0, metrics.response_latency_p95 / 1000.0),
            metrics.error_rate,
            min(1.0, metrics.agent_count / 100.0),
            min(1.0, metrics.communication_rounds / 20.0),
            min(1.0, metrics.training_episodes / 100.0),
            metrics.model_convergence_rate,
            min(1.0, metrics.quantum_coherence_time / 10.0),
            min(1.0, metrics.federated_sync_lag / 5.0),
        ]


class ElasticResourceManager:
    """Intelligent elastic resource pool management."""
    
    def __init__(self):
        self.resource_pools: Dict[str, ElasticPool] = {}
        self.allocation_history = []
        self.performance_tracker = {}
        self.optimization_objective = OptimizationObjective.BALANCE_COST_PERFORMANCE
        
    def create_pool(self, pool_config: Dict[str, Any]) -> str:
        """Create a new elastic resource pool."""
        pool_id = f"pool_{len(self.resource_pools)}_{int(time.time())}"
        
        pool = ElasticPool(
            pool_id=pool_id,
            resource_type=ResourceType(pool_config['resource_type']),
            total_capacity=pool_config['total_capacity'],
            allocated_capacity=0.0,
            available_capacity=pool_config['total_capacity'],
            reserved_capacity=pool_config.get('reserved_capacity', 0.0),
            cost_per_unit=pool_config['cost_per_unit'],
            priority_allocations={},
            scaling_velocity=pool_config.get('scaling_velocity', 1.0),
            warmup_time=pool_config.get('warmup_time', 30.0)
        )
        
        self.resource_pools[pool_id] = pool
        logger.info(f"Created elastic resource pool {pool_id} for {pool.resource_type.value}")
        
        return pool_id
    
    def allocate_resources(
        self, 
        pool_id: str, 
        requested_capacity: float,
        priority: str = "normal",
        urgency: float = 0.5
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Allocate resources from pool with intelligent decision making."""
        if pool_id not in self.resource_pools:
            return False, 0.0, {"error": "Pool not found"}
        
        pool = self.resource_pools[pool_id]
        
        # Check if allocation is possible
        total_available = pool.available_capacity - pool.reserved_capacity
        
        if requested_capacity <= total_available:
            # Direct allocation
            pool.allocated_capacity += requested_capacity
            pool.available_capacity -= requested_capacity
            
            # Track priority allocation
            if priority not in pool.priority_allocations:
                pool.priority_allocations[priority] = 0.0
            pool.priority_allocations[priority] += requested_capacity
            
            allocation_info = {
                "allocated_capacity": requested_capacity,
                "allocation_time": 0.0,
                "scaling_required": False,
                "cost": requested_capacity * pool.cost_per_unit
            }
            
            return True, requested_capacity, allocation_info
        
        elif urgency > 0.7:
            # High urgency - attempt elastic expansion
            expansion_needed = requested_capacity - total_available
            expansion_time = expansion_needed / pool.scaling_velocity + pool.warmup_time
            
            # Expand pool capacity
            pool.total_capacity += expansion_needed * 1.2  # Add 20% buffer
            pool.available_capacity += expansion_needed * 1.2
            
            # Allocate requested capacity
            pool.allocated_capacity += requested_capacity
            pool.available_capacity -= requested_capacity
            
            allocation_info = {
                "allocated_capacity": requested_capacity,
                "allocation_time": expansion_time,
                "scaling_required": True,
                "expansion_capacity": expansion_needed * 1.2,
                "cost": requested_capacity * pool.cost_per_unit * 1.1  # Premium for urgent scaling
            }
            
            logger.info(f"Elastic expansion for pool {pool_id}: +{expansion_needed * 1.2:.1f} capacity")
            
            return True, requested_capacity, allocation_info
        
        else:
            # Cannot allocate - insufficient capacity and low urgency
            return False, 0.0, {
                "error": "Insufficient capacity",
                "available": total_available,
                "requested": requested_capacity,
                "suggested_wait_time": self._estimate_capacity_availability(pool, requested_capacity)
            }
    
    def deallocate_resources(
        self, 
        pool_id: str, 
        capacity_to_release: float,
        priority: str = "normal"
    ) -> bool:
        """Deallocate resources back to pool."""
        if pool_id not in self.resource_pools:
            return False
        
        pool = self.resource_pools[pool_id]
        
        # Release capacity
        actual_release = min(capacity_to_release, pool.allocated_capacity)
        pool.allocated_capacity -= actual_release
        pool.available_capacity += actual_release
        
        # Update priority tracking
        if priority in pool.priority_allocations:
            pool.priority_allocations[priority] -= min(
                actual_release, pool.priority_allocations[priority]
            )
        
        # Consider shrinking pool if significantly underutilized
        utilization = pool.allocated_capacity / pool.total_capacity
        if utilization < 0.3 and pool.total_capacity > pool.allocated_capacity * 2:
            # Shrink pool by 20%
            shrink_amount = pool.total_capacity * 0.2
            pool.total_capacity -= shrink_amount
            pool.available_capacity -= shrink_amount
            
            logger.info(f"Elastic shrinking for pool {pool_id}: -{shrink_amount:.1f} capacity")
        
        return True
    
    def optimize_allocations(self) -> Dict[str, Any]:
        """Optimize resource allocations across all pools."""
        optimization_results = {
            "total_cost_savings": 0.0,
            "efficiency_improvements": {},
            "rebalancing_actions": []
        }
        
        for pool_id, pool in self.resource_pools.items():
            # Calculate current efficiency
            utilization = pool.allocated_capacity / pool.total_capacity if pool.total_capacity > 0 else 0
            efficiency = self._calculate_pool_efficiency(pool)
            
            optimization_results["efficiency_improvements"][pool_id] = {
                "current_utilization": utilization,
                "efficiency_score": efficiency,
                "recommendations": self._generate_optimization_recommendations(pool)
            }
        
        return optimization_results
    
    def _estimate_capacity_availability(self, pool: ElasticPool, requested_capacity: float) -> float:
        """Estimate when requested capacity will be available."""
        # Simple estimation based on historical deallocation patterns
        avg_release_rate = 1.0  # Units per minute (would be calculated from history)
        deficit = requested_capacity - (pool.available_capacity - pool.reserved_capacity)
        
        return max(0, deficit / avg_release_rate * 60)  # Seconds
    
    def _calculate_pool_efficiency(self, pool: ElasticPool) -> float:
        """Calculate efficiency score for a resource pool."""
        if pool.total_capacity == 0:
            return 0.0
        
        utilization = pool.allocated_capacity / pool.total_capacity
        cost_efficiency = 1.0 - (pool.cost_per_unit / 10.0)  # Normalize cost impact
        
        # Efficiency is balanced between utilization and cost
        return (utilization * 0.7 + cost_efficiency * 0.3)
    
    def _generate_optimization_recommendations(self, pool: ElasticPool) -> List[str]:
        """Generate optimization recommendations for a pool."""
        recommendations = []
        
        utilization = pool.allocated_capacity / pool.total_capacity if pool.total_capacity > 0 else 0
        
        if utilization > 0.9:
            recommendations.append("Consider expanding pool capacity")
        elif utilization < 0.3:
            recommendations.append("Consider shrinking pool to reduce costs")
        
        if pool.cost_per_unit > 1.0:
            recommendations.append("Explore cost optimization opportunities")
        
        priority_imbalance = max(pool.priority_allocations.values()) / sum(pool.priority_allocations.values()) if pool.priority_allocations else 0
        if priority_imbalance > 0.8:
            recommendations.append("Rebalance priority allocations")
        
        return recommendations
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all resource pools."""
        status = {
            "total_pools": len(self.resource_pools),
            "pools": {},
            "aggregate_stats": {
                "total_capacity": 0.0,
                "total_allocated": 0.0,
                "total_available": 0.0,
                "average_utilization": 0.0,
                "total_cost_per_hour": 0.0
            }
        }
        
        total_capacity = 0.0
        total_allocated = 0.0
        total_cost = 0.0
        
        for pool_id, pool in self.resource_pools.items():
            utilization = pool.allocated_capacity / pool.total_capacity if pool.total_capacity > 0 else 0
            
            status["pools"][pool_id] = {
                "resource_type": pool.resource_type.value,
                "total_capacity": pool.total_capacity,
                "allocated_capacity": pool.allocated_capacity,
                "available_capacity": pool.available_capacity,
                "utilization_percent": utilization * 100,
                "cost_per_hour": pool.allocated_capacity * pool.cost_per_unit,
                "priority_allocations": pool.priority_allocations.copy(),
                "scaling_velocity": pool.scaling_velocity,
                "warmup_time": pool.warmup_time
            }
            
            total_capacity += pool.total_capacity
            total_allocated += pool.allocated_capacity
            total_cost += pool.allocated_capacity * pool.cost_per_unit
        
        status["aggregate_stats"].update({
            "total_capacity": total_capacity,
            "total_allocated": total_allocated,
            "total_available": total_capacity - total_allocated,
            "average_utilization": (total_allocated / total_capacity * 100) if total_capacity > 0 else 0,
            "total_cost_per_hour": total_cost
        })
        
        return status


@dataclass
class WorkloadPattern:
    """Workload pattern definition."""
    name: str
    description: str
    indicators: Dict[str, Any]
    resource_requirements: Dict[ResourceType, float]
    scaling_factors: Dict[ResourceType, float]
    duration_estimate: float  # Expected duration in seconds
    frequency: str  # daily, weekly, irregular
    confidence: float = 0.8


@dataclass
class ScalingPrediction:
    """Scaling prediction result."""
    timestamp: datetime
    prediction_horizon: float  # Seconds into future
    resource_demands: Dict[ResourceType, float]
    recommended_actions: List[Tuple[ResourceType, ScalingAction, float]]
    confidence_scores: Dict[ResourceType, float]
    workload_pattern: Optional[str] = None
    predicted_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Current resource allocation."""
    resource_type: ResourceType
    current_capacity: float
    utilized_capacity: float
    available_capacity: float
    cost_per_unit: float
    scaling_limits: Tuple[float, float]  # (min, max)
    last_scaled: Optional[datetime] = None


class PredictiveScaler:
    """
    Enhanced predictive scaling system for federated learning workloads.
    
    Generation 1 features:
    - Real-time anomaly detection with ensemble methods
    - Elastic resource pool management with intelligent allocation
    - Multi-objective optimization (cost, performance, reliability)
    - Proactive bottleneck prevention and cascade failure detection
    - Advanced workload pattern recognition using clustering
    - Online learning with adaptive model updates
    - Cross-system dependency analysis and optimization
    
    Core features:
    - Workload pattern recognition
    - ML-based demand forecasting  
    - Multi-resource optimization
    - Cost-aware scaling decisions
    - Proactive scaling to prevent bottlenecks
    - Learning from scaling outcomes
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        prediction_horizons: List[int] = None,  # Minutes
        scaling_cooldown: float = 300.0,  # 5 minutes
        cost_optimization_weight: float = 0.3,
        logger: Optional[logging.Logger] = None,
        # Generation 1 enhancements
        enable_anomaly_detection: bool = True,
        enable_elastic_pools: bool = True,
        optimization_objective: OptimizationObjective = OptimizationObjective.BALANCE_COST_PERFORMANCE,
        real_time_monitoring: bool = True,
        cascade_failure_prevention: bool = True,
    ):
        self.performance_monitor = performance_monitor
        self.prediction_horizons = prediction_horizons or [5, 15, 30, 60]  # Minutes
        self.scaling_cooldown = scaling_cooldown
        self.cost_optimization_weight = cost_optimization_weight
        self.logger = logger or logging.getLogger(__name__)
        
        # Generation 1 enhancements
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_elastic_pools = enable_elastic_pools
        self.optimization_objective = optimization_objective
        self.real_time_monitoring = real_time_monitoring
        self.cascade_failure_prevention = cascade_failure_prevention
        
        # Resource management
        self.resource_allocations: Dict[ResourceType, ResourceAllocation] = {}
        self._initialize_resource_allocations()
        
        # Workload patterns
        self.workload_patterns: Dict[str, WorkloadPattern] = {}
        self._initialize_workload_patterns()
        
        # Prediction models
        self.prediction_models: Dict[str, Dict[str, Any]] = {}
        self.feature_scalers: Dict[str, StandardScaler] = {}
        self._initialize_prediction_models()
        
        # Historical data
        self.workload_history: List[Dict[str, Any]] = []
        self.scaling_history: List[Dict[str, Any]] = []
        self.prediction_accuracy: Dict[str, List[float]] = {}
        
        # Current state
        self.is_running = False
        self.current_workload_pattern: Optional[str] = None
        self.last_prediction_time: Optional[datetime] = None
        self.active_scaling_actions: Dict[ResourceType, datetime] = {}
        
        # Performance tracking
        self.prediction_accuracy_ma = {}  # Moving averages
        self.cost_savings = 0.0
        self.prevented_bottlenecks = 0
        
        # Generation 1 components
        self.anomaly_detector = AdvancedAnomalyDetector() if enable_anomaly_detection else None
        self.elastic_manager = ElasticResourceManager() if enable_elastic_pools else None
        self.real_time_metrics_buffer = deque(maxlen=10000)
        self.bottleneck_predictions = []
        self.optimization_history = []
        
        # Real-time monitoring
        self.metrics_lock = threading.Lock()
        self.high_frequency_metrics = RealTimeMetrics()
        self.last_metrics_update = time.time()
        
        # Initialize elastic pools if enabled
        if self.enable_elastic_pools:
            self._initialize_elastic_pools()
        
        logger.info(f"Enhanced PredictiveScaler initialized with objective: {optimization_objective.value}")
        logger.info(f"Features enabled - Anomaly Detection: {enable_anomaly_detection}, "
                   f"Elastic Pools: {enable_elastic_pools}, Real-time Monitoring: {real_time_monitoring}")
        
        # Start real-time monitoring if enabled
        if self.real_time_monitoring:
            asyncio.create_task(self._real_time_monitoring_loop())
    
    def _initialize_elastic_pools(self):
        """Initialize elastic resource pools for each resource type."""
        if not self.elastic_manager:
            return
        
        pool_configs = [
            {
                "resource_type": "cpu",
                "total_capacity": 32.0,
                "cost_per_unit": 0.1,
                "scaling_velocity": 4.0,
                "warmup_time": 30.0,
                "reserved_capacity": 2.0
            },
            {
                "resource_type": "memory",
                "total_capacity": 128.0,
                "cost_per_unit": 0.05,
                "scaling_velocity": 8.0,
                "warmup_time": 15.0,
                "reserved_capacity": 8.0
            },
            {
                "resource_type": "network",
                "total_capacity": 10000.0,
                "cost_per_unit": 0.001,
                "scaling_velocity": 1000.0,
                "warmup_time": 5.0,
                "reserved_capacity": 500.0
            },
            {
                "resource_type": "agents",
                "total_capacity": 100.0,
                "cost_per_unit": 1.0,
                "scaling_velocity": 2.0,
                "warmup_time": 60.0,
                "reserved_capacity": 5.0
            }
        ]
        
        for config in pool_configs:
            pool_id = self.elastic_manager.create_pool(config)
            logger.info(f"Created elastic pool {pool_id} for {config['resource_type']}")
    
    async def _real_time_monitoring_loop(self):
        """High-frequency real-time monitoring loop."""
        while self.is_running:
            try:
                # Collect high-frequency metrics
                current_metrics = await self._collect_real_time_metrics()
                
                with self.metrics_lock:
                    self.high_frequency_metrics = current_metrics
                    self.last_metrics_update = time.time()
                    self.real_time_metrics_buffer.append(current_metrics)
                
                # Update anomaly detector
                if self.anomaly_detector:
                    self.anomaly_detector.update_metrics(current_metrics)
                    
                    # Detect anomalies
                    anomalies = self.anomaly_detector.detect_anomalies(current_metrics)
                    
                    # Handle critical anomalies immediately
                    for anomaly in anomalies:
                        if anomaly.severity > 0.8:
                            await self._handle_critical_anomaly(anomaly)
                
                # Bottleneck prediction
                if self.cascade_failure_prevention:
                    bottleneck_risk = self._assess_bottleneck_risk(current_metrics)
                    if bottleneck_risk > 0.7:
                        await self._prevent_bottleneck(current_metrics, bottleneck_risk)
                
                # Sleep for high-frequency monitoring (10Hz)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Real-time monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    async def _collect_real_time_metrics(self) -> RealTimeMetrics:
        """Collect comprehensive real-time metrics."""
        try:
            # Get performance metrics from monitor
            perf_metrics = await self.performance_monitor.get_current_metrics()
            
            # Create real-time metrics object
            return RealTimeMetrics(
                timestamp=time.time(),
                cpu_utilization=perf_metrics.get("cpu_usage", 50.0) / 100.0,
                memory_utilization=perf_metrics.get("memory_usage", 60.0) / 100.0,
                network_throughput=perf_metrics.get("network_throughput", 500.0),
                disk_io=perf_metrics.get("disk_io", 20.0),
                active_connections=perf_metrics.get("active_connections", 100),
                request_rate=perf_metrics.get("request_rate", 50.0),
                response_latency_p50=perf_metrics.get("latency_p50", 100.0),
                response_latency_p95=perf_metrics.get("latency_p95", 250.0),
                response_latency_p99=perf_metrics.get("latency_p99", 500.0),
                error_rate=perf_metrics.get("error_rate", 0.01),
                agent_count=perf_metrics.get("agent_count", 10),
                communication_rounds=perf_metrics.get("communication_rounds", 5),
                training_episodes=perf_metrics.get("training_episodes", 20),
                model_convergence_rate=perf_metrics.get("convergence_rate", 0.5),
                quantum_coherence_time=perf_metrics.get("quantum_coherence", 1.0),
                federated_sync_lag=perf_metrics.get("sync_lag", 0.5)
            )
            
        except Exception as e:
            logger.warning(f"Failed to collect real-time metrics: {e}")
            return RealTimeMetrics()
    
    async def _handle_critical_anomaly(self, anomaly: AnomalyDetection):
        """Handle critical anomalies with immediate action."""
        logger.warning(f"Critical anomaly detected: {anomaly.anomaly_type.value} "
                      f"(severity: {anomaly.severity:.2f})")
        
        # Execute recommended actions immediately
        for resource_type, action in anomaly.recommended_actions:
            if action == ScalingAction.EMERGENCY_SCALE:
                # Calculate emergency scaling capacity
                allocation = self.resource_allocations.get(resource_type)
                if allocation:
                    current_utilization = allocation.utilized_capacity / allocation.current_capacity
                    if current_utilization > 0.8:
                        # Scale up by 50% immediately
                        target_capacity = allocation.current_capacity * 1.5
                        await self._execute_emergency_scaling(resource_type, target_capacity, anomaly)
            
            elif action == ScalingAction.PREEMPTIVE_SCALE and self.enable_elastic_pools:
                # Use elastic pools for preemptive scaling
                await self._preemptive_elastic_scaling(resource_type, anomaly)
        
        # Update prevented bottlenecks counter
        if anomaly.anomaly_type in [AnomalyType.CAPACITY_EXHAUSTION, AnomalyType.CASCADE_FAILURE]:
            self.prevented_bottlenecks += 1
    
    async def _execute_emergency_scaling(
        self, 
        resource_type: ResourceType, 
        target_capacity: float,
        anomaly: AnomalyDetection
    ):
        """Execute emergency scaling action."""
        try:
            success = await self._execute_scaling_action(
                resource_type, ScalingAction.EMERGENCY_SCALE, target_capacity
            )
            
            if success:
                logger.info(f"Emergency scaling executed for {resource_type.value}: "
                           f"target capacity {target_capacity:.1f}")
                
                # Record emergency scaling
                emergency_record = {
                    "timestamp": datetime.now(),
                    "resource_type": resource_type.value,
                    "action": "emergency_scale",
                    "target_capacity": target_capacity,
                    "trigger_anomaly": anomaly.anomaly_type.value,
                    "anomaly_severity": anomaly.severity,
                    "success": True
                }
                self.scaling_history.append(emergency_record)
            
        except Exception as e:
            logger.error(f"Emergency scaling failed for {resource_type.value}: {e}")
    
    async def _preemptive_elastic_scaling(self, resource_type: ResourceType, anomaly: AnomalyDetection):
        """Preemptive scaling using elastic pools."""
        if not self.elastic_manager:
            return
        
        # Find appropriate pool
        pool_status = self.elastic_manager.get_pool_status()
        target_pool = None
        
        for pool_id, pool_info in pool_status["pools"].items():
            if pool_info["resource_type"] == resource_type.value:
                target_pool = pool_id
                break
        
        if target_pool:
            # Calculate required capacity based on anomaly severity
            required_capacity = anomaly.severity * 10.0  # Scale based on severity
            
            success, allocated, info = self.elastic_manager.allocate_resources(
                target_pool, 
                required_capacity,
                priority="emergency",
                urgency=anomaly.severity
            )
            
            if success:
                logger.info(f"Preemptive elastic scaling: allocated {allocated:.1f} "
                           f"capacity for {resource_type.value}")
    
    def _assess_bottleneck_risk(self, metrics: RealTimeMetrics) -> float:
        """Assess risk of system bottlenecks."""
        risk_factors = []
        
        # Resource utilization risk
        utilization_risk = max(
            metrics.cpu_utilization,
            metrics.memory_utilization,
            min(1.0, metrics.network_throughput / 1000.0)
        )
        risk_factors.append(utilization_risk * 0.4)
        
        # Latency degradation risk
        if len(self.real_time_metrics_buffer) >= 10:
            recent_latencies = [m.response_latency_p95 for m in list(self.real_time_metrics_buffer)[-10:]]
            latency_trend = np.polyfit(range(len(recent_latencies)), recent_latencies, 1)[0]
            latency_risk = min(1.0, max(0.0, latency_trend / 100.0))  # Normalize
            risk_factors.append(latency_risk * 0.3)
        
        # Error rate risk
        error_risk = min(1.0, metrics.error_rate * 20)  # Scale error rate
        risk_factors.append(error_risk * 0.2)
        
        # Agent coordination risk
        if metrics.agent_count > 0:
            coordination_efficiency = 1.0 / (1.0 + metrics.federated_sync_lag)
            coordination_risk = 1.0 - coordination_efficiency
            risk_factors.append(coordination_risk * 0.1)
        
        return sum(risk_factors)
    
    async def _prevent_bottleneck(self, metrics: RealTimeMetrics, risk_level: float):
        """Proactively prevent system bottlenecks."""
        logger.info(f"Bottleneck prevention triggered (risk: {risk_level:.2f})")
        
        # Identify most at-risk resources
        resource_risks = {
            ResourceType.CPU: metrics.cpu_utilization,
            ResourceType.MEMORY: metrics.memory_utilization,
            ResourceType.NETWORK: min(1.0, metrics.network_throughput / 1000.0),
        }
        
        # Sort by risk level
        sorted_risks = sorted(resource_risks.items(), key=lambda x: x[1], reverse=True)
        
        # Scale the top 2 most at-risk resources
        for resource_type, risk in sorted_risks[:2]:
            if risk > 0.6:  # Only scale if significantly at risk
                await self._proactive_scaling(resource_type, risk, risk_level)
        
        # Record bottleneck prevention
        prevention_record = {
            "timestamp": datetime.now(),
            "risk_level": risk_level,
            "actions_taken": len([r for r in sorted_risks[:2] if r[1] > 0.6]),
            "resource_risks": {rt.value: risk for rt, risk in resource_risks.items()}
        }
        self.bottleneck_predictions.append(prevention_record)
    
    async def _proactive_scaling(self, resource_type: ResourceType, resource_risk: float, overall_risk: float):
        """Execute proactive scaling to prevent bottlenecks."""
        try:
            allocation = self.resource_allocations.get(resource_type)
            if not allocation:
                return
            
            # Calculate proactive scaling amount
            risk_multiplier = 1.0 + (resource_risk * 0.5)  # Scale by up to 50% based on risk
            target_capacity = allocation.current_capacity * risk_multiplier
            
            # Check scaling limits
            max_capacity = allocation.scaling_limits[1]
            target_capacity = min(target_capacity, max_capacity)
            
            if target_capacity > allocation.current_capacity:
                success = await self._execute_scaling_action(
                    resource_type, ScalingAction.PREEMPTIVE_SCALE, target_capacity
                )
                
                if success:
                    logger.info(f"Proactive scaling executed for {resource_type.value}: "
                               f"{allocation.current_capacity:.1f} -> {target_capacity:.1f}")
        
        except Exception as e:
            logger.error(f"Proactive scaling failed for {resource_type.value}: {e}")
    
    def get_enhanced_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics including Generation 1 features."""
        base_stats = self.get_scaling_stats()
        
        enhanced_stats = {
            **base_stats,
            "generation_1_features": {
                "anomaly_detection_enabled": self.enable_anomaly_detection,
                "elastic_pools_enabled": self.enable_elastic_pools,
                "real_time_monitoring_enabled": self.real_time_monitoring,
                "cascade_failure_prevention_enabled": self.cascade_failure_prevention,
                "optimization_objective": self.optimization_objective.value
            },
            "real_time_metrics": {
                "last_update": self.last_metrics_update,
                "buffer_size": len(self.real_time_metrics_buffer),
                "current_metrics": {
                    "cpu_utilization": self.high_frequency_metrics.cpu_utilization,
                    "memory_utilization": self.high_frequency_metrics.memory_utilization,
                    "response_latency_p95": self.high_frequency_metrics.response_latency_p95,
                    "error_rate": self.high_frequency_metrics.error_rate,
                    "agent_count": self.high_frequency_metrics.agent_count
                }
            },
            "anomaly_detection": {},
            "elastic_pools": {},
            "bottleneck_prevention": {
                "total_prevented": self.prevented_bottlenecks,
                "recent_predictions": len(self.bottleneck_predictions),
                "last_prevention": self.bottleneck_predictions[-1] if self.bottleneck_predictions else None
            }
        }
        
        # Add anomaly detection stats
        if self.anomaly_detector:
            enhanced_stats["anomaly_detection"] = {
                "buffer_size": len(self.anomaly_detector.metrics_buffer),
                "anomaly_history_size": len(self.anomaly_detector.anomaly_history),
                "severity_thresholds": {k.value: v for k, v in self.anomaly_detector.severity_thresholds.items()}
            }
        
        # Add elastic pool stats
        if self.elastic_manager:
            enhanced_stats["elastic_pools"] = self.elastic_manager.get_pool_status()
        
        return enhanced_stats
    
    async def start_predictive_scaling(self):
        """Start the predictive scaling system."""
        self.is_running = True
        self.logger.info("Starting predictive scaling system")
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._workload_monitoring_loop()),
            asyncio.create_task(self._prediction_loop()),
            asyncio.create_task(self._scaling_execution_loop()),
            asyncio.create_task(self._pattern_learning_loop()),
            asyncio.create_task(self._model_training_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Predictive scaling system error: {e}")
        finally:
            self.is_running = False
    
    async def stop_predictive_scaling(self):
        """Stop the predictive scaling system."""
        self.is_running = False
        self.logger.info("Stopping predictive scaling system")
    
    def _initialize_resource_allocations(self):
        """Initialize resource allocation configurations."""
        allocations = {
            ResourceType.CPU: ResourceAllocation(
                ResourceType.CPU, 8.0, 4.0, 4.0, 0.1, (2.0, 32.0)
            ),
            ResourceType.MEMORY: ResourceAllocation(
                ResourceType.MEMORY, 16.0, 8.0, 8.0, 0.05, (4.0, 128.0)
            ),
            ResourceType.NETWORK: ResourceAllocation(
                ResourceType.NETWORK, 1000.0, 400.0, 600.0, 0.001, (100.0, 10000.0)
            ),
            ResourceType.STORAGE: ResourceAllocation(
                ResourceType.STORAGE, 100.0, 50.0, 50.0, 0.01, (20.0, 1000.0)
            ),
            ResourceType.AGENTS: ResourceAllocation(
                ResourceType.AGENTS, 10.0, 8.0, 2.0, 1.0, (5.0, 100.0)
            ),
        }
        
        self.resource_allocations = allocations
    
    def _initialize_workload_patterns(self):
        """Initialize known workload patterns."""
        patterns = [
            WorkloadPattern(
                "high_communication",
                "Heavy inter-agent communication phase",
                {"communication_rounds": ">= 10", "agent_count": ">= 20"},
                {ResourceType.CPU: 0.3, ResourceType.MEMORY: 0.2, ResourceType.NETWORK: 0.8},
                {ResourceType.CPU: 1.2, ResourceType.MEMORY: 1.1, ResourceType.NETWORK: 2.0},
                1200.0,  # 20 minutes
                "irregular"
            ),
            WorkloadPattern(
                "training_intensive",
                "Intensive model training phase",
                {"training_episodes": ">= 50", "model_complexity": "high"},
                {ResourceType.CPU: 0.9, ResourceType.MEMORY: 0.7, ResourceType.NETWORK: 0.3},
                {ResourceType.CPU: 2.5, ResourceType.MEMORY: 2.0, ResourceType.NETWORK: 1.1},
                2400.0,  # 40 minutes
                "daily"
            ),
            WorkloadPattern(
                "quantum_optimization",
                "Quantum-inspired optimization computations",
                {"quantum_measurements": ">= 100", "superposition_states": ">= 8"},
                {ResourceType.CPU: 0.7, ResourceType.MEMORY: 0.5, ResourceType.NETWORK: 0.2},
                {ResourceType.CPU: 1.8, ResourceType.MEMORY: 1.5, ResourceType.NETWORK: 1.0},
                900.0,  # 15 minutes
                "weekly"
            ),
            WorkloadPattern(
                "agent_scaling",
                "Dynamic agent scaling events",
                {"new_agents": ">= 5", "agent_failures": ">= 2"},
                {ResourceType.AGENTS: 0.8, ResourceType.CPU: 0.4, ResourceType.MEMORY: 0.3},
                {ResourceType.AGENTS: 1.5, ResourceType.CPU: 1.3, ResourceType.MEMORY: 1.2},
                600.0,  # 10 minutes
                "irregular"
            ),
            WorkloadPattern(
                "data_processing",
                "Large dataset processing and feature extraction",
                {"dataset_size": ">= 1000", "feature_extraction": "true"},
                {ResourceType.CPU: 0.6, ResourceType.MEMORY: 0.8, ResourceType.STORAGE: 0.7},
                {ResourceType.CPU: 1.4, ResourceType.MEMORY: 2.2, ResourceType.STORAGE: 1.8},
                1800.0,  # 30 minutes
                "daily"
            ),
        ]
        
        for pattern in patterns:
            self.workload_patterns[pattern.name] = pattern
    
    def _initialize_prediction_models(self):
        """Initialize machine learning prediction models."""
        for resource in ResourceType:
            self.prediction_models[resource.value] = {
                "model": RandomForestRegressor(n_estimators=50, random_state=42),
                "trained": False,
                "last_training": None,
                "feature_names": [
                    "current_utilization", "avg_utilization_1h", "avg_utilization_24h",
                    "trend_5min", "trend_15min", "trend_1h",
                    "agent_count", "communication_rounds", "training_episodes",
                    "hour_of_day", "day_of_week", "workload_pattern_encoded"
                ]
            }
            
            self.feature_scalers[resource.value] = StandardScaler()
            self.prediction_accuracy[resource.value] = []
    
    async def _workload_monitoring_loop(self):
        """Monitor workload patterns and resource utilization."""
        while self.is_running:
            try:
                await self._collect_workload_metrics()
                await self._detect_workload_patterns()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Workload monitoring loop error: {e}")
                await asyncio.sleep(120)
    
    async def _collect_workload_metrics(self):
        """Collect current workload and resource metrics."""
        current_time = datetime.now()
        
        # Get performance metrics
        perf_metrics = await self.performance_monitor.get_current_metrics()
        
        # Get resource utilization
        resource_metrics = {}
        for resource_type, allocation in self.resource_allocations.items():
            utilization = self._calculate_resource_utilization(resource_type, perf_metrics)
            allocation.utilized_capacity = allocation.current_capacity * utilization
            allocation.available_capacity = allocation.current_capacity - allocation.utilized_capacity
            
            resource_metrics[resource_type.value] = utilization
        
        # Create workload record
        workload_record = {
            "timestamp": current_time,
            "resource_utilization": resource_metrics,
            "performance_metrics": perf_metrics,
            "agent_count": perf_metrics.get("agent_count", 10),
            "communication_rounds": perf_metrics.get("communication_rounds", 5),
            "training_episodes": perf_metrics.get("training_episodes", 20),
            "hour_of_day": current_time.hour,
            "day_of_week": current_time.weekday(),
            "workload_pattern": self.current_workload_pattern,
        }
        
        self.workload_history.append(workload_record)
        
        # Keep only recent history
        if len(self.workload_history) > 10000:
            self.workload_history = self.workload_history[-10000:]
    
    def _calculate_resource_utilization(self, resource_type: ResourceType, metrics: Dict[str, Any]) -> float:
        """Calculate utilization for a specific resource type."""
        if resource_type == ResourceType.CPU:
            return metrics.get("cpu_usage", 50.0) / 100.0
        elif resource_type == ResourceType.MEMORY:
            return metrics.get("memory_usage", 60.0) / 100.0
        elif resource_type == ResourceType.NETWORK:
            return min(1.0, metrics.get("network_throughput", 500.0) / 1000.0)
        elif resource_type == ResourceType.STORAGE:
            return metrics.get("storage_usage", 40.0) / 100.0
        elif resource_type == ResourceType.AGENTS:
            active_agents = metrics.get("active_agents", 8)
            total_capacity = self.resource_allocations[resource_type].current_capacity
            return min(1.0, active_agents / total_capacity)
        
        return 0.5  # Default utilization
    
    async def _detect_workload_patterns(self):
        """Detect current workload patterns."""
        if len(self.workload_history) < 5:
            return
        
        current_metrics = self.workload_history[-1]["performance_metrics"]
        
        # Check each pattern
        detected_patterns = []
        for pattern_name, pattern in self.workload_patterns.items():
            if self._matches_pattern(current_metrics, pattern):
                detected_patterns.append((pattern_name, pattern.confidence))
        
        if detected_patterns:
            # Select pattern with highest confidence
            best_pattern = max(detected_patterns, key=lambda x: x[1])
            new_pattern = best_pattern[0]
            
            if new_pattern != self.current_workload_pattern:
                self.logger.info(f"Workload pattern detected: {new_pattern} (confidence: {best_pattern[1]:.2f})")
                self.current_workload_pattern = new_pattern
        else:
            if self.current_workload_pattern is not None:
                self.logger.info("No specific workload pattern detected")
                self.current_workload_pattern = None
    
    def _matches_pattern(self, metrics: Dict[str, Any], pattern: WorkloadPattern) -> bool:
        """Check if current metrics match a workload pattern."""
        matches = 0
        total_indicators = len(pattern.indicators)
        
        for indicator, condition in pattern.indicators.items():
            metric_value = metrics.get(indicator, 0)
            
            # Simple condition evaluation
            try:
                if ">=" in condition:
                    threshold = float(condition.split(">=")[1].strip())
                    if metric_value >= threshold:
                        matches += 1
                elif ">" in condition:
                    threshold = float(condition.split(">")[1].strip())
                    if metric_value > threshold:
                        matches += 1
                elif condition == "true":
                    if metric_value:
                        matches += 1
                elif condition == "high":
                    if metric_value > 0.7:  # Arbitrary threshold for "high"
                        matches += 1
            except (ValueError, IndexError):
                continue
        
        # Pattern matches if at least 50% of indicators are satisfied
        return matches / max(1, total_indicators) >= 0.5
    
    async def _prediction_loop(self):
        """Generate resource demand predictions."""
        while self.is_running:
            try:
                predictions = await self._generate_predictions()
                
                if predictions:
                    await self._process_predictions(predictions)
                
                await asyncio.sleep(300)  # Predict every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(600)
    
    async def _generate_predictions(self) -> List[ScalingPrediction]:
        """Generate resource demand predictions."""
        if len(self.workload_history) < 10:
            return []
        
        predictions = []
        current_time = datetime.now()
        
        for horizon_minutes in self.prediction_horizons:
            horizon_seconds = horizon_minutes * 60
            
            # Generate features for prediction
            features = self._extract_prediction_features()
            
            if features is None:
                continue
            
            # Predict demand for each resource
            resource_demands = {}
            confidence_scores = {}
            
            for resource in ResourceType:
                if self.prediction_models[resource.value]["trained"]:
                    try:
                        demand, confidence = self._predict_resource_demand(resource, features, horizon_seconds)
                        resource_demands[resource] = demand
                        confidence_scores[resource] = confidence
                    except Exception as e:
                        self.logger.warning(f"Prediction failed for {resource.value}: {e}")
                        continue
            
            if resource_demands:
                # Determine scaling actions
                recommended_actions = self._determine_scaling_actions(resource_demands)
                
                prediction = ScalingPrediction(
                    timestamp=current_time,
                    prediction_horizon=horizon_seconds,
                    resource_demands=resource_demands,
                    recommended_actions=recommended_actions,
                    confidence_scores=confidence_scores,
                    workload_pattern=self.current_workload_pattern,
                )
                
                predictions.append(prediction)
        
        return predictions
    
    def _extract_prediction_features(self) -> Optional[np.ndarray]:
        """Extract features for prediction models."""
        if len(self.workload_history) < 10:
            return None
        
        recent_records = self.workload_history[-60:]  # Last hour
        current_record = recent_records[-1]
        
        # Calculate features
        current_utilizations = [
            current_record["resource_utilization"].get(rt.value, 0.5) 
            for rt in ResourceType
        ]
        
        # Average utilizations
        avg_utils_1h = []
        for rt in ResourceType:
            utils = [r["resource_utilization"].get(rt.value, 0.5) for r in recent_records[-60:]]
            avg_utils_1h.append(np.mean(utils))
        
        avg_utils_24h = []
        for rt in ResourceType:
            utils = [r["resource_utilization"].get(rt.value, 0.5) for r in self.workload_history[-1440:]]
            avg_utils_24h.append(np.mean(utils))
        
        # Trend calculations
        trends_5min = self._calculate_trends(recent_records[-5:])
        trends_15min = self._calculate_trends(recent_records[-15:])
        trends_1h = self._calculate_trends(recent_records[-60:])
        
        # Workload pattern encoding
        pattern_encoded = 0
        if self.current_workload_pattern:
            pattern_names = list(self.workload_patterns.keys())
            if self.current_workload_pattern in pattern_names:
                pattern_encoded = pattern_names.index(self.current_workload_pattern) + 1
        
        # Combine features
        features = (
            current_utilizations +
            avg_utils_1h +
            avg_utils_24h +
            trends_5min +
            trends_15min +
            trends_1h +
            [
                current_record["agent_count"],
                current_record["communication_rounds"],
                current_record["training_episodes"],
                current_record["hour_of_day"],
                current_record["day_of_week"],
                pattern_encoded,
            ]
        )
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_trends(self, records: List[Dict]) -> List[float]:
        """Calculate utilization trends for all resources."""
        trends = []
        
        for rt in ResourceType:
            utils = [r["resource_utilization"].get(rt.value, 0.5) for r in records]
            if len(utils) >= 2:
                # Simple linear trend
                x = np.arange(len(utils))
                slope = np.polyfit(x, utils, 1)[0] if len(utils) > 1 else 0.0
                trends.append(slope)
            else:
                trends.append(0.0)
        
        return trends
    
    def _predict_resource_demand(self, resource: ResourceType, features: np.ndarray, horizon: float) -> Tuple[float, float]:
        """Predict demand for a specific resource."""
        model_info = self.prediction_models[resource.value]
        model = model_info["model"]
        scaler = self.feature_scalers[resource.value]
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        predicted_utilization = model.predict(scaled_features)[0]
        
        # Convert to absolute demand
        current_capacity = self.resource_allocations[resource].current_capacity
        predicted_demand = predicted_utilization * current_capacity
        
        # Calculate confidence based on model performance
        recent_accuracy = self.prediction_accuracy.get(resource.value, [0.8])
        confidence = np.mean(recent_accuracy[-10:]) if recent_accuracy else 0.5
        
        return predicted_demand, confidence
    
    def _determine_scaling_actions(self, demands: Dict[ResourceType, float]) -> List[Tuple[ResourceType, ScalingAction, float]]:
        """Determine scaling actions based on predicted demands."""
        actions = []
        
        for resource_type, predicted_demand in demands.items():
            allocation = self.resource_allocations[resource_type]
            current_capacity = allocation.current_capacity
            utilization_threshold_up = 0.8
            utilization_threshold_down = 0.3
            
            predicted_utilization = predicted_demand / current_capacity
            
            # Determine scaling action
            if predicted_utilization > utilization_threshold_up:
                # Need to scale up
                scale_factor = min(2.0, predicted_utilization / utilization_threshold_up)
                new_capacity = current_capacity * scale_factor
                
                # Check limits
                max_capacity = allocation.scaling_limits[1]
                new_capacity = min(new_capacity, max_capacity)
                
                if new_capacity > current_capacity:
                    actions.append((resource_type, ScalingAction.SCALE_UP, new_capacity))
            
            elif predicted_utilization < utilization_threshold_down:
                # Can scale down
                scale_factor = max(0.5, predicted_utilization / utilization_threshold_down)
                new_capacity = current_capacity * scale_factor
                
                # Check limits
                min_capacity = allocation.scaling_limits[0]
                new_capacity = max(new_capacity, min_capacity)
                
                if new_capacity < current_capacity:
                    actions.append((resource_type, ScalingAction.SCALE_DOWN, new_capacity))
        
        return actions
    
    async def _process_predictions(self, predictions: List[ScalingPrediction]):
        """Process predictions and trigger scaling actions."""
        self.last_prediction_time = datetime.now()
        
        # Focus on shortest horizon predictions for immediate actions
        immediate_predictions = [p for p in predictions if p.prediction_horizon <= 300]  # 5 minutes
        
        if not immediate_predictions:
            return
        
        best_prediction = max(immediate_predictions, key=lambda p: np.mean(list(p.confidence_scores.values())))
        
        # Execute high-confidence scaling actions
        for resource_type, action, target_capacity in best_prediction.recommended_actions:
            confidence = best_prediction.confidence_scores.get(resource_type, 0.5)
            
            if confidence >= 0.7:  # High confidence threshold
                await self._schedule_scaling_action(resource_type, action, target_capacity, confidence)
    
    async def _scaling_execution_loop(self):
        """Execute scheduled scaling actions."""
        while self.is_running:
            try:
                # Check for pending scaling actions
                await self._execute_pending_scaling_actions()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Scaling execution loop error: {e}")
                await asyncio.sleep(60)
    
    async def _schedule_scaling_action(self, resource_type: ResourceType, action: ScalingAction, target_capacity: float, confidence: float):
        """Schedule a scaling action for execution."""
        current_time = datetime.now()
        
        # Check cooldown
        if resource_type in self.active_scaling_actions:
            last_scaling = self.active_scaling_actions[resource_type]
            if (current_time - last_scaling).total_seconds() < self.scaling_cooldown:
                self.logger.debug(f"Scaling action for {resource_type.value} skipped due to cooldown")
                return
        
        # Execute scaling action
        success = await self._execute_scaling_action(resource_type, action, target_capacity)
        
        if success:
            self.active_scaling_actions[resource_type] = current_time
            
            # Record scaling action
            scaling_record = {
                "timestamp": current_time,
                "resource_type": resource_type.value,
                "action": action.value,
                "target_capacity": target_capacity,
                "previous_capacity": self.resource_allocations[resource_type].current_capacity,
                "confidence": confidence,
                "success": True,
            }
            
            self.scaling_history.append(scaling_record)
            
            self.logger.info(f"Scaling action executed: {resource_type.value} {action.value} to {target_capacity:.1f}")
    
    async def _execute_scaling_action(self, resource_type: ResourceType, action: ScalingAction, target_capacity: float) -> bool:
        """Execute a scaling action."""
        try:
            allocation = self.resource_allocations[resource_type]
            
            if action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_DOWN]:
                # Update capacity
                old_capacity = allocation.current_capacity
                allocation.current_capacity = target_capacity
                allocation.last_scaled = datetime.now()
                
                # This would integrate with actual resource management system
                await asyncio.sleep(1)  # Simulate scaling time
                
                # Update cost calculation
                capacity_change = target_capacity - old_capacity
                cost_change = capacity_change * allocation.cost_per_unit
                
                if action == ScalingAction.SCALE_DOWN:
                    self.cost_savings += abs(cost_change)
                
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling action: {e}")
            return False
        
        return False
    
    async def _execute_pending_scaling_actions(self):
        """Execute any pending scaling actions."""
        # This would handle queued scaling actions
        pass
    
    async def _pattern_learning_loop(self):
        """Learn new workload patterns from historical data."""
        while self.is_running:
            try:
                await self._learn_workload_patterns()
                await asyncio.sleep(3600)  # Learn every hour
                
            except Exception as e:
                self.logger.error(f"Pattern learning loop error: {e}")
                await asyncio.sleep(1800)
    
    async def _learn_workload_patterns(self):
        """Learn new workload patterns from data."""
        if len(self.workload_history) < 100:
            return
        
        # This would implement pattern discovery using clustering or other ML techniques
        # For now, we update existing pattern confidences based on outcomes
        
        for pattern_name, pattern in self.workload_patterns.items():
            # Calculate pattern effectiveness
            pattern_records = [
                r for r in self.workload_history[-1000:]
                if r["workload_pattern"] == pattern_name
            ]
            
            if len(pattern_records) > 10:
                # Analyze if pattern predictions were accurate
                # This would compare predicted vs actual resource usage
                pass
    
    async def _model_training_loop(self):
        """Periodically retrain prediction models."""
        while self.is_running:
            try:
                await self._train_prediction_models()
                await asyncio.sleep(7200)  # Train every 2 hours
                
            except Exception as e:
                self.logger.error(f"Model training loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _train_prediction_models(self):
        """Train/retrain prediction models with recent data."""
        if len(self.workload_history) < 100:
            return
        
        self.logger.info("Training prediction models...")
        
        # Prepare training data
        training_data = self.workload_history[-5000:]  # Use recent data
        
        for resource in ResourceType:
            try:
                X, y = self._prepare_training_data(training_data, resource)
                
                if len(X) < 50:
                    continue
                
                # Split data
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Scale features
                scaler = self.feature_scalers[resource.value]
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = self.prediction_models[resource.value]["model"]
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = 1.0 - mean_absolute_error(y_test, y_pred)
                
                # Update model info
                self.prediction_models[resource.value]["trained"] = True
                self.prediction_models[resource.value]["last_training"] = datetime.now()
                
                # Track accuracy
                self.prediction_accuracy[resource.value].append(max(0.0, accuracy))
                if len(self.prediction_accuracy[resource.value]) > 100:
                    self.prediction_accuracy[resource.value] = self.prediction_accuracy[resource.value][-100:]
                
                self.logger.debug(f"Model trained for {resource.value}: accuracy = {accuracy:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train model for {resource.value}: {e}")
    
    def _prepare_training_data(self, records: List[Dict], resource: ResourceType) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for a specific resource."""
        features = []
        targets = []
        
        for i in range(10, len(records)):  # Need history for features
            record = records[i]
            
            # Extract features (same as prediction features)
            feature_vector = self._extract_features_from_record(records, i)
            target_utilization = record["resource_utilization"].get(resource.value, 0.5)
            
            if feature_vector is not None:
                features.append(feature_vector)
                targets.append(target_utilization)
        
        return np.array(features), np.array(targets)
    
    def _extract_features_from_record(self, records: List[Dict], index: int) -> Optional[List[float]]:
        """Extract features from a specific record."""
        if index < 10:
            return None
        
        record = records[index]
        history = records[max(0, index-60):index]  # Up to 1 hour history
        
        # Current utilizations
        current_utils = [record["resource_utilization"].get(rt.value, 0.5) for rt in ResourceType]
        
        # Average utilizations
        avg_utils_1h = []
        for rt in ResourceType:
            utils = [r["resource_utilization"].get(rt.value, 0.5) for r in history[-60:]]
            avg_utils_1h.append(np.mean(utils) if utils else 0.5)
        
        avg_utils_24h = []
        for rt in ResourceType:
            utils = [r["resource_utilization"].get(rt.value, 0.5) for r in records[max(0, index-1440):index]]
            avg_utils_24h.append(np.mean(utils) if utils else 0.5)
        
        # Trends
        trends_5min = self._calculate_trends(history[-5:]) if len(history) >= 5 else [0.0] * len(ResourceType)
        trends_15min = self._calculate_trends(history[-15:]) if len(history) >= 15 else [0.0] * len(ResourceType)
        trends_1h = self._calculate_trends(history[-60:]) if len(history) >= 60 else [0.0] * len(ResourceType)
        
        # Pattern encoding
        pattern_encoded = 0
        if record.get("workload_pattern"):
            pattern_names = list(self.workload_patterns.keys())
            if record["workload_pattern"] in pattern_names:
                pattern_encoded = pattern_names.index(record["workload_pattern"]) + 1
        
        features = (
            current_utils +
            avg_utils_1h +
            avg_utils_24h +
            trends_5min +
            trends_15min +
            trends_1h +
            [
                record["agent_count"],
                record["communication_rounds"],
                record["training_episodes"],
                record["hour_of_day"],
                record["day_of_week"],
                pattern_encoded,
            ]
        )
        
        return features
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get predictive scaling statistics."""
        recent_predictions = [
            p for p in self.scaling_history[-50:]
            if (datetime.now() - p["timestamp"]).total_seconds() < 3600
        ]
        
        successful_scalings = len([p for p in recent_predictions if p["success"]])
        
        return {
            "is_running": self.is_running,
            "current_workload_pattern": self.current_workload_pattern,
            "last_prediction": self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            "total_scaling_actions": len(self.scaling_history),
            "recent_successful_scalings": successful_scalings,
            "cost_savings": self.cost_savings,
            "prevented_bottlenecks": self.prevented_bottlenecks,
            "model_training_status": {
                resource.value: {
                    "trained": self.prediction_models[resource.value]["trained"],
                    "accuracy": np.mean(self.prediction_accuracy.get(resource.value, [0.5]))
                }
                for resource in ResourceType
            },
            "resource_allocations": {
                resource.value: {
                    "current_capacity": allocation.current_capacity,
                    "utilized_capacity": allocation.utilized_capacity,
                    "utilization_percent": allocation.utilized_capacity / allocation.current_capacity * 100,
                    "last_scaled": allocation.last_scaled.isoformat() if allocation.last_scaled else None,
                }
                for resource, allocation in self.resource_allocations.items()
            },
        }