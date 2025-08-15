"""
ML-Powered Load Balancing System for Generation 3 Scaling.

Features:
- Advanced ML models for traffic prediction
- Real-time capacity planning and resource optimization
- Intelligent routing based on performance patterns
- Predictive scaling with confidence intervals
- Multi-objective optimization (latency, throughput, cost)
- Adaptive learning from system behavior
"""

import asyncio
import time
import threading
import statistics
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
from collections import defaultdict, deque
import logging
import concurrent.futures
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    ML_PREDICTED = "ml_predicted"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


class PredictionModel(Enum):
    """ML prediction models for load balancing."""
    LINEAR_REGRESSION = "linear_regression"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    ARIMA = "arima"
    ENSEMBLE = "ensemble"


@dataclass
class ServerNode:
    """Server node in the load balancing pool."""
    node_id: str
    endpoint: str
    capacity: int
    current_load: float
    active_connections: int
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_utilization: float = 0.0
    health_score: float = 1.0
    last_heartbeat: float = field(default_factory=time.time)
    request_count: int = 0
    error_count: int = 0
    
    def update_metrics(self, response_time: float, cpu: float, memory: float, network: float):
        """Update server metrics."""
        self.response_times.append(response_time)
        self.cpu_utilization = cpu
        self.memory_utilization = memory
        self.network_utilization = network
        self.last_heartbeat = time.time()
        self.request_count += 1
        
        # Update health score based on performance
        avg_response_time = statistics.mean(self.response_times) if self.response_times else 1.0
        resource_pressure = max(cpu, memory, network)
        
        self.health_score = max(0.0, min(1.0, 
            1.0 - (avg_response_time / 1000.0) - (resource_pressure * 0.5)
        ))
    
    def get_load_score(self) -> float:
        """Calculate overall load score."""
        connection_load = self.active_connections / max(self.capacity, 1)
        resource_load = max(self.cpu_utilization, self.memory_utilization)
        
        return (connection_load + resource_load) / 2.0
    
    @property
    def average_response_time(self) -> float:
        """Get average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def error_rate(self) -> float:
        """Get error rate."""
        return self.error_count / max(self.request_count, 1)


@dataclass
class LoadBalancingDecision:
    """Load balancing decision result."""
    selected_node: str
    confidence: float
    predicted_response_time: float
    load_distribution: Dict[str, float]
    decision_factors: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrafficPrediction:
    """Traffic prediction result."""
    predicted_requests_per_second: float
    prediction_horizon_seconds: float
    confidence_interval: Tuple[float, float]
    peak_times: List[float]
    resource_requirements: Dict[str, float]
    scaling_recommendations: List[str]


class MLTrafficPredictor:
    """ML-based traffic prediction system."""
    
    def __init__(self, prediction_model: PredictionModel = PredictionModel.ENSEMBLE):
        self.prediction_model = prediction_model
        self.traffic_history: deque = deque(maxlen=10000)
        self.model_weights = np.random.normal(0, 0.1, 20)  # Simple linear model
        self.prediction_cache: Dict[str, Tuple[float, TrafficPrediction]] = {}
        
        # LSTM-like state for time series prediction
        self.lstm_state = np.zeros(10)
        self.seasonal_patterns = defaultdict(list)
        
        logging.info(f"MLTrafficPredictor initialized with {prediction_model.value} model")
    
    def record_traffic(self, timestamp: float, requests_per_second: float, **metrics):
        """Record traffic data for learning."""
        self.traffic_history.append({
            'timestamp': timestamp,
            'rps': requests_per_second,
            'hour_of_day': (timestamp % 86400) / 3600,
            'day_of_week': ((timestamp // 86400) % 7),
            **metrics
        })
        
        # Update seasonal patterns
        hour = int((timestamp % 86400) / 3600)
        self.seasonal_patterns[hour].append(requests_per_second)
        
        # Keep only recent seasonal data
        if len(self.seasonal_patterns[hour]) > 100:
            self.seasonal_patterns[hour] = self.seasonal_patterns[hour][-50:]
        
        # Online learning - update model weights
        if len(self.traffic_history) > 50:
            self._update_model()
    
    def predict_traffic(
        self, 
        horizon_seconds: float = 3600.0,
        confidence_level: float = 0.95
    ) -> TrafficPrediction:
        """Predict future traffic patterns."""
        
        # Check cache first
        cache_key = f"{int(time.time() / 300)}_{horizon_seconds}"  # 5-minute cache
        if cache_key in self.prediction_cache:
            cache_time, cached_prediction = self.prediction_cache[cache_key]
            if time.time() - cache_time < 300:
                return cached_prediction
        
        if len(self.traffic_history) < 10:
            # Not enough data for prediction
            return TrafficPrediction(
                predicted_requests_per_second=10.0,
                prediction_horizon_seconds=horizon_seconds,
                confidence_interval=(5.0, 20.0),
                peak_times=[],
                resource_requirements={'cpu': 0.5, 'memory': 0.5},
                scaling_recommendations=[]
            )
        
        current_time = time.time()
        
        # Feature engineering
        features = self._extract_features(current_time)
        
        # Make prediction based on model type
        if self.prediction_model == PredictionModel.NEURAL_NETWORK:
            predicted_rps = self._neural_network_predict(features)
        elif self.prediction_model == PredictionModel.LSTM:
            predicted_rps = self._lstm_predict(features)
        elif self.prediction_model == PredictionModel.ENSEMBLE:
            predicted_rps = self._ensemble_predict(features)
        else:
            predicted_rps = self._linear_predict(features)
        
        # Calculate confidence interval
        recent_rps = [r['rps'] for r in list(self.traffic_history)[-20:]]
        prediction_variance = np.var(recent_rps) if recent_rps else 1.0
        margin = 1.96 * np.sqrt(prediction_variance)  # 95% confidence
        
        confidence_interval = (
            max(0, predicted_rps - margin),
            predicted_rps + margin
        )
        
        # Identify peak times
        peak_times = self._identify_peak_times(current_time, horizon_seconds)
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(predicted_rps)
        
        # Generate scaling recommendations
        scaling_recommendations = self._generate_scaling_recommendations(
            predicted_rps, confidence_interval, peak_times
        )
        
        prediction = TrafficPrediction(
            predicted_requests_per_second=predicted_rps,
            prediction_horizon_seconds=horizon_seconds,
            confidence_interval=confidence_interval,
            peak_times=peak_times,
            resource_requirements=resource_requirements,
            scaling_recommendations=scaling_recommendations
        )
        
        # Cache prediction
        self.prediction_cache[cache_key] = (time.time(), prediction)
        
        return prediction
    
    def _extract_features(self, current_time: float) -> np.ndarray:
        """Extract features for prediction."""
        features = []
        
        # Time-based features
        hour_of_day = (current_time % 86400) / 3600
        day_of_week = (current_time // 86400) % 7
        
        features.extend([
            hour_of_day / 24.0,
            day_of_week / 7.0,
            np.sin(2 * np.pi * hour_of_day / 24),  # Cyclical hour
            np.cos(2 * np.pi * hour_of_day / 24),
            np.sin(2 * np.pi * day_of_week / 7),   # Cyclical day
            np.cos(2 * np.pi * day_of_week / 7)
        ])
        
        # Recent traffic features
        recent_data = list(self.traffic_history)[-10:]
        if recent_data:
            recent_rps = [r['rps'] for r in recent_data]
            features.extend([
                np.mean(recent_rps),
                np.std(recent_rps),
                np.max(recent_rps),
                np.min(recent_rps),
                recent_rps[-1] if recent_rps else 0,  # Last value
                np.mean(recent_rps[-3:]) if len(recent_rps) >= 3 else 0  # Short-term trend
            ])
        else:
            features.extend([0] * 6)
        
        # Seasonal patterns
        current_hour = int(hour_of_day)
        seasonal_avg = np.mean(self.seasonal_patterns[current_hour]) if self.seasonal_patterns[current_hour] else 10.0
        features.append(seasonal_avg / 100.0)  # Normalize
        
        # Trend features
        if len(self.traffic_history) >= 20:
            recent_trend = self._calculate_trend(20)
            features.append(recent_trend)
        else:
            features.append(0.0)
        
        # Pad or truncate to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    def _linear_predict(self, features: np.ndarray) -> float:
        """Linear regression prediction."""
        prediction = np.dot(features, self.model_weights)
        return max(0.1, prediction)
    
    def _neural_network_predict(self, features: np.ndarray) -> float:
        """Simple neural network prediction."""
        # Single hidden layer network
        hidden = np.tanh(np.dot(features, self.model_weights.reshape(20, 1)).flatten())
        output = np.sum(hidden) / len(hidden) * 50  # Scale to reasonable RPS
        return max(0.1, output)
    
    def _lstm_predict(self, features: np.ndarray) -> float:
        """LSTM-like prediction using state."""
        # Simplified LSTM computation
        forget_gate = 1.0 / (1.0 + np.exp(-np.dot(features[:10], self.lstm_state)))
        input_gate = 1.0 / (1.0 + np.exp(-np.dot(features[10:], self.lstm_state)))
        
        # Update state
        self.lstm_state = forget_gate * self.lstm_state + input_gate * np.tanh(features[:10])
        
        # Output
        output = np.mean(self.lstm_state) * 100
        return max(0.1, output)
    
    def _ensemble_predict(self, features: np.ndarray) -> float:
        """Ensemble prediction combining multiple models."""
        linear_pred = self._linear_predict(features)
        nn_pred = self._neural_network_predict(features)
        lstm_pred = self._lstm_predict(features)
        
        # Weighted ensemble
        ensemble_pred = 0.3 * linear_pred + 0.4 * nn_pred + 0.3 * lstm_pred
        return ensemble_pred
    
    def _update_model(self):
        """Online learning update for model weights."""
        if len(self.traffic_history) < 100:
            return
        
        # Simple gradient descent update
        recent_data = list(self.traffic_history)[-50:]
        
        for data in recent_data[-10:]:  # Use last 10 points for update
            features = self._extract_features(data['timestamp'])
            predicted = self._linear_predict(features)
            actual = data['rps']
            
            # Gradient descent step
            error = predicted - actual
            learning_rate = 0.001
            gradient = error * features
            
            self.model_weights -= learning_rate * gradient
    
    def _calculate_trend(self, window_size: int) -> float:
        """Calculate traffic trend over window."""
        if len(self.traffic_history) < window_size:
            return 0.0
        
        recent_data = list(self.traffic_history)[-window_size:]
        rps_values = [r['rps'] for r in recent_data]
        
        # Simple linear trend
        x = np.arange(len(rps_values))
        y = np.array(rps_values)
        
        if len(x) > 1:
            slope = np.corrcoef(x, y)[0, 1] if np.std(y) > 0 else 0
            return slope
        
        return 0.0
    
    def _identify_peak_times(self, current_time: float, horizon: float) -> List[float]:
        """Identify predicted peak times."""
        peak_times = []
        
        # Check seasonal patterns for peaks
        for hour, rps_history in self.seasonal_patterns.items():
            if rps_history:
                avg_rps = np.mean(rps_history)
                if avg_rps > np.mean([np.mean(h) for h in self.seasonal_patterns.values() if h]) * 1.5:
                    # This is a peak hour
                    next_occurrence = self._next_hour_occurrence(current_time, hour)
                    if next_occurrence <= current_time + horizon:
                        peak_times.append(next_occurrence)
        
        return peak_times
    
    def _next_hour_occurrence(self, current_time: float, target_hour: int) -> float:
        """Find next occurrence of a specific hour."""
        current_hour = int((current_time % 86400) / 3600)
        
        if target_hour > current_hour:
            # Today
            return current_time + (target_hour - current_hour) * 3600
        else:
            # Tomorrow
            return current_time + (24 - current_hour + target_hour) * 3600
    
    def _calculate_resource_requirements(self, predicted_rps: float) -> Dict[str, float]:
        """Calculate resource requirements for predicted traffic."""
        # Simple resource estimation model
        base_cpu = 0.1
        base_memory = 0.2
        
        # Scale with traffic (logarithmic scaling)
        cpu_requirement = base_cpu + 0.05 * math.log(predicted_rps + 1)
        memory_requirement = base_memory + 0.03 * math.log(predicted_rps + 1)
        
        return {
            'cpu': min(1.0, cpu_requirement),
            'memory': min(1.0, memory_requirement),
            'network': min(1.0, predicted_rps / 1000.0)  # 1000 RPS = 100% network
        }
    
    def _generate_scaling_recommendations(
        self,
        predicted_rps: float,
        confidence_interval: Tuple[float, float],
        peak_times: List[float]
    ) -> List[str]:
        """Generate scaling recommendations."""
        recommendations = []
        
        current_capacity_estimate = 100  # Assumed current capacity
        
        # High traffic prediction
        if predicted_rps > current_capacity_estimate * 0.8:
            recommendations.append(f"Scale up: Predicted {predicted_rps:.1f} RPS approaching capacity")
        
        # Peak time preparation
        if peak_times:
            next_peak = min(peak_times)
            time_to_peak = next_peak - time.time()
            if time_to_peak < 3600:  # Less than 1 hour
                recommendations.append(f"Prepare for peak in {time_to_peak/60:.0f} minutes")
        
        # High uncertainty
        ci_width = confidence_interval[1] - confidence_interval[0]
        if ci_width > predicted_rps * 0.5:
            recommendations.append("High prediction uncertainty - maintain flexible capacity")
        
        # Low traffic prediction
        if predicted_rps < current_capacity_estimate * 0.3:
            recommendations.append("Consider scaling down to optimize costs")
        
        return recommendations


class MLLoadBalancer:
    """
    ML-Powered Load Balancer with Advanced Traffic Prediction.
    
    Features:
    - Real-time ML-based traffic prediction
    - Intelligent server selection using multiple algorithms
    - Adaptive learning from performance patterns
    - Multi-objective optimization (latency, throughput, cost)
    - Predictive capacity planning
    """
    
    def __init__(
        self,
        servers: List[ServerNode],
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_HYBRID,
        prediction_model: PredictionModel = PredictionModel.ENSEMBLE,
        learning_rate: float = 0.01
    ):
        self.servers = {server.node_id: server for server in servers}
        self.strategy = strategy
        self.learning_rate = learning_rate
        
        # ML components
        self.traffic_predictor = MLTrafficPredictor(prediction_model)
        self.performance_history: deque = deque(maxlen=10000)
        self.routing_decisions: deque = deque(maxlen=1000)
        
        # Adaptive weights for hybrid strategy
        self.strategy_weights = {
            LoadBalancingStrategy.LEAST_CONNECTIONS: 0.25,
            LoadBalancingStrategy.LEAST_RESPONSE_TIME: 0.25,
            LoadBalancingStrategy.RESOURCE_BASED: 0.25,
            LoadBalancingStrategy.ML_PREDICTED: 0.25
        }
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Background tasks
        self.monitoring_task = None
        self.prediction_task = None
        self.optimization_task = None
        
        logging.info(f"MLLoadBalancer initialized with {len(servers)} servers")
    
    async def start(self):
        """Start the load balancer and background tasks."""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        logging.info("MLLoadBalancer started")
    
    async def stop(self):
        """Stop the load balancer."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.prediction_task:
            self.prediction_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()
        
        logging.info("MLLoadBalancer stopped")
    
    async def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> LoadBalancingDecision:
        """Select the best server for a request using ML."""
        start_time = time.time()
        
        if not self.servers:
            raise RuntimeError("No servers available")
        
        # Filter healthy servers
        healthy_servers = {
            server_id: server for server_id, server in self.servers.items()
            if server.health_score > 0.5 and time.time() - server.last_heartbeat < 30
        }
        
        if not healthy_servers:
            # All servers unhealthy, select least bad
            healthy_servers = self.servers
        
        # Select server based on strategy
        if self.strategy == LoadBalancingStrategy.ADAPTIVE_HYBRID:
            selected_server_id = await self._adaptive_hybrid_selection(healthy_servers, request_context)
        elif self.strategy == LoadBalancingStrategy.ML_PREDICTED:
            selected_server_id = await self._ml_predicted_selection(healthy_servers, request_context)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            selected_server_id = self._resource_based_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected_server_id = self._least_response_time_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_server_id = self._least_connections_selection(healthy_servers)
        else:
            selected_server_id = self._round_robin_selection(healthy_servers)
        
        # Calculate load distribution
        load_distribution = {
            server_id: server.get_load_score() 
            for server_id, server in healthy_servers.items()
        }
        
        # Predict response time
        selected_server = self.servers[selected_server_id]
        predicted_response_time = self._predict_response_time(selected_server, request_context)
        
        # Calculate decision confidence
        confidence = self._calculate_decision_confidence(selected_server_id, healthy_servers)
        
        decision = LoadBalancingDecision(
            selected_node=selected_server_id,
            confidence=confidence,
            predicted_response_time=predicted_response_time,
            load_distribution=load_distribution,
            decision_factors=self._get_decision_factors(selected_server),
            timestamp=time.time()
        )
        
        # Record decision for learning
        self.routing_decisions.append(decision)
        
        # Update server connections
        selected_server.active_connections += 1
        
        return decision
    
    async def record_request_result(
        self,
        server_id: str,
        response_time: float,
        success: bool,
        cpu_usage: float = 0.0,
        memory_usage: float = 0.0,
        network_usage: float = 0.0
    ):
        """Record the result of a request for learning."""
        if server_id in self.servers:
            server = self.servers[server_id]
            
            # Update server metrics
            server.update_metrics(response_time, cpu_usage, memory_usage, network_usage)
            server.active_connections = max(0, server.active_connections - 1)
            
            if not success:
                server.error_count += 1
            
            # Record for learning
            self.performance_history.append({
                'timestamp': time.time(),
                'server_id': server_id,
                'response_time': response_time,
                'success': success,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'network_usage': network_usage
            })
            
            # Update global counters
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            # Record traffic for prediction
            current_rps = self._calculate_current_rps()
            self.traffic_predictor.record_traffic(
                time.time(), 
                current_rps, 
                avg_response_time=response_time,
                success_rate=self.successful_requests / max(self.total_requests, 1)
            )
            
            # Adaptive learning
            await self._adaptive_learning_update(server_id, response_time, success)
    
    async def _adaptive_hybrid_selection(
        self, 
        healthy_servers: Dict[str, ServerNode], 
        request_context: Optional[Dict[str, Any]]
    ) -> str:
        """Adaptive hybrid server selection using weighted strategies."""
        
        strategy_scores = {}
        
        # Score each strategy
        strategy_scores[LoadBalancingStrategy.LEAST_CONNECTIONS] = self._least_connections_selection(healthy_servers)
        strategy_scores[LoadBalancingStrategy.LEAST_RESPONSE_TIME] = self._least_response_time_selection(healthy_servers)
        strategy_scores[LoadBalancingStrategy.RESOURCE_BASED] = self._resource_based_selection(healthy_servers)
        strategy_scores[LoadBalancingStrategy.ML_PREDICTED] = await self._ml_predicted_selection(healthy_servers, request_context)
        
        # Weighted combination
        server_scores = defaultdict(float)
        
        for strategy, selected_server in strategy_scores.items():
            weight = self.strategy_weights.get(strategy, 0.0)
            server_scores[selected_server] += weight
        
        # Select server with highest combined score
        best_server = max(server_scores.keys(), key=lambda s: server_scores[s])
        return best_server
    
    async def _ml_predicted_selection(
        self, 
        healthy_servers: Dict[str, ServerNode], 
        request_context: Optional[Dict[str, Any]]
    ) -> str:
        """ML-based server selection using learned patterns."""
        
        server_scores = {}
        
        for server_id, server in healthy_servers.items():
            # Feature vector for this server
            features = self._extract_server_features(server, request_context)
            
            # Predict performance score
            predicted_score = self._predict_server_performance(features)
            
            # Combine with current metrics
            current_load = server.get_load_score()
            health_factor = server.health_score
            
            # Final score
            server_scores[server_id] = predicted_score * health_factor * (1.0 - current_load)
        
        # Select best scoring server
        best_server = max(server_scores.keys(), key=lambda s: server_scores[s])
        return best_server
    
    def _resource_based_selection(self, healthy_servers: Dict[str, ServerNode]) -> str:
        """Select server based on resource utilization."""
        server_scores = {}
        
        for server_id, server in healthy_servers.items():
            # Resource score (lower is better)
            cpu_score = 1.0 - server.cpu_utilization
            memory_score = 1.0 - server.memory_utilization
            network_score = 1.0 - server.network_utilization
            
            resource_score = (cpu_score + memory_score + network_score) / 3.0
            server_scores[server_id] = resource_score * server.health_score
        
        return max(server_scores.keys(), key=lambda s: server_scores[s])
    
    def _least_response_time_selection(self, healthy_servers: Dict[str, ServerNode]) -> str:
        """Select server with lowest average response time."""
        server_scores = {}
        
        for server_id, server in healthy_servers.items():
            avg_response_time = server.average_response_time
            # Invert response time (lower is better)
            score = 1000.0 / max(avg_response_time, 1.0)
            server_scores[server_id] = score * server.health_score
        
        return max(server_scores.keys(), key=lambda s: server_scores[s])
    
    def _least_connections_selection(self, healthy_servers: Dict[str, ServerNode]) -> str:
        """Select server with least active connections."""
        return min(
            healthy_servers.keys(),
            key=lambda s: healthy_servers[s].active_connections / max(healthy_servers[s].capacity, 1)
        )
    
    def _round_robin_selection(self, healthy_servers: Dict[str, ServerNode]) -> str:
        """Simple round-robin selection."""
        server_ids = list(healthy_servers.keys())
        index = self.total_requests % len(server_ids)
        return server_ids[index]
    
    def _extract_server_features(self, server: ServerNode, request_context: Optional[Dict[str, Any]]) -> np.ndarray:
        """Extract features for ML prediction."""
        features = [
            server.cpu_utilization,
            server.memory_utilization,
            server.network_utilization,
            server.get_load_score(),
            server.health_score,
            server.active_connections / max(server.capacity, 1),
            server.average_response_time / 1000.0,  # Normalize to seconds
            server.error_rate,
            time.time() % 86400 / 86400,  # Time of day
            len(server.response_times) / 100.0  # History richness
        ]
        
        # Pad to fixed size
        while len(features) < 15:
            features.append(0.0)
        
        return np.array(features[:15])
    
    def _predict_server_performance(self, features: np.ndarray) -> float:
        """Predict server performance score."""
        # Simple performance prediction
        # In practice, this would use a trained ML model
        
        # Weighted feature combination
        weights = np.array([
            -0.3,  # CPU (negative - high CPU is bad)
            -0.3,  # Memory (negative - high memory is bad)
            -0.2,  # Network (negative - high network is bad)
            -0.4,  # Load score (negative - high load is bad)
            0.5,   # Health score (positive - high health is good)
            -0.3,  # Connection ratio (negative - high connections are bad)
            -0.4,  # Response time (negative - high response time is bad)
            -0.5,  # Error rate (negative - high error rate is bad)
            0.0,   # Time of day (neutral)
            0.1    # History richness (positive - more data is good)
        ])
        
        # Extend weights to match feature size
        while len(weights) < len(features):
            weights = np.append(weights, 0.0)
        
        score = np.dot(features, weights[:len(features)])
        
        # Sigmoid activation to bound score between 0 and 1
        return 1.0 / (1.0 + np.exp(-score))
    
    def _predict_response_time(self, server: ServerNode, request_context: Optional[Dict[str, Any]]) -> float:
        """Predict response time for a server."""
        base_response_time = server.average_response_time
        
        # Adjust based on current load
        load_factor = 1.0 + server.get_load_score()
        
        # Adjust based on resource utilization
        resource_factor = 1.0 + max(server.cpu_utilization, server.memory_utilization)
        
        predicted_time = base_response_time * load_factor * resource_factor
        
        return max(1.0, predicted_time)  # Minimum 1ms
    
    def _calculate_decision_confidence(self, selected_server: str, healthy_servers: Dict[str, ServerNode]) -> float:
        """Calculate confidence in the load balancing decision."""
        selected = healthy_servers[selected_server]
        
        # Compare with other servers
        selected_score = selected.health_score * (1.0 - selected.get_load_score())
        
        other_scores = [
            server.health_score * (1.0 - server.get_load_score())
            for server_id, server in healthy_servers.items()
            if server_id != selected_server
        ]
        
        if not other_scores:
            return 1.0
        
        avg_other_score = np.mean(other_scores)
        max_other_score = max(other_scores)
        
        # Confidence based on relative performance
        if max_other_score > 0:
            confidence = selected_score / max_other_score
        else:
            confidence = 1.0
        
        return min(1.0, max(0.0, confidence))
    
    def _get_decision_factors(self, server: ServerNode) -> Dict[str, float]:
        """Get factors that influenced the decision."""
        return {
            'health_score': server.health_score,
            'load_score': server.get_load_score(),
            'cpu_utilization': server.cpu_utilization,
            'memory_utilization': server.memory_utilization,
            'network_utilization': server.network_utilization,
            'active_connections': server.active_connections / max(server.capacity, 1),
            'average_response_time': server.average_response_time,
            'error_rate': server.error_rate
        }
    
    def _calculate_current_rps(self) -> float:
        """Calculate current requests per second."""
        if len(self.performance_history) < 2:
            return 1.0
        
        # Count requests in last 60 seconds
        current_time = time.time()
        recent_requests = [
            r for r in self.performance_history
            if current_time - r['timestamp'] <= 60
        ]
        
        return len(recent_requests) / 60.0
    
    async def _adaptive_learning_update(self, server_id: str, response_time: float, success: bool):
        """Update strategy weights based on performance."""
        # Find recent decisions for this server
        recent_decisions = [
            d for d in self.routing_decisions
            if d.selected_node == server_id and time.time() - d.timestamp <= 300
        ]
        
        if not recent_decisions:
            return
        
        # Calculate performance vs prediction
        actual_performance = 1.0 / max(response_time, 1.0) if success else 0.0
        
        for decision in recent_decisions:
            predicted_performance = 1.0 / max(decision.predicted_response_time, 1.0)
            
            # Calculate prediction error
            error = abs(actual_performance - predicted_performance)
            
            # Adjust strategy weights (simplified)
            if error < 0.1:  # Good prediction
                # Increase weight of strategies that led to this decision
                for strategy in self.strategy_weights:
                    if strategy in decision.decision_factors:
                        self.strategy_weights[strategy] *= 1.01
            else:  # Poor prediction
                # Decrease weight of strategies that led to this decision
                for strategy in self.strategy_weights:
                    if strategy in decision.decision_factors:
                        self.strategy_weights[strategy] *= 0.99
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight
    
    async def _monitoring_loop(self):
        """Background monitoring and health checking."""
        while True:
            try:
                current_time = time.time()
                
                # Check server health
                for server_id, server in self.servers.items():
                    if current_time - server.last_heartbeat > 60:
                        server.health_score *= 0.9  # Decay health for inactive servers
                    
                    # Remove old response times
                    while (server.response_times and 
                           current_time - server.response_times[0] > 3600):
                        server.response_times.popleft()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _prediction_loop(self):
        """Background traffic prediction."""
        while True:
            try:
                # Generate traffic predictions
                prediction = self.traffic_predictor.predict_traffic(3600.0)  # 1 hour horizon
                
                # Log predictions for monitoring
                logging.info(f"Traffic prediction: {prediction.predicted_requests_per_second:.1f} RPS")
                
                if prediction.scaling_recommendations:
                    logging.info(f"Scaling recommendations: {prediction.scaling_recommendations}")
                
                await asyncio.sleep(300)  # Predict every 5 minutes
                
            except Exception as e:
                logging.error(f"Prediction loop error: {e}")
                await asyncio.sleep(600)
    
    async def _optimization_loop(self):
        """Background optimization of load balancing parameters."""
        while True:
            try:
                # Analyze recent performance
                if len(self.performance_history) > 100:
                    await self._optimize_strategy_weights()
                
                await asyncio.sleep(600)  # Optimize every 10 minutes
                
            except Exception as e:
                logging.error(f"Optimization loop error: {e}")
                await asyncio.sleep(1200)
    
    async def _optimize_strategy_weights(self):
        """Optimize strategy weights based on recent performance."""
        recent_performance = list(self.performance_history)[-100:]
        
        # Group by server and calculate average performance
        server_performance = defaultdict(list)
        for record in recent_performance:
            server_id = record['server_id']
            performance_score = 1.0 / max(record['response_time'], 1.0) if record['success'] else 0.0
            server_performance[server_id].append(performance_score)
        
        # Calculate strategy effectiveness
        strategy_effectiveness = defaultdict(list)
        
        for decision in list(self.routing_decisions)[-50:]:
            server_id = decision.selected_node
            if server_id in server_performance:
                avg_performance = np.mean(server_performance[server_id])
                
                # Attribute performance to decision factors
                for factor, value in decision.decision_factors.items():
                    if value > 0.5:  # Factor was significant
                        strategy_effectiveness[factor].append(avg_performance)
        
        # Update weights based on effectiveness
        for factor, performances in strategy_effectiveness.items():
            if len(performances) > 5:
                avg_performance = np.mean(performances)
                
                # Map decision factors to strategies (simplified)
                if factor in ['health_score', 'load_score']:
                    # Adjust ML strategy weight
                    if avg_performance > 0.8:
                        self.strategy_weights[LoadBalancingStrategy.ML_PREDICTED] *= 1.05
                    else:
                        self.strategy_weights[LoadBalancingStrategy.ML_PREDICTED] *= 0.95
                
                elif factor in ['cpu_utilization', 'memory_utilization']:
                    # Adjust resource-based strategy weight
                    if avg_performance > 0.8:
                        self.strategy_weights[LoadBalancingStrategy.RESOURCE_BASED] *= 1.05
                    else:
                        self.strategy_weights[LoadBalancingStrategy.RESOURCE_BASED] *= 0.95
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight
    
    def get_traffic_prediction(self, horizon_seconds: float = 3600.0) -> TrafficPrediction:
        """Get traffic prediction for capacity planning."""
        return self.traffic_predictor.predict_traffic(horizon_seconds)
    
    def get_load_balancer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        current_time = time.time()
        
        # Server statistics
        server_stats = {}
        for server_id, server in self.servers.items():
            server_stats[server_id] = {
                'health_score': server.health_score,
                'load_score': server.get_load_score(),
                'active_connections': server.active_connections,
                'capacity_utilization': server.active_connections / max(server.capacity, 1),
                'average_response_time': server.average_response_time,
                'error_rate': server.error_rate,
                'cpu_utilization': server.cpu_utilization,
                'memory_utilization': server.memory_utilization,
                'last_heartbeat_age': current_time - server.last_heartbeat
            }
        
        # Overall statistics
        total_capacity = sum(server.capacity for server in self.servers.values())
        total_active_connections = sum(server.active_connections for server in self.servers.values())
        
        success_rate = self.successful_requests / max(self.total_requests, 1)
        current_rps = self._calculate_current_rps()
        
        # Strategy performance
        recent_performance = list(self.performance_history)[-100:]
        avg_response_time = np.mean([r['response_time'] for r in recent_performance]) if recent_performance else 0
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'current_rps': current_rps,
            'average_response_time': avg_response_time,
            'total_capacity': total_capacity,
            'total_active_connections': total_active_connections,
            'capacity_utilization': total_active_connections / max(total_capacity, 1),
            'strategy': self.strategy.value,
            'strategy_weights': dict(self.strategy_weights),
            'server_statistics': server_stats,
            'recent_decisions': len(self.routing_decisions)
        }