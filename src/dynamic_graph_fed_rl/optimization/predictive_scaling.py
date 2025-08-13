"""
Predictive Scaling Based on Federated Learning Workload Patterns.

Uses machine learning to predict resource needs and automatically scale
infrastructure based on federated learning workload patterns.
"""

import asyncio
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from ..quantum_planner.performance import PerformanceMonitor


class ScalingAction(Enum):
    """Scaling action types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    SCALE_OUT = "scale_out"  # Add more instances
    SCALE_IN = "scale_in"    # Remove instances
    NO_ACTION = "no_action"


class ResourceType(Enum):
    """Resource types for scaling."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    AGENTS = "agents"


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
    Predictive scaling system for federated learning workloads.
    
    Features:
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
    ):
        self.performance_monitor = performance_monitor
        self.prediction_horizons = prediction_horizons or [5, 15, 30, 60]  # Minutes
        self.scaling_cooldown = scaling_cooldown
        self.cost_optimization_weight = cost_optimization_weight
        self.logger = logger or logging.getLogger(__name__)
        
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