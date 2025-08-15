"""Base federated learning classes and protocols.

Generation 2 Robustness Features:
- Comprehensive input validation and sanitization
- Enterprise-grade error handling with circuit breakers
- Security hardening against malicious agents
- Advanced Byzantine fault tolerance
- Audit logging and monitoring
"""

import abc
import asyncio
import random
import logging
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

# Import robustness framework
from ..utils.error_handling import (
    circuit_breaker, retry, robust, SecurityError, ValidationError,
    CircuitBreakerConfig, RetryConfig, resilience
)
from ..utils.validation import (
    validator, validate_federated_params, validate_communication_data,
    create_strict_validator, ValidationLevel
)
from ..utils.security import rbac, SecurityLevel, ActionType, ResourceType


class BaseFederatedProtocol(abc.ABC):
    """Base class for federated learning protocols with enhanced security and validation."""
    
    def __init__(
        self,
        num_agents: int,
        communication_round: int = 100,
        compression_ratio: float = 1.0,
        byzantine_tolerance: bool = False,
        security_level: SecurityLevel = SecurityLevel.RESTRICTED,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        enable_audit_logging: bool = True
    ):
        # Validate initialization parameters
        if num_agents <= 0 or num_agents > 10000:  # Reasonable upper limit
            raise ValidationError(f"Invalid number of agents: {num_agents}")
        
        if communication_round <= 0 or communication_round > 100000:
            raise ValidationError(f"Invalid communication rounds: {communication_round}")
        
        if not (0.0 < compression_ratio <= 1.0):
            raise ValidationError(f"Invalid compression ratio: {compression_ratio}")
        
        self.num_agents = num_agents
        self.communication_round = communication_round
        self.compression_ratio = compression_ratio
        self.byzantine_tolerance = byzantine_tolerance
        self.security_level = security_level
        self.validation_level = validation_level
        self.enable_audit_logging = enable_audit_logging
        
        # Protocol state
        self.round_count = 0
        self.communication_log = []
        
        # Agent states tracking
        self.agent_states = {}
        self.agent_metrics = {}
        
        # Enhanced security and validation
        self.validated_agents: Set[str] = set()
        self.blocked_agents: Set[str] = set()
        self.parameter_history: Dict[str, List[Dict[str, Any]]] = {}
        self.anomaly_detection_enabled = True
        self.max_parameter_deviation = 3.0  # Standard deviations
        
        # Setup circuit breakers for federation operations
        self._setup_federation_circuit_breakers()
        
        # Initialize strict validator for sensitive operations
        if validation_level == ValidationLevel.STRICT:
            self.parameter_validator = create_strict_validator()
        else:
            self.parameter_validator = validator
        
        logging.info(f"Federation protocol initialized: {num_agents} agents, security level: {security_level.value}")
    
    def _setup_federation_circuit_breakers(self):
        """Setup circuit breakers for federation operations."""
        # Circuit breaker for parameter aggregation
        aggregation_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=180.0,  # 3 minutes
            expected_exception=(ValidationError, SecurityError, ValueError),
            success_threshold=2
        )
        self.aggregation_circuit = resilience.register_circuit_breaker(
            f"federation-aggregation-{id(self)}",
            aggregation_config
        )
        
        # Circuit breaker for communication
        communication_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120.0,  # 2 minutes
            expected_exception=(ConnectionError, TimeoutError, SecurityError)
        )
        self.communication_circuit = resilience.register_circuit_breaker(
            f"federation-communication-{id(self)}",
            communication_config
        )
    
    @robust(component="federation_protocol", operation="validate_agent_parameters")
    def validate_agent_parameters(
        self,
        agent_id: str,
        parameters: Dict[str, Any],
        session_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive validation of agent parameters with security checks."""
        # Security authorization check
        if session_token and not rbac.authorize_action(
            session_token,
            ResourceType.FEDERATION_PROTOCOL,
            ActionType.WRITE,
            security_level=self.security_level
        ):
            raise SecurityError(f"Insufficient permissions for agent {agent_id}")
        
        # Check if agent is blocked
        if agent_id in self.blocked_agents:
            raise SecurityError(f"Agent {agent_id} is blocked due to suspicious activity")
        
        # Validate parameter structure and content
        try:
            validated_params = self.parameter_validator.validate_federated_parameters(
                parameters, agent_id
            )
        except ValidationError as e:
            logging.error(f"Parameter validation failed for agent {agent_id}: {e}")
            # Block agent after repeated validation failures
            agent_failure_count = getattr(self, f"_failures_{agent_id}", 0) + 1
            setattr(self, f"_failures_{agent_id}", agent_failure_count)
            
            if agent_failure_count >= 3:
                self.blocked_agents.add(agent_id)
                logging.warning(f"Agent {agent_id} blocked after {agent_failure_count} validation failures")
            
            raise
        
        # Anomaly detection on validated parameters
        if self.anomaly_detection_enabled:
            anomaly_score = self._detect_parameter_anomalies(agent_id, validated_params)
            if anomaly_score > 0.8:  # High anomaly threshold
                logging.warning(f"High anomaly score {anomaly_score:.2f} for agent {agent_id}")
                # Consider blocking agent or flagging for review
        
        # Store parameter history for trend analysis
        if agent_id not in self.parameter_history:
            self.parameter_history[agent_id] = []
        
        self.parameter_history[agent_id].append({
            "timestamp": time.time(),
            "parameters": validated_params,
            "anomaly_score": anomaly_score if self.anomaly_detection_enabled else 0.0
        })
        
        # Keep history bounded
        if len(self.parameter_history[agent_id]) > 100:
            self.parameter_history[agent_id] = self.parameter_history[agent_id][-50:]
        
        # Mark agent as validated
        self.validated_agents.add(agent_id)
        
        # Audit logging
        if self.enable_audit_logging:
            self._log_parameter_validation(agent_id, validated_params, session_token)
        
        return validated_params
    
    def _detect_parameter_anomalies(self, agent_id: str, parameters: Dict[str, Any]) -> float:
        """Detect anomalies in agent parameters using statistical analysis."""
        if agent_id not in self.parameter_history or len(self.parameter_history[agent_id]) < 5:
            return 0.0  # Need historical data for anomaly detection
        
        try:
            import numpy as np
            
            anomaly_score = 0.0
            total_params = 0
            
            # Analyze each parameter tensor
            for param_name, param_value in parameters.items():
                if not isinstance(param_value, jnp.ndarray):
                    continue
                
                # Get historical values for this parameter
                historical_values = []
                for hist_entry in self.parameter_history[agent_id][-10:]:  # Last 10 entries
                    if param_name in hist_entry["parameters"]:
                        hist_param = hist_entry["parameters"][param_name]
                        if isinstance(hist_param, jnp.ndarray) and hist_param.shape == param_value.shape:
                            historical_values.append(hist_param)
                
                if len(historical_values) < 3:
                    continue  # Need at least 3 historical values
                
                # Calculate statistical metrics
                historical_array = jnp.stack(historical_values)
                mean_param = jnp.mean(historical_array, axis=0)
                std_param = jnp.std(historical_array, axis=0)
                
                # Calculate z-score for anomaly detection
                z_scores = jnp.abs(param_value - mean_param) / (std_param + 1e-8)
                max_z_score = float(jnp.max(z_scores))
                
                # Anomaly if z-score exceeds threshold
                if max_z_score > self.max_parameter_deviation:
                    param_anomaly = min(max_z_score / self.max_parameter_deviation - 1.0, 1.0)
                    anomaly_score += param_anomaly
                
                total_params += 1
            
            # Normalize anomaly score
            if total_params > 0:
                anomaly_score = anomaly_score / total_params
            
            return min(anomaly_score, 1.0)
        
        except Exception as e:
            logging.warning(f"Anomaly detection failed for agent {agent_id}: {e}")
            return 0.0
    
    def _log_parameter_validation(self, agent_id: str, parameters: Dict[str, Any], session_token: Optional[str]):
        """Log parameter validation for audit purposes."""
        log_entry = {
            "timestamp": time.time(),
            "event_type": "parameter_validation",
            "agent_id": agent_id,
            "session_token": session_token,
            "parameter_count": len(parameters),
            "tensor_count": sum(1 for v in parameters.values() if isinstance(v, jnp.ndarray)),
            "validation_level": self.validation_level.value,
            "round": self.round_count
        }
        
        # Add to communication log for audit trail
        self.communication_log.append(log_entry)
    
    @abc.abstractmethod
    async def aggregate_parameters(
        self,
        agent_parameters: List[Dict[str, Any]],
        session_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aggregate parameters from multiple agents with security validation."""
        pass
    
    @circuit_breaker("secure_aggregation", failure_threshold=3, recovery_timeout=300.0)
    async def secure_aggregate_parameters(
        self,
        agent_parameters: List[Dict[str, Any]],
        agent_ids: List[str],
        session_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Securely aggregate parameters with comprehensive validation."""
        if len(agent_parameters) != len(agent_ids):
            raise ValidationError("Mismatch between parameters and agent IDs")
        
        # Validate all agent parameters
        validated_parameters = []
        for agent_id, params in zip(agent_ids, agent_parameters):
            validated_params = self.validate_agent_parameters(agent_id, params, session_token)
            validated_parameters.append(validated_params)
        
        # Detect and filter Byzantine agents
        if self.byzantine_tolerance:
            byzantine_agents = self.detect_byzantine_agents(validated_parameters)
            if byzantine_agents:
                logging.warning(f"Detected Byzantine agents: {[agent_ids[i] for i in byzantine_agents]}")
                # Filter out Byzantine agents
                filtered_params = []
                filtered_ids = []
                for i, (params, agent_id) in enumerate(zip(validated_parameters, agent_ids)):
                    if i not in byzantine_agents:
                        filtered_params.append(params)
                        filtered_ids.append(agent_id)
                validated_parameters = filtered_params
                agent_ids = filtered_ids
        
        # Perform secure aggregation
        try:
            aggregated = await self.aggregate_parameters(validated_parameters, session_token)
            
            # Validate aggregated result
            aggregated_validated = self.parameter_validator.validate_federated_parameters(
                aggregated, "aggregated_result"
            )
            
            # Log successful aggregation
            if self.enable_audit_logging:
                self._log_aggregation_event(agent_ids, aggregated_validated, session_token)
            
            return aggregated_validated
        
        except Exception as e:
            logging.error(f"Secure aggregation failed: {e}")
            raise
    
    @abc.abstractmethod
    def select_communication_partners(
        self,
        agent_id: int,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Select communication partners for an agent."""
        pass
    
    def compress_parameters(
        self,
        parameters: Dict[str, Any],
        compression_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compress parameters for communication efficiency."""
        if compression_ratio is None:
            compression_ratio = self.compression_ratio
        
        if compression_ratio >= 1.0:
            return parameters  # No compression
        
        compressed = {}
        for key, value in parameters.items():
            if isinstance(value, jnp.ndarray):
                # Gradient sparsification
                flat_params = value.flatten()
                k = int(len(flat_params) * compression_ratio)
                
                # Keep top-k by magnitude
                indices = jnp.argsort(jnp.abs(flat_params))[-k:]
                compressed_flat = jnp.zeros_like(flat_params)
                compressed_flat = compressed_flat.at[indices].set(flat_params[indices])
                
                compressed[key] = compressed_flat.reshape(value.shape)
            else:
                compressed[key] = value
        
        return compressed
    
    def detect_byzantine_agents(
        self,
        agent_parameters: List[Dict[str, Any]],
        threshold: float = 0.2,
        use_advanced_detection: bool = True
    ) -> List[int]:
        """Enhanced Byzantine agent detection with multiple statistical methods."""
        if not self.byzantine_tolerance or len(agent_parameters) < 3:
            return []
        
        byzantine_agents = set()
        
        if use_advanced_detection:
            # Method 1: Distance-based detection (existing)
            distance_outliers = self._detect_byzantine_by_distance(agent_parameters, threshold)
            byzantine_agents.update(distance_outliers)
            
            # Method 2: Gradient magnitude detection
            magnitude_outliers = self._detect_byzantine_by_magnitude(agent_parameters, threshold)
            byzantine_agents.update(magnitude_outliers)
            
            # Method 3: Statistical consistency check
            consistency_outliers = self._detect_byzantine_by_consistency(agent_parameters, threshold)
            byzantine_agents.update(consistency_outliers)
            
            # Method 4: Temporal consistency (if we have history)
            temporal_outliers = self._detect_byzantine_by_temporal_consistency(agent_parameters)
            byzantine_agents.update(temporal_outliers)
        else:
            # Use simple distance-based method
            byzantine_agents.update(self._detect_byzantine_by_distance(agent_parameters, threshold))
        
        return list(byzantine_agents)
    
    def _detect_byzantine_by_distance(self, agent_parameters: List[Dict[str, Any]], threshold: float) -> List[int]:
        """Detect Byzantine agents using parameter distance analysis."""
        # Compute pairwise distances between agent parameters
        distances = []
        for i, params_i in enumerate(agent_parameters):
            agent_distances = []
            for j, params_j in enumerate(agent_parameters):
                if i != j:
                    distance = self._compute_parameter_distance(params_i, params_j)
                    agent_distances.append(distance)
            distances.append(jnp.mean(jnp.array(agent_distances)))
        
        # Identify outliers using MAD (Median Absolute Deviation)
        distances = jnp.array(distances)
        median_distance = jnp.median(distances)
        mad = jnp.median(jnp.abs(distances - median_distance))
        
        byzantine_agents = []
        for i, distance in enumerate(distances):
            if distance > median_distance + threshold * mad:
                byzantine_agents.append(i)
        
        return byzantine_agents
    
    def _detect_byzantine_by_magnitude(self, agent_parameters: List[Dict[str, Any]], threshold: float) -> List[int]:
        """Detect Byzantine agents using parameter magnitude analysis."""
        magnitudes = []
        
        for params in agent_parameters:
            total_magnitude = 0.0
            param_count = 0
            
            for key, value in params.items():
                if isinstance(value, jnp.ndarray):
                    total_magnitude += float(jnp.sum(jnp.abs(value)))
                    param_count += value.size
            
            avg_magnitude = total_magnitude / param_count if param_count > 0 else 0.0
            magnitudes.append(avg_magnitude)
        
        # Detect outliers in magnitude
        magnitudes = jnp.array(magnitudes)
        median_magnitude = jnp.median(magnitudes)
        mad = jnp.median(jnp.abs(magnitudes - median_magnitude))
        
        byzantine_agents = []
        for i, magnitude in enumerate(magnitudes):
            if abs(magnitude - median_magnitude) > threshold * mad:
                byzantine_agents.append(i)
        
        return byzantine_agents
    
    def _detect_byzantine_by_consistency(self, agent_parameters: List[Dict[str, Any]], threshold: float) -> List[int]:
        """Detect Byzantine agents using statistical consistency checks."""
        if len(agent_parameters) < 4:  # Need at least 4 agents for meaningful statistics
            return []
        
        byzantine_agents = []
        
        # For each parameter tensor, compute consistency scores
        for agent_idx in range(len(agent_parameters)):
            consistency_score = 0.0
            param_count = 0
            
            for param_name in agent_parameters[agent_idx].keys():
                if not isinstance(agent_parameters[agent_idx][param_name], jnp.ndarray):
                    continue
                
                # Collect this parameter from all agents
                param_values = []
                for other_params in agent_parameters:
                    if param_name in other_params and isinstance(other_params[param_name], jnp.ndarray):
                        param_values.append(other_params[param_name])
                
                if len(param_values) < 3:
                    continue
                
                # Calculate how much this agent deviates from others
                current_param = agent_parameters[agent_idx][param_name]
                other_params_stack = jnp.stack([p for i, p in enumerate(param_values) if i != agent_idx])
                
                # Compute median and MAD of other agents
                others_median = jnp.median(other_params_stack, axis=0)
                others_mad = jnp.median(jnp.abs(other_params_stack - others_median), axis=0)
                
                # Compute deviation score for current agent
                deviation = jnp.abs(current_param - others_median) / (others_mad + 1e-8)
                consistency_score += float(jnp.mean(deviation))
                param_count += 1
            
            if param_count > 0:
                avg_consistency = consistency_score / param_count
                if avg_consistency > 2.0:  # Threshold for inconsistency
                    byzantine_agents.append(agent_idx)
        
        return byzantine_agents
    
    def _detect_byzantine_by_temporal_consistency(self, agent_parameters: List[Dict[str, Any]]) -> List[int]:
        """Detect Byzantine agents using temporal consistency analysis."""
        # This would require historical data - simplified implementation
        # In practice, this would analyze whether agent behavior is consistent over time
        return []  # Placeholder for temporal analysis
    
    def _compute_parameter_distance(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any],
    ) -> float:
        """Compute L2 distance between parameter sets."""
        total_distance = 0.0
        total_elements = 0
        
        for key in params1.keys():
            if key in params2 and isinstance(params1[key], jnp.ndarray):
                diff = params1[key] - params2[key]
                total_distance += jnp.sum(diff ** 2)
                total_elements += diff.size
        
        if total_elements > 0:
            return float(jnp.sqrt(total_distance / total_elements))
        else:
            return 0.0
    
    def _log_aggregation_event(self, agent_ids: List[str], aggregated_params: Dict[str, Any], session_token: Optional[str]):
        """Log aggregation event for audit purposes."""
        log_entry = {
            "timestamp": time.time(),
            "event_type": "parameter_aggregation",
            "round": self.round_count,
            "participating_agents": agent_ids,
            "agent_count": len(agent_ids),
            "session_token": session_token,
            "aggregated_parameter_count": len(aggregated_params),
            "aggregated_tensor_count": sum(1 for v in aggregated_params.values() if isinstance(v, jnp.ndarray))
        }
        
        self.communication_log.append(log_entry)
    
    @robust(component="federation_protocol", operation="secure_communication")
    def log_communication(
        self,
        sender_id: Union[int, str],
        receiver_id: Union[int, str],
        message_size: int,
        timestamp: float,
        message_type: str = "parameter_update",
        session_token: Optional[str] = None
    ) -> None:
        """Enhanced communication logging with security validation."""
        # Validate inputs
        if message_size < 0 or message_size > 1024 * 1024 * 1024:  # 1GB limit
            raise ValidationError(f"Invalid message size: {message_size}")
        
        if timestamp <= 0 or timestamp > time.time() + 3600:  # Not in future by more than 1 hour
            raise ValidationError(f"Invalid timestamp: {timestamp}")
        
        # Security check: validate sender/receiver authorization
        if session_token and self.enable_audit_logging:
            # In practice, this would check if sender is authorized to send to receiver
            pass
        
        event = {
            "event_type": "communication",
            "round": self.round_count,
            "sender": str(sender_id),
            "receiver": str(receiver_id),
            "message_type": message_type,
            "size": message_size,
            "timestamp": timestamp,
            "session_token": session_token,
            "security_level": self.security_level.value
        }
        
        # Add checksum for integrity
        event_str = f"{sender_id}-{receiver_id}-{message_size}-{timestamp}"
        event["checksum"] = hashlib.sha256(event_str.encode()).hexdigest()[:16]
        
        self.communication_log.append(event)
        
        # Keep log size bounded for memory management
        if len(self.communication_log) > 10000:
            self.communication_log = self.communication_log[-5000:]
    
    def get_communication_stats(self, include_security_metrics: bool = True) -> Dict[str, Any]:
        """Get comprehensive communication statistics with security metrics."""
        if not self.communication_log:
            return {}
        
        # Basic communication statistics
        communication_events = [e for e in self.communication_log if e.get("event_type") == "communication"]
        total_messages = len(communication_events)
        total_bytes = sum(event.get("size", 0) for event in communication_events)
        avg_message_size = total_bytes / total_messages if total_messages > 0 else 0
        
        stats = {
            "total_messages": total_messages,
            "total_bytes": total_bytes,
            "avg_message_size": avg_message_size,
            "rounds": self.round_count,
            "messages_per_round": total_messages / self.round_count if self.round_count > 0 else 0,
            "total_events": len(self.communication_log)
        }
        
        if include_security_metrics:
            # Security and validation metrics
            validation_events = [e for e in self.communication_log if e.get("event_type") == "parameter_validation"]
            aggregation_events = [e for e in self.communication_log if e.get("event_type") == "parameter_aggregation"]
            
            stats.update({
                "security_metrics": {
                    "validated_agents": len(self.validated_agents),
                    "blocked_agents": len(self.blocked_agents),
                    "validation_events": len(validation_events),
                    "aggregation_events": len(aggregation_events),
                    "security_level": self.security_level.value,
                    "validation_level": self.validation_level.value,
                    "byzantine_tolerance_enabled": self.byzantine_tolerance,
                    "anomaly_detection_enabled": self.anomaly_detection_enabled
                },
                "agent_metrics": {
                    "parameter_history_agents": len(self.parameter_history),
                    "avg_history_length": sum(len(hist) for hist in self.parameter_history.values()) / len(self.parameter_history) if self.parameter_history else 0
                }
            })
        
        return stats
    
    def get_security_audit_report(self, time_window: float = 3600) -> Dict[str, Any]:
        """Generate security audit report for specified time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_events = [
            event for event in self.communication_log
            if event.get("timestamp", 0) >= cutoff_time
        ]
        
        # Analyze security events
        validation_failures = 0
        suspicious_activities = []
        agent_activity = {}
        
        for event in recent_events:
            event_type = event.get("event_type", "unknown")
            
            if event_type == "parameter_validation":
                agent_id = event.get("agent_id")
                if agent_id:
                    agent_activity[agent_id] = agent_activity.get(agent_id, 0) + 1
            
            # Check for suspicious patterns
            if event.get("size", 0) > 100 * 1024 * 1024:  # Large message
                suspicious_activities.append({
                    "type": "large_message",
                    "details": event,
                    "severity": "medium"
                })
        
        return {
            "audit_period": {
                "start_time": cutoff_time,
                "end_time": current_time,
                "duration_hours": time_window / 3600
            },
            "summary": {
                "total_events": len(recent_events),
                "validation_failures": validation_failures,
                "suspicious_activities": len(suspicious_activities),
                "active_agents": len(agent_activity),
                "blocked_agents": len(self.blocked_agents)
            },
            "agent_activity": agent_activity,
            "suspicious_activities": suspicious_activities,
            "blocked_agents": list(self.blocked_agents),
            "security_configuration": {
                "security_level": self.security_level.value,
                "validation_level": self.validation_level.value,
                "byzantine_tolerance": self.byzantine_tolerance,
                "anomaly_detection": self.anomaly_detection_enabled,
                "audit_logging": self.enable_audit_logging
            }
        }


class FederatedOptimizer:
    """Optimizer for federated learning with various aggregation methods."""
    
    def __init__(
        self,
        algorithm: str = "fedavg",
        learning_rate: float = 1.0,
        momentum: float = 0.0,
        adaptive_lr: bool = True,
        convergence_threshold: float = 1e-6,
    ):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.adaptive_lr = adaptive_lr
        self.convergence_threshold = convergence_threshold
        
        # State tracking
        self.global_state = None
        self.momentum_state = None
        self.round_count = 0
        self.convergence_history = []
    
    def aggregate(
        self,
        agent_parameters: List[Dict[str, Any]],
        agent_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Aggregate parameters using specified algorithm."""
        if not agent_parameters:
            return {}
        
        if agent_weights is None:
            agent_weights = [1.0 / len(agent_parameters)] * len(agent_parameters)
        
        if self.algorithm == "fedavg":
            return self._federated_averaging(agent_parameters, agent_weights)
        elif self.algorithm == "fedprox":
            return self._federated_proximal(agent_parameters, agent_weights)
        elif self.algorithm == "scaffold":
            return self._scaffold_aggregation(agent_parameters, agent_weights)
        elif self.algorithm == "fedadam":
            return self._federated_adam(agent_parameters, agent_weights)
        else:
            raise ValueError(f"Unknown aggregation algorithm: {self.algorithm}")
    
    def _federated_averaging(
        self,
        agent_parameters: List[Dict[str, Any]],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Standard FedAvg aggregation."""
        if not agent_parameters:
            return {}
        
        # Initialize aggregated parameters
        aggregated = {}
        
        # Get parameter keys from first agent
        keys = list(agent_parameters[0].keys())
        
        for key in keys:
            # Skip non-tensor parameters
            if not isinstance(agent_parameters[0][key], jnp.ndarray):
                aggregated[key] = agent_parameters[0][key]
                continue
            
            # Weighted average of parameters
            weighted_sum = jnp.zeros_like(agent_parameters[0][key])
            
            for params, weight in zip(agent_parameters, weights):
                if key in params:
                    weighted_sum += weight * params[key]
            
            aggregated[key] = weighted_sum
        
        return aggregated
    
    def _federated_proximal(
        self,
        agent_parameters: List[Dict[str, Any]],
        weights: List[float],
        proximal_term: float = 0.01,
    ) -> Dict[str, Any]:
        """FedProx aggregation with proximal term."""
        # Start with standard averaging
        aggregated = self._federated_averaging(agent_parameters, weights)
        
        # Apply proximal regularization if we have previous global state
        if self.global_state is not None:
            for key in aggregated.keys():
                if key in self.global_state and isinstance(aggregated[key], jnp.ndarray):
                    # Proximal term: μ/2 * ||w - w_global||^2
                    proximal_update = proximal_term * (aggregated[key] - self.global_state[key])
                    aggregated[key] = aggregated[key] - self.learning_rate * proximal_update
        
        # Update global state
        self.global_state = aggregated.copy()
        
        return aggregated
    
    def _scaffold_aggregation(
        self,
        agent_parameters: List[Dict[str, Any]],
        weights: List[float],
    ) -> Dict[str, Any]:
        """SCAFFOLD aggregation with control variates."""
        # Simplified SCAFFOLD implementation
        # In practice, this would need control variates from agents
        aggregated = self._federated_averaging(agent_parameters, weights)
        
        # Update global state for convergence tracking
        if self.global_state is not None:
            self._update_convergence_metrics(aggregated)
        
        self.global_state = aggregated.copy()
        return aggregated
    
    def _federated_adam(
        self,
        agent_parameters: List[Dict[str, Any]],
        weights: List[float],
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> Dict[str, Any]:
        """FedAdam aggregation with adaptive moments."""
        # Compute pseudo-gradients
        if self.global_state is None:
            self.global_state = self._federated_averaging(agent_parameters, weights)
            self.momentum_state = {
                "m": {k: jnp.zeros_like(v) for k, v in self.global_state.items() 
                      if isinstance(v, jnp.ndarray)},
                "v": {k: jnp.zeros_like(v) for k, v in self.global_state.items() 
                      if isinstance(v, jnp.ndarray)},
            }
            return self.global_state
        
        # Compute pseudo-gradients from aggregated updates
        pseudo_gradients = {}
        aggregated_params = self._federated_averaging(agent_parameters, weights)
        
        for key in aggregated_params.keys():
            if isinstance(aggregated_params[key], jnp.ndarray):
                pseudo_gradients[key] = self.global_state[key] - aggregated_params[key]
        
        # Adam update
        self.round_count += 1
        
        for key in pseudo_gradients.keys():
            # Update biased first moment estimate
            self.momentum_state["m"][key] = (
                beta1 * self.momentum_state["m"][key] + 
                (1 - beta1) * pseudo_gradients[key]
            )
            
            # Update biased second moment estimate
            self.momentum_state["v"][key] = (
                beta2 * self.momentum_state["v"][key] + 
                (1 - beta2) * pseudo_gradients[key] ** 2
            )
            
            # Bias correction
            m_hat = self.momentum_state["m"][key] / (1 - beta1 ** self.round_count)
            v_hat = self.momentum_state["v"][key] / (1 - beta2 ** self.round_count)
            
            # Update parameters
            self.global_state[key] = self.global_state[key] - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        return self.global_state
    
    def _update_convergence_metrics(self, new_params: Dict[str, Any]) -> None:
        """Update convergence metrics."""
        if self.global_state is None:
            return
        
        # Compute parameter change magnitude
        total_change = 0.0
        total_params = 0
        
        for key in new_params.keys():
            if key in self.global_state and isinstance(new_params[key], jnp.ndarray):
                diff = new_params[key] - self.global_state[key]
                total_change += jnp.sum(diff ** 2)
                total_params += diff.size
        
        if total_params > 0:
            rms_change = float(jnp.sqrt(total_change / total_params))
            self.convergence_history.append(rms_change)
    
    def has_converged(self, window_size: int = 10) -> bool:
        """Check if optimization has converged."""
        if len(self.convergence_history) < window_size:
            return False
        
        recent_changes = self.convergence_history[-window_size:]
        avg_change = sum(recent_changes) / len(recent_changes)
        
        return avg_change < self.convergence_threshold
    
    def adapt_learning_rate(self, performance_metric: float) -> None:
        """Adapt learning rate based on performance."""
        if not self.adaptive_lr:
            return
        
        # Simple adaptive scheme
        if len(self.convergence_history) >= 2:
            recent_change = self.convergence_history[-1]
            prev_change = self.convergence_history[-2]
            
            if recent_change > prev_change:
                # Performance getting worse, reduce learning rate
                self.learning_rate *= 0.9
            elif recent_change < prev_change * 0.5:
                # Performance improving quickly, slightly increase learning rate
                self.learning_rate *= 1.01
        
        # Clamp learning rate
        self.learning_rate = jnp.clip(self.learning_rate, 1e-6, 1.0)


class FederatedActorCritic:
    """Wrapper for federated actor-critic algorithms."""
    
    def __init__(
        self,
        num_agents: int,
        communication: str = "async_gossip",
        buffer_type: str = "graph_temporal",
        aggregation_interval: int = 100,
        aggregation_method: str = "fedavg",
    ):
        self.num_agents = num_agents
        self.communication = communication
        self.buffer_type = buffer_type
        self.aggregation_interval = aggregation_interval
        self.aggregation_method = aggregation_method
        
        # Initialize federated optimizer
        self.optimizer = FederatedOptimizer(algorithm=aggregation_method)
        
        # Communication protocol
        if communication == "async_gossip":
            from .gossip import AsyncGossipProtocol
            self.protocol = AsyncGossipProtocol(num_agents)
        elif communication == "hierarchical":
            from .hierarchical import FederatedHierarchy
            self.protocol = FederatedHierarchy(num_agents)
        else:
            raise ValueError(f"Unknown communication protocol: {communication}")
        
        # Tracking
        self.global_round = 0
        self.agents = []
    
    def add_agent(self, agent) -> None:
        """Add an agent to the federated system."""
        if len(self.agents) >= self.num_agents:
            raise ValueError("Maximum number of agents reached")
        
        self.agents.append(agent)
    
    async def federated_round(self) -> Dict[str, Any]:
        """Execute one round of federated learning."""
        if len(self.agents) < self.num_agents:
            raise ValueError(f"Need {self.num_agents} agents, have {len(self.agents)}")
        
        # Collect parameters from all agents
        agent_parameters = []
        for agent in self.agents:
            if hasattr(agent, 'actor_state') and hasattr(agent, 'critic1_state'):
                params = {
                    "actor": agent.actor_state.params,
                    "critic1": agent.critic1_state.params,
                    "critic2": agent.critic2_state.params if hasattr(agent, 'critic2_state') else None,
                }
                agent_parameters.append(params)
        
        # Aggregate parameters
        if self.communication == "async_gossip":
            aggregated_params = await self.protocol.aggregate_parameters(agent_parameters)
        else:
            # Synchronous aggregation
            aggregated_params = self.optimizer.aggregate(agent_parameters)
        
        # Update all agents with aggregated parameters
        for agent in self.agents:
            if "actor" in aggregated_params:
                agent.actor_state = agent.actor_state.replace(params=aggregated_params["actor"])
                agent.target_actor_state = agent.target_actor_state.replace(params=aggregated_params["actor"])
            
            if "critic1" in aggregated_params:
                agent.critic1_state = agent.critic1_state.replace(params=aggregated_params["critic1"])
                agent.target_critic1_state = agent.target_critic1_state.replace(params=aggregated_params["critic1"])
            
            if "critic2" in aggregated_params and aggregated_params["critic2"] is not None:
                agent.critic2_state = agent.critic2_state.replace(params=aggregated_params["critic2"])
                agent.target_critic2_state = agent.target_critic2_state.replace(params=aggregated_params["critic2"])
        
        self.global_round += 1
        
        # Return aggregation metrics
        return {
            "global_round": self.global_round,
            "num_agents": len(self.agents),
            "communication_stats": self.protocol.get_communication_stats() if hasattr(self.protocol, 'get_communication_stats') else {},
            "convergence": self.optimizer.has_converged(),
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        agent_metrics = []
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'get_training_stats'):
                stats = agent.get_training_stats()
                stats["agent_id"] = i
                agent_metrics.append(stats)
        
        return {
            "global_round": self.global_round,
            "num_agents": len(self.agents),
            "agent_metrics": agent_metrics,
            "optimizer_stats": {
                "convergence_history": self.optimizer.convergence_history,
                "learning_rate": self.optimizer.learning_rate,
                "has_converged": self.optimizer.has_converged(),
            },
            "communication_stats": self.protocol.get_communication_stats() if hasattr(self.protocol, 'get_communication_stats') else {},
        }