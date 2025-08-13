"""
Self-Healing Infrastructure for Autonomous System Recovery.

Automatically detects, diagnoses, and recovers from system failures
and performance degradations without human intervention.
"""

import asyncio
import json
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import logging

from ..quantum_planner.performance import PerformanceMonitor


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Health monitoring metric."""
    name: str
    current_value: float
    threshold_warning: float
    threshold_critical: float
    threshold_failed: float
    unit: str
    description: str
    last_updated: datetime = field(default_factory=datetime.now)
    status: HealthStatus = HealthStatus.HEALTHY
    trend: str = "stable"  # increasing, decreasing, stable, volatile


@dataclass
class SystemIncident:
    """System incident record."""
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    affected_components: List[str]
    detection_time: datetime
    resolution_time: Optional[datetime] = None
    recovery_actions: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    status: str = "open"  # open, investigating, resolved, closed
    auto_resolved: bool = False


@dataclass
class RecoveryAction:
    """Recovery action definition."""
    id: str
    name: str
    description: str
    applicable_conditions: List[str]
    executor: Callable
    success_criteria: Dict[str, float]
    max_attempts: int = 3
    cooldown_seconds: float = 300.0
    risk_level: str = "low"  # low, medium, high


class SelfHealingInfrastructure:
    """
    Self-healing infrastructure system.
    
    Features:
    - Continuous health monitoring
    - Anomaly detection and alerting
    - Automated incident detection and classification
    - Self-recovery through predefined actions
    - Root cause analysis
    - Learning from recovery patterns
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        health_check_interval: float = 30.0,
        anomaly_detection_window: int = 20,
        logger: Optional[logging.Logger] = None,
    ):
        self.performance_monitor = performance_monitor
        self.health_check_interval = health_check_interval
        self.anomaly_detection_window = anomaly_detection_window
        self.logger = logger or logging.getLogger(__name__)
        
        # Health monitoring
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._initialize_health_metrics()
        
        # Incident management
        self.active_incidents: Dict[str, SystemIncident] = {}
        self.incident_history: List[SystemIncident] = []
        self.incident_counter = 0
        
        # Recovery system
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self._initialize_recovery_actions()
        
        # Anomaly detection
        self.anomaly_baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        
        # System state
        self.is_running = False
        self.system_health_status = HealthStatus.HEALTHY
        self.last_health_check: Optional[datetime] = None
        
        # Performance tracking
        self.recovery_success_rate = 0.0
        self.mean_time_to_recovery = 0.0
        self.incidents_prevented = 0
        
    async def start_self_healing_system(self):
        """Start the self-healing infrastructure system."""
        self.is_running = True
        self.logger.info("Starting self-healing infrastructure system")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._anomaly_detection_loop()),
            asyncio.create_task(self._incident_management_loop()),
            asyncio.create_task(self._recovery_execution_loop()),
            asyncio.create_task(self._baseline_learning_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Self-healing system error: {e}")
        finally:
            self.is_running = False
    
    async def stop_self_healing_system(self):
        """Stop the self-healing system."""
        self.is_running = False
        self.logger.info("Stopping self-healing infrastructure system")
    
    def _initialize_health_metrics(self):
        """Initialize system health metrics."""
        metrics = [
            HealthMetric("cpu_usage", 0.0, 70.0, 85.0, 95.0, "%", "CPU utilization percentage"),
            HealthMetric("memory_usage", 0.0, 80.0, 90.0, 95.0, "%", "Memory utilization percentage"),
            HealthMetric("disk_usage", 0.0, 85.0, 95.0, 98.0, "%", "Disk space utilization"),
            HealthMetric("network_latency", 0.0, 100.0, 200.0, 500.0, "ms", "Network round-trip latency"),
            HealthMetric("error_rate", 0.0, 1.0, 5.0, 10.0, "%", "System error rate percentage"),
            HealthMetric("throughput", 0.0, 10.0, 5.0, 1.0, "ops/s", "System throughput (inverse threshold)"),
            HealthMetric("federated_accuracy", 0.0, 0.7, 0.5, 0.3, "score", "Federated learning accuracy"),
            HealthMetric("communication_efficiency", 0.0, 0.6, 0.4, 0.2, "score", "Communication efficiency"),
            HealthMetric("agent_connectivity", 0.0, 0.8, 0.6, 0.4, "ratio", "Connected agents ratio"),
            HealthMetric("quantum_coherence", 0.0, 5.0, 2.0, 1.0, "time", "Quantum coherence time"),
        ]
        
        for metric in metrics:
            self.health_metrics[metric.name] = metric
            self.metric_history[metric.name] = []
    
    def _initialize_recovery_actions(self):
        """Initialize automated recovery actions."""
        actions = [
            RecoveryAction(
                "restart_failed_agents",
                "Restart Failed Agents",
                "Restart federated learning agents that have stopped responding",
                ["agent_connectivity < 0.8", "error_rate > 5.0"],
                self._restart_failed_agents,
                {"agent_connectivity": 0.9, "error_rate": 2.0},
                max_attempts=3,
                risk_level="low"
            ),
            RecoveryAction(
                "scale_up_resources",
                "Scale Up Resources", 
                "Increase computational resources when system is under load",
                ["cpu_usage > 85.0", "memory_usage > 90.0"],
                self._scale_up_resources,
                {"cpu_usage": 70.0, "memory_usage": 80.0},
                max_attempts=2,
                risk_level="medium"
            ),
            RecoveryAction(
                "optimize_communication",
                "Optimize Communication",
                "Reduce communication overhead when network is congested",
                ["network_latency > 200.0", "communication_efficiency < 0.4"],
                self._optimize_communication,
                {"network_latency": 150.0, "communication_efficiency": 0.6},
                max_attempts=2,
                risk_level="low"
            ),
            RecoveryAction(
                "reset_quantum_state",
                "Reset Quantum State",
                "Reset quantum coherence when quantum subsystem is degraded",
                ["quantum_coherence < 2.0"],
                self._reset_quantum_state,
                {"quantum_coherence": 8.0},
                max_attempts=1,
                risk_level="low"
            ),
            RecoveryAction(
                "emergency_failover",
                "Emergency Failover",
                "Switch to backup systems during critical failures",
                ["system_health == FAILED"],
                self._emergency_failover,
                {"system_health": "HEALTHY"},
                max_attempts=1,
                risk_level="high"
            ),
            RecoveryAction(
                "garbage_collection",
                "Garbage Collection",
                "Clean up memory and temporary resources",
                ["memory_usage > 85.0"],
                self._perform_garbage_collection,
                {"memory_usage": 70.0},
                max_attempts=1,
                risk_level="low"
            ),
        ]
        
        for action in actions:
            self.recovery_actions[action.id] = action
    
    async def _health_monitoring_loop(self):
        """Continuously monitor system health metrics."""
        while self.is_running:
            try:
                await self._collect_health_metrics()
                await self._update_health_status()
                self.last_health_check = datetime.now()
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_health_metrics(self):
        """Collect current health metrics from system."""
        current_time = datetime.now()
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # Performance metrics from monitoring system
        perf_metrics = await self.performance_monitor.get_current_metrics()
        
        # Update health metrics
        metric_updates = {
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "disk_usage": disk_percent,
            "network_latency": perf_metrics.get("latency", 50.0),
            "error_rate": perf_metrics.get("error_rate", 0.5),
            "throughput": perf_metrics.get("throughput", 15.0),
            "federated_accuracy": perf_metrics.get("accuracy", 0.8),
            "communication_efficiency": perf_metrics.get("communication_efficiency", 0.7),
            "agent_connectivity": perf_metrics.get("agent_connectivity", 0.95),
            "quantum_coherence": perf_metrics.get("quantum_coherence", 10.0),
        }
        
        for metric_name, value in metric_updates.items():
            if metric_name in self.health_metrics:
                metric = self.health_metrics[metric_name]
                metric.current_value = value
                metric.last_updated = current_time
                
                # Update status based on thresholds
                metric.status = self._determine_metric_status(metric)
                
                # Update trend
                metric.trend = self._calculate_metric_trend(metric_name)
                
                # Store history
                self.metric_history[metric_name].append((current_time, value))
                
                # Keep only recent history
                if len(self.metric_history[metric_name]) > 1000:
                    self.metric_history[metric_name] = self.metric_history[metric_name][-1000:]
    
    def _determine_metric_status(self, metric: HealthMetric) -> HealthStatus:
        """Determine health status based on metric thresholds."""
        value = metric.current_value
        
        # Handle inverse metrics (where lower values are worse)
        if metric.name in ["throughput", "federated_accuracy", "communication_efficiency", "agent_connectivity", "quantum_coherence"]:
            if value <= metric.threshold_failed:
                return HealthStatus.FAILED
            elif value <= metric.threshold_critical:
                return HealthStatus.CRITICAL
            elif value <= metric.threshold_warning:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
        else:
            # Normal metrics (where higher values are worse)
            if value >= metric.threshold_failed:
                return HealthStatus.FAILED
            elif value >= metric.threshold_critical:
                return HealthStatus.CRITICAL
            elif value >= metric.threshold_warning:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
    
    def _calculate_metric_trend(self, metric_name: str) -> str:
        """Calculate metric trend from recent history."""
        history = self.metric_history.get(metric_name, [])
        
        if len(history) < 5:
            return "stable"
        
        recent_values = [value for _, value in history[-10:]]
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        # Calculate volatility
        volatility = np.std(recent_values) / (np.mean(recent_values) + 1e-8)
        
        if volatility > 0.2:
            return "volatile"
        elif slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    async def _update_health_status(self):
        """Update overall system health status."""
        critical_metrics = [m for m in self.health_metrics.values() if m.status == HealthStatus.CRITICAL]
        failed_metrics = [m for m in self.health_metrics.values() if m.status == HealthStatus.FAILED]
        warning_metrics = [m for m in self.health_metrics.values() if m.status == HealthStatus.WARNING]
        
        if failed_metrics:
            self.system_health_status = HealthStatus.FAILED
        elif critical_metrics:
            self.system_health_status = HealthStatus.CRITICAL
        elif warning_metrics:
            self.system_health_status = HealthStatus.WARNING
        else:
            self.system_health_status = HealthStatus.HEALTHY
        
        # Log status changes
        if hasattr(self, '_last_system_status') and self._last_system_status != self.system_health_status:
            self.logger.info(f"System health status changed: {self._last_system_status.value} -> {self.system_health_status.value}")
        
        self._last_system_status = self.system_health_status
    
    async def _anomaly_detection_loop(self):
        """Detect anomalies in system metrics."""
        while self.is_running:
            try:
                await self._detect_anomalies()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Anomaly detection loop error: {e}")
                await asyncio.sleep(120)
    
    async def _detect_anomalies(self):
        """Detect anomalies using statistical methods."""
        current_time = datetime.now()
        
        for metric_name, metric in self.health_metrics.items():
            history = self.metric_history.get(metric_name, [])
            
            if len(history) < self.anomaly_detection_window:
                continue
            
            recent_values = [value for _, value in history[-self.anomaly_detection_window:]]
            
            # Calculate baseline statistics
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)
            
            # Detect anomalies using z-score
            if std_value > 0:
                z_score = abs((metric.current_value - mean_value) / std_value)
                
                # Anomaly threshold
                anomaly_threshold = 2.5
                
                if z_score > anomaly_threshold and metric.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                    await self._create_anomaly_incident(metric_name, metric.current_value, z_score)
    
    async def _create_anomaly_incident(self, metric_name: str, value: float, z_score: float):
        """Create incident for detected anomaly."""
        incident_id = f"anomaly_{metric_name}_{int(time.time())}"
        
        # Check if similar incident already exists
        existing_incidents = [
            inc for inc in self.active_incidents.values()
            if metric_name in inc.affected_components and inc.status == "open"
        ]
        
        if existing_incidents:
            return  # Don't create duplicate incidents
        
        severity = IncidentSeverity.HIGH if z_score > 3.0 else IncidentSeverity.MEDIUM
        
        incident = SystemIncident(
            id=incident_id,
            title=f"Anomaly detected in {metric_name}",
            description=f"Metric {metric_name} showing anomalous value {value:.2f} (z-score: {z_score:.2f})",
            severity=severity,
            affected_components=[metric_name],
            detection_time=datetime.now(),
        )
        
        self.active_incidents[incident_id] = incident
        self.logger.warning(f"Anomaly incident created: {incident_id}")
    
    async def _incident_management_loop(self):
        """Manage system incidents and trigger recovery actions."""
        while self.is_running:
            try:
                await self._process_active_incidents()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Incident management loop error: {e}")
                await asyncio.sleep(60)
    
    async def _process_active_incidents(self):
        """Process and classify active incidents."""
        for incident_id, incident in list(self.active_incidents.items()):
            if incident.status == "open":
                # Analyze incident and determine recovery actions
                recovery_actions = self._determine_recovery_actions(incident)
                
                if recovery_actions:
                    incident.status = "investigating"
                    incident.recovery_actions = [action.id for action in recovery_actions]
                    
                    self.logger.info(f"Incident {incident_id}: Recovery actions determined: {incident.recovery_actions}")
                    
                    # Schedule recovery actions
                    for action in recovery_actions:
                        await self._schedule_recovery_action(action, incident_id)
    
    def _determine_recovery_actions(self, incident: SystemIncident) -> List[RecoveryAction]:
        """Determine appropriate recovery actions for an incident."""
        applicable_actions = []
        
        for action in self.recovery_actions.values():
            for condition in action.applicable_conditions:
                if self._evaluate_condition(condition):
                    applicable_actions.append(action)
                    break
        
        # Sort by risk level (prefer low risk actions)
        risk_priority = {"low": 0, "medium": 1, "high": 2}
        applicable_actions.sort(key=lambda a: risk_priority.get(a.risk_level, 3))
        
        return applicable_actions[:3]  # Limit to 3 actions per incident
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a recovery action condition."""
        try:
            # Parse condition (e.g., "cpu_usage > 85.0")
            for metric_name, metric in self.health_metrics.items():
                condition = condition.replace(metric_name, str(metric.current_value))
            
            condition = condition.replace("system_health", f"'{self.system_health_status.value}'")
            
            return eval(condition)
        except Exception as e:
            self.logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    async def _recovery_execution_loop(self):
        """Execute recovery actions."""
        while self.is_running:
            try:
                # This would be implemented with a proper job queue
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Recovery execution loop error: {e}")
                await asyncio.sleep(120)
    
    async def _schedule_recovery_action(self, action: RecoveryAction, incident_id: str):
        """Schedule a recovery action for execution."""
        try:
            self.logger.info(f"Executing recovery action: {action.name} for incident {incident_id}")
            
            # Execute the action
            success = await action.executor()
            
            # Record the execution
            execution_record = {
                "timestamp": datetime.now(),
                "action_id": action.id,
                "incident_id": incident_id,
                "success": success,
                "execution_time": time.time(),
            }
            
            self.recovery_history.append(execution_record)
            
            if success:
                self.logger.info(f"Recovery action {action.name} succeeded")
                await self._check_recovery_success(action, incident_id)
            else:
                self.logger.warning(f"Recovery action {action.name} failed")
            
        except Exception as e:
            self.logger.error(f"Failed to execute recovery action {action.name}: {e}")
    
    async def _check_recovery_success(self, action: RecoveryAction, incident_id: str):
        """Check if recovery action was successful."""
        await asyncio.sleep(30)  # Wait for metrics to update
        
        success = True
        for metric_name, target_value in action.success_criteria.items():
            if metric_name in self.health_metrics:
                current_value = self.health_metrics[metric_name].current_value
                
                # Check if metric improved to target
                if metric_name in ["throughput", "federated_accuracy", "communication_efficiency", "agent_connectivity", "quantum_coherence"]:
                    success = success and current_value >= target_value
                else:
                    success = success and current_value <= target_value
        
        if success and incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.status = "resolved"
            incident.resolution_time = datetime.now()
            incident.auto_resolved = True
            
            # Move to history
            self.incident_history.append(incident)
            del self.active_incidents[incident_id]
            
            self.logger.info(f"Incident {incident_id} automatically resolved")
    
    async def _baseline_learning_loop(self):
        """Learn normal behavior baselines for improved anomaly detection."""
        while self.is_running:
            try:
                await self._update_baselines()
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                self.logger.error(f"Baseline learning loop error: {e}")
                await asyncio.sleep(1800)
    
    async def _update_baselines(self):
        """Update baseline statistics for metrics."""
        for metric_name, history in self.metric_history.items():
            if len(history) < 100:
                continue
            
            # Use recent healthy periods for baseline
            recent_values = []
            for timestamp, value in history[-1000:]:
                # Only include values when system was healthy
                if self._was_system_healthy_at(timestamp):
                    recent_values.append(value)
            
            if len(recent_values) >= 50:
                self.anomaly_baselines[metric_name] = {
                    "mean": np.mean(recent_values),
                    "std": np.std(recent_values),
                    "percentile_95": np.percentile(recent_values, 95),
                    "percentile_5": np.percentile(recent_values, 5),
                }
    
    def _was_system_healthy_at(self, timestamp: datetime) -> bool:
        """Check if system was healthy at given timestamp."""
        # Simplified check - in practice would store historical health status
        return True
    
    # Recovery action implementations
    async def _restart_failed_agents(self) -> bool:
        """Restart failed federated learning agents."""
        try:
            self.logger.info("Restarting failed agents...")
            # This would integrate with the agent management system
            await asyncio.sleep(2)  # Simulate restart time
            return True
        except Exception as e:
            self.logger.error(f"Failed to restart agents: {e}")
            return False
    
    async def _scale_up_resources(self) -> bool:
        """Scale up computational resources."""
        try:
            self.logger.info("Scaling up resources...")
            # This would integrate with resource management system
            await asyncio.sleep(5)  # Simulate scaling time
            return True
        except Exception as e:
            self.logger.error(f"Failed to scale up resources: {e}")
            return False
    
    async def _optimize_communication(self) -> bool:
        """Optimize communication protocols."""
        try:
            self.logger.info("Optimizing communication protocols...")
            # This would adjust communication parameters
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Failed to optimize communication: {e}")
            return False
    
    async def _reset_quantum_state(self) -> bool:
        """Reset quantum coherence state."""
        try:
            self.logger.info("Resetting quantum state...")
            # This would reset quantum planning state
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset quantum state: {e}")
            return False
    
    async def _emergency_failover(self) -> bool:
        """Perform emergency failover to backup systems."""
        try:
            self.logger.critical("Performing emergency failover...")
            # This would switch to backup infrastructure
            await asyncio.sleep(10)  # Simulate failover time
            return True
        except Exception as e:
            self.logger.error(f"Failed to perform emergency failover: {e}")
            return False
    
    async def _perform_garbage_collection(self) -> bool:
        """Perform garbage collection and cleanup."""
        try:
            self.logger.info("Performing garbage collection...")
            # This would clean up memory and resources
            await asyncio.sleep(2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to perform garbage collection: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return {
            "overall_status": self.system_health_status.value,
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "metrics": {
                name: {
                    "value": metric.current_value,
                    "status": metric.status.value,
                    "trend": metric.trend,
                    "unit": metric.unit,
                }
                for name, metric in self.health_metrics.items()
            },
            "active_incidents": len(self.active_incidents),
            "total_incidents": len(self.incident_history),
            "recovery_success_rate": self.recovery_success_rate,
            "is_running": self.is_running,
        }
    
    def get_incident_summary(self) -> Dict[str, Any]:
        """Get incident management summary."""
        return {
            "active_incidents": [
                {
                    "id": inc.id,
                    "title": inc.title,
                    "severity": inc.severity.value,
                    "status": inc.status,
                    "detection_time": inc.detection_time.isoformat(),
                    "affected_components": inc.affected_components,
                }
                for inc in self.active_incidents.values()
            ],
            "recent_incidents": [
                {
                    "id": inc.id,
                    "title": inc.title,
                    "severity": inc.severity.value,
                    "auto_resolved": inc.auto_resolved,
                    "resolution_time": inc.resolution_time.isoformat() if inc.resolution_time else None,
                }
                for inc in self.incident_history[-10:]
            ],
            "recovery_actions_available": len(self.recovery_actions),
            "recent_recoveries": len([r for r in self.recovery_history[-20:] if r["success"]]),
        }