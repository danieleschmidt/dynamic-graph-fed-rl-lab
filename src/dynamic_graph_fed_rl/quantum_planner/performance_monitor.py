"""
Enhanced Performance Monitor with Generation 4 Integration.

Provides comprehensive performance monitoring and observability for
the Generation 4 AI-Enhanced Auto-Optimization System.
"""

import asyncio
import json
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import logging
from pathlib import Path


class MetricType(Enum):
    """Performance metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


@dataclass
class PerformanceMetric:
    """Performance metric definition."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    history: List[Tuple[datetime, float]] = field(default_factory=list)


@dataclass
class AlertRule:
    """Performance alert rule."""
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 0.1"
    threshold: float
    severity: str  # "warning", "critical"
    enabled: bool = True
    cooldown_seconds: float = 300.0
    last_triggered: Optional[datetime] = None


class PerformanceMonitor:
    """
    Enhanced performance monitor for Generation 4 system.
    
    Features:
    - Real-time metric collection and aggregation
    - Custom metric registration and tracking
    - Performance alerting and notifications
    - Historical data management
    - Dashboard-ready data export
    - Integration with Generation 4 components
    - OpenTelemetry-compatible metrics export
    """
    
    def __init__(
        self,
        collection_interval: float = 30.0,
        history_retention_hours: int = 168,  # 7 days
        alert_callbacks: List[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.collection_interval = collection_interval
        self.history_retention_hours = history_retention_hours
        self.alert_callbacks = alert_callbacks or []
        self.logger = logger or logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.custom_metrics: Dict[str, PerformanceMetric] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Performance tracking
        self.collection_stats = {
            "total_collections": 0,
            "failed_collections": 0,
            "collection_duration_ms": [],
            "last_collection": None,
        }
        
        # System state
        self.is_running = False
        self.deployed_algorithm_id: Optional[str] = None
        self.current_hyperparams: Dict[str, Any] = {}
        
        # Initialize default metrics
        self._initialize_default_metrics()
        self._initialize_default_alerts()
        
    def _initialize_default_metrics(self):
        """Initialize default system metrics."""
        default_metrics = [
            PerformanceMetric("cpu_usage", MetricType.GAUGE, "CPU utilization percentage", "%"),
            PerformanceMetric("memory_usage", MetricType.GAUGE, "Memory utilization percentage", "%"),
            PerformanceMetric("disk_usage", MetricType.GAUGE, "Disk space utilization percentage", "%"),
            PerformanceMetric("network_latency", MetricType.GAUGE, "Network round-trip latency", "ms"),
            PerformanceMetric("error_rate", MetricType.RATE, "System error rate", "errors/sec"),
            PerformanceMetric("throughput", MetricType.GAUGE, "System throughput", "ops/sec"),
            PerformanceMetric("accuracy", MetricType.GAUGE, "Model accuracy", "score"),
            PerformanceMetric("convergence_rate", MetricType.GAUGE, "Training convergence rate", "rate"),
            PerformanceMetric("communication_efficiency", MetricType.GAUGE, "Communication efficiency", "score"),
            PerformanceMetric("resource_utilization", MetricType.GAUGE, "Resource utilization efficiency", "score"),
            PerformanceMetric("stability_score", MetricType.GAUGE, "System stability score", "score"),
            PerformanceMetric("agent_count", MetricType.GAUGE, "Number of active agents", "count"),
            PerformanceMetric("active_agents", MetricType.GAUGE, "Number of actively participating agents", "count"),
            PerformanceMetric("communication_rounds", MetricType.COUNTER, "Total communication rounds", "count"),
            PerformanceMetric("training_episodes", MetricType.COUNTER, "Total training episodes", "count"),
            PerformanceMetric("quantum_measurements", MetricType.COUNTER, "Quantum measurements performed", "count"),
            PerformanceMetric("quantum_coherence", MetricType.GAUGE, "Quantum coherence time", "seconds"),
            PerformanceMetric("superposition_states", MetricType.GAUGE, "Active superposition states", "count"),
            PerformanceMetric("agent_connectivity", MetricType.GAUGE, "Agent connectivity ratio", "ratio"),
            PerformanceMetric("federated_accuracy", MetricType.GAUGE, "Federated learning accuracy", "score"),
        ]
        
        for metric in default_metrics:
            self.metrics[metric.name] = metric
    
    def _initialize_default_alerts(self):
        """Initialize default alert rules."""
        default_alerts = [
            AlertRule("high_cpu_usage", "cpu_usage", "> 90", 90.0, "critical"),
            AlertRule("high_memory_usage", "memory_usage", "> 85", 85.0, "warning"),
            AlertRule("high_error_rate", "error_rate", "> 5", 5.0, "critical"),
            AlertRule("low_throughput", "throughput", "< 5", 5.0, "warning"),
            AlertRule("low_accuracy", "accuracy", "< 0.6", 0.6, "critical"),
            AlertRule("poor_stability", "stability_score", "< 0.7", 0.7, "warning"),
            AlertRule("agent_disconnection", "agent_connectivity", "< 0.8", 0.8, "warning"),
        ]
        
        for alert in default_alerts:
            self.alert_rules[alert.name] = alert
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        self.is_running = True
        self.logger.info("Starting enhanced performance monitoring")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._metric_collection_loop()),
            asyncio.create_task(self._alert_processing_loop()),
            asyncio.create_task(self._data_retention_loop()),
            asyncio.create_task(self._export_metrics_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Performance monitoring error: {e}")
        finally:
            self.is_running = False
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_running = False
        self.logger.info("Stopping performance monitoring")
    
    async def _metric_collection_loop(self):
        """Main metric collection loop."""
        while self.is_running:
            try:
                collection_start = time.time()
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect federated learning metrics
                await self._collect_federated_metrics()
                
                # Collect quantum-inspired metrics
                await self._collect_quantum_metrics()
                
                # Update collection stats
                collection_duration = (time.time() - collection_start) * 1000
                self.collection_stats["collection_duration_ms"].append(collection_duration)
                if len(self.collection_stats["collection_duration_ms"]) > 100:
                    self.collection_stats["collection_duration_ms"].pop(0)
                
                self.collection_stats["total_collections"] += 1
                self.collection_stats["last_collection"] = datetime.now()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.collection_stats["failed_collections"] += 1
                self.logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(min(self.collection_interval, 60))
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            current_time = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._update_metric("cpu_usage", cpu_percent, current_time)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self._update_metric("memory_usage", memory.percent, current_time)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self._update_metric("disk_usage", disk_percent, current_time)
            
            # Network metrics (simulated)
            network_latency = np.random.normal(50, 10)  # Simulated latency
            await self._update_metric("network_latency", network_latency, current_time)
            
        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")
    
    async def _collect_federated_metrics(self):
        """Collect federated learning specific metrics."""
        try:
            current_time = datetime.now()
            
            # Simulate federated learning metrics
            # In practice, these would come from the actual federated system
            
            base_accuracy = 0.8
            accuracy_variance = np.random.normal(0, 0.05)
            accuracy = max(0.0, min(1.0, base_accuracy + accuracy_variance))
            await self._update_metric("accuracy", accuracy, current_time)
            
            # Convergence rate
            convergence_rate = max(0.01, np.random.normal(0.1, 0.02))
            await self._update_metric("convergence_rate", convergence_rate, current_time)
            
            # Communication efficiency
            comm_efficiency = max(0.1, min(1.0, np.random.normal(0.75, 0.1)))
            await self._update_metric("communication_efficiency", comm_efficiency, current_time)
            
            # Throughput
            throughput = max(1, np.random.normal(15, 3))
            await self._update_metric("throughput", throughput, current_time)
            
            # Agent metrics
            agent_count = max(5, int(np.random.normal(10, 2)))
            await self._update_metric("agent_count", agent_count, current_time)
            
            active_agents = max(4, min(agent_count, int(np.random.normal(agent_count * 0.9, 1))))
            await self._update_metric("active_agents", active_agents, current_time)
            
            # Agent connectivity
            connectivity = active_agents / agent_count if agent_count > 0 else 1.0
            await self._update_metric("agent_connectivity", connectivity, current_time)
            
            # Stability score
            stability = max(0.5, min(1.0, np.random.normal(0.85, 0.05)))
            await self._update_metric("stability_score", stability, current_time)
            
        except Exception as e:
            self.logger.error(f"Federated metrics collection error: {e}")
    
    async def _collect_quantum_metrics(self):
        """Collect quantum-inspired system metrics."""
        try:
            current_time = datetime.now()
            
            # Quantum coherence time
            coherence_time = max(1.0, np.random.normal(10.0, 2.0))
            await self._update_metric("quantum_coherence", coherence_time, current_time)
            
            # Superposition states
            superposition_states = max(1, int(np.random.normal(8, 2)))
            await self._update_metric("superposition_states", superposition_states, current_time)
            
            # Quantum measurements (counter)
            current_measurements = self.metrics["quantum_measurements"].value
            new_measurements = current_measurements + np.random.poisson(5)  # 5 measurements per interval
            await self._update_metric("quantum_measurements", new_measurements, current_time)
            
        except Exception as e:
            self.logger.error(f"Quantum metrics collection error: {e}")
    
    async def _update_metric(self, name: str, value: float, timestamp: datetime):
        """Update a metric with new value."""
        if name in self.metrics:
            metric = self.metrics[name]
        elif name in self.custom_metrics:
            metric = self.custom_metrics[name]
        else:
            self.logger.warning(f"Unknown metric: {name}")
            return
        
        # Update metric value
        old_value = metric.value
        metric.value = value
        metric.timestamp = timestamp
        
        # Add to history
        metric.history.append((timestamp, value))
        
        # Maintain history size
        max_history = int(self.history_retention_hours * 3600 / self.collection_interval)
        if len(metric.history) > max_history:
            metric.history = metric.history[-max_history:]
        
        # Handle counter metrics
        if metric.metric_type == MetricType.COUNTER:
            metric.value = max(old_value, value)  # Counters only increase
        elif metric.metric_type == MetricType.RATE:
            # Calculate rate from counter
            if len(metric.history) >= 2:
                prev_time, prev_value = metric.history[-2]
                time_diff = (timestamp - prev_time).total_seconds()
                if time_diff > 0:
                    metric.value = (value - prev_value) / time_diff
    
    async def _alert_processing_loop(self):
        """Process performance alerts."""
        while self.is_running:
            try:
                for alert_name, alert in self.alert_rules.items():
                    if alert.enabled:
                        await self._check_alert(alert)
                
                await asyncio.sleep(60)  # Check alerts every minute
                
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)
    
    async def _check_alert(self, alert: AlertRule):
        """Check if an alert condition is met."""
        try:
            # Get current metric value
            metric_value = None
            if alert.metric_name in self.metrics:
                metric_value = self.metrics[alert.metric_name].value
            elif alert.metric_name in self.custom_metrics:
                metric_value = self.custom_metrics[alert.metric_name].value
            
            if metric_value is None:
                return
            
            # Check condition
            condition_met = self._evaluate_condition(metric_value, alert.condition, alert.threshold)
            
            if condition_met:
                # Check cooldown
                current_time = datetime.now()
                if (alert.last_triggered is None or 
                    (current_time - alert.last_triggered).total_seconds() >= alert.cooldown_seconds):
                    
                    await self._trigger_alert(alert, metric_value)
                    alert.last_triggered = current_time
        
        except Exception as e:
            self.logger.error(f"Alert check error for {alert.name}: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        condition = condition.strip()
        
        if condition.startswith(">"):
            return value > threshold
        elif condition.startswith("<"):
            return value < threshold
        elif condition.startswith(">="):
            return value >= threshold
        elif condition.startswith("<="):
            return value <= threshold
        elif condition.startswith("=="):
            return abs(value - threshold) < 0.001
        elif condition.startswith("!="):
            return abs(value - threshold) >= 0.001
        
        return False
    
    async def _trigger_alert(self, alert: AlertRule, current_value: float):
        """Trigger an alert."""
        alert_data = {
            "alert_name": alert.name,
            "metric_name": alert.metric_name,
            "condition": alert.condition,
            "threshold": alert.threshold,
            "current_value": current_value,
            "severity": alert.severity,
            "timestamp": datetime.now(),
        }
        
        self.logger.warning(
            f"🚨 Alert triggered: {alert.name} | "
            f"{alert.metric_name} = {current_value:.2f} {alert.condition} {alert.threshold}"
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    async def _data_retention_loop(self):
        """Manage data retention and cleanup."""
        while self.is_running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Data retention error: {e}")
                await asyncio.sleep(1800)
    
    async def _cleanup_old_data(self):
        """Clean up old metric data."""
        cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
        
        for metric in list(self.metrics.values()) + list(self.custom_metrics.values()):
            # Remove old history entries
            metric.history = [
                (timestamp, value) for timestamp, value in metric.history
                if timestamp > cutoff_time
            ]
    
    async def _export_metrics_loop(self):
        """Export metrics for external monitoring systems."""
        while self.is_running:
            try:
                await self._export_metrics()
                await asyncio.sleep(300)  # Export every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Metrics export error: {e}")
                await asyncio.sleep(600)
    
    async def _export_metrics(self):
        """Export metrics to files for external consumption."""
        try:
            # Prepare metrics data
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {},
                "collection_stats": self.collection_stats.copy(),
            }
            
            # Add current metric values
            for name, metric in self.metrics.items():
                metrics_data["metrics"][name] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "type": metric.metric_type.value,
                }
            
            # Add custom metrics
            for name, metric in self.custom_metrics.items():
                metrics_data["metrics"][f"custom_{name}"] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "type": metric.metric_type.value,
                }
            
            # Export to file
            export_path = Path("monitoring/metrics_export.json")
            export_path.parent.mkdir(exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Export to Prometheus format
            await self._export_prometheus_format()
            
        except Exception as e:
            self.logger.error(f"Metrics export failed: {e}")
    
    async def _export_prometheus_format(self):
        """Export metrics in Prometheus format."""
        try:
            prometheus_lines = []
            
            for name, metric in self.metrics.items():
                # Add metric help
                prometheus_lines.append(f"# HELP {name} {metric.description}")
                prometheus_lines.append(f"# TYPE {name} {metric.metric_type.value}")
                
                # Add metric value with tags
                tags = ",".join([f'{k}="{v}"' for k, v in metric.tags.items()])
                tag_string = f"{{{tags}}}" if tags else ""
                
                prometheus_lines.append(f"{name}{tag_string} {metric.value}")
            
            # Write to file
            prom_path = Path("monitoring/metrics.prom")
            prom_path.parent.mkdir(exist_ok=True)
            
            with open(prom_path, 'w') as f:
                f.write("\n".join(prometheus_lines))
                
        except Exception as e:
            self.logger.error(f"Prometheus export failed: {e}")
    
    # Public API methods
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str,
        tags: Dict[str, str] = None,
    ) -> bool:
        """Register a custom metric."""
        try:
            metric = PerformanceMetric(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                tags=tags or {},
            )
            
            self.custom_metrics[name] = metric
            self.logger.info(f"Registered custom metric: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register metric {name}: {e}")
            return False
    
    def update_custom_metric(self, name: str, value: float, timestamp: datetime = None) -> bool:
        """Update a custom metric value."""
        try:
            if name not in self.custom_metrics:
                self.logger.warning(f"Custom metric not found: {name}")
                return False
            
            timestamp = timestamp or datetime.now()
            asyncio.create_task(self._update_metric(name, value, timestamp))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update custom metric {name}: {e}")
            return False
    
    def add_alert_rule(self, alert: AlertRule) -> bool:
        """Add a new alert rule."""
        try:
            self.alert_rules[alert.name] = alert
            self.logger.info(f"Added alert rule: {alert.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add alert rule {alert.name}: {e}")
            return False
    
    def remove_alert_rule(self, name: str) -> bool:
        """Remove an alert rule."""
        try:
            if name in self.alert_rules:
                del self.alert_rules[name]
                self.logger.info(f"Removed alert rule: {name}")
                return True
            else:
                self.logger.warning(f"Alert rule not found: {name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove alert rule {name}: {e}")
            return False
    
    async def get_current_metrics(self) -> Dict[str, float]:
        """Get current values of all metrics."""
        metrics = {}
        
        # Add system metrics
        for name, metric in self.metrics.items():
            metrics[name] = metric.value
        
        # Add custom metrics
        for name, metric in self.custom_metrics.items():
            metrics[f"custom_{name}"] = metric.value
        
        return metrics
    
    def get_metric_history(self, name: str, hours: int = 24) -> List[Tuple[datetime, float]]:
        """Get metric history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        metric = None
        if name in self.metrics:
            metric = self.metrics[name]
        elif name in self.custom_metrics:
            metric = self.custom_metrics[name]
        
        if not metric:
            return []
        
        return [
            (timestamp, value) for timestamp, value in metric.history
            if timestamp > cutoff_time
        ]
    
    async def update_hyperparameter(self, param_name: str, value: Any):
        """Update hyperparameter tracking."""
        self.current_hyperparams[param_name] = value
        self.logger.debug(f"Updated hyperparameter: {param_name} = {value}")
    
    async def update_deployed_algorithm(self, algorithm_id: str, hyperparams: Dict[str, Any]):
        """Update deployed algorithm tracking."""
        self.deployed_algorithm_id = algorithm_id
        self.current_hyperparams.update(hyperparams)
        self.logger.info(f"Updated deployed algorithm: {algorithm_id}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        return {
            "is_running": self.is_running,
            "total_metrics": len(self.metrics) + len(self.custom_metrics),
            "system_metrics": len(self.metrics),
            "custom_metrics": len(self.custom_metrics),
            "alert_rules": len(self.alert_rules),
            "collection_stats": self.collection_stats,
            "deployed_algorithm_id": self.deployed_algorithm_id,
            "current_hyperparams_count": len(self.current_hyperparams),
            "retention_hours": self.history_retention_hours,
            "collection_interval": self.collection_interval,
        }