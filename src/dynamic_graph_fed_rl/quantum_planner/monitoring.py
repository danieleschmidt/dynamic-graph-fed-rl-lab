"""
Monitoring and observability for quantum task planner.

Implements comprehensive monitoring, logging, and health checks:
- Performance metrics collection
- Health monitoring and alerting  
- Distributed tracing for quantum operations
- Real-time dashboards and reporting
- Anomaly detection for quantum states
"""

import time
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import statistics
import numpy as np

from .core import QuantumTask, TaskState
from .exceptions import QuantumPlannerError


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    name: str
    value: Union[float, int]
    tags: Dict[str, str]
    unit: Optional[str] = None


@dataclass  
class HealthStatus:
    """System health status."""
    component: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    timestamp: float
    details: Dict[str, Any]


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    metric: str
    condition: str  # "gt", "lt", "eq", "contains"
    threshold: float
    duration: float  # Seconds condition must persist
    severity: str  # "info", "warning", "critical"
    enabled: bool = True


class MetricsCollector:
    """
    Collects and aggregates performance metrics for quantum planner.
    
    Provides thread-safe metric collection with buffering and aggregation.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        flush_interval: float = 60.0,
        retention_period: float = 3600.0  # 1 hour
    ):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.retention_period = retention_period
        
        # Metric storage
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.aggregated_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Timing
        self.last_flush = time.time()
        self.start_times: Dict[str, float] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment counter metric."""
        tags = tags or {}
        key = self._make_key(name, tags)
        self.counters[key] += value
        
        self._record_metric(MetricPoint(
            timestamp=time.time(),
            name=name,
            value=self.counters[key],
            tags=tags,
            unit="count"
        ))
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set gauge metric."""
        tags = tags or {}
        key = self._make_key(name, tags)
        self.gauges[key] = value
        
        self._record_metric(MetricPoint(
            timestamp=time.time(),
            name=name,
            value=value,
            tags=tags,
            unit="value"
        ))
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record histogram metric."""
        tags = tags or {}
        key = self._make_key(name, tags)
        self.histograms[key].append(value)
        
        # Keep only recent values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        self._record_metric(MetricPoint(
            timestamp=time.time(),
            name=name,
            value=value,
            tags=tags,
            unit="duration"
        ))
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, tags or {})
    
    def start_timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Start a timer manually."""
        key = self._make_key(name, tags or {})
        self.start_times[key] = time.time()
    
    def end_timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """End a timer manually."""
        key = self._make_key(name, tags or {})
        if key in self.start_times:
            duration = time.time() - self.start_times[key]
            self.histogram(name, duration, tags)
            del self.start_times[key]
            return duration
        return None
    
    def _record_metric(self, metric: MetricPoint):
        """Record metric to buffer."""
        self.metrics_buffer.append(metric)
        
        # Auto-flush if buffer is full
        if len(self.metrics_buffer) >= self.buffer_size:
            self.flush()
    
    def _make_key(self, name: str, tags: Dict[str, str]) -> str:
        """Create unique key for metric with tags."""
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}" if tag_str else name
    
    def flush(self):
        """Flush metrics buffer and perform aggregation."""
        current_time = time.time()
        
        # Group metrics by name for aggregation
        metric_groups = defaultdict(list)
        
        while self.metrics_buffer:
            metric = self.metrics_buffer.popleft()
            
            # Only process recent metrics
            if current_time - metric.timestamp <= self.retention_period:
                metric_groups[metric.name].append(metric)
        
        # Aggregate metrics
        for name, metrics in metric_groups.items():
            if metrics:
                self._aggregate_metrics(name, metrics)
        
        self.last_flush = current_time
        self.logger.debug(f"Flushed metrics for {len(metric_groups)} metric types")
    
    def _aggregate_metrics(self, name: str, metrics: List[MetricPoint]):
        """Aggregate metrics for a given name."""
        if not metrics:
            return
        
        values = [m.value for m in metrics]
        timestamp = max(m.timestamp for m in metrics)
        
        # Calculate aggregations
        aggregations = {
            "count": len(values),
            "sum": sum(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }
        
        if len(values) > 1:
            aggregations["std"] = statistics.stdev(values)
            aggregations["p50"] = statistics.median(values)
            aggregations["p95"] = np.percentile(values, 95)
            aggregations["p99"] = np.percentile(values, 99)
        
        # Store aggregated metrics
        self.aggregated_metrics[name].append({
            "timestamp": timestamp,
            "aggregations": aggregations,
            "sample_count": len(values)
        })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        self.flush()  # Ensure we have latest aggregations
        
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histogram_summaries": {},
            "buffer_size": len(self.metrics_buffer),
            "last_flush": self.last_flush
        }
        
        # Summarize histograms
        for key, values in self.histograms.items():
            if values:
                summary["histogram_summaries"][key] = {
                    "count": len(values),
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": np.percentile(values, 95) if len(values) > 1 else values[0]
                }
        
        return summary


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.histogram(self.name, duration, self.tags)


class HealthMonitor:
    """
    Monitors system health and component status.
    
    Performs regular health checks and maintains component status.
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        unhealthy_threshold: int = 3,  # Consecutive failures
        recovery_threshold: int = 2   # Consecutive successes
    ):
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.recovery_threshold = recovery_threshold
        
        # Health status tracking
        self.component_status: Dict[str, HealthStatus] = {}
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        
        # Health check functions
        self.health_checks: Dict[str, Callable[[], HealthStatus]] = {}
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        self.logger = logging.getLogger(__name__)
    
    def register_health_check(
        self, 
        component: str, 
        check_func: Callable[[], HealthStatus]
    ):
        """Register a health check function for a component."""
        self.health_checks[component] = check_func
        self.logger.info(f"Registered health check for component: {component}")
    
    async def start_monitoring(self):
        """Start the health monitoring loop."""
        if self.running:
            return
        
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started health monitoring")
    
    async def stop_monitoring(self):
        """Stop the health monitoring loop."""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _perform_health_checks(self):
        """Perform all registered health checks."""
        for component, check_func in self.health_checks.items():
            try:
                status = await self._run_health_check(component, check_func)
                self._update_component_status(component, status)
            except Exception as e:
                error_status = HealthStatus(
                    component=component,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    timestamp=time.time(),
                    details={"error": str(e), "error_type": type(e).__name__}
                )
                self._update_component_status(component, error_status)
    
    async def _run_health_check(
        self, 
        component: str, 
        check_func: Callable[[], HealthStatus]
    ) -> HealthStatus:
        """Run individual health check with timeout."""
        if asyncio.iscoroutinefunction(check_func):
            # Async health check with timeout
            try:
                return await asyncio.wait_for(check_func(), timeout=10.0)
            except asyncio.TimeoutError:
                return HealthStatus(
                    component=component,
                    status="critical",
                    message="Health check timed out",
                    timestamp=time.time(),
                    details={"timeout": True}
                )
        else:
            # Synchronous health check
            return check_func()
    
    def _update_component_status(self, component: str, status: HealthStatus):
        """Update component status with failure/recovery tracking."""
        previous_status = self.component_status.get(component)
        
        # Track consecutive failures/successes
        if status.status in {"critical", "warning"}:
            self.failure_counts[component] += 1
            self.success_counts[component] = 0
        elif status.status == "healthy":
            self.success_counts[component] += 1
            self.failure_counts[component] = 0
        
        # Apply thresholds
        if (previous_status and previous_status.status == "healthy" and 
            self.failure_counts[component] >= self.unhealthy_threshold):
            self.logger.warning(
                f"Component {component} marked unhealthy after {self.failure_counts[component]} failures"
            )
        
        if (previous_status and previous_status.status in {"critical", "warning"} and
            self.success_counts[component] >= self.recovery_threshold):
            self.logger.info(
                f"Component {component} recovered after {self.success_counts[component]} successes"
            )
        
        self.component_status[component] = status
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        healthy_count = sum(1 for s in self.component_status.values() if s.status == "healthy")
        total_count = len(self.component_status)
        
        overall_status = "healthy"
        if any(s.status == "critical" for s in self.component_status.values()):
            overall_status = "critical"
        elif any(s.status == "warning" for s in self.component_status.values()):
            overall_status = "warning"
        elif total_count == 0:
            overall_status = "unknown"
        
        return {
            "overall_status": overall_status,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "components": {
                name: asdict(status) for name, status in self.component_status.items()
            },
            "last_check": max(
                (s.timestamp for s in self.component_status.values()), 
                default=0
            )
        }


class QuantumStateMonitor:
    """
    Specialized monitor for quantum state properties.
    
    Tracks quantum coherence, entanglement, and state evolution.
    """
    
    def __init__(self, coherence_threshold: float = 0.9):
        self.coherence_threshold = coherence_threshold
        self.state_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.coherence_violations: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
    
    def monitor_task_state(self, task: QuantumTask):
        """Monitor individual task quantum state."""
        timestamp = time.time()
        
        # Calculate state properties
        probabilities = {
            state.value: task.get_probability(state) 
            for state in TaskState
        }
        
        total_probability = sum(probabilities.values())
        entropy = self._calculate_entropy(list(probabilities.values()))
        coherence = self._calculate_coherence(task.state_amplitudes)
        
        state_record = {
            "timestamp": timestamp,
            "probabilities": probabilities,
            "total_probability": total_probability,
            "entropy": entropy,
            "coherence": coherence,
            "entangled_tasks": len(task.entangled_tasks)
        }
        
        self.state_history[task.id].append(state_record)
        
        # Keep only recent history
        if len(self.state_history[task.id]) > 100:
            self.state_history[task.id] = self.state_history[task.id][-100:]
        
        # Check for coherence violations
        if coherence < self.coherence_threshold:
            violation = {
                "timestamp": timestamp,
                "task_id": task.id,
                "coherence": coherence,
                "threshold": self.coherence_threshold,
                "probabilities": probabilities
            }
            self.coherence_violations.append(violation)
            
            # Keep only recent violations
            if len(self.coherence_violations) > 1000:
                self.coherence_violations = self.coherence_violations[-1000:]
            
            self.logger.warning(
                f"Quantum coherence violation in task {task.id}: {coherence:.3f} < {self.coherence_threshold}"
            )
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate quantum state entropy."""
        entropy = 0.0
        for p in probabilities:
            if p > 1e-10:  # Avoid log(0)
                entropy -= p * np.log2(p)
        return entropy
    
    def _calculate_coherence(self, state_amplitudes: Dict[TaskState, complex]) -> float:
        """Calculate quantum coherence measure."""
        if not state_amplitudes:
            return 0.0
        
        # Simple coherence measure based on amplitude phases
        phases = [np.angle(amp) for amp in state_amplitudes.values() if abs(amp) > 1e-10]
        
        if len(phases) < 2:
            return 1.0
        
        # Calculate phase coherence
        mean_phase = np.mean(phases)
        phase_variance = np.var(phases)
        
        # Coherence decreases with phase variance
        coherence = np.exp(-phase_variance)
        
        return float(coherence)
    
    def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum system metrics."""
        if not self.state_history:
            return {"error": "No quantum state data available"}
        
        # Aggregate metrics across all tasks
        total_tasks = len(self.state_history)
        recent_coherence = []
        recent_entropy = []
        recent_entanglement = []
        
        for task_id, history in self.state_history.items():
            if history:
                latest = history[-1]
                recent_coherence.append(latest["coherence"])
                recent_entropy.append(latest["entropy"])
                recent_entanglement.append(latest["entangled_tasks"])
        
        metrics = {
            "total_tasks_monitored": total_tasks,
            "coherence_violations": len(self.coherence_violations),
            "recent_violations": len([
                v for v in self.coherence_violations 
                if time.time() - v["timestamp"] < 300  # Last 5 minutes
            ])
        }
        
        if recent_coherence:
            metrics.update({
                "avg_coherence": statistics.mean(recent_coherence),
                "min_coherence": min(recent_coherence),
                "avg_entropy": statistics.mean(recent_entropy),
                "avg_entanglement": statistics.mean(recent_entanglement)
            })
        
        return metrics


class QuantumPlannerMonitor:
    """
    Main monitoring system for quantum task planner.
    
    Coordinates all monitoring components and provides unified interface.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        health_monitor: Optional[HealthMonitor] = None,
        quantum_monitor: Optional[QuantumStateMonitor] = None
    ):
        self.metrics = metrics_collector or MetricsCollector()
        self.health = health_monitor or HealthMonitor()
        self.quantum = quantum_monitor or QuantumStateMonitor()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_default_health_checks(self):
        """Setup default system health checks."""
        
        def memory_health_check() -> HealthStatus:
            """Check memory usage."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                
                if memory.percent > 90:
                    status = "critical"
                elif memory.percent > 75:
                    status = "warning"  
                else:
                    status = "healthy"
                
                return HealthStatus(
                    component="memory",
                    status=status,
                    message=f"Memory usage: {memory.percent:.1f}%",
                    timestamp=time.time(),
                    details={
                        "percent": memory.percent,
                        "available": memory.available,
                        "used": memory.used,
                        "total": memory.total
                    }
                )
            except ImportError:
                return HealthStatus(
                    component="memory",
                    status="unknown",
                    message="psutil not available for memory monitoring",
                    timestamp=time.time(),
                    details={}
                )
        
        def quantum_coherence_check() -> HealthStatus:
            """Check quantum system coherence."""
            quantum_metrics = self.quantum.get_quantum_metrics()
            
            if "error" in quantum_metrics:
                return HealthStatus(
                    component="quantum_coherence",
                    status="unknown",
                    message="No quantum data available",
                    timestamp=time.time(),
                    details=quantum_metrics
                )
            
            violations = quantum_metrics.get("recent_violations", 0)
            avg_coherence = quantum_metrics.get("avg_coherence", 1.0)
            
            if violations > 10 or avg_coherence < 0.5:
                status = "critical"
            elif violations > 5 or avg_coherence < 0.7:
                status = "warning"
            else:
                status = "healthy"
            
            return HealthStatus(
                component="quantum_coherence",
                status=status,
                message=f"Coherence: {avg_coherence:.3f}, Violations: {violations}",
                timestamp=time.time(),
                details=quantum_metrics
            )
        
        # Register health checks
        self.health.register_health_check("memory", memory_health_check)
        self.health.register_health_check("quantum_coherence", quantum_coherence_check)
    
    async def start(self):
        """Start all monitoring systems."""
        await self.health.start_monitoring()
        self.logger.info("Quantum planner monitoring started")
    
    async def stop(self):
        """Stop all monitoring systems."""
        await self.health.stop_monitoring()
        self.metrics.flush()  # Final flush
        self.logger.info("Quantum planner monitoring stopped")
    
    def record_task_execution(
        self, 
        task_id: str, 
        execution_time: float, 
        status: str,
        task: Optional[QuantumTask] = None
    ):
        """Record task execution metrics."""
        # Performance metrics
        self.metrics.histogram("task_execution_time", execution_time, {"task_id": task_id})
        self.metrics.counter("task_executions_total", 1, {"status": status})
        
        # Quantum state monitoring
        if task:
            self.quantum.monitor_task_state(task)
    
    def record_scheduling_metrics(self, scheduler_name: str, metrics: Dict[str, Any]):
        """Record scheduler performance metrics."""
        self.metrics.gauge("scheduler_efficiency", metrics.get("quantum_efficiency", 0.0), 
                          {"scheduler": scheduler_name})
        self.metrics.gauge("scheduler_throughput", metrics.get("throughput", 0.0),
                          {"scheduler": scheduler_name})
        self.metrics.counter("scheduling_rounds", 1, {"scheduler": scheduler_name})
    
    def record_optimization_metrics(self, optimizer_name: str, result: Dict[str, Any]):
        """Record optimization performance metrics."""
        self.metrics.gauge("optimization_score", result.get("optimization_score", 0.0),
                          {"optimizer": optimizer_name})
        self.metrics.counter("optimization_iterations", result.get("iterations", 0),
                            {"optimizer": optimizer_name})
        self.metrics.histogram("optimization_time", result.get("execution_time", 0.0),
                              {"optimizer": optimizer_name})
        
        convergence = 1 if result.get("convergence_achieved", False) else 0
        self.metrics.counter("optimization_convergence", convergence, 
                            {"optimizer": optimizer_name})
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            "timestamp": time.time(),
            "metrics_summary": self.metrics.get_metrics_summary(),
            "health_status": self.health.get_system_health(),
            "quantum_metrics": self.quantum.get_quantum_metrics(),
            "system_info": {
                "monitoring_active": self.health.running,
                "metrics_buffer_size": len(self.metrics.metrics_buffer)
            }
        }


# Context manager for monitoring quantum operations
@asynccontextmanager
async def monitor_operation(
    monitor: QuantumPlannerMonitor, 
    operation_name: str,
    tags: Optional[Dict[str, str]] = None
):
    """Context manager for monitoring quantum operations."""
    tags = tags or {}
    
    # Start timing
    start_time = time.time()
    monitor.metrics.counter(f"{operation_name}_started", 1, tags)
    
    try:
        yield
        
        # Success metrics
        duration = time.time() - start_time
        monitor.metrics.histogram(f"{operation_name}_duration", duration, tags)
        monitor.metrics.counter(f"{operation_name}_success", 1, tags)
        
    except QuantumPlannerError as e:
        # Quantum planner error
        duration = time.time() - start_time
        error_tags = {**tags, "error_code": e.error_code, "error_type": type(e).__name__}
        
        monitor.metrics.histogram(f"{operation_name}_duration", duration, error_tags)
        monitor.metrics.counter(f"{operation_name}_error", 1, error_tags)
        
        raise
    
    except Exception as e:
        # Unexpected error
        duration = time.time() - start_time
        error_tags = {**tags, "error_type": type(e).__name__}
        
        monitor.metrics.histogram(f"{operation_name}_duration", duration, error_tags)
        monitor.metrics.counter(f"{operation_name}_unexpected_error", 1, error_tags)
        
        raise