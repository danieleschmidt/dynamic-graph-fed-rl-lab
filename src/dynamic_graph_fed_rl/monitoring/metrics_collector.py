"""Advanced metrics collection and system monitoring."""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import psutil
import json
from pathlib import Path


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    quantum_coherence: float = 0.0
    federation_sync: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'network_io': self.network_io,
            'quantum_coherence': self.quantum_coherence,
            'federation_sync': self.federation_sync,
            'active_tasks': self.active_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
        }


class MetricsCollector:
    """
    Robust metrics collection system with automatic aggregation and alerting.
    
    Features:
    - Real-time system monitoring
    - Custom metric tracking
    - Automatic threshold alerting
    - Data persistence and export
    - Performance anomaly detection
    """
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        retention_hours: int = 24,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_persistence: bool = True,
        data_directory: str = "metrics_data"
    ):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self.enable_persistence = enable_persistence
        self.data_directory = Path(data_directory)
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=int(retention_hours * 3600 / collection_interval))
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Dict[str, Any]] = []
        
        # Collection state
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Initialize data directory
        if self.enable_persistence:
            self.data_directory.mkdir(exist_ok=True)
    
    def _default_thresholds(self) -> Dict[str, float]:
        """Default alert thresholds."""
        return {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'disk_usage': 95.0,
            'quantum_coherence': 0.3,
            'federation_sync': 0.5,
            'task_failure_rate': 0.2,
        }
    
    def start_collection(self):
        """Start automatic metrics collection."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="MetricsCollector"
        )
        self.collection_thread.start()
        print("📊 Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        print("📊 Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                metrics = self._collect_system_metrics()
                
                with self.lock:
                    self.metrics_history.append(metrics)
                    self._check_thresholds(metrics)
                    
                    if self.enable_persistence:
                        self._persist_metrics(metrics)
                
            except Exception as e:
                print(f"❌ Metrics collection error: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': float(net_io.bytes_sent),
                'bytes_recv': float(net_io.bytes_recv),
                'packets_sent': float(net_io.packets_sent),
                'packets_recv': float(net_io.packets_recv),
            }
            
        except Exception as e:
            print(f"⚠️  System metrics collection failed: {e}")
            # Fallback values
            cpu_usage = 0.0
            memory_usage = 0.0
            disk_usage = 0.0
            network_io = {'bytes_sent': 0.0, 'bytes_recv': 0.0, 'packets_sent': 0.0, 'packets_recv': 0.0}
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
        )
    
    def add_custom_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Add custom application metric."""
        metric_data = {
            'timestamp': time.time(),
            'value': value,
            'tags': tags or {}
        }
        
        with self.lock:
            self.custom_metrics[name].append(metric_data)
    
    def update_quantum_metrics(self, coherence: float, task_stats: Dict[str, int]):
        """Update quantum-specific metrics."""
        if self.metrics_history:
            latest = self.metrics_history[-1]
            latest.quantum_coherence = coherence
            latest.active_tasks = task_stats.get('active', 0)
            latest.completed_tasks = task_stats.get('completed', 0)
            latest.failed_tasks = task_stats.get('failed', 0)
    
    def update_federation_metrics(self, sync_quality: float):
        """Update federation-specific metrics."""
        if self.metrics_history:
            latest = self.metrics_history[-1]
            latest.federation_sync = sync_quality
    
    def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against alert thresholds."""
        alerts_triggered = []
        
        # System resource alerts
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts_triggered.append(('high_cpu', {'value': metrics.cpu_usage, 'threshold': self.alert_thresholds['cpu_usage']}))
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts_triggered.append(('high_memory', {'value': metrics.memory_usage, 'threshold': self.alert_thresholds['memory_usage']}))
        
        if metrics.disk_usage > self.alert_thresholds['disk_usage']:
            alerts_triggered.append(('high_disk', {'value': metrics.disk_usage, 'threshold': self.alert_thresholds['disk_usage']}))
        
        # Application-specific alerts
        if metrics.quantum_coherence < self.alert_thresholds['quantum_coherence']:
            alerts_triggered.append(('low_quantum_coherence', {'value': metrics.quantum_coherence, 'threshold': self.alert_thresholds['quantum_coherence']}))
        
        if metrics.federation_sync < self.alert_thresholds['federation_sync']:
            alerts_triggered.append(('low_federation_sync', {'value': metrics.federation_sync, 'threshold': self.alert_thresholds['federation_sync']}))
        
        # Task failure rate
        total_tasks = metrics.completed_tasks + metrics.failed_tasks
        if total_tasks > 0:
            failure_rate = metrics.failed_tasks / total_tasks
            if failure_rate > self.alert_thresholds['task_failure_rate']:
                alerts_triggered.append(('high_task_failure', {'value': failure_rate, 'threshold': self.alert_thresholds['task_failure_rate']}))
        
        # Process alerts
        for alert_type, alert_data in alerts_triggered:
            self._trigger_alert(alert_type, alert_data)
    
    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger an alert."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'data': alert_data,
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.alerts.append(alert)
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert)
            except Exception as e:
                print(f"❌ Alert callback failed: {e}")
        
        print(f"🚨 ALERT [{alert['severity']}]: {alert_type} - {alert_data}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get severity level for alert type."""
        high_severity = ['high_disk', 'high_task_failure']
        medium_severity = ['high_cpu', 'high_memory', 'low_quantum_coherence']
        
        if alert_type in high_severity:
            return 'HIGH'
        elif alert_type in medium_severity:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add alert notification callback."""
        self.alert_callbacks.append(callback)
    
    def _persist_metrics(self, metrics: SystemMetrics):
        """Persist metrics to disk."""
        try:
            timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(metrics.timestamp))
            filename = self.data_directory / f"metrics_{timestamp_str}.json"
            
            with open(filename, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
        
        except Exception as e:
            print(f"❌ Failed to persist metrics: {e}")
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics."""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, hours: float = 1.0) -> Dict[str, Any]:
        """Get aggregated metrics summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage for m in recent_metrics) / len(recent_metrics)
        avg_quantum = sum(m.quantum_coherence for m in recent_metrics) / len(recent_metrics)
        avg_federation = sum(m.federation_sync for m in recent_metrics) / len(recent_metrics)
        
        # Calculate totals
        total_completed = max(m.completed_tasks for m in recent_metrics) if recent_metrics else 0
        total_failed = max(m.failed_tasks for m in recent_metrics) if recent_metrics else 0
        
        return {
            'time_period_hours': hours,
            'sample_count': len(recent_metrics),
            'avg_cpu_usage': round(avg_cpu, 2),
            'avg_memory_usage': round(avg_memory, 2),
            'avg_disk_usage': round(avg_disk, 2),
            'avg_quantum_coherence': round(avg_quantum, 3),
            'avg_federation_sync': round(avg_federation, 3),
            'total_completed_tasks': total_completed,
            'total_failed_tasks': total_failed,
            'task_success_rate': round(total_completed / max(total_completed + total_failed, 1), 3),
            'recent_alerts': len([a for a in self.alerts if a['timestamp'] >= cutoff_time])
        }
    
    def export_metrics(self, filename: str, format: str = 'json'):
        """Export collected metrics to file."""
        export_data = {
            'export_timestamp': time.time(),
            'collection_interval': self.collection_interval,
            'metrics_count': len(self.metrics_history),
            'metrics': [m.to_dict() for m in self.metrics_history],
            'custom_metrics': dict(self.custom_metrics),
            'alerts': self.alerts,
            'summary': self.get_metrics_summary(24.0)  # 24 hour summary
        }
        
        if format.lower() == 'json':
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"📊 Metrics exported to {filename}")
    
    def clear_old_data(self):
        """Clear old metric data beyond retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self.lock:
            # Remove old metrics
            while self.metrics_history and self.metrics_history[0].timestamp < cutoff_time:
                self.metrics_history.popleft()
            
            # Remove old alerts
            self.alerts = [a for a in self.alerts if a['timestamp'] >= cutoff_time]
        
        print(f"🧹 Cleared metrics data older than {self.retention_hours} hours")