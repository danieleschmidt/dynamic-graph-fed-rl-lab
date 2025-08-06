"""
Test suite for quantum planner monitoring components.

Tests health checks, performance monitoring, and alerting systems.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import jax.numpy as jnp

from src.dynamic_graph_fed_rl.quantum_planner.monitoring import (
    QuantumHealthMonitor,
    PerformanceMonitor,
    AlertManager,
    HealthStatus,
    MetricsCollector,
    QuantumMetrics
)


class TestQuantumHealthMonitor:
    """Test quantum system health monitoring."""
    
    @pytest.fixture
    def monitor(self):
        """Create test health monitor."""
        return QuantumHealthMonitor(
            check_interval=1.0,
            coherence_threshold=0.8,
            enable_alerts=True
        )
    
    def test_monitor_initialization(self, monitor):
        """Test health monitor initialization."""
        assert monitor.check_interval == 1.0
        assert monitor.coherence_threshold == 0.8
        assert monitor.enable_alerts is True
    
    def test_check_system_health(self, monitor):
        """Test system health check."""
        # Mock quantum system state
        system_state = {
            "coherence": 0.95,
            "error_rate": 0.01,
            "active_tasks": 10,
            "memory_usage": 0.6
        }
        
        health = monitor.check_system_health(system_state)
        
        assert isinstance(health, HealthStatus)
        assert health.overall_status in ["healthy", "warning", "critical"]
        assert health.coherence_status in ["good", "degraded", "poor"]
    
    def test_coherence_monitoring(self, monitor):
        """Test quantum coherence monitoring."""
        # High coherence
        high_coherence = 0.95
        status = monitor.check_coherence_health(high_coherence)
        assert status == "good"
        
        # Medium coherence
        medium_coherence = 0.75
        status = monitor.check_coherence_health(medium_coherence)
        assert status == "degraded"
        
        # Low coherence
        low_coherence = 0.5
        status = monitor.check_coherence_health(low_coherence)
        assert status == "poor"
    
    def test_performance_metrics_collection(self, monitor):
        """Test performance metrics collection."""
        # Mock performance data
        with patch.object(monitor, '_collect_system_metrics') as mock_collect:
            mock_collect.return_value = {
                "cpu_usage": 0.4,
                "memory_usage": 0.6,
                "task_throughput": 50.0,
                "average_latency": 0.1
            }
            
            metrics = monitor.collect_performance_metrics()
            
            assert "cpu_usage" in metrics
            assert "memory_usage" in metrics
            assert "task_throughput" in metrics
            assert metrics["cpu_usage"] >= 0
    
    def test_health_trend_analysis(self, monitor):
        """Test health trend analysis."""
        # Simulate health history
        health_history = [
            {"timestamp": time.time() - 300, "coherence": 0.9, "error_rate": 0.01},
            {"timestamp": time.time() - 200, "coherence": 0.85, "error_rate": 0.02}, 
            {"timestamp": time.time() - 100, "coherence": 0.8, "error_rate": 0.03},
            {"timestamp": time.time(), "coherence": 0.75, "error_rate": 0.05}
        ]
        
        trend = monitor.analyze_health_trends(health_history)
        
        assert "coherence_trend" in trend
        assert "error_trend" in trend
        assert trend["coherence_trend"] in ["improving", "stable", "degrading"]


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    @pytest.fixture
    def monitor(self):
        """Create performance monitor."""
        return PerformanceMonitor(
            sampling_interval=0.5,
            history_size=1000,
            enable_profiling=True
        )
    
    def test_monitor_initialization(self, monitor):
        """Test performance monitor initialization."""
        assert monitor.sampling_interval == 0.5
        assert monitor.history_size == 1000
        assert monitor.enable_profiling is True
    
    def test_record_task_execution(self, monitor):
        """Test task execution recording."""
        monitor.record_task_execution(
            task_id="test_task",
            execution_time=0.15,
            memory_used=1024,
            quantum_coherence=0.92
        )
        
        stats = monitor.get_task_statistics("test_task")
        
        assert stats["execution_count"] == 1
        assert stats["avg_execution_time"] == 0.15
        assert stats["avg_memory_used"] == 1024
        assert stats["avg_coherence"] == 0.92
    
    def test_performance_alerts(self, monitor):
        """Test performance-based alerting."""
        # Record slow execution
        monitor.record_task_execution("slow_task", 5.0, 2048, 0.7)
        
        alerts = monitor.check_performance_alerts()
        
        slow_alerts = [a for a in alerts if a["type"] == "high_latency"]
        assert len(slow_alerts) > 0
    
    def test_resource_monitoring(self, monitor):
        """Test system resource monitoring."""
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_cpu.return_value = 75.5
            mock_memory.return_value = Mock(percent=65.2)
            
            resources = monitor.monitor_system_resources()
            
            assert resources["cpu_percent"] == 75.5
            assert resources["memory_percent"] == 65.2
    
    def test_quantum_metrics_tracking(self, monitor):
        """Test quantum-specific metrics tracking."""
        quantum_metrics = QuantumMetrics(
            coherence_time=2.5,
            entanglement_fidelity=0.88,
            measurement_accuracy=0.94,
            decoherence_rate=0.02
        )
        
        monitor.record_quantum_metrics(quantum_metrics)
        
        stats = monitor.get_quantum_statistics()
        
        assert "avg_coherence_time" in stats
        assert "avg_entanglement_fidelity" in stats
        assert stats["avg_coherence_time"] == 2.5


class TestAlertManager:
    """Test alerting and notification system."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager."""
        return AlertManager(
            smtp_server="localhost",
            alert_threshold={"critical": 0.9, "warning": 0.7},
            enable_notifications=True
        )
    
    def test_alert_creation(self, alert_manager):
        """Test alert creation."""
        alert = alert_manager.create_alert(
            severity="critical",
            message="System coherence below threshold",
            component="quantum_core",
            metrics={"coherence": 0.6}
        )
        
        assert alert["severity"] == "critical"
        assert alert["component"] == "quantum_core"
        assert alert["metrics"]["coherence"] == 0.6
        assert "timestamp" in alert
    
    def test_alert_filtering(self, alert_manager):
        """Test alert filtering and deduplication."""
        # Create duplicate alerts
        for _ in range(3):
            alert_manager.create_alert(
                severity="warning",
                message="High memory usage",
                component="memory_manager"
            )
        
        # Should deduplicate similar alerts
        active_alerts = alert_manager.get_active_alerts()
        memory_alerts = [a for a in active_alerts if a["component"] == "memory_manager"]
        assert len(memory_alerts) == 1  # Deduplicated
    
    @patch('smtplib.SMTP')
    def test_email_notification(self, mock_smtp, alert_manager):
        """Test email notification sending."""
        alert = {
            "severity": "critical",
            "message": "System failure detected",
            "component": "quantum_executor",
            "timestamp": time.time()
        }
        
        result = alert_manager.send_email_notification(alert, ["admin@example.com"])
        
        assert result is True
        mock_smtp.assert_called_once()
    
    def test_alert_escalation(self, alert_manager):
        """Test alert escalation rules."""
        # Create critical alert
        alert = alert_manager.create_alert(
            severity="critical", 
            message="System down",
            component="quantum_core"
        )
        
        # Check escalation
        should_escalate = alert_manager.should_escalate_alert(alert)
        assert should_escalate is True


class TestMetricsCollector:
    """Test metrics collection system."""
    
    @pytest.fixture
    def collector(self):
        """Create metrics collector."""
        return MetricsCollector(
            collection_interval=1.0,
            retention_days=7,
            enable_quantum_metrics=True
        )
    
    def test_metric_recording(self, collector):
        """Test metric recording."""
        collector.record_metric(
            name="task_completion_rate",
            value=0.95,
            tags={"component": "scheduler"}
        )
        
        metrics = collector.get_metrics("task_completion_rate")
        assert len(metrics) == 1
        assert metrics[0]["value"] == 0.95
    
    def test_quantum_metric_recording(self, collector):
        """Test quantum-specific metric recording."""
        collector.record_quantum_metric(
            "coherence_measurement",
            coherence=0.88,
            fidelity=0.92,
            gate_error=0.001
        )
        
        quantum_metrics = collector.get_quantum_metrics()
        assert "coherence_measurement" in quantum_metrics
    
    def test_metric_aggregation(self, collector):
        """Test metric aggregation."""
        # Record multiple values
        for i in range(10):
            collector.record_metric("test_metric", i * 0.1)
        
        aggregated = collector.aggregate_metrics("test_metric", "avg")
        assert 0.4 <= aggregated <= 0.5  # Average of 0 to 0.9
    
    def test_prometheus_export(self, collector):
        """Test Prometheus metrics export format."""
        collector.record_metric("http_requests_total", 100, {"method": "GET"})
        collector.record_metric("http_requests_total", 50, {"method": "POST"})
        
        prometheus_format = collector.export_prometheus_format()
        
        assert "http_requests_total" in prometheus_format
        assert "method=\"GET\"" in prometheus_format
        assert "method=\"POST\"" in prometheus_format


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    def test_end_to_end_monitoring(self):
        """Test complete monitoring workflow."""
        # Setup components
        health_monitor = QuantumHealthMonitor()
        perf_monitor = PerformanceMonitor()
        alert_manager = AlertManager(enable_notifications=False)
        metrics_collector = MetricsCollector()
        
        # Simulate system operation
        system_state = {
            "coherence": 0.6,  # Low coherence should trigger alert
            "error_rate": 0.1,  # High error rate
            "active_tasks": 50,
            "memory_usage": 0.9  # High memory usage
        }
        
        # Check health
        health = health_monitor.check_system_health(system_state)
        assert health.overall_status in ["warning", "critical"]
        
        # Record performance
        perf_monitor.record_task_execution("test_task", 2.0, 4096, 0.6)
        
        # Check alerts
        alerts = perf_monitor.check_performance_alerts()
        assert len(alerts) > 0
        
        # Record metrics
        metrics_collector.record_metric("system_health", 0.6)
        
        # Verify integration
        assert len(alerts) > 0
        metrics = metrics_collector.get_metrics("system_health")
        assert len(metrics) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])