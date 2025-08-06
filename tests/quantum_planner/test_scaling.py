"""
Test suite for quantum planner scaling components.

Tests auto-scaling, load balancing, and resource management.
"""

import pytest
import time
from unittest.mock import Mock, patch
import jax.numpy as jnp

from src.dynamic_graph_fed_rl.quantum_planner.scaling import (
    AutoScaler,
    LoadBalancer,
    ResourceManager,
    ScalingMetrics,
    ScalingDecision,
    CircuitBreaker
)


class TestAutoScaler:
    """Test auto-scaling functionality."""
    
    @pytest.fixture
    def scaler(self):
        """Create auto-scaler."""
        return AutoScaler(
            min_instances=2,
            max_instances=10,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period=300
        )
    
    def test_scaler_initialization(self, scaler):
        """Test auto-scaler initialization."""
        assert scaler.min_instances == 2
        assert scaler.max_instances == 10
        assert scaler.scale_up_threshold == 0.8
        assert scaler.scale_down_threshold == 0.3
    
    def test_scaling_decision_scale_up(self, scaler):
        """Test scale-up decision logic."""
        metrics = ScalingMetrics(
            cpu_utilization=0.9,
            memory_utilization=0.85,
            task_queue_length=100,
            active_instances=3,
            avg_response_time=2.5
        )
        
        decision = scaler.make_scaling_decision(metrics)
        
        assert decision.action == "scale_up"
        assert decision.target_instances > metrics.active_instances
    
    def test_scaling_decision_scale_down(self, scaler):
        """Test scale-down decision logic."""
        metrics = ScalingMetrics(
            cpu_utilization=0.2,
            memory_utilization=0.15,
            task_queue_length=5,
            active_instances=6,
            avg_response_time=0.1
        )
        
        decision = scaler.make_scaling_decision(metrics)
        
        assert decision.action == "scale_down"
        assert decision.target_instances < metrics.active_instances
        assert decision.target_instances >= scaler.min_instances
    
    def test_scaling_limits(self, scaler):
        """Test scaling limits enforcement."""
        # Test max limit
        high_load_metrics = ScalingMetrics(
            cpu_utilization=0.95,
            memory_utilization=0.90,
            task_queue_length=1000,
            active_instances=10,  # Already at max
            avg_response_time=5.0
        )
        
        decision = scaler.make_scaling_decision(high_load_metrics)
        assert decision.target_instances <= scaler.max_instances
        
        # Test min limit
        low_load_metrics = ScalingMetrics(
            cpu_utilization=0.1,
            memory_utilization=0.05,
            task_queue_length=0,
            active_instances=2,  # Already at min
            avg_response_time=0.05
        )
        
        decision = scaler.make_scaling_decision(low_load_metrics)
        assert decision.target_instances >= scaler.min_instances
    
    def test_cooldown_period(self, scaler):
        """Test scaling cooldown period."""
        scaler.last_scaling_action = time.time() - 100  # 100 seconds ago
        
        metrics = ScalingMetrics(
            cpu_utilization=0.9,
            memory_utilization=0.85,
            task_queue_length=50,
            active_instances=3,
            avg_response_time=1.5
        )
        
        decision = scaler.make_scaling_decision(metrics)
        
        # Should not scale due to cooldown
        assert decision.action == "no_action"


class TestLoadBalancer:
    """Test load balancing functionality."""
    
    @pytest.fixture
    def load_balancer(self):
        """Create load balancer."""
        return LoadBalancer(
            algorithm="round_robin",
            health_check_interval=30,
            enable_circuit_breaker=True
        )
    
    def test_load_balancer_initialization(self, load_balancer):
        """Test load balancer initialization."""
        assert load_balancer.algorithm == "round_robin"
        assert load_balancer.health_check_interval == 30
        assert load_balancer.enable_circuit_breaker is True
    
    def test_round_robin_distribution(self, load_balancer):
        """Test round-robin load distribution."""
        # Setup instances
        instances = [
            {"id": "instance_1", "address": "192.168.1.1", "healthy": True},
            {"id": "instance_2", "address": "192.168.1.2", "healthy": True},
            {"id": "instance_3", "address": "192.168.1.3", "healthy": True}
        ]
        
        load_balancer.register_instances(instances)
        
        # Get multiple assignments
        assignments = []
        for _ in range(6):
            instance = load_balancer.get_next_instance()
            assignments.append(instance["id"])
        
        # Should cycle through instances
        assert assignments == ["instance_1", "instance_2", "instance_3",
                              "instance_1", "instance_2", "instance_3"]
    
    def test_least_connections_distribution(self, load_balancer):
        """Test least-connections load distribution."""
        load_balancer.algorithm = "least_connections"
        
        # Setup instances with different connection counts
        instances = [
            {"id": "instance_1", "connections": 5, "healthy": True},
            {"id": "instance_2", "connections": 2, "healthy": True},  # Least loaded
            {"id": "instance_3", "connections": 8, "healthy": True}
        ]
        
        load_balancer.register_instances(instances)
        
        # Should select least loaded instance
        selected = load_balancer.get_next_instance()
        assert selected["id"] == "instance_2"
    
    def test_health_check_filtering(self, load_balancer):
        """Test filtering of unhealthy instances."""
        instances = [
            {"id": "instance_1", "healthy": True},
            {"id": "instance_2", "healthy": False},  # Unhealthy
            {"id": "instance_3", "healthy": True}
        ]
        
        load_balancer.register_instances(instances)
        
        # Should only get healthy instances
        healthy_instances = load_balancer.get_healthy_instances()
        healthy_ids = [i["id"] for i in healthy_instances]
        
        assert "instance_1" in healthy_ids
        assert "instance_2" not in healthy_ids
        assert "instance_3" in healthy_ids
    
    def test_weighted_distribution(self, load_balancer):
        """Test weighted load distribution."""
        load_balancer.algorithm = "weighted_round_robin"
        
        instances = [
            {"id": "instance_1", "weight": 1, "healthy": True},
            {"id": "instance_2", "weight": 3, "healthy": True},  # Higher weight
            {"id": "instance_3", "weight": 2, "healthy": True}
        ]
        
        load_balancer.register_instances(instances)
        
        # Get many assignments to test distribution
        assignments = []
        for _ in range(12):  # Multiple of total weight (6)
            instance = load_balancer.get_next_instance()
            assignments.append(instance["id"])
        
        # Count assignments
        counts = {inst_id: assignments.count(inst_id) for inst_id in ["instance_1", "instance_2", "instance_3"]}
        
        # Should respect weights (approximately)
        assert counts["instance_2"] > counts["instance_1"]  # Higher weight gets more
        assert counts["instance_3"] > counts["instance_1"]


class TestResourceManager:
    """Test resource management functionality."""
    
    @pytest.fixture
    def resource_manager(self):
        """Create resource manager."""
        return ResourceManager(
            cpu_limit=8.0,
            memory_limit=16.0,  # GB
            enable_resource_pooling=True,
            enable_garbage_collection=True
        )
    
    def test_resource_allocation(self, resource_manager):
        """Test resource allocation."""
        # Request resources
        allocation = resource_manager.allocate_resources(
            cpu_cores=2.0,
            memory_gb=4.0,
            task_id="test_task"
        )
        
        assert allocation["success"] is True
        assert allocation["cpu_allocated"] == 2.0
        assert allocation["memory_allocated"] == 4.0
        
        # Check remaining resources
        remaining = resource_manager.get_available_resources()
        assert remaining["cpu_cores"] == 6.0
        assert remaining["memory_gb"] == 12.0
    
    def test_resource_over_allocation(self, resource_manager):
        """Test handling of resource over-allocation."""
        # Try to allocate more than available
        allocation = resource_manager.allocate_resources(
            cpu_cores=16.0,  # More than limit of 8.0
            memory_gb=4.0,
            task_id="overallocation_task"
        )
        
        assert allocation["success"] is False
        assert "insufficient" in allocation["error"].lower()
    
    def test_resource_deallocation(self, resource_manager):
        """Test resource deallocation."""
        # Allocate resources
        resource_manager.allocate_resources(2.0, 4.0, "task_1")
        
        # Deallocate resources
        success = resource_manager.deallocate_resources("task_1")
        
        assert success is True
        
        # Check resources are back
        remaining = resource_manager.get_available_resources()
        assert remaining["cpu_cores"] == 8.0
        assert remaining["memory_gb"] == 16.0
    
    def test_resource_monitoring(self, resource_manager):
        """Test resource usage monitoring."""
        # Allocate some resources
        resource_manager.allocate_resources(3.0, 6.0, "monitor_task")
        
        # Get usage statistics
        usage = resource_manager.get_resource_usage()
        
        assert usage["cpu_utilization"] == 0.375  # 3/8
        assert usage["memory_utilization"] == 0.375  # 6/16
        assert "active_allocations" in usage
    
    @patch('gc.collect')
    def test_garbage_collection(self, mock_gc, resource_manager):
        """Test garbage collection triggering."""
        # Simulate high memory usage
        resource_manager.memory_usage_threshold = 0.8
        resource_manager.current_memory_usage = 0.85
        
        resource_manager.check_garbage_collection()
        
        mock_gc.assert_called_once()


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker."""
        return CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            enable_monitoring=True
        )
    
    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.failure_threshold == 5
        assert circuit_breaker.recovery_timeout == 30
        assert circuit_breaker.state == "closed"
    
    def test_circuit_breaker_failure_detection(self, circuit_breaker):
        """Test failure detection and state transition."""
        # Record failures
        for _ in range(5):  # Reach threshold
            circuit_breaker.record_failure()
        
        # Should transition to open
        assert circuit_breaker.state == "open"
    
    def test_circuit_breaker_request_blocking(self, circuit_breaker):
        """Test request blocking when circuit is open."""
        # Force circuit to open state
        circuit_breaker.state = "open"
        
        # Request should be blocked
        allowed = circuit_breaker.allow_request()
        assert allowed is False
    
    def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test half-open state and recovery."""
        # Force to open state and simulate timeout
        circuit_breaker.state = "open"
        circuit_breaker.last_failure_time = time.time() - 31  # Beyond recovery timeout
        
        # Should transition to half-open
        allowed = circuit_breaker.allow_request()
        assert circuit_breaker.state == "half_open"
        assert allowed is True
    
    def test_circuit_breaker_success_recovery(self, circuit_breaker):
        """Test recovery after successful requests."""
        # Set to half-open state
        circuit_breaker.state = "half_open"
        
        # Record success
        circuit_breaker.record_success()
        
        # Should close circuit
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0


class TestScalingIntegration:
    """Integration tests for scaling components."""
    
    def test_complete_scaling_workflow(self):
        """Test complete scaling workflow."""
        # Setup components
        scaler = AutoScaler(min_instances=2, max_instances=5)
        load_balancer = LoadBalancer(algorithm="round_robin")
        resource_manager = ResourceManager(cpu_limit=16.0, memory_limit=32.0)
        circuit_breaker = CircuitBreaker(failure_threshold=3)
        
        # Initial setup
        initial_instances = [
            {"id": "instance_1", "cpu": 4.0, "memory": 8.0, "healthy": True},
            {"id": "instance_2", "cpu": 4.0, "memory": 8.0, "healthy": True}
        ]
        load_balancer.register_instances(initial_instances)
        
        # Simulate high load requiring scale-up
        high_load_metrics = ScalingMetrics(
            cpu_utilization=0.9,
            memory_utilization=0.85,
            task_queue_length=100,
            active_instances=2,
            avg_response_time=3.0
        )
        
        scaling_decision = scaler.make_scaling_decision(high_load_metrics)
        
        assert scaling_decision.action == "scale_up"
        assert scaling_decision.target_instances > 2
        
        # Simulate adding new instance
        if scaling_decision.action == "scale_up":
            new_instance = {
                "id": "instance_3", 
                "cpu": 4.0, 
                "memory": 8.0, 
                "healthy": True
            }
            load_balancer.add_instance(new_instance)
            resource_manager.add_instance_resources(4.0, 8.0)
        
        # Verify scaling worked
        healthy_instances = load_balancer.get_healthy_instances()
        assert len(healthy_instances) == 3
        
        total_resources = resource_manager.get_total_resources()
        assert total_resources["cpu_cores"] >= 12.0  # 3 instances * 4 cores


class TestScalingMetrics:
    """Test scaling metrics collection and analysis."""
    
    def test_metrics_creation(self):
        """Test scaling metrics creation."""
        metrics = ScalingMetrics(
            cpu_utilization=0.75,
            memory_utilization=0.60,
            task_queue_length=25,
            active_instances=3,
            avg_response_time=1.2,
            timestamp=time.time()
        )
        
        assert metrics.cpu_utilization == 0.75
        assert metrics.memory_utilization == 0.60
        assert metrics.task_queue_length == 25
        assert metrics.active_instances == 3
        assert metrics.avg_response_time == 1.2
    
    def test_metrics_analysis(self):
        """Test metrics analysis for scaling decisions."""
        metrics = ScalingMetrics(
            cpu_utilization=0.85,
            memory_utilization=0.70,
            task_queue_length=50,
            active_instances=4,
            avg_response_time=2.0
        )
        
        # Calculate composite load score
        load_score = (
            metrics.cpu_utilization * 0.4 +
            metrics.memory_utilization * 0.3 +
            min(metrics.task_queue_length / 100, 1.0) * 0.3
        )
        
        assert load_score > 0.7  # High load should trigger scaling


if __name__ == "__main__":
    pytest.main([__file__, "-v"])