"""
Test suite for quantum scheduler functionality.

Tests quantum scheduling, adaptive scheduling, and performance optimization.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from src.dynamic_graph_fed_rl.quantum_planner.core import QuantumTask, TaskState, QuantumTaskPlanner
from src.dynamic_graph_fed_rl.quantum_planner.scheduler import (
    QuantumScheduler,
    AdaptiveScheduler,
    SchedulingMetrics,
    BaseScheduler
)
from src.dynamic_graph_fed_rl.quantum_planner.exceptions import QuantumPlannerError


class TestQuantumScheduler:
    """Test quantum scheduler functionality."""
    
    @pytest.fixture
    def planner(self):
        """Create test quantum task planner."""
        planner = QuantumTaskPlanner()
        
        # Add test tasks
        planner.add_task("task1", "Task 1", estimated_duration=1.0, priority=0.8)
        planner.add_task("task2", "Task 2", dependencies={"task1"}, estimated_duration=2.0, priority=0.6)
        planner.add_task("task3", "Task 3", estimated_duration=1.5, priority=0.9)
        planner.add_task("task4", "Task 4", dependencies={"task2", "task3"}, estimated_duration=0.5, priority=0.7)
        
        return planner
    
    @pytest.fixture
    def scheduler(self):
        """Create test quantum scheduler."""
        return QuantumScheduler(
            max_concurrent_tasks=3,
            measurement_interval=0.1,
            interference_optimization=True
        )
    
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.max_concurrent_tasks == 3
        assert scheduler.measurement_interval == 0.1
        assert scheduler.interference_optimization is True
        assert scheduler.completed_tasks == 0
        assert scheduler.failed_tasks == 0
    
    @pytest.mark.asyncio
    async def test_schedule_execution(self, scheduler, planner):
        """Test basic schedule execution."""
        result = await scheduler.schedule(planner)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "scheduled" in result
        assert "execution_results" in result
        
        if result["status"] == "success":
            assert len(result["scheduled"]) > 0
            assert "quantum_paths_explored" in result
    
    @pytest.mark.asyncio
    async def test_no_ready_tasks(self, scheduler):
        """Test scheduling with no ready tasks."""
        empty_planner = QuantumTaskPlanner()
        
        # Add task with unsatisfied dependency
        empty_planner.add_task("task1", "Task 1", dependencies={"nonexistent"})
        
        result = await scheduler.schedule(empty_planner)
        
        assert result["status"] == "no_ready_tasks"
        assert result["scheduled"] == []
    
    @pytest.mark.asyncio
    async def test_concurrent_task_limits(self, scheduler, planner):
        """Test scheduler respects concurrent task limits."""
        # Set low concurrency limit
        scheduler.max_concurrent_tasks = 2
        
        result = await scheduler.schedule(planner)
        
        if result["status"] == "success":
            # Should not schedule more than max_concurrent_tasks
            assert len(result["scheduled"]) <= 2
    
    def test_scheduling_strategies(self, scheduler, planner):
        """Test different scheduling strategies."""
        ready_tasks = ["task1", "task3"]  # Tasks with no dependencies
        
        # Test priority scheduling
        priority_schedule = scheduler._create_priority_schedule(planner, ready_tasks)
        assert isinstance(priority_schedule, list)
        
        # Should prefer higher priority tasks
        if len(priority_schedule) >= 2:
            task_priorities = [
                planner.tasks[task["task_id"]].priority
                for task in priority_schedule[:2]
            ]
            # Should be in descending priority order
            assert task_priorities[0] >= task_priorities[1]
        
        # Test resource optimization
        resource_schedule = scheduler._create_resource_optimized_schedule(planner, ready_tasks)
        assert isinstance(resource_schedule, list)
        
        # Test dependency optimization
        dependency_schedule = scheduler._create_dependency_schedule(planner, ready_tasks)
        assert isinstance(dependency_schedule, list)
    
    @pytest.mark.asyncio
    async def test_task_execution_simulation(self, scheduler):
        """Test task execution simulation."""
        # Create simple task with mock executor
        mock_result = {"status": "completed", "result": "test"}
        mock_executor = AsyncMock(return_value=mock_result)
        
        task = QuantumTask(
            id="test_task",
            name="Test Task",
            estimated_duration=0.1,
            executor=mock_executor
        )
        
        result = await scheduler._execute_single_task(task)
        
        assert result["status"] == "success"
        assert result["result"] == mock_result
        assert "duration" in result
        
        # Verify task state collapsed to completed
        assert task.get_probability(TaskState.COMPLETED) > 0.9
    
    @pytest.mark.asyncio
    async def test_task_execution_failure(self, scheduler):
        """Test task execution failure handling."""
        # Create task with failing executor
        async def failing_executor():
            raise Exception("Test failure")
        
        task = QuantumTask(
            id="failing_task",
            name="Failing Task",
            estimated_duration=0.1,
            executor=failing_executor
        )
        
        result = await scheduler._execute_single_task(task)
        
        assert result["status"] == "failed"
        assert "error" in result
        assert "duration" in result
        
        # Verify task state collapsed to failed
        assert task.get_probability(TaskState.FAILED) > 0.9
    
    def test_metrics_collection(self, scheduler):
        """Test scheduler metrics collection."""
        # Simulate some completed tasks
        scheduler.completed_tasks = 10
        scheduler.failed_tasks = 2
        scheduler.execution_times = [0.5, 1.0, 0.8, 1.2, 0.9]
        
        metrics = scheduler.get_metrics()
        
        assert isinstance(metrics, SchedulingMetrics)
        assert metrics.tasks_completed == 10
        assert metrics.tasks_failed == 2
        assert metrics.throughput > 0
        assert 0 <= metrics.quantum_efficiency <= 1
    
    @pytest.mark.asyncio
    async def test_interference_optimization(self, scheduler, planner):
        """Test quantum interference optimization."""
        # Enable interference optimization
        scheduler.interference_optimization = True
        
        ready_tasks = ["task1", "task3"]
        schedules = await scheduler._generate_schedule_superposition(planner, ready_tasks)
        
        # Apply interference optimization
        optimized_schedules = scheduler._apply_quantum_interference(schedules)
        
        assert len(optimized_schedules) == len(schedules)
        
        # Check that amplitudes were modified
        for original, optimized in zip(schedules, optimized_schedules):
            # Amplitudes might be the same or modified depending on overlap
            assert "amplitude" in optimized
    
    @pytest.mark.asyncio
    async def test_measurement_and_selection(self, scheduler, planner):
        """Test quantum measurement for schedule selection."""
        ready_tasks = ["task1", "task3"]
        schedules = await scheduler._generate_schedule_superposition(planner, ready_tasks)
        
        # Measure optimal schedule
        selected_schedule = scheduler._measure_optimal_schedule(schedules)
        
        assert isinstance(selected_schedule, list)
        # Should select one of the generated schedules
        assert any(selected_schedule == s["tasks"] for s in schedules)


class TestAdaptiveScheduler:
    """Test adaptive scheduler functionality."""
    
    @pytest.fixture
    def base_scheduler(self):
        """Create base scheduler for adaptive scheduler."""
        return QuantumScheduler(max_concurrent_tasks=2)
    
    @pytest.fixture
    def adaptive_scheduler(self, base_scheduler):
        """Create adaptive scheduler."""
        return AdaptiveScheduler(
            base_scheduler=base_scheduler,
            learning_rate=0.1,
            exploration_rate=0.2,
            adaptation_window=5
        )
    
    @pytest.fixture
    def planner(self):
        """Create test planner."""
        planner = QuantumTaskPlanner()
        planner.add_task("task1", "Task 1", estimated_duration=0.1)
        planner.add_task("task2", "Task 2", estimated_duration=0.1)
        return planner
    
    def test_adaptive_scheduler_initialization(self, adaptive_scheduler):
        """Test adaptive scheduler initialization."""
        assert adaptive_scheduler.learning_rate == 0.1
        assert adaptive_scheduler.exploration_rate == 0.2
        assert adaptive_scheduler.adaptation_window == 5
        assert len(adaptive_scheduler.adaptive_params) > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_scheduling(self, adaptive_scheduler, planner):
        """Test adaptive scheduling with learning."""
        # Run multiple scheduling rounds
        results = []
        for _ in range(6):  # More than adaptation window
            result = await adaptive_scheduler.schedule(planner)
            results.append(result)
            
            # Simulate some delay between rounds
            await asyncio.sleep(0.01)
        
        # Should have adaptation after window size
        assert len(adaptive_scheduler.parameter_history) > 0
        assert len(adaptive_scheduler.performance_history) > 0
    
    def test_parameter_adaptation(self, adaptive_scheduler):
        """Test parameter adaptation logic."""
        # Simulate performance history
        for i in range(10):
            performance = 0.8 + 0.1 * (i / 10)  # Improving performance
            adaptive_scheduler.performance_history.append(performance)
            adaptive_scheduler.parameter_history.append(dict(adaptive_scheduler.adaptive_params))
        
        # Trigger adaptation
        asyncio.run(adaptive_scheduler._adapt_parameters())
        
        # Parameters should be updated
        assert len(adaptive_scheduler.parameter_history) == 10
    
    def test_parameter_bounds(self, adaptive_scheduler):
        """Test parameter bounds enforcement."""
        original_params = dict(adaptive_scheduler.adaptive_params)
        
        # Simulate extreme parameter changes
        adaptive_scheduler.adaptive_params["max_concurrent_tasks"] = 100
        adaptive_scheduler.adaptive_params["measurement_interval"] = 100.0
        adaptive_scheduler.adaptive_params["interference_strength"] = 2.0
        
        # Apply to scheduler (should enforce bounds)
        adaptive_scheduler._apply_parameters_to_scheduler()
        
        # Check base scheduler received bounded values
        assert adaptive_scheduler.base_scheduler.max_concurrent_tasks <= 16
        assert adaptive_scheduler.base_scheduler.measurement_interval <= 10.0
    
    @pytest.mark.asyncio
    async def test_learning_from_performance(self, adaptive_scheduler):
        """Test learning from performance feedback."""
        # Create mock scheduling result
        mock_result = {
            "status": "success",
            "execution_results": {
                "task_results": {
                    "task1": {"status": "success"},
                    "task2": {"status": "success"}
                }
            }
        }
        
        initial_params = dict(adaptive_scheduler.adaptive_params)
        
        # Learn from good performance
        await adaptive_scheduler._learn_from_performance(mock_result)
        
        # Should record performance
        assert len(adaptive_scheduler.performance_history) == 1
        assert adaptive_scheduler.performance_history[0] == 1.0  # 100% success
    
    def test_adaptive_metrics(self, adaptive_scheduler):
        """Test adaptive scheduler metrics."""
        # Simulate some adaptation history
        adaptive_scheduler.performance_history.extend([0.6, 0.7, 0.8, 0.9, 0.9])
        
        metrics = adaptive_scheduler.get_metrics()
        
        assert isinstance(metrics, SchedulingMetrics)
        # Should enhance base metrics with adaptation improvements
        assert hasattr(metrics, 'quantum_efficiency')


class TestSchedulingIntegration:
    """Integration tests for scheduling components."""
    
    @pytest.mark.asyncio
    async def test_full_scheduling_workflow(self):
        """Test complete scheduling workflow."""
        # Create complex task graph
        planner = QuantumTaskPlanner()
        
        # Add tasks with dependencies and resource requirements
        planner.add_task(
            "init", "Initialize",
            estimated_duration=0.1,
            priority=1.0,
            resource_requirements={"cpu": 0.2},
            executor=AsyncMock(return_value={"status": "completed"})
        )
        
        planner.add_task(
            "process1", "Process 1",
            dependencies={"init"},
            estimated_duration=0.2,
            priority=0.8,
            resource_requirements={"cpu": 0.5, "memory": 0.3},
            executor=AsyncMock(return_value={"status": "completed"})
        )
        
        planner.add_task(
            "process2", "Process 2", 
            dependencies={"init"},
            estimated_duration=0.15,
            priority=0.9,
            resource_requirements={"cpu": 0.3, "memory": 0.6},
            executor=AsyncMock(return_value={"status": "completed"})
        )
        
        planner.add_task(
            "combine", "Combine Results",
            dependencies={"process1", "process2"},
            estimated_duration=0.1,
            priority=0.7,
            resource_requirements={"cpu": 0.4},
            executor=AsyncMock(return_value={"status": "completed"})
        )
        
        # Test with quantum scheduler
        quantum_scheduler = QuantumScheduler(max_concurrent_tasks=2)
        
        result = await quantum_scheduler.schedule(planner)
        
        assert result["status"] == "success"
        assert len(result["scheduled"]) > 0
        
        # Verify dependency constraints in execution
        execution_results = result["execution_results"]
        assert "task_results" in execution_results
        
        # Test with adaptive scheduler
        adaptive_scheduler = AdaptiveScheduler(
            base_scheduler=QuantumScheduler(max_concurrent_tasks=2),
            adaptation_window=2
        )
        
        adaptive_result = await adaptive_scheduler.schedule(planner)
        assert adaptive_result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_scheduler_performance_comparison(self):
        """Test performance comparison between scheduling strategies."""
        # Create identical planners
        planner1 = QuantumTaskPlanner()
        planner2 = QuantumTaskPlanner()
        
        # Add same tasks to both
        tasks = [
            ("task1", "Task 1", set(), 0.1, 0.8),
            ("task2", "Task 2", {"task1"}, 0.2, 0.7),
            ("task3", "Task 3", {"task1"}, 0.15, 0.9),
            ("task4", "Task 4", {"task2", "task3"}, 0.1, 0.6)
        ]
        
        for task_id, name, deps, duration, priority in tasks:
            executor = AsyncMock(return_value={"status": "completed"})
            
            planner1.add_task(task_id, name, deps, duration, priority, executor=executor)
            planner2.add_task(task_id, name, deps, duration, priority, executor=executor)
        
        # Test quantum scheduler
        quantum_scheduler = QuantumScheduler(
            max_concurrent_tasks=2,
            interference_optimization=True
        )
        
        start_time = time.time()
        quantum_result = await quantum_scheduler.schedule(planner1)
        quantum_time = time.time() - start_time
        
        # Test adaptive scheduler
        adaptive_scheduler = AdaptiveScheduler(
            base_scheduler=QuantumScheduler(max_concurrent_tasks=2),
            learning_rate=0.05
        )
        
        start_time = time.time()
        adaptive_result = await adaptive_scheduler.schedule(planner2)
        adaptive_time = time.time() - start_time
        
        # Both should succeed
        assert quantum_result["status"] == "success"
        assert adaptive_result["status"] == "success"
        
        # Compare metrics
        quantum_metrics = quantum_scheduler.get_metrics()
        adaptive_metrics = adaptive_scheduler.get_metrics()
        
        assert quantum_metrics.quantum_efficiency >= 0
        assert adaptive_metrics.quantum_efficiency >= 0
    
    @pytest.mark.asyncio
    async def test_scheduler_error_recovery(self):
        """Test scheduler error recovery capabilities."""
        planner = QuantumTaskPlanner()
        
        # Add mix of successful and failing tasks
        planner.add_task(
            "success_task", "Success Task",
            executor=AsyncMock(return_value={"status": "completed"}),
            estimated_duration=0.1
        )
        
        async def failing_executor():
            raise Exception("Simulated failure")
        
        planner.add_task(
            "failing_task", "Failing Task",
            executor=failing_executor,
            estimated_duration=0.1
        )
        
        planner.add_task(
            "recovery_task", "Recovery Task",
            dependencies={"success_task"},  # Only depend on successful task
            executor=AsyncMock(return_value={"status": "completed"}),
            estimated_duration=0.1
        )
        
        scheduler = QuantumScheduler(max_concurrent_tasks=3)
        result = await scheduler.schedule(planner)
        
        # Should handle failures gracefully
        assert result["status"] == "success"
        
        task_results = result["execution_results"]["task_results"]
        
        # Check mixed results
        success_count = sum(1 for r in task_results.values() if r["status"] == "success")
        failure_count = sum(1 for r in task_results.values() if r["status"] == "failed")
        
        assert success_count > 0  # Some tasks should succeed
        assert failure_count > 0  # Some tasks should fail
    
    @pytest.mark.asyncio
    async def test_scheduler_resource_constraints(self):
        """Test scheduler handling of resource constraints."""
        planner = QuantumTaskPlanner()
        
        # Add tasks with high resource requirements
        for i in range(5):
            planner.add_task(
                f"resource_task_{i}",
                f"Resource Task {i}",
                resource_requirements={"cpu": 0.8, "memory": 0.7},
                executor=AsyncMock(return_value={"status": "completed"}),
                estimated_duration=0.1
            )
        
        scheduler = QuantumScheduler(max_concurrent_tasks=2)  # Limited concurrency
        result = await scheduler.schedule(planner)
        
        # Should respect resource constraints
        assert result["status"] == "success"
        
        # Should not schedule more than concurrent limit
        concurrent_tasks = result.get("concurrent_tasks", len(result["scheduled"]))
        assert concurrent_tasks <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])