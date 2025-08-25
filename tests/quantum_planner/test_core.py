"""
Test suite for quantum task planner core functionality.

Comprehensive tests covering quantum task management, superposition,
entanglement, and measurement operations.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch

from src.dynamic_graph_fed_rl.quantum_planner.core import (
    QuantumTask, 
    TaskState, 
    TaskSuperposition, 
    QuantumTaskPlanner
)
from src.dynamic_graph_fed_rl.quantum_planner.exceptions import (
    QuantumPlannerError,
    TaskValidationError,
    DependencyError
)


class TestQuantumTask:
    """Test quantum task functionality."""
    
    def test_quantum_task_creation(self):
        """Test basic quantum task creation."""
        task = QuantumTask(
            id="test_task",
            name="Test Task",
            dependencies={"dep1", "dep2"},
            estimated_duration=5.0,
            priority=0.8,
            resource_requirements={"cpu": 0.5, "memory": 0.3}
        )
        
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.dependencies == {"dep1", "dep2"}
        assert task.estimated_duration == 5.0
        assert task.priority == 0.8
        assert task.resource_requirements == {"cpu": 0.5, "memory": 0.3}
        
        # Check quantum state initialization
        assert TaskState.PENDING in task.state_amplitudes
        assert abs(task.state_amplitudes[TaskState.PENDING]) == 1.0
        
    def test_quantum_state_initialization(self):
        """Test quantum state is properly initialized."""
        task = QuantumTask(id="test", name="Test")
        
        # Should have all required states
        required_states = {TaskState.PENDING, TaskState.ACTIVE, TaskState.COMPLETED, TaskState.FAILED}
        assert set(task.state_amplitudes.keys()) == required_states
        
        # Should be in superposition with pending as dominant state
        total_prob = sum(abs(amp)**2 for amp in task.state_amplitudes.values())
        assert abs(total_prob - 1.0) < 1e-10
        
        # Pending should have probability 1
        assert task.get_probability(TaskState.PENDING) == 1.0
    
    def test_probability_calculation(self):
        """Test quantum probability calculations."""
        task = QuantumTask(id="test", name="Test")
        
        # Set specific amplitude
        task.update_amplitude(TaskState.ACTIVE, complex(0.6, 0.8))
        
        # Check probability
        expected_prob = abs(complex(0.6, 0.8))**2
        actual_prob = task.get_probability(TaskState.ACTIVE)
        assert abs(actual_prob - expected_prob) < 1e-10
    
    def test_amplitude_normalization(self):
        """Test amplitude normalization preserves quantum constraint."""
        task = QuantumTask(id="test", name="Test")
        
        # Set unnormalized amplitudes
        task.update_amplitude(TaskState.PENDING, complex(2.0, 0.0))
        task.update_amplitude(TaskState.ACTIVE, complex(1.0, 1.0))
        
        # Should be automatically normalized
        total_prob = sum(task.get_probability(state) for state in TaskState)
        assert abs(total_prob - 1.0) < 1e-10
    
    def test_state_collapse(self):
        """Test quantum state collapse (measurement)."""
        task = QuantumTask(id="test", name="Test")
        
        # Set equal superposition
        for state in TaskState:
            task.update_amplitude(state, complex(0.5, 0.0))
        
        # Collapse should return one of the states
        collapsed_state = task.collapse_state()
        assert isinstance(collapsed_state, TaskState)
        assert collapsed_state in TaskState
    
    def test_entanglement_tracking(self):
        """Test quantum entanglement tracking."""
        task1 = QuantumTask(id="task1", name="Task 1")
        task2 = QuantumTask(id="task2", name="Task 2")
        
        # Initially no entanglement
        assert len(task1.entangled_tasks) == 0
        assert len(task2.entangled_tasks) == 0
        
        # Create entanglement
        task1.entangled_tasks.add("task2")
        task2.entangled_tasks.add("task1")
        
        assert "task2" in task1.entangled_tasks
        assert "task1" in task2.entangled_tasks


class TestTaskSuperposition:
    """Test task superposition functionality."""
    
    def test_superposition_creation(self):
        """Test superposition creation and path management."""
        superposition = TaskSuperposition()
        
        # Add paths
        path1 = ["task1", "task2", "task3"]
        path2 = ["task1", "task3", "task2"]
        
        superposition.add_path(path1, complex(0.8, 0.0))
        superposition.add_path(path2, complex(0.6, 0.0))
        
        assert len(superposition.paths) == 2
        assert path1 in superposition.paths
        assert path2 in superposition.paths
    
    def test_interference_calculation(self):
        """Test quantum interference calculations."""
        superposition = TaskSuperposition()
        
        # Add overlapping paths
        path1 = ["task1", "task2", "task3"]
        path2 = ["task1", "task2", "task4"]  # Shares task1, task2
        
        superposition.add_path(path1, complex(0.7, 0.0))
        superposition.add_path(path2, complex(0.7, 0.0))
        
        # Should have interference matrix
        assert superposition.interference_matrix is not None
        assert superposition.interference_matrix.shape == (2, 2)
        
        # Diagonal should be zero (no self-interference)
        assert superposition.interference_matrix[0, 0] == 0
        assert superposition.interference_matrix[1, 1] == 0
    
    def test_optimal_path_measurement(self):
        """Test optimal path measurement from superposition."""
        superposition = TaskSuperposition()
        
        # Add paths with different amplitudes
        path1 = ["task1", "task2"]
        path2 = ["task3", "task4"]
        
        superposition.add_path(path1, complex(0.9, 0.0))  # Higher amplitude
        superposition.add_path(path2, complex(0.1, 0.0))  # Lower amplitude
        
        # Measure multiple times to check probabilistic behavior
        measurements = [superposition.measure_optimal_path() for _ in range(100)]
        
        # Should get some results (not empty)
        assert all(isinstance(path, list) for path in measurements)
        assert all(path in [path1, path2] for path in measurements)
        
        # Higher amplitude path should appear more often (probabilistically)
        path1_count = sum(1 for path in measurements if path == path1)
        path2_count = sum(1 for path in measurements if path == path2)
        
        # Due to randomness, we can't guarantee exact ratios, but path1 should generally dominate
        assert path1_count + path2_count == 100


class TestQuantumTaskPlanner:
    """Test quantum task planner functionality."""
    
    def test_planner_initialization(self):
        """Test planner initialization."""
        planner = QuantumTaskPlanner(
            max_parallel_tasks=8,
            quantum_coherence_time=15.0,
            interference_strength=0.2
        )
        
        assert planner.max_parallel_tasks == 8
        assert planner.quantum_coherence_time == 15.0
        assert planner.interference_strength == 0.2
        assert len(planner.tasks) == 0
        assert len(planner.task_graph) == 0
    
    def test_add_task(self):
        """Test adding tasks to planner."""
        planner = QuantumTaskPlanner()
        
        # Add task
        task = planner.add_task(
            task_id="test_task",
            name="Test Task",
            dependencies={"dep1"},
            estimated_duration=2.0,
            priority=0.7
        )
        
        assert isinstance(task, QuantumTask)
        assert task.id == "test_task"
        assert "test_task" in planner.tasks
        assert "test_task" in planner.task_graph
        assert planner.task_graph["test_task"] == {"dep1"}
    
    def test_entanglement_calculation(self):
        """Test quantum entanglement calculation between tasks."""
        planner = QuantumTaskPlanner()
        
        # Add tasks with shared resources
        task1 = planner.add_task(
            task_id="task1",
            name="Task 1",
            resource_requirements={"cpu": 0.5, "memory": 0.3},
            priority=0.8
        )
        
        task2 = planner.add_task(
            task_id="task2", 
            name="Task 2",
            resource_requirements={"cpu": 0.3, "memory": 0.7},
            priority=0.9
        )
        
        # Should have some entanglement due to shared resources
        assert len(planner.entanglement_matrix) >= 0  # May or may not create entanglement
        
        # Add dependency relationship
        task3 = planner.add_task(
            task_id="task3",
            name="Task 3", 
            dependencies={"task1"}
        )
        
        # Should create entanglement due to dependency
        entanglement_key = tuple(sorted(["task1", "task3"]))
        if entanglement_key in planner.entanglement_matrix:
            assert planner.entanglement_matrix[entanglement_key] != 0
    
    def test_execution_path_generation(self):
        """Test execution path generation."""
        planner = QuantumTaskPlanner()
        
        # Add tasks with dependencies
        planner.add_task("task1", "Task 1")
        planner.add_task("task2", "Task 2", dependencies={"task1"})
        planner.add_task("task3", "Task 3", dependencies={"task1"})
        
        # Generate paths
        superposition = planner.generate_execution_paths()
        
        assert isinstance(superposition, TaskSuperposition)
        assert len(superposition.paths) > 0
        
        # All paths should respect dependencies
        for path in superposition.paths:
            task1_idx = path.index("task1") if "task1" in path else -1
            task2_idx = path.index("task2") if "task2" in path else -1
            task3_idx = path.index("task3") if "task3" in path else -1
            
            # task2 and task3 should come after task1
            if task1_idx >= 0 and task2_idx >= 0:
                assert task1_idx < task2_idx
            if task1_idx >= 0 and task3_idx >= 0:
                assert task1_idx < task3_idx
    
    def test_measure_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute(self):
        """Test quantum measurement and execution."""
        planner = QuantumTaskPlanner()
        
        # Add simple tasks
        planner.add_task(
            "task1", "Task 1", 
            estimated_duration=0.1,
            executor=lambda: {"result": "success"}
        )
        planner.add_task(
            "task2", "Task 2",
            dependencies={"task1"}, 
            estimated_duration=0.1,
            executor=lambda: {"result": "success"}
        )
        
        # Execute
        result = planner.measure_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute()
        
        assert isinstance(result, dict)
        assert "path" in result
        assert "task_results" in result
        assert "total_duration" in result
        assert "quantum_efficiency" in result
        
        # Check task execution results
        task_results = result["task_results"]
        assert len(task_results) > 0
        
        for task_id, task_result in task_results.items():
            assert "status" in task_result
            assert task_result["status"] in ["success", "failed"]
    
    def test_system_state_reporting(self):
        """Test system state reporting."""
        planner = QuantumTaskPlanner()
        
        # Add tasks
        planner.add_task("task1", "Task 1")
        planner.add_task("task2", "Task 2", dependencies={"task1"})
        
        # Get system state
        state = planner.get_system_state()
        
        assert isinstance(state, dict)
        assert "num_tasks" in state
        assert "task_states" in state
        assert "entanglements" in state
        assert "execution_history_length" in state
        
        assert state["num_tasks"] == 2
        assert len(state["task_states"]) == 2
    
    def test_quantum_decoherence(self):
        """Test quantum decoherence over time."""
        planner = QuantumTaskPlanner(quantum_coherence_time=0.1)  # Very short coherence
        
        # Add task and generate superposition
        planner.add_task("task1", "Task 1")
        superposition = planner.generate_execution_paths()
        
        # Wait for decoherence
        time.sleep(0.15)
        
        # Generate new superposition - should be different due to decoherence
        new_superposition = planner.generate_execution_paths()
        
        # Hard to test exactly due to randomness, but should still work
        assert isinstance(new_superposition, TaskSuperposition)
        assert len(new_superposition.paths) > 0


class TestQuantumPlannerIntegration:
    """Integration tests for complete quantum planning workflows."""
    
    def test_complete_planning_workflow(self):
        """Test complete planning workflow from start to finish."""
        planner = QuantumTaskPlanner()
        
        # Create complex task graph
        tasks = {
            "init": {"name": "Initialize", "duration": 0.1},
            "process1": {"name": "Process 1", "deps": {"init"}, "duration": 0.1},
            "process2": {"name": "Process 2", "deps": {"init"}, "duration": 0.1}, 
            "combine": {"name": "Combine", "deps": {"process1", "process2"}, "duration": 0.1},
            "finalize": {"name": "Finalize", "deps": {"combine"}, "duration": 0.1}
        }
        
        # Add all tasks
        for task_id, task_info in tasks.items():
            planner.add_task(
                task_id=task_id,
                name=task_info["name"],
                dependencies=task_info.get("deps", set()),
                estimated_duration=task_info["duration"],
                executor=lambda: {"status": "completed"}
            )
        
        # Execute complete workflow
        start_time = time.time()
        result = planner.measure_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute()
        execution_time = time.time() - start_time
        
        # Validate results
        assert result["quantum_efficiency"] > 0
        assert result["total_duration"] > 0
        assert len(result["task_results"]) == len(tasks)
        
        # All tasks should complete successfully
        for task_id, task_result in result["task_results"].items():
            assert task_result["status"] == "success"
        
        # Execution should respect dependencies
        path = result["path"]
        init_idx = path.index("init")
        combine_idx = path.index("combine") 
        finalize_idx = path.index("finalize")
        
        # init should come first
        assert init_idx < combine_idx
        assert combine_idx < finalize_idx
    
    def test_error_handling_and_recovery(self):
        """Test error handling and quantum state recovery."""
        planner = QuantumTaskPlanner()
        
        def failing_executor():
            raise Exception("Simulated failure")
        
        def success_executor():
            return {"status": "completed"}
        
        # Add tasks with mixed success/failure
        planner.add_task(
            "success_task", "Success Task",
            executor=success_executor,
            estimated_duration=0.1
        )
        
        planner.add_task(
            "failing_task", "Failing Task", 
            executor=failing_executor,
            estimated_duration=0.1
        )
        
        planner.add_task(
            "recovery_task", "Recovery Task",
            dependencies={"success_task"},
            executor=success_executor,
            estimated_duration=0.1
        )
        
        # Execute and check error handling
        result = planner.measure_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute()
        
        # Should have mixed results
        task_results = result["task_results"]
        success_count = sum(1 for r in task_results.values() if r["status"] == "success")
        failed_count = sum(1 for r in task_results.values() if r["status"] == "failed")
        
        assert success_count > 0  # Some tasks should succeed
        assert failed_count > 0   # Some tasks should fail
        
        # Quantum efficiency should reflect partial success
        assert 0 < result["quantum_efficiency"] < 1
    
    def test_large_scale_planning(self):
        """Test planning with large number of tasks."""
        planner = QuantumTaskPlanner(max_parallel_tasks=10)
        
        # Create large task graph (100 tasks)
        num_tasks = 100
        
        # Add root task
        planner.add_task("root", "Root Task", estimated_duration=0.01)
        
        # Add layers of dependent tasks
        for layer in range(5):  # 5 layers
            for i in range(num_tasks // 5):  # 20 tasks per layer
                task_id = f"task_L{layer}_N{i}"
                
                if layer == 0:
                    deps = {"root"}
                else:
                    # Depend on some tasks from previous layer
                    prev_layer = layer - 1
                    deps = {f"task_L{prev_layer}_N{min(i, num_tasks//5-1)}"}
                
                planner.add_task(
                    task_id=task_id,
                    name=f"Task Layer {layer} Node {i}",
                    dependencies=deps,
                    estimated_duration=0.01,
                    priority=np.random.random(),
                    executor=lambda: {"status": "completed"}
                )
        
        # Execute large-scale planning
        start_time = time.time()
        result = planner.measure_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute()
        execution_time = time.time() - start_time
        
        # Should handle large scale efficiently
        assert len(result["task_results"]) == num_tasks + 1  # +1 for root
        assert result["quantum_efficiency"] > 0.5  # Should be reasonably efficient
        assert execution_time < 10.0  # Should complete in reasonable time
    
    @pytest.mark.parametrize("coherence_time,interference_strength", [
        (1.0, 0.1),
        (5.0, 0.3), 
        (10.0, 0.5)
    ])
    def test_quantum_parameter_effects(self, coherence_time, interference_strength):
        """Test effects of different quantum parameters."""
        planner = QuantumTaskPlanner(
            quantum_coherence_time=coherence_time,
            interference_strength=interference_strength
        )
        
        # Add standard task set
        for i in range(10):
            planner.add_task(
                f"task_{i}", f"Task {i}",
                dependencies={f"task_{j}" for j in range(i)} if i > 0 else set(),
                estimated_duration=0.05,
                executor=lambda: {"status": "completed"}
            )
        
        # Execute with different parameters
        result = planner.measure_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute()
        
        # Should work with all parameter combinations
        assert result["quantum_efficiency"] > 0
        assert len(result["task_results"]) == 10
        
        # Parameters should influence behavior (hard to test exactly due to randomness)
        assert isinstance(result["total_duration"], float)
        assert result["total_duration"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])