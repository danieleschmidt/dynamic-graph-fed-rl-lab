"""
Test suite for quantum task execution components.

Tests task execution, quantum state management, and execution contexts.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import jax.numpy as jnp

from src.dynamic_graph_fed_rl.quantum_planner.executor import (
    QuantumExecutor,
    ExecutionContext,
    QuantumExecutionResult,
    TaskExecutionError
)
from src.dynamic_graph_fed_rl.quantum_planner.core import QuantumTask, TaskState


class TestQuantumExecutor:
    """Test quantum task execution."""
    
    @pytest.fixture
    def executor(self):
        """Create test executor."""
        return QuantumExecutor(
            max_concurrent_tasks=5,
            quantum_coherence_time=10.0,
            enable_superposition=True
        )
    
    def test_executor_initialization(self, executor):
        """Test executor initialization."""
        assert executor.max_concurrent_tasks == 5
        assert executor.quantum_coherence_time == 10.0
        assert executor.enable_superposition is True
    
    @pytest.mark.asyncio
    async def test_execute_single_task(self, executor):
        """Test single task execution."""
        task = QuantumTask(id="test_task", name="Test Task")
        
        # Mock task function
        async def mock_task_func():
            return "task_result"
        
        context = ExecutionContext(
            task_id="test_task",
            task_function=mock_task_func,
            timeout=5.0
        )
        
        result = await executor.execute_task(context)
        
        assert isinstance(result, QuantumExecutionResult)
        assert result.task_id == "test_task"
        assert result.success is True
        assert result.result == "task_result"
    
    @pytest.mark.asyncio
    async def test_execute_task_with_timeout(self, executor):
        """Test task execution with timeout."""
        # Mock slow task function
        async def slow_task_func():
            await asyncio.sleep(10)
            return "never_reached"
        
        context = ExecutionContext(
            task_id="slow_task",
            task_function=slow_task_func,
            timeout=0.1  # Very short timeout
        )
        
        result = await executor.execute_task(context)
        
        assert result.success is False
        assert "timeout" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_batch_tasks(self, executor):
        """Test batch task execution."""
        # Create multiple test tasks
        contexts = []
        for i in range(3):
            async def task_func(task_id=f"task_{i}"):
                return f"result_{task_id}"
            
            contexts.append(ExecutionContext(
                task_id=f"task_{i}",
                task_function=task_func,
                timeout=5.0
            ))
        
        results = await executor.execute_batch(contexts)
        
        assert len(results) == 3
        assert all(isinstance(r, QuantumExecutionResult) for r in results)
    
    def test_quantum_state_preparation(self, executor):
        """Test quantum state preparation for tasks."""
        task = QuantumTask(id="quantum_task", name="Quantum Task")
        
        prepared_state = executor.prepare_quantum_state(task)
        
        assert "amplitudes" in prepared_state
        assert "coherence" in prepared_state
        assert prepared_state["coherence"] >= 0.0
        assert prepared_state["coherence"] <= 1.0
    
    def test_measure_quantum_state(self, executor):
        """Test quantum state measurement."""
        quantum_state = {
            "amplitudes": jnp.array([0.7+0j, 0.3+0j, 0.0+0j, 0.0+0j]),
            "coherence": 0.95
        }
        
        measurement = executor.measure_quantum_state(quantum_state)
        
        assert "measured_state" in measurement
        assert "probability" in measurement
        assert measurement["probability"] >= 0.0
        assert measurement["probability"] <= 1.0


class TestExecutionContext:
    """Test execution context functionality."""
    
    def test_context_creation(self):
        """Test execution context creation."""
        async def dummy_func():
            return "test"
        
        context = ExecutionContext(
            task_id="test_task",
            task_function=dummy_func,
            timeout=10.0,
            metadata={"priority": "high"}
        )
        
        assert context.task_id == "test_task"
        assert context.timeout == 10.0
        assert context.metadata["priority"] == "high"
    
    def test_context_validation(self):
        """Test execution context validation."""
        with pytest.raises(ValueError):
            ExecutionContext(
                task_id="",  # Invalid empty ID
                task_function=lambda: None,
                timeout=10.0
            )
        
        with pytest.raises(ValueError):
            ExecutionContext(
                task_id="valid_id",
                task_function=None,  # Invalid None function
                timeout=10.0
            )


class TestQuantumExecutionResult:
    """Test execution result handling."""
    
    def test_successful_result(self):
        """Test successful execution result."""
        result = QuantumExecutionResult(
            task_id="success_task",
            success=True,
            result="execution_successful",
            execution_time=1.5
        )
        
        assert result.task_id == "success_task"
        assert result.success is True
        assert result.result == "execution_successful"
        assert result.execution_time == 1.5
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed execution result."""
        result = QuantumExecutionResult(
            task_id="failed_task",
            success=False,
            error="Task execution failed",
            execution_time=0.5
        )
        
        assert result.task_id == "failed_task"
        assert result.success is False
        assert result.error == "Task execution failed"
        assert result.result is None
    
    def test_result_serialization(self):
        """Test result serialization to dict."""
        result = QuantumExecutionResult(
            task_id="serialization_test",
            success=True,
            result={"data": "value"},
            execution_time=2.0,
            quantum_metrics={
                "coherence": 0.95,
                "entanglement": 0.8
            }
        )
        
        serialized = result.to_dict()
        
        assert serialized["task_id"] == "serialization_test"
        assert serialized["success"] is True
        assert serialized["quantum_metrics"]["coherence"] == 0.95


class TestExecutionErrors:
    """Test execution error handling."""
    
    def test_task_execution_error(self):
        """Test task execution error."""
        error = TaskExecutionError(
            task_id="error_task",
            message="Something went wrong",
            error_type="RuntimeError"
        )
        
        assert error.task_id == "error_task"
        assert "Something went wrong" in str(error)
        assert error.error_type == "RuntimeError"
    
    @pytest.mark.asyncio
    async def test_executor_error_handling(self):
        """Test executor error handling."""
        executor = QuantumExecutor()
        
        # Mock task that raises exception
        async def failing_task():
            raise RuntimeError("Task failed")
        
        context = ExecutionContext(
            task_id="failing_task",
            task_function=failing_task,
            timeout=5.0
        )
        
        result = await executor.execute_task(context)
        
        assert result.success is False
        assert result.error is not None
        assert "Task failed" in result.error


class TestExecutionIntegration:
    """Integration tests for execution components."""
    
    @pytest.mark.asyncio
    async def test_full_execution_workflow(self):
        """Test complete execution workflow."""
        executor = QuantumExecutor(max_concurrent_tasks=2)
        
        # Create tasks with different execution patterns
        async def fast_task():
            await asyncio.sleep(0.01)
            return "fast_result"
        
        async def medium_task():
            await asyncio.sleep(0.05)
            return "medium_result"
        
        contexts = [
            ExecutionContext("fast_task", fast_task, 1.0),
            ExecutionContext("medium_task", medium_task, 1.0),
        ]
        
        # Execute batch
        results = await executor.execute_batch(contexts)
        
        # Verify results
        assert len(results) == 2
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 2
        
        # Check execution times are reasonable
        for result in results:
            assert result.execution_time > 0
            assert result.execution_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])