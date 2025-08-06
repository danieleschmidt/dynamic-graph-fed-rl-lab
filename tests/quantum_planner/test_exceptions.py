"""
Test suite for quantum planner exception handling.

Tests custom exceptions, error handling, and recovery mechanisms.
"""

import pytest
from unittest.mock import Mock

from src.dynamic_graph_fed_rl.quantum_planner.exceptions import (
    QuantumPlannerError,
    TaskExecutionError,
    QuantumStateError,
    ValidationError,
    ResourceExhaustionError,
    SecurityViolationError,
    OptimizationError,
    ConcurrencyError,
    TimeoutError,
    CoherenceError,
    EntanglementError,
    MeasurementError,
    SchedulingError,
    ErrorHandler,
    ErrorRecovery
)


class TestQuantumPlannerExceptions:
    """Test base and specific quantum planner exceptions."""
    
    def test_base_exception(self):
        """Test base quantum planner exception."""
        error = QuantumPlannerError("Base error occurred")
        
        assert str(error) == "Base error occurred"
        assert isinstance(error, Exception)
    
    def test_task_execution_error(self):
        """Test task execution error."""
        error = TaskExecutionError(
            task_id="failed_task",
            message="Task execution failed",
            error_code="EXEC_001"
        )
        
        assert error.task_id == "failed_task"
        assert error.error_code == "EXEC_001"
        assert "failed_task" in str(error)
        assert "Task execution failed" in str(error)
    
    def test_quantum_state_error(self):
        """Test quantum state error."""
        error = QuantumStateError(
            state_id="quantum_state_1",
            message="Invalid quantum state",
            coherence=0.3
        )
        
        assert error.state_id == "quantum_state_1"
        assert error.coherence == 0.3
        assert "quantum_state_1" in str(error)
    
    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError(
            field="task_name",
            value="<script>alert('xss')</script>",
            message="Invalid characters in task name"
        )
        
        assert error.field == "task_name"
        assert error.value == "<script>alert('xss')</script>"
        assert "task_name" in str(error)
    
    def test_resource_exhaustion_error(self):
        """Test resource exhaustion error."""
        error = ResourceExhaustionError(
            resource_type="memory",
            requested=16.0,
            available=8.0,
            unit="GB"
        )
        
        assert error.resource_type == "memory"
        assert error.requested == 16.0
        assert error.available == 8.0
        assert error.unit == "GB"
        assert "memory" in str(error)
        assert "16.0" in str(error)
    
    def test_security_violation_error(self):
        """Test security violation error."""
        error = SecurityViolationError(
            violation_type="unauthorized_access",
            user_id="malicious_user",
            resource="admin_panel",
            severity="high"
        )
        
        assert error.violation_type == "unauthorized_access"
        assert error.user_id == "malicious_user"
        assert error.resource == "admin_panel"
        assert error.severity == "high"
        assert "unauthorized_access" in str(error)
    
    def test_optimization_error(self):
        """Test optimization error."""
        error = OptimizationError(
            algorithm="quantum_interference",
            iteration=50,
            message="Optimization failed to converge"
        )
        
        assert error.algorithm == "quantum_interference"
        assert error.iteration == 50
        assert "quantum_interference" in str(error)
        assert "50" in str(error)
    
    def test_concurrency_error(self):
        """Test concurrency error."""
        error = ConcurrencyError(
            operation="task_execution",
            conflicting_tasks=["task_1", "task_2"],
            message="Resource conflict detected"
        )
        
        assert error.operation == "task_execution"
        assert error.conflicting_tasks == ["task_1", "task_2"]
        assert "task_execution" in str(error)
        assert "task_1" in str(error)
    
    def test_timeout_error(self):
        """Test timeout error."""
        error = TimeoutError(
            operation="task_execution",
            timeout_duration=30.0,
            elapsed_time=45.0
        )
        
        assert error.operation == "task_execution"
        assert error.timeout_duration == 30.0
        assert error.elapsed_time == 45.0
        assert "30.0" in str(error)
        assert "45.0" in str(error)


class TestQuantumSpecificExceptions:
    """Test quantum-specific exceptions."""
    
    def test_coherence_error(self):
        """Test coherence error."""
        error = CoherenceError(
            current_coherence=0.3,
            required_coherence=0.8,
            decoherence_rate=0.1
        )
        
        assert error.current_coherence == 0.3
        assert error.required_coherence == 0.8
        assert error.decoherence_rate == 0.1
        assert "0.3" in str(error)
        assert "0.8" in str(error)
    
    def test_entanglement_error(self):
        """Test entanglement error."""
        error = EntanglementError(
            task_ids=["task_1", "task_2"],
            entanglement_strength=0.2,
            message="Entanglement too weak for operation"
        )
        
        assert error.task_ids == ["task_1", "task_2"]
        assert error.entanglement_strength == 0.2
        assert "task_1" in str(error)
        assert "0.2" in str(error)
    
    def test_measurement_error(self):
        """Test measurement error."""
        error = MeasurementError(
            measurement_type="state_collapse",
            quantum_state="superposition",
            error_probability=0.15
        )
        
        assert error.measurement_type == "state_collapse"
        assert error.quantum_state == "superposition"
        assert error.error_probability == 0.15
        assert "state_collapse" in str(error)
    
    def test_scheduling_error(self):
        """Test scheduling error."""
        error = SchedulingError(
            scheduler_type="quantum_interference",
            task_count=100,
            available_resources=50,
            message="Insufficient resources for optimal scheduling"
        )
        
        assert error.scheduler_type == "quantum_interference"
        assert error.task_count == 100
        assert error.available_resources == 50
        assert "quantum_interference" in str(error)


class TestErrorHandler:
    """Test error handling functionality."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler."""
        return ErrorHandler(
            enable_logging=True,
            enable_recovery=True,
            max_recovery_attempts=3
        )
    
    def test_error_handler_initialization(self, error_handler):
        """Test error handler initialization."""
        assert error_handler.enable_logging is True
        assert error_handler.enable_recovery is True
        assert error_handler.max_recovery_attempts == 3
    
    def test_handle_task_execution_error(self, error_handler):
        """Test handling task execution error."""
        error = TaskExecutionError(
            task_id="failed_task",
            message="Execution failed",
            error_code="EXEC_001"
        )
        
        result = error_handler.handle_error(error)
        
        assert result["error_handled"] is True
        assert result["recovery_attempted"] is True
        assert result["error_type"] == "TaskExecutionError"
    
    def test_handle_quantum_state_error(self, error_handler):
        """Test handling quantum state error."""
        error = QuantumStateError(
            state_id="corrupted_state",
            message="Quantum state corrupted",
            coherence=0.2
        )
        
        result = error_handler.handle_error(error)
        
        assert result["error_handled"] is True
        assert result["recommended_action"] == "restore_quantum_state"
    
    def test_handle_resource_exhaustion_error(self, error_handler):
        """Test handling resource exhaustion error."""
        error = ResourceExhaustionError(
            resource_type="memory",
            requested=32.0,
            available=16.0
        )
        
        result = error_handler.handle_error(error)
        
        assert result["error_handled"] is True
        assert result["recommended_action"] == "scale_resources"
    
    def test_error_recovery_attempt_limit(self, error_handler):
        """Test error recovery attempt limiting."""
        error = TaskExecutionError("test_task", "Repeated failure")
        
        # Record multiple recovery attempts
        for i in range(5):  # Exceeds max_recovery_attempts
            result = error_handler.handle_error(error)
            
            if i >= 3:  # After max attempts
                assert result["recovery_attempted"] is False
                assert result["recovery_exhausted"] is True
    
    def test_error_categorization(self, error_handler):
        """Test error categorization."""
        errors = [
            TaskExecutionError("task1", "Execution failed"),
            ValidationError("field", "value", "Invalid"),
            ResourceExhaustionError("cpu", 8.0, 4.0),
            SecurityViolationError("breach", "user", "resource")
        ]
        
        categories = error_handler.categorize_errors(errors)
        
        assert "execution" in categories
        assert "validation" in categories  
        assert "resource" in categories
        assert "security" in categories
    
    def test_error_pattern_detection(self, error_handler):
        """Test error pattern detection."""
        # Simulate recurring error pattern
        errors = [
            TaskExecutionError("task_a", "Memory error"),
            TaskExecutionError("task_b", "Memory error"),
            TaskExecutionError("task_c", "Memory error"),
            ResourceExhaustionError("memory", 16.0, 8.0)
        ]
        
        patterns = error_handler.detect_error_patterns(errors)
        
        memory_pattern = next(
            (p for p in patterns if "memory" in p["pattern_type"]), 
            None
        )
        assert memory_pattern is not None
        assert memory_pattern["frequency"] >= 3


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    @pytest.fixture
    def recovery(self):
        """Create error recovery system."""
        return ErrorRecovery(
            enable_auto_recovery=True,
            recovery_timeout=60.0,
            max_retry_attempts=3
        )
    
    def test_recovery_initialization(self, recovery):
        """Test recovery system initialization."""
        assert recovery.enable_auto_recovery is True
        assert recovery.recovery_timeout == 60.0
        assert recovery.max_retry_attempts == 3
    
    def test_task_execution_recovery(self, recovery):
        """Test task execution recovery."""
        error = TaskExecutionError("failed_task", "Network timeout")
        
        recovery_plan = recovery.create_recovery_plan(error)
        
        assert recovery_plan["strategy"] == "retry"
        assert recovery_plan["max_attempts"] <= 3
        assert recovery_plan["backoff_strategy"] == "exponential"
    
    def test_quantum_state_recovery(self, recovery):
        """Test quantum state recovery."""
        error = CoherenceError(
            current_coherence=0.4,
            required_coherence=0.8
        )
        
        recovery_plan = recovery.create_recovery_plan(error)
        
        assert recovery_plan["strategy"] == "restore_coherence"
        assert "coherence_restoration" in recovery_plan["actions"]
    
    def test_resource_recovery(self, recovery):
        """Test resource recovery."""
        error = ResourceExhaustionError("memory", 32.0, 16.0)
        
        recovery_plan = recovery.create_recovery_plan(error)
        
        assert recovery_plan["strategy"] == "resource_optimization"
        assert "garbage_collection" in recovery_plan["actions"]
    
    def test_recovery_execution(self, recovery):
        """Test recovery plan execution."""
        error = TaskExecutionError("test_task", "Temporary failure")
        recovery_plan = recovery.create_recovery_plan(error)
        
        # Mock recovery action
        def mock_retry_action():
            return {"success": True, "message": "Task retried successfully"}
        
        recovery_plan["execute"] = mock_retry_action
        
        result = recovery.execute_recovery(recovery_plan)
        
        assert result["success"] is True
        assert "retried successfully" in result["message"]
    
    def test_recovery_failure_handling(self, recovery):
        """Test handling of recovery failures."""
        error = ValidationError("field", "value", "Permanent validation error")
        
        recovery_plan = recovery.create_recovery_plan(error)
        
        # Some errors cannot be automatically recovered
        assert recovery_plan["strategy"] == "manual_intervention"
        assert recovery_plan["auto_recoverable"] is False


class TestExceptionIntegration:
    """Integration tests for exception handling system."""
    
    def test_complete_error_handling_workflow(self):
        """Test complete error handling workflow."""
        error_handler = ErrorHandler()
        recovery = ErrorRecovery()
        
        # Simulate complex error scenario
        original_error = TaskExecutionError(
            task_id="complex_task",
            message="Multiple cascading failures",
            error_code="EXEC_999"
        )
        
        # Step 1: Handle initial error
        handling_result = error_handler.handle_error(original_error)
        assert handling_result["error_handled"] is True
        
        # Step 2: Create recovery plan
        recovery_plan = recovery.create_recovery_plan(original_error)
        assert recovery_plan["strategy"] in ["retry", "fallback", "manual_intervention"]
        
        # Step 3: Log error pattern
        error_handler.log_error_occurrence(original_error)
        
        # Step 4: Check for error patterns
        patterns = error_handler.get_recent_error_patterns()
        assert isinstance(patterns, list)
    
    def test_cascading_error_handling(self):
        """Test handling of cascading errors."""
        error_handler = ErrorHandler()
        
        # Simulate cascading errors
        primary_error = ResourceExhaustionError("memory", 32.0, 16.0)
        secondary_error = TaskExecutionError("dependent_task", "Failed due to resource shortage")
        
        # Handle cascading errors
        primary_result = error_handler.handle_error(primary_error)
        secondary_result = error_handler.handle_error(secondary_error)
        
        # Should detect relationship
        related_errors = error_handler.find_related_errors(primary_error)
        assert len(related_errors) > 0
    
    def test_error_metrics_collection(self):
        """Test error metrics collection."""
        error_handler = ErrorHandler(enable_metrics=True)
        
        # Generate various errors
        errors = [
            TaskExecutionError("task1", "Error 1"),
            ValidationError("field1", "value1", "Error 2"),
            ResourceExhaustionError("cpu", 8.0, 4.0),
            TaskExecutionError("task2", "Error 3"),
        ]
        
        for error in errors:
            error_handler.handle_error(error)
        
        metrics = error_handler.get_error_metrics()
        
        assert metrics["total_errors"] == 4
        assert metrics["error_types"]["TaskExecutionError"] == 2
        assert metrics["error_types"]["ValidationError"] == 1
        assert metrics["error_types"]["ResourceExhaustionError"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])