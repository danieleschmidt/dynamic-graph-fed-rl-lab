"""
Custom exceptions for quantum task planner.

Defines specific exceptions for quantum-inspired task planning operations
with proper error handling and recovery mechanisms.
"""

from typing import Optional, Dict, Any, List


class QuantumPlannerError(Exception):
    """Base exception for quantum planner operations."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "QUANTUM_PLANNER_ERROR"
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class TaskValidationError(QuantumPlannerError):
    """Exception raised when task validation fails."""
    
    def __init__(
        self, 
        task_id: str, 
        validation_errors: List[str],
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Task validation failed for '{task_id}': {', '.join(validation_errors)}"
        super().__init__(message, "TASK_VALIDATION_ERROR", context)
        self.task_id = task_id
        self.validation_errors = validation_errors


class DependencyError(QuantumPlannerError):
    """Exception raised when task dependencies cannot be resolved."""
    
    def __init__(
        self, 
        task_id: str, 
        missing_dependencies: List[str],
        circular_dependencies: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        errors = []
        if missing_dependencies:
            errors.append(f"missing dependencies: {', '.join(missing_dependencies)}")
        if circular_dependencies:
            errors.append(f"circular dependencies: {' -> '.join(circular_dependencies)}")
        
        message = f"Dependency error for task '{task_id}': {'; '.join(errors)}"
        super().__init__(message, "DEPENDENCY_ERROR", context)
        self.task_id = task_id
        self.missing_dependencies = missing_dependencies or []
        self.circular_dependencies = circular_dependencies or []


class ResourceAllocationError(QuantumPlannerError):
    """Exception raised when resources cannot be allocated for task execution."""
    
    def __init__(
        self, 
        task_id: str, 
        required_resources: Dict[str, float],
        available_resources: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Cannot allocate resources for task '{task_id}': required {required_resources}, available {available_resources}"
        super().__init__(message, "RESOURCE_ALLOCATION_ERROR", context)
        self.task_id = task_id
        self.required_resources = required_resources
        self.available_resources = available_resources


class QuantumStateError(QuantumPlannerError):
    """Exception raised when quantum state operations fail."""
    
    def __init__(
        self, 
        operation: str,
        details: str,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Quantum state error during {operation}: {details}"
        super().__init__(message, "QUANTUM_STATE_ERROR", context)
        self.operation = operation
        self.details = details


class ExecutionError(QuantumPlannerError):
    """Exception raised during task execution."""
    
    def __init__(
        self, 
        task_id: str, 
        execution_error: str,
        error_type: str = "GENERAL",
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Execution error in task '{task_id}': {execution_error}"
        super().__init__(message, f"EXECUTION_ERROR_{error_type}", context)
        self.task_id = task_id
        self.execution_error = execution_error
        self.error_type = error_type


class OptimizationError(QuantumPlannerError):
    """Exception raised during quantum optimization."""
    
    def __init__(
        self, 
        optimization_type: str,
        details: str,
        iteration: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Optimization error in {optimization_type}: {details}"
        if iteration is not None:
            message += f" (iteration {iteration})"
        super().__init__(message, "OPTIMIZATION_ERROR", context)
        self.optimization_type = optimization_type
        self.details = details
        self.iteration = iteration


class SchedulingError(QuantumPlannerError):
    """Exception raised during task scheduling."""
    
    def __init__(
        self, 
        scheduler_type: str,
        details: str,
        affected_tasks: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Scheduling error in {scheduler_type}: {details}"
        if affected_tasks:
            message += f" (affected tasks: {', '.join(affected_tasks)})"
        super().__init__(message, "SCHEDULING_ERROR", context)
        self.scheduler_type = scheduler_type
        self.details = details
        self.affected_tasks = affected_tasks or []


class DecoherenceError(QuantumPlannerError):
    """Exception raised when quantum decoherence affects system state."""
    
    def __init__(
        self, 
        decoherence_type: str,
        affected_tasks: List[str],
        coherence_time: float,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Decoherence error ({decoherence_type}): {len(affected_tasks)} tasks affected after {coherence_time}s"
        super().__init__(message, "DECOHERENCE_ERROR", context)
        self.decoherence_type = decoherence_type
        self.affected_tasks = affected_tasks
        self.coherence_time = coherence_time


class InterferenceError(QuantumPlannerError):
    """Exception raised when quantum interference calculations fail."""
    
    def __init__(
        self, 
        interference_type: str,
        path_count: int,
        details: str,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Interference error ({interference_type}) with {path_count} paths: {details}"
        super().__init__(message, "INTERFERENCE_ERROR", context)
        self.interference_type = interference_type
        self.path_count = path_count
        self.details = details


class ConfigurationError(QuantumPlannerError):
    """Exception raised when configuration is invalid."""
    
    def __init__(
        self, 
        parameter: str,
        value: Any,
        expected: str,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Invalid configuration for parameter '{parameter}': got {value}, expected {expected}"
        super().__init__(message, "CONFIGURATION_ERROR", context)
        self.parameter = parameter
        self.value = value
        self.expected = expected


class TimeoutError(QuantumPlannerError):
    """Exception raised when operations timeout."""
    
    def __init__(
        self, 
        operation: str,
        timeout_seconds: float,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Timeout error: {operation} exceeded {timeout_seconds}s"
        super().__init__(message, "TIMEOUT_ERROR", context)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class QuantumMeasurementError(QuantumPlannerError):
    """Exception raised during quantum measurement operations."""
    
    def __init__(
        self, 
        measurement_type: str,
        quantum_state: Dict[str, float],
        details: str,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Quantum measurement error ({measurement_type}): {details}"
        super().__init__(message, "QUANTUM_MEASUREMENT_ERROR", context)
        self.measurement_type = measurement_type
        self.quantum_state = quantum_state
        self.details = details