"""
Validation system for quantum task planner.

Implements comprehensive validation and sanitization:
- Task validation with quantum constraints
- Dependency cycle detection
- Resource requirement validation
- Security input sanitization
- Configuration validation
"""

import re
import math
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .core import QuantumTask, TaskState
from .exceptions import (
    TaskValidationError, 
    DependencyError, 
    ResourceAllocationError,
    ConfigurationError
)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    code: str
    message: str
    context: Dict[str, Any]
    suggestion: Optional[str] = None


class TaskValidator:
    """
    Comprehensive task validation with quantum constraints.
    
    Validates tasks for correctness, security, and quantum compatibility.
    """
    
    def __init__(
        self,
        max_duration: float = 3600.0,  # 1 hour max
        max_dependencies: int = 50,
        max_resource_value: float = 1.0,
        allowed_resource_types: Optional[Set[str]] = None,
    ):
        self.max_duration = max_duration
        self.max_dependencies = max_dependencies
        self.max_resource_value = max_resource_value
        self.allowed_resource_types = allowed_resource_types or {
            "cpu", "memory", "io", "network", "gpu", "storage"
        }
        
        # Security patterns
        self.dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'shell=True',
            r'rm\s+-rf',
            r'DROP\s+TABLE',
            r'DELETE\s+FROM'
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
    
    def validate_task(self, task: QuantumTask) -> List[ValidationIssue]:
        """Comprehensive task validation."""
        issues = []
        
        # Basic validation
        issues.extend(self._validate_basic_properties(task))
        
        # Quantum state validation
        issues.extend(self._validate_quantum_state(task))
        
        # Security validation
        issues.extend(self._validate_security(task))
        
        # Resource validation
        issues.extend(self._validate_resources(task))
        
        # Dependency validation (structural only - no cycle check here)
        issues.extend(self._validate_dependencies_structure(task))
        
        return issues
    
    def _validate_basic_properties(self, task: QuantumTask) -> List[ValidationIssue]:
        """Validate basic task properties."""
        issues = []
        
        # Task ID validation
        if not task.id or not isinstance(task.id, str):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_TASK_ID",
                message="Task ID must be a non-empty string",
                context={"task_id": task.id}
            ))
        elif not self._is_safe_identifier(task.id):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="UNSAFE_TASK_ID", 
                message="Task ID contains unsafe characters",
                context={"task_id": task.id},
                suggestion="Use only alphanumeric characters, hyphens, and underscores"
            ))
        
        # Task name validation
        if not task.name or not isinstance(task.name, str):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_TASK_NAME",
                message="Task name must be a non-empty string",
                context={"task_name": task.name}
            ))
        elif len(task.name) > 200:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="LONG_TASK_NAME",
                message="Task name is very long (>200 chars)",
                context={"task_name": task.name, "length": len(task.name)}
            ))
        
        # Duration validation
        if not isinstance(task.estimated_duration, (int, float)):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_DURATION_TYPE",
                message="Estimated duration must be a number",
                context={"duration": task.estimated_duration, "type": type(task.estimated_duration).__name__}
            ))
        elif task.estimated_duration <= 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_DURATION_VALUE",
                message="Estimated duration must be positive",
                context={"duration": task.estimated_duration}
            ))
        elif task.estimated_duration > self.max_duration:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EXCESSIVE_DURATION",
                message=f"Task duration exceeds recommended maximum ({self.max_duration}s)",
                context={"duration": task.estimated_duration, "max_duration": self.max_duration}
            ))
        
        # Priority validation
        if not isinstance(task.priority, (int, float)):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_PRIORITY_TYPE",
                message="Priority must be a number",
                context={"priority": task.priority, "type": type(task.priority).__name__}
            ))
        elif task.priority < 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="NEGATIVE_PRIORITY",
                message="Task has negative priority",
                context={"priority": task.priority}
            ))
        
        return issues
    
    def _validate_quantum_state(self, task: QuantumTask) -> List[ValidationIssue]:
        """Validate quantum state properties."""
        issues = []
        
        if not task.state_amplitudes:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="NO_QUANTUM_STATE",
                message="Task missing quantum state amplitudes",
                context={"task_id": task.id}
            ))
            return issues
        
        # Check amplitude normalization
        total_probability = sum(abs(amp)**2 for amp in task.state_amplitudes.values())
        
        if abs(total_probability - 1.0) > 1e-10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="UNNORMALIZED_AMPLITUDES",
                message="Quantum amplitudes are not normalized",
                context={
                    "task_id": task.id,
                    "total_probability": total_probability,
                    "deviation": abs(total_probability - 1.0)
                },
                suggestion="Amplitudes should sum to probability 1.0"
            ))
        
        # Check for required quantum states
        required_states = {TaskState.PENDING, TaskState.ACTIVE, TaskState.COMPLETED, TaskState.FAILED}
        missing_states = required_states - set(task.state_amplitudes.keys())
        
        if missing_states:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MISSING_QUANTUM_STATES",
                message="Task missing required quantum states",
                context={
                    "task_id": task.id,
                    "missing_states": [state.value for state in missing_states]
                }
            ))
        
        # Check for invalid amplitudes
        for state, amplitude in task.state_amplitudes.items():
            if not isinstance(amplitude, complex):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_AMPLITUDE_TYPE",
                    message=f"Amplitude for state {state.value} is not complex",
                    context={"task_id": task.id, "state": state.value, "amplitude": amplitude}
                ))
            elif math.isnan(amplitude.real) or math.isnan(amplitude.imag):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="NAN_AMPLITUDE",
                    message=f"Amplitude for state {state.value} contains NaN",
                    context={"task_id": task.id, "state": state.value}
                ))
            elif math.isinf(amplitude.real) or math.isinf(amplitude.imag):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INFINITE_AMPLITUDE",
                    message=f"Amplitude for state {state.value} is infinite",
                    context={"task_id": task.id, "state": state.value}
                ))
        
        return issues
    
    def _validate_security(self, task: QuantumTask) -> List[ValidationIssue]:
        """Validate task for security issues."""
        issues = []
        
        # Check task name and ID for dangerous patterns
        for field_name, field_value in [("id", task.id), ("name", task.name)]:
            if isinstance(field_value, str):
                for pattern in self.compiled_patterns:
                    if pattern.search(field_value):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            code="DANGEROUS_PATTERN_DETECTED",
                            message=f"Dangerous pattern detected in task {field_name}",
                            context={
                                "task_id": task.id,
                                "field": field_name,
                                "value": field_value,
                                "pattern": pattern.pattern
                            },
                            suggestion="Remove dangerous code patterns from task definitions"
                        ))
        
        # Validate executor function (if present)
        if task.executor is not None:
            if hasattr(task.executor, '__code__'):
                # Check function code for dangerous patterns
                try:
                    code_string = str(task.executor.__code__.co_code)
                    for pattern in self.compiled_patterns:
                        if pattern.search(code_string):
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.CRITICAL,
                                code="DANGEROUS_EXECUTOR_CODE",
                                message="Executor function contains dangerous code patterns",
                                context={"task_id": task.id},
                                suggestion="Review and sanitize executor function"
                            ))
                except Exception:
                    # If we can't inspect the code, issue a warning
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="UNINSPECTABLE_EXECUTOR",
                        message="Cannot inspect executor function for security",
                        context={"task_id": task.id}
                    ))
        
        return issues
    
    def _validate_resources(self, task: QuantumTask) -> List[ValidationIssue]:
        """Validate resource requirements."""
        issues = []
        
        if not isinstance(task.resource_requirements, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_RESOURCE_TYPE",
                message="Resource requirements must be a dictionary",
                context={"task_id": task.id, "type": type(task.resource_requirements).__name__}
            ))
            return issues
        
        total_resources = 0.0
        
        for resource_name, resource_value in task.resource_requirements.items():
            # Validate resource name
            if not isinstance(resource_name, str):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_RESOURCE_NAME_TYPE",
                    message="Resource name must be a string",
                    context={"task_id": task.id, "resource_name": resource_name}
                ))
                continue
            
            if not self._is_safe_identifier(resource_name):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="UNSAFE_RESOURCE_NAME",
                    message="Resource name contains unsafe characters",
                    context={"task_id": task.id, "resource_name": resource_name}
                ))
            
            if resource_name not in self.allowed_resource_types:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="UNKNOWN_RESOURCE_TYPE",
                    message=f"Unknown resource type: {resource_name}",
                    context={
                        "task_id": task.id,
                        "resource_name": resource_name,
                        "allowed_types": list(self.allowed_resource_types)
                    }
                ))
            
            # Validate resource value
            if not isinstance(resource_value, (int, float)):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_RESOURCE_VALUE_TYPE",
                    message="Resource value must be a number",
                    context={
                        "task_id": task.id,
                        "resource_name": resource_name,
                        "value": resource_value,
                        "type": type(resource_value).__name__
                    }
                ))
                continue
            
            if resource_value < 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="NEGATIVE_RESOURCE_VALUE",
                    message="Resource value cannot be negative",
                    context={"task_id": task.id, "resource_name": resource_name, "value": resource_value}
                ))
            
            if resource_value > self.max_resource_value:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="EXCESSIVE_RESOURCE_VALUE",
                    message=f"Resource value exceeds maximum ({self.max_resource_value})",
                    context={
                        "task_id": task.id,
                        "resource_name": resource_name,
                        "value": resource_value,
                        "max_value": self.max_resource_value
                    }
                ))
            
            if math.isnan(resource_value) or math.isinf(resource_value):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_RESOURCE_VALUE",
                    message="Resource value is NaN or infinite",
                    context={"task_id": task.id, "resource_name": resource_name, "value": resource_value}
                ))
            
            total_resources += resource_value
        
        # Check total resource consumption
        if total_resources > len(self.allowed_resource_types) * self.max_resource_value:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EXCESSIVE_TOTAL_RESOURCES",
                message="Total resource requirements are very high",
                context={"task_id": task.id, "total_resources": total_resources}
            ))
        
        return issues
    
    def _validate_dependencies_structure(self, task: QuantumTask) -> List[ValidationIssue]:
        """Validate dependency structure (not cycles)."""
        issues = []
        
        if not isinstance(task.dependencies, set):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_DEPENDENCIES_TYPE",
                message="Dependencies must be a set",
                context={"task_id": task.id, "type": type(task.dependencies).__name__}
            ))
            return issues
        
        if len(task.dependencies) > self.max_dependencies:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EXCESSIVE_DEPENDENCIES",
                message=f"Task has too many dependencies ({len(task.dependencies)} > {self.max_dependencies})",
                context={"task_id": task.id, "dependency_count": len(task.dependencies)}
            ))
        
        # Check each dependency
        for dep_id in task.dependencies:
            if not isinstance(dep_id, str):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_DEPENDENCY_TYPE",
                    message="Dependency ID must be a string",
                    context={"task_id": task.id, "dependency_id": dep_id}
                ))
            elif not self._is_safe_identifier(dep_id):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="UNSAFE_DEPENDENCY_ID",
                    message="Dependency ID contains unsafe characters",
                    context={"task_id": task.id, "dependency_id": dep_id}
                ))
            elif dep_id == task.id:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SELF_DEPENDENCY",
                    message="Task cannot depend on itself",
                    context={"task_id": task.id}
                ))
        
        return issues
    
    def _is_safe_identifier(self, identifier: str) -> bool:
        """Check if identifier is safe (alphanumeric, hyphens, underscores only)."""
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', identifier))


class DependencyValidator:
    """
    Validates task dependency graphs for cycles and consistency.
    """
    
    def __init__(self):
        self.visited = set()
        self.recursion_stack = set()
    
    def validate_dependencies(
        self, 
        tasks: Dict[str, QuantumTask]
    ) -> List[ValidationIssue]:
        """Validate all task dependencies for cycles and missing references."""
        issues = []
        
        # Check for missing dependencies
        issues.extend(self._check_missing_dependencies(tasks))
        
        # Check for circular dependencies
        issues.extend(self._check_circular_dependencies(tasks))
        
        # Check for dependency depth
        issues.extend(self._check_dependency_depth(tasks))
        
        return issues
    
    def _check_missing_dependencies(self, tasks: Dict[str, QuantumTask]) -> List[ValidationIssue]:
        """Check for dependencies that reference non-existent tasks."""
        issues = []
        
        for task_id, task in tasks.items():
            missing_deps = task.dependencies - set(tasks.keys())
            if missing_deps:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_DEPENDENCIES",
                    message=f"Task references non-existent dependencies",
                    context={
                        "task_id": task_id,
                        "missing_dependencies": list(missing_deps)
                    },
                    suggestion="Ensure all dependencies exist in the task set"
                ))
        
        return issues
    
    def _check_circular_dependencies(self, tasks: Dict[str, QuantumTask]) -> List[ValidationIssue]:
        """Check for circular dependency chains."""
        issues = []
        
        # Reset state for each validation
        self.visited = set()
        self.recursion_stack = set()
        
        for task_id in tasks:
            if task_id not in self.visited:
                cycle = self._detect_cycle_dfs(task_id, tasks, [])
                if cycle:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="CIRCULAR_DEPENDENCY",
                        message="Circular dependency detected",
                        context={
                            "cycle": cycle,
                            "affected_tasks": cycle
                        },
                        suggestion="Remove circular dependencies by restructuring task relationships"
                    ))
        
        return issues
    
    def _detect_cycle_dfs(
        self, 
        task_id: str, 
        tasks: Dict[str, QuantumTask], 
        path: List[str]
    ) -> Optional[List[str]]:
        """Detect cycles using depth-first search."""
        if task_id in self.recursion_stack:
            # Found a cycle - return the cycle path
            cycle_start = path.index(task_id)
            return path[cycle_start:] + [task_id]
        
        if task_id in self.visited or task_id not in tasks:
            return None
        
        self.visited.add(task_id)
        self.recursion_stack.add(task_id)
        
        current_path = path + [task_id]
        
        for dep_id in tasks[task_id].dependencies:
            cycle = self._detect_cycle_dfs(dep_id, tasks, current_path)
            if cycle:
                return cycle
        
        self.recursion_stack.remove(task_id)
        return None
    
    def _check_dependency_depth(self, tasks: Dict[str, QuantumTask]) -> List[ValidationIssue]:
        """Check for excessive dependency chain depths."""
        issues = []
        max_depth = 20  # Reasonable maximum depth
        
        for task_id in tasks:
            depth = self._calculate_dependency_depth(task_id, tasks)
            if depth > max_depth:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="EXCESSIVE_DEPENDENCY_DEPTH",
                    message=f"Task has very deep dependency chain (depth: {depth})",
                    context={"task_id": task_id, "depth": depth, "max_depth": max_depth},
                    suggestion="Consider flattening dependency structure"
                ))
        
        return issues
    
    def _calculate_dependency_depth(
        self, 
        task_id: str, 
        tasks: Dict[str, QuantumTask], 
        visited: Optional[Set[str]] = None
    ) -> int:
        """Calculate maximum dependency depth for a task."""
        if visited is None:
            visited = set()
        
        if task_id in visited or task_id not in tasks:
            return 0
        
        visited.add(task_id)
        
        task = tasks[task_id]
        if not task.dependencies:
            return 1
        
        max_dep_depth = 0
        for dep_id in task.dependencies:
            dep_depth = self._calculate_dependency_depth(dep_id, tasks, visited.copy())
            max_dep_depth = max(max_dep_depth, dep_depth)
        
        return max_dep_depth + 1


class ConfigurationValidator:
    """
    Validates quantum planner configuration parameters.
    """
    
    def __init__(self):
        self.parameter_ranges = {
            "max_parallel_tasks": (1, 100),
            "quantum_coherence_time": (0.1, 3600.0),
            "interference_strength": (0.0, 1.0),
            "learning_rate": (0.001, 1.0),
            "exploration_rate": (0.0, 1.0),
            "max_iterations": (1, 10000),
            "convergence_threshold": (1e-12, 1e-1),
            "measurement_interval": (0.01, 60.0),
            "decoherence_time": (0.1, 3600.0),
            "error_probability": (0.0, 1.0),
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate configuration parameters."""
        issues = []
        
        for param_name, value in config.items():
            issues.extend(self._validate_parameter(param_name, value))
        
        # Check for conflicting parameters
        issues.extend(self._check_parameter_conflicts(config))
        
        return issues
    
    def _validate_parameter(self, param_name: str, value: Any) -> List[ValidationIssue]:
        """Validate individual parameter."""
        issues = []
        
        if param_name in self.parameter_ranges:
            min_val, max_val = self.parameter_ranges[param_name]
            
            # Type validation
            if not isinstance(value, (int, float)):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_PARAMETER_TYPE",
                    message=f"Parameter {param_name} must be a number",
                    context={"parameter": param_name, "value": value, "type": type(value).__name__}
                ))
                return issues
            
            # Range validation
            if value < min_val:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="PARAMETER_BELOW_MINIMUM",
                    message=f"Parameter {param_name} below minimum value",
                    context={"parameter": param_name, "value": value, "minimum": min_val}
                ))
            elif value > max_val:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="PARAMETER_ABOVE_MAXIMUM", 
                    message=f"Parameter {param_name} above maximum value",
                    context={"parameter": param_name, "value": value, "maximum": max_val}
                ))
            
            # Special validations
            if math.isnan(value) or math.isinf(value):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_PARAMETER_VALUE",
                    message=f"Parameter {param_name} is NaN or infinite",
                    context={"parameter": param_name, "value": value}
                ))
        
        return issues
    
    def _check_parameter_conflicts(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Check for conflicting parameter combinations."""
        issues = []
        
        # Coherence time vs measurement interval
        if "quantum_coherence_time" in config and "measurement_interval" in config:
            coherence_time = config["quantum_coherence_time"]
            measurement_interval = config["measurement_interval"]
            
            if measurement_interval >= coherence_time:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="MEASUREMENT_EXCEEDS_COHERENCE",
                    message="Measurement interval exceeds coherence time",
                    context={
                        "coherence_time": coherence_time,
                        "measurement_interval": measurement_interval
                    },
                    suggestion="Set measurement interval < coherence time for meaningful quantum effects"
                ))
        
        # Learning rate vs exploration rate
        if "learning_rate" in config and "exploration_rate" in config:
            learning_rate = config["learning_rate"]
            exploration_rate = config["exploration_rate"]
            
            if learning_rate > 0.5 and exploration_rate > 0.5:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="HIGH_LEARNING_AND_EXPLORATION",
                    message="Both learning rate and exploration rate are high",
                    context={
                        "learning_rate": learning_rate,
                        "exploration_rate": exploration_rate
                    },
                    suggestion="Consider reducing one rate to avoid instability"
                ))
        
        return issues


def validate_task_batch(tasks: Dict[str, QuantumTask]) -> Tuple[List[ValidationIssue], bool]:
    """
    Validate a batch of tasks comprehensively.
    
    Returns:
        Tuple of (issues, is_valid) where is_valid indicates if all critical issues are resolved
    """
    all_issues = []
    
    # Individual task validation
    task_validator = TaskValidator()
    for task in tasks.values():
        task_issues = task_validator.validate_task(task)
        all_issues.extend(task_issues)
    
    # Dependency validation
    dependency_validator = DependencyValidator()
    dependency_issues = dependency_validator.validate_dependencies(tasks)
    all_issues.extend(dependency_issues)
    
    # Check if validation passes (no critical/error issues)
    critical_issues = [
        issue for issue in all_issues
        if issue.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}
    ]
    
    is_valid = len(critical_issues) == 0
    
    return all_issues, is_valid


def sanitize_task_input(raw_task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize raw task input data for security and correctness.
    
    Args:
        raw_task_data: Raw task data from external source
        
    Returns:
        Sanitized task data
    """
    sanitized = {}
    
    # Sanitize string fields
    string_fields = ["id", "name"]
    for field in string_fields:
        if field in raw_task_data:
            value = raw_task_data[field]
            if isinstance(value, str):
                # Remove dangerous characters and limit length
                sanitized_value = re.sub(r'[^\w\s\-_.]', '', value)[:200]
                sanitized[field] = sanitized_value.strip()
    
    # Sanitize numeric fields
    numeric_fields = ["estimated_duration", "priority"]
    for field in numeric_fields:
        if field in raw_task_data:
            value = raw_task_data[field]
            try:
                if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                    sanitized[field] = float(value)
            except (ValueError, TypeError):
                pass  # Skip invalid values
    
    # Sanitize dependencies
    if "dependencies" in raw_task_data:
        deps = raw_task_data["dependencies"]
        if isinstance(deps, (list, set)):
            sanitized_deps = set()
            for dep in deps:
                if isinstance(dep, str):
                    sanitized_dep = re.sub(r'[^\w\-_]', '', dep)[:100]
                    if sanitized_dep:
                        sanitized_deps.add(sanitized_dep)
            sanitized["dependencies"] = sanitized_deps
    
    # Sanitize resource requirements
    if "resource_requirements" in raw_task_data:
        resources = raw_task_data["resource_requirements"]
        if isinstance(resources, dict):
            sanitized_resources = {}
            allowed_resources = {"cpu", "memory", "io", "network", "gpu", "storage"}
            
            for resource_name, resource_value in resources.items():
                if isinstance(resource_name, str) and resource_name in allowed_resources:
                    try:
                        if isinstance(resource_value, (int, float)):
                            if not (math.isnan(resource_value) or math.isinf(resource_value)):
                                # Clamp to reasonable range
                                clamped_value = max(0.0, min(1.0, float(resource_value)))
                                sanitized_resources[resource_name] = clamped_value
                    except (ValueError, TypeError):
                        pass
            
            sanitized["resource_requirements"] = sanitized_resources
    
    return sanitized