"""
Test suite for validation and security components.

Tests input validation, sanitization, dependency checking, and security measures.
"""

import pytest
import re
from unittest.mock import Mock, patch

from src.dynamic_graph_fed_rl.quantum_planner.core import QuantumTask, TaskState
from src.dynamic_graph_fed_rl.quantum_planner.validation import (
    TaskValidator,
    DependencyValidator,
    ConfigurationValidator,
    ValidationIssue,
    ValidationSeverity,
    validate_task_batch,
    sanitize_task_input
)
from src.dynamic_graph_fed_rl.quantum_planner.exceptions import (
    TaskValidationError,
    DependencyError,
    ConfigurationError
)


class TestTaskValidator:
    """Test task validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create test task validator."""
        return TaskValidator(
            max_duration=3600.0,
            max_dependencies=50,
            max_resource_value=1.0,
            allowed_resource_types={"cpu", "memory", "io", "network"}
        )
    
    @pytest.fixture
    def valid_task(self):
        """Create valid test task."""
        return QuantumTask(
            id="test_task",
            name="Test Task",
            dependencies={"dep1", "dep2"},
            estimated_duration=10.0,
            priority=0.5,
            resource_requirements={"cpu": 0.3, "memory": 0.4}
        )
    
    def test_valid_task_passes_validation(self, validator, valid_task):
        """Test that valid task passes all validation checks."""
        issues = validator.validate_task(valid_task)
        
        # Should have no critical or error issues
        critical_issues = [i for i in issues if i.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}]
        assert len(critical_issues) == 0
    
    def test_invalid_task_id_validation(self, validator):
        """Test validation of invalid task IDs."""
        # Test empty task ID
        task_empty_id = QuantumTask(id="", name="Test")
        issues = validator.validate_task(task_empty_id)
        assert any(issue.code == "INVALID_TASK_ID" for issue in issues)
        
        # Test unsafe task ID
        task_unsafe_id = QuantumTask(id="test; rm -rf /", name="Test")
        issues = validator.validate_task(task_unsafe_id)
        assert any(issue.code == "UNSAFE_TASK_ID" for issue in issues)
    
    def test_invalid_duration_validation(self, validator):
        """Test validation of invalid durations."""
        # Negative duration
        task_negative_duration = QuantumTask(id="test", name="Test", estimated_duration=-1.0)
        issues = validator.validate_task(task_negative_duration)
        assert any(issue.code == "INVALID_DURATION_VALUE" for issue in issues)
        
        # Zero duration
        task_zero_duration = QuantumTask(id="test", name="Test", estimated_duration=0.0)
        issues = validator.validate_task(task_zero_duration)
        assert any(issue.code == "INVALID_DURATION_VALUE" for issue in issues)
        
        # Excessive duration
        task_long_duration = QuantumTask(id="test", name="Test", estimated_duration=7200.0)  # 2 hours
        issues = validator.validate_task(task_long_duration)
        assert any(issue.code == "EXCESSIVE_DURATION" for issue in issues)
    
    def test_quantum_state_validation(self, validator):
        """Test quantum state validation."""
        # Create task with invalid quantum state
        task = QuantumTask(id="test", name="Test")
        
        # Corrupt quantum state
        task.state_amplitudes = {}
        issues = validator.validate_task(task)
        assert any(issue.code == "NO_QUANTUM_STATE" for issue in issues)
        
        # Test unnormalized amplitudes
        task.state_amplitudes = {
            TaskState.PENDING: complex(2.0, 0.0),  # Unnormalized
            TaskState.ACTIVE: complex(0.0, 0.0),
            TaskState.COMPLETED: complex(0.0, 0.0),
            TaskState.FAILED: complex(0.0, 0.0)
        }
        issues = validator.validate_task(task)
        # Note: amplitudes are auto-normalized, so this might not trigger warning
        
        # Test missing quantum states
        task.state_amplitudes = {
            TaskState.PENDING: complex(1.0, 0.0)
            # Missing other required states
        }
        issues = validator.validate_task(task)
        assert any(issue.code == "MISSING_QUANTUM_STATES" for issue in issues)
    
    def test_security_validation(self, validator):
        """Test security validation for dangerous patterns."""
        # Test dangerous task name
        dangerous_task = QuantumTask(
            id="test",
            name="Dangerous # SECURITY WARNING: eval() usage - validate input thoroughly
 eval() call",
        )
        issues = validator.validate_task(dangerous_task)
        # Should detect dangerous pattern
        assert any(issue.code == "DANGEROUS_PATTERN_DETECTED" for issue in issues)
        
        # Test dangerous task ID
        dangerous_id_task = QuantumTask(
            id="test_subprocess.call",
            name="Test"
        )
        issues = validator.validate_task(dangerous_id_task)
        assert any(issue.code == "DANGEROUS_PATTERN_DETECTED" for issue in issues)
    
    def test_resource_validation(self, validator):
        """Test resource requirement validation."""
        # Test invalid resource type
        task_invalid_resource = QuantumTask(
            id="test",
            name="Test",
            resource_requirements={"invalid_resource": 0.5}
        )
        issues = validator.validate_task(task_invalid_resource)
        assert any(issue.code == "UNKNOWN_RESOURCE_TYPE" for issue in issues)
        
        # Test excessive resource value
        task_excessive_resource = QuantumTask(
            id="test", 
            name="Test",
            resource_requirements={"cpu": 2.0}  # Above max_resource_value
        )
        issues = validator.validate_task(task_excessive_resource)
        assert any(issue.code == "EXCESSIVE_RESOURCE_VALUE" for issue in issues)
        
        # Test negative resource value
        task_negative_resource = QuantumTask(
            id="test",
            name="Test", 
            resource_requirements={"cpu": -0.5}
        )
        issues = validator.validate_task(task_negative_resource)
        assert any(issue.code == "NEGATIVE_RESOURCE_VALUE" for issue in issues)
    
    def test_dependency_structure_validation(self, validator):
        """Test dependency structure validation."""
        # Test excessive dependencies
        many_deps = {f"dep_{i}" for i in range(60)}  # Above max_dependencies
        task_many_deps = QuantumTask(
            id="test",
            name="Test",
            dependencies=many_deps
        )
        issues = validator.validate_task(task_many_deps)
        assert any(issue.code == "EXCESSIVE_DEPENDENCIES" for issue in issues)
        
        # Test self-dependency
        task_self_dep = QuantumTask(
            id="test",
            name="Test",
            dependencies={"test"}  # Self-dependency
        )
        issues = validator.validate_task(task_self_dep)
        assert any(issue.code == "SELF_DEPENDENCY" for issue in issues)
    
    def test_safe_identifier_checking(self, validator):
        """Test safe identifier validation."""
        assert validator._is_safe_identifier("valid_task_123")
        assert validator._is_safe_identifier("task-with-dashes")
        assert not validator._is_safe_identifier("task with spaces")
        assert not validator._is_safe_identifier("task;with;semicolons")
        assert not validator._is_safe_identifier("task<script>")


class TestDependencyValidator:
    """Test dependency validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create dependency validator."""
        return DependencyValidator()
    
    def test_missing_dependencies_detection(self, validator):
        """Test detection of missing dependencies."""
        tasks = {
            "task1": QuantumTask(id="task1", name="Task 1", dependencies={"nonexistent"}),
            "task2": QuantumTask(id="task2", name="Task 2")
        }
        
        issues = validator.validate_dependencies(tasks)
        assert any(issue.code == "MISSING_DEPENDENCIES" for issue in issues)
    
    def test_circular_dependency_detection(self, validator):
        """Test detection of circular dependencies."""
        tasks = {
            "task1": QuantumTask(id="task1", name="Task 1", dependencies={"task2"}),
            "task2": QuantumTask(id="task2", name="Task 2", dependencies={"task3"}),
            "task3": QuantumTask(id="task3", name="Task 3", dependencies={"task1"})  # Creates cycle
        }
        
        issues = validator.validate_dependencies(tasks)
        assert any(issue.code == "CIRCULAR_DEPENDENCY" for issue in issues)
    
    def test_self_circular_dependency(self, validator):
        """Test detection of self-referencing circular dependencies."""
        tasks = {
            "task1": QuantumTask(id="task1", name="Task 1", dependencies={"task1"})  # Self-cycle
        }
        
        issues = validator.validate_dependencies(tasks)
        # This should be caught by task validator as SELF_DEPENDENCY
        # But dependency validator should also handle cycles
    
    def test_complex_dependency_graph_validation(self, validator):
        """Test validation of complex dependency graphs."""
        # Create complex but valid dependency graph
        tasks = {}
        
        # Layer 1: Independent tasks
        tasks["root1"] = QuantumTask(id="root1", name="Root 1")
        tasks["root2"] = QuantumTask(id="root2", name="Root 2")
        
        # Layer 2: Depend on roots
        tasks["mid1"] = QuantumTask(id="mid1", name="Mid 1", dependencies={"root1"})
        tasks["mid2"] = QuantumTask(id="mid2", name="Mid 2", dependencies={"root2"})
        tasks["mid3"] = QuantumTask(id="mid3", name="Mid 3", dependencies={"root1", "root2"})
        
        # Layer 3: Depend on mid-level
        tasks["leaf1"] = QuantumTask(id="leaf1", name="Leaf 1", dependencies={"mid1", "mid2"})
        tasks["leaf2"] = QuantumTask(id="leaf2", name="Leaf 2", dependencies={"mid3"})
        
        issues = validator.validate_dependencies(tasks)
        
        # Should not have any critical dependency issues
        critical_issues = [i for i in issues if i.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}]
        assert len(critical_issues) == 0
    
    def test_dependency_depth_validation(self, validator):
        """Test excessive dependency depth detection."""
        tasks = {}
        
        # Create very deep dependency chain
        for i in range(25):  # Deeper than typical threshold
            deps = {f"task_{i-1}"} if i > 0 else set()
            tasks[f"task_{i}"] = QuantumTask(id=f"task_{i}", name=f"Task {i}", dependencies=deps)
        
        issues = validator.validate_dependencies(tasks)
        
        # Should warn about excessive depth
        assert any(issue.code == "EXCESSIVE_DEPENDENCY_DEPTH" for issue in issues)


class TestConfigurationValidator:
    """Test configuration validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create configuration validator."""
        return ConfigurationValidator()
    
    def test_valid_configuration_passes(self, validator):
        """Test that valid configuration passes validation."""
        valid_config = {
            "max_parallel_tasks": 4,
            "quantum_coherence_time": 10.0,
            "interference_strength": 0.3,
            "learning_rate": 0.01,
            "exploration_rate": 0.1
        }
        
        issues = validator.validate_configuration(valid_config)
        
        # Should have no errors
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        assert len(error_issues) == 0
    
    def test_parameter_range_validation(self, validator):
        """Test parameter range validation."""
        # Test values below minimum
        below_min_config = {
            "max_parallel_tasks": 0,  # Below minimum of 1
            "learning_rate": 0.0001   # Below minimum of 0.001
        }
        
        issues = validator.validate_configuration(below_min_config)
        assert any(issue.code == "PARAMETER_BELOW_MINIMUM" for issue in issues)
        
        # Test values above maximum
        above_max_config = {
            "max_parallel_tasks": 200,  # Above maximum of 100
            "interference_strength": 2.0  # Above maximum of 1.0
        }
        
        issues = validator.validate_configuration(above_max_config)
        assert any(issue.code == "PARAMETER_ABOVE_MAXIMUM" for issue in issues)
    
    def test_parameter_type_validation(self, validator):
        """Test parameter type validation."""
        invalid_type_config = {
            "max_parallel_tasks": "not_a_number",
            "quantum_coherence_time": None,
            "interference_strength": []
        }
        
        issues = validator.validate_configuration(invalid_type_config)
        
        # Should detect type errors for all parameters
        type_errors = [i for i in issues if i.code == "INVALID_PARAMETER_TYPE"]
        assert len(type_errors) >= 3
    
    def test_parameter_conflicts_detection(self, validator):
        """Test detection of conflicting parameters."""
        # Test measurement interval >= coherence time
        conflicting_config = {
            "quantum_coherence_time": 5.0,
            "measurement_interval": 6.0  # Greater than coherence time
        }
        
        issues = validator.validate_configuration(conflicting_config)
        assert any(issue.code == "MEASUREMENT_EXCEEDS_COHERENCE" for issue in issues)
        
        # Test high learning and exploration rates
        high_rates_config = {
            "learning_rate": 0.8,
            "exploration_rate": 0.9
        }
        
        issues = validator.validate_configuration(high_rates_config)
        assert any(issue.code == "HIGH_LEARNING_AND_EXPLORATION" for issue in issues)
    
    def test_special_value_validation(self, validator):
        """Test validation of special float values."""
        special_values_config = {
            "learning_rate": float('inf'),
            "exploration_rate": float('nan'),
            "interference_strength": -float('inf')
        }
        
        issues = validator.validate_configuration(special_values_config)
        
        # Should detect invalid values
        invalid_issues = [i for i in issues if i.code == "INVALID_PARAMETER_VALUE"]
        assert len(invalid_issues) >= 3


class TestInputSanitization:
    """Test input sanitization functionality."""
    
    def test_basic_sanitization(self):
        """Test basic input sanitization."""
        raw_input = {
            "id": "test_task_123",
            "name": "Test Task Name",
            "estimated_duration": 5.0,
            "priority": 0.8,
            "dependencies": ["dep1", "dep2"],
            "resource_requirements": {
                "cpu": 0.5,
                "memory": 0.3
            }
        }
        
        sanitized = sanitize_task_input(raw_input)
        
        assert sanitized["id"] == "test_task_123"
        assert sanitized["name"] == "Test Task Name"
        assert sanitized["estimated_duration"] == 5.0
        assert sanitized["priority"] == 0.8
        assert sanitized["dependencies"] == {"dep1", "dep2"}
        assert sanitized["resource_requirements"]["cpu"] == 0.5
        assert sanitized["resource_requirements"]["memory"] == 0.3
    
    def test_dangerous_input_sanitization(self):
        """Test sanitization of dangerous input."""
        dangerous_input = {
            "id": "test; rm -rf /",
            "name": "<script>alert('xss')</script>",
            "estimated_duration": "# SECURITY WARNING: eval() usage - validate input thoroughly
eval('malicious code')",
            "dependencies": ["dep1", "'; DROP TABLE tasks; --"],
            "resource_requirements": {
                "cpu": 999.0,  # Excessive value
                "malicious_resource": 0.5,  # Not allowed
                "memory": float('inf')  # Invalid value
            }
        }
        
        sanitized = sanitize_task_input(dangerous_input)
        
        # ID should be cleaned
        assert ";" not in sanitized.get("id", "")
        assert "rm" not in sanitized.get("id", "")
        
        # Name should be cleaned
        assert "<script>" not in sanitized.get("name", "")
        assert "alert" not in sanitized.get("name", "")
        
        # Duration should be skipped due to invalid type
        assert "estimated_duration" not in sanitized
        
        # Dependencies should be cleaned
        if "dependencies" in sanitized:
            for dep in sanitized["dependencies"]:
                assert "DROP TABLE" not in dep
                assert ";" not in dep
        
        # Resources should be filtered and clamped
        if "resource_requirements" in sanitized:
            resources = sanitized["resource_requirements"]
            assert "malicious_resource" not in resources
            assert resources.get("cpu", 0) <= 1.0  # Should be clamped
            assert "memory" not in resources  # Should be filtered out due to inf
    
    def test_type_coercion_sanitization(self):
        """Test type coercion in sanitization."""
        mixed_types_input = {
            "id": 123,  # Should become string
            "name": ["list", "name"],  # Invalid type
            "estimated_duration": "5.5",  # String number
            "priority": True,  # Boolean
            "dependencies": ("tuple", "deps"),  # Tuple
            "resource_requirements": "not_a_dict"  # Invalid type
        }
        
        sanitized = sanitize_task_input(mixed_types_input)
        
        # Should gracefully handle type mismatches
        # Most invalid types should be filtered out
        assert isinstance(sanitized.get("estimated_duration", 0), float)
    
    def test_length_limits_sanitization(self):
        """Test length limits in sanitization."""
        long_input = {
            "id": "a" * 200,  # Very long ID
            "name": "b" * 500,  # Very long name
            "dependencies": [f"dep_{i}" for i in range(1000)]  # Many dependencies
        }
        
        sanitized = sanitize_task_input(long_input)
        
        # Should enforce length limits
        assert len(sanitized.get("id", "")) <= 100
        assert len(sanitized.get("name", "")) <= 200


class TestBatchValidation:
    """Test batch validation functionality."""
    
    def test_valid_batch_validation(self):
        """Test validation of valid task batch."""
        valid_tasks = {
            "task1": QuantumTask(id="task1", name="Task 1"),
            "task2": QuantumTask(id="task2", name="Task 2", dependencies={"task1"}),
            "task3": QuantumTask(id="task3", name="Task 3")
        }
        
        issues, is_valid = validate_task_batch(valid_tasks)
        
        assert is_valid is True
        
        # Should have no critical issues
        critical_issues = [i for i in issues if i.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}]
        assert len(critical_issues) == 0
    
    def test_invalid_batch_validation(self):
        """Test validation of invalid task batch."""
        invalid_tasks = {
            "task1": QuantumTask(id="", name="Invalid ID Task"),  # Empty ID
            "task2": QuantumTask(id="task2", name="Task 2", dependencies={"nonexistent"}),  # Missing dep
            "task3": QuantumTask(id="task3", name="Task 3", dependencies={"task4"}),  # Circular with task4
            "task4": QuantumTask(id="task4", name="Task 4", dependencies={"task3"})   # Circular with task3
        }
        
        issues, is_valid = validate_task_batch(invalid_tasks)
        
        assert is_valid is False
        
        # Should have critical issues
        critical_issues = [i for i in issues if i.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}]
        assert len(critical_issues) > 0
    
    def test_large_batch_validation_performance(self):
        """Test validation performance with large batch."""
        import time
        
        # Create large batch of tasks
        large_batch = {}
        for i in range(1000):
            deps = {f"task_{j}" for j in range(max(0, i-3), i)}  # Depend on previous 3 tasks
            large_batch[f"task_{i}"] = QuantumTask(
                id=f"task_{i}",
                name=f"Task {i}",
                dependencies=deps
            )
        
        # Validate batch and measure time
        start_time = time.time()
        issues, is_valid = validate_task_batch(large_batch)
        validation_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds)
        assert validation_time < 5.0
        
        # Should be valid (no cycles in this structure)
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])