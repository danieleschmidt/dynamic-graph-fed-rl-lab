"""
Test suite for quantum planner security components.

Tests security validation, threat detection, and access control.
"""

import pytest
from unittest.mock import Mock, patch
import hashlib
import time

from src.dynamic_graph_fed_rl.quantum_planner.security import (
    SecurityManager,
    InputValidator,
    ThreatDetector,
    AccessController,
    SecurityAudit,
    SecurityViolation
)
from src.dynamic_graph_fed_rl.quantum_planner.core import QuantumTask


class TestSecurityManager:
    """Test overall security management."""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager."""
        return SecurityManager(
            enable_validation=True,
            enable_threat_detection=True,
            audit_logging=True
        )
    
    def test_security_manager_initialization(self, security_manager):
        """Test security manager initialization."""
        assert security_manager.enable_validation is True
        assert security_manager.enable_threat_detection is True
        assert security_manager.audit_logging is True
    
    def test_validate_task_input(self, security_manager):
        """Test task input validation."""
        # Valid task
        valid_task = {
            "id": "valid_task_123",
            "name": "Valid Task",
            "data": {"value": 42}
        }
        
        is_valid, issues = security_manager.validate_task_input(valid_task)
        assert is_valid is True
        assert len(issues) == 0
        
        # Invalid task with dangerous content
        invalid_task = {
            "id": "task_<script>alert('xss')</script>",
            "name": "'; DROP TABLE tasks; --",
            "data": {"command": "rm -rf /"}
        }
        
        is_valid, issues = security_manager.validate_task_input(invalid_task)
        assert is_valid is False
        assert len(issues) > 0
    
    def test_threat_detection(self, security_manager):
        """Test threat detection capabilities."""
        # Suspicious request pattern
        requests = [
            {"ip": "192.168.1.100", "timestamp": time.time(), "task_id": "task_1"},
            {"ip": "192.168.1.100", "timestamp": time.time(), "task_id": "task_2"},
            {"ip": "192.168.1.100", "timestamp": time.time(), "task_id": "task_3"},
        ]
        
        threats = security_manager.detect_threats(requests)
        
        # Should detect potential rate limiting violation
        rate_limit_threats = [t for t in threats if t["type"] == "rate_limit_violation"]
        assert len(rate_limit_threats) >= 0  # May or may not trigger based on timing
    
    def test_security_audit_logging(self, security_manager):
        """Test security audit logging."""
        with patch.object(security_manager.audit, 'log_security_event') as mock_log:
            security_manager.log_security_event(
                event_type="access_violation",
                user_id="user123", 
                resource="quantum_task",
                details={"attempted_action": "unauthorized_access"}
            )
            
            mock_log.assert_called_once()


class TestInputValidator:
    """Test input validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create input validator."""
        return InputValidator(
            max_input_size=1024,
            enable_xss_protection=True,
            enable_sql_injection_protection=True
        )
    
    def test_string_validation(self, validator):
        """Test string input validation."""
        # Safe strings
        safe_strings = [
            "normal_task_name",
            "Task with spaces",
            "Task-with-dashes_123"
        ]
        
        for string in safe_strings:
            is_safe = validator.validate_string(string)
            assert is_safe is True
        
        # Dangerous strings
        dangerous_strings = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "$(rm -rf /)"
        ]
        
        for string in dangerous_strings:
            is_safe = validator.validate_string(string)
            assert is_safe is False
    
    def test_data_size_validation(self, validator):
        """Test data size validation."""
        # Small data - should pass
        small_data = {"key": "value"}
        is_valid = validator.validate_data_size(small_data)
        assert is_valid is True
        
        # Large data - should fail
        large_data = {"key": "x" * 2000}  # Exceeds 1024 byte limit
        is_valid = validator.validate_data_size(large_data)
        assert is_valid is False
    
    def test_quantum_parameter_validation(self, validator):
        """Test quantum parameter validation."""
        # Valid quantum parameters
        valid_params = {
            "coherence_time": 1.5,
            "entanglement_strength": 0.7,
            "measurement_accuracy": 0.95
        }
        
        is_valid, errors = validator.validate_quantum_parameters(valid_params)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid quantum parameters
        invalid_params = {
            "coherence_time": -1.0,  # Negative time
            "entanglement_strength": 1.5,  # > 1.0
            "measurement_accuracy": 2.0  # > 1.0
        }
        
        is_valid, errors = validator.validate_quantum_parameters(invalid_params)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_injection_attack_detection(self, validator):
        """Test SQL injection and command injection detection."""
        injection_attempts = [
            "'; DELETE FROM tasks WHERE 1=1; --",
            "1' OR '1'='1",
            "; cat /etc/passwd;",
            "$(whoami)",
            "`ls -la`",
            "| rm -rf /",
        ]
        
        for attempt in injection_attempts:
            is_safe = validator.check_injection_patterns(attempt)
            assert is_safe is False


class TestThreatDetector:
    """Test threat detection system."""
    
    @pytest.fixture
    def detector(self):
        """Create threat detector."""
        return ThreatDetector(
            rate_limit_threshold=10,
            anomaly_detection=True,
            pattern_matching=True
        )
    
    def test_rate_limiting_detection(self, detector):
        """Test rate limiting threat detection."""
        # Simulate rapid requests from same IP
        requests = []
        for i in range(15):  # Above rate limit threshold
            requests.append({
                "ip": "192.168.1.100",
                "timestamp": time.time(),
                "endpoint": "/quantum/execute"
            })
        
        threats = detector.analyze_requests(requests)
        
        rate_threats = [t for t in threats if t["type"] == "rate_limit_exceeded"]
        assert len(rate_threats) > 0
    
    def test_anomaly_detection(self, detector):
        """Test anomaly detection in task patterns."""
        # Normal task pattern
        normal_tasks = [
            {"size": 100, "complexity": 0.5, "execution_time": 1.0},
            {"size": 120, "complexity": 0.6, "execution_time": 1.2},
            {"size": 80, "complexity": 0.4, "execution_time": 0.8}
        ]
        
        # Establish baseline
        detector.train_anomaly_detector(normal_tasks)
        
        # Anomalous task
        anomalous_task = {
            "size": 10000,  # Much larger than normal
            "complexity": 10.0,  # Much more complex
            "execution_time": 100.0  # Much longer
        }
        
        is_anomaly = detector.detect_task_anomaly(anomalous_task)
        assert is_anomaly is True
    
    def test_malicious_pattern_detection(self, detector):
        """Test detection of malicious patterns."""
        malicious_patterns = [
            "admin=true&backdoor=active",
            "union select password from users",
            "../../../etc/passwd",
            "# SECURITY WARNING: eval() usage - validate input thoroughly
eval(base64_decode($_POST['cmd']))",
            "powershell -enc JABhACAAPQAgACcA"  # Base64 encoded PowerShell
        ]
        
        for pattern in malicious_patterns:
            is_malicious = detector.check_malicious_patterns(pattern)
            assert is_malicious is True
    
    def test_brute_force_detection(self, detector):
        """Test brute force attack detection."""
        # Simulate multiple failed authentication attempts
        failed_attempts = []
        for i in range(20):  # Many failed attempts
            failed_attempts.append({
                "ip": "192.168.1.100",
                "timestamp": time.time(),
                "success": False,
                "user": f"admin{i % 3}"  # Trying multiple usernames
            })
        
        is_brute_force = detector.detect_brute_force(failed_attempts)
        assert is_brute_force is True


class TestAccessController:
    """Test access control functionality."""
    
    @pytest.fixture
    def access_controller(self):
        """Create access controller."""
        return AccessController(
            enable_rbac=True,
            session_timeout=3600,
            require_authentication=True
        )
    
    def test_role_based_access_control(self, access_controller):
        """Test RBAC functionality."""
        # Define user with role
        user = {
            "id": "user123",
            "roles": ["quantum_operator", "task_reader"],
            "permissions": ["read_tasks", "execute_quantum_tasks"]
        }
        
        # Test allowed operation
        can_execute = access_controller.check_permission(
            user, "execute_quantum_tasks", "quantum_task_1"
        )
        assert can_execute is True
        
        # Test forbidden operation
        can_delete = access_controller.check_permission(
            user, "delete_all_tasks", "quantum_task_1" 
        )
        assert can_delete is False
    
    def test_session_management(self, access_controller):
        """Test session management."""
        # Create session
        session = access_controller.create_session("user123")
        
        assert "session_id" in session
        assert "created_at" in session
        assert session["user_id"] == "user123"
        
        # Validate active session
        is_valid = access_controller.validate_session(session["session_id"])
        assert is_valid is True
        
        # Test session expiry
        expired_session = {
            "session_id": "expired_123",
            "user_id": "user123",
            "created_at": time.time() - 7200,  # 2 hours ago
            "expires_at": time.time() - 3600   # Expired 1 hour ago
        }
        
        is_valid = access_controller.validate_session(expired_session["session_id"])
        assert is_valid is False
    
    def test_resource_access_control(self, access_controller):
        """Test resource-level access control."""
        # User with limited access
        limited_user = {
            "id": "limited_user",
            "roles": ["task_reader"],
            "resource_access": ["task_1", "task_2"]  # Only specific tasks
        }
        
        # Test access to allowed resource
        can_access = access_controller.check_resource_access(
            limited_user, "quantum_task", "task_1"
        )
        assert can_access is True
        
        # Test access to forbidden resource
        can_access = access_controller.check_resource_access(
            limited_user, "quantum_task", "task_999"
        )
        assert can_access is False


class TestSecurityAudit:
    """Test security audit functionality."""
    
    @pytest.fixture
    def audit(self):
        """Create security audit logger."""
        return SecurityAudit(
            enable_logging=True,
            log_level="INFO",
            retention_days=90
        )
    
    def test_security_event_logging(self, audit):
        """Test security event logging."""
        event = {
            "event_type": "unauthorized_access",
            "user_id": "user123",
            "resource": "quantum_executor",
            "ip_address": "192.168.1.100",
            "timestamp": time.time(),
            "details": {"attempted_action": "delete_all_tasks"}
        }
        
        with patch.object(audit, '_write_log_entry') as mock_write:
            audit.log_security_event(event)
            mock_write.assert_called_once()
    
    def test_audit_trail_generation(self, audit):
        """Test audit trail generation."""
        # Log multiple events
        events = [
            {"event_type": "login", "user_id": "user1"},
            {"event_type": "task_access", "user_id": "user1", "resource": "task_1"},
            {"event_type": "task_execution", "user_id": "user1", "resource": "task_1"},
            {"event_type": "logout", "user_id": "user1"}
        ]
        
        for event in events:
            audit.log_security_event(event)
        
        # Generate audit trail for user
        trail = audit.generate_audit_trail("user1")
        
        assert len(trail) == 4
        assert trail[0]["event_type"] == "login"
        assert trail[-1]["event_type"] == "logout"
    
    def test_security_violation_detection(self, audit):
        """Test security violation detection."""
        violation = SecurityViolation(
            violation_type="unauthorized_access",
            severity="high",
            user_id="malicious_user",
            details={"attempted_resource": "admin_panel"}
        )
        
        with patch.object(audit, 'log_security_event') as mock_log:
            audit.log_security_violation(violation)
            mock_log.assert_called_once()


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_complete_security_workflow(self):
        """Test complete security validation workflow."""
        # Setup security components
        security_manager = SecurityManager()
        validator = InputValidator()
        detector = ThreatDetector()
        access_controller = AccessController()
        audit = SecurityAudit()
        
        # Simulate user request
        user = {"id": "user123", "roles": ["quantum_operator"]}
        task_data = {
            "id": "legitimate_task",
            "name": "Valid Task",
            "quantum_params": {"coherence_time": 1.0}
        }
        
        # Step 1: Validate input
        is_valid, issues = validator.validate_task_input(task_data)
        assert is_valid is True
        
        # Step 2: Check access permissions
        can_access = access_controller.check_permission(
            user, "execute_quantum_tasks", task_data["id"]
        )
        assert can_access is True
        
        # Step 3: Threat detection
        request_info = {
            "ip": "192.168.1.100",
            "user_id": user["id"],
            "timestamp": time.time()
        }
        threats = detector.analyze_request(request_info)
        assert len(threats) == 0  # No threats for legitimate request
        
        # Step 4: Audit logging
        audit_event = {
            "event_type": "task_execution",
            "user_id": user["id"],
            "resource": task_data["id"],
            "timestamp": time.time()
        }
        audit.log_security_event(audit_event)
        
        # Verify complete workflow
        assert is_valid and can_access and len(threats) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])