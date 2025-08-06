"""
Security module for quantum task planner.

Implements comprehensive security measures:
- Input validation and sanitization
- Access control and authentication
- Resource usage limits and quotas
- Audit logging for security events
- Protection against common attacks
"""

import hashlib
import secrets
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any, Callable, Union
from enum import Enum
from collections import defaultdict, deque
import re

from .exceptions import (
    QuantumPlannerError,
    TaskValidationError,
    ResourceAllocationError,
    ConfigurationError
)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class Permission(Enum):
    """System permissions."""
    READ_TASKS = "read_tasks"
    CREATE_TASKS = "create_tasks"
    MODIFY_TASKS = "modify_tasks"
    DELETE_TASKS = "delete_tasks"
    EXECUTE_TASKS = "execute_tasks"
    VIEW_METRICS = "view_metrics"
    MODIFY_CONFIG = "modify_config"
    ADMIN_ACCESS = "admin_access"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    permissions: Set[Permission]
    security_level: SecurityLevel
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_type: str
    severity: str  # "info", "warning", "critical"
    user_id: str
    resource: str
    action: str
    success: bool
    timestamp: float
    details: Dict[str, Any]
    client_ip: Optional[str] = None


class RateLimiter:
    """
    Rate limiter to prevent abuse and DoS attacks.
    
    Implements sliding window rate limiting per user/IP.
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
        cleanup_interval: float = 300.0  # 5 minutes
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval
        
        # Request tracking
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_until: Dict[str, float] = {}
        
        self.last_cleanup = time.time()
        
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier."""
        current_time = time.time()
        
        # Clean up old data periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()
        
        # Check if currently blocked
        if identifier in self.blocked_until:
            if current_time < self.blocked_until[identifier]:
                return False
            else:
                del self.blocked_until[identifier]
        
        # Get request history for this identifier
        requests = self.request_counts[identifier]
        
        # Remove requests outside the window
        cutoff_time = current_time - self.window_seconds
        while requests and requests[0] < cutoff_time:
            requests.popleft()
        
        # Check if under limit
        if len(requests) >= self.max_requests:
            # Rate limit exceeded - block for window duration
            self.blocked_until[identifier] = current_time + self.window_seconds
            self.logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Record this request
        requests.append(current_time)
        return True
    
    def _cleanup_old_requests(self):
        """Clean up old request data to prevent memory leaks."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds * 2  # Keep extra buffer
        
        # Clean request counts
        for identifier in list(self.request_counts.keys()):
            requests = self.request_counts[identifier]
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Remove empty deques
            if not requests:
                del self.request_counts[identifier]
        
        # Clean expired blocks
        for identifier in list(self.blocked_until.keys()):
            if current_time >= self.blocked_until[identifier]:
                del self.blocked_until[identifier]
        
        self.last_cleanup = current_time
    
    def get_status(self, identifier: str) -> Dict[str, Any]:
        """Get rate limiting status for identifier."""
        current_time = time.time()
        requests = self.request_counts.get(identifier, deque())
        
        # Count recent requests
        cutoff_time = current_time - self.window_seconds
        recent_count = sum(1 for req_time in requests if req_time > cutoff_time)
        
        is_blocked = identifier in self.blocked_until and current_time < self.blocked_until[identifier]
        
        return {
            "identifier": identifier,
            "recent_requests": recent_count,
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "is_blocked": is_blocked,
            "blocked_until": self.blocked_until.get(identifier, 0.0),
            "remaining_capacity": max(0, self.max_requests - recent_count)
        }


class InputSanitizer:
    """
    Sanitizes user inputs to prevent injection attacks.
    
    Provides comprehensive input validation and sanitization.
    """
    
    def __init__(self):
        # Dangerous patterns that should be blocked
        self.dangerous_patterns = [
            # Code injection
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'shell=True',
            
            # Command injection
            r'[;&|`$]',
            r'\$\(',
            r'`[^`]*`',
            
            # SQL injection
            r"'[^']*';\s*(DROP|DELETE|INSERT|UPDATE)",
            r'UNION\s+SELECT',
            r'OR\s+1\s*=\s*1',
            
            # Path traversal
            r'\.\./+',
            r'/etc/passwd',
            r'/etc/shadow',
            
            # XSS patterns
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
        ]
        
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.dangerous_patterns
        ]
        
        # Safe character sets
        self.safe_identifier = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')
        self.safe_name = re.compile(r'^[a-zA-Z0-9\s\-_.]{1,200}$')
        
        self.logger = logging.getLogger(__name__)
    
    def sanitize_identifier(self, value: str) -> str:
        """Sanitize identifier (task IDs, resource names, etc.)."""
        if not isinstance(value, str):
            raise ValueError("Identifier must be a string")
        
        # Remove dangerous characters
        sanitized = re.sub(r'[^\w\-_]', '', value)
        
        # Limit length
        sanitized = sanitized[:100]
        
        # Ensure it's not empty
        if not sanitized:
            raise ValueError("Identifier cannot be empty after sanitization")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(sanitized):
                raise ValueError(f"Identifier contains dangerous pattern: {sanitized}")
        
        return sanitized
    
    def sanitize_name(self, value: str) -> str:
        """Sanitize human-readable names."""
        if not isinstance(value, str):
            raise ValueError("Name must be a string")
        
        # Remove most dangerous characters but allow spaces
        sanitized = re.sub(r'[<>"\';\\]', '', value)
        
        # Limit length
        sanitized = sanitized[:200].strip()
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(sanitized):
                raise ValueError(f"Name contains dangerous pattern")
        
        return sanitized
    
    def validate_resource_value(self, value: Union[int, float]) -> float:
        """Validate and sanitize resource values."""
        if not isinstance(value, (int, float)):
            raise ValueError("Resource value must be a number")
        
        # Check for NaN/infinite
        if not (float('-inf') < value < float('inf')):
            raise ValueError("Resource value must be finite")
        
        # Clamp to reasonable range
        sanitized = max(0.0, min(100.0, float(value)))
        
        return sanitized
    
    def validate_duration(self, value: Union[int, float]) -> float:
        """Validate and sanitize duration values."""
        if not isinstance(value, (int, float)):
            raise ValueError("Duration must be a number")
        
        if not (0 < value < 86400):  # Max 24 hours
            raise ValueError("Duration must be between 0 and 86400 seconds")
        
        return float(value)
    
    def check_dangerous_content(self, content: str) -> bool:
        """Check if content contains dangerous patterns."""
        if not isinstance(content, str):
            return False
        
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                self.logger.warning(f"Dangerous pattern detected: {pattern.pattern}")
                return True
        
        return False


class AccessController:
    """
    Manages access control and permissions.
    
    Implements role-based access control (RBAC) with security levels.
    """
    
    def __init__(self):
        # Role definitions
        self.roles: Dict[str, Set[Permission]] = {
            "viewer": {Permission.READ_TASKS, Permission.VIEW_METRICS},
            "operator": {
                Permission.READ_TASKS, Permission.CREATE_TASKS, 
                Permission.EXECUTE_TASKS, Permission.VIEW_METRICS
            },
            "manager": {
                Permission.READ_TASKS, Permission.CREATE_TASKS, 
                Permission.MODIFY_TASKS, Permission.EXECUTE_TASKS, 
                Permission.VIEW_METRICS, Permission.MODIFY_CONFIG
            },
            "admin": set(Permission)  # All permissions
        }
        
        # User role assignments
        self.user_roles: Dict[str, Set[str]] = defaultdict(set)
        
        # Security level requirements for operations
        self.operation_security_levels: Dict[str, SecurityLevel] = {
            "read_tasks": SecurityLevel.PUBLIC,
            "create_tasks": SecurityLevel.INTERNAL,
            "modify_tasks": SecurityLevel.INTERNAL,
            "delete_tasks": SecurityLevel.CONFIDENTIAL,
            "execute_tasks": SecurityLevel.INTERNAL,
            "view_metrics": SecurityLevel.INTERNAL,
            "modify_config": SecurityLevel.CONFIDENTIAL,
            "admin_access": SecurityLevel.SECRET
        }
        
        self.logger = logging.getLogger(__name__)
    
    def assign_role(self, user_id: str, role: str):
        """Assign role to user."""
        if role not in self.roles:
            raise ValueError(f"Unknown role: {role}")
        
        self.user_roles[user_id].add(role)
        self.logger.info(f"Assigned role '{role}' to user '{user_id}'")
    
    def revoke_role(self, user_id: str, role: str):
        """Revoke role from user."""
        self.user_roles[user_id].discard(role)
        self.logger.info(f"Revoked role '{role}' from user '{user_id}'")
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        permissions = set()
        
        for role in self.user_roles[user_id]:
            if role in self.roles:
                permissions.update(self.roles[role])
        
        return permissions
    
    def check_permission(
        self, 
        context: SecurityContext, 
        permission: Permission,
        resource: Optional[str] = None
    ) -> bool:
        """Check if user has specific permission."""
        # Get user permissions
        user_permissions = self.get_user_permissions(context.user_id)
        
        # Check permission
        has_permission = permission in user_permissions
        
        # Check security level requirement
        operation_name = permission.value
        required_level = self.operation_security_levels.get(operation_name, SecurityLevel.INTERNAL)
        has_clearance = self._check_security_level(context.security_level, required_level)
        
        result = has_permission and has_clearance
        
        if not result:
            self.logger.warning(
                f"Access denied for user {context.user_id}: "
                f"permission={permission.value}, resource={resource}, "
                f"has_permission={has_permission}, has_clearance={has_clearance}"
            )
        
        return result
    
    def _check_security_level(
        self, 
        user_level: SecurityLevel, 
        required_level: SecurityLevel
    ) -> bool:
        """Check if user security level meets requirement."""
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        user_clearance = level_hierarchy.get(user_level, 0)
        required_clearance = level_hierarchy.get(required_level, 0)
        
        return user_clearance >= required_clearance


class SecurityAuditor:
    """
    Audit logger for security events.
    
    Records and analyzes security-related events for monitoring and compliance.
    """
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        
        # Threat detection
        self.suspicious_activity: Dict[str, List[float]] = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def log_event(
        self,
        event_type: str,
        severity: str,
        context: SecurityContext,
        resource: str,
        action: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            user_id=context.user_id,
            resource=resource,
            action=action,
            success=success,
            timestamp=time.time(),
            details=details or {},
            client_ip=context.client_ip
        )
        
        self.events.append(event)
        
        # Log to standard logger
        log_level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(
            log_level,
            f"Security event: {event_type} - User: {context.user_id}, "
            f"Action: {action}, Resource: {resource}, Success: {success}"
        )
        
        # Analyze for suspicious activity
        self._analyze_suspicious_activity(event)
    
    def _analyze_suspicious_activity(self, event: SecurityEvent):
        """Analyze event for suspicious patterns."""
        current_time = time.time()
        
        # Track failed attempts
        if not event.success and event.event_type in {"authentication", "authorization"}:
            user_failures = self.suspicious_activity[event.user_id]
            user_failures.append(current_time)
            
            # Keep only recent failures (last hour)
            cutoff_time = current_time - 3600
            user_failures[:] = [t for t in user_failures if t > cutoff_time]
            
            # Alert on multiple failures
            if len(user_failures) >= 5:
                self.logger.critical(
                    f"Multiple security failures detected for user {event.user_id}: "
                    f"{len(user_failures)} failures in the last hour"
                )
    
    def get_security_summary(self, hours: float = 24.0) -> Dict[str, Any]:
        """Get security summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.events if e.timestamp > cutoff_time]
        
        # Event counts by type
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        user_activity = defaultdict(int)
        failed_attempts = 0
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
            user_activity[event.user_id] += 1
            
            if not event.success:
                failed_attempts += 1
        
        # Top active users
        top_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Suspicious activity summary
        suspicious_users = {
            user_id: len(failures)
            for user_id, failures in self.suspicious_activity.items()
            if failures and max(failures) > cutoff_time
        }
        
        return {
            "period_hours": hours,
            "total_events": len(recent_events),
            "event_types": dict(event_counts),
            "severity_levels": dict(severity_counts),
            "failed_attempts": failed_attempts,
            "success_rate": (len(recent_events) - failed_attempts) / max(len(recent_events), 1),
            "top_active_users": top_users,
            "suspicious_users": suspicious_users,
            "alerts_generated": len([e for e in recent_events if e.severity == "critical"])
        }


class SecurityManager:
    """
    Main security manager coordinating all security components.
    
    Provides unified interface for security operations.
    """
    
    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        input_sanitizer: Optional[InputSanitizer] = None,
        access_controller: Optional[AccessController] = None,
        auditor: Optional[SecurityAuditor] = None
    ):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.sanitizer = input_sanitizer or InputSanitizer()
        self.access_control = access_controller or AccessController()
        self.auditor = auditor or SecurityAuditor()
        
        # Session management
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.session_timeout = 3600.0  # 1 hour
        
        self.logger = logging.getLogger(__name__)
    
    def create_session(
        self,
        user_id: str,
        permissions: Set[Permission],
        security_level: SecurityLevel,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Create new security session."""
        session_id = secrets.token_urlsafe(32)
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            permissions=permissions,
            security_level=security_level,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        self.active_sessions[session_id] = context
        
        self.auditor.log_event(
            event_type="authentication",
            severity="info",
            context=context,
            resource="session",
            action="create",
            success=True,
            details={"security_level": security_level.value}
        )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SecurityContext]:
        """Get active session context."""
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        
        # Check session timeout
        if time.time() - context.timestamp > self.session_timeout:
            self.invalidate_session(session_id)
            return None
        
        return context
    
    def invalidate_session(self, session_id: str):
        """Invalidate session."""
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            
            self.auditor.log_event(
                event_type="authentication",
                severity="info",
                context=context,
                resource="session",
                action="invalidate",
                success=True
            )
    
    def authorize_operation(
        self,
        session_id: str,
        permission: Permission,
        resource: Optional[str] = None
    ) -> bool:
        """Authorize operation for session."""
        # Get session context
        context = self.get_session(session_id)
        if not context:
            self.auditor.log_event(
                event_type="authorization",
                severity="warning",
                context=SecurityContext(
                    user_id="unknown",
                    session_id=session_id,
                    permissions=set(),
                    security_level=SecurityLevel.PUBLIC
                ),
                resource=resource or "unknown",
                action=permission.value,
                success=False,
                details={"reason": "invalid_session"}
            )
            return False
        
        # Check rate limiting
        identifier = f"{context.user_id}:{context.client_ip or 'unknown'}"
        if not self.rate_limiter.is_allowed(identifier):
            self.auditor.log_event(
                event_type="rate_limiting",
                severity="warning",
                context=context,
                resource=resource or "unknown",
                action=permission.value,
                success=False,
                details={"reason": "rate_limit_exceeded"}
            )
            return False
        
        # Check permissions
        authorized = self.access_control.check_permission(context, permission, resource)
        
        # Log authorization attempt
        self.auditor.log_event(
            event_type="authorization",
            severity="info" if authorized else "warning",
            context=context,
            resource=resource or "unknown",
            action=permission.value,
            success=authorized,
            details={
                "permission": permission.value,
                "security_level": context.security_level.value
            }
        )
        
        return authorized
    
    def validate_and_sanitize_task_input(
        self, 
        raw_task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and sanitize task input data."""
        try:
            sanitized = {}
            
            # Task ID
            if "id" in raw_task_data:
                sanitized["id"] = self.sanitizer.sanitize_identifier(raw_task_data["id"])
            
            # Task name
            if "name" in raw_task_data:
                sanitized["name"] = self.sanitizer.sanitize_name(raw_task_data["name"])
            
            # Duration
            if "estimated_duration" in raw_task_data:
                sanitized["estimated_duration"] = self.sanitizer.validate_duration(
                    raw_task_data["estimated_duration"]
                )
            
            # Priority
            if "priority" in raw_task_data:
                sanitized["priority"] = max(0.0, min(10.0, float(raw_task_data["priority"])))
            
            # Dependencies
            if "dependencies" in raw_task_data:
                deps = raw_task_data["dependencies"]
                if isinstance(deps, (list, set)):
                    sanitized_deps = set()
                    for dep in deps:
                        if isinstance(dep, str):
                            sanitized_deps.add(self.sanitizer.sanitize_identifier(dep))
                    sanitized["dependencies"] = sanitized_deps
            
            # Resource requirements
            if "resource_requirements" in raw_task_data:
                resources = raw_task_data["resource_requirements"]
                if isinstance(resources, dict):
                    sanitized_resources = {}
                    allowed_resources = {"cpu", "memory", "io", "network", "gpu", "storage"}
                    
                    for resource_name, resource_value in resources.items():
                        if isinstance(resource_name, str) and resource_name in allowed_resources:
                            sanitized_resources[resource_name] = self.sanitizer.validate_resource_value(
                                resource_value
                            )
                    
                    sanitized["resource_requirements"] = sanitized_resources
            
            return sanitized
            
        except (ValueError, TypeError) as e:
            raise TaskValidationError(
                task_id=raw_task_data.get("id", "unknown"),
                validation_errors=[f"Input sanitization failed: {str(e)}"]
            )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security system status."""
        return {
            "active_sessions": len(self.active_sessions),
            "rate_limiter_status": {
                "blocked_identifiers": len(self.rate_limiter.blocked_until),
                "active_windows": len(self.rate_limiter.request_counts)
            },
            "security_summary": self.auditor.get_security_summary(24.0),
            "timestamp": time.time()
        }