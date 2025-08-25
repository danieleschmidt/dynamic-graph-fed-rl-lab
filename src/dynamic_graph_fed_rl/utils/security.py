"""
Enterprise-grade security framework with RBAC, audit logging, and threat protection.

This module provides comprehensive security features for federated learning systems,
including role-based access control, audit logging, threat detection, and compliance.
"""

import time
import json
import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Optional, Set, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from datetime import datetime, timedelta
import jwt
import bcrypt

from .error_handling import SecurityError, ValidationError


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    secret = "SECURE_SECRET_FROM_ENV"  # TODO: Use environment variable


class ActionType(Enum):
    """Types of actions that can be performed."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"
    AUDIT = "audit"


class ResourceType(Enum):
    """Types of resources in the system."""
    QUANTUM_BACKEND = "quantum_backend"
    FEDERATION_PROTOCOL = "federation_protocol"
    MODEL_PARAMETERS = "model_parameters"
    TRAINING_DATA = "training_data"
    SYSTEM_CONFIGURATION = "system_configuration"
    HEALTH_MONITORING = "health_monitoring"
    AUDIT_LOGS = "audit_logs"
    USER_MANAGEMENT = "user_management"


@dataclass
class Permission:
    """Represents a permission for an action on a resource."""
    resource_type: ResourceType
    action: ActionType
    security_level: SecurityLevel
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.resource_type, self.action, self.security_level))


@dataclass
class Role:
    """Represents a role with associated permissions."""
    name: str
    description: str
    permissions: Set[Permission]
    security_level: SecurityLevel
    created_at: float = field(default_factory=time.time)
    is_active: bool = True


@dataclass
class User:
    """Represents a user in the system."""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: Set[str]
    security_level: SecurityLevel
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    is_locked: bool = False
    is_active: bool = True
    session_token: Optional[str] = None
    token_expires: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_id: str
    timestamp: float
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, error
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self):
        self.suspicious_patterns = {}
        self.ip_whitelist = set()
        self.ip_blacklist = set()
        self.failed_login_threshold = 5
        self.rate_limit_window = 300  # 5 minutes
        self.rate_limits = {}
        
        # Threat detection rules
        self.threat_rules = {
            'brute_force': self._detect_brute_force,
            'privilege_escalation': self._detect_privilege_escalation,
            'data_exfiltration': self._detect_data_exfiltration,
            'anomalous_access': self._detect_anomalous_access
        }
    
    def analyze_event(self, event: AuditEvent) -> Dict[str, Any]:
        """Analyze an audit event for security threats."""
        threats_detected = []
        risk_score = 0.0
        
        for threat_type, detector in self.threat_rules.items():
            threat_result = detector(event)
            if threat_result['detected']:
                threats_detected.append({
                    'type': threat_type,
                    'confidence': threat_result['confidence'],
                    'details': threat_result['details']
                })
                risk_score = max(risk_score, threat_result['confidence'])
        
        return {
            'threats_detected': threats_detected,
            'risk_score': risk_score,
            'requires_action': risk_score > 0.7
        }
    
    def _detect_brute_force(self, event: AuditEvent) -> Dict[str, Any]:
        """Detect brute force attacks."""
        if event.action != 'login' or event.result == 'success':
            return {'detected': False, 'confidence': 0.0, 'details': {}}
        
        # Track failed login attempts
        user_key = f"login_failures:{event.user_id}"
        ip_key = f"login_failures:{event.ip_address}"
        
        current_time = time.time()
        
        # Check user-based failures
        if user_key not in self.suspicious_patterns:
            self.suspicious_patterns[user_key] = []
        
        # Clean old attempts
        self.suspicious_patterns[user_key] = [
            t for t in self.suspicious_patterns[user_key]
            if current_time - t < self.rate_limit_window
        ]
        
        self.suspicious_patterns[user_key].append(current_time)
        
        user_failures = len(self.suspicious_patterns[user_key])
        
        if user_failures >= self.failed_login_threshold:
            return {
                'detected': True,
                'confidence': min(user_failures / 10.0, 1.0),
                'details': {
                    'failed_attempts': user_failures,
                    'window': self.rate_limit_window
                }
            }
        
        return {'detected': False, 'confidence': 0.0, 'details': {}}
    
    def _detect_privilege_escalation(self, event: AuditEvent) -> Dict[str, Any]:
        """Detect privilege escalation attempts."""
        escalation_actions = ['admin', 'role_assignment', 'permission_grant']
        
        if event.action in escalation_actions:
            # Check if user typically performs these actions
            user_history_key = f"user_actions:{event.user_id}"
            
            if user_history_key not in self.suspicious_patterns:
                self.suspicious_patterns[user_history_key] = []
            
            recent_actions = [
                a for a in self.suspicious_patterns[user_history_key]
                if time.time() - a['timestamp'] < 3600  # Last hour
            ]
            
            admin_actions = [a for a in recent_actions if a['action'] in escalation_actions]
            
            if len(admin_actions) > 3:  # More than 3 admin actions in an hour
                return {
                    'detected': True,
                    'confidence': 0.8,
                    'details': {
                        'admin_actions_count': len(admin_actions),
                        'suspicious_actions': admin_actions
                    }
                }
        
        return {'detected': False, 'confidence': 0.0, 'details': {}}
    
    def _detect_data_exfiltration(self, event: AuditEvent) -> Dict[str, Any]:
        """Detect potential data exfiltration."""
        if event.action == 'read' and 'data_size' in event.details:
            data_size = event.details['data_size']
            
            # Track data access patterns
            user_key = f"data_access:{event.user_id}"
            
            if user_key not in self.suspicious_patterns:
                self.suspicious_patterns[user_key] = []
            
            current_time = time.time()
            self.suspicious_patterns[user_key].append({
                'timestamp': current_time,
                'size': data_size
            })
            
            # Clean old entries
            self.suspicious_patterns[user_key] = [
                entry for entry in self.suspicious_patterns[user_key]
                if current_time - entry['timestamp'] < 3600  # Last hour
            ]
            
            # Check for large data access
            total_size = sum(entry['size'] for entry in self.suspicious_patterns[user_key])
            
            if total_size > 1024 * 1024 * 100:  # 100MB in an hour
                return {
                    'detected': True,
                    'confidence': 0.9,
                    'details': {
                        'total_data_accessed': total_size,
                        'access_count': len(self.suspicious_patterns[user_key])
                    }
                }
        
        return {'detected': False, 'confidence': 0.0, 'details': {}}
    
    def _detect_anomalous_access(self, event: AuditEvent) -> Dict[str, Any]:
        """Detect anomalous access patterns."""
        # Check for unusual time access
        current_hour = datetime.fromtimestamp(event.timestamp).hour
        
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            return {
                'detected': True,
                'confidence': 0.6,
                'details': {
                    'access_time': current_hour,
                    'reason': 'outside_business_hours'
                }
            }
        
        # Check for unusual IP addresses
        if event.ip_address and event.ip_address in self.ip_blacklist:
            return {
                'detected': True,
                'confidence': 1.0,
                'details': {
                    'ip_address': event.ip_address,
                    'reason': 'blacklisted_ip'
                }
            }
        
        return {'detected': False, 'confidence': 0.0, 'details': {}}


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Initialize default roles
        self._initialize_default_roles()
        
        # Audit logging
        self.audit_logger = AuditLogger()
        
        # Threat detection
        self.threat_detector = ThreatDetector()
        
        logging.info("RBAC Manager initialized with threat detection")
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        # Admin role
        admin_permissions = {
            Permission(ResourceType.QUANTUM_BACKEND, ActionType.ADMIN, SecurityLevel.SECRET),
            Permission(ResourceType.FEDERATION_PROTOCOL, ActionType.ADMIN, SecurityLevel.SECRET),
            Permission(ResourceType.SYSTEM_CONFIGURATION, ActionType.ADMIN, SecurityLevel.SECRET),
            Permission(ResourceType.USER_MANAGEMENT, ActionType.ADMIN, SecurityLevel.SECRET),
            Permission(ResourceType.AUDIT_LOGS, ActionType.READ, SecurityLevel.SECRET),
        }
        
        self.roles['admin'] = Role(
            name='admin',
            description='System administrator with full access',
            permissions=admin_permissions,
            security_level=SecurityLevel.SECRET
        )
        
        # Researcher role
        researcher_permissions = {
            Permission(ResourceType.QUANTUM_BACKEND, ActionType.READ, SecurityLevel.RESTRICTED),
            Permission(ResourceType.QUANTUM_BACKEND, ActionType.EXECUTE, SecurityLevel.RESTRICTED),
            Permission(ResourceType.FEDERATION_PROTOCOL, ActionType.READ, SecurityLevel.RESTRICTED),
            Permission(ResourceType.MODEL_PARAMETERS, ActionType.READ, SecurityLevel.RESTRICTED),
            Permission(ResourceType.MODEL_PARAMETERS, ActionType.WRITE, SecurityLevel.RESTRICTED),
            Permission(ResourceType.TRAINING_DATA, ActionType.READ, SecurityLevel.RESTRICTED),
        }
        
        self.roles['researcher'] = Role(
            name='researcher',
            description='Research scientist with access to quantum and federation resources',
            permissions=researcher_permissions,
            security_level=SecurityLevel.RESTRICTED
        )
        
        # Data scientist role
        data_scientist_permissions = {
            Permission(ResourceType.MODEL_PARAMETERS, ActionType.READ, SecurityLevel.INTERNAL),
            Permission(ResourceType.MODEL_PARAMETERS, ActionType.WRITE, SecurityLevel.INTERNAL),
            Permission(ResourceType.TRAINING_DATA, ActionType.READ, SecurityLevel.INTERNAL),
            Permission(ResourceType.FEDERATION_PROTOCOL, ActionType.READ, SecurityLevel.INTERNAL),
        }
        
        self.roles['data_scientist'] = Role(
            name='data_scientist',
            description='Data scientist with access to model and data resources',
            permissions=data_scientist_permissions,
            security_level=SecurityLevel.INTERNAL
        )
        
        # Viewer role
        viewer_permissions = {
            Permission(ResourceType.HEALTH_MONITORING, ActionType.READ, SecurityLevel.PUBLIC),
            Permission(ResourceType.MODEL_PARAMETERS, ActionType.READ, SecurityLevel.PUBLIC),
        }
        
        self.roles['viewer'] = Role(
            name='viewer',
            description='Read-only access to public resources',
            permissions=viewer_permissions,
            security_level=SecurityLevel.PUBLIC
        )
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[str],
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        created_by: Optional[str] = None
    ) -> str:
        """Create a new user."""
        # Validate inputs
        if not username or not email or not password:
            raise ValidationError("Username, email, and password are required")
        
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters")
        
        # Check if user already exists
        user_id = hashlib.sha256(f"{username}:{email}".encode()).hexdigest()[:16]
        
        with self.lock:
            if any(u.username == username for u in self.users.values()):
                raise ValidationError(f"Username '{username}' already exists")
            
            if any(u.email == email for u in self.users.values()):
                raise ValidationError(f"Email '{email}' already exists")
            
            # Validate roles exist
            for role_name in roles:
                if role_name not in self.roles:
                    raise ValidationError(f"Role '{role_name}' does not exist")
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            
            # Create user
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                roles=set(roles),
                security_level=security_level
            )
            
            self.users[user_id] = user
        
        # Audit log
        self.audit_logger.log_event(
            user_id=created_by or "system",
            action="create_user",
            resource=f"user:{user_id}",
            result="success",
            details={
                "username": username,
                "email": email,
                "roles": roles,
                "security_level": security_level.value
            }
        )
        
        logging.info(f"Created user '{username}' with roles {roles}")
        return user_id
    
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[str]:
        """Authenticate user and return session token."""
        with self.lock:
            user = next((u for u in self.users.values() if u.username == username), None)
            
            if not user or not user.is_active:
                self._log_failed_login(username, "invalid_user", ip_address, user_agent)
                return None
            
            if user.is_locked:
                self._log_failed_login(username, "account_locked", ip_address, user_agent)
                return None
            
            # Verify password
            if not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
                user.failed_login_attempts += 1
                
                # Lock account after too many failures
                if user.failed_login_attempts >= 5:
                    user.is_locked = True
                    logging.warning(f"Account locked for user '{username}' due to failed login attempts")
                
                self._log_failed_login(username, "invalid_password", ip_address, user_agent)
                return None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.last_login = time.time()
            
            # Generate session token
            session_token = self._generate_session_token(user.user_id)
            user.session_token = session_token
            user.token_expires = time.time() + 3600  # 1 hour
            
            # Store session
            self.sessions[session_token] = {
                'user_id': user.user_id,
                'created_at': time.time(),
                'ip_address': ip_address,
                'user_agent': user_agent
            }
        
        # Audit log
        self.audit_logger.log_event(
            user_id=user.user_id,
            action="login",
            resource="auth_system",
            result="success",
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        logging.info(f"User '{username}' authenticated successfully")
        return session_token
    
    def _log_failed_login(
        self,
        username: str,
        reason: str,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ):
        """Log failed login attempt."""
        self.audit_logger.log_event(
            user_id=username,
            action="login",
            resource="auth_system",
            result="failure",
            ip_address=ip_address,
            user_agent=user_agent,
            details={"reason": reason}
        )
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate secure session token."""
        payload = {
            'user_id': user_id,
            'iat': time.time(),
            'exp': time.time() + 3600  # 1 hour
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_session(self, session_token: str) -> Optional[str]:
        """Validate session token and return user_id."""
        try:
            payload = jwt.decode(session_token, self.secret_key, algorithms=['HS256'])
            user_id = payload['user_id']
            
            with self.lock:
                if session_token in self.sessions and user_id in self.users:
                    user = self.users[user_id]
                    
                    # Check token expiry
                    if user.token_expires and time.time() > user.token_expires:
                        self.logout_user(session_token)
                        return None
                    
                    return user_id
                
        except jwt.InvalidTokenError:
            pass
        
        return None
    
    def check_permission(
        self,
        user_id: str,
        resource_type: ResourceType,
        action: ActionType,
        security_level: SecurityLevel = SecurityLevel.PUBLIC
    ) -> bool:
        """Check if user has permission for action on resource."""
        with self.lock:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            
            if not user.is_active:
                return False
            
            # Check user's security level
            if user.security_level.value not in self._get_allowed_levels(security_level):
                return False
            
            # Check role permissions
            for role_name in user.roles:
                if role_name in self.roles:
                    role = self.roles[role_name]
                    
                    for permission in role.permissions:
                        if (permission.resource_type == resource_type and
                            permission.action == action and
                            self._has_security_clearance(user.security_level, permission.security_level)):
                            return True
        
        return False
    
    def _get_allowed_levels(self, required_level: SecurityLevel) -> List[str]:
        """Get security levels that can access the required level."""
        level_hierarchy = {
            SecurityLevel.PUBLIC: ['public', 'internal', 'restricted', 'confidential', 'secret'],
            SecurityLevel.INTERNAL: ['internal', 'restricted', 'confidential', 'secret'],
            SecurityLevel.RESTRICTED: ['restricted', 'confidential', 'secret'],
            SecurityLevel.CONFIDENTIAL: ['confidential', 'secret'],
            SecurityLevel.SECRET: ['secret']
        }
        return level_hierarchy.get(required_level, [])
    
    def _has_security_clearance(self, user_level: SecurityLevel, required_level: SecurityLevel) -> bool:
        """Check if user has required security clearance."""
        level_values = {
            SecurityLevel.PUBLIC: 1,
            SecurityLevel.INTERNAL: 2,
            SecurityLevel.RESTRICTED: 3,
            SecurityLevel.CONFIDENTIAL: 4,
            SecurityLevel.SECRET: 5
        }
        
        return level_values.get(user_level, 0) >= level_values.get(required_level, 0)
    
    def authorize_action(
        self,
        session_token: str,
        resource_type: ResourceType,
        action: ActionType,
        resource_id: str = "",
        security_level: SecurityLevel = SecurityLevel.PUBLIC,
        ip_address: Optional[str] = None
    ) -> bool:
        """Authorize an action and log the attempt."""
        user_id = self.validate_session(session_token)
        
        if not user_id:
            self.audit_logger.log_event(
                user_id="unknown",
                action=action.value,
                resource=f"{resource_type.value}:{resource_id}",
                result="failure",
                ip_address=ip_address,
                details={"reason": "invalid_session"}
            )
            return False
        
        # Check permission
        has_permission = self.check_permission(user_id, resource_type, action, security_level)
        
        # Log the attempt
        result = "success" if has_permission else "failure"
        self.audit_logger.log_event(
            user_id=user_id,
            action=action.value,
            resource=f"{resource_type.value}:{resource_id}",
            result=result,
            ip_address=ip_address,
            details={
                "security_level": security_level.value,
                "permission_granted": has_permission
            }
        )
        
        return has_permission
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user and invalidate session."""
        with self.lock:
            if session_token in self.sessions:
                session_info = self.sessions[session_token]
                user_id = session_info['user_id']
                
                # Clear session
                del self.sessions[session_token]
                
                # Clear user session token
                if user_id in self.users:
                    user = self.users[user_id]
                    user.session_token = None
                    user.token_expires = None
                
                # Audit log
                self.audit_logger.log_event(
                    user_id=user_id,
                    action="logout",
                    resource="auth_system",
                    result="success"
                )
                
                return True
        
        return False
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        permissions = set()
        
        with self.lock:
            if user_id in self.users:
                user = self.users[user_id]
                
                for role_name in user.roles:
                    if role_name in self.roles:
                        permissions.update(self.roles[role_name].permissions)
        
        return permissions
    
    def add_role_to_user(self, user_id: str, role_name: str, added_by: str) -> bool:
        """Add role to user."""
        with self.lock:
            if user_id not in self.users or role_name not in self.roles:
                return False
            
            user = self.users[user_id]
            user.roles.add(role_name)
        
        self.audit_logger.log_event(
            user_id=added_by,
            action="add_role",
            resource=f"user:{user_id}",
            result="success",
            details={"role": role_name}
        )
        
        return True
    
    def remove_role_from_user(self, user_id: str, role_name: str, removed_by: str) -> bool:
        """Remove role from user."""
        with self.lock:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            user.roles.discard(role_name)
        
        self.audit_logger.log_event(
            user_id=removed_by,
            action="remove_role",
            resource=f"user:{user_id}",
            result="success",
            details={"role": role_name}
        )
        
        return True


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self.events: List[AuditEvent] = []
        self.lock = threading.Lock()
        
        # Risk analysis
        self.threat_detector = ThreatDetector()
        
        logging.info("Audit logger initialized")
    
    def log_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an audit event."""
        event_id = hashlib.sha256(f"{time.time()}:{user_id}:{action}".encode()).hexdigest()[:16]
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        
        # Analyze for threats
        threat_analysis = self.threat_detector.analyze_event(event)
        event.risk_score = threat_analysis['risk_score']
        
        with self.lock:
            self.events.append(event)
            
            # Maintain size limit
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events//2:]
        
        # Log high-risk events
        if threat_analysis['requires_action']:
            logging.warning(
                f"High-risk security event detected: {action} by {user_id} "
                f"(risk: {event.risk_score:.2f})"
            )
        
        return event_id
    
    def get_events(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Get audit events with filtering."""
        with self.lock:
            filtered_events = self.events.copy()
        
        # Apply filters
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if action:
            filtered_events = [e for e in filtered_events if e.action == action]
        
        if resource:
            filtered_events = [e for e in filtered_events if resource in e.resource]
        
        if time_range:
            start_time, end_time = time_range
            filtered_events = [
                e for e in filtered_events
                if start_time <= e.timestamp <= end_time
            ]
        
        # Sort by timestamp (most recent first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return filtered_events[:limit]
    
    def get_security_report(self, time_window: float = 86400) -> Dict[str, Any]:
        """Generate security report for specified time window."""
        current_time = time.time()
        start_time = current_time - time_window
        
        recent_events = self.get_events(time_range=(start_time, current_time))
        
        # Calculate metrics
        total_events = len(recent_events)
        failed_events = len([e for e in recent_events if e.result == 'failure'])
        high_risk_events = len([e for e in recent_events if e.risk_score > 0.7])
        
        # User activity
        unique_users = len(set(e.user_id for e in recent_events))
        
        # Action breakdown
        action_counts = {}
        for event in recent_events:
            action_counts[event.action] = action_counts.get(event.action, 0) + 1
        
        # Resource access
        resource_counts = {}
        for event in recent_events:
            resource_type = event.resource.split(':')[0]
            resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
        
        return {
            'time_window_hours': time_window / 3600,
            'total_events': total_events,
            'failed_events': failed_events,
            'high_risk_events': high_risk_events,
            'unique_users': unique_users,
            'action_breakdown': action_counts,
            'resource_access': resource_counts,
            'security_score': max(0, 100 - (failed_events + high_risk_events * 2)),
            'top_risks': [
                {
                    'event_id': e.event_id,
                    'user_id': e.user_id,
                    'action': e.action,
                    'risk_score': e.risk_score,
                    'timestamp': e.timestamp
                }
                for e in sorted(recent_events, key=lambda x: x.risk_score, reverse=True)[:10]
            ]
        }


# Global RBAC instance
rbac = RBACManager()


# Decorator for authorization
def require_permission(
    resource_type: ResourceType,
    action: ActionType,
    security_level: SecurityLevel = SecurityLevel.PUBLIC
):
    """Decorator to require specific permission for function access."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract session token from kwargs or request context
            session_token = kwargs.get('session_token')
            if not session_token:
                raise SecurityError("Session token required")
            
            # Extract resource ID if available
            resource_id = kwargs.get('resource_id', '')
            ip_address = kwargs.get('ip_address')
            
            # Check authorization
            if not rbac.authorize_action(
                session_token, resource_type, action, resource_id, security_level, ip_address
            ):
                raise SecurityError(f"Access denied: insufficient permissions for {action.value} on {resource_type.value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Convenience functions
def create_admin_user(username: str, email: str, password: str) -> str:
    """Create an admin user."""
    return rbac.create_user(
        username=username,
        email=email,
        password=password,
        roles=['admin'],
        security_level=SecurityLevel.SECRET
    )

def authenticate(username: str, password: str, ip_address: Optional[str] = None) -> Optional[str]:
    """Authenticate user and return session token."""
    return rbac.authenticate_user(username, password, ip_address)

def authorize(
    session_token: str,
    resource_type: ResourceType,
    action: ActionType,
    security_level: SecurityLevel = SecurityLevel.PUBLIC,
    resource_id: str = "",
    ip_address: Optional[str] = None
) -> bool:
    """Authorize an action."""
    return rbac.authorize_action(session_token, resource_type, action, resource_id, security_level, ip_address)

def get_security_report(time_window: float = 86400) -> Dict[str, Any]:
    """Get security report."""
    return rbac.audit_logger.get_security_report(time_window)