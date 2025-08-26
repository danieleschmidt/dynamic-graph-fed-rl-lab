"""
Zero-Trust Security Architecture for Federated Learning Systems.

This module implements enterprise-grade zero-trust security principles including:
- Identity verification and continuous authentication
- Micro-segmentation and network isolation
- Real-time threat detection and response
- Least-privilege access controls
- End-to-end encryption and data protection
- Behavioral analysis and anomaly detection
"""

import asyncio
import json
import time
import hashlib
import secrets
import logging
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union, Callable
from enum import Enum
import jwt
import ipaddress
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .error_handling import (
    circuit_breaker, retry, robust, SecurityError, ValidationError,
    CircuitBreakerConfig, RetryConfig, resilience
)
from .security import rbac, SecurityLevel, ActionType, ResourceType


class TrustLevel(Enum):
    """Trust levels in zero-trust architecture."""
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NetworkZone(Enum):
    """Network security zones."""
    DMZ = "dmz"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CRITICAL = "critical"
    QUARANTINE = "quarantine"


class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """Comprehensive security context for zero-trust decisions."""
    user_id: str
    session_id: str
    device_id: str
    ip_address: str
    user_agent: str
    timestamp: float
    trust_level: TrustLevel
    network_zone: NetworkZone
    authentication_factors: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    behavioral_score: float = 0.0
    geolocation: Optional[Dict[str, Any]] = None
    device_fingerprint: Optional[Dict[str, Any]] = None
    previous_activities: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AccessRequest:
    """Access request for zero-trust evaluation."""
    request_id: str
    security_context: SecurityContext
    resource_type: ResourceType
    action: ActionType
    resource_id: str
    requested_permissions: List[str]
    additional_context: Dict[str, Any] = field(default_factory=dict)
    urgency_level: str = "normal"  # normal, high, emergency


@dataclass
class ThreatEvent:
    """Security threat event."""
    event_id: str
    timestamp: float
    threat_type: str
    threat_level: ThreatLevel
    source_ip: str
    target_resource: str
    description: str
    indicators: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False


class EncryptionManager:
    """Advanced encryption management for zero-trust security."""
    
    def __init__(self):
        self.master_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        self.key_rotation_interval = 3600  # 1 hour
        self.last_key_rotation = time.time()
        
        # Generate RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # Key history for graceful rotation
        self.key_history: List[bytes] = []
        
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt data with current key."""
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = self.cipher_suite.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> bytes:
        """Decrypt data, trying current and historical keys."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        
        # Try current key first
        try:
            return self.cipher_suite.decrypt(encrypted_bytes)
        except Exception:
            pass
        
        # Try historical keys
        for old_key in self.key_history:
            try:
                old_cipher = Fernet(old_key)
                return old_cipher.decrypt(encrypted_bytes)
            except Exception:
                continue
        
        raise SecurityError("Failed to decrypt data with any available key")
    
    def encrypt_asymmetric(self, data: bytes, public_key=None) -> bytes:
        """Encrypt data using RSA public key."""
        key = public_key or self.public_key
        
        # RSA can only encrypt small amounts, so we use hybrid encryption
        # Generate a random symmetric key
        symmetric_key = Fernet.generate_key()
        symmetric_cipher = Fernet(symmetric_key)
        
        # Encrypt data with symmetric key
        encrypted_data = symmetric_cipher.encrypt(data)
        
        # Encrypt symmetric key with RSA
        encrypted_key = key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combine encrypted key and data
        return encrypted_key + b':::' + encrypted_data
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using RSA private key."""
        # Split encrypted key and data
        parts = encrypted_data.split(b':::', 1)
        if len(parts) != 2:
            raise SecurityError("Invalid encrypted data format")
        
        encrypted_key, encrypted_data = parts
        
        # Decrypt symmetric key with RSA
        symmetric_key = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt data with symmetric key
        symmetric_cipher = Fernet(symmetric_key)
        return symmetric_cipher.decrypt(encrypted_data)
    
    def rotate_keys(self):
        """Rotate encryption keys for security."""
        # Save current key to history
        self.key_history.append(self.master_key)
        
        # Generate new key
        self.master_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        self.last_key_rotation = time.time()
        
        # Keep only recent keys in history
        if len(self.key_history) > 10:
            self.key_history = self.key_history[-10:]
        
        logging.info("Encryption keys rotated successfully")
    
    def should_rotate_keys(self) -> bool:
        """Check if keys should be rotated."""
        return time.time() - self.last_key_rotation > self.key_rotation_interval


class BehavioralAnalyzer:
    """Behavioral analysis for anomaly detection in zero-trust."""
    
    def __init__(self):
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
        self.activity_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.anomaly_thresholds = {
            "login_time": 2.0,  # Standard deviations
            "access_pattern": 2.5,
            "data_volume": 3.0,
            "geographic_location": 1.5
        }
    
    def analyze_user_behavior(self, security_context: SecurityContext) -> float:
        """Analyze user behavior and return anomaly score (0.0 = normal, 1.0 = highly anomalous)."""
        user_id = security_context.user_id
        current_time = datetime.fromtimestamp(security_context.timestamp)
        
        # Initialize baseline if first time seeing user
        if user_id not in self.user_baselines:
            self._initialize_user_baseline(user_id, security_context)
            return 0.0  # No baseline yet
        
        baseline = self.user_baselines[user_id]
        anomaly_score = 0.0
        
        # Analyze login time patterns
        time_anomaly = self._analyze_time_pattern(user_id, current_time)
        anomaly_score += time_anomaly * 0.3
        
        # Analyze access patterns
        access_anomaly = self._analyze_access_pattern(user_id, security_context)
        anomaly_score += access_anomaly * 0.3
        
        # Analyze geographic patterns
        geo_anomaly = self._analyze_geographic_pattern(user_id, security_context)
        anomaly_score += geo_anomaly * 0.2
        
        # Analyze device patterns
        device_anomaly = self._analyze_device_pattern(user_id, security_context)
        anomaly_score += device_anomaly * 0.2
        
        # Update baseline with current activity
        self._update_user_baseline(user_id, security_context)
        
        return min(anomaly_score, 1.0)
    
    def _initialize_user_baseline(self, user_id: str, security_context: SecurityContext):
        """Initialize behavioral baseline for new user."""
        self.user_baselines[user_id] = {
            "first_seen": security_context.timestamp,
            "typical_login_hours": [],
            "typical_ips": set(),
            "typical_user_agents": set(),
            "typical_devices": set(),
            "activity_volume": [],
            "geographic_locations": []
        }
        self.activity_patterns[user_id] = []
    
    def _analyze_time_pattern(self, user_id: str, current_time: datetime) -> float:
        """Analyze if current time is unusual for this user."""
        baseline = self.user_baselines[user_id]
        typical_hours = baseline.get("typical_login_hours", [])
        
        if len(typical_hours) < 5:  # Not enough data
            return 0.0
        
        current_hour = current_time.hour
        
        # Calculate how unusual this hour is
        import numpy as np
        hour_array = np.array(typical_hours)
        mean_hour = np.mean(hour_array)
        std_hour = np.std(hour_array)
        
        if std_hour > 0:
            z_score = abs(current_hour - mean_hour) / std_hour
            return min(z_score / self.anomaly_thresholds["login_time"], 1.0)
        
        return 0.0
    
    def _analyze_access_pattern(self, user_id: str, security_context: SecurityContext) -> float:
        """Analyze if current access pattern is unusual."""
        # Simplified analysis - in practice, this would be more sophisticated
        baseline = self.user_baselines[user_id]
        typical_ips = baseline.get("typical_ips", set())
        
        if len(typical_ips) == 0:
            return 0.0
        
        # Check if IP is completely new
        if security_context.ip_address not in typical_ips and len(typical_ips) > 3:
            return 0.7  # Moderately suspicious
        
        return 0.0
    
    def _analyze_geographic_pattern(self, user_id: str, security_context: SecurityContext) -> float:
        """Analyze geographic location patterns."""
        if not security_context.geolocation:
            return 0.0
        
        baseline = self.user_baselines[user_id]
        typical_locations = baseline.get("geographic_locations", [])
        
        if len(typical_locations) < 3:
            return 0.0
        
        # Simplified geographic analysis
        current_country = security_context.geolocation.get("country")
        typical_countries = [loc.get("country") for loc in typical_locations]
        
        if current_country not in typical_countries:
            return 0.8  # Highly suspicious if from new country
        
        return 0.0
    
    def _analyze_device_pattern(self, user_id: str, security_context: SecurityContext) -> float:
        """Analyze device fingerprint patterns."""
        baseline = self.user_baselines[user_id]
        typical_devices = baseline.get("typical_devices", set())
        
        if len(typical_devices) == 0:
            return 0.0
        
        # Check if device is completely new
        if security_context.device_id not in typical_devices and len(typical_devices) > 2:
            return 0.6  # Moderately suspicious
        
        return 0.0
    
    def _update_user_baseline(self, user_id: str, security_context: SecurityContext):
        """Update user baseline with current activity."""
        baseline = self.user_baselines[user_id]
        current_time = datetime.fromtimestamp(security_context.timestamp)
        
        # Update typical hours
        baseline["typical_login_hours"].append(current_time.hour)
        if len(baseline["typical_login_hours"]) > 100:
            baseline["typical_login_hours"] = baseline["typical_login_hours"][-50:]
        
        # Update typical IPs
        baseline["typical_ips"].add(security_context.ip_address)
        if len(baseline["typical_ips"]) > 20:
            # Keep most recent IPs (simplified)
            baseline["typical_ips"] = set(list(baseline["typical_ips"])[-10:])
        
        # Update devices
        baseline["typical_devices"].add(security_context.device_id)
        
        # Update geographic locations
        if security_context.geolocation:
            baseline["geographic_locations"].append(security_context.geolocation)
            if len(baseline["geographic_locations"]) > 50:
                baseline["geographic_locations"] = baseline["geographic_locations"][-25:]


class NetworkSegmentationManager:
    """Network micro-segmentation for zero-trust architecture."""
    
    def __init__(self):
        self.network_policies: Dict[str, Dict[str, Any]] = {}
        self.zone_rules: Dict[NetworkZone, Set[str]] = {
            NetworkZone.DMZ: set(),
            NetworkZone.INTERNAL: set(),
            NetworkZone.RESTRICTED: set(),
            NetworkZone.CRITICAL: set(),
            NetworkZone.QUARANTINE: set()
        }
        self.ip_zone_mapping: Dict[str, NetworkZone] = {}
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default network segmentation policies."""
        # DMZ policies - most restrictive
        self.network_policies["dmz_to_internal"] = {
            "allowed": False,
            "requires_authentication": True,
            "max_connections": 10,
            "allowed_ports": [443, 80],
            "rate_limit": 100  # requests per minute
        }
        
        # Internal to restricted
        self.network_policies["internal_to_restricted"] = {
            "allowed": True,
            "requires_authentication": True,
            "max_connections": 50,
            "allowed_ports": [443, 22],
            "rate_limit": 1000
        }
        
        # Quarantine policies - very restrictive
        self.network_policies["quarantine_outbound"] = {
            "allowed": False,
            "requires_authentication": True,
            "max_connections": 1,
            "allowed_ports": [],
            "rate_limit": 10
        }
    
    def classify_ip_zone(self, ip_address: str) -> NetworkZone:
        """Classify IP address into network zone."""
        if ip_address in self.ip_zone_mapping:
            return self.ip_zone_mapping[ip_address]
        
        # Default classification based on IP ranges
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Private networks - internal by default
            if ip.is_private:
                return NetworkZone.INTERNAL
            
            # Loopback - critical
            if ip.is_loopback:
                return NetworkZone.CRITICAL
            
            # Public IPs - DMZ by default
            return NetworkZone.DMZ
        
        except ValueError:
            # Invalid IP - quarantine
            return NetworkZone.QUARANTINE
    
    def evaluate_network_access(
        self,
        source_ip: str,
        destination_zone: NetworkZone,
        port: int,
        protocol: str = "tcp"
    ) -> Dict[str, Any]:
        """Evaluate if network access should be allowed."""
        source_zone = self.classify_ip_zone(source_ip)
        
        # Generate policy key
        policy_key = f"{source_zone.value}_to_{destination_zone.value}"
        
        # Get applicable policy
        policy = self.network_policies.get(policy_key, {
            "allowed": False,
            "requires_authentication": True,
            "max_connections": 0,
            "allowed_ports": [],
            "rate_limit": 0
        })
        
        # Evaluate access
        access_decision = {
            "allowed": policy.get("allowed", False),
            "source_zone": source_zone.value,
            "destination_zone": destination_zone.value,
            "requires_mfa": False,  # Default
            "rate_limit": policy.get("rate_limit", 0),
            "conditions": []
        }
        
        # Check port restrictions
        allowed_ports = policy.get("allowed_ports", [])
        if allowed_ports and port not in allowed_ports:
            access_decision["allowed"] = False
            access_decision["conditions"].append(f"Port {port} not allowed")
        
        # Apply additional restrictions for high-risk zones
        if source_zone == NetworkZone.QUARANTINE:
            access_decision["allowed"] = False
            access_decision["conditions"].append("Source in quarantine")
        
        if destination_zone == NetworkZone.CRITICAL:
            access_decision["requires_mfa"] = True
            access_decision["conditions"].append("Critical zone requires MFA")
        
        return access_decision
    
    def quarantine_ip(self, ip_address: str, reason: str):
        """Move IP address to quarantine zone."""
        self.ip_zone_mapping[ip_address] = NetworkZone.QUARANTINE
        self.zone_rules[NetworkZone.QUARANTINE].add(ip_address)
        
        logging.warning(f"IP {ip_address} quarantined: {reason}")
    
    def release_from_quarantine(self, ip_address: str):
        """Release IP address from quarantine."""
        if ip_address in self.ip_zone_mapping:
            del self.ip_zone_mapping[ip_address]
        
        self.zone_rules[NetworkZone.QUARANTINE].discard(ip_address)
        
        logging.info(f"IP {ip_address} released from quarantine")


class ZeroTrustEngine:
    """
    Core zero-trust security engine that makes access decisions.
    
    Features:
    - Continuous identity verification
    - Risk-based access control
    - Behavioral analysis and anomaly detection
    - Network micro-segmentation
    - Real-time threat response
    - Policy enforcement and audit logging
    """
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.network_manager = NetworkSegmentationManager()
        
        # Trust and risk management
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.trust_policies: Dict[str, Dict[str, Any]] = {}
        self.threat_events: List[ThreatEvent] = []
        
        # Zero-trust policies
        self.default_trust_level = TrustLevel.UNTRUSTED
        self.min_trust_levels = {
            ResourceType.QUANTUM_BACKEND: TrustLevel.HIGH,
            ResourceType.FEDERATION_PROTOCOL: TrustLevel.MEDIUM,
            ResourceType.MODEL_PARAMETERS: TrustLevel.MEDIUM,
            ResourceType.TRAINING_DATA: TrustLevel.LOW,
            ResourceType.SYSTEM_CONFIGURATION: TrustLevel.CRITICAL,
            ResourceType.HEALTH_MONITORING: TrustLevel.LOW,
            ResourceType.AUDIT_LOGS: TrustLevel.HIGH,
            ResourceType.USER_MANAGEMENT: TrustLevel.CRITICAL
        }
        
        # Initialize circuit breakers
        self._setup_zero_trust_circuit_breakers()
        
        logging.info("Zero-Trust Security Engine initialized")
    
    def _setup_zero_trust_circuit_breakers(self):
        """Setup circuit breakers for zero-trust operations."""
        # Access decision circuit breaker
        access_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=300.0,  # 5 minutes
            expected_exception=(SecurityError, ValidationError)
        )
        self.access_circuit = resilience.register_circuit_breaker(
            "zero-trust-access-decisions",
            access_config
        )
        
        # Threat detection circuit breaker
        threat_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=180.0,  # 3 minutes
            expected_exception=(Exception,)
        )
        self.threat_circuit = resilience.register_circuit_breaker(
            "zero-trust-threat-detection",
            threat_config
        )
    
    @circuit_breaker("zero_trust_evaluation", failure_threshold=3, recovery_timeout=300.0)
    async def evaluate_access_request(self, access_request: AccessRequest) -> Dict[str, Any]:
        """Comprehensive zero-trust access evaluation."""
        start_time = time.time()
        
        try:
            # Step 1: Identity verification
            identity_result = await self._verify_identity(access_request.security_context)
            
            # Step 2: Risk assessment
            risk_score = await self._assess_risk(access_request)
            
            # Step 3: Trust level determination
            trust_level = await self._determine_trust_level(access_request, risk_score)
            
            # Step 4: Policy evaluation
            policy_result = await self._evaluate_policies(access_request, trust_level)
            
            # Step 5: Network access control
            network_result = self._evaluate_network_access(access_request)
            
            # Step 6: Behavioral analysis
            behavioral_score = self.behavioral_analyzer.analyze_user_behavior(
                access_request.security_context
            )
            
            # Step 7: Make final access decision
            access_decision = self._make_access_decision(
                access_request,
                identity_result,
                risk_score,
                trust_level,
                policy_result,
                network_result,
                behavioral_score
            )
            
            # Step 8: Log decision and update context
            await self._log_access_decision(access_request, access_decision)
            
            # Step 9: Apply enforcement actions
            if access_decision["granted"]:
                await self._apply_access_controls(access_request, access_decision)
            else:
                await self._apply_denial_actions(access_request, access_decision)
            
            access_decision["evaluation_time"] = time.time() - start_time
            return access_decision
        
        except Exception as e:
            logging.error(f"Zero-trust evaluation failed: {e}")
            # Fail secure - deny access on evaluation errors
            return {
                "granted": False,
                "reason": "evaluation_error",
                "error": str(e),
                "trust_level": TrustLevel.UNTRUSTED.value,
                "required_actions": ["contact_administrator"],
                "evaluation_time": time.time() - start_time
            }
    
    async def _verify_identity(self, security_context: SecurityContext) -> Dict[str, Any]:
        """Verify user identity with multiple factors."""
        verification_result = {
            "verified": False,
            "confidence": 0.0,
            "factors_used": [],
            "additional_factors_required": []
        }
        
        # Check session validity
        session_valid = security_context.session_id in self.active_sessions
        if session_valid:
            verification_result["factors_used"].append("valid_session")
            verification_result["confidence"] += 0.3
        
        # Check authentication factors
        auth_factors = security_context.authentication_factors
        
        if "password" in auth_factors:
            verification_result["factors_used"].append("password")
            verification_result["confidence"] += 0.4
        
        if "mfa_token" in auth_factors or "totp" in auth_factors:
            verification_result["factors_used"].append("mfa")
            verification_result["confidence"] += 0.3
        
        if "biometric" in auth_factors:
            verification_result["factors_used"].append("biometric")
            verification_result["confidence"] += 0.2
        
        if "certificate" in auth_factors:
            verification_result["factors_used"].append("certificate")
            verification_result["confidence"] += 0.3
        
        # Determine if verified
        verification_result["verified"] = verification_result["confidence"] >= 0.7
        
        # Require additional factors if confidence is low
        if verification_result["confidence"] < 0.7:
            if "mfa" not in verification_result["factors_used"]:
                verification_result["additional_factors_required"].append("mfa")
        
        return verification_result
    
    async def _assess_risk(self, access_request: AccessRequest) -> float:
        """Assess risk score for access request."""
        risk_score = 0.0
        
        # Base risk from security context
        risk_score += access_request.security_context.risk_score * 0.3
        
        # Resource sensitivity risk
        resource_risk = {
            ResourceType.QUANTUM_BACKEND: 0.8,
            ResourceType.FEDERATION_PROTOCOL: 0.6,
            ResourceType.MODEL_PARAMETERS: 0.5,
            ResourceType.TRAINING_DATA: 0.4,
            ResourceType.SYSTEM_CONFIGURATION: 0.9,
            ResourceType.HEALTH_MONITORING: 0.2,
            ResourceType.AUDIT_LOGS: 0.7,
            ResourceType.USER_MANAGEMENT: 0.9
        }
        risk_score += resource_risk.get(access_request.resource_type, 0.5) * 0.3
        
        # Action risk
        action_risk = {
            ActionType.READ: 0.1,
            ActionType.WRITE: 0.5,
            ActionType.EXECUTE: 0.7,
            ActionType.DELETE: 0.9,
            ActionType.ADMIN: 1.0,
            ActionType.AUDIT: 0.3
        }
        risk_score += action_risk.get(access_request.action, 0.5) * 0.2
        
        # Time-based risk (access outside normal hours)
        current_hour = datetime.fromtimestamp(access_request.security_context.timestamp).hour
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            risk_score += 0.2
        
        # Network zone risk
        network_zone = self.network_manager.classify_ip_zone(
            access_request.security_context.ip_address
        )
        zone_risk = {
            NetworkZone.CRITICAL: 0.0,
            NetworkZone.INTERNAL: 0.1,
            NetworkZone.RESTRICTED: 0.2,
            NetworkZone.DMZ: 0.6,
            NetworkZone.QUARANTINE: 1.0
        }
        risk_score += zone_risk.get(network_zone, 0.5) * 0.2
        
        return min(risk_score, 1.0)
    
    async def _determine_trust_level(self, access_request: AccessRequest, risk_score: float) -> TrustLevel:
        """Determine trust level based on various factors."""
        # Start with context trust level
        trust_score = 0.5  # Neutral starting point
        
        # Adjust based on current trust level
        trust_values = {
            TrustLevel.UNTRUSTED: 0.0,
            TrustLevel.LOW: 0.2,
            TrustLevel.MEDIUM: 0.5,
            TrustLevel.HIGH: 0.8,
            TrustLevel.CRITICAL: 1.0
        }
        trust_score = trust_values.get(access_request.security_context.trust_level, 0.0)
        
        # Adjust based on risk
        trust_score -= risk_score * 0.5
        
        # Adjust based on behavioral score
        trust_score -= access_request.security_context.behavioral_score * 0.3
        
        # Adjust based on authentication factors
        auth_factors = access_request.security_context.authentication_factors
        if "mfa" in auth_factors or "totp" in auth_factors:
            trust_score += 0.2
        if "biometric" in auth_factors:
            trust_score += 0.1
        if "certificate" in auth_factors:
            trust_score += 0.15
        
        # Map trust score to trust level
        trust_score = max(0.0, min(1.0, trust_score))
        
        if trust_score >= 0.9:
            return TrustLevel.CRITICAL
        elif trust_score >= 0.7:
            return TrustLevel.HIGH
        elif trust_score >= 0.4:
            return TrustLevel.MEDIUM
        elif trust_score >= 0.2:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNTRUSTED
    
    async def _evaluate_policies(self, access_request: AccessRequest, trust_level: TrustLevel) -> Dict[str, Any]:
        """Evaluate zero-trust policies for access request."""
        policy_result = {
            "allowed": False,
            "conditions": [],
            "required_trust_level": self.min_trust_levels.get(access_request.resource_type, TrustLevel.MEDIUM),
            "current_trust_level": trust_level,
            "additional_requirements": []
        }
        
        # Check minimum trust level
        required_trust = policy_result["required_trust_level"]
        trust_order = [TrustLevel.UNTRUSTED, TrustLevel.LOW, TrustLevel.MEDIUM, TrustLevel.HIGH, TrustLevel.CRITICAL]
        
        if trust_order.index(trust_level) >= trust_order.index(required_trust):
            policy_result["allowed"] = True
        else:
            policy_result["conditions"].append(f"Insufficient trust level: {trust_level.value} < {required_trust.value}")
        
        # Check RBAC permissions
        try:
            rbac_allowed = rbac.check_permission(
                access_request.security_context.user_id,
                access_request.resource_type,
                access_request.action
            )
            
            if not rbac_allowed:
                policy_result["allowed"] = False
                policy_result["conditions"].append("RBAC permission denied")
        except Exception as e:
            policy_result["allowed"] = False
            policy_result["conditions"].append(f"RBAC check failed: {e}")
        
        # Additional requirements for sensitive resources
        if access_request.resource_type in [ResourceType.SYSTEM_CONFIGURATION, ResourceType.USER_MANAGEMENT]:
            if "mfa" not in access_request.security_context.authentication_factors:
                policy_result["additional_requirements"].append("mfa_required")
        
        return policy_result
    
    def _evaluate_network_access(self, access_request: AccessRequest) -> Dict[str, Any]:
        """Evaluate network-level access controls."""
        return self.network_manager.evaluate_network_access(
            source_ip=access_request.security_context.ip_address,
            destination_zone=NetworkZone.INTERNAL,  # Simplified
            port=443,  # HTTPS
            protocol="tcp"
        )
    
    def _make_access_decision(
        self,
        access_request: AccessRequest,
        identity_result: Dict[str, Any],
        risk_score: float,
        trust_level: TrustLevel,
        policy_result: Dict[str, Any],
        network_result: Dict[str, Any],
        behavioral_score: float
    ) -> Dict[str, Any]:
        """Make final access decision based on all evaluations."""
        
        decision = {
            "granted": False,
            "reason": "default_deny",
            "trust_level": trust_level.value,
            "risk_score": risk_score,
            "behavioral_score": behavioral_score,
            "conditions": [],
            "required_actions": [],
            "session_duration": 3600,  # 1 hour default
            "monitoring_level": "standard"
        }
        
        # Check all required conditions
        if not identity_result["verified"]:
            decision["reason"] = "identity_not_verified"
            decision["required_actions"].extend(identity_result["additional_factors_required"])
            return decision
        
        if not policy_result["allowed"]:
            decision["reason"] = "policy_violation"
            decision["conditions"].extend(policy_result["conditions"])
            decision["required_actions"].extend(policy_result["additional_requirements"])
            return decision
        
        if not network_result["allowed"]:
            decision["reason"] = "network_policy_violation"
            decision["conditions"].extend(network_result["conditions"])
            return decision
        
        # Check risk thresholds
        if risk_score > 0.8:
            decision["reason"] = "high_risk_score"
            decision["required_actions"].append("additional_verification")
            return decision
        
        if behavioral_score > 0.7:
            decision["reason"] = "anomalous_behavior"
            decision["required_actions"].append("behavioral_verification")
            decision["monitoring_level"] = "enhanced"
        
        # Grant access with conditions
        decision["granted"] = True
        decision["reason"] = "access_granted"
        
        # Adjust session duration based on risk and trust
        if risk_score > 0.5 or behavioral_score > 0.5:
            decision["session_duration"] = 1800  # 30 minutes for risky access
        
        if trust_level in [TrustLevel.HIGH, TrustLevel.CRITICAL]:
            decision["session_duration"] = 7200  # 2 hours for high trust
        
        # Set monitoring level
        if risk_score > 0.6 or behavioral_score > 0.6:
            decision["monitoring_level"] = "enhanced"
        
        return decision
    
    async def _log_access_decision(self, access_request: AccessRequest, access_decision: Dict[str, Any]):
        """Log access decision for audit purposes."""
        log_entry = {
            "timestamp": time.time(),
            "request_id": access_request.request_id,
            "user_id": access_request.security_context.user_id,
            "resource_type": access_request.resource_type.value,
            "action": access_request.action.value,
            "resource_id": access_request.resource_id,
            "granted": access_decision["granted"],
            "reason": access_decision["reason"],
            "trust_level": access_decision["trust_level"],
            "risk_score": access_decision["risk_score"],
            "behavioral_score": access_decision["behavioral_score"],
            "ip_address": access_request.security_context.ip_address,
            "user_agent": access_request.security_context.user_agent
        }
        
        # Use existing audit system
        rbac.audit_logger.log_event(
            user_id=access_request.security_context.user_id,
            action=f"zero_trust_{access_request.action.value}",
            resource=f"{access_request.resource_type.value}:{access_request.resource_id}",
            result="success" if access_decision["granted"] else "denied",
            ip_address=access_request.security_context.ip_address,
            details=access_decision
        )
    
    async def _apply_access_controls(self, access_request: AccessRequest, access_decision: Dict[str, Any]):
        """Apply access controls for granted requests."""
        # Update session with new trust level and expiration
        session_context = access_request.security_context
        session_context.trust_level = TrustLevel(access_decision["trust_level"])
        
        # Calculate session expiration
        session_expiry = time.time() + access_decision["session_duration"]
        
        # Store active session
        self.active_sessions[session_context.session_id] = session_context
        
        # Apply monitoring based on decision
        if access_decision["monitoring_level"] == "enhanced":
            await self._enable_enhanced_monitoring(session_context)
    
    async def _apply_denial_actions(self, access_request: AccessRequest, access_decision: Dict[str, Any]):
        """Apply denial actions for rejected requests."""
        # Increment failed access attempts
        user_id = access_request.security_context.user_id
        failed_attempts = getattr(self, f"_failed_attempts_{user_id}", 0) + 1
        setattr(self, f"_failed_attempts_{user_id}", failed_attempts)
        
        # Lock account after too many failures
        if failed_attempts >= 5:
            await self._lock_user_account(user_id, "too_many_failed_access_attempts")
        
        # Quarantine suspicious IPs
        if access_decision["risk_score"] > 0.8:
            self.network_manager.quarantine_ip(
                access_request.security_context.ip_address,
                f"High risk access attempt: {access_decision['reason']}"
            )
    
    async def _enable_enhanced_monitoring(self, security_context: SecurityContext):
        """Enable enhanced monitoring for high-risk sessions."""
        logging.info(f"Enhanced monitoring enabled for user {security_context.user_id}")
        # Implementation would integrate with monitoring systems
    
    async def _lock_user_account(self, user_id: str, reason: str):
        """Lock user account due to security concerns."""
        logging.warning(f"User account locked: {user_id} - {reason}")
        # Implementation would integrate with user management system
    
    def detect_threats(self, security_events: List[Dict[str, Any]]) -> List[ThreatEvent]:
        """Detect security threats from events."""
        threats = []
        
        for event in security_events:
            # Implement threat detection algorithms
            threat = self._analyze_security_event(event)
            if threat:
                threats.append(threat)
                self.threat_events.append(threat)
        
        return threats
    
    def _analyze_security_event(self, event: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Analyze individual security event for threats."""
        # Simplified threat detection - in practice, this would be more sophisticated
        
        # Detect brute force attacks
        if event.get("event_type") == "login_failure":
            # Check for multiple failures from same IP
            source_ip = event.get("ip_address")
            recent_failures = [
                e for e in self.threat_events[-100:]  # Last 100 events
                if e.source_ip == source_ip and 
                time.time() - e.timestamp < 300  # Last 5 minutes
            ]
            
            if len(recent_failures) >= 3:
                return ThreatEvent(
                    event_id=f"threat_{int(time.time())}_{secrets.token_hex(4)}",
                    timestamp=time.time(),
                    threat_type="brute_force_attack",
                    threat_level=ThreatLevel.HIGH,
                    source_ip=source_ip,
                    target_resource="authentication_system",
                    description=f"Multiple login failures from {source_ip}",
                    indicators=["repeated_login_failures", "same_source_ip"],
                    mitigation_actions=["quarantine_ip", "alert_security_team"]
                )
        
        return None
    
    def get_zero_trust_metrics(self) -> Dict[str, Any]:
        """Get comprehensive zero-trust security metrics."""
        current_time = time.time()
        
        # Calculate trust distribution
        trust_distribution = {}
        for session in self.active_sessions.values():
            trust_level = session.trust_level.value
            trust_distribution[trust_level] = trust_distribution.get(trust_level, 0) + 1
        
        # Calculate threat metrics
        recent_threats = [
            t for t in self.threat_events
            if current_time - t.timestamp < 3600  # Last hour
        ]
        
        threat_levels = {}
        for threat in recent_threats:
            level = threat.threat_level.value
            threat_levels[level] = threat_levels.get(level, 0) + 1
        
        return {
            "timestamp": current_time,
            "active_sessions": len(self.active_sessions),
            "trust_distribution": trust_distribution,
            "recent_threats": {
                "total": len(recent_threats),
                "by_level": threat_levels,
                "resolved": len([t for t in recent_threats if t.resolved])
            },
            "network_zones": {
                zone.value: len(ips) for zone, ips in self.network_manager.zone_rules.items()
            },
            "behavioral_analysis": {
                "users_with_baselines": len(self.behavioral_analyzer.user_baselines),
                "total_activity_patterns": sum(
                    len(patterns) for patterns in self.behavioral_analyzer.activity_patterns.values()
                )
            },
            "encryption_status": {
                "keys_rotated": len(self.encryption_manager.key_history),
                "last_rotation": self.encryption_manager.last_key_rotation,
                "rotation_due": self.encryption_manager.should_rotate_keys()
            }
        }


# Global zero-trust engine instance
zero_trust = ZeroTrustEngine()


# Convenience functions for zero-trust operations
async def evaluate_zero_trust_access(
    user_id: str,
    resource_type: ResourceType,
    action: ActionType,
    resource_id: str = "",
    ip_address: str = "127.0.0.1",
    session_token: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function for zero-trust access evaluation."""
    
    # Create security context
    security_context = SecurityContext(
        user_id=user_id,
        session_id=session_token or f"session_{secrets.token_hex(8)}",
        device_id=f"device_{hashlib.sha256(f'{user_id}:{ip_address}'.encode()).hexdigest()[:12]}",
        ip_address=ip_address,
        user_agent=additional_context.get("user_agent", "Unknown") if additional_context else "Unknown",
        timestamp=time.time(),
        trust_level=TrustLevel.LOW,  # Start with low trust
        network_zone=zero_trust.network_manager.classify_ip_zone(ip_address),
        authentication_factors=additional_context.get("auth_factors", []) if additional_context else []
    )
    
    # Create access request
    access_request = AccessRequest(
        request_id=f"req_{int(time.time())}_{secrets.token_hex(6)}",
        security_context=security_context,
        resource_type=resource_type,
        action=action,
        resource_id=resource_id,
        requested_permissions=[action.value],
        additional_context=additional_context or {}
    )
    
    return await zero_trust.evaluate_access_request(access_request)


def get_zero_trust_status() -> Dict[str, Any]:
    """Get zero-trust system status."""
    return zero_trust.get_zero_trust_metrics()


def secure_encrypt_data(data: Union[str, bytes]) -> str:
    """Encrypt data using zero-trust encryption."""
    return zero_trust.encryption_manager.encrypt_data(data)


def secure_decrypt_data(encrypted_data: str) -> bytes:
    """Decrypt data using zero-trust encryption."""
    return zero_trust.encryption_manager.decrypt_data(encrypted_data)