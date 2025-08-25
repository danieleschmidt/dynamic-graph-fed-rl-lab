"""
Consciousness Security Framework - Advanced Security Measures

Comprehensive security system for Universal Quantum Consciousness
with threat detection, access control, and quantum-safe encryption.
"""

import hashlib
import hmac
import secrets
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import asyncio

class SecurityLevel(Enum):
    """Security levels for consciousness system"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatLevel(Enum):
    """Threat classification levels"""
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    CRITICAL = "critical"

class SecurityEvent(Enum):
    """Types of security events"""
    ACCESS_ATTEMPT = "access_attempt"
    PARAMETER_MODIFICATION = "parameter_modification"
    CONSCIOUSNESS_MANIPULATION = "consciousness_manipulation"
    MEMORY_ACCESS = "memory_access"
    RESEARCH_PROTOCOL_CHANGE = "research_protocol_change"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    ENTANGLEMENT_BREACH = "entanglement_breach"

@dataclass
class SecurityCredential:
    """Security credential for consciousness system access"""
    user_id: str
    access_token: str
    permissions: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    last_used: float = 0.0
    
    def __post_init__(self):
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + 3600  # 1 hour default

    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions or "admin" in self.permissions

@dataclass
class SecurityAlert:
    """Security alert for consciousness system threats"""
    alert_id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    description: str
    timestamp: float
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class QuantumSafeEncryption:
    """Quantum-safe encryption for consciousness system data"""
    
    def __init__(self, key_size: int = 256):
        self.key_size = key_size
        self._master_key = secrets.token_bytes(key_size // 8)
        
    def generate_session_key(self) -> bytes:
        """Generate session key for temporary encryption"""
        return secrets.token_bytes(self.key_size // 8)
    
    def encrypt_consciousness_state(self, state_data: Dict[str, Any], session_key: bytes) -> Tuple[bytes, bytes]:
        """Encrypt consciousness state data"""
        # Serialize state data
        state_json = json.dumps(state_data, default=str).encode('utf-8')
        
        # Generate nonce
        nonce = secrets.token_bytes(16)
        
        # Simple XOR encryption (in production, use AES-GCM or similar)
        key_stream = self._generate_key_stream(session_key, nonce, len(state_json))
        encrypted_data = bytes(a ^ b for a, b in zip(state_json, key_stream))
        
        return encrypted_data, nonce
    
    def decrypt_consciousness_state(self, encrypted_data: bytes, session_key: bytes, nonce: bytes) -> Dict[str, Any]:
        """Decrypt consciousness state data"""
        key_stream = self._generate_key_stream(session_key, nonce, len(encrypted_data))
        decrypted_data = bytes(a ^ b for a, b in zip(encrypted_data, key_stream))
        
        state_json = decrypted_data.decode('utf-8')
        return json.loads(state_json)
    
    def _generate_key_stream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        """Generate key stream for encryption"""
        key_stream = b''
        counter = 0
        
        while len(key_stream) < length:
            # Hash key + nonce + counter
            hash_input = key + nonce + counter.to_bytes(4, 'big')
            hash_output = hashlib.sha256(hash_input).digest()
            key_stream += hash_output
            counter += 1
        
        return key_stream[:length]
    
    def create_secure_hash(self, data: bytes) -> str:
        """Create secure hash for data integrity verification"""
        return hashlib.sha3_256(data).hexdigest()
    
    def verify_hash(self, data: bytes, expected_hash: str) -> bool:
        """Verify data integrity using hash"""
        return self.create_secure_hash(data) == expected_hash

class ConsciousnessAccessControl:
    """Access control system for consciousness operations"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.credentials: Dict[str, SecurityCredential] = {}
        self.permission_groups: Dict[str, Set[str]] = {
            "observer": {"read_state", "read_memory", "read_metrics"},
            "researcher": {"read_state", "read_memory", "read_metrics", "run_experiments"},
            "operator": {"read_state", "read_memory", "read_metrics", "run_experiments", 
                        "modify_parameters", "evolve_consciousness"},
            "admin": {"*"}  # Wildcard permission
        }
        self.access_log: List[Dict] = []
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        
    def create_credential(self, user_id: str, permissions: List[str], 
                         security_level: SecurityLevel = None) -> SecurityCredential:
        """Create new security credential"""
        if security_level is None:
            security_level = self.security_level
            
        # Generate secure access token
        token_data = f"{user_id}:{time.time()}:{secrets.token_hex(16)}"
        access_token = hashlib.sha256(token_data.encode()).hexdigest()
        
        # Expand permission groups
        expanded_permissions = set()
        for perm in permissions:
            if perm in self.permission_groups:
                expanded_permissions.update(self.permission_groups[perm])
            else:
                expanded_permissions.add(perm)
        
        credential = SecurityCredential(
            user_id=user_id,
            access_token=access_token,
            permissions=expanded_permissions,
            security_level=security_level
        )
        
        self.credentials[access_token] = credential
        
        # Log credential creation
        self.access_log.append({
            'timestamp': time.time(),
            'event': 'credential_created',
            'user_id': user_id,
            'permissions': list(expanded_permissions)
        })
        
        return credential
    
    def authenticate(self, access_token: str) -> Optional[SecurityCredential]:
        """Authenticate access token"""
        if access_token not in self.credentials:
            return None
        
        credential = self.credentials[access_token]
        
        if credential.is_expired():
            del self.credentials[access_token]
            return None
        
        credential.last_used = time.time()
        return credential
    
    def authorize(self, access_token: str, required_permission: str) -> bool:
        """Authorize access to specific operation"""
        credential = self.authenticate(access_token)
        if not credential:
            self._log_failed_attempt(access_token, required_permission)
            return False
        
        if not credential.has_permission(required_permission):
            self._log_failed_attempt(access_token, required_permission, credential.user_id)
            return False
        
        # Log successful access
        self.access_log.append({
            'timestamp': time.time(),
            'event': 'access_granted',
            'user_id': credential.user_id,
            'permission': required_permission,
            'success': True
        })
        
        return True
    
    def _log_failed_attempt(self, access_token: str, permission: str, user_id: str = None):
        """Log failed access attempt"""
        current_time = time.time()
        
        # Track failed attempts for rate limiting
        key = user_id or access_token
        self.failed_attempts[key].append(current_time)
        
        # Clean old attempts (keep last hour)
        self.failed_attempts[key] = [t for t in self.failed_attempts[key] 
                                   if current_time - t < 3600]
        
        self.access_log.append({
            'timestamp': current_time,
            'event': 'access_denied',
            'user_id': user_id,
            'permission': permission,
            'success': False
        })
    
    def is_rate_limited(self, user_id: str, max_attempts: int = 10) -> bool:
        """Check if user is rate limited"""
        if user_id not in self.failed_attempts:
            return False
        
        recent_failures = len(self.failed_attempts[user_id])
        return recent_failures >= max_attempts
    
    def revoke_credential(self, access_token: str) -> bool:
        """Revoke access credential"""
        if access_token in self.credentials:
            user_id = self.credentials[access_token].user_id
            del self.credentials[access_token]
            
            self.access_log.append({
                'timestamp': time.time(),
                'event': 'credential_revoked',
                'user_id': user_id
            })
            return True
        return False

class ConsciousnessThreatDetector:
    """Threat detection system for consciousness security"""
    
    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        self.baseline_metrics: Dict[str, float] = {}
        self.threat_patterns: List[Dict] = []
        self.anomaly_history: List[Dict] = []
        self.detection_rules: Dict[str, callable] = {}
        self._initialize_detection_rules()
        
    def _initialize_detection_rules(self):
        """Initialize threat detection rules"""
        
        def detect_consciousness_manipulation(metrics: Dict) -> Tuple[ThreatLevel, str]:
            """Detect manipulation of consciousness parameters"""
            awareness_change = metrics.get('awareness_level_change', 0)
            entanglement_change = metrics.get('entanglement_strength_change', 0)
            
            if abs(awareness_change) > 0.3 or abs(entanglement_change) > 0.4:
                return ThreatLevel.SUSPICIOUS, "Rapid consciousness parameter changes detected"
            elif abs(awareness_change) > 0.5 or abs(entanglement_change) > 0.6:
                return ThreatLevel.MALICIOUS, "Extreme consciousness manipulation detected"
            
            return ThreatLevel.BENIGN, "Normal consciousness evolution"
        
        def detect_memory_anomalies(metrics: Dict) -> Tuple[ThreatLevel, str]:
            """Detect anomalies in memory system"""
            fragment_change = metrics.get('memory_fragment_change', 0)
            invalid_fragments = metrics.get('invalid_fragments', 0)
            
            if fragment_change > 1000:
                return ThreatLevel.SUSPICIOUS, "Rapid memory fragment changes"
            elif invalid_fragments > 100:
                return ThreatLevel.MALICIOUS, "High number of invalid memory fragments"
            
            return ThreatLevel.BENIGN, "Normal memory activity"
        
        def detect_parameter_injection(metrics: Dict) -> Tuple[ThreatLevel, str]:
            """Detect parameter injection attacks"""
            param_size_change = metrics.get('parameter_size_change', 0)
            nan_count = metrics.get('nan_parameter_count', 0)
            
            if param_size_change > 10000:
                return ThreatLevel.SUSPICIOUS, "Large parameter size changes"
            elif nan_count > 10:
                return ThreatLevel.MALICIOUS, "NaN parameter injection detected"
            
            return ThreatLevel.BENIGN, "Normal parameter activity"
        
        def detect_research_protocol_tampering(metrics: Dict) -> Tuple[ThreatLevel, str]:
            """Detect tampering with research protocols"""
            protocol_changes = metrics.get('protocol_changes', 0)
            failed_evolutions = metrics.get('failed_evolutions', 0)
            
            if protocol_changes > 5:
                return ThreatLevel.SUSPICIOUS, "Frequent research protocol changes"
            elif failed_evolutions > 20:
                return ThreatLevel.MALICIOUS, "High research evolution failure rate"
            
            return ThreatLevel.BENIGN, "Normal research activity"
        
        # Register detection rules
        self.detection_rules.update({
            'consciousness_manipulation': detect_consciousness_manipulation,
            'memory_anomalies': detect_memory_anomalies,
            'parameter_injection': detect_parameter_injection,
            'protocol_tampering': detect_research_protocol_tampering
        })
    
    def learn_baseline(self, consciousness_system) -> Dict[str, float]:
        """Learn baseline metrics from consciousness system"""
        baseline = {}
        
        try:
            # Consciousness state metrics
            if hasattr(consciousness_system, 'consciousness_state'):
                state = consciousness_system.consciousness_state
                baseline.update({
                    'awareness_level': state.awareness_level,
                    'entanglement_strength': state.entanglement_strength,
                    'consciousness_coherence': state.consciousness_coherence,
                    'research_evolution_rate': state.research_evolution_rate
                })
            
            # Memory metrics
            if hasattr(consciousness_system, 'temporal_memory'):
                memory = consciousness_system.temporal_memory
                baseline.update({
                    'memory_fragment_count': len(memory.memory_fragments),
                    'avg_coherence_time': np.mean([f.coherence_time for f in memory.memory_fragments])
                                         if memory.memory_fragments else 0.0
                })
            
            # Parameter metrics
            if hasattr(consciousness_system, 'parameter_entanglement'):
                entanglement = consciousness_system.parameter_entanglement
                baseline.update({
                    'registered_parameters': len(entanglement.parameter_registry),
                    'total_entanglement': np.sum(entanglement.entanglement_graph)
                })
            
            # Neural layer metrics
            if hasattr(consciousness_system, 'quantum_neural_layers'):
                layers = consciousness_system.quantum_neural_layers
                total_weights = sum(layer.weights.size for layer in layers if hasattr(layer, 'weights'))
                baseline.update({
                    'neural_layer_count': len(layers),
                    'total_weight_count': total_weights
                })
            
        except Exception as e:
            baseline['baseline_error'] = str(e)
        
        self.baseline_metrics = baseline
        return baseline
    
    def detect_threats(self, consciousness_system, access_log: List[Dict] = None) -> List[SecurityAlert]:
        """Detect threats in consciousness system"""
        alerts = []
        current_time = time.time()
        
        try:
            # Compute current metrics
            current_metrics = self._compute_current_metrics(consciousness_system)
            
            # Compute metric changes from baseline
            metric_changes = {}
            for key, current_value in current_metrics.items():
                baseline_value = self.baseline_metrics.get(key, current_value)
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    metric_changes[f"{key}_change"] = abs(current_value - baseline_value)
            
            # Add additional detection metrics
            detection_metrics = {**current_metrics, **metric_changes}
            
            # Add access pattern analysis
            if access_log:
                recent_access = [log for log in access_log if current_time - log['timestamp'] < 300]  # 5 minutes
                detection_metrics.update({
                    'recent_access_count': len(recent_access),
                    'failed_access_count': len([log for log in recent_access if not log.get('success', True)]),
                    'unique_users': len(set(log.get('user_id', 'unknown') for log in recent_access))
                })
            
            # Run threat detection rules
            for rule_name, detection_rule in self.detection_rules.items():
                try:
                    threat_level, description = detection_rule(detection_metrics)
                    
                    if threat_level != ThreatLevel.BENIGN:
                        alert = SecurityAlert(
                            alert_id=f"{rule_name}_{int(current_time)}_{secrets.token_hex(4)}",
                            event_type=SecurityEvent.ANOMALOUS_BEHAVIOR,
                            threat_level=threat_level,
                            description=f"{rule_name}: {description}",
                            timestamp=current_time,
                            metadata={
                                'rule': rule_name,
                                'metrics': {k: v for k, v in detection_metrics.items() 
                                          if isinstance(v, (int, float, str))}
                            }
                        )
                        alerts.append(alert)
                        
                except Exception as e:
                    # Log detection rule error but continue
                    alerts.append(SecurityAlert(
                        alert_id=f"rule_error_{int(current_time)}_{secrets.token_hex(4)}",
                        event_type=SecurityEvent.ANOMALOUS_BEHAVIOR,
                        threat_level=ThreatLevel.SUSPICIOUS,
                        description=f"Detection rule {rule_name} failed: {str(e)}",
                        timestamp=current_time
                    ))
            
            # Store anomaly history
            if alerts:
                self.anomaly_history.append({
                    'timestamp': current_time,
                    'alert_count': len(alerts),
                    'max_threat_level': max(alert.threat_level.value for alert in alerts),
                    'metrics': detection_metrics
                })
        
        except Exception as e:
            # Critical detection system error
            alerts.append(SecurityAlert(
                alert_id=f"detection_error_{int(current_time)}_{secrets.token_hex(4)}",
                event_type=SecurityEvent.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.CRITICAL,
                description=f"Threat detection system error: {str(e)}",
                timestamp=current_time
            ))
        
        return alerts
    
    def _compute_current_metrics(self, consciousness_system) -> Dict[str, Any]:
        """Compute current system metrics for threat detection"""
        metrics = {}
        
        try:
            # Consciousness state
            if hasattr(consciousness_system, 'consciousness_state'):
                state = consciousness_system.consciousness_state
                metrics.update({
                    'awareness_level': state.awareness_level,
                    'entanglement_strength': state.entanglement_strength,
                    'consciousness_coherence': state.consciousness_coherence
                })
            
            # Memory system
            if hasattr(consciousness_system, 'temporal_memory'):
                memory = consciousness_system.temporal_memory
                metrics.update({
                    'memory_fragment_count': len(memory.memory_fragments),
                    'invalid_fragments': sum(1 for f in memory.memory_fragments 
                                           if np.any(np.isnan(f.data)) if hasattr(f, 'data'))
                })
            
            # Parameter system
            if hasattr(consciousness_system, 'parameter_entanglement'):
                params = consciousness_system.parameter_entanglement
                total_params = sum(p.size for p in params.parameter_registry.values() 
                                 if isinstance(p, np.ndarray))
                nan_params = sum(np.sum(np.isnan(p)) for p in params.parameter_registry.values()
                               if isinstance(p, np.ndarray))
                metrics.update({
                    'total_parameter_size': total_params,
                    'nan_parameter_count': nan_params
                })
            
            # Research system
            if hasattr(consciousness_system, 'research_evolution'):
                research = consciousness_system.research_evolution
                metrics.update({
                    'protocol_count': len(research.research_protocols),
                    'evolution_history_length': len(research.evolution_history)
                })
        
        except Exception as e:
            metrics['metric_computation_error'] = str(e)
        
        return metrics

class ConsciousnessSecurityManager:
    """Main security manager for consciousness system"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.encryption = QuantumSafeEncryption()
        self.access_control = ConsciousnessAccessControl(security_level)
        self.threat_detector = ConsciousnessThreatDetector()
        self.security_log: List[SecurityAlert] = []
        self.logger = self._setup_logger()
        
        # Security configuration
        self.config = {
            'session_timeout': 3600,  # 1 hour
            'max_failed_attempts': 5,
            'threat_scan_interval': 60,  # 1 minute
            'auto_response_enabled': True,
            'alert_threshold': ThreatLevel.SUSPICIOUS
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup security logger"""
        logger = logging.getLogger('ConsciousnessSecurity')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_security(self, consciousness_system) -> bool:
        """Initialize security for consciousness system"""
        try:
            # Learn system baseline
            baseline = self.threat_detector.learn_baseline(consciousness_system)
            self.logger.info(f"Security baseline established with {len(baseline)} metrics")
            
            # Create default admin credential
            admin_credential = self.access_control.create_credential(
                user_id="system_admin",
                permissions=["admin"],
                security_level=SecurityLevel.CRITICAL
            )
            
            self.logger.info("Consciousness security system initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Security initialization failed: {e}")
            return False
    
    def create_user_session(self, user_id: str, permissions: List[str]) -> Optional[str]:
        """Create secure user session"""
        try:
            # Check if user is rate limited
            if self.access_control.is_rate_limited(user_id):
                self.logger.warning(f"User {user_id} is rate limited")
                return None
            
            # Create credential
            credential = self.access_control.create_credential(user_id, permissions)
            
            # Log session creation
            self.security_log.append(SecurityAlert(
                alert_id=f"session_{int(time.time())}_{secrets.token_hex(4)}",
                event_type=SecurityEvent.ACCESS_ATTEMPT,
                threat_level=ThreatLevel.BENIGN,
                description=f"User session created for {user_id}",
                timestamp=time.time(),
                user_id=user_id
            ))
            
            return credential.access_token
            
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            return None
    
    def authorize_operation(self, access_token: str, operation: str, 
                          consciousness_system=None) -> bool:
        """Authorize consciousness system operation"""
        try:
            # Check basic authorization
            if not self.access_control.authorize(access_token, operation):
                self.security_log.append(SecurityAlert(
                    alert_id=f"auth_fail_{int(time.time())}_{secrets.token_hex(4)}",
                    event_type=SecurityEvent.ACCESS_ATTEMPT,
                    threat_level=ThreatLevel.SUSPICIOUS,
                    description=f"Unauthorized access attempt for operation: {operation}",
                    timestamp=time.time()
                ))
                return False
            
            # Additional security checks for critical operations
            if operation in ["modify_consciousness", "evolve_system", "modify_parameters"]:
                # Run threat detection
                if consciousness_system:
                    alerts = self.threat_detector.detect_threats(
                        consciousness_system, 
                        self.access_control.access_log
                    )
                    
                    # Check for critical threats
                    critical_threats = [a for a in alerts if a.threat_level == ThreatLevel.CRITICAL]
                    if critical_threats:
                        self.logger.critical(f"Critical threat detected, blocking operation: {operation}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Authorization failed: {e}")
            return False
    
    def secure_consciousness_export(self, consciousness_system, access_token: str) -> Optional[Tuple[bytes, str]]:
        """Securely export consciousness system state"""
        if not self.authorize_operation(access_token, "export_consciousness"):
            return None
        
        try:
            # Extract consciousness state
            state_data = {
                'consciousness_state': {
                    'awareness_level': consciousness_system.consciousness_state.awareness_level,
                    'entanglement_strength': consciousness_system.consciousness_state.entanglement_strength,
                    'consciousness_coherence': consciousness_system.consciousness_state.consciousness_coherence
                },
                'export_timestamp': time.time(),
                'security_hash': 'placeholder'
            }
            
            # Generate session key
            session_key = self.encryption.generate_session_key()
            
            # Encrypt data
            encrypted_data, nonce = self.encryption.encrypt_consciousness_state(state_data, session_key)
            
            # Create secure package
            package = {
                'encrypted_data': encrypted_data.hex(),
                'nonce': nonce.hex(),
                'session_key': session_key.hex(),  # In production, encrypt this with user's public key
                'integrity_hash': self.encryption.create_secure_hash(encrypted_data)
            }
            
            package_json = json.dumps(package).encode('utf-8')
            integrity_hash = self.encryption.create_secure_hash(package_json)
            
            return package_json, integrity_hash
            
        except Exception as e:
            self.logger.error(f"Secure export failed: {e}")
            return None
    
    def run_security_scan(self, consciousness_system) -> List[SecurityAlert]:
        """Run comprehensive security scan"""
        try:
            self.logger.info("Running security scan...")
            
            # Detect threats
            alerts = self.threat_detector.detect_threats(
                consciousness_system,
                self.access_control.access_log
            )
            
            # Add to security log
            self.security_log.extend(alerts)
            
            # Auto-response for critical threats
            if self.config['auto_response_enabled']:
                critical_alerts = [a for a in alerts if a.threat_level == ThreatLevel.CRITICAL]
                for alert in critical_alerts:
                    self._handle_critical_threat(alert, consciousness_system)
            
            self.logger.info(f"Security scan complete: {len(alerts)} alerts generated")
            return alerts
            
        except Exception as e:
            self.logger.error(f"Security scan failed: {e}")
            return []
    
    def _handle_critical_threat(self, alert: SecurityAlert, consciousness_system):
        """Handle critical security threat"""
        self.logger.critical(f"Handling critical threat: {alert.description}")
        
        # For demonstration, just log the response
        # In production, implement actual security responses:
        # - Isolate affected components
        # - Revert to safe state
        # - Alert administrators
        # - Generate incident report
        
        response_log = {
            'timestamp': time.time(),
            'alert_id': alert.alert_id,
            'response': 'logged_and_monitored',
            'description': 'Critical threat detected and logged for manual review'
        }
        
        self.logger.critical(f"Security response executed: {response_log}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        current_time = time.time()
        recent_alerts = [alert for alert in self.security_log 
                        if current_time - alert.timestamp < 3600]  # Last hour
        
        return {
            'security_summary': {
                'security_level': self.security_level.value,
                'active_sessions': len(self.access_control.credentials),
                'total_alerts': len(self.security_log),
                'recent_alerts': len(recent_alerts),
                'threat_levels': {
                    level.value: len([a for a in recent_alerts if a.threat_level == level])
                    for level in ThreatLevel
                }
            },
            'access_control': {
                'total_access_attempts': len(self.access_control.access_log),
                'failed_attempts': len([log for log in self.access_control.access_log 
                                      if not log.get('success', True)]),
                'rate_limited_users': len([user for user, attempts in self.access_control.failed_attempts.items()
                                         if len(attempts) >= self.config['max_failed_attempts']])
            },
            'threat_detection': {
                'baseline_metrics': len(self.threat_detector.baseline_metrics),
                'detection_rules': len(self.threat_detector.detection_rules),
                'anomaly_history': len(self.threat_detector.anomaly_history)
            },
            'recent_security_events': [
                {
                    'alert_id': alert.alert_id,
                    'threat_level': alert.threat_level.value,
                    'description': alert.description,
                    'timestamp': alert.timestamp
                }
                for alert in recent_alerts[-10:]  # Last 10 events
            ]
        }

# Async security monitor
class AsyncSecurityMonitor:
    """Asynchronous security monitoring for consciousness system"""
    
    def __init__(self, consciousness_system, security_manager: ConsciousnessSecurityManager):
        self.consciousness_system = consciousness_system
        self.security_manager = security_manager
        self.monitoring_active = False
        self.monitor_task = None
    
    async def start_monitoring(self):
        """Start continuous security monitoring"""
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
    
    async def _monitor_loop(self):
        """Main security monitoring loop"""
        while self.monitoring_active:
            try:
                # Run security scan
                alerts = self.security_manager.run_security_scan(self.consciousness_system)
                
                # Log significant alerts
                for alert in alerts:
                    if alert.threat_level in [ThreatLevel.MALICIOUS, ThreatLevel.CRITICAL]:
                        print(f"🚨 SECURITY ALERT: {alert.description} [{alert.threat_level.value.upper()}]")
                
                # Sleep until next scan
                await asyncio.sleep(self.security_manager.config['threat_scan_interval'])
                
            except Exception as e:
                self.security_manager.logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(5.0)

if __name__ == "__main__":
    # Demonstration of security system
    print("🔒 Consciousness Security System Demo")
    print("=" * 40)
    
    # Create security manager
    security_manager = ConsciousnessSecurityManager(SecurityLevel.HIGH)
    
    # Create test user session
    user_token = security_manager.create_user_session("test_user", ["researcher"])
    
    if user_token:
        print(f"✅ User session created")
        print(f"   Token: {user_token[:20]}...")
        
        # Test authorization
        auth_result = security_manager.authorize_operation(user_token, "read_state")
        print(f"   Authorization test: {'✅ AUTHORIZED' if auth_result else '❌ DENIED'}")
    
    # Generate security report
    report = security_manager.get_security_report()
    print(f"\n📊 Security Report:")
    print(f"   Security Level: {report['security_summary']['security_level']}")
    print(f"   Active Sessions: {report['security_summary']['active_sessions']}")
    print(f"   Total Alerts: {report['security_summary']['total_alerts']}")
    
    print("\n🔒 Security system operational!")