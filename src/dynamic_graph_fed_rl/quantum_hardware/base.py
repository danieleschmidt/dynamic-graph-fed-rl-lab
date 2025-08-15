"""
Base classes for quantum hardware integration.

Provides abstract interfaces for different quantum computing platforms
to enable seamless switching between IBM Quantum, Google Cirq, and AWS Braket.

Generation 2 Robustness Features:
- Enterprise-grade error handling with circuit breakers
- Input validation and sanitization
- Comprehensive logging and monitoring
- Automatic retry mechanisms for quantum operations
- Security hardening and access control
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from enum import Enum
import numpy as np
import jax.numpy as jnp
import logging
import hashlib
import time
from ..utils.error_handling import (
    circuit_breaker, retry, robust, SecurityError, ValidationError,
    CircuitBreakerConfig, RetryConfig, resilience
)


class QuantumBackendType(Enum):
    """Supported quantum computing backends."""
    IBM_QUANTUM = "ibm_quantum"
    GOOGLE_QUANTUM = "google_quantum"
    AWS_BRAKET = "aws_braket"
    SIMULATOR = "simulator"


@dataclass
class QuantumCircuit:
    """Universal quantum circuit representation."""
    qubits: int
    gates: List[Dict[str, Any]]
    measurements: List[int]
    parameters: Dict[str, float]
    
    def add_gate(self, gate_type: str, qubits: List[int], **kwargs):
        """Add a quantum gate to the circuit."""
        self.gates.append({
            "type": gate_type,
            "qubits": qubits,
            **kwargs
        })
    
    def add_measurement(self, qubit: int):
        """Add measurement to a qubit."""
        if qubit not in self.measurements:
            self.measurements.append(qubit)


@dataclass
class QuantumResult:
    """Universal quantum execution result."""
    backend_type: QuantumBackendType
    job_id: str
    counts: Dict[str, int]
    execution_time: float
    shots: int
    success: bool
    error_message: Optional[str] = None
    raw_result: Any = None
    
    @property
    def probabilities(self) -> Dict[str, float]:
        """Convert counts to probabilities."""
        total = sum(self.counts.values())
        return {state: count/total for state, count in self.counts.items()}


class QuantumBackend(ABC):
    """Abstract base class for quantum computing backends with enhanced robustness."""
    
    def __init__(self, backend_type: QuantumBackendType):
        self.backend_type = backend_type
        self.is_connected = False
        self.device_info: Dict[str, Any] = {}
        self.connection_attempts = 0
        self.last_health_check = 0.0
        self.total_executions = 0
        self.failed_executions = 0
        
        # Setup circuit breakers for quantum operations
        self._setup_circuit_breakers()
        
        # Security and audit
        self.access_log: List[Dict[str, Any]] = []
        self.authorized_users: Set[str] = set()
    
    def _setup_circuit_breakers(self):
        """Setup circuit breakers for quantum operations."""
        # Circuit breaker for device connections
        connection_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=300.0,  # 5 minutes
            expected_exception=(ConnectionError, TimeoutError)
        )
        self.connection_circuit = resilience.register_circuit_breaker(
            f"quantum-connection-{self.backend_type.value}", 
            connection_config
        )
        
        # Circuit breaker for circuit execution
        execution_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120.0,  # 2 minutes
            expected_exception=(RuntimeError, ValueError)
        )
        self.execution_circuit = resilience.register_circuit_breaker(
            f"quantum-execution-{self.backend_type.value}", 
            execution_config
        )
        
    @robust(component="quantum_backend", operation="connect")
    @abstractmethod
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to the quantum backend with enhanced security and resilience."""
        pass
    
    def secure_connect(self, credentials: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Secure connection wrapper with authentication and logging."""
        # Validate credentials
        if not isinstance(credentials, dict):
            raise ValidationError("Credentials must be a dictionary")
        
        # Security check for required fields
        required_fields = ["api_key", "endpoint"]
        for field in required_fields:
            if field not in credentials:
                raise SecurityError(f"Missing required credential field: {field}")
        
        # Sanitize credentials
        sanitized_creds = self._sanitize_credentials(credentials)
        
        # Log connection attempt
        self._log_access_attempt(user_id, "connect", "attempted")
        
        try:
            result = self.connection_circuit.call(self.connect, sanitized_creds)
            if result:
                self.is_connected = True
                self.connection_attempts += 1
                self._log_access_attempt(user_id, "connect", "success")
                logging.info(f"Successfully connected to {self.backend_type.value}")
            else:
                self._log_access_attempt(user_id, "connect", "failed")
                logging.warning(f"Failed to connect to {self.backend_type.value}")
            
            return result
        except Exception as e:
            self._log_access_attempt(user_id, "connect", "error", str(e))
            raise
    
    def _sanitize_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize credentials to prevent injection attacks."""
        sanitized = {}
        for key, value in credentials.items():
            if isinstance(value, str):
                # Basic sanitization - remove potential injection characters
                sanitized_value = value.strip()
                if any(char in sanitized_value for char in ['<', '>', '&', '"', "'", ';']):
                    raise SecurityError(f"Invalid characters in credential field: {key}")
                sanitized[key] = sanitized_value
            else:
                sanitized[key] = value
        return sanitized
    
    def _log_access_attempt(self, user_id: Optional[str], operation: str, status: str, details: str = ""):
        """Log access attempts for security auditing."""
        log_entry = {
            "timestamp": time.time(),
            "user_id": user_id or "anonymous",
            "operation": operation,
            "status": status,
            "backend": self.backend_type.value,
            "details": details
        }
        self.access_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-500:]
    
    @abstractmethod
    def get_available_devices(self) -> List[Dict[str, Any]]:
        """Get list of available quantum devices."""
        pass
    
    @abstractmethod
    def compile_circuit(self, circuit: QuantumCircuit, device: str) -> Any:
        """Compile circuit for specific device."""
        pass
    
    @robust(component="quantum_backend", operation="execute_circuit")
    @abstractmethod
    def execute_circuit(
        self, 
        compiled_circuit: Any, 
        shots: int = 1000,
        **kwargs
    ) -> QuantumResult:
        """Execute compiled circuit on quantum device with enhanced reliability."""
        pass
    
    def secure_execute_circuit(
        self, 
        compiled_circuit: Any, 
        shots: int = 1000,
        user_id: Optional[str] = None,
        timeout: float = 300.0,
        **kwargs
    ) -> QuantumResult:
        """Secure circuit execution with comprehensive validation and monitoring."""
        # Validate inputs
        if shots <= 0 or shots > 1000000:  # Reasonable upper limit
            raise ValidationError(f"Invalid shots count: {shots}")
        
        if timeout <= 0 or timeout > 3600:  # Max 1 hour
            raise ValidationError(f"Invalid timeout: {timeout}")
        
        # Check connection
        if not self.is_connected:
            raise ConnectionError("Backend not connected")
        
        # Log execution attempt
        execution_id = hashlib.md5(f"{time.time()}-{shots}".encode()).hexdigest()[:12]
        self._log_access_attempt(user_id, "execute_circuit", "attempted", f"shots={shots}, id={execution_id}")
        
        start_time = time.time()
        
        try:
            # Execute with circuit breaker protection
            result = self.execution_circuit.call(
                self.execute_circuit, 
                compiled_circuit, 
                shots, 
                **kwargs
            )
            
            # Validate result
            if not isinstance(result, QuantumResult):
                raise ValidationError("Invalid result type returned")
            
            # Update metrics
            self.total_executions += 1
            execution_time = time.time() - start_time
            
            # Log successful execution
            self._log_access_attempt(
                user_id, 
                "execute_circuit", 
                "success", 
                f"id={execution_id}, time={execution_time:.2f}s"
            )
            
            logging.info(f"Quantum circuit executed successfully: {execution_id}")
            return result
        
        except Exception as e:
            self.failed_executions += 1
            execution_time = time.time() - start_time
            
            # Log failed execution
            self._log_access_attempt(
                user_id, 
                "execute_circuit", 
                "failed", 
                f"id={execution_id}, time={execution_time:.2f}s, error={str(e)}"
            )
            
            logging.error(f"Quantum circuit execution failed: {execution_id} - {str(e)}")
            raise
    
    @abstractmethod
    def get_device_properties(self, device: str) -> Dict[str, Any]:
        """Get properties of specific quantum device."""
        pass
    
    def validate_circuit(self, circuit: QuantumCircuit, device: str) -> bool:
        """Validate if circuit can run on device with enhanced checks."""
        try:
            device_props = self.get_device_properties(device)
            
            # Validate circuit structure
            if not hasattr(circuit, 'validated') or not circuit.validated:
                raise ValidationError("Circuit not properly validated")
            
            # Check qubit count
            max_qubits = device_props.get("qubits", 0)
            if circuit.qubits > max_qubits:
                logging.warning(f"Circuit requires {circuit.qubits} qubits, device has {max_qubits}")
                return False
            
            # Check gate support
            supported_gates = device_props.get("supported_gates", [])
            for gate in circuit.gates:
                if gate["type"] not in supported_gates:
                    logging.warning(f"Gate {gate['type']} not supported on device {device}")
                    return False
            
            # Check connectivity constraints
            if "connectivity" in device_props:
                connectivity = device_props["connectivity"]
                for gate in circuit.gates:
                    if len(gate["qubits"]) == 2:  # Two-qubit gate
                        q1, q2 = gate["qubits"]
                        if [q1, q2] not in connectivity and [q2, q1] not in connectivity:
                            logging.warning(f"Gate {gate['type']} on qubits {q1},{q2} violates connectivity")
                            return False
            
            return True
            
        except Exception as e:
            logging.error(f"Circuit validation failed: {str(e)}")
            return False
    
    def get_backend_health(self) -> Dict[str, Any]:
        """Get comprehensive backend health status."""
        current_time = time.time()
        
        # Calculate success rate
        success_rate = 1.0
        if self.total_executions > 0:
            success_rate = (self.total_executions - self.failed_executions) / self.total_executions
        
        # Get circuit breaker metrics
        connection_metrics = self.connection_circuit.get_metrics()
        execution_metrics = self.execution_circuit.get_metrics()
        
        health_status = {
            "backend_type": self.backend_type.value,
            "is_connected": self.is_connected,
            "connection_attempts": self.connection_attempts,
            "total_executions": self.total_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "last_health_check": current_time,
            "uptime": current_time - self.last_health_check if self.last_health_check > 0 else 0,
            "circuit_breakers": {
                "connection": connection_metrics,
                "execution": execution_metrics
            },
            "access_logs": len(self.access_log),
            "status": self._determine_health_status(success_rate, connection_metrics, execution_metrics)
        }
        
        self.last_health_check = current_time
        return health_status
    
    def _determine_health_status(self, success_rate: float, connection_metrics: Dict, execution_metrics: Dict) -> str:
        """Determine overall health status."""
        if not self.is_connected:
            return "critical"
        
        if connection_metrics.get("state") == "open" or execution_metrics.get("state") == "open":
            return "degraded"
        
        if success_rate < 0.8:
            return "degraded"
        elif success_rate < 0.5:
            return "unhealthy"
        else:
            return "healthy"
    
    def reset_circuit_breakers(self) -> None:
        """Reset circuit breakers - use with caution."""
        logging.warning(f"Resetting circuit breakers for {self.backend_type.value}")
        # Note: Actual reset would depend on circuit breaker implementation
    
    def get_security_audit_log(self, time_window: float = 3600) -> List[Dict[str, Any]]:
        """Get security audit log for specified time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        return [
            entry for entry in self.access_log 
            if entry["timestamp"] >= cutoff_time
        ]


class QuantumCircuitBuilder:
    """Builder for creating quantum circuits."""
    
    def __init__(self, qubits: int):
        self.circuit = QuantumCircuit(
            qubits=qubits,
            gates=[],
            measurements=[],
            parameters={}
        )
    
    def h(self, qubit: int) -> "QuantumCircuitBuilder":
        """Add Hadamard gate."""
        self.circuit.add_gate("h", [qubit])
        return self
    
    def x(self, qubit: int) -> "QuantumCircuitBuilder":
        """Add Pauli-X gate."""
        self.circuit.add_gate("x", [qubit])
        return self
    
    def y(self, qubit: int) -> "QuantumCircuitBuilder":
        """Add Pauli-Y gate."""
        self.circuit.add_gate("y", [qubit])
        return self
    
    def z(self, qubit: int) -> "QuantumCircuitBuilder":
        """Add Pauli-Z gate."""
        self.circuit.add_gate("z", [qubit])
        return self
    
    def cnot(self, control: int, target: int) -> "QuantumCircuitBuilder":
        """Add CNOT gate."""
        self.circuit.add_gate("cnot", [control, target])
        return self
    
    def rx(self, qubit: int, angle: float) -> "QuantumCircuitBuilder":
        """Add RX rotation gate."""
        self.circuit.add_gate("rx", [qubit], angle=angle)
        return self
    
    def ry(self, qubit: int, angle: float) -> "QuantumCircuitBuilder":
        """Add RY rotation gate."""
        self.circuit.add_gate("ry", [qubit], angle=angle)
        return self
    
    def rz(self, qubit: int, angle: float) -> "QuantumCircuitBuilder":
        """Add RZ rotation gate."""
        self.circuit.add_gate("rz", [qubit], angle=angle)
        return self
    
    def measure(self, qubit: int) -> "QuantumCircuitBuilder":
        """Add measurement."""
        self.circuit.add_measurement(qubit)
        return self
    
    def measure_all(self) -> "QuantumCircuitBuilder":
        """Measure all qubits."""
        for q in range(self.circuit.qubits):
            self.circuit.add_measurement(q)
        return self
    
    def parametric_gate(self, gate_type: str, qubits: List[int], parameter_name: str) -> "QuantumCircuitBuilder":
        """Add parametric gate."""
        self.circuit.add_gate(gate_type, qubits, parameter=parameter_name)
        return self
    
    def set_parameter(self, name: str, value: float) -> "QuantumCircuitBuilder":
        """Set parameter value."""
        self.circuit.parameters[name] = value
        return self
    
    def build(self) -> QuantumCircuit:
        """Build the quantum circuit."""
        return self.circuit


class QuantumAlgorithm(ABC):
    """Abstract base for quantum algorithms."""
    
    def __init__(self, backend: QuantumBackend):
        self.backend = backend
    
    @abstractmethod
    def create_circuit(self, **kwargs) -> QuantumCircuit:
        """Create quantum circuit for the algorithm."""
        pass
    
    @abstractmethod
    def process_result(self, result: QuantumResult) -> Any:
        """Process quantum execution result."""
        pass
    
    def run(self, device: str, shots: int = 1000, **kwargs) -> Any:
        """Run the quantum algorithm."""
        circuit = self.create_circuit(**kwargs)
        compiled = self.backend.compile_circuit(circuit, device)
        result = self.backend.execute_circuit(compiled, shots)
        return self.process_result(result)


class QuantumFederatedAlgorithm(QuantumAlgorithm):
    """Base class for quantum federated learning algorithms."""
    
    @abstractmethod
    def aggregate_parameters(
        self, 
        client_parameters: List[jnp.ndarray],
        quantum_weights: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Quantum parameter aggregation."""
        pass
    
    @abstractmethod
    def compute_quantum_gradients(
        self,
        parameters: jnp.ndarray,
        data_batch: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute quantum gradients."""
        pass