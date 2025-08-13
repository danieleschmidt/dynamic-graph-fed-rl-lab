"""
Base classes for quantum hardware integration.

Provides abstract interfaces for different quantum computing platforms
to enable seamless switching between IBM Quantum, Google Cirq, and AWS Braket.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import numpy as np
import jax.numpy as jnp


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
    """Abstract base class for quantum computing backends."""
    
    def __init__(self, backend_type: QuantumBackendType):
        self.backend_type = backend_type
        self.is_connected = False
        self.device_info: Dict[str, Any] = {}
        
    @abstractmethod
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to the quantum backend."""
        pass
    
    @abstractmethod
    def get_available_devices(self) -> List[Dict[str, Any]]:
        """Get list of available quantum devices."""
        pass
    
    @abstractmethod
    def compile_circuit(self, circuit: QuantumCircuit, device: str) -> Any:
        """Compile circuit for specific device."""
        pass
    
    @abstractmethod
    def execute_circuit(
        self, 
        compiled_circuit: Any, 
        shots: int = 1000,
        **kwargs
    ) -> QuantumResult:
        """Execute compiled circuit on quantum device."""
        pass
    
    @abstractmethod
    def get_device_properties(self, device: str) -> Dict[str, Any]:
        """Get properties of specific quantum device."""
        pass
    
    def validate_circuit(self, circuit: QuantumCircuit, device: str) -> bool:
        """Validate if circuit can run on device."""
        device_props = self.get_device_properties(device)
        
        if circuit.qubits > device_props.get("qubits", 0):
            return False
        
        supported_gates = device_props.get("supported_gates", [])
        for gate in circuit.gates:
            if gate["type"] not in supported_gates:
                return False
                
        return True


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