"""
Tests for quantum backend implementations.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch

from dynamic_graph_fed_rl.quantum_hardware.base import (
    QuantumBackend, QuantumCircuit, QuantumResult, QuantumBackendType,
    QuantumCircuitBuilder
)


class MockQuantumBackend(QuantumBackend):
    """Mock quantum backend for testing."""
    
    def __init__(self):
        super().__init__(QuantumBackendType.SIMULATOR)
        self.mock_devices = [
            {"name": "mock_simulator", "qubits": 32, "simulator": True},
            {"name": "mock_quantum", "qubits": 8, "simulator": False}
        ]
    
    def connect(self, credentials):
        self.is_connected = True
        return True
    
    def get_available_devices(self):
        return self.mock_devices
    
    def get_device_properties(self, device):
        for dev in self.mock_devices:
            if dev["name"] == device:
                return {**dev, "supported_gates": ["h", "x", "y", "z", "cnot", "rx", "ry", "rz"]}
        return {}
    
    def compile_circuit(self, circuit, device):
        return circuit  # Mock compilation
    
    def execute_circuit(self, compiled_circuit, shots=1000, **kwargs):
        # Mock quantum execution with random results
        num_qubits = compiled_circuit.qubits
        counts = {}
        
        # Generate random measurement outcomes
        for _ in range(shots):
            bitstring = ''.join(np.random.choice(['0', '1']) for _ in range(num_qubits))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return QuantumResult(
            backend_type=self.backend_type,
            job_id="mock_job_123",
            counts=counts,
            execution_time=0.1,
            shots=shots,
            success=True
        )


class TestQuantumCircuitBuilder:
    """Test quantum circuit builder."""
    
    def test_circuit_creation(self):
        """Test basic circuit creation."""
        builder = QuantumCircuitBuilder(3)
        circuit = builder.h(0).cnot(0, 1).measure_all().build()
        
        assert circuit.qubits == 3
        assert len(circuit.gates) == 2
        assert len(circuit.measurements) == 3
        
        # Check gate types
        assert circuit.gates[0]["type"] == "h"
        assert circuit.gates[1]["type"] == "cnot"
    
    def test_parametric_gates(self):
        """Test parametric gate creation."""
        builder = QuantumCircuitBuilder(2)
        circuit = builder.parametric_gate("ry", [0], "theta").set_parameter("theta", np.pi/4).build()
        
        assert "theta" in circuit.parameters
        assert circuit.parameters["theta"] == np.pi/4
        assert circuit.gates[0]["parameter"] == "theta"
    
    def test_rotation_gates(self):
        """Test rotation gate creation."""
        builder = QuantumCircuitBuilder(2)
        circuit = builder.rx(0, np.pi/2).ry(1, np.pi/3).rz(0, np.pi/4).build()
        
        assert len(circuit.gates) == 3
        assert circuit.gates[0]["type"] == "rx"
        assert circuit.gates[1]["type"] == "ry" 
        assert circuit.gates[2]["type"] == "rz"


class TestQuantumBackend:
    """Test quantum backend functionality."""
    
    def test_backend_connection(self):
        """Test backend connection."""
        backend = MockQuantumBackend()
        assert not backend.is_connected
        
        success = backend.connect({})
        assert success
        assert backend.is_connected
    
    def test_get_devices(self):
        """Test device listing."""
        backend = MockQuantumBackend()
        backend.connect({})
        
        devices = backend.get_available_devices()
        assert len(devices) == 2
        assert devices[0]["name"] == "mock_simulator"
        assert devices[1]["name"] == "mock_quantum"
    
    def test_circuit_validation(self):
        """Test circuit validation."""
        backend = MockQuantumBackend()
        backend.connect({})
        
        # Valid circuit
        circuit = QuantumCircuitBuilder(3).h(0).cnot(0, 1).measure_all().build()
        assert backend.validate_circuit(circuit, "mock_simulator")
        
        # Invalid circuit (too many qubits)
        large_circuit = QuantumCircuitBuilder(100).h(0).build()
        assert not backend.validate_circuit(large_circuit, "mock_quantum")
    
    def test_circuit_execution(self):
        """Test circuit execution."""
        backend = MockQuantumBackend()
        backend.connect({})
        
        circuit = QuantumCircuitBuilder(2).h(0).cnot(0, 1).measure_all().build()
        compiled = backend.compile_circuit(circuit, "mock_simulator")
        result = backend.execute_circuit(compiled, shots=100)
        
        assert result.success
        assert result.shots == 100
        assert len(result.counts) > 0
        assert sum(result.counts.values()) == 100


@pytest.mark.skipif(True, reason="Requires actual quantum hardware credentials")
class TestRealQuantumBackends:
    """Tests for real quantum backends (requires credentials)."""
    
    def test_ibm_quantum_connection(self):
        """Test IBM Quantum connection (requires real credentials)."""
        from dynamic_graph_fed_rl.quantum_hardware.ibm_quantum import IBMQuantumBackend
        
        backend = IBMQuantumBackend()
        # Would need real IBM Quantum token
        credentials = {"token": "your_ibm_token_here"}
        
        # This would fail without real credentials
        # success = backend.connect(credentials)
        # assert success
    
    def test_google_quantum_connection(self):
        """Test Google Quantum connection (requires real credentials)."""
        from dynamic_graph_fed_rl.quantum_hardware.google_quantum import GoogleQuantumBackend
        
        backend = GoogleQuantumBackend()
        # Would need real Google Cloud project
        credentials = {"project_id": "your_project_id"}
        
        # This would fail without real credentials
        # success = backend.connect(credentials)
        # assert success
    
    def test_aws_braket_connection(self):
        """Test AWS Braket connection (requires real credentials)."""
        from dynamic_graph_fed_rl.quantum_hardware.aws_braket import AWSBraketBackend
        
        backend = AWSBraketBackend()
        credentials = {
            "aws_access_key_id": "your_access_key",
            "aws_secret_access_key": "your_secret_key",
            "region": "us-east-1"
        }
        
        # This would fail without real credentials
        # success = backend.connect(credentials)
        # assert success