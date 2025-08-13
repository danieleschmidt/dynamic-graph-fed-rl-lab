"""
Tests for quantum federated learning implementation.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch

from dynamic_graph_fed_rl.quantum_hardware.quantum_fed_learning import (
    QuantumFederatedLearning, QuantumFederatedConfig, QuantumAggregationStrategy,
    QuantumWeightedAggregator, QuantumVariationalAggregator,
    QuantumParameterEncoder
)
from dynamic_graph_fed_rl.quantum_hardware.base import QuantumBackendType
from tests.quantum_hardware.test_quantum_backends import MockQuantumBackend


class TestQuantumParameterEncoder:
    """Test quantum parameter encoding methods."""
    
    def test_amplitude_encoding(self):
        """Test amplitude encoding of parameters."""
        params = jnp.array([0.5, 0.3, 0.2])
        circuit = QuantumParameterEncoder.amplitude_encoding(params, 4)
        
        assert circuit.qubits == 4
        assert len(circuit.gates) > 0
    
    def test_angle_encoding(self):
        """Test angle encoding of parameters."""
        params = jnp.array([1.0, 2.0, 3.0])
        circuit = QuantumParameterEncoder.angle_encoding(params, 3)
        
        assert circuit.qubits == 3
        assert len(circuit.gates) == 3
        
        # Check that gates are rotation gates
        for gate in circuit.gates:
            assert gate["type"] == "ry"


class TestQuantumWeightedAggregator:
    """Test quantum weighted aggregation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MockQuantumBackend()
        self.backend.connect({})
        
        self.config = QuantumFederatedConfig(
            aggregation_strategy=QuantumAggregationStrategy.QUANTUM_WEIGHTED_AVERAGE,
            num_qubits=4,
            circuit_depth=2,
            shots=1000,
            optimization_iterations=10,
            quantum_advantage_threshold=0.1,
            noise_mitigation=False,
            error_correction=False
        )
        
        self.aggregator = QuantumWeightedAggregator(self.backend, self.config)
    
    def test_circuit_creation(self):
        """Test quantum weighting circuit creation."""
        circuit = self.aggregator.create_circuit(num_clients=3)
        
        assert circuit.qubits >= 3
        assert len(circuit.gates) > 0
        assert len(circuit.measurements) > 0
    
    def test_parameter_aggregation(self):
        """Test quantum parameter aggregation."""
        # Create test client parameters
        client_params = [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.5, 2.5, 3.5]),
            jnp.array([0.5, 1.5, 2.5])
        ]
        
        aggregated = self.aggregator.aggregate_parameters(client_params)
        
        assert aggregated.shape == (3,)
        assert jnp.all(jnp.isfinite(aggregated))
        
        # Should be somewhere between min and max of client parameters
        client_array = jnp.array(client_params)
        min_vals = jnp.min(client_array, axis=0)
        max_vals = jnp.max(client_array, axis=0)
        
        assert jnp.all(aggregated >= min_vals)
        assert jnp.all(aggregated <= max_vals)
    
    def test_weighted_aggregation(self):
        """Test weighted quantum aggregation."""
        client_params = [
            jnp.array([1.0, 2.0]),
            jnp.array([3.0, 4.0])
        ]
        
        weights = jnp.array([0.7, 0.3])
        
        aggregated = self.aggregator.aggregate_parameters(client_params, weights)
        
        assert aggregated.shape == (2,)
        assert jnp.all(jnp.isfinite(aggregated))
    
    def test_gradient_computation(self):
        """Test quantum gradient computation."""
        params = jnp.array([0.5, 1.0, 1.5])
        data = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        
        gradients = self.aggregator.compute_quantum_gradients(params, data)
        
        assert gradients.shape == params.shape
        assert jnp.all(jnp.isfinite(gradients))


class TestQuantumVariationalAggregator:
    """Test variational quantum aggregation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MockQuantumBackend()
        self.backend.connect({})
        
        self.config = QuantumFederatedConfig(
            aggregation_strategy=QuantumAggregationStrategy.VARIATIONAL_QUANTUM_EIGENSOLVER,
            num_qubits=4,
            circuit_depth=2,
            shots=1000,
            optimization_iterations=5,  # Reduced for testing
            quantum_advantage_threshold=0.1,
            noise_mitigation=False,
            error_correction=False
        )
        
        self.aggregator = QuantumVariationalAggregator(self.backend, self.config)
    
    def test_variational_circuit_creation(self):
        """Test variational ansatz circuit creation."""
        circuit = self.aggregator.create_circuit()
        
        assert circuit.qubits == self.config.num_qubits
        
        # Should have parameterized gates
        param_gates = [g for g in circuit.gates if "parameter" in g]
        expected_param_gates = self.config.circuit_depth * self.config.num_qubits * 2
        assert len(param_gates) == expected_param_gates
    
    def test_vqe_aggregation(self):
        """Test VQE-based aggregation."""
        client_params = [
            jnp.array([1.0, 2.0]),
            jnp.array([2.0, 3.0]),
            jnp.array([1.5, 2.5])
        ]
        
        aggregated = self.aggregator.aggregate_parameters(client_params)
        
        assert aggregated.shape == (2,)
        assert jnp.all(jnp.isfinite(aggregated))
        
        # Check that variational parameters were initialized
        assert self.aggregator.variational_params is not None
        assert len(self.aggregator.variational_params) > 0


class TestQuantumFederatedLearning:
    """Test main quantum federated learning orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backends = {
            "mock_simulator": MockQuantumBackend(),
            "mock_quantum": MockQuantumBackend()
        }
        
        for backend in self.backends.values():
            backend.connect({})
        
        self.config = QuantumFederatedConfig(
            aggregation_strategy=QuantumAggregationStrategy.QUANTUM_WEIGHTED_AVERAGE,
            num_qubits=4,
            circuit_depth=2,
            shots=100,  # Reduced for testing
            optimization_iterations=3,  # Reduced for testing
            quantum_advantage_threshold=0.1,
            noise_mitigation=False,
            error_correction=False
        )
        
        self.qfl = QuantumFederatedLearning(self.backends, self.config)
    
    def test_initialization(self):
        """Test quantum federated learning initialization."""
        assert len(self.qfl.backends) == 2
        assert len(self.qfl.aggregators) > 0
        assert self.qfl.performance_history == []
    
    def test_federated_round(self):
        """Test federated learning round execution."""
        client_params = [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.1, 2.1, 3.1]),
            jnp.array([0.9, 1.9, 2.9])
        ]
        
        client_weights = jnp.array([0.4, 0.3, 0.3])
        
        aggregated, round_info = self.qfl.federated_round(
            client_params, client_weights, round_number=0
        )
        
        assert aggregated.shape == (3,)
        assert jnp.all(jnp.isfinite(aggregated))
        
        # Check round info
        assert round_info["round_number"] == 0
        assert round_info["num_clients"] == 3
        assert "execution_time" in round_info
        assert "results" in round_info
        
        # Performance history should be updated
        assert len(self.qfl.performance_history) == 1
    
    def test_multiple_federated_rounds(self):
        """Test multiple federated rounds."""
        client_params = [
            jnp.array([1.0, 2.0]),
            jnp.array([2.0, 3.0])
        ]
        
        for round_num in range(3):
            aggregated, round_info = self.qfl.federated_round(
                client_params, round_number=round_num
            )
            
            assert aggregated is not None
            assert round_info["round_number"] == round_num
        
        assert len(self.qfl.performance_history) == 3
    
    def test_performance_metrics(self):
        """Test performance metrics computation."""
        # Run some rounds first
        client_params = [jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0])]
        
        for i in range(5):
            self.qfl.federated_round(client_params, round_number=i)
        
        metrics = self.qfl.get_performance_metrics()
        
        assert "total_rounds" in metrics
        assert metrics["total_rounds"] == 5
        assert "quantum_success_rate" in metrics
        assert "average_execution_time" in metrics
        assert "backend_success_rates" in metrics
        assert "quantum_advantage_achieved" in metrics
    
    def test_fallback_to_classical(self):
        """Test fallback to classical aggregation when quantum fails."""
        # Create backends that will fail
        failing_backends = {
            "failing_backend": Mock()
        }
        
        # Mock the backend to fail during aggregation
        failing_backend = failing_backends["failing_backend"]
        failing_backend.backend_type = QuantumBackendType.SIMULATOR
        
        config = QuantumFederatedConfig(
            aggregation_strategy=QuantumAggregationStrategy.QUANTUM_WEIGHTED_AVERAGE,
            num_qubits=4,
            circuit_depth=2,
            shots=100,
            optimization_iterations=3,
            quantum_advantage_threshold=0.1,
            noise_mitigation=False,
            error_correction=False
        )
        
        qfl = QuantumFederatedLearning(failing_backends, config)
        
        # Mock aggregator to raise exception
        for aggregator in qfl.aggregators.values():
            aggregator.aggregate_parameters = Mock(side_effect=Exception("Quantum failure"))
        
        client_params = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]
        client_weights = jnp.array([0.6, 0.4])
        
        aggregated, round_info = qfl.federated_round(client_params, client_weights)
        
        # Should fallback to classical
        assert aggregated is not None
        assert "classical_fallback" in round_info["results"]
        assert round_info["results"]["classical_fallback"]["success"]
        
        # Verify classical aggregation result
        expected_classical = jnp.average(jnp.array(client_params), axis=0, weights=client_weights)
        np.testing.assert_allclose(aggregated, expected_classical, rtol=1e-10)