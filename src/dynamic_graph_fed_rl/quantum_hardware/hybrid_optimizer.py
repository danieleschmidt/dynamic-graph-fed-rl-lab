"""
Hybrid classical-quantum optimization for dynamic graph processing.

Combines classical optimization methods with quantum algorithms for enhanced
performance in federated learning over dynamic graphs.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from enum import Enum
import time

from .base import QuantumBackend, QuantumCircuit, QuantumResult, QuantumCircuitBuilder
from .quantum_fed_learning import QuantumFederatedConfig


class OptimizationStrategy(Enum):
    """Hybrid optimization strategies."""
    QAOA_CLASSICAL = "qaoa_classical"
    VQE_GRADIENT_DESCENT = "vqe_gradient_descent"
    QUANTUM_NATURAL_GRADIENT = "quantum_natural_gradient"
    ALTERNATING_OPTIMIZATION = "alternating_optimization"


@dataclass
class HybridOptimizationConfig:
    """Configuration for hybrid classical-quantum optimization."""
    strategy: OptimizationStrategy
    max_classical_iterations: int
    max_quantum_iterations: int
    classical_learning_rate: float
    quantum_learning_rate: float
    convergence_threshold: float
    use_parameter_shift: bool
    noise_mitigation: bool


class QuantumApproximateOptimizationAlgorithm:
    """QAOA implementation for graph optimization problems."""
    
    def __init__(self, backend: QuantumBackend, p_layers: int = 2):
        self.backend = backend
        self.p_layers = p_layers
        
    def create_qaoa_circuit(
        self, 
        graph_edges: List[Tuple[int, int]], 
        gamma: jnp.ndarray, 
        beta: jnp.ndarray
    ) -> QuantumCircuit:
        """Create QAOA circuit for Max-Cut problem."""
        # Determine number of nodes
        nodes = set()
        for edge in graph_edges:
            nodes.update(edge)
        num_qubits = len(nodes)
        
        builder = QuantumCircuitBuilder(num_qubits)
        
        # Initial state: uniform superposition
        for q in range(num_qubits):
            builder.h(q)
        
        # QAOA layers
        for p in range(self.p_layers):
            # Cost Hamiltonian (Problem layer)
            for edge in graph_edges:
                q1, q2 = edge
                if q1 < num_qubits and q2 < num_qubits:
                    builder.cnot(q1, q2)
                    builder.rz(q2, 2 * gamma[p])
                    builder.cnot(q1, q2)
            
            # Mixer Hamiltonian (Driver layer)
            for q in range(num_qubits):
                builder.rx(q, 2 * beta[p])
        
        builder.measure_all()
        return builder.build()
    
    def optimize_graph_partition(
        self,
        graph_edges: List[Tuple[int, int]],
        classical_optimizer: Callable = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """Optimize graph partitioning using QAOA."""
        
        # Initialize QAOA parameters
        gamma = jnp.random.uniform(0, jnp.pi, self.p_layers)
        beta = jnp.random.uniform(0, jnp.pi/2, self.p_layers)
        
        best_cost = float('-inf')  # Maximizing cut value
        best_params = None
        best_solution = None
        
        def qaoa_objective(params):
            """QAOA objective function."""
            g, b = params[:self.p_layers], params[self.p_layers:]
            
            circuit = self.create_qaoa_circuit(graph_edges, g, b)
            device = self._get_best_device()
            
            try:
                compiled = self.backend.compile_circuit(circuit, device)
                result = self.backend.execute_circuit(compiled, shots=1000)
                
                if result.success:
                    return self._evaluate_cut_value(result.counts, graph_edges)
                else:
                    return -1000  # Penalty for failed execution
            except Exception:
                return -1000
        
        # Classical optimization of QAOA parameters
        if classical_optimizer is None:
            # Simple grid search + gradient-free optimization
            for iteration in range(max_iterations):
                # Random parameter update
                if iteration == 0:
                    current_params = jnp.concatenate([gamma, beta])
                else:
                    # Add noise for exploration
                    noise_scale = max(0.1, 1.0 - iteration / max_iterations)
                    current_params = best_params + jnp.random.normal(0, noise_scale, size=best_params.shape)
                
                cost = qaoa_objective(current_params)
                
                if cost > best_cost:
                    best_cost = cost
                    best_params = current_params.copy()
                    
                    # Get best solution
                    g, b = best_params[:self.p_layers], best_params[self.p_layers:]
                    circuit = self.create_qaoa_circuit(graph_edges, g, b)
                    device = self._get_best_device()
                    compiled = self.backend.compile_circuit(circuit, device)
                    result = self.backend.execute_circuit(compiled, shots=1000)
                    
                    if result.success:
                        best_solution = max(result.counts.items(), key=lambda x: x[1])[0]
        else:
            # Use provided classical optimizer
            result = classical_optimizer(
                qaoa_objective,
                jnp.concatenate([gamma, beta]),
                maxiter=max_iterations
            )
            best_params = result.x
            best_cost = -result.fun  # Negate if minimizer
        
        return {
            "best_cost": best_cost,
            "best_parameters": best_params,
            "best_solution": best_solution,
            "num_iterations": max_iterations,
            "optimization_success": best_cost > -999
        }
    
    def _evaluate_cut_value(self, counts: Dict[str, int], edges: List[Tuple[int, int]]) -> float:
        """Evaluate average cut value from quantum measurements."""
        total_cut_value = 0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            cut_value = 0
            for edge in edges:
                q1, q2 = edge
                if q1 < len(bitstring) and q2 < len(bitstring):
                    if bitstring[q1] != bitstring[q2]:  # Edge is cut
                        cut_value += 1
            
            total_cut_value += cut_value * (count / total_shots)
        
        return total_cut_value
    
    def _get_best_device(self) -> str:
        """Select best quantum device for QAOA."""
        devices = self.backend.get_available_devices()
        
        # Prefer quantum hardware with low noise
        quantum_devices = [d for d in devices if not d.get("simulator", True)]
        if quantum_devices:
            # Select device with best quantum volume or connectivity
            return max(quantum_devices, key=lambda d: d.get("quantum_volume", d.get("qubits", 0)))["name"]
        
        # Fallback to simulator
        simulators = [d for d in devices if d.get("simulator", False)]
        if simulators:
            return simulators[0]["name"]
        
        raise RuntimeError("No suitable quantum device available")


class VariationalQuantumEigensolver:
    """VQE for finding optimal federated parameters."""
    
    def __init__(self, backend: QuantumBackend, ansatz_depth: int = 3):
        self.backend = backend
        self.ansatz_depth = ansatz_depth
        
    def create_ansatz(self, num_qubits: int, parameters: jnp.ndarray) -> QuantumCircuit:
        """Create hardware-efficient variational ansatz."""
        builder = QuantumCircuitBuilder(num_qubits)
        
        param_idx = 0
        
        for layer in range(self.ansatz_depth):
            # Layer of single-qubit rotations
            for q in range(num_qubits):
                if param_idx < len(parameters):
                    builder.ry(q, parameters[param_idx])
                    param_idx += 1
                if param_idx < len(parameters):
                    builder.rz(q, parameters[param_idx])
                    param_idx += 1
            
            # Entangling layer
            for q in range(num_qubits - 1):
                builder.cnot(q, q + 1)
            
            # Ring closure
            if num_qubits > 2:
                builder.cnot(num_qubits - 1, 0)
        
        return builder.build()
    
    def optimize_hamiltonian(
        self,
        hamiltonian_terms: List[Dict[str, Any]],
        num_qubits: int,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """Optimize expectation value of Hamiltonian using VQE."""
        
        # Initialize variational parameters
        num_params = self.ansatz_depth * num_qubits * 2
        theta = jnp.random.uniform(0, 2*jnp.pi, num_params)
        
        best_energy = float('inf')
        best_params = None
        energy_history = []
        
        def vqe_objective(params):
            """VQE objective function - compute Hamiltonian expectation."""
            circuit = self.create_ansatz(num_qubits, params)
            circuit.measure_all()
            
            device = self._get_best_device()
            
            try:
                compiled = self.backend.compile_circuit(circuit, device)
                result = self.backend.execute_circuit(compiled, shots=1000)
                
                if result.success:
                    return self._compute_hamiltonian_expectation(result, hamiltonian_terms)
                else:
                    return 1000  # Penalty for failed execution
            except Exception:
                return 1000
        
        # VQE optimization loop
        for iteration in range(max_iterations):
            energy = vqe_objective(theta)
            energy_history.append(energy)
            
            if energy < best_energy:
                best_energy = energy
                best_params = theta.copy()
            
            # Compute gradients using parameter shift rule
            if iteration < max_iterations - 1:
                gradients = self._compute_parameter_gradients(theta, hamiltonian_terms, num_qubits)
                
                # Gradient descent update
                learning_rate = 0.1 * (1 - iteration / max_iterations)  # Decaying learning rate
                theta = theta - learning_rate * gradients
                
                # Keep parameters in valid range
                theta = theta % (2 * jnp.pi)
        
        return {
            "optimal_energy": best_energy,
            "optimal_parameters": best_params,
            "energy_history": energy_history,
            "num_iterations": max_iterations,
            "convergence_achieved": abs(energy_history[-1] - energy_history[-10]) < 1e-6 if len(energy_history) >= 10 else False
        }
    
    def _compute_parameter_gradients(
        self, 
        parameters: jnp.ndarray, 
        hamiltonian_terms: List[Dict[str, Any]], 
        num_qubits: int
    ) -> jnp.ndarray:
        """Compute gradients using parameter shift rule."""
        gradients = jnp.zeros_like(parameters)
        shift = jnp.pi / 2
        
        for i in range(len(parameters)):
            # Forward shift
            params_plus = parameters.at[i].add(shift)
            expectation_plus = self._evaluate_hamiltonian_at_params(params_plus, hamiltonian_terms, num_qubits)
            
            # Backward shift
            params_minus = parameters.at[i].add(-shift)
            expectation_minus = self._evaluate_hamiltonian_at_params(params_minus, hamiltonian_terms, num_qubits)
            
            # Parameter shift rule
            gradients = gradients.at[i].set((expectation_plus - expectation_minus) / 2)
        
        return gradients
    
    def _evaluate_hamiltonian_at_params(
        self, 
        parameters: jnp.ndarray, 
        hamiltonian_terms: List[Dict[str, Any]], 
        num_qubits: int
    ) -> float:
        """Evaluate Hamiltonian expectation at given parameters."""
        circuit = self.create_ansatz(num_qubits, parameters)
        circuit.measure_all()
        
        device = self._get_best_device()
        
        try:
            compiled = self.backend.compile_circuit(circuit, device)
            result = self.backend.execute_circuit(compiled, shots=100)  # Fewer shots for gradient
            
            if result.success:
                return self._compute_hamiltonian_expectation(result, hamiltonian_terms)
            
        except Exception:
            pass
        
        return 1000  # High penalty for failed evaluation
    
    def _compute_hamiltonian_expectation(
        self, 
        result: QuantumResult, 
        hamiltonian_terms: List[Dict[str, Any]]
    ) -> float:
        """Compute Hamiltonian expectation value from measurement results."""
        total_expectation = 0.0
        
        for term in hamiltonian_terms:
            coefficient = term.get("coefficient", 1.0)
            pauli_string = term.get("pauli_string", "")
            
            term_expectation = 0.0
            total_shots = sum(result.counts.values())
            
            for bitstring, count in result.counts.items():
                # Compute expectation for this Pauli term
                parity = 1.0
                for i, pauli in enumerate(pauli_string):
                    if i < len(bitstring):
                        if pauli == 'Z' and bitstring[i] == '1':
                            parity *= -1
                        # For X and Y, would need different measurement basis
                
                term_expectation += parity * (count / total_shots)
            
            total_expectation += coefficient * term_expectation
        
        return total_expectation
    
    def _get_best_device(self) -> str:
        """Select best quantum device for VQE."""
        devices = self.backend.get_available_devices()
        
        # Prefer quantum hardware with good coherence
        quantum_devices = [d for d in devices if not d.get("simulator", True)]
        if quantum_devices:
            return max(quantum_devices, key=lambda d: d.get("qubits", 0))["name"]
        
        # Fallback to simulator
        simulators = [d for d in devices if d.get("simulator", False)]
        if simulators:
            return simulators[0]["name"]
        
        raise RuntimeError("No suitable quantum device available")


class HybridClassicalQuantumOptimizer:
    """Main hybrid optimization orchestrator."""
    
    def __init__(
        self,
        quantum_backends: Dict[str, QuantumBackend],
        config: HybridOptimizationConfig
    ):
        self.quantum_backends = quantum_backends
        self.config = config
        self.qaoa = {name: QuantumApproximateOptimizationAlgorithm(backend) 
                    for name, backend in quantum_backends.items()}
        self.vqe = {name: VariationalQuantumEigensolver(backend) 
                   for name, backend in quantum_backends.items()}
        
    def optimize_dynamic_graph_partition(
        self,
        graph_snapshots: List[List[Tuple[int, int]]],
        temporal_weights: Optional[jnp.ndarray] = None
    ) -> Dict[str, Any]:
        """Optimize partitioning across dynamic graph snapshots."""
        
        if temporal_weights is None:
            temporal_weights = jnp.ones(len(graph_snapshots)) / len(graph_snapshots)
        
        results = {}
        total_optimization_value = 0.0
        
        for backend_name in self.quantum_backends.keys():
            try:
                backend_results = []
                backend_total_value = 0.0
                
                for t, (graph, weight) in enumerate(zip(graph_snapshots, temporal_weights)):
                    qaoa_result = self.qaoa[backend_name].optimize_graph_partition(
                        graph,
                        max_iterations=self.config.max_quantum_iterations
                    )
                    
                    backend_results.append(qaoa_result)
                    backend_total_value += weight * qaoa_result["best_cost"]
                
                results[backend_name] = {
                    "success": True,
                    "temporal_results": backend_results,
                    "weighted_total_value": backend_total_value,
                    "backend_type": self.quantum_backends[backend_name].backend_type.value
                }
                
                # Update best total value
                if backend_total_value > total_optimization_value:
                    total_optimization_value = backend_total_value
                    
            except Exception as e:
                results[backend_name] = {
                    "success": False,
                    "error": str(e),
                    "backend_type": self.quantum_backends[backend_name].backend_type.value
                }
        
        return {
            "optimization_value": total_optimization_value,
            "backend_results": results,
            "strategy": self.config.strategy.value,
            "quantum_advantage": any(r["success"] for r in results.values())
        }
    
    def optimize_federated_parameters(
        self,
        client_parameters: List[jnp.ndarray],
        objective_function: Callable[[jnp.ndarray], float],
        constraints: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """Hybrid optimization of federated parameters."""
        
        num_clients = len(client_parameters)
        parameter_dim = client_parameters[0].size
        
        # Create Hamiltonian encoding the optimization problem
        hamiltonian_terms = self._encode_optimization_hamiltonian(
            client_parameters, 
            objective_function
        )
        
        results = {}
        best_parameters = None
        best_objective_value = float('inf')
        
        # Try VQE optimization with each backend
        for backend_name in self.quantum_backends.keys():
            try:
                vqe_result = self.vqe[backend_name].optimize_hamiltonian(
                    hamiltonian_terms,
                    num_qubits=max(4, num_clients.bit_length()),
                    max_iterations=self.config.max_quantum_iterations
                )
                
                # Convert VQE result to parameter space
                optimized_params = self._decode_vqe_parameters(
                    vqe_result["optimal_parameters"],
                    client_parameters
                )
                
                objective_value = objective_function(optimized_params)
                
                results[backend_name] = {
                    "success": True,
                    "vqe_result": vqe_result,
                    "optimized_parameters": optimized_params,
                    "objective_value": objective_value,
                    "backend_type": self.quantum_backends[backend_name].backend_type.value
                }
                
                if objective_value < best_objective_value:
                    best_objective_value = objective_value
                    best_parameters = optimized_params
                    
            except Exception as e:
                results[backend_name] = {
                    "success": False,
                    "error": str(e),
                    "backend_type": self.quantum_backends[backend_name].backend_type.value
                }
        
        # Classical refinement of best quantum solution
        if best_parameters is not None:
            refined_params, refined_value = self._classical_refinement(
                best_parameters,
                objective_function,
                constraints
            )
            
            results["classical_refinement"] = {
                "initial_value": best_objective_value,
                "refined_value": refined_value,
                "improvement": best_objective_value - refined_value,
                "refined_parameters": refined_params
            }
            
            if refined_value < best_objective_value:
                best_objective_value = refined_value
                best_parameters = refined_params
        
        return {
            "best_parameters": best_parameters,
            "best_objective_value": best_objective_value,
            "backend_results": results,
            "hybrid_optimization_used": "classical_refinement" in results,
            "quantum_advantage": any(r["success"] for r in results.values() if "backend_type" in r and r["backend_type"] != "classical")
        }
    
    def _encode_optimization_hamiltonian(
        self,
        client_parameters: List[jnp.ndarray],
        objective_function: Callable
    ) -> List[Dict[str, Any]]:
        """Encode optimization problem as quantum Hamiltonian."""
        # Simplified encoding - in practice would be problem-specific
        hamiltonian_terms = []
        
        # Add terms representing parameter differences (diversity regularization)
        for i in range(len(client_parameters)):
            for j in range(i + 1, len(client_parameters)):
                # Term encouraging diversity between client parameters
                hamiltonian_terms.append({
                    "coefficient": -0.1,  # Negative to encourage diversity
                    "pauli_string": "Z" * (i + 1) + "I" * (j - i - 1) + "Z" + "I" * (len(client_parameters) - j - 1)
                })
        
        # Add terms representing objective function (simplified quadratic form)
        for i in range(len(client_parameters)):
            hamiltonian_terms.append({
                "coefficient": 1.0,
                "pauli_string": "I" * i + "Z" + "I" * (len(client_parameters) - i - 1)
            })
        
        return hamiltonian_terms
    
    def _decode_vqe_parameters(
        self,
        vqe_params: jnp.ndarray,
        client_parameters: List[jnp.ndarray]
    ) -> jnp.ndarray:
        """Decode VQE solution back to parameter space."""
        # Simplified decoding - use VQE parameters to weight client contributions
        num_clients = len(client_parameters)
        
        # Create weights from VQE parameters
        weights = jnp.abs(vqe_params[:num_clients])
        weights = weights / jnp.sum(weights)
        
        # Weighted average of client parameters
        return jnp.average(jnp.array(client_parameters), axis=0, weights=weights)
    
    def _classical_refinement(
        self,
        initial_params: jnp.ndarray,
        objective_function: Callable,
        constraints: Optional[List[Callable]] = None
    ) -> Tuple[jnp.ndarray, float]:
        """Refine quantum solution using classical optimization."""
        
        # Simple gradient descent refinement
        params = initial_params.copy()
        best_value = objective_function(params)
        
        learning_rate = self.config.classical_learning_rate
        
        for iteration in range(self.config.max_classical_iterations):
            # Compute numerical gradient
            gradient = jnp.zeros_like(params)
            epsilon = 1e-6
            
            for i in range(len(params)):
                params_plus = params.at[i].add(epsilon)
                params_minus = params.at[i].add(-epsilon)
                
                grad_i = (objective_function(params_plus) - objective_function(params_minus)) / (2 * epsilon)
                gradient = gradient.at[i].set(grad_i)
            
            # Update parameters
            params = params - learning_rate * gradient
            
            # Apply constraints if provided
            if constraints:
                for constraint in constraints:
                    params = constraint(params)
            
            # Check for improvement
            current_value = objective_function(params)
            if current_value < best_value:
                best_value = current_value
            
            # Check convergence
            if jnp.linalg.norm(gradient) < self.config.convergence_threshold:
                break
        
        return params, best_value