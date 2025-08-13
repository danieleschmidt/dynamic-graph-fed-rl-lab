"""
AWS Braket backend implementation.

Provides integration with AWS Braket quantum computing service for real quantum hardware execution.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from .base import QuantumBackend, QuantumBackendType, QuantumCircuit, QuantumResult

try:
    from braket.circuits import Circuit as BraketCircuit
    from braket.circuits.instruction import Instruction
    from braket.circuits import gates
    from braket.aws import AwsDevice
    from braket.devices import Device
    import boto3
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False


class AWSBraketBackend(QuantumBackend):
    """AWS Braket backend implementation."""
    
    def __init__(self):
        super().__init__(QuantumBackendType.AWS_BRAKET)
        self.session = None
        self.s3_bucket = None
        self.s3_prefix = None
        
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to AWS Braket service."""
        if not BRAKET_AVAILABLE:
            raise ImportError("Braket not available. Install with: pip install amazon-braket-sdk boto3")
        
        try:
            # Configure AWS session
            self.session = boto3.Session(
                aws_access_key_id=credentials.get("aws_access_key_id"),
                aws_secret_access_key=credentials.get("aws_secret_access_key"),
                region_name=credentials.get("region", "us-east-1")
            )
            
            # S3 bucket for job results
            self.s3_bucket = credentials.get("s3_bucket", "amazon-braket-quantum-results")
            self.s3_prefix = credentials.get("s3_prefix", "quantum-federated-learning")
            
            # Test connection by listing devices
            devices = AwsDevice.get_devices()
            
            self.is_connected = True
            return True
            
        except Exception as e:
            print(f"Failed to connect to AWS Braket: {e}")
            return False
    
    def get_available_devices(self) -> List[Dict[str, Any]]:
        """Get available AWS Braket devices."""
        if not self.is_connected:
            return []
        
        devices = []
        try:
            for device in AwsDevice.get_devices():
                device_info = {
                    "name": device.name,
                    "arn": device.arn,
                    "type": device.type.value,
                    "provider": device.provider_name,
                    "status": device.status.value,
                    "qubits": self._get_qubit_count(device),
                    "simulator": "simulator" in device.type.value.lower(),
                }
                
                # Add device-specific properties
                if hasattr(device, 'properties'):
                    props = device.properties
                    if hasattr(props, 'paradigm'):
                        device_info["paradigm"] = props.paradigm.name
                    if hasattr(props, 'connectivity'):
                        device_info["connectivity"] = str(props.connectivity)
                
                devices.append(device_info)
                
        except Exception as e:
            print(f"Error getting AWS Braket devices: {e}")
        
        return devices
    
    def _get_qubit_count(self, device: AwsDevice) -> int:
        """Extract qubit count from device properties."""
        try:
            if hasattr(device, 'properties') and hasattr(device.properties, 'paradigm'):
                paradigm = device.properties.paradigm
                if hasattr(paradigm, 'qubit_count'):
                    return paradigm.qubit_count
                elif hasattr(paradigm, 'connectivity') and hasattr(paradigm.connectivity, 'fully_connected_qubits_count'):
                    return paradigm.connectivity.fully_connected_qubits_count
        except Exception:
            pass
        return 0
    
    def get_device_properties(self, device: str) -> Dict[str, Any]:
        """Get detailed properties of AWS Braket device."""
        if not self.is_connected:
            return {}
        
        try:
            aws_device = AwsDevice(device)
            properties = aws_device.properties
            
            device_props = {
                "name": aws_device.name,
                "arn": device,
                "type": aws_device.type.value,
                "provider": aws_device.provider_name,
                "status": aws_device.status.value,
                "qubits": self._get_qubit_count(aws_device),
                "simulator": "simulator" in aws_device.type.value.lower(),
            }
            
            # Add paradigm-specific properties
            if hasattr(properties, 'paradigm'):
                paradigm = properties.paradigm
                device_props["paradigm"] = paradigm.name if hasattr(paradigm, 'name') else str(paradigm)
                
                # Gate-based quantum computer properties
                if hasattr(paradigm, 'native_gate_set'):
                    device_props["native_gates"] = [str(gate) for gate in paradigm.native_gate_set]
                
                # Connectivity information
                if hasattr(paradigm, 'connectivity'):
                    connectivity = paradigm.connectivity
                    device_props["connectivity_graph"] = str(connectivity)
            
            # Add service-specific properties
            if hasattr(properties, 'service'):
                service = properties.service
                if hasattr(service, 'execution_windows'):
                    device_props["execution_windows"] = str(service.execution_windows)
                if hasattr(service, 'shotsRange'):
                    device_props["shots_range"] = (service.shotsRange.min, service.shotsRange.max)
            
            return device_props
            
        except Exception as e:
            print(f"Error getting device properties for {device}: {e}")
            return {}
    
    def compile_circuit(self, circuit: QuantumCircuit, device: str) -> BraketCircuit:
        """Compile universal circuit to Braket format."""
        if not self.is_connected:
            raise RuntimeError("Not connected to AWS Braket")
        
        # Create Braket circuit
        braket_circuit = BraketCircuit()
        
        # Add gates
        for gate in circuit.gates:
            gate_type = gate["type"]
            qubits = gate["qubits"]
            
            if gate_type == "h":
                braket_circuit.h(qubits[0])
            elif gate_type == "x":
                braket_circuit.x(qubits[0])
            elif gate_type == "y":
                braket_circuit.y(qubits[0])
            elif gate_type == "z":
                braket_circuit.z(qubits[0])
            elif gate_type == "cnot":
                braket_circuit.cnot(qubits[0], qubits[1])
            elif gate_type == "rx":
                angle = gate.get("angle", circuit.parameters.get(gate.get("parameter", ""), 0))
                braket_circuit.rx(qubits[0], angle)
            elif gate_type == "ry":
                angle = gate.get("angle", circuit.parameters.get(gate.get("parameter", ""), 0))
                braket_circuit.ry(qubits[0], angle)
            elif gate_type == "rz":
                angle = gate.get("angle", circuit.parameters.get(gate.get("parameter", ""), 0))
                braket_circuit.rz(qubits[0], angle)
            else:
                print(f"Warning: Unsupported gate type {gate_type}")
        
        return braket_circuit
    
    def execute_circuit(
        self, 
        compiled_circuit: BraketCircuit, 
        shots: int = 1000,
        **kwargs
    ) -> QuantumResult:
        """Execute compiled circuit on AWS Braket device."""
        if not self.is_connected:
            raise RuntimeError("Not connected to AWS Braket")
        
        start_time = time.time()
        
        try:
            device_arn = kwargs.get("device_arn")
            if not device_arn:
                raise ValueError("device_arn required for AWS Braket execution")
            
            # Get the device
            device = AwsDevice(device_arn)
            
            # Execute the circuit
            task = device.run(
                compiled_circuit,
                shots=shots,
                s3_destination_folder=(self.s3_bucket, self.s3_prefix)
            )
            
            # Wait for completion
            result = task.result()
            
            # Extract measurement counts
            counts = {}
            if hasattr(result, 'measurement_counts'):
                counts = dict(result.measurement_counts)
            elif hasattr(result, 'measurements'):
                # Process raw measurements to counts
                measurements = result.measurements
                for measurement in measurements:
                    bitstring = ''.join(str(int(bit)) for bit in measurement)
                    counts[bitstring] = counts.get(bitstring, 0) + 1
            
            execution_time = time.time() - start_time
            
            return QuantumResult(
                backend_type=self.backend_type,
                job_id=task.id,
                counts=counts,
                execution_time=execution_time,
                shots=shots,
                success=True,
                raw_result=result
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QuantumResult(
                backend_type=self.backend_type,
                job_id="failed",
                counts={},
                execution_time=execution_time,
                shots=shots,
                success=False,
                error_message=str(e)
            )
    
    def get_job_status(self, job_id: str) -> str:
        """Get status of a submitted job."""
        try:
            # Get task by ID and check status
            task = AwsDevice.get_task(job_id)
            return task.state().lower()
        except Exception:
            return "not_found"
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a submitted job."""
        try:
            task = AwsDevice.get_task(job_id)
            task.cancel()
            return True
        except Exception:
            return False


class AWSBraketHybridOptimizer:
    """Hybrid classical-quantum optimizer using AWS Braket."""
    
    def __init__(self, backend: AWSBraketBackend, device_arn: str):
        self.backend = backend
        self.device_arn = device_arn
        
    def create_qaoa_circuit(self, graph_edges: List[tuple], gamma: float, beta: float) -> QuantumCircuit:
        """Create QAOA circuit for graph optimization problems."""
        from .base import QuantumCircuitBuilder
        
        # Determine number of nodes
        nodes = set()
        for edge in graph_edges:
            nodes.update(edge)
        num_qubits = len(nodes)
        
        builder = QuantumCircuitBuilder(num_qubits)
        
        # Initial state: uniform superposition
        for q in range(num_qubits):
            builder.h(q)
        
        # Problem Hamiltonian (Cost layer)
        for edge in graph_edges:
            q1, q2 = edge
            builder.cnot(q1, q2)
            builder.rz(q2, 2 * gamma)
            builder.cnot(q1, q2)
        
        # Mixer Hamiltonian (Driver layer)
        for q in range(num_qubits):
            builder.rx(q, 2 * beta)
        
        builder.measure_all()
        return builder.build()
    
    def optimize_federated_graph(
        self,
        client_graphs: List[List[tuple]],
        max_iterations: int = 20
    ) -> Dict[str, Any]:
        """Optimize federated graph problems using QAOA on AWS Braket."""
        
        # Combine client graphs into global graph
        global_edges = []
        for client_graph in client_graphs:
            global_edges.extend(client_graph)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_edges = []
        for edge in global_edges:
            edge_tuple = tuple(sorted(edge))
            if edge_tuple not in seen:
                seen.add(edge_tuple)
                unique_edges.append(edge)
        
        best_cost = float('inf')
        best_params = None
        best_solution = None
        
        # QAOA optimization loop
        for iteration in range(max_iterations):
            # Random parameter initialization
            gamma = np.random.uniform(0, np.pi)
            beta = np.random.uniform(0, np.pi/2)
            
            # Create and execute QAOA circuit
            circuit = self.create_qaoa_circuit(unique_edges, gamma, beta)
            compiled = self.backend.compile_circuit(circuit, self.device_arn)
            result = self.backend.execute_circuit(
                compiled, 
                shots=1000,
                device_arn=self.device_arn
            )
            
            if result.success:
                # Evaluate cost function
                cost = self._evaluate_cut_cost(result.counts, unique_edges)
                
                if cost < best_cost:
                    best_cost = cost
                    best_params = (gamma, beta)
                    best_solution = max(result.counts.items(), key=lambda x: x[1])[0]
        
        return {
            "best_cost": best_cost,
            "best_parameters": best_params,
            "best_solution": best_solution,
            "optimization_completed": True
        }
    
    def _evaluate_cut_cost(self, counts: Dict[str, int], edges: List[tuple]) -> float:
        """Evaluate average cut cost from measurement results."""
        total_cost = 0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            cut_value = 0
            for edge in edges:
                q1, q2 = edge
                if q1 < len(bitstring) and q2 < len(bitstring):
                    if bitstring[q1] != bitstring[q2]:  # Edge is cut
                        cut_value += 1
            
            # We want to maximize cuts (minimize negative cut value)
            total_cost -= cut_value * (count / total_shots)
        
        return total_cost


class AWSBraketQuantumML:
    """Quantum Machine Learning algorithms on AWS Braket."""
    
    def __init__(self, backend: AWSBraketBackend, device_arn: str):
        self.backend = backend
        self.device_arn = device_arn
    
    def create_variational_classifier(self, features: int, layers: int = 2) -> QuantumCircuit:
        """Create variational quantum classifier circuit."""
        from .base import QuantumCircuitBuilder
        
        qubits = max(4, features)  # Ensure minimum qubits for expressivity
        builder = QuantumCircuitBuilder(qubits)
        
        # Data encoding layer
        for i in range(min(features, qubits)):
            builder.parametric_gate("ry", [i], f"data_{i}")
        
        # Variational layers
        for layer in range(layers):
            # Parameterized rotations
            for q in range(qubits):
                builder.parametric_gate("ry", [q], f"theta_{layer}_{q}")
                builder.parametric_gate("rz", [q], f"phi_{layer}_{q}")
            
            # Entangling gates
            for q in range(qubits - 1):
                builder.cnot(q, (q + 1) % qubits)
        
        # Measurement on first qubit for classification
        builder.measure(0)
        
        return builder.build()
    
    def quantum_federated_classification(
        self,
        client_data: List[np.ndarray],
        client_labels: List[np.ndarray],
        training_iterations: int = 100
    ) -> Dict[str, Any]:
        """Perform quantum federated learning for classification."""
        
        feature_dim = client_data[0].shape[1] if client_data else 4
        circuit = self.create_variational_classifier(feature_dim)
        
        # Initialize variational parameters
        num_params = len([p for p in circuit.parameters.keys() if "theta" in p or "phi" in p])
        params = np.random.uniform(0, 2*np.pi, num_params)
        
        training_results = []
        
        for iteration in range(training_iterations):
            client_gradients = []
            
            # Compute gradients for each client
            for client_idx, (data, labels) in enumerate(zip(client_data, client_labels)):
                gradient = self._compute_quantum_gradient(circuit, params, data, labels)
                client_gradients.append(gradient)
            
            # Aggregate gradients (simple average for now)
            avg_gradient = np.mean(client_gradients, axis=0)
            
            # Update parameters
            learning_rate = 0.1
            params -= learning_rate * avg_gradient
            
            # Evaluate performance
            accuracy = self._evaluate_quantum_classifier(circuit, params, client_data, client_labels)
            training_results.append({
                "iteration": iteration,
                "accuracy": accuracy,
                "parameters": params.copy()
            })
        
        return {
            "final_parameters": params,
            "training_history": training_results,
            "final_accuracy": training_results[-1]["accuracy"] if training_results else 0.0
        }
    
    def _compute_quantum_gradient(
        self, 
        circuit: QuantumCircuit, 
        params: np.ndarray, 
        data: np.ndarray, 
        labels: np.ndarray
    ) -> np.ndarray:
        """Compute quantum gradients using parameter shift rule."""
        gradient = np.zeros_like(params)
        shift = np.pi / 2
        
        for i in range(len(params)):
            # Positive shift
            params_plus = params.copy()
            params_plus[i] += shift
            
            # Negative shift
            params_minus = params.copy()
            params_minus[i] -= shift
            
            # Compute expectation values (simplified)
            exp_plus = self._compute_expectation(circuit, params_plus, data, labels)
            exp_minus = self._compute_expectation(circuit, params_minus, data, labels)
            
            # Parameter shift rule
            gradient[i] = (exp_plus - exp_minus) / 2
        
        return gradient
    
    def _compute_expectation(
        self, 
        circuit: QuantumCircuit, 
        params: np.ndarray, 
        data: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """Compute expectation value for given parameters."""
        # Set circuit parameters
        param_idx = 0
        for param_name in circuit.parameters:
            if "theta" in param_name or "phi" in param_name:
                circuit.parameters[param_name] = params[param_idx]
                param_idx += 1
        
        # Set data parameters (simplified - would encode actual data)
        for i, param_name in enumerate(circuit.parameters):
            if "data_" in param_name and i < len(data[0]):
                circuit.parameters[param_name] = float(data[0][i])  # Use first data point
        
        try:
            # Execute quantum circuit
            compiled = self.backend.compile_circuit(circuit, self.device_arn)
            result = self.backend.execute_circuit(compiled, shots=100, device_arn=self.device_arn)
            
            if result.success and result.counts:
                # Simple expectation: probability of measuring |1⟩
                prob_one = sum(count for state, count in result.counts.items() 
                             if state.startswith('1')) / sum(result.counts.values())
                return prob_one
            
        except Exception as e:
            print(f"Error computing expectation: {e}")
        
        return 0.5  # Default neutral expectation
    
    def _evaluate_quantum_classifier(
        self, 
        circuit: QuantumCircuit, 
        params: np.ndarray,
        client_data: List[np.ndarray], 
        client_labels: List[np.ndarray]
    ) -> float:
        """Evaluate quantum classifier accuracy."""
        correct_predictions = 0
        total_predictions = 0
        
        for data, labels in zip(client_data, client_labels):
            for i in range(len(data)):
                expectation = self._compute_expectation(circuit, params, data[i:i+1], labels[i:i+1])
                prediction = 1 if expectation > 0.5 else 0
                
                if prediction == int(labels[i]):
                    correct_predictions += 1
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0