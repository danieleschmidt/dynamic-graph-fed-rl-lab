"""
IBM Quantum backend implementation using Qiskit.

Provides integration with IBM Quantum platform for real quantum hardware execution.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from .base import QuantumBackend, QuantumBackendType, QuantumCircuit, QuantumResult

try:
    from qiskit import QuantumCircuit as QiskitQuantumCircuit
    from qiskit import transpile, execute, ClassicalRegister
    from qiskit.providers import JobStatus
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
    from qiskit_ibm_runtime.accounts import AccountManager
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class IBMQuantumBackend(QuantumBackend):
    """IBM Quantum backend using Qiskit Runtime."""
    
    def __init__(self):
        super().__init__(QuantumBackendType.IBM_QUANTUM)
        self.service: Optional[QiskitRuntimeService] = None
        self.selected_backend = None
        
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to IBM Quantum using API token."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available. Install with: pip install qiskit qiskit-ibm-runtime")
        
        try:
            # Save account if token provided
            if "token" in credentials:
                QiskitRuntimeService.save_account(
                    token=credentials["token"],
                    instance=credentials.get("instance", "ibm-q/open/main"),
                    overwrite=True
                )
            
            # Initialize service
            self.service = QiskitRuntimeService(
                instance=credentials.get("instance", "ibm-q/open/main")
            )
            
            self.is_connected = True
            return True
            
        except Exception as e:
            print(f"Failed to connect to IBM Quantum: {e}")
            return False
    
    def get_available_devices(self) -> List[Dict[str, Any]]:
        """Get available IBM Quantum devices."""
        if not self.is_connected or not self.service:
            return []
        
        devices = []
        for backend in self.service.backends():
            config = backend.configuration()
            devices.append({
                "name": backend.name,
                "qubits": config.n_qubits,
                "simulator": config.simulator,
                "operational": backend.status().operational,
                "pending_jobs": backend.status().pending_jobs,
                "supported_gates": list(config.basis_gates),
                "coupling_map": config.coupling_map.get_edges() if config.coupling_map else None,
                "quantum_volume": getattr(config, 'quantum_volume', None)
            })
        
        return devices
    
    def get_device_properties(self, device: str) -> Dict[str, Any]:
        """Get detailed properties of IBM Quantum device."""
        if not self.is_connected or not self.service:
            return {}
        
        try:
            backend = self.service.backend(device)
            config = backend.configuration()
            properties = backend.properties() if hasattr(backend, 'properties') else None
            
            device_props = {
                "name": device,
                "qubits": config.n_qubits,
                "simulator": config.simulator,
                "supported_gates": list(config.basis_gates),
                "coupling_map": config.coupling_map.get_edges() if config.coupling_map else None,
                "quantum_volume": getattr(config, 'quantum_volume', None),
                "max_shots": config.max_shots,
                "max_experiments": config.max_experiments,
            }
            
            if properties:
                # Add calibration data
                device_props.update({
                    "gate_times": {gate.gate: gate.parameters[0].value 
                                 for gate in properties.gates},
                    "gate_errors": {gate.gate: [param.value for param in gate.parameters 
                                              if param.name == 'gate_error']
                                  for gate in properties.gates},
                    "readout_errors": [qubit.readout_error for qubit in properties.qubits],
                    "t1_times": [qubit.T1 for qubit in properties.qubits],
                    "t2_times": [qubit.T2 for qubit in properties.qubits],
                })
            
            return device_props
            
        except Exception as e:
            print(f"Error getting device properties for {device}: {e}")
            return {}
    
    def compile_circuit(self, circuit: QuantumCircuit, device: str) -> QiskitQuantumCircuit:
        """Compile universal circuit to Qiskit format."""
        if not self.is_connected or not self.service:
            raise RuntimeError("Not connected to IBM Quantum")
        
        # Create Qiskit circuit
        qc = QiskitQuantumCircuit(circuit.qubits, len(circuit.measurements))
        
        # Add gates
        for gate in circuit.gates:
            gate_type = gate["type"]
            qubits = gate["qubits"]
            
            if gate_type == "h":
                qc.h(qubits[0])
            elif gate_type == "x":
                qc.x(qubits[0])
            elif gate_type == "y":
                qc.y(qubits[0])
            elif gate_type == "z":
                qc.z(qubits[0])
            elif gate_type == "cnot":
                qc.cx(qubits[0], qubits[1])
            elif gate_type == "rx":
                angle = gate.get("angle", circuit.parameters.get(gate.get("parameter", ""), 0))
                qc.rx(angle, qubits[0])
            elif gate_type == "ry":
                angle = gate.get("angle", circuit.parameters.get(gate.get("parameter", ""), 0))
                qc.ry(angle, qubits[0])
            elif gate_type == "rz":
                angle = gate.get("angle", circuit.parameters.get(gate.get("parameter", ""), 0))
                qc.rz(angle, qubits[0])
            else:
                print(f"Warning: Unsupported gate type {gate_type}")
        
        # Add measurements
        for i, qubit in enumerate(circuit.measurements):
            qc.measure(qubit, i)
        
        # Transpile for target backend
        backend = self.service.backend(device)
        compiled_circuit = transpile(qc, backend=backend, optimization_level=3)
        
        return compiled_circuit
    
    def execute_circuit(
        self, 
        compiled_circuit: QiskitQuantumCircuit, 
        shots: int = 1000,
        **kwargs
    ) -> QuantumResult:
        """Execute compiled circuit on IBM Quantum device."""
        if not self.is_connected or not self.service:
            raise RuntimeError("Not connected to IBM Quantum")
        
        start_time = time.time()
        
        try:
            # Set up runtime options
            options = Options()
            options.execution.shots = shots
            options.optimization_level = kwargs.get("optimization_level", 3)
            options.resilience_level = kwargs.get("resilience_level", 1)
            
            # Use Qiskit Runtime Sampler
            sampler = Sampler(options=options)
            
            # Execute circuit
            job = sampler.run([compiled_circuit])
            result = job.result()
            
            # Extract counts
            counts = {}
            if hasattr(result, 'quasi_dists') and result.quasi_dists:
                quasi_dist = result.quasi_dists[0]
                for bitstring, count in quasi_dist.items():
                    counts[format(bitstring, f'0{compiled_circuit.num_clbits}b')] = int(count * shots)
            
            execution_time = time.time() - start_time
            
            return QuantumResult(
                backend_type=self.backend_type,
                job_id=job.job_id(),
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
        if not self.service:
            return "unknown"
        
        try:
            job = self.service.job(job_id)
            return job.status().name.lower()
        except Exception:
            return "not_found"
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a submitted job."""
        if not self.service:
            return False
        
        try:
            job = self.service.job(job_id)
            job.cancel()
            return True
        except Exception:
            return False
    
    def get_backend_queue_info(self, device: str) -> Dict[str, Any]:
        """Get queue information for a backend."""
        if not self.is_connected or not self.service:
            return {}
        
        try:
            backend = self.service.backend(device)
            status = backend.status()
            
            return {
                "operational": status.operational,
                "pending_jobs": status.pending_jobs,
                "status_msg": status.status_msg
            }
        except Exception:
            return {}


class IBMQuantumFederatedAggregator:
    """Quantum parameter aggregation using IBM Quantum hardware."""
    
    def __init__(self, backend: IBMQuantumBackend, device: str):
        self.backend = backend
        self.device = device
        
    def create_aggregation_circuit(self, num_clients: int, parameter_dim: int) -> QuantumCircuit:
        """Create quantum circuit for parameter aggregation."""
        # Determine number of qubits needed
        qubits_needed = max(num_clients, parameter_dim).bit_length()
        
        # Build aggregation circuit using quantum parallelism
        from .base import QuantumCircuitBuilder
        
        builder = QuantumCircuitBuilder(qubits_needed)
        
        # Initialize superposition state
        for q in range(qubits_needed):
            builder.h(q)
        
        # Add entanglement for correlation effects
        for i in range(qubits_needed - 1):
            builder.cnot(i, i + 1)
        
        # Parametric gates for client weighting
        for i in range(num_clients):
            qubit = i % qubits_needed
            builder.parametric_gate("ry", [qubit], f"client_{i}_weight")
        
        # Measurement
        builder.measure_all()
        
        return builder.build()
    
    def quantum_aggregate(
        self, 
        client_parameters: List[np.ndarray],
        quantum_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Perform quantum parameter aggregation."""
        num_clients = len(client_parameters)
        
        if quantum_weights is None:
            quantum_weights = np.ones(num_clients) / num_clients
        
        # Create and execute aggregation circuit
        circuit = self.create_aggregation_circuit(num_clients, client_parameters[0].size)
        
        # Set quantum weight parameters
        for i, weight in enumerate(quantum_weights):
            circuit.parameters[f"client_{i}_weight"] = float(weight) * np.pi
        
        # Execute on quantum hardware
        compiled = self.backend.compile_circuit(circuit, self.device)
        result = self.backend.execute_circuit(compiled, shots=1000)
        
        if not result.success:
            # Fallback to classical aggregation
            return np.average(client_parameters, axis=0, weights=quantum_weights)
        
        # Process quantum result to extract aggregated parameters
        probabilities = list(result.probabilities.values())
        
        # Use quantum measurement probabilities as aggregation weights
        quantum_aggregation_weights = np.array(probabilities[:num_clients])
        quantum_aggregation_weights = quantum_aggregation_weights / np.sum(quantum_aggregation_weights)
        
        # Weighted aggregation using quantum-derived weights
        aggregated = np.average(client_parameters, axis=0, weights=quantum_aggregation_weights)
        
        return aggregated