"""
Quantum error correction and noise mitigation for real quantum hardware.

Implements error correction codes and noise mitigation techniques to achieve
quantum advantage on NISQ devices.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from enum import Enum

from .base import QuantumBackend, QuantumCircuit, QuantumResult, QuantumCircuitBuilder


class ErrorCorrectionCode(Enum):
    """Quantum error correction codes."""
    REPETITION_CODE = "repetition_code"
    SURFACE_CODE = "surface_code"
    STEANE_CODE = "steane_code"
    SHOR_CODE = "shor_code"


class NoiseMitigationTechnique(Enum):
    """Noise mitigation techniques."""
    ZERO_NOISE_EXTRAPOLATION = "zero_noise_extrapolation"
    READOUT_ERROR_MITIGATION = "readout_error_mitigation"
    SYMMETRY_VERIFICATION = "symmetry_verification"
    VIRTUAL_DISTILLATION = "virtual_distillation"


@dataclass
class ErrorCorrectionConfig:
    """Configuration for quantum error correction."""
    code_type: ErrorCorrectionCode
    code_distance: int
    syndrome_extraction_rounds: int
    error_threshold: float
    mitigation_techniques: List[NoiseMitigationTechnique]
    use_logical_qubits: bool


class RepetitionCode:
    """Repetition code for bit-flip error correction."""
    
    def __init__(self, code_distance: int):
        self.code_distance = code_distance
        self.num_physical_qubits = code_distance
        self.num_ancilla_qubits = code_distance - 1
        
    def encode_logical_qubit(self, logical_state: str = "0") -> QuantumCircuit:
        """Encode logical qubit using repetition code."""
        total_qubits = self.num_physical_qubits + self.num_ancilla_qubits
        builder = QuantumCircuitBuilder(total_qubits)
        
        # Initialize logical state
        if logical_state == "1":
            for q in range(self.num_physical_qubits):
                builder.x(q)
        elif logical_state == "+":
            for q in range(self.num_physical_qubits):
                builder.h(q)
        elif logical_state == "-":
            for q in range(self.num_physical_qubits):
                builder.h(q)
                builder.z(q)
        
        return builder.build()
    
    def syndrome_extraction(self, encoded_circuit: QuantumCircuit) -> QuantumCircuit:
        """Add syndrome extraction to detect bit-flip errors."""
        builder = QuantumCircuitBuilder(encoded_circuit.qubits)
        
        # Copy existing gates
        for gate in encoded_circuit.gates:
            builder.circuit.gates.append(gate)
        
        # Syndrome extraction using ancilla qubits
        ancilla_start = self.num_physical_qubits
        
        for i in range(self.num_ancilla_qubits):
            # CNOT between adjacent data qubits and ancilla
            builder.cnot(i, ancilla_start + i)
            builder.cnot(i + 1, ancilla_start + i)
        
        # Measure ancilla qubits for syndrome
        for i in range(self.num_ancilla_qubits):
            builder.measure(ancilla_start + i)
        
        return builder.build()
    
    def error_correction(self, syndrome: List[int]) -> List[Tuple[str, int]]:
        """Determine error correction operations from syndrome."""
        corrections = []
        
        for i, syndrome_bit in enumerate(syndrome):
            if syndrome_bit == 1:
                # Bit-flip error detected
                if i == 0:
                    corrections.append(("x", 0))  # Error on first qubit
                elif i == len(syndrome) - 1:
                    corrections.append(("x", i + 1))  # Error on last qubit
                else:
                    # Error between qubits i and i+1 - apply to qubit i
                    corrections.append(("x", i))
        
        return corrections


class SurfaceCode:
    """Surface code implementation for fault-tolerant quantum computing."""
    
    def __init__(self, code_distance: int):
        self.code_distance = code_distance
        # Surface code requires (d^2 + (d-1)^2) qubits
        self.num_data_qubits = code_distance ** 2
        self.num_ancilla_qubits = (code_distance - 1) ** 2 + (code_distance - 1) ** 2
        self.total_qubits = self.num_data_qubits + self.num_ancilla_qubits
        
    def create_surface_code_layout(self) -> Dict[str, List[Tuple[int, int]]]:
        """Create surface code qubit layout."""
        data_qubits = []
        x_ancillas = []
        z_ancillas = []
        
        # Create checkerboard pattern
        for row in range(self.code_distance):
            for col in range(self.code_distance):
                if (row + col) % 2 == 0:
                    data_qubits.append((row, col))
                else:
                    if row % 2 == 0:
                        x_ancillas.append((row, col))
                    else:
                        z_ancillas.append((row, col))
        
        return {
            "data_qubits": data_qubits,
            "x_ancillas": x_ancillas,
            "z_ancillas": z_ancillas
        }
    
    def syndrome_measurement_circuit(self) -> QuantumCircuit:
        """Create syndrome measurement circuit for surface code."""
        builder = QuantumCircuitBuilder(self.total_qubits)
        layout = self.create_surface_code_layout()
        
        # Map 2D coordinates to qubit indices
        qubit_map = {}
        qubit_idx = 0
        
        for pos in layout["data_qubits"] + layout["x_ancillas"] + layout["z_ancillas"]:
            qubit_map[pos] = qubit_idx
            qubit_idx += 1
        
        # X-type stabilizer measurements
        for x_ancilla in layout["x_ancillas"]:
            ancilla_idx = qubit_map[x_ancilla]
            builder.h(ancilla_idx)
            
            # Find neighboring data qubits
            neighbors = self._get_neighbors(x_ancilla, layout["data_qubits"])
            for neighbor in neighbors:
                data_idx = qubit_map[neighbor]
                builder.cnot(ancilla_idx, data_idx)
            
            builder.h(ancilla_idx)
            builder.measure(ancilla_idx)
        
        # Z-type stabilizer measurements
        for z_ancilla in layout["z_ancillas"]:
            ancilla_idx = qubit_map[z_ancilla]
            
            # Find neighboring data qubits
            neighbors = self._get_neighbors(z_ancilla, layout["data_qubits"])
            for neighbor in neighbors:
                data_idx = qubit_map[neighbor]
                builder.cnot(data_idx, ancilla_idx)
            
            builder.measure(ancilla_idx)
        
        return builder.build()
    
    def _get_neighbors(
        self, 
        position: Tuple[int, int], 
        valid_positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Get neighboring positions in surface code lattice."""
        row, col = position
        neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + dr, col + dc)
            if neighbor in valid_positions:
                neighbors.append(neighbor)
        
        return neighbors
    
    def decode_surface_code_syndrome(self, syndrome: List[int]) -> List[Tuple[str, int]]:
        """Decode surface code syndrome to determine error corrections."""
        # Simplified minimum weight perfect matching decoder
        corrections = []
        
        # Find syndrome positions
        syndrome_positions = [i for i, bit in enumerate(syndrome) if bit == 1]
        
        # Pair up syndromes (simplified - real decoder would use MWPM)
        for i in range(0, len(syndrome_positions), 2):
            if i + 1 < len(syndrome_positions):
                pos1, pos2 = syndrome_positions[i], syndrome_positions[i + 1]
                # Apply correction along path between syndromes
                corrections.append(("x", (pos1 + pos2) // 2))
        
        return corrections


class ZeroNoiseExtrapolation:
    """Zero noise extrapolation for error mitigation."""
    
    def __init__(self, noise_scaling_factors: List[float] = None):
        if noise_scaling_factors is None:
            self.noise_scaling_factors = [1.0, 1.5, 2.0, 2.5]
        else:
            self.noise_scaling_factors = noise_scaling_factors
    
    def scale_circuit_noise(self, circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
        """Scale noise in quantum circuit by inserting identity operations."""
        if scale_factor <= 1.0:
            return circuit
        
        builder = QuantumCircuitBuilder(circuit.qubits)
        
        for gate in circuit.gates:
            # Add original gate
            builder.circuit.gates.append(gate)
            
            # Add noise scaling (simplified - insert Pauli gate pairs)
            if scale_factor > 1.0:
                num_insertions = int((scale_factor - 1.0) * 2)
                for _ in range(num_insertions):
                    for qubit in gate["qubits"]:
                        # Insert X-X pair (identity with added noise)
                        builder.x(qubit)
                        builder.x(qubit)
        
        # Copy measurements
        for measurement in circuit.measurements:
            builder.measure(measurement)
        
        return builder.build()
    
    def extrapolate_to_zero_noise(
        self,
        backend: QuantumBackend,
        circuit: QuantumCircuit,
        device: str,
        shots: int = 1000
    ) -> QuantumResult:
        """Perform zero noise extrapolation."""
        expectation_values = []
        
        # Run circuit at different noise levels
        for scale_factor in self.noise_scaling_factors:
            scaled_circuit = self.scale_circuit_noise(circuit, scale_factor)
            compiled = backend.compile_circuit(scaled_circuit, device)
            result = backend.execute_circuit(compiled, shots)
            
            if result.success:
                # Compute expectation value (simplified)
                expectation = self._compute_expectation_value(result)
                expectation_values.append((scale_factor, expectation))
        
        if len(expectation_values) >= 2:
            # Extrapolate to zero noise using linear fit
            noise_levels = jnp.array([nv[0] for nv in expectation_values])
            expectations = jnp.array([nv[1] for nv in expectation_values])
            
            # Linear extrapolation: y = ax + b, extrapolate to x=0
            A = jnp.vstack([noise_levels, jnp.ones(len(noise_levels))]).T
            coeffs = jnp.linalg.lstsq(A, expectations, rcond=None)[0]
            zero_noise_expectation = coeffs[1]  # b coefficient
            
            # Create extrapolated result
            extrapolated_result = QuantumResult(
                backend_type=backend.backend_type,
                job_id="zero_noise_extrapolated",
                counts={"extrapolated": 1},
                execution_time=sum(result.execution_time for _, result in [
                    (_, backend.execute_circuit(
                        backend.compile_circuit(self.scale_circuit_noise(circuit, scale), device), 
                        shots
                    )) for scale in self.noise_scaling_factors[:2]
                ]),
                shots=shots,
                success=True,
                raw_result={"zero_noise_expectation": zero_noise_expectation}
            )
            
            return extrapolated_result
        
        # Fallback to original circuit result
        compiled = backend.compile_circuit(circuit, device)
        return backend.execute_circuit(compiled, shots)
    
    def _compute_expectation_value(self, result: QuantumResult) -> float:
        """Compute expectation value from quantum result."""
        # Simplified: compute <Z> expectation on first qubit
        zero_prob = sum(count for state, count in result.counts.items() 
                       if state.startswith('0')) / sum(result.counts.values())
        one_prob = 1 - zero_prob
        
        return zero_prob - one_prob  # <Z> = P(0) - P(1)


class ReadoutErrorMitigation:
    """Readout error mitigation using calibration matrices."""
    
    def __init__(self):
        self.calibration_matrices = {}
    
    def characterize_readout_errors(
        self,
        backend: QuantumBackend,
        device: str,
        num_qubits: int,
        shots: int = 1000
    ) -> Dict[int, jnp.ndarray]:
        """Characterize readout error rates for each qubit."""
        calibration_matrices = {}
        
        for qubit in range(num_qubits):
            # Measure |0⟩ state
            builder_0 = QuantumCircuitBuilder(num_qubits)
            builder_0.measure(qubit)
            circuit_0 = builder_0.build()
            
            compiled_0 = backend.compile_circuit(circuit_0, device)
            result_0 = backend.execute_circuit(compiled_0, shots)
            
            # Measure |1⟩ state
            builder_1 = QuantumCircuitBuilder(num_qubits)
            builder_1.x(qubit)
            builder_1.measure(qubit)
            circuit_1 = builder_1.build()
            
            compiled_1 = backend.compile_circuit(circuit_1, device)
            result_1 = backend.execute_circuit(compiled_1, shots)
            
            if result_0.success and result_1.success:
                # Extract readout fidelities
                p00 = sum(count for state, count in result_0.counts.items() 
                         if state[qubit] == '0') / sum(result_0.counts.values())
                p11 = sum(count for state, count in result_1.counts.items() 
                         if state[qubit] == '1') / sum(result_1.counts.values())
                
                p01 = 1 - p00
                p10 = 1 - p11
                
                # Create calibration matrix
                calibration_matrix = jnp.array([[p00, p01], [p10, p11]])
                calibration_matrices[qubit] = calibration_matrix
        
        self.calibration_matrices = calibration_matrices
        return calibration_matrices
    
    def mitigate_readout_errors(self, result: QuantumResult) -> QuantumResult:
        """Apply readout error mitigation to quantum result."""
        if not self.calibration_matrices:
            return result  # No calibration available
        
        mitigated_counts = {}
        
        for bitstring, count in result.counts.items():
            # For each measured bitstring, compute the corrected probability
            corrected_prob = count / sum(result.counts.values())
            
            # Apply inverse calibration (simplified single-qubit correction)
            for qubit, bit in enumerate(bitstring):
                if qubit in self.calibration_matrices:
                    calib_matrix = self.calibration_matrices[qubit]
                    
                    # Invert calibration matrix
                    try:
                        inv_calib = jnp.linalg.inv(calib_matrix)
                        
                        # Apply correction (simplified)
                        if bit == '0':
                            corrected_prob *= inv_calib[0, 0]
                        else:
                            corrected_prob *= inv_calib[1, 1]
                    except:
                        pass  # Skip if matrix not invertible
            
            mitigated_count = max(0, int(corrected_prob * sum(result.counts.values())))
            if mitigated_count > 0:
                mitigated_counts[bitstring] = mitigated_count
        
        # Create mitigated result
        return QuantumResult(
            backend_type=result.backend_type,
            job_id=result.job_id + "_mitigated",
            counts=mitigated_counts,
            execution_time=result.execution_time,
            shots=result.shots,
            success=result.success,
            error_message=result.error_message,
            raw_result=result.raw_result
        )


class QuantumErrorCorrection:
    """Main quantum error correction orchestrator."""
    
    def __init__(self, config: ErrorCorrectionConfig):
        self.config = config
        self.error_correction_code = self._initialize_error_correction_code()
        self.mitigation_techniques = self._initialize_mitigation_techniques()
        
    def _initialize_error_correction_code(self):
        """Initialize the specified error correction code."""
        if self.config.code_type == ErrorCorrectionCode.REPETITION_CODE:
            return RepetitionCode(self.config.code_distance)
        elif self.config.code_type == ErrorCorrectionCode.SURFACE_CODE:
            return SurfaceCode(self.config.code_distance)
        else:
            raise NotImplementedError(f"Error correction code {self.config.code_type} not implemented")
    
    def _initialize_mitigation_techniques(self) -> Dict[str, Any]:
        """Initialize noise mitigation techniques."""
        techniques = {}
        
        for technique in self.config.mitigation_techniques:
            if technique == NoiseMitigationTechnique.ZERO_NOISE_EXTRAPOLATION:
                techniques["zne"] = ZeroNoiseExtrapolation()
            elif technique == NoiseMitigationTechnique.READOUT_ERROR_MITIGATION:
                techniques["readout"] = ReadoutErrorMitigation()
        
        return techniques
    
    def apply_error_correction(
        self,
        backend: QuantumBackend,
        logical_circuit: QuantumCircuit,
        device: str,
        shots: int = 1000
    ) -> QuantumResult:
        """Apply error correction to logical circuit."""
        
        if self.config.use_logical_qubits:
            # Encode logical qubits
            encoded_circuit = self._encode_logical_circuit(logical_circuit)
            
            # Add syndrome extraction
            protected_circuit = self.error_correction_code.syndrome_extraction(encoded_circuit)
        else:
            protected_circuit = logical_circuit
        
        # Apply noise mitigation
        if "zne" in self.mitigation_techniques:
            result = self.mitigation_techniques["zne"].extrapolate_to_zero_noise(
                backend, protected_circuit, device, shots
            )
        else:
            compiled = backend.compile_circuit(protected_circuit, device)
            result = backend.execute_circuit(compiled, shots)
        
        # Apply readout error mitigation
        if "readout" in self.mitigation_techniques and result.success:
            # Characterize readout errors if not done
            if not self.mitigation_techniques["readout"].calibration_matrices:
                self.mitigation_techniques["readout"].characterize_readout_errors(
                    backend, device, protected_circuit.qubits
                )
            
            result = self.mitigation_techniques["readout"].mitigate_readout_errors(result)
        
        return result
    
    def _encode_logical_circuit(self, logical_circuit: QuantumCircuit) -> QuantumCircuit:
        """Encode logical circuit using error correction code."""
        # This is a simplified encoding - real implementation would be more complex
        if self.config.code_type == ErrorCorrectionCode.REPETITION_CODE:
            return self.error_correction_code.encode_logical_qubit("0")
        elif self.config.code_type == ErrorCorrectionCode.SURFACE_CODE:
            return self.error_correction_code.syndrome_measurement_circuit()
        
        return logical_circuit
    
    def get_error_correction_overhead(self) -> Dict[str, int]:
        """Get resource overhead for error correction."""
        if self.config.code_type == ErrorCorrectionCode.REPETITION_CODE:
            return {
                "physical_qubits": self.error_correction_code.num_physical_qubits,
                "ancilla_qubits": self.error_correction_code.num_ancilla_qubits,
                "total_qubits": self.error_correction_code.num_physical_qubits + self.error_correction_code.num_ancilla_qubits,
                "code_distance": self.config.code_distance
            }
        elif self.config.code_type == ErrorCorrectionCode.SURFACE_CODE:
            return {
                "data_qubits": self.error_correction_code.num_data_qubits,
                "ancilla_qubits": self.error_correction_code.num_ancilla_qubits,
                "total_qubits": self.error_correction_code.total_qubits,
                "code_distance": self.config.code_distance
            }
        
        return {"total_qubits": 0}