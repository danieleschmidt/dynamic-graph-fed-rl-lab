"""
Quantum error correction and noise mitigation for real quantum hardware.

Implements error correction codes and noise mitigation techniques to achieve
quantum advantage on NISQ devices. Enhanced with Generation 1 improvements
including advanced error correction protocols, real-time syndrome decoding,
and adaptive mitigation strategies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from abc import ABC, abstractmethod

from .base import QuantumBackend, QuantumCircuit, QuantumResult, QuantumCircuitBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorCorrectionCode(Enum):
    """Quantum error correction codes."""
    REPETITION_CODE = "repetition_code"
    SURFACE_CODE = "surface_code"
    STEANE_CODE = "steane_code"
    SHOR_CODE = "shor_code"
    COLOR_CODE = "color_code"
    TOPOLOGICAL_CODE = "topological_code"


class NoiseMitigationTechnique(Enum):
    """Noise mitigation techniques."""
    ZERO_NOISE_EXTRAPOLATION = "zero_noise_extrapolation"
    READOUT_ERROR_MITIGATION = "readout_error_mitigation"
    SYMMETRY_VERIFICATION = "symmetry_verification"
    VIRTUAL_DISTILLATION = "virtual_distillation"
    PROBABILISTIC_ERROR_CANCELLATION = "probabilistic_error_cancellation"
    MACHINE_LEARNING_DECODER = "machine_learning_decoder"
    ADAPTIVE_PROTOCOLS = "adaptive_protocols"


@dataclass
class ErrorCorrectionConfig:
    """Configuration for quantum error correction."""
    code_type: ErrorCorrectionCode
    code_distance: int
    syndrome_extraction_rounds: int
    error_threshold: float
    mitigation_techniques: List[NoiseMitigationTechnique]
    use_logical_qubits: bool
    adaptive_threshold: bool = True
    real_time_decoding: bool = True
    ml_decoder_enabled: bool = False
    performance_monitoring: bool = True
    error_rate_tracking: Dict[str, float] = field(default_factory=dict)
    syndrome_history_size: int = 100
    correction_confidence_threshold: float = 0.8


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
    """Main quantum error correction orchestrator with Generation 1 enhancements."""
    
    def __init__(self, config: ErrorCorrectionConfig):
        self.config = config
        self.error_correction_code = self._initialize_error_correction_code()
        self.mitigation_techniques = self._initialize_mitigation_techniques()
        
        # Generation 1 enhancements
        self.adaptive_correction = AdaptiveErrorCorrection(config) if config.real_time_decoding else None
        self.performance_monitor = {
            'total_executions': 0,
            'total_errors_corrected': 0,
            'average_fidelity': 0.0,
            'execution_history': []
        }
        
        logger.info(f"Initialized quantum error correction with {config.code_type.value}, distance {config.code_distance}")
        
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
            elif technique == NoiseMitigationTechnique.MACHINE_LEARNING_DECODER:
                techniques["ml_decoder"] = MLSyndromeDecoder(self.config.code_distance)
        
        return techniques
    
    def apply_error_correction(
        self,
        backend: QuantumBackend,
        logical_circuit: QuantumCircuit,
        device: str,
        shots: int = 1000
    ) -> QuantumResult:
        """Apply error correction to logical circuit with Generation 1 enhancements."""
        
        start_time = time.time()
        execution_id = f"exec_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"Starting error correction execution {execution_id}")
            
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
                        backend, device, protected_circuit.qubits, shots=min(500, shots)
                    )
                
                result = self.mitigation_techniques["readout"].mitigate_readout_errors(result)
            
            # Real-time adaptive correction if enabled
            if self.adaptive_correction and result.success:
                syndrome = self._extract_syndrome_from_result(result)
                if syndrome:
                    corrections, performance_report = self.adaptive_correction.process_syndrome(syndrome)
                    
                    # Apply corrections if high confidence
                    if performance_report.get('confidence', 0) > self.config.correction_confidence_threshold:
                        result = self._apply_corrections_to_result(result, corrections)
                    
                    logger.info(f"Adaptive correction applied {len(corrections)} corrections with confidence {performance_report.get('confidence', 0):.3f}")
            
            # Update performance monitoring
            execution_time = time.time() - start_time
            self._update_performance_monitoring(execution_id, result, execution_time)
            
            # Log execution summary
            logger.info(f"Error correction execution {execution_id} completed in {execution_time:.3f}s, success: {result.success}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in error correction execution {execution_id}: {e}")
            # Return original circuit result as fallback
            try:
                compiled = backend.compile_circuit(logical_circuit, device)
                return backend.execute_circuit(compiled, shots)
            except:
                return QuantumResult(
                    backend_type=backend.backend_type,
                    job_id=f"failed_{execution_id}",
                    counts={},
                    execution_time=time.time() - start_time,
                    shots=shots,
                    success=False,
                    error_message=str(e)
                )
    
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
    
    def _extract_syndrome_from_result(self, result: QuantumResult) -> Optional[List[int]]:
        """Extract syndrome information from quantum execution result."""
        try:
            if not result.success or not result.counts:
                return None
            
            # Simple syndrome extraction from measurement results
            # In real implementation, this would be more sophisticated
            syndrome = []
            
            # Extract syndrome from measurement patterns
            for bitstring, count in result.counts.items():
                if count > 0:
                    # Simple parity check syndrome
                    parity = sum(int(bit) for bit in bitstring) % 2
                    syndrome.append(parity)
            
            # Pad syndrome to expected length
            expected_length = self.config.code_distance - 1
            while len(syndrome) < expected_length:
                syndrome.append(0)
            
            return syndrome[:expected_length]
            
        except Exception as e:
            logger.error(f"Error extracting syndrome: {e}")
            return None
    
    def _apply_corrections_to_result(
        self, 
        result: QuantumResult, 
        corrections: List[Tuple[str, int]]
    ) -> QuantumResult:
        """Apply error corrections to quantum result."""
        try:
            if not corrections:
                return result
            
            corrected_counts = {}
            
            for bitstring, count in result.counts.items():
                corrected_bitstring = list(bitstring)
                
                # Apply corrections
                for gate_type, qubit in corrections:
                    if qubit < len(corrected_bitstring):
                        if gate_type == "x":
                            # Flip bit
                            corrected_bitstring[qubit] = '1' if corrected_bitstring[qubit] == '0' else '0'
                        elif gate_type == "z":
                            # Phase flip (no change to computational basis measurement)
                            pass
                
                corrected_key = ''.join(corrected_bitstring)
                corrected_counts[corrected_key] = corrected_counts.get(corrected_key, 0) + count
            
            # Create corrected result
            return QuantumResult(
                backend_type=result.backend_type,
                job_id=result.job_id + "_corrected",
                counts=corrected_counts,
                execution_time=result.execution_time,
                shots=result.shots,
                success=result.success,
                error_message=result.error_message,
                raw_result=result.raw_result
            )
            
        except Exception as e:
            logger.error(f"Error applying corrections: {e}")
            return result
    
    def _update_performance_monitoring(self, execution_id: str, result: QuantumResult, execution_time: float):
        """Update performance monitoring metrics."""
        try:
            self.performance_monitor['total_executions'] += 1
            
            # Track execution in history
            execution_record = {
                'id': execution_id,
                'timestamp': time.time(),
                'execution_time': execution_time,
                'success': result.success,
                'shots': result.shots,
                'counts': len(result.counts) if result.counts else 0
            }
            
            self.performance_monitor['execution_history'].append(execution_record)
            
            # Keep only recent history
            if len(self.performance_monitor['execution_history']) > 100:
                self.performance_monitor['execution_history'].pop(0)
            
            # Calculate average fidelity (simplified metric)
            if result.success and result.counts:
                # Simple fidelity estimate based on measurement distribution
                total_counts = sum(result.counts.values())
                max_count = max(result.counts.values()) if result.counts else 0
                fidelity_estimate = max_count / total_counts if total_counts > 0 else 0.0
                
                current_avg = self.performance_monitor['average_fidelity']
                self.performance_monitor['average_fidelity'] = (
                    0.9 * current_avg + 0.1 * fidelity_estimate
                )
            
        except Exception as e:
            logger.error(f"Error updating performance monitoring: {e}")
    
    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance and diagnostics report."""
        try:
            report = {
                'error_correction_config': {
                    'code_type': self.config.code_type.value,
                    'code_distance': self.config.code_distance,
                    'syndrome_extraction_rounds': self.config.syndrome_extraction_rounds,
                    'adaptive_threshold': self.config.adaptive_threshold,
                    'ml_decoder_enabled': self.config.ml_decoder_enabled
                },
                'performance_metrics': self.performance_monitor.copy(),
                'mitigation_techniques_active': list(self.mitigation_techniques.keys()),
                'resource_overhead': self.get_error_correction_overhead()
            }
            
            # Add adaptive correction performance if available
            if self.adaptive_correction:
                report['adaptive_correction_performance'] = self.adaptive_correction.get_performance_summary()
            
            # Calculate success rate
            recent_executions = self.performance_monitor['execution_history'][-20:]
            if recent_executions:
                success_count = sum(1 for exec_record in recent_executions if exec_record['success'])
                report['recent_success_rate'] = success_count / len(recent_executions)
            else:
                report['recent_success_rate'] = 0.0
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}


class SyndromeDecoder(ABC):
    """Abstract base class for syndrome decoders."""
    
    @abstractmethod
    def decode_syndrome(self, syndrome: List[int], **kwargs) -> List[Tuple[str, int]]:
        """Decode syndrome to determine error corrections."""
        pass
    
    @abstractmethod
    def update_decoder(self, syndrome_history: List[List[int]], corrections: List[List[Tuple[str, int]]]):
        """Update decoder based on syndrome history and applied corrections."""
        pass


class MLSyndromeDecoder(SyndromeDecoder):
    """Machine learning-based syndrome decoder with adaptive capabilities."""
    
    def __init__(self, code_distance: int, learning_rate: float = 0.01):
        self.code_distance = code_distance
        self.learning_rate = learning_rate
        self.syndrome_history = []
        self.correction_history = []
        self.error_patterns = {}
        self.prediction_accuracy = 0.0
        
        # Initialize simple neural network weights (simplified implementation)
        self.weights = jnp.random.normal(0, 0.1, (code_distance * 2, code_distance))
        self.bias = jnp.zeros(code_distance)
        
    def decode_syndrome(self, syndrome: List[int], **kwargs) -> List[Tuple[str, int]]:
        """Decode syndrome using ML model."""
        try:
            start_time = time.time()
            
            # Convert syndrome to feature vector
            syndrome_array = jnp.array(syndrome, dtype=jnp.float32)
            
            # Pad syndrome if necessary
            if len(syndrome_array) < self.code_distance * 2:
                padding = jnp.zeros(self.code_distance * 2 - len(syndrome_array))
                syndrome_array = jnp.concatenate([syndrome_array, padding])
            elif len(syndrome_array) > self.code_distance * 2:
                syndrome_array = syndrome_array[:self.code_distance * 2]
            
            # Forward pass through simple neural network
            hidden = jnp.tanh(jnp.dot(syndrome_array, self.weights) + self.bias)
            error_probabilities = jnp.sigmoid(hidden)
            
            # Determine corrections based on probability threshold
            corrections = []
            confidence_threshold = kwargs.get('confidence_threshold', 0.5)
            
            for i, prob in enumerate(error_probabilities):
                if prob > confidence_threshold:
                    corrections.append(("x", i))
            
            # Log performance metrics
            decode_time = time.time() - start_time
            logger.info(f"ML decoder processed syndrome in {decode_time:.4f}s with {len(corrections)} corrections")
            
            return corrections
            
        except Exception as e:
            logger.error(f"ML decoder error: {e}")
            # Fallback to simple lookup decoder
            return self._fallback_decode(syndrome)
    
    def _fallback_decode(self, syndrome: List[int]) -> List[Tuple[str, int]]:
        """Fallback decoder for when ML model fails."""
        corrections = []
        for i, bit in enumerate(syndrome):
            if bit == 1:
                corrections.append(("x", i))
        return corrections
    
    def update_decoder(self, syndrome_history: List[List[int]], corrections: List[List[Tuple[str, int]]]):
        """Update ML model based on historical data."""
        try:
            if len(syndrome_history) != len(corrections):
                return
            
            # Update learning from recent examples
            for syndrome, correction in zip(syndrome_history[-10:], corrections[-10:]):
                self._update_weights(syndrome, correction)
            
            # Update prediction accuracy
            self._evaluate_accuracy(syndrome_history, corrections)
            
            logger.info(f"Updated ML decoder, accuracy: {self.prediction_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating ML decoder: {e}")
    
    def _update_weights(self, syndrome: List[int], corrections: List[Tuple[str, int]]):
        """Update neural network weights based on training example."""
        try:
            # Convert to arrays
            syndrome_array = jnp.array(syndrome, dtype=jnp.float32)
            
            # Pad syndrome if necessary
            if len(syndrome_array) < self.code_distance * 2:
                padding = jnp.zeros(self.code_distance * 2 - len(syndrome_array))
                syndrome_array = jnp.concatenate([syndrome_array, padding])
            elif len(syndrome_array) > self.code_distance * 2:
                syndrome_array = syndrome_array[:self.code_distance * 2]
            
            # Create target vector
            target = jnp.zeros(self.code_distance)
            for _, qubit in corrections:
                if qubit < self.code_distance:
                    target = target.at[qubit].set(1.0)
            
            # Forward pass
            hidden = jnp.tanh(jnp.dot(syndrome_array, self.weights) + self.bias)
            output = jnp.sigmoid(hidden)
            
            # Compute gradients and update weights (simplified gradient descent)
            error = target - output
            weight_gradient = jnp.outer(syndrome_array, error * output * (1 - output))
            bias_gradient = error * output * (1 - output)
            
            self.weights += self.learning_rate * weight_gradient
            self.bias += self.learning_rate * bias_gradient
            
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
    
    def _evaluate_accuracy(self, syndrome_history: List[List[int]], corrections: List[List[Tuple[str, int]]]):
        """Evaluate prediction accuracy on recent examples."""
        if len(syndrome_history) < 5:
            return
        
        correct_predictions = 0
        total_predictions = 0
        
        for syndrome, actual_corrections in zip(syndrome_history[-20:], corrections[-20:]):
            predicted_corrections = self.decode_syndrome(syndrome, confidence_threshold=0.5)
            
            # Simple accuracy metric: fraction of corrections that match
            actual_qubits = set(qubit for _, qubit in actual_corrections)
            predicted_qubits = set(qubit for _, qubit in predicted_corrections)
            
            if actual_qubits == predicted_qubits:
                correct_predictions += 1
            total_predictions += 1
        
        if total_predictions > 0:
            self.prediction_accuracy = correct_predictions / total_predictions


class AdaptiveErrorCorrection:
    """Adaptive error correction with real-time performance optimization."""
    
    def __init__(self, config: ErrorCorrectionConfig):
        self.config = config
        self.syndrome_history = []
        self.correction_history = []
        self.error_rates = {}
        self.performance_metrics = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'average_decode_time': 0.0,
            'syndrome_patterns': {}
        }
        
        # Initialize decoders
        self.decoders = {}
        if config.ml_decoder_enabled:
            self.decoders['ml'] = MLSyndromeDecoder(config.code_distance)
        
    def process_syndrome(
        self, 
        syndrome: List[int], 
        timestamp: Optional[float] = None
    ) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
        """Process syndrome with adaptive decoding and performance tracking."""
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.time()
        
        try:
            # Update syndrome history
            self.syndrome_history.append((syndrome, timestamp))
            if len(self.syndrome_history) > self.config.syndrome_history_size:
                self.syndrome_history.pop(0)
            
            # Choose decoder based on performance
            decoder_name, corrections = self._select_and_decode(syndrome)
            
            # Update correction history
            self.correction_history.append(corrections)
            if len(self.correction_history) > self.config.syndrome_history_size:
                self.correction_history.pop(0)
            
            # Update performance metrics
            decode_time = time.time() - start_time
            self._update_performance_metrics(syndrome, corrections, decode_time, decoder_name)
            
            # Adaptive threshold adjustment
            if self.config.adaptive_threshold:
                self._adjust_error_threshold(syndrome, corrections)
            
            # Generate performance report
            performance_report = {
                'decoder_used': decoder_name,
                'decode_time': decode_time,
                'num_corrections': len(corrections),
                'confidence': self._calculate_correction_confidence(syndrome, corrections),
                'error_rate_estimate': self.error_rates.get('current', 0.0)
            }
            
            logger.info(f"Processed syndrome with {len(corrections)} corrections using {decoder_name} decoder")
            
            return corrections, performance_report
            
        except Exception as e:
            logger.error(f"Error processing syndrome: {e}")
            return [], {'error': str(e)}
    
    def _select_and_decode(self, syndrome: List[int]) -> Tuple[str, List[Tuple[str, int]]]:
        """Select best decoder and decode syndrome."""
        # Use ML decoder if available and trained
        if 'ml' in self.decoders and self.decoders['ml'].prediction_accuracy > 0.7:
            try:
                corrections = self.decoders['ml'].decode_syndrome(
                    syndrome, 
                    confidence_threshold=self.config.correction_confidence_threshold
                )
                return 'ml', corrections
            except Exception as e:
                logger.warning(f"ML decoder failed: {e}, falling back to classical")
        
        # Fallback to classical lookup decoder
        corrections = self._classical_decode(syndrome)
        return 'classical', corrections
    
    def _classical_decode(self, syndrome: List[int]) -> List[Tuple[str, int]]:
        """Classical syndrome decoder with pattern matching."""
        corrections = []
        
        # Simple pattern matching for common error patterns
        syndrome_pattern = tuple(syndrome)
        
        if syndrome_pattern in self.performance_metrics['syndrome_patterns']:
            # Use previously successful correction pattern
            pattern_info = self.performance_metrics['syndrome_patterns'][syndrome_pattern]
            if pattern_info['success_rate'] > 0.8:
                return pattern_info['corrections']
        
        # Default decoding logic
        for i, bit in enumerate(syndrome):
            if bit == 1:
                corrections.append(("x", i))
        
        return corrections
    
    def _update_performance_metrics(
        self, 
        syndrome: List[int], 
        corrections: List[Tuple[str, int]], 
        decode_time: float,
        decoder_name: str
    ):
        """Update performance tracking metrics."""
        self.performance_metrics['total_corrections'] += len(corrections)
        
        # Update decode time moving average
        current_avg = self.performance_metrics['average_decode_time']
        self.performance_metrics['average_decode_time'] = (
            0.9 * current_avg + 0.1 * decode_time
        )
        
        # Track syndrome patterns
        syndrome_pattern = tuple(syndrome)
        if syndrome_pattern not in self.performance_metrics['syndrome_patterns']:
            self.performance_metrics['syndrome_patterns'][syndrome_pattern] = {
                'corrections': corrections,
                'count': 0,
                'success_rate': 0.5,
                'decoder_used': decoder_name
            }
        
        pattern_info = self.performance_metrics['syndrome_patterns'][syndrome_pattern]
        pattern_info['count'] += 1
        pattern_info['decoder_used'] = decoder_name
    
    def _calculate_correction_confidence(
        self, 
        syndrome: List[int], 
        corrections: List[Tuple[str, int]]
    ) -> float:
        """Calculate confidence in correction decisions."""
        if not corrections:
            return 1.0 if sum(syndrome) == 0 else 0.1
        
        # Base confidence on syndrome consistency and historical patterns
        syndrome_pattern = tuple(syndrome)
        
        if syndrome_pattern in self.performance_metrics['syndrome_patterns']:
            pattern_info = self.performance_metrics['syndrome_patterns'][syndrome_pattern]
            return pattern_info['success_rate']
        
        # Default confidence based on syndrome weight
        syndrome_weight = sum(syndrome)
        if syndrome_weight == 0:
            return 1.0
        elif syndrome_weight <= self.config.code_distance // 2:
            return 0.8
        else:
            return 0.4
    
    def _adjust_error_threshold(self, syndrome: List[int], corrections: List[Tuple[str, int]]):
        """Adaptively adjust error correction threshold."""
        try:
            current_error_rate = len(corrections) / max(1, len(syndrome))
            
            # Update moving average of error rate
            if 'current' not in self.error_rates:
                self.error_rates['current'] = current_error_rate
            else:
                self.error_rates['current'] = (
                    0.9 * self.error_rates['current'] + 0.1 * current_error_rate
                )
            
            # Adjust threshold based on error trend
            if self.error_rates['current'] > self.config.error_threshold * 1.5:
                self.config.correction_confidence_threshold = min(
                    0.9, self.config.correction_confidence_threshold + 0.05
                )
            elif self.error_rates['current'] < self.config.error_threshold * 0.5:
                self.config.correction_confidence_threshold = max(
                    0.3, self.config.correction_confidence_threshold - 0.05
                )
                
        except Exception as e:
            logger.error(f"Error adjusting threshold: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'total_syndromes_processed': len(self.syndrome_history),
            'total_corrections_applied': self.performance_metrics['total_corrections'],
            'average_decode_time': self.performance_metrics['average_decode_time'],
            'current_error_rate': self.error_rates.get('current', 0.0),
            'unique_syndrome_patterns': len(self.performance_metrics['syndrome_patterns']),
            'ml_decoder_accuracy': self.decoders['ml'].prediction_accuracy if 'ml' in self.decoders else None,
            'adaptive_threshold': self.config.correction_confidence_threshold
        }