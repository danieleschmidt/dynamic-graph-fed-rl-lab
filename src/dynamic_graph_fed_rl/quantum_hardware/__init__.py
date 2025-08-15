"""
Quantum Hardware Integration Module

Real quantum computing hardware integration moving beyond quantum-inspired
algorithms to true quantum advantage.

Supports:
- IBM Quantum (Qiskit)
- Google Quantum AI (Cirq)  
- AWS Braket
"""

from .base import QuantumBackend, QuantumCircuit, QuantumResult
from .ibm_quantum import IBMQuantumBackend
from .google_quantum import GoogleQuantumBackend  
from .aws_braket import AWSBraketBackend
from .quantum_fed_learning import QuantumFederatedLearning
from .hybrid_optimizer import HybridClassicalQuantumOptimizer
from .error_correction import QuantumErrorCorrection
from .benchmarking import QuantumAdvantageBenchmark
from .quantum_accelerated_optimizer import (
    QuantumAcceleratedOptimizer, 
    QuantumAccelerationConfig, 
    QuantumOptimizationResult,
    QuantumAccelerationType
)

__all__ = [
    "QuantumBackend",
    "QuantumCircuit", 
    "QuantumResult",
    "IBMQuantumBackend",
    "GoogleQuantumBackend",
    "AWSBraketBackend",
    "QuantumFederatedLearning",
    "HybridClassicalQuantumOptimizer",
    "QuantumErrorCorrection",
    "QuantumAdvantageBenchmark",
    "QuantumAcceleratedOptimizer",
    "QuantumAccelerationConfig",
    "QuantumOptimizationResult",
    "QuantumAccelerationType",
]