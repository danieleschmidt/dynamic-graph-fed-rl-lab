"""
Research Module for Novel Federated Graph RL Contributions

This module contains cutting-edge research implementations for:
1. Quantum Coherence Optimization in Federated Graph Learning
2. Adversarial Robustness in Multi-Scale Dynamic Graph Environments  
3. Communication-Efficient Temporal Graph Compression

Each research direction addresses critical gaps in current literature
and has strong potential for top-tier conference publications.
"""

from .quantum_coherence import (
    QuantumCoherenceAggregator,
    EntanglementWeightedFederation,
    SuperpositionAveraging,
)

from .adversarial_robustness import (
    MultiScaleAdversarialDefense,
    TemporalGraphAttackSuite,
    CertifiedRobustnessAnalyzer,
)

from .communication_efficiency import (
    TemporalGraphCompressor,
    QuantumSparsificationProtocol,
    AdaptiveBandwidthManager,
)

from .experimental_framework import (
    ResearchExperimentRunner,
    BaselineComparator,
    StatisticalValidator,
)

__all__ = [
    # Quantum Coherence Research
    "QuantumCoherenceAggregator",
    "EntanglementWeightedFederation", 
    "SuperpositionAveraging",
    
    # Adversarial Robustness Research
    "MultiScaleAdversarialDefense",
    "TemporalGraphAttackSuite",
    "CertifiedRobustnessAnalyzer",
    
    # Communication Efficiency Research
    "TemporalGraphCompressor",
    "QuantumSparsificationProtocol", 
    "AdaptiveBandwidthManager",
    
    # Experimental Framework
    "ResearchExperimentRunner",
    "BaselineComparator",
    "StatisticalValidator",
]