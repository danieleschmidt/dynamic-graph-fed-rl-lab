"""
Research Benchmarks for Dynamic Graph Federated Reinforcement Learning

Comprehensive benchmarking suite for evaluating novel research contributions:
1. Quantum Coherence Optimization in Federated Graph Learning
2. Adversarial Robustness in Multi-Scale Dynamic Graph Environments  
3. Communication-Efficient Temporal Graph Compression

Provides standardized datasets, evaluation protocols, and baseline implementations
for reproducible research comparisons.
"""

from .datasets import (
    QuantumCoherenceBenchmark,
    AdversarialRobustnessBenchmark, 
    CommunicationEfficiencyBenchmark,
    UnifiedResearchBenchmark
)

from .evaluation import (
    StandardizedEvaluator,
    StatisticalSignificanceTester,
    ReproducibilityValidator
)

from .baselines import (
    ClassicalFederatedBaselines,
    QuantumInspiredBaselines,
    AdversarialDefenseBaselines,
    CompressionBaselines
)

__all__ = [
    # Dataset benchmarks
    "QuantumCoherenceBenchmark",
    "AdversarialRobustnessBenchmark",
    "CommunicationEfficiencyBenchmark", 
    "UnifiedResearchBenchmark",
    
    # Evaluation tools
    "StandardizedEvaluator",
    "StatisticalSignificanceTester",
    "ReproducibilityValidator",
    
    # Baseline implementations
    "ClassicalFederatedBaselines",
    "QuantumInspiredBaselines", 
    "AdversarialDefenseBaselines",
    "CompressionBaselines",
]