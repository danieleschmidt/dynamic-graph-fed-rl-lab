"""
Dynamic Graph Federated Reinforcement Learning Lab

A cutting-edge federated reinforcement learning framework for controlling 
systems with time-evolving graph structures with autonomous SDLC capabilities.
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

from . import algorithms
from . import environments
from . import federation
from . import models
from . import utils
from . import quantum_planner
from . import monitoring
from . import optimization
from . import scaling
from . import global_deployment

# Core exports
from .algorithms.base import BaseGraphAlgorithm
from .environments.base import BaseGraphEnvironment, GraphState, GraphTransition
from .federation.base import BaseFederatedLearning

# Quantum planner exports
from .quantum_planner import QuantumTaskPlanner, QuantumTask, TaskSuperposition
from .quantum_planner import QuantumScheduler, QuantumOptimizer, QuantumExecutor

# Enhanced SDLC exports
from .monitoring import HealthMonitor, MetricsCollector
from .optimization import AutonomousTesting, Generation4System
from .scaling import HorizontalAutoscaler, PerformanceOptimizer
from .global_deployment import MultiRegionManager, ComplianceFramework

__all__ = [
    "algorithms",
    "environments", 
    "federation",
    "models",
    "utils",
    "quantum_planner",
    "monitoring",
    "optimization", 
    "scaling",
    "global_deployment",
    # Core algorithm exports
    "BaseGraphAlgorithm",
    "BaseGraphEnvironment",
    "GraphState",
    "GraphTransition",
    "BaseFederatedLearning",
    # Quantum planner core
    "QuantumTaskPlanner",
    "QuantumTask", 
    "TaskSuperposition",
    "QuantumScheduler",
    "QuantumOptimizer",
    "QuantumExecutor",
    # Enhanced SDLC
    "HealthMonitor",
    "MetricsCollector",
    "AutonomousTesting",
    "Generation4System",
    "HorizontalAutoscaler", 
    "PerformanceOptimizer",
    "MultiRegionManager",
    "ComplianceFramework",
]