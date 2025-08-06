"""
Quantum-Inspired Task Planner

Core module implementing quantum principles for adaptive task planning:
- Superposition of task states
- Entanglement between dependent tasks  
- Quantum interference for optimization
- Measurement collapse for execution
"""

from .core import QuantumTaskPlanner, QuantumTask, TaskSuperposition
from .scheduler import QuantumScheduler, AdaptiveScheduler  
from .optimizer import QuantumOptimizer, InterferenceOptimizer
from .executor import QuantumExecutor, ParallelExecutor

__all__ = [
    "QuantumTaskPlanner",
    "QuantumTask", 
    "TaskSuperposition",
    "QuantumScheduler",
    "AdaptiveScheduler",
    "QuantumOptimizer",
    "InterferenceOptimizer", 
    "QuantumExecutor",
    "ParallelExecutor",
]