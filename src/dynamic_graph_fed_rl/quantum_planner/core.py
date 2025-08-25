"""
Core quantum-inspired task planning implementation.

Implements quantum principles for task management:
- Superposition: Tasks exist in multiple states simultaneously
- Entanglement: Dependencies create correlated task states
- Interference: Path optimization through quantum interference
- Measurement: Collapse to deterministic execution plan
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import random


class TaskState(Enum):
    """Quantum task states."""
    PENDING = "pending"
    ACTIVE = "active" 
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QuantumTask:
    """
    Quantum task with superposition of states and amplitudes.
    
    Each task exists in a superposition until measurement (execution).
    """
    id: str
    name: str
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 1.0
    priority: float = 1.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Quantum properties
    state_amplitudes: Dict[TaskState, complex] = field(default_factory=dict)
    entangled_tasks: Set[str] = field(default_factory=set)
    interference_factor: float = 0.0
    
    # Execution properties
    actual_duration: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    executor: Optional[Callable] = None
    result: Any = None
    
    def __post_init__(self):
        """Initialize quantum state."""
        if not self.state_amplitudes:
            # Initialize in superposition
            self.state_amplitudes = {
                TaskState.PENDING: complex(1.0, 0.0),
                TaskState.ACTIVE: complex(0.0, 0.0),
                TaskState.BLOCKED: complex(0.0, 0.0),
                TaskState.COMPLETED: complex(0.0, 0.0),
                TaskState.FAILED: complex(0.0, 0.0),
            }
    
    def get_probability(self, state: TaskState) -> float:
        """Get probability of task being in given state."""
        amplitude = self.state_amplitudes.get(state, complex(0.0, 0.0))
        return abs(amplitude) ** 2
    
    def collapse_state(self) -> TaskState:
        """Collapse superposition to definite state (measurement)."""
        probabilities = [self.get_probability(state) for state in TaskState]
        states = list(TaskState)
        
        # Quantum measurement - probabilistic collapse
        key = random.PRNGKey(int(time.time() * 1e6) % 2**32)
        collapsed_idx = random.choice(key, len(states), p=np.array(probabilities))
        return states[collapsed_idx]
    
    def update_amplitude(self, state: TaskState, amplitude: complex):
        """Update state amplitude."""
        self.state_amplitudes[state] = amplitude
        self._normalize_amplitudes()
    
    def _normalize_amplitudes(self):
        """Normalize amplitudes to maintain quantum constraint."""
        total_prob = sum(abs(amp)**2 for amp in self.state_amplitudes.values())
        if total_prob > 0:
            norm_factor = 1.0 / np.sqrt(total_prob)
            for state in self.state_amplitudes:
                self.state_amplitudes[state] *= norm_factor


@dataclass 
class TaskSuperposition:
    """
    Represents superposition of multiple task execution paths.
    
    Maintains coherent superposition until measurement forces collapse
    to specific execution timeline.
    """
    paths: List[List[str]] = field(default_factory=list)
    path_amplitudes: List[complex] = field(default_factory=list)
    interference_matrix: np.ndarray = field(default=None)
    
    def add_path(self, task_sequence: List[str], amplitude: complex):
        """Add execution path to superposition."""
        self.paths.append(task_sequence)
        self.path_amplitudes.append(amplitude)
        self._update_interference_matrix()
    
    def _update_interference_matrix(self):
        """Calculate quantum interference between paths."""
        n_paths = len(self.paths)
        if n_paths < 2:
            return
            
        self.interference_matrix = np.zeros((n_paths, n_paths), dtype=complex)
        
        for i in range(n_paths):
            for j in range(n_paths):
                if i != j:
                    # Calculate overlap between paths
                    path_i = set(self.paths[i])
                    path_j = set(self.paths[j])
                    overlap = len(path_i.intersection(path_j)) / max(len(path_i), len(path_j))
                    
                    # Interference amplitude
                    self.interference_matrix[i, j] = overlap * np.conj(self.path_amplitudes[i]) * self.path_amplitudes[j]
    
    def measure_optimal_path(self) -> List[str]:
        """Collapse superposition to optimal path."""
        if not self.paths:
            return []
        
        # Calculate path probabilities with interference
        probabilities = []
        for i, amplitude in enumerate(self.path_amplitudes):
            # Base probability
            prob = abs(amplitude) ** 2
            
            # Add interference effects
            if self.interference_matrix is not None:
                interference = np.sum(self.interference_matrix[i, :]).real
                prob += interference
            
            probabilities.append(max(0.0, prob))  # Ensure non-negative
        
        # Normalize
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)
        
        # Quantum measurement
        key = random.PRNGKey(int(time.time() * 1e6) % 2**32)
        selected_idx = random.choice(key, len(self.paths), p=np.array(probabilities))
        
        return self.paths[selected_idx]


class QuantumTaskPlanner:
    """
    Main quantum-inspired task planner.
    
    Uses quantum principles to optimize task scheduling and execution:
    - Maintains task superposition states
    - Calculates entanglement between dependent tasks  
    - Applies interference for path optimization
    - Collapses to execution plans via measurement
    """
    
    def __init__(
        self,
        max_parallel_tasks: int = 4,
        quantum_coherence_time: float = 10.0,
        interference_strength: float = 0.1,
    ):
        self.max_parallel_tasks = max_parallel_tasks
        self.quantum_coherence_time = quantum_coherence_time
        self.interference_strength = interference_strength
        
        # Task management
        self.tasks: Dict[str, QuantumTask] = {}
        self.task_graph: Dict[str, Set[str]] = {}  # Dependencies
        self.execution_history: List[Dict] = []
        
        # Quantum state
        self.global_superposition: Optional[TaskSuperposition] = None
        self.entanglement_matrix: Dict[Tuple[str, str], complex] = {}
        self.last_measurement_time: float = 0.0
        
    def add_task(
        self,
        task_id: str,
        name: str,
        dependencies: Optional[Set[str]] = None,
        estimated_duration: float = 1.0,
        priority: float = 1.0,
        resource_requirements: Optional[Dict[str, float]] = None,
        executor: Optional[Callable] = None,
    ) -> QuantumTask:
        """Add task to quantum planner."""
        dependencies = dependencies or set()
        resource_requirements = resource_requirements or {}
        
        task = QuantumTask(
            id=task_id,
            name=name,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            priority=priority,
            resource_requirements=resource_requirements,
            executor=executor,
        )
        
        self.tasks[task_id] = task
        self.task_graph[task_id] = dependencies
        
        # Update quantum entanglement
        self._update_entanglement(task_id)
        
        return task
    
    def _update_entanglement(self, new_task_id: str):
        """Update quantum entanglement between tasks."""
        new_task = self.tasks[new_task_id]
        
        for other_id, other_task in self.tasks.items():
            if other_id == new_task_id:
                continue
                
            # Calculate entanglement strength
            entanglement_strength = 0.0
            
            # Direct dependency creates strong entanglement
            if other_id in new_task.dependencies or new_task_id in other_task.dependencies:
                entanglement_strength += 0.8
            
            # Resource competition creates entanglement
            shared_resources = set(new_task.resource_requirements.keys()) & set(other_task.resource_requirements.keys())
            if shared_resources:
                entanglement_strength += 0.3 * len(shared_resources)
            
            # Priority similarity creates weak entanglement
            priority_diff = abs(new_task.priority - other_task.priority)
            entanglement_strength += 0.1 * np.exp(-priority_diff)
            
            # Store entanglement
            if entanglement_strength > 0.1:
                entanglement_key = tuple(sorted([new_task_id, other_id]))
                self.entanglement_matrix[entanglement_key] = complex(entanglement_strength, 0.0)
                
                # Mark tasks as entangled
                new_task.entangled_tasks.add(other_id)
                other_task.entangled_tasks.add(new_task_id)
    
    def generate_execution_paths(self) -> TaskSuperposition:
        """Generate superposition of possible execution paths."""
        ready_tasks = self._get_ready_tasks()
        if not ready_tasks:
            return TaskSuperposition()
        
        # Generate multiple execution paths
        paths = []
        amplitudes = []
        
        # Path 1: Priority-based ordering
        priority_path = self._generate_priority_path(ready_tasks)
        paths.append(priority_path)
        amplitudes.append(complex(0.6, 0.0))
        
        # Path 2: Dependency-optimized ordering  
        dependency_path = self._generate_dependency_path(ready_tasks)
        paths.append(dependency_path)
        amplitudes.append(complex(0.5, 0.1))
        
        # Path 3: Resource-optimized ordering
        resource_path = self._generate_resource_path(ready_tasks)
        paths.append(resource_path)
        amplitudes.append(complex(0.4, 0.2))
        
        # Path 4: Random exploration path
        random_path = self._generate_random_path(ready_tasks)
        paths.append(random_path)
        amplitudes.append(complex(0.3, 0.0))
        
        # Create superposition
        superposition = TaskSuperposition()
        for path, amplitude in zip(paths, amplitudes):
            superposition.add_path(path, amplitude)
        
        self.global_superposition = superposition
        return superposition
    
    def _get_ready_tasks(self) -> List[str]:
        """Get tasks ready for execution."""
        ready = []
        for task_id, task in self.tasks.items():
            if task.get_probability(TaskState.PENDING) > 0.5:
                # Check dependencies
                deps_completed = all(
                    self.tasks[dep_id].get_probability(TaskState.COMPLETED) > 0.5
                    for dep_id in task.dependencies
                    if dep_id in self.tasks
                )
                if deps_completed:
                    ready.append(task_id)
        return ready
    
    def _generate_priority_path(self, ready_tasks: List[str]) -> List[str]:
        """Generate execution path based on priority."""
        return sorted(ready_tasks, key=lambda tid: self.tasks[tid].priority, reverse=True)
    
    def _generate_dependency_path(self, ready_tasks: List[str]) -> List[str]:
        """Generate path optimizing dependency resolution."""
        # Topological sort with priority tie-breaking
        in_degree = {tid: len(self.task_graph[tid]) for tid in ready_tasks}
        path = []
        remaining = ready_tasks.copy()
        
        while remaining:
            # Find tasks with no remaining dependencies
            available = [tid for tid in remaining if in_degree[tid] == 0]
            if not available:
                # Break cycles by priority
                available = [min(remaining, key=lambda tid: self.tasks[tid].priority)]
            
            # Select highest priority available task
            selected = max(available, key=lambda tid: self.tasks[tid].priority)
            path.append(selected)
            remaining.remove(selected)
            
            # Update in-degrees
            for tid in remaining:
                if selected in self.task_graph[tid]:
                    in_degree[tid] -= 1
        
        return path
    
    def _generate_resource_path(self, ready_tasks: List[str]) -> List[str]:
        """Generate path optimizing resource utilization."""
        # Sort by resource requirements (ascending) to enable parallelization
        return sorted(ready_tasks, key=lambda tid: sum(self.tasks[tid].resource_requirements.values()))
    
    def _generate_random_path(self, ready_tasks: List[str]) -> List[str]:
        """Generate random exploration path."""
        key = random.PRNGKey(int(time.time() * 1e6) % 2**32)
        return list(random.permutation(key, np.array(ready_tasks)))
    
    def measure_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute(self) -> Dict[str, Any]:
        """Collapse quantum superposition and execute optimal path."""
        current_time = time.time()
        
        # Check quantum coherence
        if current_time - self.last_measurement_time > self.quantum_coherence_time:
            # Decoherence - regenerate superposition
            self.global_superposition = None
        
        # Generate or use existing superposition
        if self.global_superposition is None:
            self.global_superposition = self.generate_execution_paths()
        
        # Quantum measurement - collapse to execution path
        optimal_path = self.global_superposition.measure_optimal_path()
        
        # Update measurement time
        self.last_measurement_time = current_time
        
        # Execute collapsed path
        execution_result = self._execute_path(optimal_path)
        
        # Record execution
        self.execution_history.append({
            "timestamp": current_time,
            "path": optimal_path,
            "result": execution_result,
        })
        
        return execution_result
    
    def _execute_path(self, path: List[str]) -> Dict[str, Any]:
        """Execute collapsed task path."""
        results = {}
        start_time = time.time()
        
        for task_id in path:
            if task_id not in self.tasks:
                continue
                
            task = self.tasks[task_id]
            
            # Collapse task state to ACTIVE
            task.update_amplitude(TaskState.ACTIVE, complex(1.0, 0.0))
            task.update_amplitude(TaskState.PENDING, complex(0.0, 0.0))
            
            task.start_time = time.time()
            
            try:
                # Execute task
                if task.executor:
                    task.result = task.executor()
                else:
                    # Simulate execution
                    time.sleep(min(task.estimated_duration, 0.1))  # Cap simulation time
                    task.result = {"status": "completed", "task_id": task_id}
                
                task.end_time = time.time()
                task.actual_duration = task.end_time - task.start_time
                
                # Collapse to COMPLETED state
                task.update_amplitude(TaskState.COMPLETED, complex(1.0, 0.0))
                task.update_amplitude(TaskState.ACTIVE, complex(0.0, 0.0))
                
                results[task_id] = {
                    "status": "success",
                    "result": task.result,
                    "duration": task.actual_duration,
                }
                
            except Exception as e:
                task.end_time = time.time()
                task.actual_duration = task.end_time - task.start_time
                
                # Collapse to FAILED state  
                task.update_amplitude(TaskState.FAILED, complex(1.0, 0.0))
                task.update_amplitude(TaskState.ACTIVE, complex(0.0, 0.0))
                
                results[task_id] = {
                    "status": "failed",
                    "error": str(e),
                    "duration": task.actual_duration,
                }
        
        total_duration = time.time() - start_time
        
        return {
            "path": path,
            "task_results": results,
            "total_duration": total_duration,
            "quantum_efficiency": self._calculate_quantum_efficiency(results),
        }
    
    def _calculate_quantum_efficiency(self, results: Dict[str, Any]) -> float:
        """Calculate quantum planning efficiency metric."""
        if not results:
            return 0.0
        
        successful_tasks = sum(1 for r in results.values() if r["status"] == "success")
        total_tasks = len(results)
        
        # Base efficiency
        success_rate = successful_tasks / total_tasks
        
        # Quantum enhancement factor
        if self.global_superposition and len(self.global_superposition.paths) > 1:
            # Bonus for exploring multiple paths
            quantum_bonus = 0.1 * len(self.global_superposition.paths)
        else:
            quantum_bonus = 0.0
        
        # Entanglement utilization factor
        entanglement_bonus = 0.05 * len(self.entanglement_matrix)
        
        return min(1.0, success_rate + quantum_bonus + entanglement_bonus)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current quantum system state."""
        return {
            "num_tasks": len(self.tasks),
            "task_states": {
                task_id: {
                    state.value: task.get_probability(state)
                    for state in TaskState
                }
                for task_id, task in self.tasks.items()
            },
            "entanglements": len(self.entanglement_matrix),
            "superposition_paths": len(self.global_superposition.paths) if self.global_superposition else 0,
            "execution_history_length": len(self.execution_history),
            "quantum_coherence_remaining": max(0, self.quantum_coherence_time - (time.time() - self.last_measurement_time)),
        }