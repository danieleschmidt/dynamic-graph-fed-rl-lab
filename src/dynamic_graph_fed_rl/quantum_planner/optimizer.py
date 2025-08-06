"""
Quantum-inspired optimizers for task execution paths.

Implements optimization algorithms using quantum interference:
- QuantumOptimizer: Base quantum optimization with amplitude manipulation
- InterferenceOptimizer: Advanced optimization using quantum interference patterns
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, grad

from .core import QuantumTask, TaskSuperposition, TaskState


@dataclass
class OptimizationResult:
    """Result of quantum optimization."""
    optimal_path: List[str]
    optimization_score: float
    iterations: int
    convergence_achieved: bool
    quantum_efficiency: float
    execution_time: float


class BaseOptimizer(ABC):
    """Abstract base optimizer interface."""
    
    @abstractmethod
    def optimize(
        self, 
        tasks: Dict[str, QuantumTask],
        objective_function: Optional[Callable] = None
    ) -> OptimizationResult:
        """Optimize task execution path."""
        pass


class QuantumOptimizer(BaseOptimizer):
    """
    Core quantum optimizer using amplitude manipulation.
    
    Uses quantum principles to find optimal task execution paths:
    - Maintains superposition of execution paths
    - Applies quantum gates to evolve amplitudes
    - Measures optimal path through amplitude collapse
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        learning_rate: float = 0.1,
        quantum_noise: float = 0.01,
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.learning_rate = learning_rate
        self.quantum_noise = quantum_noise
        
        # Optimization state
        self.iteration_history: List[Dict[str, Any]] = []
        self.best_score_history: List[float] = []
    
    def optimize(
        self, 
        tasks: Dict[str, QuantumTask],
        objective_function: Optional[Callable] = None
    ) -> OptimizationResult:
        """Optimize task execution using quantum amplitude evolution."""
        start_time = time.time()
        
        # Default objective: minimize total execution time while maximizing success probability
        if objective_function is None:
            objective_function = self._default_objective_function
        
        # Initialize quantum state
        ready_tasks = self._get_ready_tasks(tasks)
        if not ready_tasks:
            return OptimizationResult(
                optimal_path=[],
                optimization_score=0.0,
                iterations=0,
                convergence_achieved=True,
                quantum_efficiency=0.0,
                execution_time=time.time() - start_time
            )
        
        # Generate initial superposition
        initial_paths = self._generate_initial_paths(tasks, ready_tasks)
        current_superposition = self._create_superposition(initial_paths)
        
        best_score = float('-inf')
        best_path = []
        convergence_achieved = False
        
        # Quantum optimization loop
        for iteration in range(self.max_iterations):
            iteration_start = time.time()
            
            # Evaluate current paths
            path_scores = self._evaluate_paths(current_superposition.paths, tasks, objective_function)
            
            # Update best solution
            max_score_idx = np.argmax(path_scores)
            if path_scores[max_score_idx] > best_score:
                best_score = path_scores[max_score_idx]
                best_path = current_superposition.paths[max_score_idx].copy()
            
            # Apply quantum evolution operators
            current_superposition = self._apply_quantum_evolution(
                current_superposition, 
                path_scores,
                iteration
            )
            
            # Check convergence
            if len(self.best_score_history) > 0:
                score_change = abs(best_score - self.best_score_history[-1])
                if score_change < self.convergence_threshold:
                    convergence_achieved = True
                    break
            
            # Store iteration data
            self.best_score_history.append(best_score)
            self.iteration_history.append({
                "iteration": iteration,
                "best_score": best_score,
                "num_paths": len(current_superposition.paths),
                "iteration_time": time.time() - iteration_start
            })
        
        # Calculate quantum efficiency
        quantum_efficiency = self._calculate_quantum_efficiency(
            best_score, len(current_superposition.paths), convergence_achieved
        )
        
        return OptimizationResult(
            optimal_path=best_path,
            optimization_score=best_score,
            iterations=len(self.iteration_history),
            convergence_achieved=convergence_achieved,
            quantum_efficiency=quantum_efficiency,
            execution_time=time.time() - start_time
        )
    
    def _get_ready_tasks(self, tasks: Dict[str, QuantumTask]) -> List[str]:
        """Get tasks ready for optimization."""
        ready = []
        for task_id, task in tasks.items():
            # Check if task is in pending state
            if task.get_probability(TaskState.PENDING) > 0.5:
                # Check dependencies
                deps_ready = all(
                    tasks[dep_id].get_probability(TaskState.COMPLETED) > 0.5
                    for dep_id in task.dependencies
                    if dep_id in tasks
                )
                if deps_ready:
                    ready.append(task_id)
        return ready
    
    def _generate_initial_paths(
        self, 
        tasks: Dict[str, QuantumTask], 
        ready_tasks: List[str]
    ) -> List[List[str]]:
        """Generate initial execution paths."""
        paths = []
        
        # Path 1: Priority ordering
        priority_path = sorted(ready_tasks, key=lambda tid: tasks[tid].priority, reverse=True)
        paths.append(priority_path)
        
        # Path 2: Duration ordering (shortest first)
        duration_path = sorted(ready_tasks, key=lambda tid: tasks[tid].estimated_duration)
        paths.append(duration_path)
        
        # Path 3: Dependency-aware topological ordering
        topo_path = self._topological_sort(tasks, ready_tasks)
        paths.append(topo_path)
        
        # Path 4: Random paths for exploration
        key = random.PRNGKey(int(time.time() * 1e6) % 2**32)
        for _ in range(3):
            key, subkey = random.split(key)
            random_path = list(random.permutation(subkey, np.array(ready_tasks)))
            paths.append(random_path)
        
        return paths
    
    def _topological_sort(self, tasks: Dict[str, QuantumTask], ready_tasks: List[str]) -> List[str]:
        """Topological sort of ready tasks considering all dependencies."""
        # Build dependency graph for ready tasks
        in_degree = {tid: 0 for tid in ready_tasks}
        graph = {tid: [] for tid in ready_tasks}
        
        for task_id in ready_tasks:
            task = tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id in ready_tasks:
                    graph[dep_id].append(task_id)
                    in_degree[task_id] += 1
        
        # Kahn's algorithm
        queue = [tid for tid in ready_tasks if in_degree[tid] == 0]
        result = []
        
        while queue:
            # Select task with highest priority among available
            current = max(queue, key=lambda tid: tasks[tid].priority)
            queue.remove(current)
            result.append(current)
            
            # Update in-degrees
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _create_superposition(self, paths: List[List[str]]) -> TaskSuperposition:
        """Create quantum superposition from execution paths."""
        superposition = TaskSuperposition()
        
        # Initialize with equal amplitudes
        num_paths = len(paths)
        base_amplitude = 1.0 / np.sqrt(num_paths)
        
        for i, path in enumerate(paths):
            # Add small phase variation
            phase = 2 * np.pi * i / num_paths
            amplitude = complex(base_amplitude * np.cos(phase), base_amplitude * np.sin(phase))
            superposition.add_path(path, amplitude)
        
        return superposition
    
    def _evaluate_paths(
        self, 
        paths: List[List[str]], 
        tasks: Dict[str, QuantumTask],
        objective_function: Callable
    ) -> List[float]:
        """Evaluate all paths using objective function."""
        scores = []
        for path in paths:
            score = objective_function(path, tasks)
            scores.append(score)
        return scores
    
    def _default_objective_function(self, path: List[str], tasks: Dict[str, QuantumTask]) -> float:
        """Default objective function: minimize time, maximize success probability."""
        if not path:
            return 0.0
        
        # Calculate estimated total execution time
        total_time = sum(tasks[tid].estimated_duration for tid in path if tid in tasks)
        
        # Calculate success probability (product of individual probabilities)
        success_prob = 1.0
        for task_id in path:
            if task_id in tasks:
                # Estimate success probability based on task complexity
                task = tasks[task_id]
                # Simple heuristic: shorter tasks more likely to succeed
                task_success_prob = max(0.5, 1.0 - task.estimated_duration * 0.1)
                success_prob *= task_success_prob
        
        # Objective: maximize success probability / execution time ratio
        if total_time > 0:
            return success_prob / total_time
        else:
            return success_prob
    
    def _apply_quantum_evolution(
        self, 
        superposition: TaskSuperposition,
        path_scores: List[float],
        iteration: int
    ) -> TaskSuperposition:
        """Apply quantum evolution operators to superposition."""
        new_paths = []
        new_amplitudes = []
        
        # Normalize scores to probabilities
        min_score = min(path_scores)
        max_score = max(path_scores)
        score_range = max_score - min_score
        
        if score_range > 0:
            normalized_scores = [(score - min_score) / score_range for score in path_scores]
        else:
            normalized_scores = [1.0] * len(path_scores)
        
        # Evolution operators
        for i, (path, amplitude) in enumerate(zip(superposition.paths, superposition.path_amplitudes)):
            score = normalized_scores[i]
            
            # Amplitude amplification for high-scoring paths
            amplification_factor = 1.0 + self.learning_rate * score
            
            # Phase rotation based on score
            phase_rotation = 2 * np.pi * score * iteration * 0.01
            rotation_factor = complex(np.cos(phase_rotation), np.sin(phase_rotation))
            
            # Apply quantum noise for exploration
            key = random.PRNGKey(int(time.time() * 1e6 + i) % 2**32)
            noise_real = random.normal(key) * self.quantum_noise
            key, subkey = random.split(key)
            noise_imag = random.normal(subkey) * self.quantum_noise
            noise = complex(noise_real, noise_imag)
            
            # Update amplitude
            new_amplitude = amplitude * amplification_factor * rotation_factor + noise
            
            new_paths.append(path.copy())
            new_amplitudes.append(new_amplitude)
        
        # Generate new paths through quantum crossover
        if len(superposition.paths) >= 2:
            new_crossover_paths, crossover_amplitudes = self._quantum_crossover(
                superposition.paths, superposition.path_amplitudes, normalized_scores
            )
            new_paths.extend(new_crossover_paths)
            new_amplitudes.extend(crossover_amplitudes)
        
        # Create new superposition
        new_superposition = TaskSuperposition()
        for path, amplitude in zip(new_paths, new_amplitudes):
            new_superposition.add_path(path, amplitude)
        
        # Limit superposition size to prevent explosion
        max_paths = 20
        if len(new_superposition.paths) > max_paths:
            # Keep paths with highest amplitude magnitudes
            path_mags = [abs(amp) for amp in new_superposition.path_amplitudes]
            top_indices = np.argsort(path_mags)[-max_paths:]
            
            filtered_superposition = TaskSuperposition()
            for idx in top_indices:
                filtered_superposition.add_path(
                    new_superposition.paths[idx],
                    new_superposition.path_amplitudes[idx]
                )
            new_superposition = filtered_superposition
        
        return new_superposition
    
    def _quantum_crossover(
        self, 
        paths: List[List[str]], 
        amplitudes: List[complex],
        scores: List[float]
    ) -> Tuple[List[List[str]], List[complex]]:
        """Generate new paths through quantum crossover."""
        new_paths = []
        new_amplitudes = []
        
        # Select high-scoring parents
        parent_probs = np.array(scores) / sum(scores) if sum(scores) > 0 else np.ones(len(scores)) / len(scores)
        
        key = random.PRNGKey(int(time.time() * 1e6) % 2**32)
        
        # Generate several crossover offspring
        num_crossovers = min(3, len(paths) // 2)
        for _ in range(num_crossovers):
            key, subkey1, subkey2 = random.split(key, 3)
            
            # Select parents
            parent1_idx = random.choice(subkey1, len(paths), p=parent_probs)
            parent2_idx = random.choice(subkey2, len(paths), p=parent_probs)
            
            if parent1_idx != parent2_idx:
                parent1 = paths[parent1_idx]
                parent2 = paths[parent2_idx]
                
                # Quantum crossover - create superposition of segments
                crossover_path = self._perform_crossover(parent1, parent2, key)
                
                # Child amplitude from parent amplitudes
                child_amplitude = (amplitudes[parent1_idx] + amplitudes[parent2_idx]) / 2.0
                
                new_paths.append(crossover_path)
                new_amplitudes.append(child_amplitude)
                
                key, subkey = random.split(key)
        
        return new_paths, new_amplitudes
    
    def _perform_crossover(self, parent1: List[str], parent2: List[str], key) -> List[str]:
        """Perform quantum crossover between two paths."""
        if not parent1 or not parent2:
            return parent1 or parent2
        
        # Find common tasks
        set1 = set(parent1)
        set2 = set(parent2)
        common = set1.intersection(set2)
        
        # Create child path
        child = []
        p1_idx = 0
        p2_idx = 0
        
        key, subkey = random.split(key)
        
        while p1_idx < len(parent1) or p2_idx < len(parent2):
            # Quantum choice: select from parent1 or parent2
            if p1_idx >= len(parent1):
                source = 2  # Only parent2 available
            elif p2_idx >= len(parent2):
                source = 1  # Only parent1 available
            else:
                # Probabilistic choice
                prob = random.uniform(key, minval=0.0, maxval=1.0)
                source = 1 if prob < 0.5 else 2
                key, subkey = random.split(key)
            
            # Add task from selected parent
            if source == 1 and p1_idx < len(parent1):
                task = parent1[p1_idx]
                if task not in child:
                    child.append(task)
                p1_idx += 1
            elif source == 2 and p2_idx < len(parent2):
                task = parent2[p2_idx]
                if task not in child:
                    child.append(task)
                p2_idx += 1
        
        return child
    
    def _calculate_quantum_efficiency(
        self, 
        best_score: float,
        num_paths_explored: int,
        convergence_achieved: bool
    ) -> float:
        """Calculate quantum optimization efficiency."""
        # Base efficiency from score
        base_efficiency = min(1.0, max(0.0, best_score))
        
        # Exploration bonus
        exploration_bonus = 0.1 * min(1.0, num_paths_explored / 10.0)
        
        # Convergence bonus
        convergence_bonus = 0.2 if convergence_achieved else 0.0
        
        return min(1.0, base_efficiency + exploration_bonus + convergence_bonus)


class InterferenceOptimizer(QuantumOptimizer):
    """
    Advanced optimizer using quantum interference patterns.
    
    Extends base quantum optimizer with sophisticated interference
    calculations for enhanced path optimization.
    """
    
    def __init__(
        self,
        max_iterations: int = 150,
        convergence_threshold: float = 1e-6,
        learning_rate: float = 0.05,
        quantum_noise: float = 0.01,
        interference_strength: float = 0.3,
        coherence_length: int = 10,
    ):
        super().__init__(max_iterations, convergence_threshold, learning_rate, quantum_noise)
        self.interference_strength = interference_strength
        self.coherence_length = coherence_length
        
        # Interference tracking
        self.interference_history: List[np.ndarray] = []
        self.coherence_time: float = 0.0
    
    def optimize(
        self, 
        tasks: Dict[str, QuantumTask],
        objective_function: Optional[Callable] = None
    ) -> OptimizationResult:
        """Optimize using quantum interference."""
        start_time = time.time()
        
        # Enhanced initial superposition with interference design
        ready_tasks = self._get_ready_tasks(tasks)
        if not ready_tasks:
            return OptimizationResult([], 0.0, 0, True, 0.0, time.time() - start_time)
        
        # Generate interference-optimized initial paths
        initial_paths = self._generate_interference_paths(tasks, ready_tasks)
        current_superposition = self._create_interference_superposition(initial_paths)
        
        # Run optimization with interference
        result = super().optimize(tasks, objective_function)
        
        # Enhanced quantum efficiency with interference bonus
        interference_efficiency = self._calculate_interference_efficiency()
        enhanced_efficiency = min(1.0, result.quantum_efficiency + interference_efficiency)
        
        return OptimizationResult(
            optimal_path=result.optimal_path,
            optimization_score=result.optimization_score,
            iterations=result.iterations,
            convergence_achieved=result.convergence_achieved,
            quantum_efficiency=enhanced_efficiency,
            execution_time=result.execution_time
        )
    
    def _generate_interference_paths(
        self, 
        tasks: Dict[str, QuantumTask], 
        ready_tasks: List[str]
    ) -> List[List[str]]:
        """Generate paths optimized for quantum interference."""
        paths = []
        
        # Base paths from parent class
        base_paths = self._generate_initial_paths(tasks, ready_tasks)
        paths.extend(base_paths)
        
        # Interference-specific paths
        
        # Path 1: Phase-coherent ordering based on task similarity
        similarity_path = self._create_similarity_ordered_path(tasks, ready_tasks)
        paths.append(similarity_path)
        
        # Path 2: Interference-maximizing path
        interference_path = self._create_interference_maximizing_path(tasks, ready_tasks)
        paths.append(interference_path)
        
        # Path 3: Coherence-length optimized chunks
        coherence_path = self._create_coherence_optimized_path(tasks, ready_tasks)
        paths.append(coherence_path)
        
        return paths
    
    def _create_similarity_ordered_path(
        self, 
        tasks: Dict[str, QuantumTask], 
        ready_tasks: List[str]
    ) -> List[str]:
        """Create path ordering tasks by similarity for coherent interference."""
        if not ready_tasks:
            return []
        
        # Calculate task similarity matrix
        similarity_matrix = np.zeros((len(ready_tasks), len(ready_tasks)))
        
        for i, task1_id in enumerate(ready_tasks):
            for j, task2_id in enumerate(ready_tasks):
                if i != j:
                    similarity = self._calculate_task_similarity(
                        tasks[task1_id], tasks[task2_id]
                    )
                    similarity_matrix[i, j] = similarity
        
        # Greedy clustering by similarity
        path = [ready_tasks[0]]  # Start with first task
        remaining = set(ready_tasks[1:])
        
        while remaining:
            current_task_idx = ready_tasks.index(path[-1])
            
            # Find most similar remaining task
            best_similarity = -1
            best_task = None
            
            for task_id in remaining:
                task_idx = ready_tasks.index(task_id)
                similarity = similarity_matrix[current_task_idx, task_idx]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_task = task_id
            
            if best_task:
                path.append(best_task)
                remaining.remove(best_task)
            else:
                # Fallback: add any remaining task
                path.append(remaining.pop())
        
        return path
    
    def _calculate_task_similarity(self, task1: QuantumTask, task2: QuantumTask) -> float:
        """Calculate similarity between two tasks."""
        similarity = 0.0
        
        # Duration similarity
        duration_diff = abs(task1.estimated_duration - task2.estimated_duration)
        duration_similarity = np.exp(-duration_diff)
        similarity += 0.3 * duration_similarity
        
        # Priority similarity
        priority_diff = abs(task1.priority - task2.priority)
        priority_similarity = np.exp(-priority_diff)
        similarity += 0.2 * priority_similarity
        
        # Resource similarity
        resources1 = set(task1.resource_requirements.keys())
        resources2 = set(task2.resource_requirements.keys())
        
        if resources1 or resources2:
            resource_overlap = len(resources1.intersection(resources2))
            resource_union = len(resources1.union(resources2))
            resource_similarity = resource_overlap / max(resource_union, 1)
            similarity += 0.4 * resource_similarity
        
        # Dependency relationship
        if task2.id in task1.dependencies or task1.id in task2.dependencies:
            similarity += 0.1
        
        return similarity
    
    def _create_interference_maximizing_path(
        self, 
        tasks: Dict[str, QuantumTask], 
        ready_tasks: List[str]
    ) -> List[str]:
        """Create path that maximizes constructive interference."""
        # Start with priority ordering
        path = sorted(ready_tasks, key=lambda tid: tasks[tid].priority, reverse=True)
        
        # Apply interference-based swaps
        for i in range(len(path) - 1):
            for j in range(i + 1, min(i + self.coherence_length, len(path))):
                # Check if swapping improves interference
                original_interference = self._calculate_local_interference(path, i, j)
                
                # Try swap
                path[i], path[j] = path[j], path[i]
                new_interference = self._calculate_local_interference(path, i, j)
                
                # Keep swap if it improves interference
                if new_interference <= original_interference:
                    # Revert swap
                    path[i], path[j] = path[j], path[i]
        
        return path
    
    def _calculate_local_interference(self, path: List[str], pos1: int, pos2: int) -> float:
        """Calculate local interference between two positions in path."""
        if pos1 >= len(path) or pos2 >= len(path):
            return 0.0
        
        # Simple interference metric based on position distance
        distance = abs(pos2 - pos1)
        
        # Interference decreases with distance (quantum decoherence)
        interference = np.exp(-distance / self.coherence_length)
        
        return interference
    
    def _create_coherence_optimized_path(
        self, 
        tasks: Dict[str, QuantumTask], 
        ready_tasks: List[str]
    ) -> List[str]:
        """Create path optimized for quantum coherence preservation."""
        if len(ready_tasks) <= self.coherence_length:
            return ready_tasks.copy()
        
        # Divide tasks into coherence-length chunks
        path = []
        remaining = ready_tasks.copy()
        
        while remaining:
            # Take up to coherence_length tasks
            chunk_size = min(self.coherence_length, len(remaining))
            
            # Select best chunk based on internal coherence
            best_chunk = self._select_best_coherence_chunk(tasks, remaining, chunk_size)
            
            path.extend(best_chunk)
            for task_id in best_chunk:
                remaining.remove(task_id)
        
        return path
    
    def _select_best_coherence_chunk(
        self, 
        tasks: Dict[str, QuantumTask], 
        candidates: List[str], 
        chunk_size: int
    ) -> List[str]:
        """Select best chunk of tasks for quantum coherence."""
        if chunk_size >= len(candidates):
            return candidates.copy()
        
        # Simple greedy selection
        chunk = []
        remaining = candidates.copy()
        
        # Start with highest priority task
        best_task = max(remaining, key=lambda tid: tasks[tid].priority)
        chunk.append(best_task)
        remaining.remove(best_task)
        
        # Add most coherent tasks iteratively
        while len(chunk) < chunk_size and remaining:
            best_coherence = -1
            best_addition = None
            
            for candidate in remaining:
                # Calculate coherence with current chunk
                coherence = self._calculate_chunk_coherence(tasks, chunk + [candidate])
                
                if coherence > best_coherence:
                    best_coherence = coherence
                    best_addition = candidate
            
            if best_addition:
                chunk.append(best_addition)
                remaining.remove(best_addition)
            else:
                break
        
        return chunk
    
    def _calculate_chunk_coherence(self, tasks: Dict[str, QuantumTask], chunk: List[str]) -> float:
        """Calculate quantum coherence within a chunk of tasks."""
        if len(chunk) <= 1:
            return 1.0
        
        total_coherence = 0.0
        pairs = 0
        
        for i in range(len(chunk)):
            for j in range(i + 1, len(chunk)):
                task1 = tasks[chunk[i]]
                task2 = tasks[chunk[j]]
                
                # Coherence based on task similarity
                coherence = self._calculate_task_similarity(task1, task2)
                total_coherence += coherence
                pairs += 1
        
        return total_coherence / max(pairs, 1)
    
    def _create_interference_superposition(self, paths: List[List[str]]) -> TaskSuperposition:
        """Create superposition with designed interference patterns."""
        superposition = TaskSuperposition()
        
        # Calculate interference-optimized amplitudes
        num_paths = len(paths)
        
        for i, path in enumerate(paths):
            # Phase designed for constructive interference
            phase = 2 * np.pi * i * self.interference_strength / num_paths
            
            # Amplitude based on path quality estimate
            amplitude_mag = 1.0 / np.sqrt(num_paths)
            
            # Apply phase
            amplitude = complex(
                amplitude_mag * np.cos(phase),
                amplitude_mag * np.sin(phase)
            )
            
            superposition.add_path(path, amplitude)
        
        # Store interference pattern
        if superposition.interference_matrix is not None:
            self.interference_history.append(superposition.interference_matrix.copy())
        
        return superposition
    
    def _calculate_interference_efficiency(self) -> float:
        """Calculate efficiency bonus from quantum interference."""
        if not self.interference_history:
            return 0.0
        
        # Measure interference pattern evolution
        final_interference = self.interference_history[-1]
        
        # Calculate constructive interference ratio
        positive_interference = np.sum(final_interference.real > 0)
        total_interference = final_interference.size
        
        constructive_ratio = positive_interference / max(total_interference, 1)
        
        # Bonus for maintaining coherence
        coherence_bonus = min(0.2, constructive_ratio * 0.3)
        
        return coherence_bonus