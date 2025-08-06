"""
Quantum-inspired schedulers for adaptive task execution.

Implements multiple scheduling strategies using quantum principles:
- QuantumScheduler: Core quantum scheduling with superposition
- AdaptiveScheduler: Self-tuning scheduler that learns optimal parameters
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import jax.numpy as jnp
from jax import random

from .core import QuantumTaskPlanner, QuantumTask, TaskState


@dataclass
class SchedulingMetrics:
    """Metrics for scheduler performance tracking."""
    total_execution_time: float
    tasks_completed: int
    tasks_failed: int  
    resource_utilization: Dict[str, float]
    quantum_efficiency: float
    scheduling_overhead: float
    throughput: float  # tasks per second


class BaseScheduler(ABC):
    """Abstract base scheduler interface."""
    
    @abstractmethod
    async def schedule(self, planner: QuantumTaskPlanner) -> Dict[str, Any]:
        """Execute scheduling algorithm."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> SchedulingMetrics:
        """Get scheduler performance metrics."""
        pass


class QuantumScheduler(BaseScheduler):
    """
    Core quantum scheduler using superposition and measurement.
    
    Maintains multiple potential schedules in superposition until
    measurement collapses to optimal execution timeline.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 4,
        measurement_interval: float = 1.0,
        decoherence_threshold: float = 10.0,
        interference_optimization: bool = True,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.measurement_interval = measurement_interval
        self.decoherence_threshold = decoherence_threshold
        self.interference_optimization = interference_optimization
        
        # Metrics tracking
        self.execution_times: List[float] = []
        self.completed_tasks: int = 0
        self.failed_tasks: int = 0
        self.total_scheduling_time: float = 0.0
        self.resource_usage_history: List[Dict[str, float]] = []
        
    async def schedule(self, planner: QuantumTaskPlanner) -> Dict[str, Any]:
        """Execute quantum scheduling algorithm."""
        scheduling_start = time.time()
        
        # Get current ready tasks
        ready_tasks = self._get_ready_tasks(planner)
        if not ready_tasks:
            return {"status": "no_ready_tasks", "scheduled": []}
        
        # Generate quantum superposition of schedules
        schedule_superposition = await self._generate_schedule_superposition(planner, ready_tasks)
        
        # Quantum interference optimization
        if self.interference_optimization:
            schedule_superposition = self._apply_quantum_interference(schedule_superposition)
        
        # Measurement - collapse to optimal schedule
        optimal_schedule = self._measure_optimal_schedule(schedule_superposition)
        
        # Execute scheduled tasks
        execution_results = await self._execute_concurrent_tasks(planner, optimal_schedule)
        
        # Update metrics
        scheduling_time = time.time() - scheduling_start
        self.total_scheduling_time += scheduling_time
        
        return {
            "status": "success",
            "scheduled": optimal_schedule,
            "execution_results": execution_results,
            "scheduling_time": scheduling_time,
            "quantum_paths_explored": len(schedule_superposition),
        }
    
    def _get_ready_tasks(self, planner: QuantumTaskPlanner) -> List[str]:
        """Get tasks ready for execution."""
        ready = []
        for task_id, task in planner.tasks.items():
            # Check if task is in pending state
            if task.get_probability(TaskState.PENDING) > 0.5:
                # Verify dependencies are completed
                deps_satisfied = all(
                    planner.tasks[dep_id].get_probability(TaskState.COMPLETED) > 0.5
                    for dep_id in task.dependencies
                    if dep_id in planner.tasks
                )
                if deps_satisfied:
                    ready.append(task_id)
        return ready
    
    async def _generate_schedule_superposition(
        self, 
        planner: QuantumTaskPlanner, 
        ready_tasks: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate superposition of possible schedules."""
        schedules = []
        
        # Schedule 1: Priority-first with resource constraints
        priority_schedule = self._create_priority_schedule(planner, ready_tasks)
        schedules.append({
            "tasks": priority_schedule,
            "amplitude": complex(0.4, 0.0),
            "strategy": "priority_first"
        })
        
        # Schedule 2: Resource-optimized concurrent execution
        resource_schedule = self._create_resource_optimized_schedule(planner, ready_tasks)
        schedules.append({
            "tasks": resource_schedule, 
            "amplitude": complex(0.35, 0.1),
            "strategy": "resource_optimized"
        })
        
        # Schedule 3: Dependency-chain optimized
        dependency_schedule = self._create_dependency_schedule(planner, ready_tasks)
        schedules.append({
            "tasks": dependency_schedule,
            "amplitude": complex(0.3, 0.15),
            "strategy": "dependency_chain"
        })
        
        # Schedule 4: Load-balanced with entanglement awareness
        entangled_schedule = self._create_entanglement_aware_schedule(planner, ready_tasks)
        schedules.append({
            "tasks": entangled_schedule,
            "amplitude": complex(0.25, 0.2),
            "strategy": "entanglement_aware"
        })
        
        return schedules
    
    def _create_priority_schedule(self, planner: QuantumTaskPlanner, ready_tasks: List[str]) -> List[Dict[str, Any]]:
        """Create schedule based on task priorities."""
        sorted_tasks = sorted(
            ready_tasks, 
            key=lambda tid: planner.tasks[tid].priority, 
            reverse=True
        )
        
        schedule = []
        resource_allocation = {}
        
        for task_id in sorted_tasks:
            task = planner.tasks[task_id]
            
            # Check resource availability
            can_schedule = True
            for resource, required in task.resource_requirements.items():
                current_usage = resource_allocation.get(resource, 0.0)
                if current_usage + required > 1.0:  # Assume normalized resources
                    can_schedule = False
                    break
            
            if can_schedule and len(schedule) < self.max_concurrent_tasks:
                # Schedule task
                schedule.append({
                    "task_id": task_id,
                    "start_time": 0.0,  # Immediate start
                    "allocated_resources": dict(task.resource_requirements)
                })
                
                # Update resource allocation
                for resource, required in task.resource_requirements.items():
                    resource_allocation[resource] = resource_allocation.get(resource, 0.0) + required
        
        return schedule
    
    def _create_resource_optimized_schedule(self, planner: QuantumTaskPlanner, ready_tasks: List[str]) -> List[Dict[str, Any]]:
        """Create schedule optimizing resource utilization."""
        # Sort by resource efficiency (priority/resources ratio)
        def resource_efficiency(task_id):
            task = planner.tasks[task_id]
            total_resources = sum(task.resource_requirements.values()) or 1.0
            return task.priority / total_resources
        
        sorted_tasks = sorted(ready_tasks, key=resource_efficiency, reverse=True)
        
        schedule = []
        resource_allocation = {}
        
        for task_id in sorted_tasks[:self.max_concurrent_tasks]:
            task = planner.tasks[task_id]
            
            # Pack tasks efficiently into available resources
            can_fit = True
            temp_allocation = dict(resource_allocation)
            
            for resource, required in task.resource_requirements.items():
                current = temp_allocation.get(resource, 0.0)
                if current + required <= 1.0:
                    temp_allocation[resource] = current + required
                else:
                    can_fit = False
                    break
            
            if can_fit:
                schedule.append({
                    "task_id": task_id,
                    "start_time": 0.0,
                    "allocated_resources": dict(task.resource_requirements)
                })
                resource_allocation = temp_allocation
        
        return schedule
    
    def _create_dependency_schedule(self, planner: QuantumTaskPlanner, ready_tasks: List[str]) -> List[Dict[str, Any]]:
        """Create schedule optimizing dependency resolution."""
        # Build dependency chains
        chains = self._build_dependency_chains(planner, ready_tasks)
        
        schedule = []
        scheduled_count = 0
        
        # Schedule longest chains first to maximize parallelism
        sorted_chains = sorted(chains, key=len, reverse=True)
        
        for chain in sorted_chains:
            if scheduled_count >= self.max_concurrent_tasks:
                break
                
            # Schedule first task in chain (others will follow)
            if chain:
                task_id = chain[0]
                schedule.append({
                    "task_id": task_id,
                    "start_time": 0.0,
                    "allocated_resources": dict(planner.tasks[task_id].resource_requirements),
                    "chain_length": len(chain)
                })
                scheduled_count += 1
        
        return schedule
    
    def _create_entanglement_aware_schedule(self, planner: QuantumTaskPlanner, ready_tasks: List[str]) -> List[Dict[str, Any]]:
        """Create schedule considering quantum entanglement between tasks."""
        schedule = []
        entanglement_groups = self._group_entangled_tasks(planner, ready_tasks)
        
        # Schedule entangled groups together for quantum coherence
        for group in entanglement_groups:
            if len(schedule) >= self.max_concurrent_tasks:
                break
                
            # Select representative task from group
            group_tasks = [tid for tid in group if tid in ready_tasks]
            if group_tasks:
                # Choose highest priority task from entangled group
                best_task = max(group_tasks, key=lambda tid: planner.tasks[tid].priority)
                
                schedule.append({
                    "task_id": best_task,
                    "start_time": 0.0,
                    "allocated_resources": dict(planner.tasks[best_task].resource_requirements),
                    "entangled_group": group
                })
        
        return schedule
    
    def _build_dependency_chains(self, planner: QuantumTaskPlanner, ready_tasks: List[str]) -> List[List[str]]:
        """Build chains of dependent tasks."""
        chains = []
        visited = set()
        
        for task_id in ready_tasks:
            if task_id in visited:
                continue
                
            chain = self._trace_dependency_chain(planner, task_id, visited)
            if chain:
                chains.append(chain)
        
        return chains
    
    def _trace_dependency_chain(self, planner: QuantumTaskPlanner, start_task: str, visited: set) -> List[str]:
        """Trace chain of dependencies from starting task."""
        chain = []
        current = start_task
        
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            
            # Find next task that depends on current
            next_task = None
            for task_id, task in planner.tasks.items():
                if current in task.dependencies and task_id not in visited:
                    next_task = task_id
                    break
            
            current = next_task
        
        return chain
    
    def _group_entangled_tasks(self, planner: QuantumTaskPlanner, ready_tasks: List[str]) -> List[List[str]]:
        """Group tasks by quantum entanglement."""
        groups = []
        processed = set()
        
        for task_id in ready_tasks:
            if task_id in processed:
                continue
                
            # Find all entangled tasks
            entangled_group = {task_id}
            queue = [task_id]
            
            while queue:
                current = queue.pop(0)
                if current in processed:
                    continue
                    
                processed.add(current)
                task = planner.tasks[current]
                
                # Add entangled tasks to group
                for entangled_id in task.entangled_tasks:
                    if entangled_id not in entangled_group and entangled_id in ready_tasks:
                        entangled_group.add(entangled_id)
                        queue.append(entangled_id)
            
            if entangled_group:
                groups.append(list(entangled_group))
        
        return groups
    
    def _apply_quantum_interference(self, schedule_superposition: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply quantum interference to optimize schedule selection."""
        if len(schedule_superposition) < 2:
            return schedule_superposition
        
        # Calculate interference between schedules
        for i in range(len(schedule_superposition)):
            for j in range(i + 1, len(schedule_superposition)):
                schedule_i = schedule_superposition[i]
                schedule_j = schedule_superposition[j]
                
                # Calculate schedule overlap
                tasks_i = {t["task_id"] for t in schedule_i["tasks"]}
                tasks_j = {t["task_id"] for t in schedule_j["tasks"]}
                overlap = len(tasks_i.intersection(tasks_j)) / max(len(tasks_i), len(tasks_j), 1)
                
                # Apply constructive/destructive interference
                if overlap > 0.5:  # High overlap - constructive interference
                    interference_factor = 1.1
                else:  # Low overlap - slight destructive interference
                    interference_factor = 0.95
                
                # Update amplitudes
                schedule_i["amplitude"] *= interference_factor
                schedule_j["amplitude"] *= interference_factor
        
        return schedule_superposition
    
    def _measure_optimal_schedule(self, schedule_superposition: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Quantum measurement to collapse to optimal schedule."""
        if not schedule_superposition:
            return []
        
        # Calculate probabilities from amplitudes
        probabilities = []
        for schedule in schedule_superposition:
            prob = abs(schedule["amplitude"]) ** 2
            probabilities.append(prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)
        
        # Quantum measurement
        key = random.PRNGKey(int(time.time() * 1e6) % 2**32)
        selected_idx = random.choice(key, len(schedule_superposition), p=np.array(probabilities))
        
        return schedule_superposition[selected_idx]["tasks"]
    
    async def _execute_concurrent_tasks(
        self, 
        planner: QuantumTaskPlanner, 
        schedule: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute scheduled tasks concurrently."""
        execution_start = time.time()
        
        # Create execution tasks
        execution_tasks = []
        for task_spec in schedule:
            task_id = task_spec["task_id"]
            if task_id in planner.tasks:
                execution_task = asyncio.create_task(
                    self._execute_single_task(planner.tasks[task_id])
                )
                execution_tasks.append((task_id, execution_task))
        
        # Wait for all tasks to complete
        results = {}
        for task_id, execution_task in execution_tasks:
            try:
                result = await execution_task
                results[task_id] = result
                if result["status"] == "success":
                    self.completed_tasks += 1
                else:
                    self.failed_tasks += 1
            except Exception as e:
                results[task_id] = {"status": "error", "error": str(e)}
                self.failed_tasks += 1
        
        execution_time = time.time() - execution_start
        self.execution_times.append(execution_time)
        
        return {
            "task_results": results,
            "execution_time": execution_time,
            "concurrent_tasks": len(schedule)
        }
    
    async def _execute_single_task(self, task: QuantumTask) -> Dict[str, Any]:
        """Execute a single quantum task."""
        start_time = time.time()
        
        # Collapse task state to ACTIVE
        task.update_amplitude(TaskState.ACTIVE, complex(1.0, 0.0))
        task.update_amplitude(TaskState.PENDING, complex(0.0, 0.0))
        
        task.start_time = start_time
        
        try:
            # Execute task
            if task.executor:
                result = task.executor()
                if asyncio.iscoroutine(result):
                    result = await result
            else:
                # Simulate execution
                await asyncio.sleep(min(task.estimated_duration, 0.1))
                result = {"status": "completed", "task_id": task.id}
            
            # Success - collapse to COMPLETED
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            task.result = result
            
            task.update_amplitude(TaskState.COMPLETED, complex(1.0, 0.0))
            task.update_amplitude(TaskState.ACTIVE, complex(0.0, 0.0))
            
            return {
                "status": "success",
                "result": result,
                "duration": task.actual_duration
            }
            
        except Exception as e:
            # Failure - collapse to FAILED
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            
            task.update_amplitude(TaskState.FAILED, complex(1.0, 0.0))
            task.update_amplitude(TaskState.ACTIVE, complex(0.0, 0.0))
            
            return {
                "status": "failed", 
                "error": str(e),
                "duration": task.actual_duration
            }
    
    def get_metrics(self) -> SchedulingMetrics:
        """Get scheduler performance metrics."""
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0.0
        
        # Calculate throughput
        total_time = sum(self.execution_times) or 1.0
        throughput = (self.completed_tasks + self.failed_tasks) / total_time
        
        # Resource utilization (simplified)
        avg_resource_util = np.mean([
            sum(usage.values()) / len(usage) if usage else 0.0
            for usage in self.resource_usage_history
        ]) if self.resource_usage_history else 0.0
        
        # Quantum efficiency
        total_tasks = self.completed_tasks + self.failed_tasks
        success_rate = self.completed_tasks / max(total_tasks, 1)
        quantum_efficiency = min(1.0, success_rate + 0.1)  # Quantum bonus
        
        return SchedulingMetrics(
            total_execution_time=avg_execution_time,
            tasks_completed=self.completed_tasks,
            tasks_failed=self.failed_tasks,
            resource_utilization={"average": avg_resource_util},
            quantum_efficiency=quantum_efficiency,
            scheduling_overhead=self.total_scheduling_time,
            throughput=throughput
        )


class AdaptiveScheduler(BaseScheduler):
    """
    Self-tuning scheduler that learns optimal quantum parameters.
    
    Uses reinforcement learning to adapt scheduling strategy based on
    system performance and workload characteristics.
    """
    
    def __init__(
        self,
        base_scheduler: Optional[QuantumScheduler] = None,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1,
        adaptation_window: int = 100,
    ):
        self.base_scheduler = base_scheduler or QuantumScheduler()
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.adaptation_window = adaptation_window
        
        # Learning parameters
        self.parameter_history: List[Dict[str, float]] = []
        self.performance_history: List[float] = []
        self.adaptation_count = 0
        
        # Current parameters
        self.adaptive_params = {
            "max_concurrent_tasks": 4,
            "measurement_interval": 1.0,
            "interference_strength": 0.1,
        }
    
    async def schedule(self, planner: QuantumTaskPlanner) -> Dict[str, Any]:
        """Execute adaptive scheduling with parameter learning."""
        # Apply current parameters to base scheduler
        self._apply_parameters_to_scheduler()
        
        # Execute scheduling
        result = await self.base_scheduler.schedule(planner)
        
        # Learn from performance
        await self._learn_from_performance(result)
        
        return result
    
    def _apply_parameters_to_scheduler(self):
        """Apply learned parameters to base scheduler."""
        self.base_scheduler.max_concurrent_tasks = int(self.adaptive_params["max_concurrent_tasks"])
        self.base_scheduler.measurement_interval = self.adaptive_params["measurement_interval"]
        self.base_scheduler.interference_optimization = self.adaptive_params["interference_strength"] > 0.05
    
    async def _learn_from_performance(self, scheduling_result: Dict[str, Any]):
        """Learn and adapt parameters based on scheduling performance."""
        if scheduling_result["status"] != "success":
            return
        
        # Calculate performance metric
        execution_results = scheduling_result.get("execution_results", {})
        total_tasks = len(execution_results.get("task_results", {}))
        successful_tasks = sum(
            1 for result in execution_results.get("task_results", {}).values()
            if result.get("status") == "success"
        )
        
        performance = successful_tasks / max(total_tasks, 1)
        
        # Store performance
        self.performance_history.append(performance)
        self.parameter_history.append(dict(self.adaptive_params))
        
        # Adapt parameters
        self.adaptation_count += 1
        if self.adaptation_count % self.adaptation_window == 0:
            await self._adapt_parameters()
    
    async def _adapt_parameters(self):
        """Adapt scheduling parameters based on performance history."""
        if len(self.performance_history) < self.adaptation_window:
            return
        
        recent_performance = self.performance_history[-self.adaptation_window:]
        recent_params = self.parameter_history[-self.adaptation_window:]
        
        # Simple gradient-based adaptation
        best_idx = np.argmax(recent_performance)
        best_params = recent_params[best_idx]
        
        # Move towards best parameters with exploration
        for param_name in self.adaptive_params:
            if param_name in best_params:
                current_val = self.adaptive_params[param_name]
                best_val = best_params[param_name]
                
                # Gradient step towards best
                gradient = (best_val - current_val) * self.learning_rate
                
                # Add exploration noise
                key = random.PRNGKey(int(time.time() * 1e6) % 2**32)
                noise = random.normal(key) * self.exploration_rate * abs(current_val)
                
                # Update parameter
                new_val = current_val + gradient + noise
                
                # Apply bounds
                if param_name == "max_concurrent_tasks":
                    new_val = max(1, min(16, int(new_val)))
                elif param_name == "measurement_interval":
                    new_val = max(0.1, min(10.0, new_val))
                elif param_name == "interference_strength":
                    new_val = max(0.0, min(1.0, new_val))
                
                self.adaptive_params[param_name] = new_val
    
    def get_metrics(self) -> SchedulingMetrics:
        """Get adaptive scheduler metrics."""
        base_metrics = self.base_scheduler.get_metrics()
        
        # Add adaptation metrics
        adaptation_efficiency = 0.0
        if len(self.performance_history) >= 2:
            recent_perf = np.mean(self.performance_history[-10:])
            initial_perf = np.mean(self.performance_history[:10])
            adaptation_efficiency = max(0.0, recent_perf - initial_perf)
        
        # Enhanced quantum efficiency with adaptation bonus
        enhanced_efficiency = min(1.0, base_metrics.quantum_efficiency + adaptation_efficiency)
        
        return SchedulingMetrics(
            total_execution_time=base_metrics.total_execution_time,
            tasks_completed=base_metrics.tasks_completed,
            tasks_failed=base_metrics.tasks_failed,
            resource_utilization=base_metrics.resource_utilization,
            quantum_efficiency=enhanced_efficiency,
            scheduling_overhead=base_metrics.scheduling_overhead,
            throughput=base_metrics.throughput,
        )