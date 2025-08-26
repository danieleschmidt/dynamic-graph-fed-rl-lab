import secrets
"""
Quantum-inspired executors for task execution.

Implements execution engines with quantum principles:
- QuantumExecutor: Probabilistic execution with state collapse
- ParallelExecutor: Concurrent execution with quantum resource management
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Set
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .core import QuantumTask, TaskState


@dataclass
class ExecutionMetrics:
    """Metrics for task execution performance."""
    total_execution_time: float
    tasks_executed: int
    tasks_successful: int
    tasks_failed: int
    resource_efficiency: float
    quantum_fidelity: float  # How well quantum properties were preserved
    parallelism_achieved: float
    error_rate: float


class BaseExecutor(ABC):
    """Abstract base executor interface."""
    
    @abstractmethod
    async def execute_tasks(
        self, 
        tasks: Dict[str, QuantumTask],
        execution_order: List[str]
    ) -> Dict[str, Any]:
        """Execute tasks according to specified order."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> ExecutionMetrics:
        """Get execution performance metrics."""
        pass


class QuantumExecutor(BaseExecutor):
    """
    Quantum-inspired task executor with probabilistic state collapse.
    
    Maintains quantum superposition until execution measurement,
    then collapses to deterministic execution states.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 4,
        measurement_interval: float = 1.0,
        decoherence_time: float = 10.0,
        error_probability: float = 0.05,
        resource_limits: Optional[Dict[str, float]] = None,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.measurement_interval = measurement_interval
        self.decoherence_time = decoherence_time
        self.error_probability = error_probability
        self.resource_limits = resource_limits or {"cpu": 1.0, "memory": 1.0, "io": 1.0}
        
        # Execution state
        self.active_tasks: Set[str] = set()
        self.resource_usage: Dict[str, float] = {res: 0.0 for res in self.resource_limits}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Metrics
        self.total_execution_time = 0.0
        self.tasks_executed = 0
        self.tasks_successful = 0
        self.tasks_failed = 0
        self.quantum_measurements = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def execute_tasks(
        self, 
        tasks: Dict[str, QuantumTask],
        execution_order: List[str]
    ) -> Dict[str, Any]:
        """Execute tasks with quantum state management."""
        execution_start = time.time()
        
        # Initialize quantum states
        self._initialize_quantum_states(tasks, execution_order)
        
        # Execute tasks with quantum evolution
        results = await self._quantum_execution_loop(tasks, execution_order)
        
        # Calculate metrics
        total_time = time.time() - execution_start
        self.total_execution_time += total_time
        
        return {
            "execution_results": results,
            "total_time": total_time,
            "quantum_measurements": self.quantum_measurements,
            "resource_utilization": dict(self.resource_usage),
        }
    
    def _initialize_quantum_states(
        self, 
        tasks: Dict[str, QuantumTask],
        execution_order: List[str]
    ):
        """Initialize quantum states for execution."""
        for task_id in execution_order:
            if task_id in tasks:
                task = tasks[task_id]
                
                # Set initial superposition state
                if task.get_probability(TaskState.PENDING) < 0.9:
                    # Reset to pending state for execution
                    task.update_amplitude(TaskState.PENDING, complex(1.0, 0.0))
                    for state in TaskState:
                        if state != TaskState.PENDING:
                            task.update_amplitude(state, complex(0.0, 0.0))
    
    async def _quantum_execution_loop(
        self, 
        tasks: Dict[str, QuantumTask],
        execution_order: List[str]
    ) -> Dict[str, Any]:
        """Main quantum execution loop with state evolution."""
        results = {}
        execution_queue = execution_order.copy()
        last_measurement = time.time()
        
        while execution_queue or self.active_tasks:
            current_time = time.time()
            
            # Quantum decoherence check
            if current_time - last_measurement > self.decoherence_time:
                await self._apply_decoherence(tasks)
                last_measurement = current_time
            
            # Start new tasks if resources available
            await self._start_ready_tasks(tasks, execution_queue)
            
            # Check for completed tasks
            completed_tasks = await self._check_completed_tasks(tasks)
            
            for task_id, result in completed_tasks.items():
                results[task_id] = result
                self.active_tasks.discard(task_id)
                self._release_resources(tasks[task_id])
            
            # Quantum measurement interval
            if current_time - last_measurement >= self.measurement_interval:
                await self._perform_quantum_measurement(tasks)
                last_measurement = current_time
                self.quantum_measurements += 1
            
            # Brief pause to prevent busy waiting
            await asyncio.sleep(0.01)
        
        return results
    
    async def _start_ready_tasks(
        self, 
        tasks: Dict[str, QuantumTask],
        execution_queue: List[str]
    ):
        """Start tasks that are ready for execution."""
        tasks_to_start = []
        
        # Find tasks that can be started
        for task_id in execution_queue[:]:
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                break
            
            if task_id not in tasks:
                execution_queue.remove(task_id)
                continue
            
            task = tasks[task_id]
            
            # Check if task is ready (dependencies completed)
            deps_ready = all(
                tasks[dep_id].get_probability(TaskState.COMPLETED) > 0.9
                for dep_id in task.dependencies
                if dep_id in tasks
            )
            
            if deps_ready and self._can_allocate_resources(task):
                tasks_to_start.append(task_id)
                execution_queue.remove(task_id)
        
        # Start selected tasks
        for task_id in tasks_to_start:
            await self._start_task_execution(tasks[task_id])
    
    def _can_allocate_resources(self, task: QuantumTask) -> bool:
        """Check if resources are available for task."""
        for resource, required in task.resource_requirements.items():
            if resource in self.resource_usage:
                available = self.resource_limits[resource] - self.resource_usage[resource]
                if required > available:
                    return False
        return True
    
    def _allocate_resources(self, task: QuantumTask):
        """Allocate resources for task execution."""
        for resource, required in task.resource_requirements.items():
            if resource in self.resource_usage:
                self.resource_usage[resource] += required
    
    def _release_resources(self, task: QuantumTask):
        """Release resources after task completion."""
        for resource, required in task.resource_requirements.items():
            if resource in self.resource_usage:
                self.resource_usage[resource] = max(0.0, self.resource_usage[resource] - required)
    
    async def _start_task_execution(self, task: QuantumTask):
        """Start execution of a single task."""
        # Quantum state collapse to ACTIVE
        await self._collapse_task_state(task, TaskState.ACTIVE)
        
        # Allocate resources
        self._allocate_resources(task)
        
        # Mark as active
        self.active_tasks.add(task.id)
        task.start_time = time.time()
        
        # Create execution coroutine
        if task.executor:
            if asyncio.iscoroutinefunction(task.executor):
                asyncio.create_task(self._execute_task_async(task))
            else:
                # Run synchronous executor in thread pool
                asyncio.create_task(self._execute_task_sync(task))
        else:
            # Default execution simulation
            asyncio.create_task(self._simulate_task_execution(task))
        
        self.logger.info(f"Started execution of task {task.id}")
    
    async def _execute_task_async(self, task: QuantumTask):
        """Execute async task."""
        try:
            result = await task.executor()
            task.result = result
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            
            # Collapse to success state
            await self._collapse_task_state(task, TaskState.COMPLETED)
            
        except Exception as e:
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            task.result = {"error": str(e)}
            
            # Collapse to failure state
            await self._collapse_task_state(task, TaskState.FAILED)
            
            self.logger.error(f"Task {task.id} failed: {e}")
    
    async def _execute_task_sync(self, task: QuantumTask):
        """Execute synchronous task in thread pool."""
        loop = asyncio.get_event_loop()
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(executor, task.executor)
            
            task.result = result
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            
            # Collapse to success state
            await self._collapse_task_state(task, TaskState.COMPLETED)
            
        except Exception as e:
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            task.result = {"error": str(e)}
            
            # Collapse to failure state
            await self._collapse_task_state(task, TaskState.FAILED)
            
            self.logger.error(f"Task {task.id} failed: {e}")
    
    async def _simulate_task_execution(self, task: QuantumTask):
        """Simulate task execution."""
        try:
            # Simulate work
            execution_time = task.estimated_duration
            
            # Add quantum uncertainty
            uncertainty = np.random.normal(0, 0.1 * execution_time)
            actual_time = max(0.01, execution_time + uncertainty)
            
            await asyncio.sleep(min(actual_time, 0.5))  # Cap simulation time
            
            # Quantum error probability
            if np.secrets.SystemRandom().random() < self.error_probability:
                raise Exception(f"Quantum execution error in task {task.id}")
            
            task.result = {
                "status": "completed", 
                "task_id": task.id,
                "simulated": True
            }
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            
            # Collapse to success state
            await self._collapse_task_state(task, TaskState.COMPLETED)
            
        except Exception as e:
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            task.result = {"error": str(e), "simulated": True}
            
            # Collapse to failure state
            await self._collapse_task_state(task, TaskState.FAILED)
    
    async def _collapse_task_state(self, task: QuantumTask, target_state: TaskState):
        """Collapse quantum task state to definite state."""
        # Set target state to 1.0 amplitude
        task.update_amplitude(target_state, complex(1.0, 0.0))
        
        # Set all other states to 0.0 amplitude
        for state in TaskState:
            if state != target_state:
                task.update_amplitude(state, complex(0.0, 0.0))
        
        self.logger.debug(f"Task {task.id} collapsed to state {target_state.value}")
    
    async def _check_completed_tasks(self, tasks: Dict[str, QuantumTask]) -> Dict[str, Dict[str, Any]]:
        """Check for completed tasks and collect results."""
        completed = {}
        
        for task_id in list(self.active_tasks):
            if task_id in tasks:
                task = tasks[task_id]
                
                # Check if task has collapsed to final state
                completed_prob = task.get_probability(TaskState.COMPLETED)
                failed_prob = task.get_probability(TaskState.FAILED)
                
                if completed_prob > 0.9:
                    completed[task_id] = {
                        "status": "success",
                        "result": task.result,
                        "duration": task.actual_duration,
                        "quantum_fidelity": completed_prob
                    }
                    self.tasks_successful += 1
                    self.tasks_executed += 1
                    
                elif failed_prob > 0.9:
                    completed[task_id] = {
                        "status": "failed",
                        "result": task.result,
                        "duration": task.actual_duration,
                        "quantum_fidelity": failed_prob
                    }
                    self.tasks_failed += 1
                    self.tasks_executed += 1
        
        return completed
    
    async def _perform_quantum_measurement(self, tasks: Dict[str, QuantumTask]):
        """Perform quantum measurement on active tasks."""
        for task_id in self.active_tasks:
            if task_id in tasks:
                task = tasks[task_id]
                
                # Measure task state evolution
                active_prob = task.get_probability(TaskState.ACTIVE)
                
                if active_prob < 0.5:
                    # Task state is becoming uncertain - apply measurement
                    measured_state = task.collapse_state()
                    
                    self.logger.debug(
                        f"Quantum measurement on task {task_id}: collapsed to {measured_state.value}"
                    )
    
    async def _apply_decoherence(self, tasks: Dict[str, QuantumTask]):
        """Apply quantum decoherence to long-running tasks."""
        current_time = time.time()
        
        for task_id in self.active_tasks:
            if task_id in tasks:
                task = tasks[task_id]
                
                if task.start_time and (current_time - task.start_time) > self.decoherence_time:
                    # Apply decoherence - increase failure probability
                    current_amplitudes = dict(task.state_amplitudes)
                    
                    # Gradually shift probability towards failure
                    decoherence_factor = 0.1
                    failure_boost = decoherence_factor * abs(current_amplitudes[TaskState.ACTIVE])
                    
                    new_failure_amp = current_amplitudes[TaskState.FAILED] + complex(failure_boost, 0)
                    new_active_amp = current_amplitudes[TaskState.ACTIVE] * (1 - decoherence_factor)
                    
                    task.update_amplitude(TaskState.FAILED, new_failure_amp)
                    task.update_amplitude(TaskState.ACTIVE, new_active_amp)
                    
                    self.logger.warning(f"Applied decoherence to task {task_id}")
    
    def get_metrics(self) -> ExecutionMetrics:
        """Get execution performance metrics."""
        total_tasks = self.tasks_executed
        success_rate = self.tasks_successful / max(total_tasks, 1)
        error_rate = self.tasks_failed / max(total_tasks, 1)
        
        # Resource efficiency
        avg_resource_usage = np.mean(list(self.resource_usage.values()))
        max_resource_capacity = np.mean(list(self.resource_limits.values()))
        resource_efficiency = avg_resource_usage / max(max_resource_capacity, 1)
        
        # Quantum fidelity (how well quantum properties were preserved)
        quantum_fidelity = success_rate  # Simplified metric
        
        # Parallelism achieved
        parallelism = min(1.0, len(self.active_tasks) / max(self.max_concurrent_tasks, 1))
        
        return ExecutionMetrics(
            total_execution_time=self.total_execution_time,
            tasks_executed=total_tasks,
            tasks_successful=self.tasks_successful,
            tasks_failed=self.tasks_failed,
            resource_efficiency=resource_efficiency,
            quantum_fidelity=quantum_fidelity,
            parallelism_achieved=parallelism,
            error_rate=error_rate
        )


class ParallelExecutor(QuantumExecutor):
    """
    Enhanced parallel executor with quantum resource management.
    
    Extends QuantumExecutor with advanced parallel processing
    and intelligent resource allocation.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 8,
        max_worker_threads: int = 4,
        max_worker_processes: int = 2,
        adaptive_concurrency: bool = True,
        resource_balancing: bool = True,
        **kwargs
    ):
        super().__init__(max_concurrent_tasks=max_concurrent_tasks, **kwargs)
        
        self.max_worker_threads = max_worker_threads
        self.max_worker_processes = max_worker_processes
        self.adaptive_concurrency = adaptive_concurrency
        self.resource_balancing = resource_balancing
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_worker_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=max_worker_processes)
        
        # Adaptive parameters
        self.performance_history: List[float] = []
        self.concurrency_history: List[int] = []
        
        # Resource tracking
        self.task_resource_usage: Dict[str, Dict[str, float]] = {}
    
    async def execute_tasks(
        self, 
        tasks: Dict[str, QuantumTask],
        execution_order: List[str]
    ) -> Dict[str, Any]:
        """Execute tasks with advanced parallel processing."""
        try:
            # Adaptive concurrency adjustment
            if self.adaptive_concurrency:
                self._adjust_concurrency()
            
            # Resource balancing
            if self.resource_balancing:
                execution_order = self._reorder_for_resource_balance(tasks, execution_order)
            
            # Execute with parallel quantum management
            result = await super().execute_tasks(tasks, execution_order)
            
            # Update adaptive parameters
            self._update_performance_history(result)
            
            return result
            
        finally:
            # Cleanup resources
            await self._cleanup_execution_pools()
    
    def _adjust_concurrency(self):
        """Adjust concurrency based on performance history."""
        if len(self.performance_history) < 5:
            return  # Need more history
        
        recent_performance = np.mean(self.performance_history[-5:])
        previous_performance = np.mean(self.performance_history[-10:-5]) if len(self.performance_history) >= 10 else recent_performance
        
        # If performance is improving, try increasing concurrency
        if recent_performance > previous_performance * 1.1:
            self.max_concurrent_tasks = min(16, self.max_concurrent_tasks + 1)
        # If performance is degrading, decrease concurrency
        elif recent_performance < previous_performance * 0.9:
            self.max_concurrent_tasks = max(2, self.max_concurrent_tasks - 1)
        
        self.logger.info(f"Adjusted max_concurrent_tasks to {self.max_concurrent_tasks}")
    
    def _reorder_for_resource_balance(
        self, 
        tasks: Dict[str, QuantumTask],
        execution_order: List[str]
    ) -> List[str]:
        """Reorder tasks for better resource balance."""
        if not execution_order:
            return execution_order
        
        # Group tasks by resource requirements
        resource_groups = {}
        
        for task_id in execution_order:
            if task_id in tasks:
                task = tasks[task_id]
                
                # Create resource signature
                resource_sig = tuple(sorted(task.resource_requirements.items()))
                
                if resource_sig not in resource_groups:
                    resource_groups[resource_sig] = []
                resource_groups[resource_sig].append(task_id)
        
        # Interleave groups for better resource distribution
        balanced_order = []
        group_iterators = [iter(group) for group in resource_groups.values()]
        
        while group_iterators:
            for iterator in group_iterators[:]:
                try:
                    task_id = next(iterator)
                    balanced_order.append(task_id)
                except StopIteration:
                    group_iterators.remove(iterator)
        
        return balanced_order
    
    async def _execute_task_sync(self, task: QuantumTask):
        """Execute synchronous task with intelligent pool selection."""
        # Determine execution pool based on task characteristics
        use_process_pool = self._should_use_process_pool(task)
        
        try:
            if use_process_pool and self.process_pool:
                # CPU-intensive task - use process pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.process_pool, task.executor)
            else:
                # I/O-bound task - use thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.thread_pool, task.executor)
            
            task.result = result
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            
            # Track resource usage
            self._track_task_resources(task)
            
            # Collapse to success state
            await self._collapse_task_state(task, TaskState.COMPLETED)
            
        except Exception as e:
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            task.result = {"error": str(e)}
            
            # Collapse to failure state
            await self._collapse_task_state(task, TaskState.FAILED)
            
            self.logger.error(f"Task {task.id} failed: {e}")
    
    def _should_use_process_pool(self, task: QuantumTask) -> bool:
        """Determine if task should use process pool."""
        # Use process pool for CPU-intensive tasks
        cpu_requirement = task.resource_requirements.get("cpu", 0.0)
        memory_requirement = task.resource_requirements.get("memory", 0.0)
        
        # Heuristic: high CPU or memory requirement suggests process pool
        return cpu_requirement > 0.5 or memory_requirement > 0.7
    
    def _track_task_resources(self, task: QuantumTask):
        """Track actual resource usage for task."""
        if task.actual_duration:
            # Calculate resource efficiency
            estimated_resources = sum(task.resource_requirements.values())
            actual_time_ratio = task.actual_duration / max(task.estimated_duration, 0.1)
            
            # Store resource usage data
            self.task_resource_usage[task.id] = {
                "estimated": estimated_resources,
                "time_ratio": actual_time_ratio,
                "efficiency": estimated_resources / actual_time_ratio if actual_time_ratio > 0 else 0.0
            }
    
    def _update_performance_history(self, result: Dict[str, Any]):
        """Update performance metrics for adaptive adjustment."""
        if "execution_results" in result:
            execution_results = result["execution_results"]
            
            # Calculate performance metric
            successful_tasks = sum(
                1 for res in execution_results.values()
                if isinstance(res, dict) and res.get("status") == "success"
            )
            total_tasks = len(execution_results)
            
            if total_tasks > 0:
                performance = successful_tasks / total_tasks
                self.performance_history.append(performance)
                
                # Keep only recent history
                if len(self.performance_history) > 50:
                    self.performance_history = self.performance_history[-50:]
            
            # Track concurrency
            self.concurrency_history.append(len(self.active_tasks))
    
    async def _cleanup_execution_pools(self):
        """Cleanup execution pools."""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
            
            if self.process_pool:
                self.process_pool.shutdown(wait=False)
                
        except Exception as e:
            self.logger.warning(f"Error during pool cleanup: {e}")
    
    def get_metrics(self) -> ExecutionMetrics:
        """Get enhanced execution metrics."""
        base_metrics = super().get_metrics()
        
        # Enhanced parallelism metric
        if self.concurrency_history:
            avg_concurrency = np.mean(self.concurrency_history)
            parallelism_achieved = avg_concurrency / max(self.max_concurrent_tasks, 1)
        else:
            parallelism_achieved = base_metrics.parallelism_achieved
        
        # Enhanced resource efficiency
        if self.task_resource_usage:
            resource_efficiencies = [
                usage_data["efficiency"] 
                for usage_data in self.task_resource_usage.values()
            ]
            avg_resource_efficiency = np.mean(resource_efficiencies)
        else:
            avg_resource_efficiency = base_metrics.resource_efficiency
        
        return ExecutionMetrics(
            total_execution_time=base_metrics.total_execution_time,
            tasks_executed=base_metrics.tasks_executed,
            tasks_successful=base_metrics.tasks_successful,
            tasks_failed=base_metrics.tasks_failed,
            resource_efficiency=avg_resource_efficiency,
            quantum_fidelity=base_metrics.quantum_fidelity,
            parallelism_achieved=parallelism_achieved,
            error_rate=base_metrics.error_rate
        )