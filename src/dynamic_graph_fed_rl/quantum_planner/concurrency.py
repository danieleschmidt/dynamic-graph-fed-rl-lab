"""
Concurrency and parallelization for quantum task planner.

Implements advanced concurrency patterns:
- Async quantum state management with proper isolation
- Thread-safe quantum operations with lock-free data structures
- Process pool management for CPU-intensive quantum computations
- Actor model for distributed quantum computation
- Deadlock prevention and resource management
"""

import asyncio
import threading
import multiprocessing
import time
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable, Union, Tuple, Awaitable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, deque
import weakref
import uuid

import jax
import jax.numpy as jnp
from jax import random

from .core import QuantumTask, TaskState, QuantumTaskPlanner
from .exceptions import QuantumPlannerError
from .performance import PerformanceManager


@dataclass
class ConcurrencyMetrics:
    """Metrics for concurrency performance tracking."""
    active_threads: int
    active_processes: int
    queue_sizes: Dict[str, int]
    lock_contentions: int
    deadlock_detections: int
    task_throughput: float  # tasks per second
    resource_utilization: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class QuantumStateLock:
    """
    Specialized lock for quantum state operations.
    
    Provides reader-writer semantics with quantum coherence awareness.
    """
    
    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(threading.RLock())
        self._write_ready = threading.Condition(threading.RLock())
        self._coherence_version = 0
        self._pending_measurements = 0
    
    def acquire_read(self, coherence_version: Optional[int] = None):
        """Acquire read lock with coherence check."""
        self._read_ready.acquire()
        try:
            while self._writers > 0:
                self._read_ready.wait()
            
            # Check coherence version if provided
            if coherence_version is not None and coherence_version != self._coherence_version:
                # Coherence has changed - need to re-acquire with new version
                return False
            
            self._readers += 1
            return True
        finally:
            self._read_ready.release()
    
    def release_read(self):
        """Release read lock."""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()
    
    def acquire_write(self):
        """Acquire write lock (exclusive)."""
        self._write_ready.acquire()
        try:
            while self._writers > 0 or self._readers > 0:
                self._write_ready.wait()
            self._writers = 1
            return True
        finally:
            self._write_ready.release()
    
    def release_write(self, increment_coherence: bool = True):
        """Release write lock and optionally increment coherence."""
        self._write_ready.acquire()
        try:
            self._writers = 0
            if increment_coherence:
                self._coherence_version += 1
            self._write_ready.notify_all()
        finally:
            self._write_ready.release()
    
    def get_coherence_version(self) -> int:
        """Get current coherence version."""
        return self._coherence_version


class AsyncQuantumStateManager:
    """
    Async manager for quantum task states with proper isolation.
    
    Manages concurrent access to quantum states while maintaining coherence.
    """
    
    def __init__(self, max_concurrent_operations: int = 100):
        self.max_concurrent_operations = max_concurrent_operations
        
        # State storage with locks
        self.task_states: Dict[str, QuantumTask] = {}
        self.state_locks: Dict[str, QuantumStateLock] = {}
        self.operation_semaphore = asyncio.Semaphore(max_concurrent_operations)
        
        # Async event system
        self.state_change_events: Dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.operation_queue = asyncio.Queue()
        
        # Statistics
        self.operation_count = 0
        self.lock_contentions = 0
        
    async def register_task(self, task: QuantumTask):
        """Register task for concurrent state management."""
        async with self.operation_semaphore:
            task_id = task.id
            
            # Create state lock if needed
            if task_id not in self.state_locks:
                self.state_locks[task_id] = QuantumStateLock()
            
            # Store task with write lock
            lock = self.state_locks[task_id]
            if await asyncio.get_event_loop().run_in_executor(None, lock.acquire_write):
                try:
                    self.task_states[task_id] = task
                    self.operation_count += 1
                finally:
                    await asyncio.get_event_loop().run_in_executor(None, lock.release_write)
    
    async def update_task_state(
        self, 
        task_id: str, 
        new_state: TaskState, 
        amplitude: complex
    ):
        """Update task quantum state asynchronously."""
        if task_id not in self.state_locks:
            return False
        
        async with self.operation_semaphore:
            lock = self.state_locks[task_id]
            
            # Acquire write lock
            acquired = await asyncio.get_event_loop().run_in_executor(None, lock.acquire_write)
            if not acquired:
                self.lock_contentions += 1
                return False
            
            try:
                if task_id in self.task_states:
                    task = self.task_states[task_id]
                    task.update_amplitude(new_state, amplitude)
                    
                    # Notify listeners of state change
                    self.state_change_events[task_id].set()
                    self.state_change_events[task_id] = asyncio.Event()  # Reset for next change
                    
                    return True
                return False
            finally:
                await asyncio.get_event_loop().run_in_executor(None, lock.release_write)
    
    async def read_task_state(
        self, 
        task_id: str,
        coherence_version: Optional[int] = None
    ) -> Optional[Dict[TaskState, complex]]:
        """Read task quantum state with coherence check."""
        if task_id not in self.state_locks:
            return None
        
        async with self.operation_semaphore:
            lock = self.state_locks[task_id]
            
            # Acquire read lock
            acquired = await asyncio.get_event_loop().run_in_executor(
                None, lock.acquire_read, coherence_version
            )
            if not acquired:
                return None
            
            try:
                if task_id in self.task_states:
                    task = self.task_states[task_id]
                    return dict(task.state_amplitudes)
                return None
            finally:
                await asyncio.get_event_loop().run_in_executor(None, lock.release_read)
    
    async def wait_for_state_change(self, task_id: str, timeout: float = 10.0):
        """Wait for task state to change."""
        if task_id in self.state_change_events:
            try:
                await asyncio.wait_for(
                    self.state_change_events[task_id].wait(),
                    timeout=timeout
                )
                return True
            except asyncio.TimeoutError:
                return False
        return False
    
    async def batch_update_states(
        self, 
        updates: List[Tuple[str, TaskState, complex]]
    ):
        """Update multiple task states in batch for efficiency."""
        # Sort updates by task_id to prevent deadlocks
        updates.sort(key=lambda x: x[0])
        
        # Acquire all locks first (in order)
        acquired_locks = []
        try:
            for task_id, _, _ in updates:
                if task_id in self.state_locks:
                    lock = self.state_locks[task_id]
                    acquired = await asyncio.get_event_loop().run_in_executor(None, lock.acquire_write)
                    if acquired:
                        acquired_locks.append((task_id, lock))
                    else:
                        # Failed to acquire lock - abort batch update
                        break
            
            # Apply updates
            successful_updates = 0
            for (task_id, new_state, amplitude) in updates:
                if task_id in self.task_states and any(tid == task_id for tid, _ in acquired_locks):
                    task = self.task_states[task_id]
                    task.update_amplitude(new_state, amplitude)
                    
                    # Notify state change
                    self.state_change_events[task_id].set()
                    self.state_change_events[task_id] = asyncio.Event()
                    
                    successful_updates += 1
            
            return successful_updates
            
        finally:
            # Release all acquired locks
            for task_id, lock in reversed(acquired_locks):
                await asyncio.get_event_loop().run_in_executor(None, lock.release_write)


class QuantumWorkerPool:
    """
    Worker pool for quantum computations with intelligent task distribution.
    
    Manages both thread and process pools for optimal quantum computation performance.
    """
    
    def __init__(
        self,
        max_threads: int = 8,
        max_processes: int = 4,
        quantum_batch_size: int = 32
    ):
        self.max_threads = max_threads
        self.max_processes = max_processes
        self.quantum_batch_size = quantum_batch_size
        
        # Worker pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=max_processes)
        
        # Task queues
        self.light_tasks = asyncio.Queue()  # Thread pool tasks
        self.heavy_tasks = asyncio.Queue()  # Process pool tasks
        self.quantum_tasks = asyncio.Queue()  # Quantum computation tasks
        
        # Worker management
        self.active_workers = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Resource tracking
        self.resource_usage = defaultdict(float)
        self.task_execution_times = defaultdict(list)
        
    async def submit_light_task(
        self, 
        func: Callable, 
        *args, 
        task_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Submit lightweight task to thread pool."""
        task_id = task_id or str(uuid.uuid4())
        
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        try:
            future = loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
            result = await future
            
            execution_time = time.time() - start_time
            self.task_execution_times["light"].append(execution_time)
            self.completed_tasks += 1
            
            return result
            
        except Exception as e:
            self.failed_tasks += 1
            raise QuantumPlannerError(
                f"Light task {task_id} failed: {str(e)}",
                "TASK_EXECUTION_ERROR",
                {"task_id": task_id, "error": str(e)}
            )
    
    async def submit_heavy_task(
        self, 
        func: Callable, 
        *args, 
        task_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Submit CPU-intensive task to process pool."""
        task_id = task_id or str(uuid.uuid4())
        
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        try:
            future = loop.run_in_executor(self.process_pool, func, *args, **kwargs)
            result = await future
            
            execution_time = time.time() - start_time
            self.task_execution_times["heavy"].append(execution_time)
            self.completed_tasks += 1
            
            return result
            
        except Exception as e:
            self.failed_tasks += 1
            raise QuantumPlannerError(
                f"Heavy task {task_id} failed: {str(e)}",
                "TASK_EXECUTION_ERROR",
                {"task_id": task_id, "error": str(e)}
            )
    
    async def submit_quantum_batch(
        self, 
        quantum_operations: List[Tuple[Callable, tuple, dict]],
        task_id: Optional[str] = None
    ) -> List[Any]:
        """Submit batch of quantum operations for parallel execution."""
        task_id = task_id or str(uuid.uuid4())
        start_time = time.time()
        
        if len(quantum_operations) <= 4:
            # Small batch - use thread pool
            tasks = []
            for func, args, kwargs in quantum_operations:
                task = self.submit_light_task(func, *args, task_id=f"{task_id}_sub", **kwargs)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        else:
            # Large batch - use process pool with chunking
            chunk_size = max(1, len(quantum_operations) // self.max_processes)
            chunks = [
                quantum_operations[i:i + chunk_size]
                for i in range(0, len(quantum_operations), chunk_size)
            ]
            
            chunk_tasks = []
            for i, chunk in enumerate(chunks):
                chunk_task = self.submit_heavy_task(
                    self._process_quantum_chunk,
                    chunk,
                    task_id=f"{task_id}_chunk_{i}"
                )
                chunk_tasks.append(chunk_task)
            
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Flatten results
            results = []
            for chunk_result in chunk_results:
                if isinstance(chunk_result, list):
                    results.extend(chunk_result)
                else:
                    results.append(chunk_result)
        
        execution_time = time.time() - start_time
        self.task_execution_times["quantum_batch"].append(execution_time)
        
        return results
    
    def _process_quantum_chunk(
        self, 
        chunk: List[Tuple[Callable, tuple, dict]]
    ) -> List[Any]:
        """Process chunk of quantum operations in separate process."""
        results = []
        
        for func, args, kwargs in chunk:
            try:
                result = func(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results
    
    async def map_quantum_operations(
        self,
        func: Callable,
        task_args_list: List[tuple],
        concurrent_limit: int = None
    ) -> List[Any]:
        """Map quantum operation across multiple arguments with concurrency control."""
        concurrent_limit = concurrent_limit or self.max_threads
        
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def bounded_task(args):
            async with semaphore:
                return await self.submit_light_task(func, *args)
        
        tasks = [bounded_task(args) for args in task_args_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        def calculate_avg_time(times_list):
            return sum(times_list) / len(times_list) if times_list else 0.0
        
        return {
            "active_workers": self.active_workers,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / max(self.completed_tasks + self.failed_tasks, 1),
            "avg_execution_times": {
                task_type: calculate_avg_time(times)
                for task_type, times in self.task_execution_times.items()
            },
            "queue_sizes": {
                "light_tasks": self.light_tasks.qsize(),
                "heavy_tasks": self.heavy_tasks.qsize(),
                "quantum_tasks": self.quantum_tasks.qsize()
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown worker pools."""
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)
        
        # Shutdown process pool
        self.process_pool.shutdown(wait=False)
        
        # Clear queues
        while not self.light_tasks.empty():
            try:
                self.light_tasks.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.heavy_tasks.empty():
            try:
                self.heavy_tasks.get_nowait()
            except asyncio.QueueEmpty:
                break


class QuantumActor:
    """
    Actor model implementation for distributed quantum computation.
    
    Each actor manages its own quantum state and communicates via message passing.
    """
    
    def __init__(self, actor_id: str, max_queue_size: int = 1000):
        self.actor_id = actor_id
        self.max_queue_size = max_queue_size
        
        # Actor state
        self.quantum_state: Dict[str, QuantumTask] = {}
        self.message_queue = asyncio.Queue(maxsize=max_queue_size)
        self.running = False
        
        # Message handling
        self.message_handlers: Dict[str, Callable] = {
            "update_task": self._handle_update_task,
            "compute_superposition": self._handle_compute_superposition,
            "measure_state": self._handle_measure_state,
            "entangle_tasks": self._handle_entangle_tasks
        }
        
        # Statistics
        self.messages_processed = 0
        self.processing_errors = 0
        
    async def start(self):
        """Start the actor message processing loop."""
        self.running = True
        asyncio.create_task(self._message_loop())
    
    async def stop(self):
        """Stop the actor."""
        self.running = False
    
    async def send_message(
        self, 
        message_type: str, 
        data: Dict[str, Any],
        timeout: float = 10.0
    ) -> Any:
        """Send message to actor and wait for response."""
        message_id = str(uuid.uuid4())
        response_future = asyncio.Future()
        
        message = {
            "id": message_id,
            "type": message_type,
            "data": data,
            "response_future": response_future,
            "timestamp": time.time()
        }
        
        try:
            await asyncio.wait_for(
                self.message_queue.put(message),
                timeout=timeout
            )
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            raise QuantumPlannerError(
                f"Actor {self.actor_id} message timeout",
                "ACTOR_TIMEOUT",
                {"message_type": message_type, "timeout": timeout}
            )
    
    async def _message_loop(self):
        """Main message processing loop."""
        while self.running:
            try:
                # Get next message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Process message
                await self._process_message(message)
                self.messages_processed += 1
                
            except asyncio.TimeoutError:
                # Timeout is normal - just continue
                continue
            except Exception as e:
                self.processing_errors += 1
                # Log error but continue processing
                print(f"Actor {self.actor_id} processing error: {e}")
    
    async def _process_message(self, message: Dict[str, Any]):
        """Process individual message."""
        message_type = message["type"]
        message_data = message["data"]
        response_future = message["response_future"]
        
        try:
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                result = await handler(message_data)
                response_future.set_result(result)
            else:
                response_future.set_exception(
                    QuantumPlannerError(
                        f"Unknown message type: {message_type}",
                        "UNKNOWN_MESSAGE_TYPE"
                    )
                )
        except Exception as e:
            response_future.set_exception(e)
    
    async def _handle_update_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task update message."""
        task_id = data["task_id"]
        task_data = data["task"]
        
        # Update local quantum state
        if isinstance(task_data, QuantumTask):
            self.quantum_state[task_id] = task_data
        
        return {"status": "updated", "task_id": task_id}
    
    async def _handle_compute_superposition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle superposition computation message."""
        task_ids = data["task_ids"]
        
        # Compute superposition for specified tasks
        superposition_data = {}
        
        for task_id in task_ids:
            if task_id in self.quantum_state:
                task = self.quantum_state[task_id]
                
                # Calculate superposition properties
                probabilities = {
                    state.value: task.get_probability(state)
                    for state in TaskState
                }
                
                superposition_data[task_id] = {
                    "probabilities": probabilities,
                    "entangled_tasks": list(task.entangled_tasks)
                }
        
        return {"superposition": superposition_data}
    
    async def _handle_measure_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum measurement message."""
        task_id = data["task_id"]
        
        if task_id in self.quantum_state:
            task = self.quantum_state[task_id]
            
            # Perform quantum measurement
            measured_state = task.collapse_state()
            
            return {
                "task_id": task_id,
                "measured_state": measured_state.value,
                "timestamp": time.time()
            }
        else:
            raise QuantumPlannerError(
                f"Task {task_id} not found in actor {self.actor_id}",
                "TASK_NOT_FOUND"
            )
    
    async def _handle_entangle_tasks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task entanglement message."""
        task1_id = data["task1_id"]
        task2_id = data["task2_id"]
        entanglement_strength = data.get("strength", 0.5)
        
        if task1_id in self.quantum_state and task2_id in self.quantum_state:
            task1 = self.quantum_state[task1_id]
            task2 = self.quantum_state[task2_id]
            
            # Create entanglement
            task1.entangled_tasks.add(task2_id)
            task2.entangled_tasks.add(task1_id)
            
            return {
                "entangled": [task1_id, task2_id],
                "strength": entanglement_strength
            }
        else:
            missing = []
            if task1_id not in self.quantum_state:
                missing.append(task1_id)
            if task2_id not in self.quantum_state:
                missing.append(task2_id)
            
            raise QuantumPlannerError(
                f"Tasks not found: {missing}",
                "TASKS_NOT_FOUND",
                {"missing_tasks": missing}
            )
    
    def get_actor_stats(self) -> Dict[str, Any]:
        """Get actor performance statistics."""
        return {
            "actor_id": self.actor_id,
            "running": self.running,
            "quantum_states": len(self.quantum_state),
            "queue_size": self.message_queue.qsize(),
            "messages_processed": self.messages_processed,
            "processing_errors": self.processing_errors,
            "error_rate": self.processing_errors / max(self.messages_processed, 1)
        }


class ConcurrencyManager:
    """
    Main concurrency manager coordinating all concurrent operations.
    
    Provides unified interface for quantum parallel processing.
    """
    
    def __init__(
        self,
        max_threads: int = 8,
        max_processes: int = 4,
        max_actors: int = 10,
        performance_manager: Optional[PerformanceManager] = None
    ):
        # Core components
        self.state_manager = AsyncQuantumStateManager()
        self.worker_pool = QuantumWorkerPool(max_threads, max_processes)
        self.performance_manager = performance_manager
        
        # Actor system
        self.actors: Dict[str, QuantumActor] = {}
        self.max_actors = max_actors
        
        # Concurrency control
        self.global_lock = asyncio.Lock()
        self.resource_semaphores: Dict[str, asyncio.Semaphore] = {
            "cpu": asyncio.Semaphore(max_threads + max_processes),
            "memory": asyncio.Semaphore(100),  # Memory units
            "quantum_coherence": asyncio.Semaphore(50)  # Coherence operations
        }
        
        # Metrics
        self.operation_metrics = ConcurrencyMetrics(
            active_threads=0,
            active_processes=0,
            queue_sizes={},
            lock_contentions=0,
            deadlock_detections=0,
            task_throughput=0.0,
            resource_utilization={}
        )
        
        # Deadlock prevention
        self.lock_graph: Dict[str, Set[str]] = defaultdict(set)
        self.resource_owners: Dict[str, str] = {}
    
    async def start(self):
        """Start concurrency management system."""
        # Initialize core components
        await self._initialize_default_actors()
        
        # Start background monitoring
        asyncio.create_task(self._monitor_performance())
    
    async def _initialize_default_actors(self):
        """Initialize default actor system."""
        actor_types = ["optimizer", "scheduler", "executor", "monitor"]
        
        for actor_type in actor_types:
            actor_id = f"{actor_type}_actor"
            actor = QuantumActor(actor_id)
            self.actors[actor_id] = actor
            await actor.start()
    
    async def execute_quantum_operation(
        self,
        operation: Callable,
        *args,
        concurrency_type: str = "auto",
        resource_requirements: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> Any:
        """Execute quantum operation with optimal concurrency."""
        resource_requirements = resource_requirements or {"cpu": 1}
        
        # Acquire required resources
        acquired_resources = []
        try:
            for resource, count in resource_requirements.items():
                if resource in self.resource_semaphores:
                    semaphore = self.resource_semaphores[resource]
                    for _ in range(count):
                        await semaphore.acquire()
                        acquired_resources.append(resource)
            
            # Determine optimal execution strategy
            if concurrency_type == "auto":
                concurrency_type = self._determine_concurrency_type(operation, args)
            
            # Execute operation
            start_time = time.time()
            
            if concurrency_type == "light":
                result = await self.worker_pool.submit_light_task(operation, *args, **kwargs)
            elif concurrency_type == "heavy":
                result = await self.worker_pool.submit_heavy_task(operation, *args, **kwargs)
            elif concurrency_type == "actor":
                result = await self._execute_via_actor(operation, *args, **kwargs)
            else:
                # Fallback to direct execution
                result = operation(*args, **kwargs)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_throughput_metrics(execution_time)
            
            return result
            
        finally:
            # Release acquired resources
            for resource in acquired_resources:
                if resource in self.resource_semaphores:
                    self.resource_semaphores[resource].release()
    
    def _determine_concurrency_type(self, operation: Callable, args: tuple) -> str:
        """Automatically determine best concurrency type."""
        # Simple heuristic based on operation characteristics
        operation_name = getattr(operation, "__name__", "unknown")
        
        if "jit" in operation_name.lower() or "vectorized" in operation_name.lower():
            return "heavy"  # JIT compilation benefits from process isolation
        elif "quantum" in operation_name.lower():
            return "light"  # Quantum operations are often I/O bound
        elif len(args) > 10:
            return "heavy"  # Large argument sets suggest heavy computation
        else:
            return "light"  # Default to thread pool
    
    async def _execute_via_actor(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation via actor system."""
        # Find least busy actor
        best_actor = None
        min_queue_size = float('inf')
        
        for actor in self.actors.values():
            queue_size = actor.message_queue.qsize()
            if queue_size < min_queue_size:
                min_queue_size = queue_size
                best_actor = actor
        
        if best_actor:
            # Send computation message to actor
            message_data = {
                "operation": operation.__name__,
                "args": args,
                "kwargs": kwargs
            }
            
            return await best_actor.send_message("compute", message_data)
        else:
            # Fallback to worker pool
            return await self.worker_pool.submit_light_task(operation, *args, **kwargs)
    
    async def parallel_quantum_evolution(
        self,
        tasks: Dict[str, QuantumTask],
        evolution_func: Callable,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Execute quantum evolution in parallel batches."""
        task_items = list(tasks.items())
        results = {}
        
        # Process in batches for optimal memory usage
        for i in range(0, len(task_items), batch_size):
            batch = task_items[i:i + batch_size]
            
            # Create batch evolution tasks
            batch_tasks = []
            for task_id, task in batch:
                evolution_task = self.execute_quantum_operation(
                    evolution_func,
                    task,
                    concurrency_type="light"
                )
                batch_tasks.append((task_id, evolution_task))
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(
                *[task for _, task in batch_tasks],
                return_exceptions=True
            )
            
            # Collect results
            for (task_id, _), result in zip(batch_tasks, batch_results):
                if not isinstance(result, Exception):
                    results[task_id] = result
                else:
                    results[task_id] = {"error": str(result)}
        
        return results
    
    async def distributed_optimization(
        self,
        optimization_func: Callable,
        parameter_space: List[Dict[str, Any]],
        max_parallel: int = None
    ) -> List[Any]:
        """Run distributed optimization across parameter space."""
        max_parallel = max_parallel or len(self.actors)
        
        # Create optimization tasks
        optimization_operations = [
            (optimization_func, (params,), {})
            for params in parameter_space
        ]
        
        # Execute via worker pool batch processing
        results = await self.worker_pool.submit_quantum_batch(
            optimization_operations
        )
        
        return results
    
    def _update_throughput_metrics(self, execution_time: float):
        """Update throughput and performance metrics."""
        current_time = time.time()
        
        # Simple throughput calculation
        if execution_time > 0:
            throughput = 1.0 / execution_time
            
            # Exponential moving average
            alpha = 0.1
            self.operation_metrics.task_throughput = (
                alpha * throughput + 
                (1 - alpha) * self.operation_metrics.task_throughput
            )
        
        self.operation_metrics.timestamp = current_time
    
    async def _monitor_performance(self):
        """Background performance monitoring."""
        while True:
            try:
                # Update resource utilization
                for resource, semaphore in self.resource_semaphores.items():
                    # Estimate utilization based on available permits
                    available = semaphore._value
                    total = available + len(semaphore._waiters) if hasattr(semaphore, '_waiters') else available
                    utilization = 1.0 - (available / max(total, 1))
                    self.operation_metrics.resource_utilization[resource] = utilization
                
                # Update queue sizes
                self.operation_metrics.queue_sizes = {
                    "state_manager": 0,  # Placeholder
                    **self.worker_pool.get_pool_stats()["queue_sizes"]
                }
                
                # Update active workers
                pool_stats = self.worker_pool.get_pool_stats()
                self.operation_metrics.active_threads = pool_stats.get("active_workers", 0)
                
                # Auto-tune performance if manager available
                if self.performance_manager:
                    self.performance_manager.auto_tune_performance()
                
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                await asyncio.sleep(10.0)  # Longer pause on error
    
    async def shutdown(self):
        """Gracefully shutdown concurrency system."""
        # Stop actors
        for actor in self.actors.values():
            await actor.stop()
        
        # Shutdown worker pool
        await self.worker_pool.shutdown()
    
    def get_concurrency_metrics(self) -> ConcurrencyMetrics:
        """Get current concurrency performance metrics."""
        return self.operation_metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "concurrency_metrics": self.operation_metrics,
            "worker_pool_stats": self.worker_pool.get_pool_stats(),
            "actor_stats": {
                actor_id: actor.get_actor_stats()
                for actor_id, actor in self.actors.items()
            },
            "resource_utilization": dict(self.operation_metrics.resource_utilization),
            "system_health": "healthy" if self.operation_metrics.task_throughput > 0 else "degraded"
        }