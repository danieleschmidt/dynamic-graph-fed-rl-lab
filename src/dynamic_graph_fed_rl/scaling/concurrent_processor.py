"""Advanced concurrent processing for scalable federated RL systems."""

import asyncio
import threading
import multiprocessing
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import queue
from collections import defaultdict


class ProcessingMode(Enum):
    """Processing modes for different workload characteristics."""
    THREAD_PARALLEL = "thread_parallel"  # I/O bound tasks
    PROCESS_PARALLEL = "process_parallel"  # CPU bound tasks
    ASYNC_CONCURRENT = "async_concurrent"  # Async I/O tasks
    HYBRID_PARALLEL = "hybrid_parallel"  # Mixed workloads
    ADAPTIVE = "adaptive"  # Automatically choose best mode


@dataclass
class ProcessingTask:
    """Task for concurrent processing."""
    
    id: str
    function: Callable
    args: Tuple
    kwargs: Dict
    priority: int = 1
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __hash__(self):
        return hash(self.id)


@dataclass 
class ProcessingResult:
    """Result of concurrent processing task."""
    
    task_id: str
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    retry_count: int = 0
    worker_id: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None


class ConcurrentProcessor:
    """
    Advanced concurrent processor with intelligent workload distribution.
    
    Features:
    - Multiple processing modes (threads, processes, async)
    - Adaptive workload balancing
    - Priority-based task scheduling
    - Automatic retry and error handling
    - Performance monitoring and optimization
    - Resource-aware scaling
    """
    
    def __init__(
        self,
        max_thread_workers: int = 8,
        max_process_workers: int = None,
        default_mode: ProcessingMode = ProcessingMode.ADAPTIVE,
        enable_monitoring: bool = True,
        task_queue_size: int = 10000
    ):
        self.max_thread_workers = max_thread_workers
        self.max_process_workers = max_process_workers or multiprocessing.cpu_count()
        self.default_mode = default_mode
        self.enable_monitoring = enable_monitoring
        
        # Executors
        self.thread_executor = ThreadPoolExecutor(max_workers=max_thread_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_process_workers)
        
        # Task management
        self.task_queue = queue.PriorityQueue(maxsize=task_queue_size)
        self.active_tasks: Dict[str, Future] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        
        # Performance monitoring
        self.performance_stats = ProcessingStatistics() if enable_monitoring else None
        self.workload_analyzer = WorkloadAnalyzer()
        
        # Worker management
        self.worker_states: Dict[str, Dict] = {}
        self.load_balancer = LoadBalancer()
        
        # Control flags
        self.is_running = True
        self.processor_thread = None
        
        print(f"🚀 Concurrent processor initialized: {max_thread_workers} threads, {self.max_process_workers} processes")
        
        # Start background processor
        self._start_background_processor()
    
    def _start_background_processor(self):
        """Start background task processor."""
        self.processor_thread = threading.Thread(
            target=self._process_tasks,
            daemon=True,
            name="ConcurrentProcessor"
        )
        self.processor_thread.start()
    
    def submit_task(self, 
                   task_id: str,
                   function: Callable,
                   args: Tuple = (),
                   kwargs: Dict = None,
                   priority: int = 1,
                   mode: Optional[ProcessingMode] = None,
                   timeout: Optional[float] = None) -> str:
        """Submit task for concurrent processing."""
        
        kwargs = kwargs or {}
        mode = mode or self.default_mode
        
        task = ProcessingTask(
            id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        # Analyze task characteristics for optimal processing mode
        if mode == ProcessingMode.ADAPTIVE:
            mode = self.workload_analyzer.recommend_processing_mode(function, args, kwargs)
        
        try:
            # Priority queue uses negative priority for max-heap behavior
            self.task_queue.put((-priority, time.time(), task, mode), timeout=1.0)
            
            if self.performance_stats:
                self.performance_stats.record_task_submitted()
            
            print(f"📝 Task '{task_id}' submitted with {mode.value} mode (priority: {priority})")
            return task_id
            
        except queue.Full:
            raise RuntimeError(f"Task queue is full, cannot submit task '{task_id}'")
    
    def submit_batch(self,
                    tasks: List[Tuple[str, Callable, Tuple, Dict]],
                    mode: Optional[ProcessingMode] = None,
                    batch_priority: int = 1) -> List[str]:
        """Submit batch of tasks for concurrent processing."""
        
        task_ids = []
        
        for i, (task_id, function, args, kwargs) in enumerate(tasks):
            try:
                submitted_id = self.submit_task(
                    task_id=task_id or f"batch_task_{i}",
                    function=function,
                    args=args,
                    kwargs=kwargs,
                    priority=batch_priority,
                    mode=mode
                )
                task_ids.append(submitted_id)
            except Exception as e:
                print(f"❌ Failed to submit batch task {i}: {e}")
        
        print(f"📦 Submitted batch: {len(task_ids)} tasks")
        return task_ids
    
    def _process_tasks(self):
        """Background task processor."""
        while self.is_running:
            try:
                # Get task from queue with timeout
                try:
                    priority, timestamp, task, mode = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Execute task based on mode
                result = self._execute_task(task, mode)
                
                # Store result
                self.completed_tasks[task.id] = result
                
                # Clean up old completed tasks (keep last 1000)
                if len(self.completed_tasks) > 1000:
                    oldest_tasks = sorted(self.completed_tasks.keys())[:100]
                    for old_task_id in oldest_tasks:
                        del self.completed_tasks[old_task_id]
                
                if self.performance_stats:
                    self.performance_stats.record_task_completed(result)
                
            except Exception as e:
                print(f"❌ Task processor error: {e}")
    
    def _execute_task(self, task: ProcessingTask, mode: ProcessingMode) -> ProcessingResult:
        """Execute task with specified processing mode."""
        start_time = time.time()
        result = ProcessingResult(task_id=task.id)
        
        try:
            if mode == ProcessingMode.THREAD_PARALLEL:
                future = self.thread_executor.submit(task.function, *task.args, **task.kwargs)
                result.worker_id = f"thread_{threading.current_thread().ident}"
            
            elif mode == ProcessingMode.PROCESS_PARALLEL:
                future = self.process_executor.submit(task.function, *task.args, **task.kwargs)
                result.worker_id = f"process_{multiprocessing.current_process().pid}"
            
            elif mode == ProcessingMode.ASYNC_CONCURRENT:
                # For async tasks, run in thread pool but handle async
                future = self.thread_executor.submit(self._run_async_task, task.function, task.args, task.kwargs)
                result.worker_id = f"async_{threading.current_thread().ident}"
            
            else:  # HYBRID_PARALLEL or fallback
                # Choose based on task characteristics
                if self.workload_analyzer.is_cpu_intensive(task.function):
                    future = self.process_executor.submit(task.function, *task.args, **task.kwargs)
                    result.worker_id = f"hybrid_process_{multiprocessing.current_process().pid}"
                else:
                    future = self.thread_executor.submit(task.function, *task.args, **task.kwargs)
                    result.worker_id = f"hybrid_thread_{threading.current_thread().ident}"
            
            # Store active task
            self.active_tasks[task.id] = future
            
            # Wait for completion with timeout
            try:
                result.result = future.result(timeout=task.timeout)
            except Exception as e:
                result.error = e
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    print(f"🔄 Retrying task '{task.id}' (attempt {task.retry_count}/{task.max_retries})")
                    return self._execute_task(task, mode)
            
            # Remove from active tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
        except Exception as e:
            result.error = e
            print(f"❌ Task execution failed: {task.id} - {e}")
        
        result.execution_time = time.time() - start_time
        result.retry_count = task.retry_count
        
        return result
    
    def _run_async_task(self, async_function: Callable, args: Tuple, kwargs: Dict) -> Any:
        """Run async function in thread pool."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(async_function(*args, **kwargs))
            finally:
                loop.close()
        except Exception as e:
            raise e
    
    async def execute_parallel_batch(self,
                                   functions: List[Callable],
                                   batch_args: List[Tuple],
                                   batch_kwargs: List[Dict] = None,
                                   mode: ProcessingMode = ProcessingMode.THREAD_PARALLEL) -> List[ProcessingResult]:
        """Execute batch of functions in parallel."""
        
        batch_kwargs = batch_kwargs or [{}] * len(functions)
        
        # Submit all tasks
        task_ids = []
        for i, (func, args, kwargs) in enumerate(zip(functions, batch_args, batch_kwargs)):
            task_id = f"parallel_batch_{i}_{time.time()}"
            self.submit_task(task_id, func, args, kwargs, mode=mode)
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        timeout_start = time.time()
        timeout_duration = 300.0  # 5 minutes default timeout
        
        while len(results) < len(task_ids) and (time.time() - timeout_start) < timeout_duration:
            for task_id in task_ids:
                if task_id in self.completed_tasks and task_id not in [r.task_id for r in results]:
                    results.append(self.completed_tasks[task_id])
            
            if len(results) < len(task_ids):
                await asyncio.sleep(0.1)  # Brief pause before checking again
        
        print(f"📦 Parallel batch completed: {len(results)}/{len(task_ids)} tasks")
        return results
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> ProcessingResult:
        """Get result of completed task."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Wait for task completion
        start_time = time.time()
        while timeout is None or (time.time() - start_time) < timeout:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            time.sleep(0.1)
        
        # Check if task is still active
        if task_id in self.active_tasks:
            raise TimeoutError(f"Task '{task_id}' is still running after {timeout}s")
        else:
            raise KeyError(f"Task '{task_id}' not found")
    
    def get_active_tasks(self) -> Dict[str, str]:
        """Get information about currently active tasks."""
        active_info = {}
        for task_id, future in self.active_tasks.items():
            if future.running():
                active_info[task_id] = "running"
            elif future.done():
                active_info[task_id] = "done"
            else:
                active_info[task_id] = "pending"
        return active_info
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel active task."""
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            if future.cancel():
                del self.active_tasks[task_id]
                print(f"❌ Cancelled task: {task_id}")
                return True
        return False
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.performance_stats:
            return {"error": "Performance monitoring disabled"}
        
        return {
            **self.performance_stats.get_statistics(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queue_size": self.task_queue.qsize(),
            "thread_workers": self.max_thread_workers,
            "process_workers": self.max_process_workers,
            "workload_analysis": self.workload_analyzer.get_analysis(),
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize processor performance based on current workload."""
        optimization_results = {}
        
        if self.performance_stats:
            stats = self.performance_stats.get_statistics()
            
            # Analyze task completion rates
            if stats["average_execution_time"] > 10.0:
                optimization_results["recommendation"] = "Consider increasing worker count for slow tasks"
            
            if stats["error_rate"] > 0.1:
                optimization_results["warning"] = "High error rate detected, check task implementations"
            
            # Queue analysis
            queue_size = self.task_queue.qsize()
            if queue_size > self.task_queue.maxsize * 0.8:
                optimization_results["queue_warning"] = "Task queue is nearly full, consider increasing capacity"
            
            # Worker utilization
            active_ratio = len(self.active_tasks) / max(self.max_thread_workers + self.max_process_workers, 1)
            if active_ratio > 0.9:
                optimization_results["utilization"] = "High worker utilization - consider scaling up"
            elif active_ratio < 0.3:
                optimization_results["utilization"] = "Low worker utilization - consider scaling down"
            
            optimization_results["current_utilization"] = active_ratio
        
        return optimization_results
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """Shutdown concurrent processor."""
        print("🧹 Shutting down concurrent processor...")
        
        self.is_running = False
        
        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            self.cancel_task(task_id)
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=wait)
        self.process_executor.shutdown(wait=wait)
        
        # Wait for processor thread
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=timeout)
        
        print("✅ Concurrent processor shutdown complete")


class ProcessingStatistics:
    """Statistics tracking for concurrent processing."""
    
    def __init__(self):
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        self.start_time = time.time()
        self.execution_times = []
    
    def record_task_submitted(self):
        self.tasks_submitted += 1
    
    def record_task_completed(self, result: ProcessingResult):
        self.tasks_completed += 1
        self.total_execution_time += result.execution_time
        self.execution_times.append(result.execution_time)
        
        # Keep only recent execution times
        if len(self.execution_times) > 1000:
            self.execution_times = self.execution_times[-500:]
        
        if result.error:
            self.tasks_failed += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        
        return {
            "tasks_submitted": self.tasks_submitted,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": (self.tasks_completed - self.tasks_failed) / max(self.tasks_completed, 1),
            "error_rate": self.tasks_failed / max(self.tasks_completed, 1),
            "average_execution_time": self.total_execution_time / max(self.tasks_completed, 1),
            "tasks_per_second": self.tasks_completed / max(uptime, 1),
            "uptime_seconds": uptime,
        }


class WorkloadAnalyzer:
    """Analyze workload characteristics for optimal processing."""
    
    def __init__(self):
        self.function_profiles = defaultdict(dict)
        self.cpu_intensive_functions = set()
        self.io_intensive_functions = set()
    
    def recommend_processing_mode(self, function: Callable, args: Tuple, kwargs: Dict) -> ProcessingMode:
        """Recommend optimal processing mode based on function characteristics."""
        
        func_name = function.__name__
        
        # Check known classifications
        if func_name in self.cpu_intensive_functions:
            return ProcessingMode.PROCESS_PARALLEL
        elif func_name in self.io_intensive_functions:
            return ProcessingMode.THREAD_PARALLEL
        
        # Analyze function characteristics
        if asyncio.iscoroutinefunction(function):
            return ProcessingMode.ASYNC_CONCURRENT
        
        # Heuristic analysis
        if self._appears_cpu_intensive(function, args):
            self.cpu_intensive_functions.add(func_name)
            return ProcessingMode.PROCESS_PARALLEL
        else:
            self.io_intensive_functions.add(func_name)
            return ProcessingMode.THREAD_PARALLEL
    
    def _appears_cpu_intensive(self, function: Callable, args: Tuple) -> bool:
        """Heuristically determine if function is CPU intensive."""
        
        # Check function name for keywords
        func_name = function.__name__.lower()
        cpu_keywords = ['compute', 'calculate', 'process', 'optimize', 'train', 'learn']
        
        if any(keyword in func_name for keyword in cpu_keywords):
            return True
        
        # Check argument sizes (large data might indicate CPU work)
        total_size = 0
        for arg in args:
            if hasattr(arg, '__len__'):
                total_size += len(arg)
        
        return total_size > 1000  # Arbitrary threshold
    
    def is_cpu_intensive(self, function: Callable) -> bool:
        """Check if function is classified as CPU intensive."""
        return function.__name__ in self.cpu_intensive_functions
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get workload analysis summary."""
        return {
            "cpu_intensive_functions": len(self.cpu_intensive_functions),
            "io_intensive_functions": len(self.io_intensive_functions),
            "total_analyzed_functions": len(self.function_profiles),
        }


class LoadBalancer:
    """Simple load balancer for worker assignment."""
    
    def __init__(self):
        self.worker_loads = defaultdict(int)
    
    def assign_worker(self, available_workers: List[str]) -> str:
        """Assign task to least loaded worker."""
        if not available_workers:
            return available_workers[0] if available_workers else "default"
        
        # Find worker with minimum load
        min_load_worker = min(available_workers, key=lambda w: self.worker_loads[w])
        self.worker_loads[min_load_worker] += 1
        
        return min_load_worker
    
    def release_worker(self, worker_id: str):
        """Release worker after task completion."""
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - 1)