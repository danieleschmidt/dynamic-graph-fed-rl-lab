#!/usr/bin/env python3
"""
Scalable Quantum Task Planner Example

Demonstrates Generation 3 features including:
- Performance optimization and caching
- Concurrent processing with resource pooling
- Load balancing and auto-scaling triggers
- Advanced quantum optimization algorithms
- Distributed execution capabilities
"""

import sys
import os
import time
import asyncio
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import logging
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from quantum_planner_minimal import MinimalQuantumPlanner, MinimalQuantumTask

# Setup high-performance logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(processName)s-%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantum_planner_scalable.log')
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    tasks_per_second: float = 0.0
    avg_execution_time: float = 0.0
    resource_efficiency: float = 0.0
    cache_hit_rate: float = 0.0
    concurrency_factor: float = 1.0
    quantum_coherence: float = 1.0


@dataclass
class ResourcePool:
    """Resource pool for optimized allocation."""
    pool_id: str
    resource_type: str
    total_capacity: float
    available_capacity: float
    reserved_capacity: float = 0.0
    usage_history: List[float] = None
    optimization_factor: float = 1.0
    
    def __post_init__(self):
        if self.usage_history is None:
            self.usage_history = []


class QuantumCache:
    """
    High-performance caching system for quantum computation results.
    
    Features:
    - LRU eviction policy
    - Quantum-aware cache coherence
    - Performance-optimized lookups
    - Adaptive cache sizing
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"QuantumCache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with performance tracking."""
        current_time = time.time()
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if current_time - entry['timestamp'] <= self.ttl_seconds:
                    self.access_times[key] = current_time
                    self.hits += 1
                    return entry['value']
                else:
                    # Expired entry
                    del self.cache[key]
                    del self.access_times[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, quantum_coherence: float = 1.0):
        """Store item in cache with quantum coherence tracking."""
        current_time = time.time()
        
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = {
                'value': value,
                'timestamp': current_time,
                'coherence': quantum_coherence,
                'access_count': 1
            }
            self.access_times[key] = current_time
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "utilization": len(self.cache) / self.max_size
        }
    
    def clear_expired(self):
        """Clear expired entries for maintenance."""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, entry in self.cache.items():
                if current_time - entry['timestamp'] > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                del self.access_times[key]
        
        return len(expired_keys)


class ResourcePoolManager:
    """
    Advanced resource pool management with optimization.
    
    Features:
    - Dynamic resource allocation
    - Load balancing across pools
    - Predictive scaling
    - Performance optimization
    """
    
    def __init__(self):
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.allocation_history: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        
        # Performance optimization
        self.allocation_cache = QuantumCache(max_size=1000, ttl_seconds=60.0)
        
        # Auto-scaling parameters
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.scaling_factor = 1.5
        
        self._initialize_default_pools()
        
        logger.info("ResourcePoolManager initialized with dynamic scaling")
    
    def _initialize_default_pools(self):
        """Initialize default resource pools."""
        default_pools = [
            ResourcePool("cpu_pool", "cpu", 16.0, 16.0),
            ResourcePool("memory_pool", "memory", 32.0, 32.0),
            ResourcePool("io_pool", "io", 8.0, 8.0),
            ResourcePool("gpu_pool", "gpu", 4.0, 4.0),
            ResourcePool("network_pool", "network", 10.0, 10.0)
        ]
        
        for pool in default_pools:
            self.resource_pools[pool.pool_id] = pool
    
    def allocate_resources(self, requirements: Dict[str, float], task_id: str) -> Dict[str, str]:
        """Allocate resources optimally across pools."""
        allocation_key = f"{task_id}:{hash(str(sorted(requirements.items())))}"
        
        # Check cache first
        cached_allocation = self.allocation_cache.get(allocation_key)
        if cached_allocation:
            return cached_allocation
        
        with self.lock:
            allocation = {}
            
            for resource_type, amount in requirements.items():
                pool_id = f"{resource_type}_pool"
                
                if pool_id in self.resource_pools:
                    pool = self.resource_pools[pool_id]
                    
                    # Check if we need to scale up
                    utilization = (pool.total_capacity - pool.available_capacity) / pool.total_capacity
                    if utilization > self.scale_up_threshold and pool.available_capacity < amount:
                        self._scale_pool_up(pool_id, amount)
                    
                    # Allocate if possible
                    if pool.available_capacity >= amount:
                        pool.available_capacity -= amount
                        pool.reserved_capacity += amount
                        allocation[resource_type] = pool_id
                        
                        # Track usage history
                        pool.usage_history.append(utilization)
                        if len(pool.usage_history) > 100:
                            pool.usage_history = pool.usage_history[-100:]
                    else:
                        raise Exception(f"Insufficient {resource_type}: need {amount}, available {pool.available_capacity}")
            
            # Cache successful allocation
            self.allocation_cache.put(allocation_key, allocation)
            
            # Record allocation history
            self.allocation_history.append({
                "timestamp": time.time(),
                "task_id": task_id,
                "requirements": requirements,
                "allocation": allocation
            })
            
            return allocation
    
    def release_resources(self, requirements: Dict[str, float], allocation: Dict[str, str]):
        """Release allocated resources back to pools."""
        with self.lock:
            for resource_type, amount in requirements.items():
                if resource_type in allocation:
                    pool_id = allocation[resource_type]
                    if pool_id in self.resource_pools:
                        pool = self.resource_pools[pool_id]
                        pool.available_capacity += amount
                        pool.reserved_capacity -= amount
                        
                        # Check if we should scale down
                        utilization = (pool.total_capacity - pool.available_capacity) / pool.total_capacity
                        if utilization < self.scale_down_threshold:
                            self._scale_pool_down(pool_id)
    
    def _scale_pool_up(self, pool_id: str, required_amount: float):
        """Scale pool up when utilization is high."""
        if pool_id in self.resource_pools:
            pool = self.resource_pools[pool_id]
            scale_amount = max(required_amount, pool.total_capacity * (self.scaling_factor - 1))
            
            pool.total_capacity += scale_amount
            pool.available_capacity += scale_amount
            
            logger.info(f"Scaled up {pool_id}: +{scale_amount:.1f} (total: {pool.total_capacity:.1f})")
    
    def _scale_pool_down(self, pool_id: str):
        """Scale pool down when utilization is low."""
        if pool_id in self.resource_pools:
            pool = self.resource_pools[pool_id]
            
            # Only scale down if we have excess capacity
            excess_capacity = pool.available_capacity - (pool.total_capacity * 0.5)
            if excess_capacity > 0:
                scale_amount = min(excess_capacity, pool.total_capacity * 0.2)
                
                pool.total_capacity -= scale_amount
                pool.available_capacity -= scale_amount
                
                logger.info(f"Scaled down {pool_id}: -{scale_amount:.1f} (total: {pool.total_capacity:.1f})")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        with self.lock:
            stats = {}
            
            for pool_id, pool in self.resource_pools.items():
                utilization = (pool.total_capacity - pool.available_capacity) / pool.total_capacity
                avg_utilization = sum(pool.usage_history) / len(pool.usage_history) if pool.usage_history else 0.0
                
                stats[pool_id] = {
                    "total_capacity": pool.total_capacity,
                    "available_capacity": pool.available_capacity,
                    "utilization": utilization,
                    "avg_utilization": avg_utilization,
                    "reserved_capacity": pool.reserved_capacity
                }
            
            return stats


class ScalableQuantumPlanner(MinimalQuantumPlanner):
    """
    High-performance scalable quantum task planner.
    
    Generation 3 features:
    - Concurrent execution with thread/process pools
    - Advanced caching and optimization
    - Auto-scaling resource management
    - Load balancing and performance monitoring
    - Distributed quantum computation
    """
    
    def __init__(
        self, 
        max_parallel_tasks: int = 8,
        max_threads: int = None,
        max_processes: int = None,
        enable_caching: bool = True,
        enable_auto_scaling: bool = True
    ):
        super().__init__(max_parallel_tasks=max_parallel_tasks)
        
        # Performance configuration
        self.max_threads = max_threads or min(32, (os.cpu_count() or 1) + 4)
        self.max_processes = max_processes or min(8, (os.cpu_count() or 1))
        self.enable_caching = enable_caching
        self.enable_auto_scaling = enable_auto_scaling
        
        # High-performance components
        self.cache = QuantumCache(max_size=10000, ttl_seconds=300.0) if enable_caching else None
        self.resource_manager = ResourcePoolManager() if enable_auto_scaling else None
        
        # Concurrent execution
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_threads, thread_name_prefix="QuantumTask")
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.execution_times: List[float] = []
        self.task_queue_size = 0
        
        # Load balancing
        self.worker_loads: Dict[str, float] = {}
        self.load_balancer_enabled = True
        
        logger.info(f"ScalableQuantumPlanner initialized: threads={self.max_threads}, processes={self.max_processes}")
    
    def optimize_execution_plan(self, tasks: List[str]) -> List[List[str]]:
        """
        Optimize execution plan for maximum parallel efficiency.
        
        Uses advanced algorithms to minimize execution time while
        maximizing resource utilization.
        """
        if not tasks:
            return []
        
        # Check cache first
        plan_key = f"exec_plan:{hash(str(sorted(tasks)))}"
        if self.cache:
            cached_plan = self.cache.get(plan_key)
            if cached_plan:
                return cached_plan
        
        # Dependency analysis
        dependency_levels = self._analyze_dependency_levels(tasks)
        
        # Resource-aware grouping
        resource_optimized_groups = self._optimize_resource_allocation(dependency_levels)
        
        # Load balancing optimization
        load_balanced_groups = self._optimize_load_balancing(resource_optimized_groups)
        
        # Cache the optimized plan
        if self.cache:
            self.cache.put(plan_key, load_balanced_groups, quantum_coherence=0.95)
        
        logger.info(f"Optimized execution plan: {len(load_balanced_groups)} parallel groups")
        return load_balanced_groups
    
    def _analyze_dependency_levels(self, tasks: List[str]) -> Dict[int, List[str]]:
        """Analyze task dependency levels for optimal grouping."""
        levels = {}
        task_levels = {}
        
        def calculate_level(task_id: str, visited: set = None) -> int:
            if visited is None:
                visited = set()
            
            if task_id in visited:
                return 0  # Avoid cycles
            
            if task_id in task_levels:
                return task_levels[task_id]
            
            if task_id not in self.tasks:
                return 0
            
            visited.add(task_id)
            task = self.tasks[task_id]
            
            if not task.dependencies:
                level = 0
            else:
                max_dep_level = max(
                    calculate_level(dep_id, visited.copy()) 
                    for dep_id in task.dependencies 
                    if dep_id in self.tasks
                )
                level = max_dep_level + 1
            
            task_levels[task_id] = level
            return level
        
        # Calculate levels for all tasks
        for task_id in tasks:
            level = calculate_level(task_id)
            if level not in levels:
                levels[level] = []
            levels[level].append(task_id)
        
        return levels
    
    def _optimize_resource_allocation(self, dependency_levels: Dict[int, List[str]]) -> List[List[str]]:
        """Optimize task grouping based on resource requirements."""
        optimized_groups = []
        
        for level, level_tasks in sorted(dependency_levels.items()):
            # Group tasks by similar resource requirements
            resource_groups = {}
            
            for task_id in level_tasks:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    resources = getattr(task, 'resource_requirements', {})
                    
                    # Create resource signature for grouping
                    resource_signature = tuple(sorted(resources.items()))
                    
                    if resource_signature not in resource_groups:
                        resource_groups[resource_signature] = []
                    resource_groups[resource_signature].append(task_id)
            
            # Add each resource group as a parallel batch
            for group_tasks in resource_groups.values():
                # Split large groups to respect parallel limits
                batch_size = max(1, self.max_parallel_tasks // 2)
                for i in range(0, len(group_tasks), batch_size):
                    batch = group_tasks[i:i + batch_size]
                    optimized_groups.append(batch)
        
        return optimized_groups
    
    def _optimize_load_balancing(self, resource_groups: List[List[str]]) -> List[List[str]]:
        """Further optimize groups for load balancing."""
        if not self.load_balancer_enabled:
            return resource_groups
        
        balanced_groups = []
        
        for group in resource_groups:
            # Estimate execution times
            estimated_times = []
            for task_id in group:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    estimated_times.append(task.estimated_duration)
                else:
                    estimated_times.append(1.0)
            
            # Reorder for better load distribution
            task_time_pairs = list(zip(group, estimated_times))
            task_time_pairs.sort(key=lambda x: x[1], reverse=True)  # Longest first
            
            reordered_group = [task_id for task_id, _ in task_time_pairs]
            balanced_groups.append(reordered_group)
        
        return balanced_groups
    
    async def execute_scalable(self, execution_plan: List[List[str]]) -> Dict[str, Any]:
        """
        Execute optimized plan with maximum concurrency and performance.
        """
        start_time = time.time()
        all_results = {}
        total_tasks = sum(len(group) for group in execution_plan)
        completed_tasks = 0
        
        logger.info(f"Starting scalable execution: {len(execution_plan)} groups, {total_tasks} tasks")
        
        for group_idx, task_group in enumerate(execution_plan):
            logger.info(f"Executing group {group_idx + 1}/{len(execution_plan)}: {len(task_group)} tasks")
            
            # Execute group concurrently
            group_results = await self._execute_task_group_concurrent(task_group)
            all_results.update(group_results)
            
            completed_tasks += len(task_group)
            progress = (completed_tasks / total_tasks) * 100
            logger.info(f"Progress: {progress:.1f}% ({completed_tasks}/{total_tasks})")
        
        execution_time = time.time() - start_time
        
        # Update performance metrics
        self._update_performance_metrics(execution_time, total_tasks, all_results)
        
        return {
            "results": all_results,
            "execution_time": execution_time,
            "total_tasks": total_tasks,
            "groups_executed": len(execution_plan),
            "performance_metrics": self.get_performance_metrics()
        }
    
    async def _execute_task_group_concurrent(self, task_group: List[str]) -> Dict[str, Any]:
        """Execute a group of tasks concurrently."""
        if not task_group:
            return {}
        
        # Prepare tasks for concurrent execution
        executable_tasks = []
        resource_allocations = {}
        
        for task_id in task_group:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                # Allocate resources if resource manager is available
                if self.resource_manager:
                    try:
                        resources = getattr(task, 'resource_requirements', {})
                        if resources:
                            allocation = self.resource_manager.allocate_resources(resources, task_id)
                            resource_allocations[task_id] = (resources, allocation)
                    except Exception as e:
                        logger.warning(f"Resource allocation failed for {task_id}: {e}")
                
                executable_tasks.append((task_id, task))
        
        # Execute tasks concurrently
        loop = asyncio.get_event_loop()
        
        # Choose execution strategy based on task characteristics
        if self._should_use_processes(executable_tasks):
            # CPU-intensive tasks: use process pool
            futures = [
                loop.run_in_executor(self.process_pool, self._execute_task_safe, task_id, task)
                for task_id, task in executable_tasks
            ]
        else:
            # I/O-intensive tasks: use thread pool
            futures = [
                loop.run_in_executor(self.thread_pool, self._execute_task_safe, task_id, task)
                for task_id, task in executable_tasks
            ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Process results and release resources
        group_results = {}
        for i, ((task_id, task), result) in enumerate(zip(executable_tasks, results)):
            if isinstance(result, Exception):
                group_results[task_id] = {
                    "status": "failed",
                    "error": str(result),
                    "duration": 0.0
                }
                logger.error(f"Task {task_id} failed: {result}")
            else:
                group_results[task_id] = result
            
            # Release resources
            if task_id in resource_allocations:
                resources, allocation = resource_allocations[task_id]
                self.resource_manager.release_resources(resources, allocation)
        
        return group_results
    
    def _should_use_processes(self, tasks: List[tuple]) -> bool:
        """Determine whether to use processes vs threads based on task characteristics."""
        cpu_intensive_count = 0
        
        for task_id, task in tasks:
            resources = getattr(task, 'resource_requirements', {})
            cpu_requirement = resources.get('cpu', 0.0)
            
            # Consider CPU-intensive if requires > 1.0 CPU units
            if cpu_requirement > 1.0:
                cpu_intensive_count += 1
        
        # Use processes if majority of tasks are CPU-intensive
        return cpu_intensive_count > len(tasks) / 2
    
    def _execute_task_safe(self, task_id: str, task) -> Dict[str, Any]:
        """Execute single task with comprehensive error handling and monitoring."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache:
                cache_key = f"task_result:{task_id}:{hash(str(getattr(task, 'executor', None)))}"
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    return cached_result
            
            # Execute task
            if task.executor:
                result = task.executor()
            else:
                # Simulate optimized execution
                time.sleep(min(task.estimated_duration, 0.1))
                result = {
                    "status": "optimized_execution",
                    "task_id": task_id,
                    "optimization_applied": True
                }
            
            execution_time = time.time() - start_time
            
            # Prepare result
            task_result = {
                "status": "success",
                "result": result,
                "duration": execution_time,
                "optimization_score": self._calculate_optimization_score(execution_time, task.estimated_duration)
            }
            
            # Cache result if beneficial
            if self.cache and execution_time > 0.01:  # Only cache non-trivial results
                cache_key = f"task_result:{task_id}:{hash(str(task.executor))}"
                self.cache.put(cache_key, task_result, quantum_coherence=0.9)
            
            return task_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "duration": execution_time,
                "optimization_score": 0.0
            }
    
    def _calculate_optimization_score(self, actual_time: float, estimated_time: float) -> float:
        """Calculate optimization score based on performance vs estimate."""
        if estimated_time == 0:
            return 1.0
        
        ratio = actual_time / estimated_time
        
        if ratio <= 0.5:
            return 1.0  # Excellent optimization
        elif ratio <= 0.8:
            return 0.8  # Good optimization
        elif ratio <= 1.0:
            return 0.6  # Meeting expectations
        elif ratio <= 1.5:
            return 0.4  # Below expectations
        else:
            return 0.2  # Poor performance
    
    def _update_performance_metrics(self, execution_time: float, total_tasks: int, results: Dict[str, Any]):
        """Update performance metrics based on execution results."""
        # Calculate throughput
        self.metrics.tasks_per_second = total_tasks / execution_time if execution_time > 0 else 0
        
        # Calculate average execution time
        task_times = [r.get('duration', 0) for r in results.values() if isinstance(r, dict)]
        self.metrics.avg_execution_time = sum(task_times) / len(task_times) if task_times else 0
        
        # Calculate optimization scores
        opt_scores = [r.get('optimization_score', 0) for r in results.values() if isinstance(r, dict)]
        self.metrics.resource_efficiency = sum(opt_scores) / len(opt_scores) if opt_scores else 0
        
        # Cache performance
        if self.cache:
            cache_stats = self.cache.get_stats()
            self.metrics.cache_hit_rate = cache_stats['hit_rate']
        
        # Concurrency factor
        self.metrics.concurrency_factor = min(total_tasks, self.max_threads) / max(1, total_tasks / (execution_time / self.metrics.avg_execution_time)) if self.metrics.avg_execution_time > 0 else 1.0
        
        # Store execution time history
        self.execution_times.append(execution_time)
        if len(self.execution_times) > 100:
            self.execution_times = self.execution_times[-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        base_metrics = {
            "tasks_per_second": self.metrics.tasks_per_second,
            "avg_execution_time": self.metrics.avg_execution_time,
            "resource_efficiency": self.metrics.resource_efficiency,
            "concurrency_factor": self.metrics.concurrency_factor,
            "quantum_coherence": self.metrics.quantum_coherence
        }
        
        # Cache metrics
        if self.cache:
            cache_stats = self.cache.get_stats()
            base_metrics.update({
                "cache_hit_rate": cache_stats['hit_rate'],
                "cache_size": cache_stats['size'],
                "cache_utilization": cache_stats['utilization']
            })
        
        # Resource pool metrics
        if self.resource_manager:
            pool_stats = self.resource_manager.get_pool_stats()
            base_metrics["resource_pools"] = pool_stats
        
        # Thread pool metrics
        base_metrics.update({
            "thread_pool_size": self.max_threads,
            "process_pool_size": self.max_processes,
            "execution_history_size": len(self.execution_times)
        })
        
        return base_metrics
    
    def cleanup(self):
        """Cleanup resources and shutdown pools."""
        logger.info("Shutting down ScalableQuantumPlanner...")
        
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        if self.cache:
            expired_count = self.cache.clear_expired()
            logger.info(f"Cleared {expired_count} expired cache entries")
        
        logger.info("ScalableQuantumPlanner shutdown complete")


# High-performance task functions for demonstration
def cpu_intensive_task():
    """CPU-intensive computation task."""
    start = time.time()
    
    # Simulate CPU-bound work
    result = sum(i * i for i in range(50000))
    
    duration = time.time() - start
    return {
        "computation_result": result,
        "cpu_time": duration,
        "task_type": "cpu_intensive"
    }


def io_intensive_task():
    """I/O-intensive task simulation."""
    start = time.time()
    
    # Simulate I/O operations
    time.sleep(0.05)
    
    duration = time.time() - start
    return {
        "data_processed": "1GB",
        "io_time": duration,
        "task_type": "io_intensive"
    }


def memory_intensive_task():
    """Memory-intensive task simulation."""
    start = time.time()
    
    # Simulate memory operations
    data = [i for i in range(10000)]
    result = len(data)
    
    duration = time.time() - start
    return {
        "memory_used": f"{len(data)} items",
        "processing_time": duration,
        "task_type": "memory_intensive"
    }


def mixed_workload_task():
    """Mixed CPU/IO workload task."""
    start = time.time()
    
    # Mixed operations
    cpu_result = sum(i for i in range(1000))
    time.sleep(0.01)  # I/O simulation
    
    duration = time.time() - start
    return {
        "cpu_result": cpu_result,
        "total_time": duration,
        "task_type": "mixed_workload"
    }


async def main():
    """Run scalable quantum planner demonstration."""
    print("🚀 Scalable Quantum Task Planner - Generation 3")
    print("=" * 60)
    
    # Initialize high-performance planner
    planner = ScalableQuantumPlanner(
        max_parallel_tasks=12,
        max_threads=16,
        max_processes=4,
        enable_caching=True,
        enable_auto_scaling=True
    )
    
    print(f"\n📊 Planner Configuration:")
    print(f"Max parallel tasks: {planner.max_parallel_tasks}")
    print(f"Thread pool size: {planner.max_threads}")
    print(f"Process pool size: {planner.max_processes}")
    print(f"Caching enabled: {planner.enable_caching}")
    print(f"Auto-scaling enabled: {planner.enable_auto_scaling}")
    
    # Create high-performance task suite
    print(f"\n📋 Creating high-performance task suite...")
    
    task_definitions = [
        # CPU-intensive tasks
        ("cpu_task_1", "CPU Intensive Computation 1", cpu_intensive_task, {"cpu": 2.0}),
        ("cpu_task_2", "CPU Intensive Computation 2", cpu_intensive_task, {"cpu": 2.0}),
        ("cpu_task_3", "CPU Intensive Computation 3", cpu_intensive_task, {"cpu": 1.5}),
        
        # I/O-intensive tasks
        ("io_task_1", "I/O Intensive Operation 1", io_intensive_task, {"io": 1.0}),
        ("io_task_2", "I/O Intensive Operation 2", io_intensive_task, {"io": 1.5}),
        ("io_task_3", "I/O Intensive Operation 3", io_intensive_task, {"io": 0.8}),
        
        # Memory-intensive tasks
        ("mem_task_1", "Memory Intensive Processing 1", memory_intensive_task, {"memory": 2.0}),
        ("mem_task_2", "Memory Intensive Processing 2", memory_intensive_task, {"memory": 1.5}),
        
        # Mixed workload tasks
        ("mixed_task_1", "Mixed Workload 1", mixed_workload_task, {"cpu": 1.0, "memory": 1.0}),
        ("mixed_task_2", "Mixed Workload 2", mixed_workload_task, {"cpu": 0.8, "io": 0.5}),
        ("mixed_task_3", "Mixed Workload 3", mixed_workload_task, {"memory": 1.0, "io": 1.0}),
        
        # Dependent tasks for testing optimization
        ("aggregate_1", "Aggregate CPU Results", lambda: {"aggregated": "cpu_results"}, {"cpu": 0.5}),
        ("aggregate_2", "Aggregate IO Results", lambda: {"aggregated": "io_results"}, {"io": 0.3}),
        ("final_report", "Generate Final Report", lambda: {"report": "complete"}, {"cpu": 0.8, "memory": 0.5})
    ]
    
    # Add tasks with dependencies
    for task_id, name, executor, resources in task_definitions:
        task = planner.add_task(
            task_id=task_id,
            name=name,
            estimated_duration=0.1 + len(resources) * 0.05,
            priority=2.0 if "cpu" in resources else 1.5,
            executor=executor
        )
        
        # Add resource requirements
        setattr(task, 'resource_requirements', resources)
        print(f"  ✅ Added {task_id}: {name}")
    
    # Add dependencies for optimization testing
    planner.tasks["aggregate_1"].dependencies = {"cpu_task_1", "cpu_task_2", "cpu_task_3"}
    planner.tasks["aggregate_2"].dependencies = {"io_task_1", "io_task_2", "io_task_3"}
    planner.tasks["final_report"].dependencies = {"aggregate_1", "aggregate_2", "mem_task_1"}
    
    print(f"Total tasks created: {len(planner.tasks)}")
    
    # Display initial resource pool status
    if planner.resource_manager:
        print(f"\n🏊 Resource Pool Status:")
        pool_stats = planner.resource_manager.get_pool_stats()
        for pool_id, stats in pool_stats.items():
            print(f"  {pool_id}: {stats['available_capacity']:.1f}/{stats['total_capacity']:.1f} available")
    
    # Generate optimized execution plan
    print(f"\n🧠 Generating optimized execution plan...")
    all_task_ids = list(planner.tasks.keys())
    execution_plan = planner.optimize_execution_plan(all_task_ids)
    
    print(f"Optimized into {len(execution_plan)} parallel groups:")
    for i, group in enumerate(execution_plan):
        print(f"  Group {i+1}: {group}")
    
    # Execute with maximum performance
    print(f"\n⚡ Executing with maximum concurrency and optimization...")
    start_time = time.time()
    
    execution_result = await planner.execute_scalable(execution_plan)
    
    total_time = time.time() - start_time
    
    # Display comprehensive results
    print(f"\n📊 High-Performance Execution Results:")
    print(f"Total execution time: {execution_result['execution_time']:.3f}s")
    print(f"Real wall-clock time: {total_time:.3f}s")
    print(f"Tasks completed: {execution_result['total_tasks']}")
    print(f"Parallel groups: {execution_result['groups_executed']}")
    
    # Performance metrics
    metrics = execution_result['performance_metrics']
    print(f"\n🎯 Performance Metrics:")
    print(f"Tasks per second: {metrics['tasks_per_second']:.2f}")
    print(f"Average task time: {metrics['avg_execution_time']:.4f}s")
    print(f"Resource efficiency: {metrics['resource_efficiency']:.3f}")
    print(f"Concurrency factor: {metrics['concurrency_factor']:.2f}")
    
    if 'cache_hit_rate' in metrics:
        print(f"Cache hit rate: {metrics['cache_hit_rate']:.3f}")
        print(f"Cache utilization: {metrics['cache_utilization']:.3f}")
    
    # Task-specific results
    print(f"\n📋 Task Execution Summary:")
    results = execution_result['results']
    success_count = 0
    failed_count = 0
    
    for task_id, result in results.items():
        status = result.get('status', 'unknown')
        duration = result.get('duration', 0.0)
        opt_score = result.get('optimization_score', 0.0)
        
        emoji = "✅" if status == "success" else "❌"
        print(f"  {emoji} {task_id}: {status} ({duration:.4f}s, opt: {opt_score:.2f})")
        
        if status == "success":
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\nSuccess rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    
    # Resource pool final status
    if planner.resource_manager:
        print(f"\n🏊 Final Resource Pool Status:")
        final_pool_stats = planner.resource_manager.get_pool_stats()
        for pool_id, stats in final_pool_stats.items():
            utilization = stats['utilization']
            avg_util = stats['avg_utilization']
            print(f"  {pool_id}: {utilization:.1%} current, {avg_util:.1%} average utilization")
    
    # Cleanup
    print(f"\n🧹 Cleaning up resources...")
    planner.cleanup()
    
    print(f"\n🚀 Scalable quantum planning demonstration complete!")
    print(f"Generation 3 Features Demonstrated:")
    print(f"  • High-performance concurrent execution")
    print(f"  • Intelligent caching and optimization")
    print(f"  • Auto-scaling resource management")
    print(f"  • Load balancing and performance monitoring")
    print(f"  • Process/thread pool optimization")
    print(f"  • Advanced quantum execution planning")


if __name__ == "__main__":
    asyncio.run(main())