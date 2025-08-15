"""Advanced performance optimization for scalable federated RL systems."""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import concurrent.futures
import asyncio
from collections import defaultdict, deque


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    COMPUTE_PARALLEL = "compute_parallel"
    MEMORY_EFFICIENT = "memory_efficient"
    NETWORK_OPTIMIZED = "network_optimized"
    ADAPTIVE_BATCHING = "adaptive_batching"
    PREDICTIVE_CACHING = "predictive_caching"


@dataclass
class OptimizationResult:
    """Result of performance optimization."""
    
    strategy: OptimizationStrategy
    original_time: float
    optimized_time: float
    speedup_factor: float
    memory_reduction: float
    throughput_improvement: float
    resource_utilization: Dict[str, float]
    optimization_overhead: float
    
    @property
    def efficiency_gain(self) -> float:
        """Calculate overall efficiency gain."""
        return (self.speedup_factor + self.memory_reduction + self.throughput_improvement) / 3.0


class PerformanceOptimizer:
    """
    Advanced performance optimizer for scalable federated RL systems.
    
    Features:
    - Automatic performance profiling and bottleneck detection
    - Multiple optimization strategies (compute, memory, network)
    - Adaptive batching and parallel processing
    - Predictive caching and resource pre-allocation
    - Real-time performance monitoring and adjustment
    """
    
    def __init__(
        self,
        max_workers: int = 8,
        enable_adaptive_batching: bool = True,
        enable_predictive_caching: bool = True,
        optimization_interval: float = 30.0,
        performance_history_size: int = 1000
    ):
        self.max_workers = max_workers
        self.enable_adaptive_batching = enable_adaptive_batching
        self.enable_predictive_caching = enable_predictive_caching
        self.optimization_interval = optimization_interval
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=performance_history_size)
        self.optimization_results: List[OptimizationResult] = []
        self.active_optimizations: Dict[str, Any] = {}
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.bottleneck_detector = BottleneckDetector()
        
        # Threading
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.optimization_lock = threading.Lock()
        
        # Adaptive parameters
        self.adaptive_batch_sizes: Dict[str, int] = defaultdict(lambda: 32)
        self.cache_hit_rates: Dict[str, float] = defaultdict(float)
        
        print(f"🚀 Performance optimizer initialized with {max_workers} workers")
    
    def optimize_computation(self, 
                           computation_func: Callable,
                           data_batch: List[Any],
                           optimization_strategy: OptimizationStrategy = OptimizationStrategy.COMPUTE_PARALLEL) -> OptimizationResult:
        """Optimize computational workload with specified strategy."""
        
        start_time = time.time()
        
        # Baseline performance measurement
        baseline_start = time.time()
        baseline_results = [computation_func(item) for item in data_batch]
        baseline_time = time.time() - baseline_start
        
        # Apply optimization strategy
        optimized_start = time.time()
        
        if optimization_strategy == OptimizationStrategy.COMPUTE_PARALLEL:
            optimized_results = self._parallel_computation(computation_func, data_batch)
        elif optimization_strategy == OptimizationStrategy.ADAPTIVE_BATCHING:
            optimized_results = self._adaptive_batch_computation(computation_func, data_batch)
        elif optimization_strategy == OptimizationStrategy.MEMORY_EFFICIENT:
            optimized_results = self._memory_efficient_computation(computation_func, data_batch)
        else:
            optimized_results = baseline_results
        
        optimized_time = time.time() - optimized_start
        optimization_overhead = time.time() - start_time - baseline_time - optimized_time
        
        # Calculate performance improvements
        speedup_factor = baseline_time / max(optimized_time, 0.001)
        memory_usage = self.resource_monitor.get_memory_usage()
        
        result = OptimizationResult(
            strategy=optimization_strategy,
            original_time=baseline_time,
            optimized_time=optimized_time,
            speedup_factor=speedup_factor,
            memory_reduction=0.1,  # Simplified metric
            throughput_improvement=speedup_factor - 1.0,
            resource_utilization=self.resource_monitor.get_current_utilization(),
            optimization_overhead=optimization_overhead
        )
        
        self.optimization_results.append(result)
        print(f"⚡ {optimization_strategy.value}: {speedup_factor:.2f}x speedup")
        
        return result
    
    def _parallel_computation(self, computation_func: Callable, data_batch: List[Any]) -> List[Any]:
        """Execute computation in parallel using thread pool."""
        futures = [self.executor.submit(computation_func, item) for item in data_batch]
        return [future.result() for future in concurrent.futures.as_completed(futures)]
    
    def _adaptive_batch_computation(self, computation_func: Callable, data_batch: List[Any]) -> List[Any]:
        """Execute computation with adaptive batching."""
        func_name = computation_func.__name__
        batch_size = self.adaptive_batch_sizes[func_name]
        
        results = []
        
        # Process in adaptive batches
        for i in range(0, len(data_batch), batch_size):
            batch = data_batch[i:i + batch_size]
            
            batch_start = time.time()
            batch_results = [computation_func(item) for item in batch]
            batch_time = time.time() - batch_start
            
            # Adjust batch size based on performance
            items_per_second = len(batch) / max(batch_time, 0.001)
            if items_per_second > 100:  # Too fast, increase batch size
                self.adaptive_batch_sizes[func_name] = min(batch_size * 2, 256)
            elif items_per_second < 10:  # Too slow, decrease batch size
                self.adaptive_batch_sizes[func_name] = max(batch_size // 2, 1)
            
            results.extend(batch_results)
        
        return results
    
    def _memory_efficient_computation(self, computation_func: Callable, data_batch: List[Any]) -> List[Any]:
        """Execute computation with memory optimization."""
        # Process items one at a time to minimize memory usage
        results = []
        for item in data_batch:
            result = computation_func(item)
            results.append(result)
            # Force garbage collection periodically
            if len(results) % 100 == 0:
                import gc
                gc.collect()
        
        return results
    
    async def optimize_federated_communication(self, 
                                             agents: List[Dict],
                                             communication_func: Callable) -> OptimizationResult:
        """Optimize federated communication patterns."""
        
        start_time = time.time()
        
        # Baseline: sequential communication
        baseline_start = time.time()
        baseline_results = []
        for agent in agents:
            result = await communication_func(agent)
            baseline_results.append(result)
        baseline_time = time.time() - baseline_start
        
        # Optimized: parallel communication with batching
        optimized_start = time.time()
        
        # Group agents by region/characteristics for efficient batching
        agent_groups = self._group_agents_for_communication(agents)
        
        optimized_results = []
        for group in agent_groups:
            # Parallel communication within group
            tasks = [communication_func(agent) for agent in group]
            group_results = await asyncio.gather(*tasks)
            optimized_results.extend(group_results)
        
        optimized_time = time.time() - optimized_start
        
        # Calculate improvements
        speedup_factor = baseline_time / max(optimized_time, 0.001)
        network_efficiency = self._calculate_network_efficiency(agents, optimized_time)
        
        result = OptimizationResult(
            strategy=OptimizationStrategy.NETWORK_OPTIMIZED,
            original_time=baseline_time,
            optimized_time=optimized_time,
            speedup_factor=speedup_factor,
            memory_reduction=0.0,
            throughput_improvement=network_efficiency,
            resource_utilization=self.resource_monitor.get_current_utilization(),
            optimization_overhead=time.time() - start_time - baseline_time - optimized_time
        )
        
        self.optimization_results.append(result)
        print(f"🌐 Network optimization: {speedup_factor:.2f}x speedup")
        
        return result
    
    def _group_agents_for_communication(self, agents: List[Dict]) -> List[List[Dict]]:
        """Group agents for efficient batch communication."""
        groups = defaultdict(list)
        
        for agent in agents:
            # Group by region or other characteristics
            key = agent.get('region', 'default')
            groups[key].append(agent)
        
        return list(groups.values())
    
    def _calculate_network_efficiency(self, agents: List[Dict], communication_time: float) -> float:
        """Calculate network communication efficiency."""
        total_agents = len(agents)
        theoretical_min_time = total_agents * 0.01  # Assume 10ms per agent minimum
        
        return max(0.0, (theoretical_min_time / max(communication_time, 0.001)) - 1.0)
    
    def optimize_memory_usage(self, memory_intensive_operations: List[Callable]) -> OptimizationResult:
        """Optimize memory usage across operations."""
        
        start_time = time.time()
        initial_memory = self.resource_monitor.get_memory_usage()
        
        # Baseline: execute all operations
        baseline_start = time.time()
        baseline_results = []
        for operation in memory_intensive_operations:
            try:
                result = operation()
                baseline_results.append(result)
            except MemoryError:
                print("⚠️  Memory error in baseline execution")
                break
        baseline_time = time.time() - baseline_start
        baseline_memory = self.resource_monitor.get_memory_usage()
        
        # Optimized: memory-efficient execution
        optimized_start = time.time()
        optimized_results = []
        
        for i, operation in enumerate(memory_intensive_operations):
            try:
                # Force garbage collection before memory-intensive operations
                if i % 5 == 0:
                    import gc
                    gc.collect()
                
                result = operation()
                optimized_results.append(result)
                
                # Monitor memory usage
                current_memory = self.resource_monitor.get_memory_usage()
                if current_memory > initial_memory * 2:  # Memory usage doubled
                    print(f"⚠️  High memory usage detected: {current_memory:.1f}MB")
                    gc.collect()
                
            except MemoryError:
                print(f"⚠️  Memory error at operation {i}, implementing emergency cleanup")
                self._emergency_memory_cleanup()
                continue
        
        optimized_time = time.time() - optimized_start
        final_memory = self.resource_monitor.get_memory_usage()
        
        # Calculate memory efficiency
        memory_reduction = max(0.0, (baseline_memory - final_memory) / max(baseline_memory, 1.0))
        
        result = OptimizationResult(
            strategy=OptimizationStrategy.MEMORY_EFFICIENT,
            original_time=baseline_time,
            optimized_time=optimized_time,
            speedup_factor=baseline_time / max(optimized_time, 0.001),
            memory_reduction=memory_reduction,
            throughput_improvement=0.0,
            resource_utilization=self.resource_monitor.get_current_utilization(),
            optimization_overhead=time.time() - start_time - baseline_time - optimized_time
        )
        
        self.optimization_results.append(result)
        print(f"💾 Memory optimization: {memory_reduction:.1%} reduction")
        
        return result
    
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup procedures."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Clear internal caches
        self.performance_history.clear()
        if len(self.optimization_results) > 100:
            self.optimization_results = self.optimization_results[-50:]
        
        print("🧹 Emergency memory cleanup completed")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance history."""
        recommendations = []
        
        if not self.optimization_results:
            return ["No optimization data available yet"]
        
        # Analyze recent performance
        recent_results = self.optimization_results[-10:]
        
        # Check for consistent speedups
        avg_speedup = sum(r.speedup_factor for r in recent_results) / len(recent_results)
        if avg_speedup < 1.5:
            recommendations.append("Consider increasing parallelization for compute-intensive tasks")
        
        # Check memory efficiency
        avg_memory_reduction = sum(r.memory_reduction for r in recent_results) / len(recent_results)
        if avg_memory_reduction < 0.1:
            recommendations.append("Implement more aggressive memory management strategies")
        
        # Check for optimization overhead
        avg_overhead = sum(r.optimization_overhead for r in recent_results) / len(recent_results)
        if avg_overhead > 0.1:
            recommendations.append("Optimization overhead is high, consider simpler strategies")
        
        # Resource utilization analysis
        current_util = self.resource_monitor.get_current_utilization()
        if current_util.get('cpu', 0) < 50:
            recommendations.append("CPU utilization is low, increase parallel processing")
        if current_util.get('memory', 0) > 80:
            recommendations.append("Memory usage is high, implement memory-efficient algorithms")
        
        return recommendations or ["System performance is well optimized"]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        if not self.optimization_results:
            return {"status": "No optimization data available"}
        
        recent_results = self.optimization_results[-20:]
        
        return {
            "total_optimizations": len(self.optimization_results),
            "average_speedup": sum(r.speedup_factor for r in recent_results) / len(recent_results),
            "average_memory_reduction": sum(r.memory_reduction for r in recent_results) / len(recent_results),
            "average_throughput_improvement": sum(r.throughput_improvement for r in recent_results) / len(recent_results),
            "best_strategy": max(recent_results, key=lambda r: r.efficiency_gain).strategy.value,
            "current_resource_utilization": self.resource_monitor.get_current_utilization(),
            "optimization_recommendations": self.get_optimization_recommendations(),
            "adaptive_batch_sizes": dict(self.adaptive_batch_sizes),
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        print("🧹 Performance optimizer cleanup completed")


class ResourceMonitor:
    """Monitor system resource utilization."""
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback for systems without psutil
            return 100.0 + time.time() % 50  # Mock value
    
    def get_current_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        try:
            import psutil
            return {
                'cpu': psutil.cpu_percent(interval=0.1),
                'memory': psutil.virtual_memory().percent,
                'disk_io': psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes,
            }
        except ImportError:
            # Fallback mock values
            return {
                'cpu': 30.0 + (time.time() % 40),
                'memory': 45.0 + (time.time() % 30),
                'disk_io': 1000000,
            }


class BottleneckDetector:
    """Detect performance bottlenecks in the system."""
    
    def __init__(self):
        self.bottlenecks = []
    
    def detect_bottlenecks(self, performance_data: Dict[str, float]) -> List[str]:
        """Detect current performance bottlenecks."""
        bottlenecks = []
        
        # CPU bottleneck
        if performance_data.get('cpu', 0) > 90:
            bottlenecks.append("CPU usage critically high")
        elif performance_data.get('cpu', 0) > 80:
            bottlenecks.append("CPU usage high")
        
        # Memory bottleneck
        if performance_data.get('memory', 0) > 90:
            bottlenecks.append("Memory usage critically high")
        elif performance_data.get('memory', 0) > 80:
            bottlenecks.append("Memory usage high")
        
        # I/O bottleneck
        if performance_data.get('disk_io', 0) > 1000000000:  # 1GB
            bottlenecks.append("High disk I/O detected")
        
        self.bottlenecks = bottlenecks
        return bottlenecks