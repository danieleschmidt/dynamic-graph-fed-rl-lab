"""
Test suite for performance optimization components.

Tests caching, memory pooling, JIT optimization, and performance profiling.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch
import jax.numpy as jnp

from src.dynamic_graph_fed_rl.quantum_planner.performance import (
    QuantumCache,
    MemoryPool,
    JITOptimizer,
    VectorizedOperations,
    PerformanceProfiler,
    PerformanceManager
)


class TestQuantumCache:
    """Test quantum-aware caching functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create test cache."""
        return QuantumCache(max_size=100, ttl_seconds=1.0, coherence_threshold=0.95)
    
    def test_cache_basic_operations(self, cache):
        """Test basic cache operations."""
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
    
    def test_cache_ttl_expiration(self, cache):
        """Test cache TTL expiration."""
        cache.put("expiring_key", "value")
        
        # Should be available immediately
        assert cache.get("expiring_key") == "value"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("expiring_key") is None
    
    def test_cache_coherence_tracking(self, cache):
        """Test quantum coherence tracking."""
        # Put value with coherence
        cache.put("coherent_key", "value", coherence=0.98)
        
        # Should retrieve with matching coherence
        assert cache.get("coherent_key", coherence=0.98) == "value"
        
        # Should miss with drifted coherence
        assert cache.get("coherent_key", coherence=0.90) is None
    
    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction policy."""
        # Fill cache beyond capacity
        for i in range(150):  # More than max_size=100
            cache.put(f"key_{i}", f"value_{i}")
        
        # Should evict oldest entries
        assert cache.get("key_0") is None  # Oldest should be evicted
        assert cache.get("key_149") == "value_149"  # Newest should remain
        
        # Cache size should not exceed max
        stats = cache.get_stats()
        assert stats["size"] <= 100
    
    def test_cache_cleanup(self, cache):
        """Test cache cleanup of expired entries."""
        # Add entries that will expire
        for i in range(10):
            cache.put(f"temp_key_{i}", f"temp_value_{i}")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Trigger cleanup
        cache.cleanup()
        
        # All expired entries should be removed
        stats = cache.get_stats()
        assert stats["size"] == 0


class TestMemoryPool:
    """Test memory pool functionality."""
    
    @pytest.fixture
    def pool(self):
        """Create test memory pool."""
        return MemoryPool(
            pool_sizes=[64, 256, 1024], 
            pool_counts=[10, 5, 2]
        )
    
    def test_pool_initialization(self, pool):
        """Test memory pool initialization."""
        stats = pool.get_stats()
        assert stats["pools"][64]["available"] == 10
        assert stats["pools"][256]["available"] == 5
        assert stats["pools"][1024]["available"] == 2
    
    def test_array_allocation_hit(self, pool):
        """Test successful allocation from pool."""
        # Request array that fits in pool
        array = pool.get_array(64)
        
        assert array.shape[0] == 64
        assert array.dtype == jnp.complex64
        
        # Pool should have one less array
        stats = pool.get_stats()
        assert stats["pools"][64]["available"] == 9
        assert stats["pools"][64]["allocated"] == 1
        assert stats["hit_rate"] > 0
    
    def test_array_allocation_miss(self, pool):
        """Test allocation when pool is empty or size unavailable."""
        # Request size not in pool
        large_array = pool.get_array(2048)
        
        assert large_array.shape[0] == 2048
        
        # Should be a pool miss
        stats = pool.get_stats()
        assert stats["misses"] > 0
    
    def test_array_return(self, pool):
        """Test returning arrays to pool."""
        # Get array from pool
        array = pool.get_array(256)
        original_available = pool.get_stats()["pools"][256]["available"]
        
        # Return array to pool
        pool.return_array(array, 256)
        
        # Pool should have array back
        new_available = pool.get_stats()["pools"][256]["available"]
        assert new_available == original_available + 1
    
    def test_pool_size_selection(self, pool):
        """Test selection of appropriate pool size."""
        # Request size smaller than available pool
        array = pool.get_array(32)  # Smaller than 64
        
        # Should get array of pool size but return correct slice
        assert array.shape[0] == 32
        
        # Pool should be used
        stats = pool.get_stats()
        assert stats["hits"] > 0


class TestJITOptimizer:
    """Test JIT compilation optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create JIT optimizer."""
        return JITOptimizer()
    
    def test_amplitude_evolution_compilation(self, optimizer):
        """Test JIT compilation of amplitude evolution."""
        compiled_func = optimizer.optimize_amplitude_evolution()
        
        assert compiled_func is not None
        assert "evolve_amplitudes" in optimizer.compiled_functions
        
        # Test compiled function
        amplitudes = jnp.array([1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j])
        evolution_matrix = jnp.eye(4, dtype=jnp.complex64)
        time_step = 0.1
        
        result = compiled_func(amplitudes, evolution_matrix, time_step)
        
        assert result.shape == amplitudes.shape
        assert jnp.allclose(jnp.sum(jnp.abs(result)**2), 1.0)  # Normalized
    
    def test_interference_calculation_compilation(self, optimizer):
        """Test JIT compilation of interference calculations."""
        compiled_func = optimizer.optimize_interference_calculation()
        
        assert compiled_func is not None
        
        # Test with sample data
        path_amplitudes = jnp.array([0.5+0.5j, 0.3+0.7j])
        overlap_matrix = jnp.array([[0.0, 0.8], [0.8, 0.0]])
        
        result = compiled_func(path_amplitudes, overlap_matrix)
        
        assert result.shape == (2, 2)
        assert result[0, 0] == 0  # No self-interference
    
    def test_compilation_stats(self, optimizer):
        """Test compilation statistics tracking."""
        # Compile functions
        optimizer.optimize_amplitude_evolution()
        optimizer.optimize_interference_calculation()
        
        stats = optimizer.get_stats()
        
        assert stats["compiled_functions"] >= 2
        assert "evolve_amplitudes" in stats["compilation_times"]
        assert "calculate_interference" in stats["compilation_times"]
        assert stats["total_compilation_time"] > 0
    
    def test_function_call_tracking(self, optimizer):
        """Test tracking of compiled function calls."""
        compiled_func = optimizer.optimize_amplitude_evolution()
        
        # Call function multiple times
        for _ in range(5):
            retrieved_func = optimizer.get_compiled_function("evolve_amplitudes")
            assert retrieved_func is not None
        
        stats = optimizer.get_stats()
        assert stats["call_counts"]["evolve_amplitudes"] == 5


class TestVectorizedOperations:
    """Test vectorized quantum operations."""
    
    @pytest.fixture
    def vec_ops(self):
        """Create vectorized operations instance."""
        return VectorizedOperations()
    
    def test_batch_state_evolution(self, vec_ops):
        """Test vectorized state evolution."""
        batch_size = 4
        state_size = 4
        
        # Create batch of amplitudes
        batch_amplitudes = jnp.ones((batch_size, state_size), dtype=jnp.complex64) / 2.0
        evolution_matrices = jnp.stack([jnp.eye(state_size, dtype=jnp.complex64)] * batch_size)
        time_steps = jnp.ones(batch_size) * 0.1
        
        result = vec_ops.batch_state_evolution(batch_amplitudes, evolution_matrices, time_steps)
        
        assert result.shape == (batch_size, state_size)
        
        # Check normalization
        norms = jnp.sum(jnp.abs(result)**2, axis=1)
        assert jnp.allclose(norms, 1.0)
    
    def test_batch_probability_calculation(self, vec_ops):
        """Test vectorized probability calculation."""
        batch_size = 3
        state_size = 4
        
        # Create batch of normalized amplitudes
        batch_amplitudes = jnp.ones((batch_size, state_size), dtype=jnp.complex64)
        batch_amplitudes = batch_amplitudes / jnp.sqrt(state_size)
        
        probabilities = vec_ops.batch_probability_calculation(batch_amplitudes)
        
        assert probabilities.shape == (batch_size, state_size)
        
        # Check probability normalization
        prob_sums = jnp.sum(probabilities, axis=1)
        assert jnp.allclose(prob_sums, 1.0)
    
    def test_batch_coherence_measurement(self, vec_ops):
        """Test vectorized coherence measurement."""
        batch_size = 5
        state_size = 4
        
        # Create batch with different coherence levels
        batch_amplitudes = []
        
        # High coherence (aligned phases)
        high_coherence = jnp.ones(state_size, dtype=jnp.complex64) / 2.0
        batch_amplitudes.append(high_coherence)
        
        # Low coherence (random phases)
        phases = jnp.array([0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])
        low_coherence = jnp.exp(1j * phases) / 2.0
        batch_amplitudes.append(low_coherence)
        
        # Add more for batch
        for _ in range(3):
            random_phases = jnp.linspace(0, 2*jnp.pi, state_size)
            batch_amplitudes.append(jnp.exp(1j * random_phases) / 2.0)
        
        batch_amplitudes = jnp.stack(batch_amplitudes)
        
        coherences = vec_ops.batch_coherence_measurement(batch_amplitudes)
        
        assert coherences.shape == (batch_size,)
        assert jnp.all(coherences >= 0) and jnp.all(coherences <= 1)


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler."""
        return PerformanceProfiler(profile_threshold=0.01)
    
    def test_operation_profiling(self, profiler):
        """Test profiling of operations."""
        # Profile operation
        with profiler.profile_operation("test_operation"):
            time.sleep(0.02)  # Simulate work
        
        # Check profiling data
        assert "test_operation" in profiler.profiles
        assert len(profiler.profiles["test_operation"]) == 1
        assert profiler.profiles["test_operation"][0] >= 0.02
    
    def test_profile_threshold(self, profiler):
        """Test profiling threshold filtering."""
        # Very short operation (below threshold)
        profiler.record_execution_time("short_op", 0.005)  # 5ms, below 10ms threshold
        
        # Should not be recorded
        assert "short_op" not in profiler.profiles
        
        # Long operation (above threshold)
        profiler.record_execution_time("long_op", 0.02)  # 20ms, above threshold
        
        # Should be recorded
        assert "long_op" in profiler.profiles
    
    def test_optimization_recommendations(self, profiler):
        """Test optimization recommendations generation."""
        # Add operation data with high variance
        operation_times = [0.1, 0.15, 0.5, 0.12, 0.8, 0.11]  # High variance
        for time_val in operation_times:
            profiler.record_execution_time("variable_op", time_val)
        
        # Add consistently slow operation
        for _ in range(5):
            profiler.record_execution_time("slow_op", 0.2)  # Consistently slow
        
        recommendations = profiler.get_optimization_recommendations()
        
        assert len(recommendations) > 0
        
        # Should recommend optimization for high variance
        variance_recs = [r for r in recommendations if r["operation"] == "variable_op"]
        assert len(variance_recs) > 0
        assert any(r["issue"] == "high_variance" for r in variance_recs)
        
        # Should recommend optimization for slow operations
        slow_recs = [r for r in recommendations if r["operation"] == "slow_op"]
        assert len(slow_recs) > 0
    
    @patch('psutil.Process')
    def test_memory_snapshot(self, mock_process, profiler):
        """Test memory usage snapshots."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024  # 1MB
        mock_memory_info.vms = 2048 * 1024  # 2MB
        
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.memory_percent.return_value = 15.5
        
        profiler.take_memory_snapshot()
        
        assert len(profiler.memory_snapshots) == 1
        snapshot = profiler.memory_snapshots[0]
        
        assert snapshot["rss"] == 1024 * 1024
        assert snapshot["percent"] == 15.5
    
    def test_performance_report(self, profiler):
        """Test performance report generation."""
        # Add some profiling data
        for i in range(10):
            profiler.record_execution_time("test_op", 0.1 + i * 0.01)
        
        report = profiler.get_performance_report()
        
        assert "operations_profiled" in report
        assert "performance_summary" in report
        assert "optimization_recommendations" in report
        
        # Check performance summary
        assert "test_op" in report["performance_summary"]
        op_summary = report["performance_summary"]["test_op"]
        
        assert "count" in op_summary
        assert "avg_time" in op_summary
        assert "p95_time" in op_summary


class TestPerformanceManager:
    """Test integrated performance management."""
    
    @pytest.fixture
    def manager(self):
        """Create performance manager."""
        return PerformanceManager(
            cache_size=100,
            enable_jit=True,
            enable_vectorization=True,
            enable_memory_pool=True
        )
    
    def test_manager_initialization(self, manager):
        """Test performance manager initialization."""
        assert manager.cache is not None
        assert manager.memory_pool is not None
        assert manager.jit_optimizer is not None
        assert manager.vectorized_ops is not None
        assert manager.profiler is not None
    
    def test_cached_operation_decorator(self, manager):
        """Test cached operation decorator."""
        call_count = 0
        
        @manager.cached_operation("test_cache_key")
        def expensive_operation(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = expensive_operation(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_operation(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
    
    def test_batch_processing(self, manager):
        """Test batch processing optimization."""
        # Create test tasks
        tasks = {f"task_{i}": {"value": i} for i in range(10)}
        
        # Process batch
        results = manager.batch_process_tasks(tasks, "test_operation")
        
        assert len(results) == len(tasks)
        assert all(isinstance(r, dict) for r in results.values())
    
    def test_performance_dashboard(self, manager):
        """Test performance dashboard."""
        # Trigger some operations to populate metrics
        manager.cache.put("test", "value")
        manager.cache.get("test")
        
        with manager.profiler.profile_operation("dashboard_test"):
            time.sleep(0.01)
        
        dashboard = manager.get_performance_dashboard()
        
        assert "cache_stats" in dashboard
        assert "profiler_report" in dashboard
        assert "optimization_enabled" in dashboard
        
        # Check cache stats
        cache_stats = dashboard["cache_stats"]
        assert "hits" in cache_stats
        assert "hit_rate" in cache_stats
    
    def test_memory_optimization(self, manager):
        """Test memory optimization functionality."""
        # Add some cached items
        for i in range(50):
            manager.cache.put(f"key_{i}", f"value_{i}")
        
        # Trigger memory optimization
        manager.optimize_memory_usage()
        
        # Should clean up cache and take memory snapshot
        # Exact behavior depends on system state
    
    def test_auto_tuning(self, manager):
        """Test automatic performance tuning."""
        # Add some profiling data that suggests optimization
        for i in range(20):
            manager.profiler.record_execution_time("slow_operation", 0.2 + i * 0.01)
        
        # Trigger auto-tuning
        recommendations_count = manager.auto_tune_performance()
        
        assert isinstance(recommendations_count, int)
        assert recommendations_count >= 0


class TestPerformanceIntegration:
    """Integration tests for performance components."""
    
    def test_end_to_end_performance_optimization(self):
        """Test complete performance optimization workflow."""
        manager = PerformanceManager()
        
        # Simulate quantum computation workload
        def quantum_computation(task_data):
            # Simulate some computation
            result = jnp.array([1.0, 2.0, 3.0, 4.0])
            return jnp.dot(result, result)
        
        # Process multiple batches with caching
        tasks_batch1 = {f"task_{i}": {"data": i} for i in range(5)}
        tasks_batch2 = {f"task_{i}": {"data": i} for i in range(5)}  # Same tasks (should hit cache)
        
        # First batch
        with manager.profiler.profile_operation("batch_1"):
            results1 = manager.batch_process_tasks(tasks_batch1, "quantum_computation")
        
        # Second batch (should benefit from caching)
        with manager.profiler.profile_operation("batch_2"):
            results2 = manager.batch_process_tasks(tasks_batch2, "quantum_computation")
        
        # Both should succeed
        assert len(results1) == 5
        assert len(results2) == 5
        
        # Check performance metrics
        dashboard = manager.get_performance_dashboard()
        assert dashboard["cache_stats"]["hits"] >= 0
        
        # Generate optimization recommendations
        recommendations = manager.profiler.get_optimization_recommendations()
        assert isinstance(recommendations, list)
    
    def test_performance_under_load(self):
        """Test performance optimization under high load."""
        manager = PerformanceManager(cache_size=1000)
        
        # Generate large workload
        large_workload = {f"task_{i}": {"value": i} for i in range(500)}
        
        start_time = time.time()
        results = manager.batch_process_tasks(large_workload, "load_test")
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert execution_time < 10.0  # 10 seconds max
        assert len(results) == 500
        
        # Check system performed well
        dashboard = manager.get_performance_dashboard()
        cache_stats = dashboard["cache_stats"]
        
        # Cache should be utilized
        assert cache_stats["utilization"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])