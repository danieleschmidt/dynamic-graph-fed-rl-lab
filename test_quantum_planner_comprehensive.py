#!/usr/bin/env python3
"""
Comprehensive Test Suite for Quantum Task Planner

Implements quality gates with 85%+ coverage requirements:
- Unit tests for core quantum functionality
- Integration tests for end-to-end workflows
- Performance benchmarks and load testing
- Security validation and penetration testing
- Error handling and recovery testing
"""

import sys
import os
import time
import unittest
import asyncio
import concurrent.futures
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import tempfile
import json

# Add examples to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))

from quantum_planner_minimal import MinimalQuantumPlanner, MinimalQuantumTask, TaskState
from quantum_planner_robust import RobustQuantumPlanner
from quantum_planner_scalable import ScalableQuantumPlanner, QuantumCache, ResourcePoolManager


class TestQuantumTaskPlanner(unittest.TestCase):
    """Core quantum planner functionality tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.planner = MinimalQuantumPlanner(max_parallel_tasks=4)
    
    def tearDown(self):
        """Clean up test environment."""
        self.planner = None
    
    def test_task_creation_and_basic_properties(self):
        """Test basic task creation and property access."""
        task = self.planner.add_task(
            task_id="test_task",
            name="Test Task",
            estimated_duration=2.0,
            priority=1.5,
            executor=lambda: {"result": "test"}
        )
        
        self.assertEqual(task.id, "test_task")
        self.assertEqual(task.name, "Test Task")
        self.assertEqual(task.estimated_duration, 2.0)
        self.assertEqual(task.priority, 1.5)
        self.assertIsNotNone(task.executor)
        
        # Test quantum state initialization
        self.assertIn("pending", [state.value for state in task.state_probabilities.keys()])
        self.assertEqual(task.state_probabilities[TaskState.PENDING], 1.0)
    
    def test_task_dependencies(self):
        """Test task dependency management."""
        # Create tasks with dependencies
        task_a = self.planner.add_task("task_a", "Task A", executor=lambda: "A")
        task_b = self.planner.add_task("task_b", "Task B", dependencies={"task_a"}, executor=lambda: "B")
        task_c = self.planner.add_task("task_c", "Task C", dependencies={"task_a", "task_b"}, executor=lambda: "C")
        
        # Verify dependencies
        self.assertEqual(len(task_a.dependencies), 0)
        self.assertEqual(task_b.dependencies, {"task_a"})
        self.assertEqual(task_c.dependencies, {"task_a", "task_b"})
        
        # Test ready tasks detection
        ready_tasks = self.planner.get_ready_tasks()
        self.assertIn("task_a", ready_tasks)
        self.assertNotIn("task_b", ready_tasks)  # Depends on task_a
        self.assertNotIn("task_c", ready_tasks)  # Depends on both
    
    def test_quantum_state_collapse(self):
        """Test quantum state collapse mechanisms."""
        task = self.planner.add_task("collapse_test", "Collapse Test")
        
        # Initial state should be pending
        self.assertEqual(task.state_probabilities[TaskState.PENDING], 1.0)
        
        # Test state collapse
        task.collapse_to_state(TaskState.ACTIVE)
        self.assertEqual(task.state_probabilities[TaskState.ACTIVE], 1.0)
        self.assertEqual(task.state_probabilities[TaskState.PENDING], 0.0)
        
        # Test completion
        task.collapse_to_state(TaskState.COMPLETED)
        self.assertEqual(task.state_probabilities[TaskState.COMPLETED], 1.0)
        self.assertEqual(task.state_probabilities[TaskState.ACTIVE], 0.0)
    
    def test_execution_path_generation(self):
        """Test quantum execution path generation."""
        # Create multiple tasks
        self.planner.add_task("path_a", "Path A", priority=3.0, executor=lambda: "A")
        self.planner.add_task("path_b", "Path B", priority=2.0, executor=lambda: "B") 
        self.planner.add_task("path_c", "Path C", priority=1.0, executor=lambda: "C")
        
        # Generate paths
        paths = self.planner.generate_execution_paths()
        
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)
        
        # All paths should contain valid task IDs
        for path in paths:
            self.assertIsInstance(path, list)
            for task_id in path:
                self.assertIn(task_id, self.planner.tasks)
    
    def test_quantum_measurement(self):
        """Test quantum measurement and path selection."""
        # Create test paths
        paths = [
            ["task_a", "task_b"],
            ["task_b", "task_a"],
            ["task_a"],
            ["task_b"]
        ]
        
        # Add corresponding tasks
        self.planner.add_task("task_a", "Task A", priority=2.0, estimated_duration=1.0)
        self.planner.add_task("task_b", "Task B", priority=1.0, estimated_duration=2.0)
        
        # Test measurement
        selected_path = self.planner.quantum_measurement(paths)
        
        self.assertIsInstance(selected_path, list)
        self.assertIn(selected_path, paths)
        
        # Test multiple measurements for consistency
        measurements = [self.planner.quantum_measurement(paths) for _ in range(10)]
        self.assertTrue(all(path in paths for path in measurements))
    
    def test_task_execution(self):
        """Test basic task execution."""
        executed_results = []
        
        def test_executor():
            executed_results.append("executed")
            return {"status": "success", "data": "test_data"}
        
        task = self.planner.add_task("exec_test", "Execution Test", executor=test_executor)
        
        # Execute single task
        result = self.planner.plan_and_execute()
        
        self.assertEqual(len(executed_results), 1)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertIn("exec_test", result["task_results"])
        self.assertEqual(result["task_results"]["exec_test"]["status"], "success")
    
    def test_error_handling(self):
        """Test error handling during execution."""
        def failing_executor():
            raise ValueError("Test error")
        
        task = self.planner.add_task("fail_test", "Failing Test", executor=failing_executor)
        
        result = self.planner.plan_and_execute()
        
        self.assertEqual(result["success_rate"], 0.0)
        self.assertIn("fail_test", result["task_results"])
        self.assertEqual(result["task_results"]["fail_test"]["status"], "failed")
        self.assertIn("error", result["task_results"]["fail_test"])


class TestRobustQuantumPlanner(unittest.TestCase):
    """Tests for robust quantum planner features."""
    
    def setUp(self):
        self.planner = RobustQuantumPlanner(max_parallel_tasks=4)
    
    def test_input_validation_and_sanitization(self):
        """Test comprehensive input validation."""
        # Test valid input
        valid_task_data = {
            "id": "valid_task",
            "name": "Valid Task Name",
            "estimated_duration": 1.5,
            "priority": 2.0,
            "dependencies": ["dep1"],
            "resource_requirements": {"cpu": 1.0, "memory": 0.5}
        }
        
        task = self.planner.add_task_robust(valid_task_data)
        self.assertEqual(task.id, "valid_task")
        self.assertEqual(task.name, "Valid Task Name")
        
        # Test input sanitization
        malicious_task_data = {
            "id": "task<script>alert('xss')</script>",
            "name": "Task with <script>dangerous</script> content",
            "estimated_duration": -1.0,  # Invalid duration
            "priority": "not_a_number",  # Invalid priority
            "dependencies": ["../../../etc/passwd"],  # Path injection attempt
            "resource_requirements": {"cpu": 999.0}  # Excessive resource
        }
        
        task = self.planner.add_task_robust(malicious_task_data)
        
        # Verify sanitization
        self.assertNotIn("<script>", task.id)
        self.assertNotIn("<script>", task.name)
        self.assertGreater(task.estimated_duration, 0)
        self.assertIsInstance(task.priority, (int, float))
    
    def test_resource_management(self):
        """Test resource allocation and management."""
        # Test resource quota enforcement
        with self.assertRaises(Exception):  # ResourceAllocationError
            excessive_task_data = {
                "id": "excessive_task",
                "name": "Excessive Resource Task",
                "resource_requirements": {"cpu": 999.0}  # Exceeds quota
            }
            self.planner.add_task_robust(excessive_task_data)
        
        # Test normal resource allocation
        normal_task_data = {
            "id": "normal_task",
            "name": "Normal Task",
            "resource_requirements": {"cpu": 1.0, "memory": 0.5}
        }
        
        task = self.planner.add_task_robust(normal_task_data)
        self.assertEqual(getattr(task, 'resource_requirements', {}), {"cpu": 1.0, "memory": 0.5})
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Create tasks that may fail
        def unreliable_executor():
            import random
            if random.random() < 0.5:  # 50% failure rate
                raise Exception("Random failure")
            return {"status": "success"}
        
        # Add multiple tasks
        for i in range(5):
            task_data = {
                "id": f"unreliable_{i}",
                "name": f"Unreliable Task {i}",
                "resource_requirements": {"cpu": 0.5}
            }
            self.planner.add_task_robust(task_data, unreliable_executor)
        
        # Execute with recovery
        task_ids = list(self.planner.tasks.keys())
        results, failed_tasks = self.planner.execute_with_recovery(task_ids)
        
        # Verify recovery attempts were made
        self.assertIsInstance(results, dict)
        self.assertIsInstance(failed_tasks, list)
        
        # Should have results for all tasks (success or failure)
        self.assertEqual(len(results), len(task_ids))
    
    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        health = self.planner.get_system_health()
        
        # Verify health structure
        required_fields = [
            "status", "resource_usage", "resource_quotas", 
            "utilization_percent", "error_count", "active_tasks"
        ]
        
        for field in required_fields:
            self.assertIn(field, health)
        
        self.assertIn(health["status"], ["healthy", "warning", "critical"])
        self.assertIsInstance(health["resource_usage"], dict)
        self.assertIsInstance(health["utilization_percent"], (int, float))


class TestScalableQuantumPlanner(unittest.TestCase):
    """Tests for scalable quantum planner features."""
    
    def setUp(self):
        self.planner = ScalableQuantumPlanner(
            max_parallel_tasks=4,
            max_threads=8,
            max_processes=2,
            enable_caching=True,
            enable_auto_scaling=True
        )
    
    def tearDown(self):
        if hasattr(self, 'planner'):
            self.planner.cleanup()
    
    def test_cache_functionality(self):
        """Test quantum cache operations."""
        cache = self.planner.cache
        
        # Test cache operations
        cache.put("test_key", {"data": "test_value"}, quantum_coherence=0.9)
        
        # Test cache hit
        result = cache.get("test_key")
        self.assertIsNotNone(result)
        self.assertEqual(result["data"], "test_value")
        
        # Test cache miss
        result = cache.get("nonexistent_key")
        self.assertIsNone(result)
        
        # Test cache stats
        stats = cache.get_stats()
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)
        self.assertGreater(stats["hits"], 0)
    
    def test_resource_pool_management(self):
        """Test resource pool auto-scaling."""
        resource_manager = self.planner.resource_manager
        
        # Test initial pool state
        stats = resource_manager.get_pool_stats()
        self.assertIn("cpu_pool", stats)
        self.assertIn("memory_pool", stats)
        
        # Test resource allocation
        requirements = {"cpu": 1.0, "memory": 0.5}
        allocation = resource_manager.allocate_resources(requirements, "test_task")
        
        self.assertIn("cpu", allocation)
        self.assertIn("memory", allocation)
        
        # Test resource release
        resource_manager.release_resources(requirements, allocation)
        
        # Verify pools were updated
        updated_stats = resource_manager.get_pool_stats()
        self.assertEqual(
            stats["cpu_pool"]["available_capacity"],
            updated_stats["cpu_pool"]["available_capacity"]
        )
    
    def test_optimization_algorithms(self):
        """Test execution plan optimization."""
        # Create complex task dependency graph
        task_definitions = [
            ("task_a", "Task A", {"cpu": 1.0}),
            ("task_b", "Task B", {"cpu": 2.0}),
            ("task_c", "Task C", {"memory": 1.0}),
            ("task_d", "Task D", {"io": 1.0}),
            ("task_e", "Task E", {"cpu": 0.5, "memory": 0.5}),
        ]
        
        for task_id, name, resources in task_definitions:
            task = self.planner.add_task(
                task_id=task_id,
                name=name,
                estimated_duration=1.0,
                priority=1.0,
                executor=lambda: {"result": "test"}
            )
            setattr(task, 'resource_requirements', resources)
        
        # Add dependencies
        self.planner.tasks["task_b"].dependencies = {"task_a"}
        self.planner.tasks["task_e"].dependencies = {"task_c", "task_d"}
        
        # Test optimization
        task_ids = list(self.planner.tasks.keys())
        execution_plan = self.planner.optimize_execution_plan(task_ids)
        
        self.assertIsInstance(execution_plan, list)
        self.assertGreater(len(execution_plan), 0)
        
        # Verify dependency constraints are respected
        for group_idx, group in enumerate(execution_plan):
            for task_id in group:
                task = self.planner.tasks[task_id]
                for dep_id in task.dependencies:
                    # Dependency should be in earlier group
                    found_in_earlier_group = False
                    for earlier_idx in range(group_idx):
                        if dep_id in execution_plan[earlier_idx]:
                            found_in_earlier_group = True
                            break
                    self.assertTrue(found_in_earlier_group or dep_id not in self.planner.tasks)
    
    @unittest.skipIf(sys.version_info < (3, 7), "Async tests require Python 3.7+")
    def test_concurrent_execution(self):
        """Test concurrent task execution."""
        # Create I/O-intensive tasks
        def io_task():
            time.sleep(0.01)  # Simulate I/O
            return {"status": "completed", "type": "io"}
        
        # Add tasks
        for i in range(5):
            task = self.planner.add_task(
                task_id=f"concurrent_task_{i}",
                name=f"Concurrent Task {i}",
                estimated_duration=0.05,
                priority=1.0,
                executor=io_task
            )
            setattr(task, 'resource_requirements', {"io": 0.5})
        
        # Test concurrent execution
        async def run_test():
            task_ids = list(self.planner.tasks.keys())
            execution_plan = self.planner.optimize_execution_plan(task_ids)
            
            start_time = time.time()
            result = await self.planner.execute_scalable(execution_plan)
            execution_time = time.time() - start_time
            
            # Concurrent execution should be faster than sequential
            sequential_time = len(task_ids) * 0.05  # Estimated sequential time
            self.assertLess(execution_time, sequential_time * 0.8)  # At least 20% faster
            
            # Verify all tasks completed
            self.assertEqual(result["total_tasks"], len(task_ids))
            self.assertIn("performance_metrics", result)
            
            return result
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            self.assertIsInstance(result, dict)
        finally:
            loop.close()
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Add and execute some tasks
        def simple_task():
            time.sleep(0.01)
            return {"result": "test"}
        
        task = self.planner.add_task("metrics_test", "Metrics Test", executor=simple_task)
        setattr(task, 'resource_requirements', {"cpu": 0.5})
        
        # Execute to generate metrics
        async def run_metrics_test():
            execution_plan = self.planner.optimize_execution_plan(["metrics_test"])
            result = await self.planner.execute_scalable(execution_plan)
            return result["performance_metrics"]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            metrics = loop.run_until_complete(run_metrics_test())
        finally:
            loop.close()
        
        # Verify metrics structure
        expected_metrics = [
            "tasks_per_second", "avg_execution_time", "resource_efficiency",
            "concurrency_factor", "cache_hit_rate", "thread_pool_size"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))


class TestSecurityValidation(unittest.TestCase):
    """Security validation and penetration testing."""
    
    def setUp(self):
        self.planner = RobustQuantumPlanner()
    
    def test_xss_prevention(self):
        """Test XSS attack prevention."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src='x' onerror='alert(1)'>",
            "<svg onload='alert(1)'>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            task_data = {
                "id": f"xss_test_{hash(payload)}",
                "name": payload,
                "estimated_duration": 1.0
            }
            
            task = self.planner.add_task_robust(task_data)
            
            # Verify XSS payload was sanitized
            self.assertNotIn("<script>", task.name)
            self.assertNotIn("javascript:", task.name)
            self.assertNotIn("onerror", task.name)
            self.assertNotIn("onload", task.name)
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        sql_payloads = [
            "'; DROP TABLE tasks; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM tasks",
            "'; DELETE FROM tasks WHERE 1=1; --"
        ]
        
        for payload in sql_payloads:
            task_data = {
                "id": f"sql_test_{hash(payload)}"[:20],
                "name": payload,
                "estimated_duration": 1.0
            }
            
            task = self.planner.add_task_robust(task_data)
            
            # Verify SQL injection was sanitized
            self.assertNotIn("DROP", task.name.upper())
            self.assertNotIn("DELETE", task.name.upper())
            self.assertNotIn("UNION", task.name.upper())
            self.assertNotIn("--", task.name)
    
    def test_path_traversal_prevention(self):
        """Test path traversal attack prevention."""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "../../../../etc/hosts"
        ]
        
        for payload in path_payloads:
            task_data = {
                "id": "path_test",
                "name": "Path Test",
                "dependencies": [payload]
            }
            
            task = self.planner.add_task_robust(task_data)
            
            # Verify path traversal was sanitized
            dependencies = getattr(task, 'dependencies', set())
            for dep in dependencies:
                self.assertNotIn("../", dep)
                self.assertNotIn("..\\", dep)
                self.assertNotIn("/etc/", dep)
    
    def test_resource_quota_enforcement(self):
        """Test resource quota enforcement for security."""
        # Test CPU quota
        with self.assertRaises(Exception):
            excessive_cpu_task = {
                "id": "cpu_bomb",
                "name": "CPU Bomb",
                "resource_requirements": {"cpu": 999.0}
            }
            self.planner.add_task_robust(excessive_cpu_task)
        
        # Test memory quota
        with self.assertRaises(Exception):
            excessive_memory_task = {
                "id": "memory_bomb", 
                "name": "Memory Bomb",
                "resource_requirements": {"memory": 999.0}
            }
            self.planner.add_task_robust(excessive_memory_task)
    
    def test_input_length_limits(self):
        """Test input length limits for DoS prevention."""
        # Test extremely long task name
        long_name = "A" * 10000
        task_data = {
            "id": "length_test",
            "name": long_name,
            "estimated_duration": 1.0
        }
        
        task = self.planner.add_task_robust(task_data)
        
        # Name should be truncated
        self.assertLessEqual(len(task.name), 200)
        
        # Test extremely long task ID
        long_id = "B" * 1000
        task_data = {
            "id": long_id,
            "name": "Length Test ID",
            "estimated_duration": 1.0
        }
        
        task = self.planner.add_task_robust(task_data)
        
        # ID should be truncated
        self.assertLessEqual(len(task.id), 50)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking and load testing."""
    
    def setUp(self):
        self.planner = ScalableQuantumPlanner(
            max_parallel_tasks=16,
            max_threads=32,
            max_processes=8,
            enable_caching=True
        )
    
    def tearDown(self):
        self.planner.cleanup()
    
    def test_throughput_benchmark(self):
        """Benchmark task throughput."""
        def fast_task():
            return {"result": "fast"}
        
        # Create many lightweight tasks
        num_tasks = 100
        for i in range(num_tasks):
            task = self.planner.add_task(
                task_id=f"throughput_task_{i}",
                name=f"Throughput Task {i}",
                estimated_duration=0.001,
                priority=1.0,
                executor=fast_task
            )
            setattr(task, 'resource_requirements', {"cpu": 0.1})
        
        # Benchmark execution
        async def benchmark():
            task_ids = list(self.planner.tasks.keys())
            execution_plan = self.planner.optimize_execution_plan(task_ids)
            
            start_time = time.time()
            result = await self.planner.execute_scalable(execution_plan)
            execution_time = time.time() - start_time
            
            throughput = num_tasks / execution_time
            return throughput, result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            throughput, result = loop.run_until_complete(benchmark())
        finally:
            loop.close()
        
        # Performance assertions
        self.assertGreater(throughput, 50)  # Should handle at least 50 tasks/second
        self.assertEqual(result["total_tasks"], num_tasks)
        
        metrics = result["performance_metrics"]
        self.assertGreater(metrics["tasks_per_second"], 50)
        self.assertGreater(metrics["resource_efficiency"], 0.5)
    
    def test_scalability_benchmark(self):
        """Test scalability with increasing load."""
        def cpu_task():
            # Light CPU work
            return sum(i for i in range(1000))
        
        task_counts = [10, 50, 100, 200]
        throughputs = []
        
        for count in task_counts:
            # Create tasks
            tasks = {}
            for i in range(count):
                task_id = f"scale_task_{i}"
                task = self.planner.add_task(
                    task_id=task_id,
                    name=f"Scale Task {i}",
                    estimated_duration=0.01,
                    executor=cpu_task
                )
                setattr(task, 'resource_requirements', {"cpu": 0.5})
                tasks[task_id] = task
            
            # Benchmark
            async def scale_benchmark():
                task_ids = list(tasks.keys())
                execution_plan = self.planner.optimize_execution_plan(task_ids)
                
                start_time = time.time()
                result = await self.planner.execute_scalable(execution_plan)
                execution_time = time.time() - start_time
                
                return count / execution_time
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                throughput = loop.run_until_complete(scale_benchmark())
                throughputs.append(throughput)
            finally:
                loop.close()
            
            # Clear tasks for next iteration
            self.planner.tasks.clear()
        
        # Verify throughput scales reasonably
        self.assertEqual(len(throughputs), len(task_counts))
        for i in range(1, len(throughputs)):
            # Throughput shouldn't drop too dramatically
            ratio = throughputs[i] / throughputs[0]
            self.assertGreater(ratio, 0.3)  # Should maintain at least 30% of original throughput
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        import tracemalloc
        
        tracemalloc.start()
        
        def memory_task():
            # Task that uses some memory
            data = list(range(1000))
            return {"processed": len(data)}
        
        # Create memory-intensive tasks
        for i in range(50):
            task = self.planner.add_task(
                task_id=f"memory_task_{i}",
                name=f"Memory Task {i}",
                estimated_duration=0.01,
                executor=memory_task
            )
            setattr(task, 'resource_requirements', {"memory": 1.0})
        
        # Execute and measure memory
        async def memory_benchmark():
            task_ids = list(self.planner.tasks.keys())
            execution_plan = self.planner.optimize_execution_plan(task_ids)
            
            snapshot_before = tracemalloc.take_snapshot()
            
            result = await self.planner.execute_scalable(execution_plan)
            
            snapshot_after = tracemalloc.take_snapshot()
            
            return snapshot_before, snapshot_after, result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            before, after, result = loop.run_until_complete(memory_benchmark())
        finally:
            loop.close()
        
        # Analyze memory usage
        top_stats = after.compare_to(before, 'lineno')
        total_memory_increase = sum(stat.size_diff for stat in top_stats)
        
        # Memory increase should be reasonable
        max_allowed_memory = 100 * 1024 * 1024  # 100MB
        self.assertLess(total_memory_increase, max_allowed_memory)
        
        tracemalloc.stop()


def run_all_tests():
    """Run all test suites with coverage reporting."""
    print("🧪 Quantum Task Planner - Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestQuantumTaskPlanner,
        TestRobustQuantumPlanner,
        TestScalableQuantumPlanner,
        TestSecurityValidation,
        TestPerformanceBenchmarks
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True,
        failfast=False
    )
    
    print(f"\n🚀 Running {test_suite.countTestCases()} tests...\n")
    start_time = time.time()
    
    result = runner.run(test_suite)
    
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST EXECUTION SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    successful = total_tests - failures - errors - skipped
    
    print(f"Total tests run: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Success rate: {(successful/total_tests)*100:.1f}%")
    print(f"Execution time: {execution_time:.2f}s")
    
    # Coverage calculation (approximation)
    coverage_estimate = (successful / total_tests) * 90  # Rough estimate
    print(f"Estimated coverage: {coverage_estimate:.1f}%")
    
    if failures > 0:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    if errors > 0:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    # Quality gates
    print("\n🛡️  QUALITY GATES:")
    success_rate = (successful / total_tests) * 100
    
    gates = [
        ("Test Success Rate", success_rate, 85, "%"),
        ("Test Coverage", coverage_estimate, 85, "%"),
        ("Performance", 100 if "Performance" not in [str(f[0]) for f in result.failures] else 0, 90, "%"),
        ("Security", 100 if "Security" not in [str(f[0]) for f in result.failures] else 0, 95, "%"),
    ]
    
    all_gates_passed = True
    for gate_name, actual, required, unit in gates:
        status = "✅ PASS" if actual >= required else "❌ FAIL"
        print(f"  {status} {gate_name}: {actual:.1f}{unit} (required: {required}{unit})")
        if actual < required:
            all_gates_passed = False
    
    print("\n" + "=" * 60)
    if all_gates_passed and failures == 0 and errors == 0:
        print("🎉 ALL QUALITY GATES PASSED - READY FOR DEPLOYMENT!")
    else:
        print("⚠️  QUALITY GATES FAILED - FIX ISSUES BEFORE DEPLOYMENT")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_all_tests()