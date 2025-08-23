#!/usr/bin/env python3
"""
Comprehensive Quality Gates - Testing, security, performance validation
for autonomous federated RL system.
"""

import json
import time
import math
import random
import threading
import subprocess
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result structure."""
    test_name: str
    passed: bool
    duration: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    test_results: List[TestResult]
    security_score: float
    performance_score: float
    coverage_score: float
    overall_score: float
    timestamp: float


class SecurityValidator:
    """Security validation and testing."""
    
    def __init__(self):
        self.security_tests = []
    
    def test_parameter_injection_attack(self) -> TestResult:
        """Test resistance to parameter injection attacks."""
        start_time = time.time()
        
        try:
            # Simulate malicious parameter injection
            malicious_params = [float('inf'), float('-inf'), 1e20, -1e20]
            
            # Test parameter bounds validation
            from scalable_gen3_system import ScalableAgent
            agent = ScalableAgent(0)
            
            # Should handle malicious parameters gracefully
            original_params = agent.get_parameters()
            agent.set_parameters(malicious_params)
            new_params = agent.get_parameters()
            
            # Parameters should be bounded
            max_param = max(abs(p) for p in new_params)
            passed = max_param <= 100.0  # Reasonable bound
            
            return TestResult(
                test_name="parameter_injection_resistance",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'max_parameter_value': max_param,
                    'malicious_params_tested': len(malicious_params),
                    'bounds_enforced': passed
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="parameter_injection_resistance",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def test_data_sanitization(self) -> TestResult:
        """Test input data sanitization."""
        start_time = time.time()
        
        try:
            from scalable_gen3_system import ScalableAgent, ScalableGraphState
            
            agent = ScalableAgent(0)
            
            # Test with malicious input data
            malicious_features = [
                [float('inf'), float('-inf')],
                [1e50, -1e50],
                [float('nan'), 0]
            ]
            
            state = ScalableGraphState(
                node_features=malicious_features,
                edges=[(0, 1)],
                timestamp=time.time()
            )
            
            # Should handle gracefully
            action = agent.select_action(state)
            
            passed = not (math.isnan(action) or math.isinf(action))
            
            return TestResult(
                test_name="data_sanitization",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'action_is_finite': passed,
                    'malicious_inputs_tested': len(malicious_features)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="data_sanitization",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def test_byzantine_resistance(self) -> TestResult:
        """Test resistance to Byzantine agents."""
        start_time = time.time()
        
        try:
            from scalable_gen3_system import ScalableAgent, ScalableFederationProtocol
            
            # Create normal agents
            normal_agents = [ScalableAgent(i) for i in range(3)]
            
            # Create Byzantine agent that returns extreme parameters
            byzantine_agent = ScalableAgent(3)
            byzantine_agent.parameters = [1000.0] * 10  # Extreme values
            
            # Test federation with Byzantine agent
            all_agents = normal_agents + [byzantine_agent]
            federation = ScalableFederationProtocol(all_agents)
            
            # Attempt federation
            success = federation.federated_averaging_concurrent()
            
            # Check if global parameters are still reasonable
            if success:
                max_global_param = max(abs(p) for p in federation.global_parameters)
                passed = max_global_param <= 100.0  # Should be bounded
            else:
                passed = True  # Rejection is acceptable
            
            return TestResult(
                test_name="byzantine_resistance",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'federation_completed': success,
                    'parameters_bounded': passed,
                    'byzantine_agents': 1,
                    'normal_agents': 3
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="byzantine_resistance",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def run_all_security_tests(self) -> List[TestResult]:
        """Run all security tests."""
        logger.info("Running security validation tests...")
        
        tests = [
            self.test_parameter_injection_attack(),
            self.test_data_sanitization(),
            self.test_byzantine_resistance()
        ]
        
        passed_tests = sum(1 for test in tests if test.passed)
        logger.info(f"Security tests: {passed_tests}/{len(tests)} passed")
        
        return tests


class PerformanceValidator:
    """Performance testing and validation."""
    
    def __init__(self):
        self.benchmarks = []
    
    def benchmark_action_selection(self) -> TestResult:
        """Benchmark action selection performance."""
        start_time = time.time()
        
        try:
            from scalable_gen3_system import ScalableAgent, ScalableGraphState
            
            agent = ScalableAgent(0)
            
            # Create test state
            state = ScalableGraphState(
                node_features=[[random.random() for _ in range(4)] for _ in range(10)],
                edges=[(i, i+1) for i in range(9)],
                timestamp=time.time()
            )
            
            # Benchmark multiple action selections
            num_iterations = 1000
            action_times = []
            
            for _ in range(num_iterations):
                iter_start = time.perf_counter()
                action = agent.select_action(state)
                iter_end = time.perf_counter()
                action_times.append(iter_end - iter_start)
            
            avg_time = sum(action_times) / len(action_times)
            max_time = max(action_times)
            
            # Performance threshold: < 1ms per action on average
            passed = avg_time < 0.001
            
            return TestResult(
                test_name="action_selection_performance",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'iterations': num_iterations,
                    'avg_time_seconds': avg_time,
                    'max_time_seconds': max_time,
                    'actions_per_second': 1.0 / avg_time if avg_time > 0 else float('inf')
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="action_selection_performance",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def benchmark_federation_performance(self) -> TestResult:
        """Benchmark federated averaging performance."""
        start_time = time.time()
        
        try:
            from scalable_gen3_system import ScalableAgent, ScalableFederationProtocol
            
            # Create agents
            agents = [ScalableAgent(i) for i in range(10)]
            federation = ScalableFederationProtocol(agents)
            
            # Benchmark federation rounds
            num_rounds = 100
            federation_times = []
            
            for _ in range(num_rounds):
                round_start = time.perf_counter()
                success = federation.federated_averaging_concurrent()
                round_end = time.perf_counter()
                
                if success:
                    federation_times.append(round_end - round_start)
            
            if federation_times:
                avg_time = sum(federation_times) / len(federation_times)
                max_time = max(federation_times)
                
                # Performance threshold: < 10ms per federation round
                passed = avg_time < 0.01
            else:
                passed = False
                avg_time = float('inf')
                max_time = float('inf')
            
            return TestResult(
                test_name="federation_performance",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'successful_rounds': len(federation_times),
                    'total_rounds': num_rounds,
                    'avg_time_seconds': avg_time,
                    'max_time_seconds': max_time,
                    'rounds_per_second': 1.0 / avg_time if avg_time > 0 else 0
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="federation_performance",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def benchmark_memory_usage(self) -> TestResult:
        """Benchmark memory usage patterns."""
        start_time = time.time()
        
        try:
            import gc
            from scalable_gen3_system import ScalableAgent, ScalableTrafficEnvironment
            
            gc.collect()  # Clean start
            initial_objects = len(gc.get_objects())
            
            # Create system components
            env = ScalableTrafficEnvironment(num_intersections=20)
            agents = [ScalableAgent(i) for i in range(10)]
            
            # Run simulation
            state = env.reset()
            for _ in range(100):
                actions = {agent.agent_id: agent.select_action(state) for agent in agents}
                state, rewards, done = env._step_single_optimized(actions)
                
                for agent in agents:
                    if agent.agent_id in rewards:
                        agent.local_update_async(state, actions[agent.agent_id], rewards[agent.agent_id])
                
                if done:
                    state = env.reset()
            
            # Check memory growth
            gc.collect()
            final_objects = len(gc.get_objects())
            object_growth = final_objects - initial_objects
            
            # Memory threshold: object growth should be reasonable
            passed = object_growth < 10000  # Allow some growth but not excessive
            
            # Cleanup
            for agent in agents:
                agent.cleanup()
            
            return TestResult(
                test_name="memory_usage",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'initial_objects': initial_objects,
                    'final_objects': final_objects,
                    'object_growth': object_growth,
                    'growth_rate': object_growth / 100.0  # per iteration
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="memory_usage",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def run_all_performance_tests(self) -> List[TestResult]:
        """Run all performance tests."""
        logger.info("Running performance validation tests...")
        
        tests = [
            self.benchmark_action_selection(),
            self.benchmark_federation_performance(),
            self.benchmark_memory_usage()
        ]
        
        passed_tests = sum(1 for test in tests if test.passed)
        logger.info(f"Performance tests: {passed_tests}/{len(tests)} passed")
        
        return tests


class FunctionalTester:
    """Functional correctness testing."""
    
    def test_generation1_functionality(self) -> TestResult:
        """Test Generation 1 basic functionality."""
        start_time = time.time()
        
        try:
            # Import Generation 1 demo
            import simple_demo_gen1
            
            # Run demo and check results
            results = simple_demo_gen1.run_generation1_demo()
            
            if results and len(results) > 0:
                final_result = results[-1]
                federation_stats = final_result['federation_stats']
                
                # Validate results
                passed = (
                    federation_stats['federation_rounds'] > 0 and
                    len(results) >= 15 and  # At least 15 episodes
                    'avg_reward' in federation_stats
                )
            else:
                passed = False
            
            return TestResult(
                test_name="generation1_functionality",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'episodes_completed': len(results) if results else 0,
                    'federation_rounds': federation_stats.get('federation_rounds', 0) if results else 0,
                    'final_reward': federation_stats.get('avg_reward', 0) if results else 0
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="generation1_functionality",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def test_generation2_robustness(self) -> TestResult:
        """Test Generation 2 robustness features."""
        start_time = time.time()
        
        try:
            from robust_gen2_system import RobustFederatedAgent, RobustTrafficEnvironment
            
            # Create robust components
            env = RobustTrafficEnvironment(num_intersections=5)
            agents = [RobustFederatedAgent(i) for i in range(3)]
            
            # Test error handling
            error_scenarios = 0
            successful_handles = 0
            
            # Test 1: Invalid state handling
            try:
                invalid_state = None
                action = agents[0].select_action(invalid_state)
                if action == 0.0:  # Expected fallback
                    successful_handles += 1
                error_scenarios += 1
            except:
                error_scenarios += 1
            
            # Test 2: Extreme reward handling
            try:
                from robust_gen2_system import RobustGraphState
                state = RobustGraphState(
                    node_features=[[0.5] * 4] * 3,
                    edges=[(0, 1), (1, 2)]
                )
                extreme_reward = float('inf')
                agents[0].local_update(state, 0.5, extreme_reward)
                successful_handles += 1
                error_scenarios += 1
            except:
                error_scenarios += 1
            
            # Test 3: Health monitoring
            health_status = agents[0].health_monitor.get_health_status()
            if 'status' in health_status:
                successful_handles += 1
            error_scenarios += 1
            
            passed = successful_handles >= error_scenarios * 0.8  # 80% success rate
            
            return TestResult(
                test_name="generation2_robustness",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'error_scenarios_tested': error_scenarios,
                    'successful_handles': successful_handles,
                    'success_rate': successful_handles / max(1, error_scenarios)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="generation2_robustness",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def test_generation3_scalability(self) -> TestResult:
        """Test Generation 3 scalability features."""
        start_time = time.time()
        
        try:
            from scalable_gen3_system import ScalableAgent, CacheManager, LoadBalancer
            
            # Test caching
            cache = CacheManager(max_size=100, ttl=10.0)
            cache.put("test_key", "test_value")
            cached_value = cache.get("test_key")
            caching_works = cached_value == "test_value"
            
            # Test load balancing
            agents = [ScalableAgent(i) for i in range(5)]
            load_balancer = LoadBalancer(agents)
            
            selected_agents = []
            for _ in range(10):
                agent = load_balancer.select_agent("least_loaded")
                selected_agents.append(agent.agent_id)
            
            # Should distribute load
            unique_agents = len(set(selected_agents))
            load_balancing_works = unique_agents >= 3  # Should use multiple agents
            
            # Test resource pooling
            from scalable_gen3_system import ResourcePool
            pool = ResourcePool(lambda: [0.0] * 10, max_size=10)
            
            resource1 = pool.acquire()
            resource2 = pool.acquire()
            pool.release(resource1)
            resource3 = pool.acquire()
            
            pooling_works = resource3 is not None
            
            passed = caching_works and load_balancing_works and pooling_works
            
            return TestResult(
                test_name="generation3_scalability",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'caching_functional': caching_works,
                    'load_balancing_functional': load_balancing_works,
                    'resource_pooling_functional': pooling_works,
                    'unique_agents_selected': unique_agents
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="generation3_scalability",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error_message=str(e)
            )
    
    def run_all_functional_tests(self) -> List[TestResult]:
        """Run all functional tests."""
        logger.info("Running functional validation tests...")
        
        tests = [
            self.test_generation1_functionality(),
            self.test_generation2_robustness(),
            self.test_generation3_scalability()
        ]
        
        passed_tests = sum(1 for test in tests if test.passed)
        logger.info(f"Functional tests: {passed_tests}/{len(tests)} passed")
        
        return tests


class QualityGateRunner:
    """Main quality gate runner."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
        self.performance_validator = PerformanceValidator()
        self.functional_tester = FunctionalTester()
    
    def run_all_quality_gates(self) -> QualityReport:
        """Run comprehensive quality validation."""
        logger.info("🧪 Running comprehensive quality gates...")
        
        start_time = time.time()
        
        # Run all test suites
        security_tests = self.security_validator.run_all_security_tests()
        performance_tests = self.performance_validator.run_all_performance_tests()
        functional_tests = self.functional_tester.run_all_functional_tests()
        
        all_tests = security_tests + performance_tests + functional_tests
        
        # Calculate scores
        security_score = sum(1 for test in security_tests if test.passed) / len(security_tests)
        performance_score = sum(1 for test in performance_tests if test.passed) / len(performance_tests)
        coverage_score = sum(1 for test in functional_tests if test.passed) / len(functional_tests)
        
        overall_score = (security_score + performance_score + coverage_score) / 3.0
        
        # Create report
        report = QualityReport(
            test_results=all_tests,
            security_score=security_score,
            performance_score=performance_score,
            coverage_score=coverage_score,
            overall_score=overall_score,
            timestamp=time.time()
        )
        
        # Log summary
        total_tests = len(all_tests)
        passed_tests = sum(1 for test in all_tests if test.passed)
        duration = time.time() - start_time
        
        logger.info(f"Quality gates completed in {duration:.2f}s")
        logger.info(f"Overall results: {passed_tests}/{total_tests} tests passed")
        logger.info(f"Security score: {security_score:.1%}")
        logger.info(f"Performance score: {performance_score:.1%}")
        logger.info(f"Coverage score: {coverage_score:.1%}")
        logger.info(f"Overall quality score: {overall_score:.1%}")
        
        return report
    
    def save_report(self, report: QualityReport, filename: str = "quality_gates_report.json"):
        """Save quality report to file."""
        try:
            # Convert to serializable format
            report_dict = {
                "timestamp": report.timestamp,
                "overall_score": report.overall_score,
                "security_score": report.security_score,
                "performance_score": report.performance_score,
                "coverage_score": report.coverage_score,
                "test_results": [
                    {
                        "test_name": test.test_name,
                        "passed": test.passed,
                        "duration": test.duration,
                        "details": test.details,
                        "error_message": test.error_message
                    }
                    for test in report.test_results
                ]
            }
            
            with open(f"/root/repo/{filename}", 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"Quality report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")


def main():
    """Main quality gate execution."""
    print("🧪 Comprehensive Quality Gates - Autonomous SDLC Validation")
    print("=" * 70)
    
    runner = QualityGateRunner()
    report = runner.run_all_quality_gates()
    
    # Save report
    runner.save_report(report)
    
    # Print summary
    print(f"\n🎯 Quality Gate Results:")
    print(f"Overall Score: {report.overall_score:.1%}")
    print(f"Security: {report.security_score:.1%}")
    print(f"Performance: {report.performance_score:.1%}")
    print(f"Coverage: {report.coverage_score:.1%}")
    
    # Validate minimum thresholds
    if report.overall_score >= 0.85:  # 85% threshold
        print("✅ Quality gates PASSED")
        return 0
    else:
        print("❌ Quality gates FAILED")
        return 1


if __name__ == "__main__":
    exit(main())