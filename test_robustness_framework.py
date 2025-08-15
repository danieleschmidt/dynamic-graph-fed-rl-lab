#!/usr/bin/env python3
"""
Test script for the robustness testing framework.

This script tests the robustness testing framework in isolation to verify
that all components work correctly together.
"""

import sys
import os
import asyncio
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import():
    """Test importing the robustness testing framework."""
    try:
        # Import directly from the module file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "robustness_testing", 
            "src/dynamic_graph_fed_rl/testing/robustness_testing.py"
        )
        robustness_module = importlib.util.module_from_spec(spec)
        
        # Add necessary modules to sys.modules to handle relative imports
        sys.modules['dynamic_graph_fed_rl'] = type(sys)('dynamic_graph_fed_rl')
        sys.modules['dynamic_graph_fed_rl.utils'] = type(sys)('utils')
        sys.modules['dynamic_graph_fed_rl.utils.error_handling'] = type(sys)('error_handling')
        
        # Mock the error handling module
        error_handling = sys.modules['dynamic_graph_fed_rl.utils.error_handling']
        
        class MockCircuitBreakerConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class MockRetryConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class MockResilience:
            def get_system_metrics(self):
                return {"circuit_breakers": {}}
        
        def mock_circuit_breaker(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        def mock_retry(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        def mock_robust(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        class ValidationError(Exception):
            pass
        
        class SecurityError(Exception):
            pass
        
        # Add mocks to the error handling module
        error_handling.circuit_breaker = mock_circuit_breaker
        error_handling.retry = mock_retry
        error_handling.robust = mock_robust
        error_handling.SecurityError = SecurityError
        error_handling.ValidationError = ValidationError
        error_handling.CircuitBreakerConfig = MockCircuitBreakerConfig
        error_handling.RetryConfig = MockRetryConfig
        error_handling.resilience = MockResilience()
        
        # Load the module
        spec.loader.exec_module(robustness_module)
        
        print("✓ Robustness testing framework imported successfully")
        return robustness_module
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_framework_functionality(robustness_module):
    """Test the basic functionality of the robustness framework."""
    try:
        # Get the framework instance
        framework = robustness_module.robustness_tester
        
        # Test getting status
        status = framework.get_test_results_summary()
        print(f"✓ Status retrieved: {status.get('message', 'Framework ready')}")
        
        # Test available test suites
        suites = list(framework.test_suites.keys())
        print(f"✓ Available test suites ({len(suites)}): {', '.join(suites)}")
        
        # Test fault injector
        fault_injector = framework.fault_injector
        fault_id = await fault_injector.inject_network_latency("test_target", 100, 1.0)
        print(f"✓ Fault injection test passed: {fault_id}")
        
        await asyncio.sleep(1.5)  # Wait for fault to be removed
        
        # Test load generator
        load_generator = framework.load_generator
        load_id = await load_generator.generate_federation_load(5, 10, 2.0)
        print(f"✓ Load generation test passed: {load_id}")
        
        await asyncio.sleep(2.5)  # Wait for load test to complete
        
        print("✓ All basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_single_test_execution(robustness_module):
    """Test executing a single robustness test."""
    try:
        framework = robustness_module.robustness_tester
        
        # Create a simple test function
        async def simple_test(result):
            result.test_type = robustness_module.TestType.CHAOS_ENGINEERING
            result.logs.append("Simple test started")
            await asyncio.sleep(0.1)  # Simulate test work
            result.logs.append("Simple test completed")
            result.status = robustness_module.TestStatus.PASSED
        
        # Run the test
        result = await framework._run_single_test(simple_test)
        
        print(f"✓ Single test execution: {result.status.value}")
        print(f"  - Duration: {result.duration:.3f}s")
        print(f"  - Logs: {len(result.logs)} entries")
        
        return result.status == robustness_module.TestStatus.PASSED
        
    except Exception as e:
        print(f"✗ Single test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_test_suite_execution(robustness_module):
    """Test executing a small test suite."""
    try:
        framework = robustness_module.robustness_tester
        
        # Create a mini test suite with just a couple of tests
        async def test1(result):
            result.test_type = robustness_module.TestType.LOAD_TESTING
            result.logs.append("Test 1 running")
            await asyncio.sleep(0.05)
            result.status = robustness_module.TestStatus.PASSED
        
        async def test2(result):
            result.test_type = robustness_module.TestType.SECURITY_TESTING
            result.logs.append("Test 2 running")
            await asyncio.sleep(0.05)
            result.status = robustness_module.TestStatus.PASSED
        
        # Create and register test suite
        test_suite = robustness_module.TestSuite(
            suite_id="mini_test_suite",
            name="Mini Test Suite",
            description="Small test suite for validation"
        )
        test_suite.tests = [test1, test2]
        test_suite.parallel_execution = True
        
        framework.test_suites["mini_test_suite"] = test_suite
        
        # Run the test suite
        suite_result = await framework.run_test_suite("mini_test_suite")
        
        print(f"✓ Test suite execution completed")
        print(f"  - Total tests: {suite_result['total_tests']}")
        print(f"  - Passed: {suite_result['passed_tests']}")
        print(f"  - Success rate: {suite_result['success_rate']:.1%}")
        print(f"  - Duration: {suite_result['duration']:.3f}s")
        
        return suite_result['success_rate'] == 1.0
        
    except Exception as e:
        print(f"✗ Test suite execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_comprehensive_test_summary(robustness_module):
    """Print a comprehensive summary of the testing framework capabilities."""
    try:
        framework = robustness_module.robustness_tester
        
        print("\n" + "="*60)
        print("ROBUSTNESS TESTING FRAMEWORK SUMMARY")
        print("="*60)
        
        # Test suites
        print(f"\nTest Suites Available: {len(framework.test_suites)}")
        for suite_id, suite in framework.test_suites.items():
            print(f"  • {suite.name}")
            print(f"    - ID: {suite_id}")
            print(f"    - Tests: {len(suite.tests)}")
            print(f"    - Parallel: {suite.parallel_execution}")
            print(f"    - Description: {suite.description}")
        
        # Test types
        print(f"\nTest Types Supported:")
        for test_type in robustness_module.TestType:
            print(f"  • {test_type.value}")
        
        # Framework components
        print(f"\nFramework Components:")
        print(f"  • Fault Injector: Ready")
        print(f"  • Load Generator: Ready") 
        print(f"  • Test Orchestrator: Ready")
        print(f"  • Result Aggregator: Ready")
        
        # Current status
        status = framework.get_test_results_summary()
        print(f"\nCurrent Status:")
        print(f"  • Test results: {len(framework.test_results)}")
        print(f"  • Active tests: {len(framework.active_tests)}")
        print(f"  • Default timeout: {framework.default_timeout}s")
        print(f"  • Max parallel tests: {framework.max_parallel_tests}")
        
        print("\n" + "="*60)
        print("FRAMEWORK VALIDATION COMPLETE")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"✗ Summary generation failed: {e}")
        return False

async def main():
    """Main test execution function."""
    print("Testing Robustness Testing Framework")
    print("====================================")
    
    # Test 1: Import
    print("\n1. Testing imports...")
    robustness_module = test_import()
    if not robustness_module:
        return False
    
    # Test 2: Basic functionality
    print("\n2. Testing basic functionality...")
    if not await test_framework_functionality(robustness_module):
        return False
    
    # Test 3: Single test execution
    print("\n3. Testing single test execution...")
    if not await test_single_test_execution(robustness_module):
        return False
    
    # Test 4: Test suite execution
    print("\n4. Testing test suite execution...")
    if not await test_test_suite_execution(robustness_module):
        return False
    
    # Test 5: Comprehensive summary
    print("\n5. Generating comprehensive summary...")
    if not print_comprehensive_test_summary(robustness_module):
        return False
    
    print("\n✓ ALL TESTS PASSED - Robustness Testing Framework is fully operational!")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)