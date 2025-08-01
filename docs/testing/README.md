# Testing Documentation

This directory contains comprehensive documentation for the testing framework and guidelines for the Dynamic Graph Fed-RL project.

## Testing Strategy

### Test Pyramid Structure

```
              ┌─────────────────┐
              │   E2E Tests     │ ← Few, high-level, complete workflows  
              │   (Expensive)   │
              └─────────────────┘
                       │
              ┌─────────────────┐
              │ Integration     │ ← Moderate, component interactions
              │ Tests           │
              └─────────────────┘
                       │
              ┌─────────────────┐
              │   Unit Tests    │ ← Many, fast, isolated components
              │   (Cheap)       │
              └─────────────────┘
```

### Test Categories

#### 1. Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (< 1s per test)
- **Coverage**: Aim for 90%+ code coverage
- **Scope**: Single functions, classes, or small modules

#### 2. Integration Tests (`tests/integration/`)
- **Purpose**: Test interaction between components
- **Speed**: Medium (1-10s per test)
- **Coverage**: Critical paths and component boundaries
- **Scope**: Multiple modules working together

#### 3. End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Speed**: Slow (10s-5min per test)
- **Coverage**: Critical user journeys
- **Scope**: Entire system functionality

#### 4. Performance Tests (`tests/benchmarks/`)
- **Purpose**: Measure performance and detect regressions
- **Speed**: Variable (can be very slow)
- **Coverage**: Performance-critical paths
- **Scope**: System performance under various conditions

#### 5. Load Tests (`tests/load/`)
- **Purpose**: Test system behavior under load
- **Speed**: Very slow (minutes to hours)
- **Coverage**: Scalability and reliability
- **Scope**: System limits and breaking points

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── fixtures/                   # Reusable test fixtures
│   ├── __init__.py
│   ├── graph_fixtures.py       # Graph-related test data
│   ├── environment_fixtures.py # Environment mocks
│   ├── federation_fixtures.py  # Federated learning setups
│   └── model_fixtures.py       # Model and agent mocks
├── unit/                       # Unit tests
│   ├── test_algorithms.py
│   ├── test_environments.py
│   ├── test_federation.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/                # Integration tests
│   ├── test_federated_training.py
│   ├── test_environment_integration.py
│   └── test_model_federation_integration.py
├── e2e/                        # End-to-end tests
│   ├── test_complete_pipeline.py
│   ├── test_deployment_scenarios.py
│   └── test_user_workflows.py
├── benchmarks/                 # Performance benchmarks
│   ├── test_performance_benchmarks.py
│   ├── test_scalability.py
│   └── test_memory_usage.py
├── load/                       # Load testing
│   ├── locustfile.py
│   ├── k6_scripts/
│   └── load_test_configs/
└── data/                       # Test data files
    ├── graphs/
    ├── scenarios/
    └── expected_outputs/
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
pytest tests/e2e/              # E2E tests only
pytest tests/benchmarks/       # Performance benchmarks

# Run with coverage
pytest --cov=src --cov-report=html

# Run with parallel execution
pytest -n auto

# Run specific test file
pytest tests/unit/test_algorithms.py

# Run specific test function
pytest tests/unit/test_algorithms.py::test_graph_td3_forward_pass

# Run tests matching pattern
pytest -k "federated"

# Run with verbose output
pytest -v

# Run with profiling
pytest --profile
```

### Advanced Test Options

```bash
# Run only fast tests (skip slow benchmarks)
pytest -m "not slow"

# Run only slow tests
pytest -m "slow"

# Run with different log levels
pytest --log-level=DEBUG

# Generate JUnit XML report
pytest --junitxml=test-results.xml

# Run with test discovery
pytest --collect-only

# Run failed tests from last run
pytest --lf

# Run with timeout
pytest --timeout=300

# Run with memory profiling
pytest --memprof
```

## Test Configuration

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-fail-under=80",
    "--strict-markers",
    "--disable-warnings",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "e2e: marks tests as end-to-end tests",
    "benchmark: marks tests as performance benchmarks",
    "gpu: marks tests that require GPU",
    "distributed: marks tests that require distributed setup",
]
```

### Test Markers

Use markers to categorize tests:

```python
import pytest

@pytest.mark.slow
def test_long_training():
    """Test that takes a long time."""
    pass

@pytest.mark.gpu
def test_gpu_acceleration():
    """Test that requires GPU."""
    pass

@pytest.mark.integration
def test_component_interaction():
    """Test component integration."""
    pass
```

## Writing Good Tests

### Test Naming Conventions

```python
def test_should_do_something_when_condition():
    """Test naming: test_should_<expected_behavior>_when_<condition>"""
    pass

def test_graph_td3_should_select_valid_actions_when_given_state():
    """Specific example with clear behavior description."""
    pass

class TestGraphTD3:
    """Test class for GraphTD3 algorithm."""
    
    def test_should_initialize_with_correct_parameters(self):
        pass
    
    def test_should_update_policy_when_training(self):
        pass
```

### Test Structure (AAA Pattern)

```python
def test_federated_aggregation():
    # Arrange - Set up test data and conditions
    num_agents = 5
    parameter_shape = (10, 10)
    agents = create_mock_agents(num_agents, parameter_shape)
    
    # Act - Execute the behavior being tested
    aggregated_params = federated_average(agents)
    
    # Assert - Verify the results
    assert aggregated_params.shape == parameter_shape
    assert not jnp.isnan(aggregated_params).any()
    assert jnp.allclose(aggregated_params, expected_average, rtol=1e-5)
```

### Fixture Usage

```python
def test_with_fixtures(small_graph, rng_key, federation_config):
    """Use fixtures for common test setup."""
    # Fixtures provide reusable test data
    env = create_environment(small_graph)
    fed_system = create_federation(federation_config)
    
    # Test logic here
    assert env.num_nodes == small_graph["num_nodes"]
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

def test_with_mocking():
    """Mock external dependencies for isolated testing."""
    # Mock external API
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {"status": "ok"}
        
        result = call_external_api()
        
        assert result["status"] == "ok"
        mock_get.assert_called_once()

def test_with_mock_objects():
    """Use mock objects for complex dependencies."""
    mock_env = Mock()
    mock_env.reset.return_value = {"nodes": jnp.ones((5, 4))}
    mock_env.step.return_value = (mock_env.reset.return_value, {"reward": 1.0}, False, {})
    
    agent = GraphTD3()
    result = agent.train_step(mock_env)
    
    assert mock_env.reset.called
    assert mock_env.step.called
```

## Test Data Management

### Using Fixtures for Test Data

```python
# In conftest.py or test files
@pytest.fixture
def traffic_network_data():
    """Realistic traffic network test data."""
    return {
        "nodes": create_traffic_nodes(num_intersections=25),
        "edges": create_road_network(grid_size=5),
        "demands": generate_traffic_demands(time_period="rush_hour")
    }

@pytest.fixture
def temporal_graph_sequence():
    """Sequence of graphs showing topology changes."""
    base_graph = create_base_graph()
    return create_temporal_sequence(base_graph, length=10)
```

### Test Data Files

```python
import json
import pickle
from pathlib import Path

def load_test_data(filename):
    """Load test data from files."""
    data_path = Path(__file__).parent / "data" / filename
    
    if filename.endswith('.json'):
        with open(data_path) as f:
            return json.load(f)
    elif filename.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

def test_with_file_data():
    """Use external test data files."""
    test_scenario = load_test_data("power_grid_scenario.json")
    expected_output = load_test_data("expected_power_flow.pkl")
    
    result = simulate_power_flow(test_scenario)
    
    assert jnp.allclose(result, expected_output, rtol=1e-4)
```

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run unit tests
      run: pytest tests/unit/ -v
    
    - name: Run integration tests
      run: pytest tests/integration/ -v
    
    - name: Run E2E tests
      run: pytest tests/e2e/ -v -m "not slow"
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Testing

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [tests/unit/, -x, --tb=short]
```

## Performance Testing

### Benchmark Guidelines

```python
import pytest
import time
from memory_profiler import profile

class TestPerformance:
    @pytest.mark.benchmark
    def test_training_speed_benchmark(self, benchmark):
        """Benchmark training iteration speed."""
        def training_iteration():
            # Setup
            agent = create_test_agent()
            batch = create_test_batch()
            
            # Training step
            return agent.train_step(batch)
        
        # Run benchmark
        result = benchmark(training_iteration)
        
        # Assertions about performance
        assert benchmark.stats.mean < 0.1  # Should complete in < 100ms
        assert result is not None

    @profile
    def test_memory_usage(self):
        """Profile memory usage during training."""
        agent = create_large_agent()
        
        for i in range(100):
            batch = create_test_batch()
            agent.train_step(batch)
            
            if i % 10 == 0:
                # Force garbage collection
                import gc
                gc.collect()
```

### Load Testing with Locust

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class FederatedLearningUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup user session."""
        self.client.post("/api/auth/login", {
            "username": "test_user",
            "password": "test_pass"
        })
    
    @task(3)
    def submit_training_batch(self):
        """Submit training batch to server."""
        batch_data = generate_test_batch()
        self.client.post("/api/training/batch", json=batch_data)
    
    @task(1)
    def get_model_parameters(self):
        """Retrieve current model parameters."""
        self.client.get("/api/model/parameters")
```

## Test Reporting

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=src --cov-report=xml

# Show missing lines
pytest --cov=src --cov-report=term-missing
```

### Test Results

```bash
# Generate JUnit XML for CI integration
pytest --junitxml=test-results.xml

# Generate test report with timing
pytest --durations=10

# Generate detailed test report
pytest --tb=long -v
```

## Debugging Tests

### Interactive Debugging

```python
def test_with_debugging():
    """Example of debugging test."""
    import pdb; pdb.set_trace()  # Breakpoint
    
    # Test code here
    result = complex_function()
    
    # Another breakpoint
    import ipdb; ipdb.set_trace()  # Enhanced debugger
    
    assert result == expected_value
```

### Logging in Tests

```python
import logging

def test_with_logging(caplog):
    """Capture and test log output."""
    with caplog.at_level(logging.INFO):
        function_that_logs()
    
    assert "Expected log message" in caplog.text
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO"
```

## Best Practices

### DO:
- ✅ Write tests first (TDD approach)
- ✅ Use descriptive test names
- ✅ Keep tests simple and focused
- ✅ Use fixtures for common setup
- ✅ Mock external dependencies
- ✅ Test edge cases and error conditions
- ✅ Maintain high test coverage (>80%)
- ✅ Run tests frequently during development
- ✅ Use appropriate test categories (unit/integration/e2e)
- ✅ Document complex test scenarios

### DON'T:
- ❌ Write overly complex tests
- ❌ Test implementation details
- ❌ Use sleep() for timing (use proper synchronization)
- ❌ Ignore failing tests
- ❌ Test multiple things in one test
- ❌ Use hard-coded paths or data
- ❌ Skip error handling in tests
- ❌ Write tests that depend on external services
- ❌ Use production data in tests
- ❌ Commit commented-out test code

## Getting Help

- **Documentation**: Check this testing guide first
- **Code Examples**: Look at existing tests for patterns
- **pytest Documentation**: https://docs.pytest.org/
- **JAX Testing**: https://jax.readthedocs.io/en/latest/debugging/
- **Team Chat**: Ask in the #testing channel
- **Code Review**: Request test review in PRs

Happy testing! 🚀