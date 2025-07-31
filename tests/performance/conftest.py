"""Performance test configuration and fixtures."""
import pytest
import numpy as np


def pytest_configure(config):
    """Configure performance test markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: mark test as memory intensive"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark"
    )


@pytest.fixture(scope="session")
def performance_config():
    """Configuration for performance tests."""
    return {
        "max_training_time": 60,  # seconds
        "min_steps_per_second": 1000,
        "max_memory_usage_mb": 500,
        "max_aggregation_time": 0.1,  # seconds
    }


@pytest.fixture
def mock_large_graph():
    """Generate a large graph for performance testing."""
    n_nodes = 1000
    n_edges = 5000
    
    # Generate random graph
    node_features = np.random.random((n_nodes, 64))
    edge_indices = np.random.randint(0, n_nodes, (2, n_edges))
    edge_features = np.random.random((n_edges, 16))
    
    return {
        "nodes": node_features,
        "edges": edge_indices,
        "edge_attr": edge_features,
        "num_nodes": n_nodes,
        "num_edges": n_edges,
    }


@pytest.fixture
def mock_federated_setup():
    """Mock federated learning setup for testing."""
    num_agents = 10
    param_size = 50000
    
    agents = []
    for i in range(num_agents):
        agent = {
            "id": i,
            "parameters": np.random.random(param_size),
            "gradients": np.random.random(param_size),
            "learning_rate": 0.001,
        }
        agents.append(agent)
    
    return agents


class PerformanceReporter:
    """Helper class for reporting performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, name, value, unit="seconds"):
        """Record a performance metric."""
        self.metrics[name] = {"value": value, "unit": unit}
    
    def report(self):
        """Generate performance report."""
        report = "Performance Test Results:\n"
        report += "=" * 50 + "\n"
        
        for name, metric in self.metrics.items():
            report += f"{name}: {metric['value']:.6f} {metric['unit']}\n"
        
        return report


@pytest.fixture
def performance_reporter():
    """Performance metrics reporter."""
    return PerformanceReporter()