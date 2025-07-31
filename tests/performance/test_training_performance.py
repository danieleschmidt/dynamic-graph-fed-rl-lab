"""Performance tests for training loops and core algorithms."""
import time
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow
]


class TestTrainingPerformance:
    """Test training performance and scalability."""

    @pytest.fixture
    def mock_environment(self):
        """Mock environment for performance testing."""
        env = Mock()
        env.reset.return_value = Mock()
        env.step.return_value = (Mock(), {"agent_0": 1.0}, False, {})
        env.max_steps = 1000
        return env

    @pytest.mark.timeout(60)
    def test_single_agent_training_speed(self, mock_environment):
        """Test single agent training throughput."""
        start_time = time.perf_counter()
        
        # Simulate training loop
        episodes = 100
        steps_per_episode = 100
        total_steps = 0
        
        for episode in range(episodes):
            state = mock_environment.reset()
            for step in range(steps_per_episode):
                action = np.random.random(4)  # Mock action
                next_state, reward, done, info = mock_environment.step({"agent_0": action})
                total_steps += 1
                
                if done:
                    break

        duration = time.perf_counter() - start_time
        steps_per_second = total_steps / duration
        
        # Performance assertion: should achieve at least 1000 steps/sec
        assert steps_per_second > 1000, f"Training too slow: {steps_per_second:.2f} steps/sec"
        
    def test_federated_aggregation_speed(self):
        """Test parameter aggregation performance with multiple agents."""
        num_agents = [5, 10, 20, 50, 100]
        param_size = 10000  # Size of parameter vector
        
        aggregation_times = []
        
        for n_agents in num_agents:
            # Generate mock parameters
            parameters = [np.random.random(param_size) for _ in range(n_agents)]
            
            start_time = time.perf_counter()
            
            # Simple aggregation (averaging)
            aggregated = np.mean(parameters, axis=0)
            
            duration = time.perf_counter() - start_time
            aggregation_times.append(duration)
            
            # Should complete aggregation within reasonable time
            assert duration < 0.1, f"Aggregation too slow for {n_agents} agents: {duration:.4f}s"
        
        # Verify linear scaling (approximately)
        assert aggregation_times[-1] < aggregation_times[0] * 10, "Non-linear scaling detected"

    def test_graph_encoding_performance(self):
        """Test graph neural network encoding performance."""
        graph_sizes = [100, 500, 1000, 2000]
        encoding_times = []
        
        for n_nodes in graph_sizes:
            # Mock graph data
            node_features = np.random.random((n_nodes, 64))
            edge_indices = np.random.randint(0, n_nodes, (2, n_nodes * 2))
            
            start_time = time.perf_counter()
            
            # Simulate graph encoding (matrix operations)
            adjacency = np.zeros((n_nodes, n_nodes))
            adjacency[edge_indices[0], edge_indices[1]] = 1
            encoded = np.dot(adjacency, node_features)
            
            duration = time.perf_counter() - start_time
            encoding_times.append(duration)
            
            # Should encode within reasonable time
            assert duration < 1.0, f"Graph encoding too slow for {n_nodes} nodes: {duration:.4f}s"

    @pytest.mark.memory_intensive
    def test_replay_buffer_memory_efficiency(self):
        """Test memory usage of replay buffer with large datasets."""
        buffer_size = 100000
        transition_size = 1024  # bytes per transition
        
        # Simulate buffer operations
        buffer_data = []
        
        start_memory = self._get_memory_usage()
        
        for i in range(buffer_size):
            # Add mock transition
            transition = np.random.bytes(transition_size)
            buffer_data.append(transition)
            
            # Simulate circular buffer
            if len(buffer_data) > buffer_size:
                buffer_data.pop(0)
        
        end_memory = self._get_memory_usage()
        memory_usage = end_memory - start_memory
        
        # Memory usage should be reasonable (< 200MB for this test)
        assert memory_usage < 200 * 1024 * 1024, f"Excessive memory usage: {memory_usage / 1024 / 1024:.2f}MB"
    
    def _get_memory_usage(self):
        """Get current memory usage in bytes."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss


@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for comparing performance across versions."""

    def test_algorithm_comparison(self, benchmark):
        """Benchmark different RL algorithms."""
        def training_step():
            # Mock training step
            state = np.random.random(64)
            action = np.random.random(4)
            reward = np.random.random()
            next_state = np.random.random(64)
            
            # Simulate Q-learning update
            q_value = np.dot(state, np.random.random(64))
            target = reward + 0.99 * np.max(np.dot(next_state, np.random.random(64)))
            loss = (q_value - target) ** 2
            
            return loss
        
        result = benchmark(training_step)
        
        # Benchmark should complete quickly
        assert result < 0.001, f"Training step benchmark failed: {result:.6f}s"

    def test_communication_overhead(self, benchmark):
        """Benchmark communication overhead in federated setup."""
        def gossip_round():
            # Simulate parameter exchange
            params = np.random.random(10000)
            
            # Simulate compression
            compressed = params[params > 0.5]  # Simple sparsification
            
            # Simulate decompression
            decompressed = np.zeros_like(params)
            decompressed[params > 0.5] = compressed
            
            return decompressed
        
        result = benchmark(gossip_round)
        
        # Communication should be efficient
        assert result < 0.01, f"Communication benchmark failed: {result:.6f}s"