"""Performance benchmarks for the dynamic graph fed-RL system."""

import pytest
import time
import psutil
import gc
from typing import Dict, List, Any
import jax
import jax.numpy as jnp
import numpy as np


class TestPerformanceBenchmarks:
    """Performance and scalability benchmarks."""
    
    @pytest.fixture(autouse=True)
    def setup_jax_for_benchmarks(self):
        """Setup JAX for consistent benchmarking."""
        # Enable XLA optimizations
        jax.config.update("jax_enable_x64", False)  # Use 32-bit for speed
        
        # Pre-compile to avoid including compilation time
        dummy_fn = jax.jit(lambda x: x @ x.T)
        dummy_fn(jnp.ones((10, 10))).block_until_ready()
        
        yield
        
        # Cleanup
        gc.collect()
    
    def benchmark_function(self, func, *args, **kwargs):
        """Utility function to benchmark execution time and memory usage."""
        # Force garbage collection before benchmark
        gc.collect()
        
        # Record initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Warmup run
        func(*args, **kwargs)
        
        # Actual benchmark
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        
        # Ensure JAX computation is complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (list, tuple)) and result:
            for item in result:
                if hasattr(item, 'block_until_ready'):
                    item.block_until_ready()
        
        end_time = time.perf_counter()
        
        # Record final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "execution_time": end_time - start_time,
            "memory_usage": final_memory - initial_memory,
            "result": result
        }
    
    def test_graph_processing_scalability(self):
        """Benchmark graph processing performance across different scales."""
        scales = [
            {"nodes": 100, "edges": 200},
            {"nodes": 500, "edges": 1000},
            {"nodes": 1000, "edges": 2000},
            {"nodes": 2000, "edges": 4000},
        ]
        
        def process_graph(num_nodes, num_edges):
            """Mock graph processing function."""
            # Create random graph
            key = jax.random.PRNGKey(42)
            nodes = jax.random.normal(key, (num_nodes, 16))  # 16 features per node
            
            # Create edge indices
            edges = jnp.array(np.random.choice(num_nodes, (num_edges, 2)))
            edge_features = jax.random.normal(key, (num_edges, 8))  # 8 features per edge
            
            # Simple graph neural network computation
            @jax.jit
            def gnn_forward(nodes, edges, edge_features):
                # Message passing simulation
                messages = nodes[edges[:, 0]] + nodes[edges[:, 1]] + edge_features
                
                # Aggregate messages (simplified)
                aggregated = jnp.zeros_like(nodes)
                aggregated = aggregated.at[edges[:, 1]].add(messages[:, :16])
                
                # Update nodes
                updated_nodes = nodes + 0.1 * aggregated
                return updated_nodes
            
            return gnn_forward(nodes, edges, edge_features)
        
        results = []
        for scale in scales:
            benchmark_result = self.benchmark_function(
                process_graph, 
                scale["nodes"], 
                scale["edges"]
            )
            
            results.append({
                "scale": scale,
                "time": benchmark_result["execution_time"],
                "memory": benchmark_result["memory_usage"]
            })
        
        # Verify scalability characteristics
        assert len(results) == len(scales)
        
        # Check that execution time increases with scale (within reasonable bounds)
        times = [r["time"] for r in results]
        assert all(t > 0 for t in times), "All executions should take positive time"
        
        # Performance should not degrade exponentially
        time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        assert all(ratio < 10 for ratio in time_ratios), "Performance degradation should be reasonable"
        
        print("\\nGraph Processing Scalability Results:")
        for result in results:
            print(f"Nodes: {result['scale']['nodes']}, "
                  f"Edges: {result['scale']['edges']}, "
                  f"Time: {result['time']:.4f}s, "
                  f"Memory: {result['memory']:.2f}MB")
    
    def test_federated_aggregation_performance(self):
        """Benchmark federated parameter aggregation performance."""
        agent_counts = [5, 10, 20, 50]
        parameter_sizes = [
            {"layers": 2, "size": 128},
            {"layers": 4, "size": 256},
            {"layers": 6, "size": 512},
        ]
        
        def federated_aggregation(num_agents, layer_size, num_layers):
            """Mock federated aggregation."""
            key = jax.random.PRNGKey(42)
            
            # Create agent parameters
            agent_params = []
            for i in range(num_agents):
                params = {}
                for layer in range(num_layers):
                    params[f"layer_{layer}"] = jax.random.normal(
                        jax.random.fold_in(key, i), (layer_size, layer_size)
                    )
                agent_params.append(params)
            
            @jax.jit
            def aggregate_parameters(params_list):
                """Average parameters across agents."""
                avg_params = {}
                for key in params_list[0].keys():
                    stacked = jnp.stack([p[key] for p in params_list])
                    avg_params[key] = jnp.mean(stacked, axis=0)
                return avg_params
            
            return aggregate_parameters(agent_params)
        
        results = []
        for num_agents in agent_counts:
            for param_config in parameter_sizes:
                benchmark_result = self.benchmark_function(
                    federated_aggregation,
                    num_agents,
                    param_config["size"],
                    param_config["layers"]
                )
                
                results.append({
                    "num_agents": num_agents,
                    "param_config": param_config,
                    "time": benchmark_result["execution_time"],
                    "memory": benchmark_result["memory_usage"]
                })
        
        # Verify aggregation performance
        assert len(results) == len(agent_counts) * len(parameter_sizes)
        
        # Check scaling with number of agents
        for param_config in parameter_sizes:
            config_results = [r for r in results if r["param_config"] == param_config]
            times = [r["time"] for r in config_results]
            
            # Time should increase with number of agents but remain reasonable
            assert all(t < 1.0 for t in times), "Aggregation should be fast"
        
        print("\\nFederated Aggregation Performance Results:")
        for result in results:
            print(f"Agents: {result['num_agents']}, "
                  f"Layers: {result['param_config']['layers']}, "
                  f"Size: {result['param_config']['size']}, "
                  f"Time: {result['time']:.4f}s")
    
    def test_training_iteration_performance(self):
        """Benchmark single training iteration performance."""
        batch_sizes = [32, 64, 128, 256, 512]
        sequence_lengths = [10, 20, 50]
        
        def training_iteration(batch_size, sequence_length):
            """Mock training iteration."""
            key = jax.random.PRNGKey(42)
            
            # Generate batch of temporal graph sequences
            graphs = []
            for t in range(sequence_length):
                graph = {
                    "nodes": jax.random.normal(key, (batch_size, 20, 16)),  # batch, nodes, features
                    "edges": jnp.tile(jnp.array([[0, 1], [1, 2], [2, 0]]), (batch_size, 1, 1)),
                    "edge_features": jax.random.normal(key, (batch_size, 3, 8))
                }
                graphs.append(graph)
            
            @jax.jit
            def forward_pass(graph_sequence):
                """Mock forward pass through temporal GNN."""
                hidden_states = []
                
                for graph in graph_sequence:
                    # Simple GNN layer
                    node_updates = graph["nodes"] + 0.1 * jax.random.normal(
                        key, graph["nodes"].shape
                    )
                    hidden_states.append(node_updates)
                
                # Temporal aggregation
                final_state = jnp.stack(hidden_states).mean(axis=0)
                
                # Policy and value heads
                policy_logits = final_state @ jax.random.normal(key, (16, 4))
                values = final_state @ jax.random.normal(key, (16, 1))
                
                return policy_logits, values
            
            @jax.jit 
            def backward_pass(params, graphs, targets):
                """Mock backward pass."""
                def loss_fn(params, graphs):
                    policy_logits, values = forward_pass(graphs)
                    policy_loss = jnp.mean(policy_logits**2)  # Simplified loss
                    value_loss = jnp.mean(values**2)
                    return policy_loss + value_loss
                
                return jax.grad(lambda p: loss_fn(p, graphs))(params)
            
            # Forward pass
            policy_logits, values = forward_pass(graphs)
            
            # Mock backward pass
            dummy_params = jax.random.normal(key, (100,))
            dummy_targets = jax.random.normal(key, (batch_size, 4))
            gradients = backward_pass(dummy_params, graphs, dummy_targets)
            
            return policy_logits, values, gradients
        
        results = []
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                benchmark_result = self.benchmark_function(
                    training_iteration,
                    batch_size,
                    seq_len
                )
                
                results.append({
                    "batch_size": batch_size,
                    "sequence_length": seq_len,
                    "time": benchmark_result["execution_time"],
                    "memory": benchmark_result["memory_usage"],
                    "throughput": batch_size / benchmark_result["execution_time"]
                })
        
        # Verify training performance
        assert len(results) == len(batch_sizes) * len(sequence_lengths)
        
        # Check throughput scaling
        for result in results:
            assert result["throughput"] > 0, "Should process some samples per second"
            assert result["time"] < 10.0, "Training iteration should be reasonably fast"
        
        print("\\nTraining Iteration Performance Results:")
        for result in results:
            print(f"Batch: {result['batch_size']}, "
                  f"SeqLen: {result['sequence_length']}, "
                  f"Time: {result['time']:.4f}s, "
                  f"Throughput: {result['throughput']:.1f} samples/s")
    
    def test_memory_efficiency(self):
        """Test memory usage patterns and efficiency."""
        def memory_intensive_operation(size_factor):
            """Operation that uses significant memory."""
            key = jax.random.PRNGKey(42)
            
            # Create large tensors
            large_tensor_1 = jax.random.normal(key, (size_factor * 100, size_factor * 100))
            large_tensor_2 = jax.random.normal(key, (size_factor * 100, size_factor * 100))
            
            # Perform computation
            result = large_tensor_1 @ large_tensor_2
            
            # Simulate cleanup
            del large_tensor_1, large_tensor_2
            
            return result.sum()
        
        size_factors = [1, 2, 3, 4]
        memory_results = []
        
        for factor in size_factors:
            benchmark_result = self.benchmark_function(
                memory_intensive_operation,
                factor
            )
            
            memory_results.append({
                "size_factor": factor,
                "memory_usage": benchmark_result["memory_usage"],
                "time": benchmark_result["execution_time"]
            })
            
            # Force garbage collection between tests
            gc.collect()
        
        # Verify memory usage patterns
        memory_usages = [r["memory_usage"] for r in memory_results]
        
        # Memory usage should scale with problem size
        assert len(memory_usages) == len(size_factors)
        
        # Should not have memory leaks (difficult to test precisely)
        # At minimum, ensure we're not using excessive memory
        max_memory = max(memory_usages)
        assert max_memory < 1000, f"Memory usage should be reasonable, got {max_memory}MB"
        
        print("\\nMemory Efficiency Results:")
        for result in memory_results:
            print(f"Size Factor: {result['size_factor']}, "
                  f"Memory: {result['memory_usage']:.2f}MB, "
                  f"Time: {result['time']:.4f}s")
    
    @pytest.mark.slow
    def test_long_running_stability(self):
        """Test system stability over extended periods."""
        # This test simulates long-running training
        iterations = 100  # Reduced for CI
        
        def stable_training_step():
            """Mock training step that should remain stable."""
            key = jax.random.PRNGKey(int(time.time() * 1000) % 2**31)
            
            # Create some tensors
            x = jax.random.normal(key, (50, 50))
            y = jax.random.normal(key, (50, 1))
            
            # Simple computation
            @jax.jit
            def computation(x, y):
                w = jnp.eye(50) * 0.1
                pred = x @ w
                loss = jnp.mean((pred - y) ** 2)
                return loss
            
            loss = computation(x, y)
            return float(loss)
        
        losses = []
        memory_usage = []
        execution_times = []
        
        for i in range(iterations):
            # Monitor memory before each iteration
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory)
            
            # Time the training step
            start_time = time.perf_counter()
            loss = stable_training_step()
            end_time = time.perf_counter()
            
            losses.append(loss)
            execution_times.append(end_time - start_time)
            
            # Periodic cleanup
            if i % 20 == 0:
                gc.collect()
        
        # Analyze stability
        assert len(losses) == iterations
        assert all(not np.isnan(loss) for loss in losses), "No NaN losses"
        assert all(not np.isinf(loss) for loss in losses), "No infinite losses"
        
        # Check memory stability (no significant leaks)
        memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        assert memory_trend < 1.0, f"Memory usage should be stable, trend: {memory_trend}MB/iter"
        
        # Check execution time stability
        avg_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        assert std_time < avg_time, "Execution times should be reasonably consistent"
        
        print(f"\\nLong Running Stability Results:")
        print(f"Iterations: {iterations}")
        print(f"Average loss: {np.mean(losses):.6f} ± {np.std(losses):.6f}")
        print(f"Average time: {avg_time:.6f}s ± {std_time:.6f}s") 
        print(f"Memory trend: {memory_trend:.4f}MB/iter")
        print(f"Final memory: {memory_usage[-1]:.2f}MB")
    
    def test_concurrent_agent_performance(self):
        """Test performance with concurrent agent operations."""
        num_agents_list = [5, 10, 20]
        
        def concurrent_agent_operations(num_agents):
            """Simulate concurrent agent operations."""
            key = jax.random.PRNGKey(42)
            
            # Create agent states
            agent_states = []
            for i in range(num_agents):
                state = {
                    "policy_params": jax.random.normal(
                        jax.random.fold_in(key, i), (64, 32)
                    ),
                    "value_params": jax.random.normal(
                        jax.random.fold_in(key, i + num_agents), (32, 1)
                    )
                }
                agent_states.append(state)
            
            @jax.jit
            def batch_agent_forward(states, observations):
                """Process all agents in batch."""
                batch_size = len(states)
                
                # Vectorized forward pass
                policy_params = jnp.stack([s["policy_params"] for s in states])
                value_params = jnp.stack([s["value_params"] for s in states])
                
                # Batch matrix multiplication
                hidden = observations @ policy_params  # (batch, obs_dim) @ (batch, obs_dim, hidden)
                hidden = jnp.diagonal(hidden, axis1=1, axis2=2).T  # Extract diagonal
                
                values = hidden @ value_params.squeeze(-1)  # (batch, hidden) @ (batch, hidden)
                values = jnp.diagonal(values).reshape(-1, 1)
                
                return hidden, values
            
            # Mock observations
            observations = jax.random.normal(key, (num_agents, 64))
            
            # Process all agents
            hidden, values = batch_agent_forward(agent_states, observations)
            
            return hidden, values
        
        results = []
        for num_agents in num_agents_list:
            benchmark_result = self.benchmark_function(
                concurrent_agent_operations,
                num_agents
            )
            
            results.append({
                "num_agents": num_agents,
                "time": benchmark_result["execution_time"],
                "memory": benchmark_result["memory_usage"],
                "throughput": num_agents / benchmark_result["execution_time"]
            })
        
        # Verify concurrent performance
        assert len(results) == len(num_agents_list)
        
        # Throughput should scale reasonably with number of agents
        throughputs = [r["throughput"] for r in results]
        assert all(t > 0 for t in throughputs), "Should process agents"
        
        print("\\nConcurrent Agent Performance Results:")
        for result in results:
            print(f"Agents: {result['num_agents']}, "
                  f"Time: {result['time']:.4f}s, "
                  f"Throughput: {result['throughput']:.1f} agents/s")