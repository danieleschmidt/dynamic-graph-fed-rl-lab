"""
Tests for quantum advantage benchmarking system.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch

from dynamic_graph_fed_rl.quantum_hardware.benchmarking import (
    QuantumAdvantageBenchmark, BenchmarkConfig, BenchmarkType,
    ParameterAggregationBenchmark, GraphOptimizationBenchmark
)
from dynamic_graph_fed_rl.quantum_hardware.quantum_fed_learning import (
    QuantumFederatedLearning, QuantumFederatedConfig, QuantumAggregationStrategy
)
from dynamic_graph_fed_rl.quantum_hardware.hybrid_optimizer import (
    HybridClassicalQuantumOptimizer, HybridOptimizationConfig, OptimizationStrategy
)
from tests.quantum_hardware.test_quantum_backends import MockQuantumBackend


class TestParameterAggregationBenchmark:
    """Test parameter aggregation benchmarking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = ParameterAggregationBenchmark()
    
    def test_classical_methods(self):
        """Test classical aggregation methods."""
        client_params = [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([2.0, 3.0, 4.0]),
            jnp.array([1.5, 2.5, 3.5])
        ]
        
        weights = jnp.array([0.4, 0.3, 0.3])
        
        # Test FedAvg
        fedavg_result = self.benchmark._fedavg(client_params, weights)
        expected = jnp.average(jnp.array(client_params), axis=0, weights=weights)
        np.testing.assert_allclose(fedavg_result, expected)
        
        # Test weighted average
        weighted_result = self.benchmark._weighted_average(client_params, weights)
        np.testing.assert_allclose(weighted_result, expected)
        
        # Test median aggregation
        median_result = self.benchmark._median_aggregation(client_params)
        expected_median = jnp.median(jnp.array(client_params), axis=0)
        np.testing.assert_allclose(median_result, expected_median)
        
        # Test trimmed mean
        trimmed_result = self.benchmark._trimmed_mean(client_params)
        assert trimmed_result.shape == (3,)
    
    def test_aggregation_quality_evaluation(self):
        """Test aggregation quality evaluation."""
        client_params = [
            jnp.array([1.0, 2.0]),
            jnp.array([1.1, 2.1]),
            jnp.array([0.9, 1.9])
        ]
        
        # Good aggregation (close to clients)
        good_aggregation = jnp.array([1.0, 2.0])
        good_quality = self.benchmark._evaluate_aggregation_quality(good_aggregation, client_params)
        
        # Bad aggregation (far from clients)
        bad_aggregation = jnp.array([10.0, 20.0])
        bad_quality = self.benchmark._evaluate_aggregation_quality(bad_aggregation, client_params)
        
        assert good_quality > bad_quality
        assert 0 <= good_quality <= 1
        assert 0 <= bad_quality <= 1
    
    def test_benchmark_execution(self):
        """Test full benchmark execution."""
        # Create mock quantum federated learning
        backend = MockQuantumBackend()
        backend.connect({})
        
        config = QuantumFederatedConfig(
            aggregation_strategy=QuantumAggregationStrategy.QUANTUM_WEIGHTED_AVERAGE,
            num_qubits=4,
            circuit_depth=2,
            shots=100,
            optimization_iterations=3,
            quantum_advantage_threshold=0.1,
            noise_mitigation=False,
            error_correction=False
        )
        
        qfl = QuantumFederatedLearning({"mock": backend}, config)
        
        client_params = [
            jnp.array([1.0, 2.0]),
            jnp.array([2.0, 3.0]),
            jnp.array([1.5, 2.5])
        ]
        
        results = self.benchmark.run_benchmark(qfl, client_params, num_trials=3)
        
        # Check result structure
        assert "quantum_results" in results
        assert "classical_results" in results
        assert "quantum_summary" in results
        assert "problem_size" in results
        assert "num_clients" in results
        
        # Check quantum results
        assert len(results["quantum_results"]) == 3
        for result in results["quantum_results"]:
            assert "trial" in result
            assert "execution_time" in result
            assert "quality" in result
        
        # Check classical results
        assert len(results["classical_results"]) > 0
        for method_name in self.benchmark.classical_methods.keys():
            assert method_name in results["classical_results"]
            method_result = results["classical_results"][method_name]
            assert "avg_execution_time" in method_result
            assert "avg_quality" in method_result


class TestGraphOptimizationBenchmark:
    """Test graph optimization benchmarking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = GraphOptimizationBenchmark()
    
    def test_cut_evaluation(self):
        """Test cut value evaluation."""
        edges = [(0, 1), (1, 2), (2, 0)]
        
        # Perfect cut (all edges cut)
        perfect_partition = {0: 0, 1: 1, 2: 0}
        perfect_cut = self.benchmark._evaluate_cut(edges, perfect_partition)
        assert perfect_cut == 2  # Two edges cut: (0,1) and (1,2)
        
        # No cut
        no_cut_partition = {0: 0, 1: 0, 2: 0}
        no_cut = self.benchmark._evaluate_cut(edges, no_cut_partition)
        assert no_cut == 0
    
    def test_greedy_max_cut(self):
        """Test greedy Max-Cut algorithm."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        cut_value = self.benchmark._greedy_max_cut(edges)
        
        assert cut_value >= 0
        assert cut_value <= len(edges)
    
    def test_simulated_annealing(self):
        """Test simulated annealing Max-Cut algorithm."""
        edges = [(0, 1), (1, 2), (2, 0)]
        cut_value = self.benchmark._simulated_annealing_max_cut(edges)
        
        assert cut_value >= 0
        assert cut_value <= len(edges)
    
    def test_spectral_max_cut(self):
        """Test spectral Max-Cut algorithm."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        cut_value = self.benchmark._spectral_max_cut(edges)
        
        assert cut_value >= 0
        assert cut_value <= len(edges)
    
    def test_empty_graph(self):
        """Test algorithms on empty graph."""
        empty_edges = []
        
        assert self.benchmark._greedy_max_cut(empty_edges) == 0
        assert self.benchmark._simulated_annealing_max_cut(empty_edges) == 0
        assert self.benchmark._spectral_max_cut(empty_edges) == 0
    
    def test_benchmark_execution(self):
        """Test full graph optimization benchmark."""
        # Create mock hybrid optimizer
        backends = {"mock": MockQuantumBackend()}
        backends["mock"].connect({})
        
        optimizer_config = HybridOptimizationConfig(
            strategy=OptimizationStrategy.QAOA_CLASSICAL,
            max_classical_iterations=5,
            max_quantum_iterations=3,
            classical_learning_rate=0.1,
            quantum_learning_rate=0.1,
            convergence_threshold=1e-6,
            use_parameter_shift=True,
            noise_mitigation=False
        )
        
        optimizer = HybridClassicalQuantumOptimizer(backends, optimizer_config)
        
        # Create test graphs
        test_graphs = [
            [(0, 1), (1, 2)],  # Simple path
            [(0, 1), (1, 2), (2, 0)]  # Triangle
        ]
        
        results = self.benchmark.run_benchmark(optimizer, test_graphs, num_trials=2)
        
        # Check result structure
        assert "quantum_results" in results
        assert "classical_results" in results
        assert "graph_sizes" in results
        
        assert len(results["quantum_results"]) == len(test_graphs)
        assert len(results["graph_sizes"]) == len(test_graphs)


class TestQuantumAdvantageBenchmark:
    """Test main quantum advantage benchmark system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BenchmarkConfig(
            benchmark_type=BenchmarkType.PARAMETER_AGGREGATION,
            num_trials=3,
            problem_sizes=[4, 8],
            classical_baselines=["fedavg", "weighted_average"],
            quantum_algorithms=["quantum_weighted_average"],
            noise_levels=[0.0, 0.1],
            time_limit_seconds=10.0,
            quality_threshold=1.1,
            statistical_significance_level=0.05
        )
        
        self.benchmark = QuantumAdvantageBenchmark(self.config)
    
    def test_default_test_data_generation(self):
        """Test default test data generation."""
        # Parameter aggregation data
        param_config = BenchmarkConfig(
            benchmark_type=BenchmarkType.PARAMETER_AGGREGATION,
            num_trials=3,
            problem_sizes=[10],
            classical_baselines=[],
            quantum_algorithms=[],
            noise_levels=[],
            time_limit_seconds=1.0,
            quality_threshold=1.0,
            statistical_significance_level=0.05
        )
        
        param_benchmark = QuantumAdvantageBenchmark(param_config)
        param_data = param_benchmark._generate_default_test_data(10)
        
        assert "client_parameters" in param_data
        assert "client_weights" in param_data
        assert len(param_data["client_parameters"]) > 0
        assert param_data["client_parameters"][0].size == 10
        
        # Graph optimization data
        graph_config = BenchmarkConfig(
            benchmark_type=BenchmarkType.GRAPH_OPTIMIZATION,
            num_trials=3,
            problem_sizes=[5],
            classical_baselines=[],
            quantum_algorithms=[],
            noise_levels=[],
            time_limit_seconds=1.0,
            quality_threshold=1.0,
            statistical_significance_level=0.05
        )
        
        graph_benchmark = QuantumAdvantageBenchmark(graph_config)
        graph_data = graph_benchmark._generate_default_test_data(5)
        
        assert "graphs" in graph_data
        assert len(graph_data["graphs"]) > 0
    
    def test_quantum_advantage_metrics_computation(self):
        """Test quantum advantage metrics computation."""
        # Mock benchmark results
        mock_results = {
            "benchmark_results": [
                {
                    "problem_size": 4,
                    "parameter_aggregation": {
                        "quantum_summary": {
                            "avg_quality": 0.8,
                            "avg_execution_time": 0.1,
                            "success_rate": 1.0
                        },
                        "classical_results": {
                            "fedavg": {"avg_quality": 0.7, "avg_execution_time": 0.05},
                            "weighted_average": {"avg_quality": 0.6, "avg_execution_time": 0.03}
                        }
                    }
                }
            ]
        }
        
        metrics = self.benchmark._compute_quantum_advantage_metrics(mock_results)
        
        assert "overall_quantum_advantage" in metrics
        assert "advantage_by_problem_size" in metrics
        assert "average_advantage" in metrics
        
        # Check problem size specific metrics
        size_4_metrics = metrics["advantage_by_problem_size"][4]
        assert "quality_advantage" in size_4_metrics
        assert "time_advantage" in size_4_metrics
        assert "combined_advantage" in size_4_metrics
        
        # Quality advantage should be 0.8 / 0.7 > 1
        assert size_4_metrics["quality_advantage"] > 1.0
    
    def test_comprehensive_benchmark_execution(self):
        """Test comprehensive benchmark execution."""
        # Create minimal quantum federated learning system
        backend = MockQuantumBackend()
        backend.connect({})
        
        qfl_config = QuantumFederatedConfig(
            aggregation_strategy=QuantumAggregationStrategy.QUANTUM_WEIGHTED_AVERAGE,
            num_qubits=4,
            circuit_depth=1,
            shots=50,  # Minimal for testing
            optimization_iterations=2,
            quantum_advantage_threshold=0.1,
            noise_mitigation=False,
            error_correction=False
        )
        
        qfl = QuantumFederatedLearning({"mock": backend}, qfl_config)
        
        # Custom test data generator
        def test_data_generator(problem_size):
            return {
                "client_parameters": [
                    jnp.array(np.random.normal(0, 1, problem_size)) for _ in range(3)
                ],
                "client_weights": jnp.array([0.4, 0.3, 0.3])
            }
        
        results = self.benchmark.run_comprehensive_benchmark(
            quantum_fed_learning=qfl,
            test_data_generator=test_data_generator
        )
        
        # Check result structure
        assert "config" in results
        assert "benchmark_results" in results
        assert "quantum_advantage_metrics" in results
        
        # Check that we have results for each problem size
        assert len(results["benchmark_results"]) == len(self.config.problem_sizes)
        
        for size_result in results["benchmark_results"]:
            assert "problem_size" in size_result
            assert size_result["problem_size"] in self.config.problem_sizes
    
    def test_benchmark_report_generation(self):
        """Test benchmark report generation."""
        # Run a minimal benchmark first
        backend = MockQuantumBackend()
        backend.connect({})
        
        qfl_config = QuantumFederatedConfig(
            aggregation_strategy=QuantumAggregationStrategy.QUANTUM_WEIGHTED_AVERAGE,
            num_qubits=4,
            circuit_depth=1,
            shots=50,
            optimization_iterations=2,
            quantum_advantage_threshold=0.1,
            noise_mitigation=False,
            error_correction=False
        )
        
        qfl = QuantumFederatedLearning({"mock": backend}, qfl_config)
        
        # Simplified config for testing
        simple_config = BenchmarkConfig(
            benchmark_type=BenchmarkType.PARAMETER_AGGREGATION,
            num_trials=2,
            problem_sizes=[4],
            classical_baselines=["fedavg"],
            quantum_algorithms=["quantum_weighted_average"],
            noise_levels=[0.0],
            time_limit_seconds=5.0,
            quality_threshold=1.1,
            statistical_significance_level=0.05
        )
        
        simple_benchmark = QuantumAdvantageBenchmark(simple_config)
        
        results = simple_benchmark.run_comprehensive_benchmark(quantum_fed_learning=qfl)
        
        report = simple_benchmark.generate_benchmark_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "QUANTUM ADVANTAGE BENCHMARK REPORT" in report
        assert "PARAMETER_AGGREGATION" in report
        
    def test_results_export(self):
        """Test results export functionality."""
        import tempfile
        import os
        import json
        
        # Add some mock results
        self.benchmark.results_history = [
            {"test": "data", "timestamp": 123456}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            self.benchmark.export_results(temp_filename)
            
            # Verify file was created and contains expected data
            assert os.path.exists(temp_filename)
            
            with open(temp_filename, 'r') as f:
                exported_data = json.load(f)
            
            assert len(exported_data) == 1
            assert exported_data[0]["test"] == "data"
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)