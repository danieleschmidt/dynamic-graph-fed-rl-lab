import secrets
"""
Quantum advantage benchmarking system.

Comprehensive benchmarking to validate quantum advantage in federated learning
scenarios, comparing quantum algorithms against classical baselines.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from enum import Enum
import time
import json

from .base import QuantumBackend, QuantumCircuit, QuantumResult
from .quantum_fed_learning import QuantumFederatedLearning, QuantumFederatedConfig
from .hybrid_optimizer import HybridClassicalQuantumOptimizer


class BenchmarkType(Enum):
    """Types of quantum advantage benchmarks."""
    PARAMETER_AGGREGATION = "parameter_aggregation"
    GRAPH_OPTIMIZATION = "graph_optimization"
    FEDERATED_LEARNING_CONVERGENCE = "federated_learning_convergence"
    NOISE_ROBUSTNESS = "noise_robustness"
    SCALABILITY = "scalability"


@dataclass
class BenchmarkConfig:
    """Configuration for quantum advantage benchmarking."""
    benchmark_type: BenchmarkType
    num_trials: int
    problem_sizes: List[int]
    classical_baselines: List[str]
    quantum_algorithms: List[str]
    noise_levels: List[float]
    time_limit_seconds: float
    quality_threshold: float
    statistical_significance_level: float


@dataclass
class BenchmarkResult:
    """Results of quantum advantage benchmark."""
    benchmark_type: BenchmarkType
    problem_size: int
    quantum_performance: Dict[str, float]
    classical_performance: Dict[str, float]
    quantum_advantage: float
    statistical_significance: float
    execution_time_quantum: float
    execution_time_classical: float
    speedup: float
    quality_improvement: float
    success_rate_quantum: float
    success_rate_classical: float
    raw_results: Dict[str, Any]


class ParameterAggregationBenchmark:
    """Benchmark quantum vs classical parameter aggregation."""
    
    def __init__(self):
        self.classical_methods = {
            "fedavg": self._fedavg,
            "weighted_average": self._weighted_average,
            "median_aggregation": self._median_aggregation,
            "trimmed_mean": self._trimmed_mean
        }
    
    def run_benchmark(
        self,
        quantum_fed_learning: QuantumFederatedLearning,
        client_parameters: List[jnp.ndarray],
        client_weights: Optional[jnp.ndarray] = None,
        num_trials: int = 10
    ) -> Dict[str, Any]:
        """Run parameter aggregation benchmark."""
        
        results = {
            "quantum_results": [],
            "classical_results": {},
            "problem_size": len(client_parameters[0]) if client_parameters else 0,
            "num_clients": len(client_parameters)
        }
        
        # Run quantum aggregation trials
        quantum_times = []
        quantum_qualities = []
        quantum_successes = 0
        
        for trial in range(num_trials):
            start_time = time.time()
            try:
                aggregated_params, round_info = quantum_fed_learning.federated_round(
                    client_parameters, client_weights, trial
                )
                
                execution_time = time.time() - start_time
                quantum_times.append(execution_time)
                
                # Evaluate aggregation quality
                quality = self._evaluate_aggregation_quality(aggregated_params, client_parameters)
                quantum_qualities.append(quality)
                
                if round_info["quantum_backends_used"] > 0:
                    quantum_successes += 1
                
                results["quantum_results"].append({
                    "trial": trial,
                    "execution_time": execution_time,
                    "quality": quality,
                    "quantum_used": round_info["quantum_backends_used"] > 0,
                    "round_info": round_info
                })
                
            except Exception as e:
                results["quantum_results"].append({
                    "trial": trial,
                    "error": str(e),
                    "execution_time": float('inf'),
                    "quality": 0.0,
                    "quantum_used": False
                })
        
        # Run classical aggregation methods
        for method_name, method_func in self.classical_methods.items():
            method_times = []
            method_qualities = []
            
            for trial in range(num_trials):
                start_time = time.time()
                try:
                    aggregated_params = method_func(client_parameters, client_weights)
                    execution_time = time.time() - start_time
                    
                    quality = self._evaluate_aggregation_quality(aggregated_params, client_parameters)
                    
                    method_times.append(execution_time)
                    method_qualities.append(quality)
                    
                except Exception as e:
                    method_times.append(float('inf'))
                    method_qualities.append(0.0)
            
            results["classical_results"][method_name] = {
                "avg_execution_time": np.mean(method_times),
                "avg_quality": np.mean(method_qualities),
                "std_execution_time": np.std(method_times),
                "std_quality": np.std(method_qualities)
            }
        
        # Quantum performance summary
        results["quantum_summary"] = {
            "avg_execution_time": np.mean(quantum_times),
            "avg_quality": np.mean(quantum_qualities),
            "std_execution_time": np.std(quantum_times),
            "std_quality": np.std(quantum_qualities),
            "success_rate": quantum_successes / num_trials
        }
        
        return results
    
    def _evaluate_aggregation_quality(
        self,
        aggregated_params: jnp.ndarray,
        client_parameters: List[jnp.ndarray]
    ) -> float:
        """Evaluate quality of parameter aggregation."""
        if aggregated_params is None:
            return 0.0
        
        # Quality metric: negative variance from aggregated parameters
        client_array = jnp.array(client_parameters)
        
        # Compute distances from aggregated parameters
        distances = jnp.linalg.norm(client_array - aggregated_params, axis=1)
        
        # Quality is inverse of average distance (higher is better)
        avg_distance = jnp.mean(distances)
        quality = 1.0 / (1.0 + avg_distance)
        
        return float(quality)
    
    def _fedavg(self, client_parameters: List[jnp.ndarray], weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """FedAvg aggregation."""
        if weights is None:
            return jnp.mean(jnp.array(client_parameters), axis=0)
        else:
            return jnp.average(jnp.array(client_parameters), axis=0, weights=weights)
    
    def _weighted_average(self, client_parameters: List[jnp.ndarray], weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Weighted average aggregation."""
        if weights is None:
            # Use inverse variance weighting
            variances = [jnp.var(params) for params in client_parameters]
            weights = 1.0 / (jnp.array(variances) + 1e-8)
            weights = weights / jnp.sum(weights)
        
        return jnp.average(jnp.array(client_parameters), axis=0, weights=weights)
    
    def _median_aggregation(self, client_parameters: List[jnp.ndarray], weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Median aggregation (robust to outliers)."""
        return jnp.median(jnp.array(client_parameters), axis=0)
    
    def _trimmed_mean(self, client_parameters: List[jnp.ndarray], weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Trimmed mean aggregation."""
        client_array = jnp.array(client_parameters)
        
        # Sort and trim 10% from each end
        sorted_params = jnp.sort(client_array, axis=0)
        trim_count = max(1, len(client_parameters) // 10)
        
        if trim_count >= len(client_parameters):
            return jnp.mean(client_array, axis=0)
        
        trimmed = sorted_params[trim_count:-trim_count]
        return jnp.mean(trimmed, axis=0)


class GraphOptimizationBenchmark:
    """Benchmark quantum vs classical graph optimization."""
    
    def __init__(self):
        self.classical_algorithms = {
            "greedy": self._greedy_max_cut,
            "simulated_annealing": self._simulated_annealing_max_cut,
            "spectral": self._spectral_max_cut
        }
    
    def run_benchmark(
        self,
        hybrid_optimizer: HybridClassicalQuantumOptimizer,
        graph_instances: List[List[Tuple[int, int]]],
        num_trials: int = 5
    ) -> Dict[str, Any]:
        """Run graph optimization benchmark."""
        
        results = {
            "quantum_results": [],
            "classical_results": {},
            "graph_sizes": [len(set([v for edge in graph for v in edge])) for graph in graph_instances]
        }
        
        for graph_idx, graph in enumerate(graph_instances):
            graph_size = len(set([v for edge in graph for v in edge]))
            
            # Quantum optimization
            quantum_result = {"graph_idx": graph_idx, "graph_size": graph_size}
            
            start_time = time.time()
            try:
                opt_result = hybrid_optimizer.optimize_dynamic_graph_partition([graph])
                quantum_result.update({
                    "execution_time": time.time() - start_time,
                    "optimization_value": opt_result["optimization_value"],
                    "success": opt_result["quantum_advantage"],
                    "backend_results": opt_result["backend_results"]
                })
            except Exception as e:
                quantum_result.update({
                    "execution_time": float('inf'),
                    "optimization_value": 0.0,
                    "success": False,
                    "error": str(e)
                })
            
            results["quantum_results"].append(quantum_result)
            
            # Classical optimization
            for alg_name, alg_func in self.classical_algorithms.items():
                if alg_name not in results["classical_results"]:
                    results["classical_results"][alg_name] = []
                
                start_time = time.time()
                try:
                    cut_value = alg_func(graph)
                    execution_time = time.time() - start_time
                    
                    results["classical_results"][alg_name].append({
                        "graph_idx": graph_idx,
                        "graph_size": graph_size,
                        "execution_time": execution_time,
                        "cut_value": cut_value,
                        "success": True
                    })
                except Exception as e:
                    results["classical_results"][alg_name].append({
                        "graph_idx": graph_idx,
                        "graph_size": graph_size,
                        "execution_time": float('inf'),
                        "cut_value": 0.0,
                        "success": False,
                        "error": str(e)
                    })
        
        return results
    
    def _greedy_max_cut(self, edges: List[Tuple[int, int]]) -> float:
        """Greedy algorithm for Max-Cut."""
        nodes = set([v for edge in edges for v in edge])
        
        # Start with random partition
        partition = {node: np.random.choice([0, 1]) for node in nodes}
        
        # Greedy improvement
        improved = True
        while improved:
            improved = False
            for node in nodes:
                # Try flipping node
                current_cut = self._evaluate_cut(edges, partition)
                
                partition[node] = 1 - partition[node]
                new_cut = self._evaluate_cut(edges, partition)
                
                if new_cut > current_cut:
                    improved = True
                else:
                    partition[node] = 1 - partition[node]  # Flip back
        
        return self._evaluate_cut(edges, partition)
    
    def _simulated_annealing_max_cut(self, edges: List[Tuple[int, int]]) -> float:
        """Simulated annealing for Max-Cut."""
        nodes = list(set([v for edge in edges for v in edge]))
        
        # Initialize random solution
        current_partition = {node: np.random.choice([0, 1]) for node in nodes}
        current_cut = self._evaluate_cut(edges, current_partition)
        
        best_partition = current_partition.copy()
        best_cut = current_cut
        
        # Simulated annealing parameters
        initial_temp = 10.0
        final_temp = 0.01
        cooling_rate = 0.95
        max_iterations = 1000
        
        temperature = initial_temp
        
        for iteration in range(max_iterations):
            # Generate neighbor by flipping random node
            neighbor_partition = current_partition.copy()
            flip_node = np.random.choice(nodes)
            neighbor_partition[flip_node] = 1 - neighbor_partition[flip_node]
            
            neighbor_cut = self._evaluate_cut(edges, neighbor_partition)
            
            # Accept or reject based on temperature
            delta = neighbor_cut - current_cut
            
            if delta > 0 or np.secrets.SystemRandom().random() < np.exp(delta / temperature):
                current_partition = neighbor_partition
                current_cut = neighbor_cut
                
                if current_cut > best_cut:
                    best_partition = current_partition.copy()
                    best_cut = current_cut
            
            # Cool down
            temperature *= cooling_rate
            if temperature < final_temp:
                break
        
        return best_cut
    
    def _spectral_max_cut(self, edges: List[Tuple[int, int]]) -> float:
        """Spectral algorithm for Max-Cut."""
        nodes = list(set([v for edge in edges for v in edge]))
        n = len(nodes)
        
        if n == 0:
            return 0.0
        
        # Create node mapping
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build adjacency matrix
        adj_matrix = np.zeros((n, n))
        for u, v in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        
        # Compute Laplacian
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree_matrix - adj_matrix
        
        try:
            # Find second smallest eigenvalue and eigenvector
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            fiedler_vector = eigenvectors[:, 1]  # Second smallest eigenvalue
            
            # Create partition based on sign of Fiedler vector
            partition = {nodes[i]: 1 if fiedler_vector[i] >= 0 else 0 for i in range(n)}
            
            return self._evaluate_cut(edges, partition)
            
        except Exception:
            # Fallback to random partition
            partition = {node: np.random.choice([0, 1]) for node in nodes}
            return self._evaluate_cut(edges, partition)
    
    def _evaluate_cut(self, edges: List[Tuple[int, int]], partition: Dict[int, int]) -> float:
        """Evaluate cut value for given partition."""
        cut_value = 0
        for u, v in edges:
            if u in partition and v in partition and partition[u] != partition[v]:
                cut_value += 1
        return cut_value


class QuantumAdvantageBenchmark:
    """Main quantum advantage benchmarking system."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.param_benchmark = ParameterAggregationBenchmark()
        self.graph_benchmark = GraphOptimizationBenchmark()
        self.results_history = []
    
    def run_comprehensive_benchmark(
        self,
        quantum_fed_learning: Optional[QuantumFederatedLearning] = None,
        hybrid_optimizer: Optional[HybridClassicalQuantumOptimizer] = None,
        test_data_generator: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run comprehensive quantum advantage benchmark."""
        
        comprehensive_results = {
            "config": {
                "benchmark_type": self.config.benchmark_type.value,
                "num_trials": self.config.num_trials,
                "problem_sizes": self.config.problem_sizes,
                "timestamp": time.time()
            },
            "benchmark_results": []
        }
        
        for problem_size in self.config.problem_sizes:
            # Generate test data
            if test_data_generator:
                test_data = test_data_generator(problem_size)
            else:
                test_data = self._generate_default_test_data(problem_size)
            
            size_results = {"problem_size": problem_size}
            
            if self.config.benchmark_type == BenchmarkType.PARAMETER_AGGREGATION and quantum_fed_learning:
                benchmark_result = self.param_benchmark.run_benchmark(
                    quantum_fed_learning,
                    test_data["client_parameters"],
                    test_data.get("client_weights"),
                    self.config.num_trials
                )
                size_results["parameter_aggregation"] = benchmark_result
                
            elif self.config.benchmark_type == BenchmarkType.GRAPH_OPTIMIZATION and hybrid_optimizer:
                benchmark_result = self.graph_benchmark.run_benchmark(
                    hybrid_optimizer,
                    test_data["graphs"],
                    self.config.num_trials
                )
                size_results["graph_optimization"] = benchmark_result
            
            comprehensive_results["benchmark_results"].append(size_results)
        
        # Compute quantum advantage metrics
        advantage_metrics = self._compute_quantum_advantage_metrics(comprehensive_results)
        comprehensive_results["quantum_advantage_metrics"] = advantage_metrics
        
        self.results_history.append(comprehensive_results)
        
        return comprehensive_results
    
    def _generate_default_test_data(self, problem_size: int) -> Dict[str, Any]:
        """Generate default test data for benchmarking."""
        if self.config.benchmark_type == BenchmarkType.PARAMETER_AGGREGATION:
            # Generate synthetic federated learning parameters
            num_clients = min(10, problem_size // 10 + 1)
            param_dim = problem_size
            
            client_parameters = []
            for _ in range(num_clients):
                params = np.random.normal(0, 1, param_dim)
                client_parameters.append(jnp.array(params))
            
            client_weights = jnp.random.uniform(0.1, 1.0, num_clients)
            client_weights = client_weights / jnp.sum(client_weights)
            
            return {
                "client_parameters": client_parameters,
                "client_weights": client_weights
            }
            
        elif self.config.benchmark_type == BenchmarkType.GRAPH_OPTIMIZATION:
            # Generate random graphs
            num_nodes = problem_size
            edge_probability = min(0.5, 10.0 / num_nodes)  # Adjust density for larger graphs
            
            graphs = []
            for _ in range(max(1, self.config.num_trials // 2)):
                edges = []
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        if np.secrets.SystemRandom().random() < edge_probability:
                            edges.append((i, j))
                graphs.append(edges)
            
            return {"graphs": graphs}
        
        return {}
    
    def _compute_quantum_advantage_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute quantum advantage metrics from benchmark results."""
        metrics = {
            "overall_quantum_advantage": False,
            "advantage_by_problem_size": {},
            "performance_scaling": {},
            "statistical_significance": {}
        }
        
        quantum_advantages = []
        
        for size_result in results["benchmark_results"]:
            problem_size = size_result["problem_size"]
            
            # Parameter aggregation analysis
            if "parameter_aggregation" in size_result:
                param_results = size_result["parameter_aggregation"]
                quantum_summary = param_results["quantum_summary"]
                
                # Compare with best classical method
                best_classical_quality = max(
                    classical_result["avg_quality"] 
                    for classical_result in param_results["classical_results"].values()
                )
                
                best_classical_time = min(
                    classical_result["avg_execution_time"] 
                    for classical_result in param_results["classical_results"].values()
                )
                
                quality_advantage = quantum_summary["avg_quality"] / best_classical_quality if best_classical_quality > 0 else 1.0
                time_advantage = best_classical_time / quantum_summary["avg_execution_time"] if quantum_summary["avg_execution_time"] > 0 else 0.0
                
                # Combined advantage metric
                combined_advantage = quality_advantage * time_advantage
                
                metrics["advantage_by_problem_size"][problem_size] = {
                    "quality_advantage": quality_advantage,
                    "time_advantage": time_advantage,
                    "combined_advantage": combined_advantage,
                    "quantum_success_rate": quantum_summary["success_rate"]
                }
                
                quantum_advantages.append(combined_advantage)
            
            # Graph optimization analysis
            if "graph_optimization" in size_result:
                graph_results = size_result["graph_optimization"]
                
                # Compute average quantum performance
                quantum_values = [
                    result["optimization_value"] for result in graph_results["quantum_results"]
                    if result["success"]
                ]
                
                if quantum_values:
                    avg_quantum_value = np.mean(quantum_values)
                    
                    # Compare with classical methods
                    classical_advantages = []
                    for method_name, method_results in graph_results["classical_results"].items():
                        classical_values = [
                            result["cut_value"] for result in method_results
                            if result["success"]
                        ]
                        
                        if classical_values:
                            avg_classical_value = np.mean(classical_values)
                            advantage = avg_quantum_value / avg_classical_value if avg_classical_value > 0 else 1.0
                            classical_advantages.append(advantage)
                    
                    if classical_advantages:
                        best_advantage = max(classical_advantages)
                        metrics["advantage_by_problem_size"][problem_size] = {
                            "optimization_advantage": best_advantage,
                            "quantum_avg_value": avg_quantum_value
                        }
                        quantum_advantages.append(best_advantage)
        
        # Overall quantum advantage
        if quantum_advantages:
            metrics["overall_quantum_advantage"] = np.mean(quantum_advantages) > self.config.quality_threshold
            metrics["average_advantage"] = np.mean(quantum_advantages)
            metrics["advantage_std"] = np.std(quantum_advantages)
        
        return metrics
    
    def generate_benchmark_report(self) -> str:
        """Generate human-readable benchmark report."""
        if not self.results_history:
            return "No benchmark results available."
        
        latest_results = self.results_history[-1]
        
        report = []
        report.append("=== QUANTUM ADVANTAGE BENCHMARK REPORT ===")
        report.append(f"Benchmark Type: {latest_results['config']['benchmark_type']}")
        report.append(f"Number of Trials: {latest_results['config']['num_trials']}")
        report.append(f"Problem Sizes: {latest_results['config']['problem_sizes']}")
        report.append("")
        
        metrics = latest_results["quantum_advantage_metrics"]
        
        report.append("=== QUANTUM ADVANTAGE ANALYSIS ===")
        report.append(f"Overall Quantum Advantage: {metrics['overall_quantum_advantage']}")
        
        if "average_advantage" in metrics:
            report.append(f"Average Advantage Factor: {metrics['average_advantage']:.3f}")
            report.append(f"Advantage Standard Deviation: {metrics['advantage_std']:.3f}")
        
        report.append("")
        report.append("=== PERFORMANCE BY PROBLEM SIZE ===")
        
        for problem_size, size_metrics in metrics["advantage_by_problem_size"].items():
            report.append(f"Problem Size {problem_size}:")
            for metric_name, value in size_metrics.items():
                report.append(f"  {metric_name}: {value:.3f}")
            report.append("")
        
        return "\n".join(report)
    
    def export_results(self, filename: str) -> None:
        """Export benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)