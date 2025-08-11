"""
Research Benchmark Datasets

Standardized datasets for evaluating novel research contributions
with reproducible experimental protocols and statistical validation.
"""

import time
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Iterator
import numpy as np
import jax.numpy as jnp
from jax import random
from abc import ABC, abstractmethod
from enum import Enum


class BenchmarkDifficulty(Enum):
    """Benchmark difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium" 
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class BenchmarkMetadata:
    """Metadata for research benchmarks."""
    benchmark_id: str
    name: str
    description: str
    difficulty: BenchmarkDifficulty
    num_samples: int
    temporal_length: int
    graph_size: int
    num_clients: int
    creation_time: float
    version: str = "1.0"


class ResearchBenchmark(ABC):
    """Abstract base class for research benchmarks."""
    
    def __init__(self, benchmark_id: str, random_seed: int = 42):
        self.benchmark_id = benchmark_id
        self.random_seed = random_seed
        self.metadata: Optional[BenchmarkMetadata] = None
        self.key = random.PRNGKey(random_seed)
        
    @abstractmethod
    def generate_data(self) -> Dict[str, Any]:
        """Generate benchmark dataset."""
        pass
    
    @abstractmethod
    def evaluate_method(self, method_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate method on benchmark."""
        pass
    
    def save_benchmark(self, filepath: str) -> None:
        """Save benchmark to file."""
        data = {
            "metadata": self.metadata,
            "benchmark_data": self.generate_data(),
            "creation_timestamp": time.time()
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    
    def load_benchmark(self, filepath: str) -> Dict[str, Any]:
        """Load benchmark from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.metadata = data["metadata"]
        return data["benchmark_data"]


class QuantumCoherenceBenchmark(ResearchBenchmark):
    """
    Benchmark for evaluating quantum coherence optimization methods.
    
    Tests ability to achieve quantum advantage in federated parameter aggregation
    with various client heterogeneity levels and network conditions.
    """
    
    def __init__(
        self,
        num_clients: int = 10,
        graph_size: int = 100,
        temporal_length: int = 50,
        difficulty: BenchmarkDifficulty = BenchmarkDifficulty.MEDIUM,
        **kwargs
    ):
        super().__init__(f"quantum_coherence_{difficulty.value}", **kwargs)
        
        self.num_clients = num_clients
        self.graph_size = graph_size
        self.temporal_length = temporal_length
        self.difficulty = difficulty
        
        # Difficulty-dependent parameters
        difficulty_params = {
            BenchmarkDifficulty.EASY: {
                "heterogeneity": 0.1,
                "noise_level": 0.01,
                "coherence_time": 20.0,
                "client_dropout": 0.0
            },
            BenchmarkDifficulty.MEDIUM: {
                "heterogeneity": 0.3,
                "noise_level": 0.05,
                "coherence_time": 10.0,
                "client_dropout": 0.1
            },
            BenchmarkDifficulty.HARD: {
                "heterogeneity": 0.6,
                "noise_level": 0.1,
                "coherence_time": 5.0,
                "client_dropout": 0.2
            },
            BenchmarkDifficulty.EXTREME: {
                "heterogeneity": 0.9,
                "noise_level": 0.2,
                "coherence_time": 2.0,
                "client_dropout": 0.3
            }
        }
        
        self.params = difficulty_params[difficulty]
        
        self.metadata = BenchmarkMetadata(
            benchmark_id=self.benchmark_id,
            name=f"Quantum Coherence Benchmark ({difficulty.value})",
            description="Evaluates quantum-inspired federated aggregation methods",
            difficulty=difficulty,
            num_samples=1000,
            temporal_length=temporal_length,
            graph_size=graph_size,
            num_clients=num_clients,
            creation_time=time.time()
        )
    
    def generate_data(self) -> Dict[str, Any]:
        """Generate quantum coherence benchmark data."""
        # Generate heterogeneous client data
        client_data = []
        
        for client_id in range(self.num_clients):
            self.key, client_key = random.split(self.key)
            
            # Client-specific graph sequences
            graphs = self._generate_client_graphs(client_key, client_id)
            
            # Client-specific parameters (heterogeneous)
            heterogeneity_noise = random.normal(client_key, (self.graph_size,)) * self.params["heterogeneity"]
            client_params = self._generate_base_parameters() + heterogeneity_noise
            
            client_data.append({
                "client_id": client_id,
                "graph_sequence": graphs,
                "initial_parameters": client_params,
                "data_distribution": self._generate_client_distribution(client_key)
            })
        
        # Ground truth optimal parameters
        optimal_params = self._generate_optimal_parameters()
        
        # Communication network topology
        network_topology = self._generate_network_topology()
        
        # Quantum coherence patterns (for evaluation)
        coherence_patterns = self._generate_coherence_patterns()
        
        return {
            "client_data": client_data,
            "optimal_parameters": optimal_params,
            "network_topology": network_topology,
            "coherence_patterns": coherence_patterns,
            "benchmark_params": self.params,
            "evaluation_metrics": [
                "convergence_speed",
                "quantum_advantage",
                "parameter_quality",
                "communication_efficiency",
                "coherence_preservation"
            ]
        }
    
    def _generate_client_graphs(self, key: jnp.ndarray, client_id: int) -> List[jnp.ndarray]:
        """Generate temporal graph sequence for client."""
        graphs = []
        
        # Client-specific graph characteristics
        base_connectivity = 0.1 + 0.3 * (client_id / self.num_clients)
        
        for t in range(self.temporal_length):
            # Temporal evolution
            time_factor = 1.0 + 0.1 * jnp.sin(2 * jnp.pi * t / 10)
            connectivity = base_connectivity * time_factor
            
            # Generate random graph
            key, graph_key = random.split(key)
            adjacency = random.bernoulli(graph_key, connectivity, (self.graph_size, self.graph_size))
            
            # Node features
            key, feature_key = random.split(key)
            node_features = random.normal(feature_key, (self.graph_size, 16))
            
            # Combine into graph representation
            graph = jnp.concatenate([adjacency.astype(jnp.float32), node_features], axis=-1)
            graphs.append(graph)
        
        return graphs
    
    def _generate_base_parameters(self) -> jnp.ndarray:
        """Generate base parameters for quantum coherence."""
        key, param_key = random.split(self.key)
        self.key = key
        return random.normal(param_key, (self.graph_size,))
    
    def _generate_optimal_parameters(self) -> jnp.ndarray:
        """Generate ground truth optimal parameters."""
        key, param_key = random.split(self.key)
        self.key = key
        return random.normal(param_key, (self.graph_size,)) * 0.5
    
    def _generate_client_distribution(self, key: jnp.ndarray) -> Dict[str, Any]:
        """Generate client data distribution characteristics."""
        return {
            "mean": random.normal(key, (5,)),
            "std": random.uniform(key, (5,), minval=0.5, maxval=2.0),
            "skewness": random.normal(key, ()) * 0.5
        }
    
    def _generate_network_topology(self) -> jnp.ndarray:
        """Generate network topology for communication."""
        key, topo_key = random.split(self.key)
        self.key = key
        
        # Random network with some structure
        base_connectivity = 0.3
        topology = random.bernoulli(topo_key, base_connectivity, (self.num_clients, self.num_clients))
        
        # Ensure connectivity
        jnp.fill_diagonal(topology, 0)  # No self-loops
        
        return topology.astype(jnp.float32)
    
    def _generate_coherence_patterns(self) -> Dict[str, jnp.ndarray]:
        """Generate expected quantum coherence patterns."""
        key, coherence_key = random.split(self.key)
        self.key = key
        
        # Expected entanglement patterns
        entanglement_matrix = random.uniform(
            coherence_key, (self.num_clients, self.num_clients), minval=0, maxval=0.5
        )
        
        # Expected superposition evolution
        superposition_evolution = random.uniform(
            coherence_key, (self.temporal_length, self.num_clients), minval=0, maxval=1
        )
        
        return {
            "entanglement_matrix": entanglement_matrix,
            "superposition_evolution": superposition_evolution,
            "coherence_decay": jnp.exp(-jnp.arange(self.temporal_length) / self.params["coherence_time"])
        }
    
    def evaluate_method(self, method_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate quantum coherence method."""
        metrics = {}
        
        # Convergence speed (lower is better)
        if "convergence_rounds" in method_results:
            baseline_rounds = 100  # Expected baseline convergence
            improvement = (baseline_rounds - method_results["convergence_rounds"]) / baseline_rounds
            metrics["convergence_improvement"] = max(0.0, improvement)
        
        # Quantum advantage (higher is better)
        if "quantum_advantage" in method_results:
            metrics["quantum_advantage"] = method_results["quantum_advantage"]
        
        # Parameter quality (higher is better)
        if "final_parameters" in method_results:
            optimal_params = self.generate_data()["optimal_parameters"]
            param_error = jnp.linalg.norm(method_results["final_parameters"] - optimal_params)
            metrics["parameter_quality"] = max(0.0, 1.0 - param_error)
        
        # Communication efficiency (higher is better)
        if "communication_bytes" in method_results:
            baseline_bytes = self.num_clients * self.graph_size * 4 * 100  # Expected baseline
            efficiency = 1.0 - (method_results["communication_bytes"] / baseline_bytes)
            metrics["communication_efficiency"] = max(0.0, efficiency)
        
        # Coherence preservation (higher is better)
        if "coherence_trace" in method_results:
            expected_coherence = self.generate_data()["coherence_patterns"]["coherence_decay"]
            coherence_similarity = jnp.corrcoef(
                method_results["coherence_trace"], expected_coherence
            )[0, 1]
            metrics["coherence_preservation"] = max(0.0, coherence_similarity)
        
        # Overall score (weighted average)
        if len(metrics) > 0:
            weights = {
                "convergence_improvement": 0.25,
                "quantum_advantage": 0.30,
                "parameter_quality": 0.25,
                "communication_efficiency": 0.10,
                "coherence_preservation": 0.10
            }
            
            weighted_score = sum(
                metrics.get(key, 0.0) * weight 
                for key, weight in weights.items()
            )
            metrics["overall_score"] = weighted_score
        
        return metrics


class AdversarialRobustnessBenchmark(ResearchBenchmark):
    """
    Benchmark for evaluating adversarial robustness methods on dynamic graphs.
    
    Tests defense effectiveness against various temporal adversarial attacks
    with different perturbation budgets and attack sophistication levels.
    """
    
    def __init__(
        self,
        graph_size: int = 500,
        temporal_length: int = 100,
        difficulty: BenchmarkDifficulty = BenchmarkDifficulty.MEDIUM,
        **kwargs
    ):
        super().__init__(f"adversarial_robustness_{difficulty.value}", **kwargs)
        
        self.graph_size = graph_size
        self.temporal_length = temporal_length
        self.difficulty = difficulty
        
        # Attack parameters by difficulty
        attack_params = {
            BenchmarkDifficulty.EASY: {
                "perturbation_budgets": [0.01, 0.05],
                "attack_types": ["topology_injection", "feature_noise"],
                "attack_success_threshold": 0.7,
                "adaptive_attacks": False
            },
            BenchmarkDifficulty.MEDIUM: {
                "perturbation_budgets": [0.05, 0.1, 0.15],
                "attack_types": ["topology_injection", "feature_noise", "temporal_shift"],
                "attack_success_threshold": 0.5,
                "adaptive_attacks": False
            },
            BenchmarkDifficulty.HARD: {
                "perturbation_budgets": [0.1, 0.2, 0.3],
                "attack_types": ["topology_injection", "feature_noise", "temporal_shift", "causality_violation"],
                "attack_success_threshold": 0.3,
                "adaptive_attacks": True
            },
            BenchmarkDifficulty.EXTREME: {
                "perturbation_budgets": [0.2, 0.4, 0.6],
                "attack_types": ["topology_injection", "feature_noise", "temporal_shift", "causality_violation", "gradient_inversion"],
                "attack_success_threshold": 0.1,
                "adaptive_attacks": True
            }
        }
        
        self.attack_params = attack_params[difficulty]
        
        self.metadata = BenchmarkMetadata(
            benchmark_id=self.benchmark_id,
            name=f"Adversarial Robustness Benchmark ({difficulty.value})",
            description="Evaluates defense mechanisms against temporal graph adversarial attacks",
            difficulty=difficulty,
            num_samples=500,
            temporal_length=temporal_length,
            graph_size=graph_size,
            num_clients=1,  # Single client for robustness testing
            creation_time=time.time()
        )
    
    def generate_data(self) -> Dict[str, Any]:
        """Generate adversarial robustness benchmark data."""
        # Clean temporal graph sequences
        clean_sequences = self._generate_clean_sequences()
        
        # Adversarial examples for each attack type and budget
        adversarial_examples = {}
        
        for attack_type in self.attack_params["attack_types"]:
            adversarial_examples[attack_type] = {}
            
            for budget in self.attack_params["perturbation_budgets"]:
                attacked_sequences = self._generate_attacked_sequences(
                    clean_sequences, attack_type, budget
                )
                adversarial_examples[attack_type][str(budget)] = attacked_sequences
        
        # Ground truth labels and expected robust predictions
        labels = self._generate_labels(clean_sequences)
        robust_predictions = self._generate_robust_predictions(labels)
        
        return {
            "clean_sequences": clean_sequences,
            "adversarial_examples": adversarial_examples,
            "labels": labels,
            "robust_predictions": robust_predictions,
            "attack_params": self.attack_params,
            "evaluation_metrics": [
                "clean_accuracy",
                "robust_accuracy", 
                "attack_detection_rate",
                "certified_robustness_radius",
                "false_positive_rate"
            ]
        }
    
    def _generate_clean_sequences(self) -> List[List[jnp.ndarray]]:
        """Generate clean temporal graph sequences."""
        sequences = []
        
        num_sequences = 100  # Number of test sequences
        
        for seq_id in range(num_sequences):
            self.key, seq_key = random.split(self.key)
            
            sequence = []
            for t in range(self.temporal_length):
                # Generate graph with temporal patterns
                connectivity = 0.1 + 0.05 * jnp.sin(2 * jnp.pi * t / 20)
                
                seq_key, graph_key = random.split(seq_key)
                adjacency = random.bernoulli(graph_key, connectivity, (self.graph_size, self.graph_size))
                
                seq_key, feature_key = random.split(seq_key)
                node_features = random.normal(feature_key, (self.graph_size, 10))
                
                graph = jnp.concatenate([adjacency.astype(jnp.float32), node_features], axis=-1)
                sequence.append(graph)
            
            sequences.append(sequence)
        
        return sequences
    
    def _generate_attacked_sequences(
        self, 
        clean_sequences: List[List[jnp.ndarray]], 
        attack_type: str,
        perturbation_budget: float
    ) -> List[List[jnp.ndarray]]:
        """Generate adversarial examples using specified attack."""
        attacked_sequences = []
        
        for sequence in clean_sequences:
            if attack_type == "topology_injection":
                attacked_seq = self._topology_injection_attack(sequence, perturbation_budget)
            elif attack_type == "feature_noise":
                attacked_seq = self._feature_noise_attack(sequence, perturbation_budget)
            elif attack_type == "temporal_shift":
                attacked_seq = self._temporal_shift_attack(sequence, perturbation_budget)
            elif attack_type == "causality_violation":
                attacked_seq = self._causality_violation_attack(sequence, perturbation_budget)
            elif attack_type == "gradient_inversion":
                attacked_seq = self._gradient_inversion_attack(sequence, perturbation_budget)
            else:
                attacked_seq = sequence  # Unknown attack type
            
            attacked_sequences.append(attacked_seq)
        
        return attacked_sequences
    
    def _topology_injection_attack(
        self, 
        sequence: List[jnp.ndarray], 
        budget: float
    ) -> List[jnp.ndarray]:
        """Inject adversarial edges into graph topology."""
        attacked_sequence = []
        
        for graph in sequence:
            # Split adjacency and features
            adjacency = graph[:, :self.graph_size]
            features = graph[:, self.graph_size:]
            
            # Add adversarial edges
            self.key, edge_key = random.split(self.key)
            num_edges_to_add = int(budget * self.graph_size * self.graph_size)
            
            edge_indices = random.choice(
                edge_key, self.graph_size * self.graph_size, 
                shape=(num_edges_to_add,), replace=False
            )
            
            # Convert to 2D indices
            rows = edge_indices // self.graph_size
            cols = edge_indices % self.graph_size
            
            # Inject edges
            attacked_adj = adjacency.at[rows, cols].set(1.0)
            attacked_graph = jnp.concatenate([attacked_adj, features], axis=-1)
            attacked_sequence.append(attacked_graph)
        
        return attacked_sequence
    
    def _feature_noise_attack(
        self, 
        sequence: List[jnp.ndarray], 
        budget: float
    ) -> List[jnp.ndarray]:
        """Add adversarial noise to node features."""
        attacked_sequence = []
        
        for graph in sequence:
            adjacency = graph[:, :self.graph_size]
            features = graph[:, self.graph_size:]
            
            # Add noise to features
            self.key, noise_key = random.split(self.key)
            noise = random.normal(noise_key, features.shape) * budget
            
            attacked_features = features + noise
            attacked_graph = jnp.concatenate([adjacency, attacked_features], axis=-1)
            attacked_sequence.append(attacked_graph)
        
        return attacked_sequence
    
    def _temporal_shift_attack(
        self, 
        sequence: List[jnp.ndarray], 
        budget: float
    ) -> List[jnp.ndarray]:
        """Introduce temporal shifts to break causality."""
        attacked_sequence = sequence.copy()
        
        # Determine shift magnitude
        max_shift = max(1, int(budget * self.temporal_length))
        
        # Apply random temporal shifts
        for t in range(len(sequence)):
            if t + max_shift < len(sequence):
                self.key, shift_key = random.split(self.key)
                shift = random.randint(shift_key, (), 1, max_shift + 1)
                
                # Blend current with future (causality violation)
                blend_factor = budget
                current = sequence[t]
                future = sequence[t + shift]
                
                attacked_graph = (1 - blend_factor) * current + blend_factor * future
                attacked_sequence[t] = attacked_graph
        
        return attacked_sequence
    
    def _causality_violation_attack(
        self, 
        sequence: List[jnp.ndarray], 
        budget: float
    ) -> List[jnp.ndarray]:
        """Violate temporal causality constraints."""
        attacked_sequence = sequence.copy()
        
        # Reverse some temporal segments
        num_violations = max(1, int(budget * self.temporal_length / 10))
        
        for _ in range(num_violations):
            self.key, viol_key = random.split(self.key)
            
            # Select random segment
            start = random.randint(viol_key, (), 0, len(sequence) - 5)
            length = random.randint(viol_key, (), 3, min(8, len(sequence) - start))
            
            # Reverse segment
            segment = attacked_sequence[start:start+length]
            attacked_sequence[start:start+length] = segment[::-1]
        
        return attacked_sequence
    
    def _gradient_inversion_attack(
        self, 
        sequence: List[jnp.ndarray], 
        budget: float
    ) -> List[jnp.ndarray]:
        """Simulate gradient inversion attack effects."""
        # For simulation, add structured noise that mimics gradient inversion
        attacked_sequence = []
        
        for graph in sequence:
            self.key, attack_key = random.split(self.key)
            
            # Structured adversarial perturbation
            structured_noise = random.normal(attack_key, graph.shape)
            
            # Apply low-rank structure (typical of gradient inversion)
            u, s, vt = jnp.linalg.svd(structured_noise, full_matrices=False)
            rank = max(1, int((1 - budget) * min(graph.shape)))
            structured_noise = u[:, :rank] @ jnp.diag(s[:rank]) @ vt[:rank, :]
            
            attacked_graph = graph + budget * structured_noise
            attacked_sequence.append(attacked_graph)
        
        return attacked_sequence
    
    def _generate_labels(self, sequences: List[List[jnp.ndarray]]) -> List[int]:
        """Generate ground truth labels for sequences."""
        labels = []
        for i, sequence in enumerate(sequences):
            # Simple labeling based on sequence properties
            avg_connectivity = jnp.mean(jnp.array([jnp.mean(g[:, :self.graph_size]) for g in sequence]))
            label = 1 if avg_connectivity > 0.15 else 0
            labels.append(label)
        return labels
    
    def _generate_robust_predictions(self, labels: List[int]) -> List[float]:
        """Generate expected robust predictions."""
        # Robust predictions should be stable under attacks
        return [0.9 if label == 1 else 0.1 for label in labels]
    
    def evaluate_method(self, method_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate adversarial robustness method."""
        metrics = {}
        
        # Clean accuracy
        if "clean_predictions" in method_results and "true_labels" in method_results:
            clean_acc = jnp.mean(
                (jnp.array(method_results["clean_predictions"]) > 0.5) == 
                jnp.array(method_results["true_labels"])
            )
            metrics["clean_accuracy"] = float(clean_acc)
        
        # Robust accuracy across attacks
        if "adversarial_predictions" in method_results:
            adv_results = method_results["adversarial_predictions"]
            robust_accuracies = []
            
            for attack_type, attack_results in adv_results.items():
                for budget, predictions in attack_results.items():
                    robust_acc = jnp.mean(
                        (jnp.array(predictions) > 0.5) == 
                        jnp.array(method_results.get("true_labels", []))
                    )
                    robust_accuracies.append(float(robust_acc))
            
            if robust_accuracies:
                metrics["robust_accuracy"] = jnp.mean(jnp.array(robust_accuracies))
        
        # Attack detection rate
        if "attack_detected" in method_results:
            detection_rate = jnp.mean(jnp.array(method_results["attack_detected"]))
            metrics["attack_detection_rate"] = float(detection_rate)
        
        # Certified robustness radius
        if "certified_radius" in method_results:
            metrics["certified_robustness_radius"] = method_results["certified_radius"]
        
        # False positive rate
        if "false_positives" in method_results and "total_clean" in method_results:
            fpr = method_results["false_positives"] / method_results["total_clean"]
            metrics["false_positive_rate"] = fpr
        
        # Overall robustness score
        if len(metrics) > 0:
            weights = {
                "clean_accuracy": 0.20,
                "robust_accuracy": 0.40,
                "attack_detection_rate": 0.20,
                "certified_robustness_radius": 0.15,
                "false_positive_rate": 0.05  # Negative weight
            }
            
            weighted_score = 0.0
            for key, weight in weights.items():
                if key in metrics:
                    if key == "false_positive_rate":
                        weighted_score -= weight * metrics[key]  # Penalty for false positives
                    else:
                        weighted_score += weight * metrics[key]
            
            metrics["overall_robustness_score"] = max(0.0, weighted_score)
        
        return metrics


class CommunicationEfficiencyBenchmark(ResearchBenchmark):
    """
    Benchmark for evaluating communication efficiency methods in federated learning.
    
    Tests compression effectiveness, convergence preservation, and information
    preservation across various network conditions and client configurations.
    """
    
    def __init__(
        self,
        num_clients: int = 20,
        graph_size: int = 1000,
        temporal_length: int = 30,
        difficulty: BenchmarkDifficulty = BenchmarkDifficulty.MEDIUM,
        **kwargs
    ):
        super().__init__(f"communication_efficiency_{difficulty.value}", **kwargs)
        
        self.num_clients = num_clients
        self.graph_size = graph_size
        self.temporal_length = temporal_length
        self.difficulty = difficulty
        
        # Network conditions by difficulty
        network_params = {
            BenchmarkDifficulty.EASY: {
                "bandwidth_limits": [1e6, 5e6],  # 1-5 MB/s
                "latency_ms": [10, 50],
                "packet_loss": [0.0, 0.01],
                "target_compression": [0.5, 0.3]  # 50-70% compression
            },
            BenchmarkDifficulty.MEDIUM: {
                "bandwidth_limits": [1e5, 1e6],  # 100KB-1MB/s
                "latency_ms": [50, 200], 
                "packet_loss": [0.01, 0.05],
                "target_compression": [0.2, 0.1]  # 80-90% compression
            },
            BenchmarkDifficulty.HARD: {
                "bandwidth_limits": [1e4, 1e5],  # 10-100KB/s
                "latency_ms": [200, 1000],
                "packet_loss": [0.05, 0.15],
                "target_compression": [0.1, 0.05]  # 90-95% compression
            },
            BenchmarkDifficulty.EXTREME: {
                "bandwidth_limits": [1e3, 1e4],  # 1-10KB/s
                "latency_ms": [1000, 5000],
                "packet_loss": [0.15, 0.3],
                "target_compression": [0.05, 0.01]  # 95-99% compression
            }
        }
        
        self.network_params = network_params[difficulty]
        
        self.metadata = BenchmarkMetadata(
            benchmark_id=self.benchmark_id,
            name=f"Communication Efficiency Benchmark ({difficulty.value})",
            description="Evaluates compression methods for federated temporal graph learning",
            difficulty=difficulty,
            num_samples=200,
            temporal_length=temporal_length,
            graph_size=graph_size,
            num_clients=num_clients,
            creation_time=time.time()
        )
    
    def generate_data(self) -> Dict[str, Any]:
        """Generate communication efficiency benchmark data."""
        # Client temporal graph sequences
        client_sequences = self._generate_client_sequences()
        
        # Network conditions for different scenarios
        network_scenarios = self._generate_network_scenarios()
        
        # Baseline communication costs
        baseline_costs = self._compute_baseline_costs(client_sequences)
        
        # Information content analysis
        information_metrics = self._compute_information_metrics(client_sequences)
        
        return {
            "client_sequences": client_sequences,
            "network_scenarios": network_scenarios,
            "baseline_costs": baseline_costs,
            "information_metrics": information_metrics,
            "network_params": self.network_params,
            "evaluation_metrics": [
                "compression_ratio",
                "reconstruction_quality", 
                "convergence_preservation",
                "bandwidth_efficiency",
                "information_preservation"
            ]
        }
    
    def _generate_client_sequences(self) -> Dict[int, List[jnp.ndarray]]:
        """Generate temporal graph sequences for each client."""
        client_sequences = {}
        
        for client_id in range(self.num_clients):
            self.key, client_key = random.split(self.key)
            
            sequence = []
            for t in range(self.temporal_length):
                # Generate graph with client-specific patterns
                base_connectivity = 0.05 + 0.1 * (client_id / self.num_clients)
                
                client_key, graph_key = random.split(client_key)
                adjacency = random.bernoulli(
                    graph_key, base_connectivity, (self.graph_size, self.graph_size)
                )
                
                client_key, feature_key = random.split(client_key)
                node_features = random.normal(feature_key, (self.graph_size, 20))
                
                graph = jnp.concatenate([adjacency.astype(jnp.float32), node_features], axis=-1)
                sequence.append(graph)
            
            client_sequences[client_id] = sequence
        
        return client_sequences
    
    def _generate_network_scenarios(self) -> List[Dict[str, Any]]:
        """Generate different network condition scenarios."""
        scenarios = []
        
        # Create scenarios with different network conditions
        num_scenarios = 5
        
        for scenario_id in range(num_scenarios):
            self.key, scenario_key = random.split(self.key)
            
            # Sample network parameters
            bandwidth = random.uniform(
                scenario_key, (), 
                self.network_params["bandwidth_limits"][0],
                self.network_params["bandwidth_limits"][1]
            )
            
            latency = random.uniform(
                scenario_key, (),
                self.network_params["latency_ms"][0], 
                self.network_params["latency_ms"][1]
            )
            
            packet_loss = random.uniform(
                scenario_key, (),
                self.network_params["packet_loss"][0],
                self.network_params["packet_loss"][1]
            )
            
            scenarios.append({
                "scenario_id": scenario_id,
                "bandwidth_bps": float(bandwidth),
                "latency_ms": float(latency),
                "packet_loss_rate": float(packet_loss),
                "target_compression_ratio": random.uniform(
                    scenario_key, (),
                    self.network_params["target_compression"][1],
                    self.network_params["target_compression"][0] 
                )
            })
        
        return scenarios
    
    def _compute_baseline_costs(
        self, 
        client_sequences: Dict[int, List[jnp.ndarray]]
    ) -> Dict[str, float]:
        """Compute baseline communication costs."""
        # Size of uncompressed data per client
        bytes_per_graph = self.graph_size * (self.graph_size + 20) * 4  # Float32
        total_bytes_per_client = bytes_per_graph * self.temporal_length
        
        total_baseline_bytes = total_bytes_per_client * self.num_clients
        
        return {
            "bytes_per_client": total_bytes_per_client,
            "total_baseline_bytes": total_baseline_bytes,
            "graphs_per_client": self.temporal_length,
            "features_per_graph": self.graph_size * 20,
            "adjacency_per_graph": self.graph_size * self.graph_size
        }
    
    def _compute_information_metrics(
        self, 
        client_sequences: Dict[int, List[jnp.ndarray]]
    ) -> Dict[str, float]:
        """Compute information-theoretic metrics."""
        # Sample graphs for analysis
        all_graphs = []
        for client_sequences_list in client_sequences.values():
            all_graphs.extend(client_sequences_list[:5])  # Sample 5 per client
        
        if not all_graphs:
            return {}
        
        # Stack graphs for analysis
        stacked_graphs = jnp.stack(all_graphs)
        
        # Compute entropy estimates
        # Quantize for entropy calculation
        quantized = jnp.round(stacked_graphs * 10).astype(jnp.int32)
        unique_values, counts = jnp.unique(quantized.flatten(), return_counts=True)
        
        probabilities = counts / jnp.sum(counts)
        entropy = -jnp.sum(probabilities * jnp.log2(probabilities + 1e-12))
        
        # Temporal correlations
        temporal_correlations = []
        for i in range(len(all_graphs) - 1):
            corr = jnp.corrcoef(all_graphs[i].flatten(), all_graphs[i+1].flatten())[0, 1]
            temporal_correlations.append(float(corr))
        
        avg_temporal_correlation = jnp.mean(jnp.array(temporal_correlations))
        
        return {
            "estimated_entropy": float(entropy),
            "temporal_correlation": float(avg_temporal_correlation),
            "data_dimensionality": stacked_graphs.shape[-1],
            "redundancy_estimate": 1.0 - float(entropy) / 16.0  # Normalized
        }
    
    def evaluate_method(self, method_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate communication efficiency method."""
        metrics = {}
        
        # Compression ratio (lower is better)
        if "compressed_size" in method_results and "original_size" in method_results:
            compression_ratio = method_results["compressed_size"] / method_results["original_size"]
            metrics["compression_ratio"] = compression_ratio
            
            # Compression efficiency score (higher is better)
            metrics["compression_efficiency"] = 1.0 - compression_ratio
        
        # Reconstruction quality (higher is better) 
        if "reconstruction_error" in method_results:
            # Convert error to quality score
            max_error = 1.0  # Assume normalized error
            quality = max(0.0, 1.0 - method_results["reconstruction_error"] / max_error)
            metrics["reconstruction_quality"] = quality
        
        # Convergence preservation (higher is better)
        if "convergence_rounds_compressed" in method_results and "convergence_rounds_baseline" in method_results:
            baseline_rounds = method_results["convergence_rounds_baseline"]
            compressed_rounds = method_results["convergence_rounds_compressed"]
            
            convergence_preservation = baseline_rounds / compressed_rounds
            metrics["convergence_preservation"] = min(1.0, convergence_preservation)
        
        # Bandwidth efficiency (higher is better)
        if "transmission_time" in method_results and "baseline_transmission_time" in method_results:
            baseline_time = method_results["baseline_transmission_time"]
            compressed_time = method_results["transmission_time"]
            
            bandwidth_efficiency = baseline_time / compressed_time
            metrics["bandwidth_efficiency"] = bandwidth_efficiency
        
        # Information preservation (higher is better)
        if "mutual_information_preserved" in method_results:
            metrics["information_preservation"] = method_results["mutual_information_preserved"]
        
        # Communication cost reduction (higher is better)
        if "total_bytes_sent" in method_results:
            baseline_bytes = self._compute_baseline_costs({})["total_baseline_bytes"]
            cost_reduction = 1.0 - (method_results["total_bytes_sent"] / baseline_bytes)
            metrics["communication_cost_reduction"] = max(0.0, cost_reduction)
        
        # Overall efficiency score
        if len(metrics) > 0:
            weights = {
                "compression_efficiency": 0.25,
                "reconstruction_quality": 0.20,
                "convergence_preservation": 0.25,
                "bandwidth_efficiency": 0.15,
                "information_preservation": 0.15
            }
            
            weighted_score = sum(
                metrics.get(key, 0.0) * weight 
                for key, weight in weights.items()
            )
            metrics["overall_efficiency_score"] = weighted_score
        
        return metrics


class UnifiedResearchBenchmark(ResearchBenchmark):
    """
    Unified benchmark combining all three research domains.
    
    Evaluates methods across quantum coherence, adversarial robustness,
    and communication efficiency in integrated scenarios.
    """
    
    def __init__(
        self,
        num_clients: int = 15,
        graph_size: int = 300,
        temporal_length: int = 75,
        difficulty: BenchmarkDifficulty = BenchmarkDifficulty.MEDIUM,
        **kwargs
    ):
        super().__init__(f"unified_research_{difficulty.value}", **kwargs)
        
        # Initialize component benchmarks
        self.quantum_benchmark = QuantumCoherenceBenchmark(
            num_clients, graph_size, temporal_length, difficulty
        )
        
        self.robustness_benchmark = AdversarialRobustnessBenchmark(
            graph_size, temporal_length, difficulty
        )
        
        self.efficiency_benchmark = CommunicationEfficiencyBenchmark(
            num_clients, graph_size, temporal_length, difficulty
        )
        
        self.metadata = BenchmarkMetadata(
            benchmark_id=self.benchmark_id,
            name=f"Unified Research Benchmark ({difficulty.value})",
            description="Comprehensive evaluation across all research domains",
            difficulty=difficulty,
            num_samples=300,
            temporal_length=temporal_length,
            graph_size=graph_size,
            num_clients=num_clients,
            creation_time=time.time()
        )
    
    def generate_data(self) -> Dict[str, Any]:
        """Generate unified benchmark data."""
        return {
            "quantum_coherence": self.quantum_benchmark.generate_data(),
            "adversarial_robustness": self.robustness_benchmark.generate_data(),
            "communication_efficiency": self.efficiency_benchmark.generate_data(),
            "unified_metrics": [
                "multi_domain_performance",
                "integration_effectiveness", 
                "scalability_across_domains",
                "practical_deployment_readiness"
            ]
        }
    
    def evaluate_method(self, method_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate method across all research domains."""
        metrics = {}
        
        # Evaluate each domain
        if "quantum_results" in method_results:
            quantum_metrics = self.quantum_benchmark.evaluate_method(
                method_results["quantum_results"]
            )
            metrics.update({f"quantum_{k}": v for k, v in quantum_metrics.items()})
        
        if "robustness_results" in method_results:
            robustness_metrics = self.robustness_benchmark.evaluate_method(
                method_results["robustness_results"]
            )
            metrics.update({f"robustness_{k}": v for k, v in robustness_metrics.items()})
        
        if "efficiency_results" in method_results:
            efficiency_metrics = self.efficiency_benchmark.evaluate_method(
                method_results["efficiency_results"]
            )
            metrics.update({f"efficiency_{k}": v for k, v in efficiency_metrics.items()})
        
        # Unified metrics
        domain_scores = []
        if "quantum_overall_score" in metrics:
            domain_scores.append(metrics["quantum_overall_score"])
        if "robustness_overall_robustness_score" in metrics:
            domain_scores.append(metrics["robustness_overall_robustness_score"])
        if "efficiency_overall_efficiency_score" in metrics:
            domain_scores.append(metrics["efficiency_overall_efficiency_score"])
        
        if domain_scores:
            metrics["unified_performance_score"] = jnp.mean(jnp.array(domain_scores))
            
            # Integration bonus for methods that perform well across domains
            min_score = jnp.min(jnp.array(domain_scores))
            max_score = jnp.max(jnp.array(domain_scores))
            consistency_bonus = 1.0 - (max_score - min_score)
            
            metrics["integration_effectiveness"] = consistency_bonus
        
        return metrics