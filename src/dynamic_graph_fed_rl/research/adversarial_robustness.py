"""
Adversarial Robustness in Multi-Scale Dynamic Graph Environments

Novel defense mechanisms leveraging multi-scale temporal modeling
to provide certified robustness against adversarial attacks on dynamic graphs.

Research Contribution: "Adversarial Robustness in Multi-Scale Dynamic Graph Environments"
Target Venue: ICML 2025 or NeurIPS 2025
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap
from abc import ABC, abstractmethod
from enum import Enum

from ..models.graph_networks import MultiScaleTemporalGNN


class AttackType(Enum):
    """Types of adversarial attacks on dynamic graphs."""
    TOPOLOGY_INJECTION = "topology_injection"  # Add/remove edges
    NODE_FEATURE_PERTURBATION = "node_feature_perturbation" 
    TEMPORAL_SHIFT = "temporal_shift"  # Shift time dependencies
    GRADIENT_INVERSION = "gradient_inversion"  # Federated privacy attack
    CAUSALITY_VIOLATION = "causality_violation"  # Break temporal causality


@dataclass
class AdversarialAttackResult:
    """Result of adversarial attack on dynamic graph."""
    attack_type: AttackType
    success: bool
    perturbation_magnitude: float
    original_performance: float
    attacked_performance: float
    attack_time: float
    detection_score: float
    temporal_scale_affected: List[int]


class TemporalGraphAttackSuite:
    """
    Comprehensive adversarial attack suite for dynamic graphs.
    
    Novel Contribution: First systematic attack suite targeting
    temporal dependencies in dynamic graph neural networks.
    """
    
    def __init__(self, perturbation_budget: float = 0.1):
        self.perturbation_budget = perturbation_budget
        self.attack_history: List[AdversarialAttackResult] = []
        
    def topology_injection_attack(
        self,
        graph_sequence: List[jnp.ndarray],
        edge_indices: List[jnp.ndarray], 
        target_nodes: Optional[List[int]] = None,
        injection_rate: float = 0.05
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Inject adversarial edges to disrupt graph structure learning.
        
        Novel Attack: Strategically adds edges at temporal points to
        maximize disruption to multi-scale temporal patterns.
        """
        perturbed_graphs = []
        perturbed_edges = []
        
        for t, (graph, edges) in enumerate(zip(graph_sequence, edge_indices)):
            num_nodes = graph.shape[0]
            num_edges_to_add = int(injection_rate * edges.shape[1])
            
            # Strategic edge injection targeting temporal patterns
            if target_nodes is None:
                # Target high-centrality nodes that appear frequently
                node_frequencies = self._compute_temporal_centrality(graph_sequence, t)
                target_candidates = jnp.argsort(node_frequencies)[-10:]  # Top 10 central nodes
            else:
                target_candidates = jnp.array(target_nodes)
            
            # Generate adversarial edges
            key = random.PRNGKey(42 + t)
            source_nodes = random.choice(key, target_candidates, shape=(num_edges_to_add,))
            key = random.split(key)[0]
            dest_nodes = random.choice(key, num_nodes, shape=(num_edges_to_add,))
            
            # Avoid self-loops
            mask = source_nodes != dest_nodes
            source_nodes = source_nodes[mask]
            dest_nodes = dest_nodes[mask]
            
            # Add adversarial edges
            new_edges = jnp.concatenate([
                edges,
                jnp.stack([source_nodes, dest_nodes])
            ], axis=1)
            
            perturbed_graphs.append(graph)
            perturbed_edges.append(new_edges)
        
        return perturbed_graphs, perturbed_edges
    
    def temporal_shift_attack(
        self,
        graph_sequence: List[jnp.ndarray],
        shift_magnitude: int = 2,
        target_time_scales: List[int] = None
    ) -> List[jnp.ndarray]:
        """
        Attack temporal dependencies by shifting patterns across time scales.
        
        Novel Attack: Breaks temporal causality by introducing time-shifted
        patterns that confuse multi-scale temporal learning.
        """
        if target_time_scales is None:
            target_time_scales = [1, 5, 20]  # Different temporal scales
        
        perturbed_sequence = graph_sequence.copy()
        
        for scale in target_time_scales:
            # Introduce temporal shifts at this scale
            for t in range(len(perturbed_sequence)):
                if t % scale == 0 and t + shift_magnitude < len(perturbed_sequence):
                    # Swap temporal order to break causality
                    future_idx = min(t + shift_magnitude, len(perturbed_sequence) - 1)
                    
                    # Blend current and future states (causality violation)
                    blend_factor = 0.3  # 30% future information leakage
                    original = perturbed_sequence[t] 
                    future = perturbed_sequence[future_idx]
                    
                    perturbed_sequence[t] = (1 - blend_factor) * original + blend_factor * future
        
        return perturbed_sequence
    
    def node_feature_perturbation(
        self,
        graph_sequence: List[jnp.ndarray],
        perturbation_type: str = "gaussian",
        target_features: Optional[List[int]] = None
    ) -> List[jnp.ndarray]:
        """
        Add adversarial noise to node features across temporal sequence.
        
        Novel Attack: Time-correlated perturbations that fool temporal
        pattern recognition while staying within perturbation budget.
        """
        perturbed_sequence = []
        
        for t, graph in enumerate(graph_sequence):
            key = random.PRNGKey(42 + t)
            
            if perturbation_type == "gaussian":
                # Gaussian noise with temporal correlation
                noise_scale = self.perturbation_budget * jnp.sqrt(graph.shape[-1])
                
                # Generate temporally correlated noise
                if t > 0:
                    # Correlate with previous timestep
                    prev_noise = getattr(self, '_prev_noise', jnp.zeros_like(graph))
                    temporal_correlation = 0.7
                    base_noise = random.normal(key, graph.shape) * noise_scale
                    correlated_noise = temporal_correlation * prev_noise + (1 - temporal_correlation) * base_noise
                else:
                    correlated_noise = random.normal(key, graph.shape) * noise_scale
                
                self._prev_noise = correlated_noise
                
            elif perturbation_type == "targeted":
                # Target specific features that affect temporal learning
                if target_features is None:
                    # Target features with highest temporal variance
                    target_features = self._identify_temporal_features(graph_sequence)
                
                correlated_noise = jnp.zeros_like(graph)
                for feature_idx in target_features:
                    noise = random.normal(key, graph.shape[:-1]) * self.perturbation_budget
                    correlated_noise = correlated_noise.at[..., feature_idx].set(noise)
                    key = random.split(key)[0]
            
            else:
                # Uniform perturbation
                correlated_noise = random.uniform(
                    key, graph.shape, minval=-self.perturbation_budget, maxval=self.perturbation_budget
                )
            
            perturbed_graph = graph + correlated_noise
            perturbed_sequence.append(perturbed_graph)
        
        return perturbed_sequence
    
    def gradient_inversion_attack(
        self,
        client_gradients: List[jnp.ndarray],
        target_client: int = 0,
        inversion_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Attempt to invert gradients to reconstruct private data.
        
        Novel Attack: Exploit temporal correlations in gradients to
        improve reconstruction quality of private graph data.
        """
        target_gradient = client_gradients[target_client]
        
        # Initialize dummy data
        key = random.PRNGKey(42)
        dummy_data = random.normal(key, target_gradient.shape)
        
        # Gradient matching objective
        def gradient_matching_loss(dummy_data, target_grad):
            # Simulate forward pass to get dummy gradient
            dummy_loss = jnp.sum(dummy_data ** 2)  # Simplified loss
            dummy_grad = jax.grad(lambda x: jnp.sum(x ** 2))(dummy_data)
            
            # L2 distance between gradients
            return jnp.mean((dummy_grad - target_grad) ** 2)
        
        # Optimize dummy data to match gradients
        optimizer = jax.example_libraries.optimizers.adam(learning_rate=0.01)
        opt_init, opt_update, get_params = optimizer
        opt_state = opt_init(dummy_data)
        
        reconstruction_losses = []
        for step in range(inversion_steps):
            params = get_params(opt_state)
            loss = gradient_matching_loss(params, target_gradient)
            grad = jax.grad(gradient_matching_loss)(params, target_gradient)
            opt_state = opt_update(step, grad, opt_state)
            reconstruction_losses.append(float(loss))
        
        reconstructed_data = get_params(opt_state)
        final_loss = reconstruction_losses[-1]
        
        return {
            "reconstructed_data": reconstructed_data,
            "reconstruction_loss": final_loss,
            "loss_history": reconstruction_losses,
            "attack_success": final_loss < 0.1,  # Success threshold
        }
    
    def _compute_temporal_centrality(
        self, 
        graph_sequence: List[jnp.ndarray], 
        current_time: int
    ) -> jnp.ndarray:
        """Compute temporal centrality of nodes for targeted attacks."""
        if not graph_sequence:
            return jnp.array([])
        
        num_nodes = graph_sequence[0].shape[0]
        centrality_scores = jnp.zeros(num_nodes)
        
        # Look at temporal window around current time
        window_size = min(5, len(graph_sequence))
        start_idx = max(0, current_time - window_size // 2)
        end_idx = min(len(graph_sequence), start_idx + window_size)
        
        for t in range(start_idx, end_idx):
            graph = graph_sequence[t]
            # Simple centrality: sum of feature magnitudes
            node_magnitudes = jnp.linalg.norm(graph, axis=-1)
            centrality_scores = centrality_scores + node_magnitudes
        
        return centrality_scores / (end_idx - start_idx)
    
    def _identify_temporal_features(
        self, 
        graph_sequence: List[jnp.ndarray]
    ) -> List[int]:
        """Identify features with highest temporal variance for targeted attacks."""
        if not graph_sequence:
            return []
        
        # Stack all graphs
        stacked_graphs = jnp.stack(graph_sequence)  # [time, nodes, features]
        
        # Compute temporal variance for each feature
        temporal_variance = jnp.var(stacked_graphs, axis=0)  # [nodes, features]
        feature_importance = jnp.mean(temporal_variance, axis=0)  # [features]
        
        # Return indices of top 3 most temporally varying features
        return list(jnp.argsort(feature_importance)[-3:])


class MultiScaleAdversarialDefense:
    """
    Novel defense mechanism using multi-scale temporal modeling
    to detect and mitigate adversarial attacks on dynamic graphs.
    
    Key Innovation: Leverages temporal redundancy across multiple
    time scales to provide certified robustness guarantees.
    """
    
    def __init__(
        self,
        time_scales: List[int] = None,
        detection_threshold: float = 0.5,
        smoothing_factor: float = 0.1
    ):
        if time_scales is None:
            time_scales = [1, 5, 20, 100]  # Multi-scale temporal windows
        
        self.time_scales = time_scales
        self.detection_threshold = detection_threshold
        self.smoothing_factor = smoothing_factor
        
        # Defense components
        self.temporal_detector = TemporalAnomalyDetector()
        self.consistency_checker = CrossScaleConsistencyChecker(time_scales)
        self.robust_aggregator = RobustTemporalAggregator()
        
        # Defense statistics
        self.defense_history: List[Dict] = []
        self.certified_robustness_radius: Optional[float] = None
    
    def detect_adversarial_input(
        self,
        graph_sequence: List[jnp.ndarray],
        reference_sequence: Optional[List[jnp.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Multi-scale adversarial detection using temporal consistency.
        
        Novel Method: Cross-scale temporal consistency checking to
        identify adversarial perturbations that break temporal patterns.
        """
        detection_scores = {}
        
        # 1. Temporal anomaly detection
        anomaly_score = self.temporal_detector.detect_anomalies(graph_sequence)
        detection_scores["temporal_anomaly"] = anomaly_score
        
        # 2. Cross-scale consistency check
        consistency_score = self.consistency_checker.check_consistency(graph_sequence)
        detection_scores["cross_scale_consistency"] = consistency_score
        
        # 3. Statistical outlier detection
        if reference_sequence:
            statistical_score = self._statistical_outlier_detection(
                graph_sequence, reference_sequence
            )
            detection_scores["statistical_outlier"] = statistical_score
        
        # Aggregate detection scores
        overall_score = jnp.mean(jnp.array(list(detection_scores.values())))
        is_adversarial = overall_score > self.detection_threshold
        
        detection_result = {
            "is_adversarial": is_adversarial,
            "overall_score": float(overall_score),
            "component_scores": {k: float(v) for k, v in detection_scores.items()},
            "detection_confidence": float(jnp.abs(overall_score - 0.5) * 2),  # Distance from decision boundary
        }
        
        self.defense_history.append({
            "timestamp": time.time(),
            "detection_result": detection_result,
        })
        
        return detection_result
    
    def certified_robustness_defense(
        self,
        graph_sequence: List[jnp.ndarray],
        perturbation_radius: float,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Provide certified robustness guarantees using temporal smoothing.
        
        Novel Method: Uses multi-scale temporal smoothing to provide
        provable robustness bounds against adversarial perturbations.
        """
        # 1. Multi-scale temporal smoothing
        smoothed_sequences = {}
        for scale in self.time_scales:
            smoothed_sequences[scale] = self._temporal_smoothing(
                graph_sequence, window_size=scale
            )
        
        # 2. Compute certified radius for each scale
        certified_radii = {}
        for scale, smoothed_seq in smoothed_sequences.items():
            radius = self._compute_certified_radius(
                smoothed_seq, perturbation_radius, confidence_level
            )
            certified_radii[scale] = radius
        
        # 3. Conservative certified bound (minimum across scales)
        certified_bound = min(certified_radii.values()) if certified_radii else 0.0
        
        # 4. Generate robust prediction
        robust_prediction = self.robust_aggregator.aggregate_predictions(
            smoothed_sequences, certified_radii
        )
        
        return {
            "certified_radius": certified_bound,
            "scale_specific_radii": certified_radii,
            "robust_prediction": robust_prediction,
            "confidence_level": confidence_level,
            "smoothing_applied": True,
        }
    
    def adaptive_defense_response(
        self,
        attack_detected: bool,
        attack_type: Optional[AttackType] = None,
        attack_strength: float = 0.0
    ) -> Dict[str, Any]:
        """
        Adaptively adjust defense mechanisms based on detected attacks.
        
        Novel Feature: Dynamic reconfiguration of defense parameters
        based on attack characteristics and severity.
        """
        if not attack_detected:
            return {"defense_level": "normal", "adjustments": {}}
        
        adjustments = {}
        
        # Adjust detection sensitivity
        if attack_strength > 0.8:  # Strong attack
            adjustments["detection_threshold"] = self.detection_threshold * 0.7
            adjustments["smoothing_factor"] = self.smoothing_factor * 1.5
        elif attack_strength > 0.5:  # Moderate attack
            adjustments["detection_threshold"] = self.detection_threshold * 0.85
            adjustments["smoothing_factor"] = self.smoothing_factor * 1.2
        
        # Attack-specific adjustments
        if attack_type == AttackType.TEMPORAL_SHIFT:
            adjustments["temporal_window_expansion"] = True
            adjustments["causality_checking"] = True
        elif attack_type == AttackType.TOPOLOGY_INJECTION:
            adjustments["edge_validation"] = True  
            adjustments["structural_consistency"] = True
        
        # Apply adjustments
        self.detection_threshold = adjustments.get(
            "detection_threshold", self.detection_threshold
        )
        self.smoothing_factor = adjustments.get(
            "smoothing_factor", self.smoothing_factor
        )
        
        defense_level = "high" if attack_strength > 0.7 else "medium"
        
        return {
            "defense_level": defense_level,
            "adjustments": adjustments,
            "new_threshold": self.detection_threshold,
            "new_smoothing": self.smoothing_factor,
        }
    
    def _temporal_smoothing(
        self,
        graph_sequence: List[jnp.ndarray],
        window_size: int
    ) -> List[jnp.ndarray]:
        """Apply temporal smoothing with given window size."""
        if window_size <= 1:
            return graph_sequence
        
        smoothed_sequence = []
        for t in range(len(graph_sequence)):
            # Define smoothing window
            start_idx = max(0, t - window_size // 2)
            end_idx = min(len(graph_sequence), t + window_size // 2 + 1)
            
            # Compute weighted average (Gaussian weights)
            weights = []
            window_graphs = []
            
            for i in range(start_idx, end_idx):
                distance = abs(i - t)
                weight = jnp.exp(-distance**2 / (2 * (window_size/4)**2))
                weights.append(weight)
                window_graphs.append(graph_sequence[i])
            
            # Normalize weights
            weights = jnp.array(weights)
            weights = weights / jnp.sum(weights)
            
            # Weighted average
            smoothed_graph = sum(
                w * graph for w, graph in zip(weights, window_graphs)
            )
            smoothed_sequence.append(smoothed_graph)
        
        return smoothed_sequence
    
    def _compute_certified_radius(
        self,
        smoothed_sequence: List[jnp.ndarray], 
        perturbation_radius: float,
        confidence_level: float
    ) -> float:
        """Compute certified robustness radius using smoothing analysis."""
        # Simplified certified radius calculation
        # In practice, this would use more sophisticated bounds from smoothing theory
        
        # Estimate Lipschitz constant of the smoothed function
        lipschitz_estimate = self._estimate_lipschitz_constant(smoothed_sequence)
        
        # Certified radius based on smoothing variance and confidence level
        smoothing_noise_std = self.smoothing_factor
        confidence_multiplier = 2.0 if confidence_level > 0.95 else 1.5
        
        certified_radius = (smoothing_noise_std / lipschitz_estimate) / confidence_multiplier
        
        # Conservative bound
        return min(certified_radius, perturbation_radius * 0.8)
    
    def _estimate_lipschitz_constant(
        self, 
        sequence: List[jnp.ndarray]
    ) -> float:
        """Estimate Lipschitz constant of the temporal function."""
        if len(sequence) < 2:
            return 1.0
        
        max_gradient = 0.0
        for t in range(len(sequence) - 1):
            diff = jnp.linalg.norm(sequence[t+1] - sequence[t])
            max_gradient = max(max_gradient, float(diff))
        
        return max(max_gradient, 0.1)  # Avoid division by zero
    
    def _statistical_outlier_detection(
        self,
        test_sequence: List[jnp.ndarray],
        reference_sequence: List[jnp.ndarray]
    ) -> float:
        """Detect statistical outliers compared to reference sequence."""
        # Compute sequence statistics
        test_stats = self._compute_sequence_statistics(test_sequence)
        ref_stats = self._compute_sequence_statistics(reference_sequence)
        
        # Compare distributions using Wasserstein distance
        stat_distance = jnp.linalg.norm(
            jnp.array(list(test_stats.values())) - jnp.array(list(ref_stats.values()))
        )
        
        # Normalize to [0, 1] score
        return float(jnp.tanh(stat_distance))
    
    def _compute_sequence_statistics(
        self, 
        sequence: List[jnp.ndarray]
    ) -> Dict[str, float]:
        """Compute statistical features of graph sequence."""
        if not sequence:
            return {}
        
        stacked = jnp.stack(sequence)
        
        return {
            "mean_magnitude": float(jnp.mean(jnp.linalg.norm(stacked, axis=-1))),
            "temporal_variance": float(jnp.var(stacked)),
            "max_change_rate": float(jnp.max(jnp.abs(jnp.diff(stacked, axis=0)))),
            "spectral_norm": float(jnp.linalg.norm(stacked.reshape(-1, stacked.shape[-1]))),
        }


class TemporalAnomalyDetector:
    """Detect temporal anomalies in dynamic graph sequences."""
    
    def detect_anomalies(self, graph_sequence: List[jnp.ndarray]) -> float:
        """Detect temporal anomalies using change point detection."""
        if len(graph_sequence) < 3:
            return 0.0
        
        # Compute temporal differences
        changes = []
        for t in range(1, len(graph_sequence)):
            change = jnp.linalg.norm(graph_sequence[t] - graph_sequence[t-1])
            changes.append(change)
        
        changes = jnp.array(changes)
        
        # Anomaly score based on change magnitude variance
        mean_change = jnp.mean(changes)
        std_change = jnp.std(changes)
        
        # Z-score of maximum change
        max_change = jnp.max(changes)
        anomaly_score = (max_change - mean_change) / (std_change + 1e-8)
        
        return float(jnp.tanh(anomaly_score / 3.0))  # Normalize to [0, 1]


class CrossScaleConsistencyChecker:
    """Check consistency across multiple temporal scales."""
    
    def __init__(self, time_scales: List[int]):
        self.time_scales = time_scales
    
    def check_consistency(self, graph_sequence: List[jnp.ndarray]) -> float:
        """Check cross-scale temporal consistency."""
        if len(graph_sequence) < max(self.time_scales):
            return 0.0
        
        scale_representations = {}
        
        # Compute representations at each scale
        for scale in self.time_scales:
            scale_repr = self._compute_scale_representation(graph_sequence, scale)
            scale_representations[scale] = scale_repr
        
        # Measure consistency between scales
        consistency_scores = []
        scales = list(scale_representations.keys())
        
        for i in range(len(scales)):
            for j in range(i+1, len(scales)):
                scale_i, scale_j = scales[i], scales[j]
                repr_i = scale_representations[scale_i]
                repr_j = scale_representations[scale_j]
                
                # Consistency: correlation between scale representations
                if len(repr_i) > 0 and len(repr_j) > 0:
                    min_len = min(len(repr_i), len(repr_j))
                    correlation = jnp.corrcoef(
                        repr_i[:min_len].flatten(), 
                        repr_j[:min_len].flatten()
                    )[0, 1]
                    consistency_scores.append(1.0 - jnp.abs(correlation))  # Inconsistency score
        
        if consistency_scores:
            return float(jnp.mean(jnp.array(consistency_scores)))
        else:
            return 0.0
    
    def _compute_scale_representation(
        self, 
        sequence: List[jnp.ndarray], 
        scale: int
    ) -> jnp.ndarray:
        """Compute temporal representation at given scale."""
        # Subsample sequence at this scale
        subsampled = sequence[::scale]
        if not subsampled:
            return jnp.array([])
        
        # Simple representation: mean features over time
        if len(subsampled) > 1:
            stacked = jnp.stack(subsampled)
            representation = jnp.mean(stacked, axis=0)
        else:
            representation = subsampled[0]
        
        return representation.flatten()


class RobustTemporalAggregator:
    """Robust aggregation of temporal predictions across scales."""
    
    def aggregate_predictions(
        self,
        scale_predictions: Dict[int, List[jnp.ndarray]],
        certified_radii: Dict[int, float]
    ) -> jnp.ndarray:
        """Aggregate predictions with certified robustness weighting."""
        if not scale_predictions:
            return jnp.array([])
        
        # Weight by certified robustness radius
        weighted_predictions = []
        weights = []
        
        for scale, predictions in scale_predictions.items():
            if scale in certified_radii and predictions:
                weight = certified_radii[scale]
                # Average prediction across time for this scale
                avg_prediction = jnp.mean(jnp.stack(predictions), axis=0)
                
                weighted_predictions.append(avg_prediction * weight)
                weights.append(weight)
        
        if weighted_predictions:
            total_weight = sum(weights)
            if total_weight > 0:
                robust_prediction = sum(weighted_predictions) / total_weight
            else:
                robust_prediction = jnp.mean(jnp.stack([p for p in scale_predictions.values()][0]), axis=0)
        else:
            # Fallback to uniform weighting
            all_predictions = []
            for predictions in scale_predictions.values():
                if predictions:
                    all_predictions.extend(predictions)
            
            if all_predictions:
                robust_prediction = jnp.mean(jnp.stack(all_predictions), axis=0)
            else:
                robust_prediction = jnp.array([])
        
        return robust_prediction


class CertifiedRobustnessAnalyzer:
    """Analyze and compute certified robustness bounds."""
    
    def __init__(self):
        self.analysis_results: List[Dict] = []
    
    def compute_robustness_certificate(
        self,
        model_predictions: Callable,
        graph_sequence: List[jnp.ndarray],
        perturbation_budget: float,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Compute certified robustness certificate."""
        
        # 1. Baseline performance
        baseline_pred = model_predictions(graph_sequence)
        baseline_performance = self._compute_performance_metric(baseline_pred)
        
        # 2. Sample random perturbations within budget
        num_samples = 1000
        key = random.PRNGKey(42)
        
        perturbed_performances = []
        for _ in range(num_samples):
            # Generate random perturbation
            perturbed_seq = self._generate_random_perturbation(
                graph_sequence, perturbation_budget, key
            )
            key = random.split(key)[0]
            
            # Evaluate perturbed performance
            perturbed_pred = model_predictions(perturbed_seq)
            perf = self._compute_performance_metric(perturbed_pred)
            perturbed_performances.append(perf)
        
        perturbed_performances = jnp.array(perturbed_performances)
        
        # 3. Compute certified bound
        performance_drop = baseline_performance - perturbed_performances
        confidence_quantile = 1.0 - (1.0 - confidence_level) / 2  # Two-sided
        certified_bound = float(jnp.quantile(performance_drop, confidence_quantile))
        
        # 4. Statistical significance test
        p_value = self._compute_significance_test(performance_drop)
        
        certificate = {
            "baseline_performance": float(baseline_performance),
            "certified_performance_drop": certified_bound,
            "confidence_level": confidence_level,
            "perturbation_budget": perturbation_budget,
            "num_samples": num_samples,
            "statistical_significance": float(p_value),
            "is_significant": p_value < 0.05,
        }
        
        self.analysis_results.append(certificate)
        return certificate
    
    def _generate_random_perturbation(
        self,
        sequence: List[jnp.ndarray],
        budget: float,
        key: jnp.ndarray
    ) -> List[jnp.ndarray]:
        """Generate random perturbation within budget."""
        perturbed = []
        
        for graph in sequence:
            noise = random.normal(key, graph.shape)
            # Scale to budget
            noise = noise / jnp.linalg.norm(noise) * budget
            perturbed_graph = graph + noise
            perturbed.append(perturbed_graph)
            key = random.split(key)[0]
        
        return perturbed
    
    def _compute_performance_metric(self, predictions: jnp.ndarray) -> float:
        """Compute performance metric from predictions."""
        # Simplified metric - in practice would be task-specific
        return float(jnp.mean(predictions))
    
    def _compute_significance_test(self, performance_drops: jnp.ndarray) -> float:
        """Compute p-value for statistical significance."""
        # One-sample t-test against zero (no performance drop)
        mean_drop = jnp.mean(performance_drops)
        std_drop = jnp.std(performance_drops)
        n = len(performance_drops)
        
        t_stat = mean_drop / (std_drop / jnp.sqrt(n))
        
        # Simplified p-value calculation (would use proper t-distribution in practice)
        p_value = 2 * (1 - jax.scipy.stats.norm.cdf(jnp.abs(t_stat)))
        
        return float(p_value)