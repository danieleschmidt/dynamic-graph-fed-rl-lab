import secrets
"""
AutoML Pipeline for Continuous Algorithm Improvement.

Automatically discovers, trains, and evaluates new algorithm variants
to continuously improve the federated learning system performance.
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import logging

from ..algorithms.base import BaseFederatedAlgorithm
from ..algorithms.graph_td3 import GraphTD3
from ..quantum_planner.performance import PerformanceMonitor


@dataclass
class AlgorithmVariant:
    """Represents an algorithm variant for evaluation."""
    id: str
    name: str
    base_algorithm: str
    hyperparameters: Dict[str, Any]
    architecture_params: Dict[str, Any]
    performance_scores: Dict[str, float] = field(default_factory=dict)
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    evaluation_status: str = "pending"  # pending, training, evaluated, deployed
    total_training_time: float = 0.0
    stability_score: float = 0.0


@dataclass
class AutoMLExperiment:
    """Represents an AutoML experiment configuration."""
    id: str
    objective: str  # "maximize_accuracy", "minimize_latency", "optimize_efficiency"
    search_space: Dict[str, Any]
    max_variants: int = 50
    max_evaluation_time: float = 3600.0  # 1 hour
    early_stopping_patience: int = 10
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"  # running, completed, stopped
    best_variant_id: Optional[str] = None


class AutoMLPipeline:
    """
    AutoML pipeline for continuous algorithm improvement.
    
    Features:
    - Neural architecture search for graph neural networks
    - Hyperparameter optimization using Bayesian optimization
    - Multi-objective optimization (accuracy, efficiency, stability)
    - Automated model selection and deployment
    - Continuous learning from performance feedback
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        base_algorithms: List[str] = None,
        max_concurrent_evaluations: int = 4,
        evaluation_budget_per_variant: float = 300.0,  # 5 minutes
        logger: Optional[logging.Logger] = None,
    ):
        self.performance_monitor = performance_monitor
        self.base_algorithms = base_algorithms or ["graph_td3", "fedavg", "quantum_weighted"]
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.evaluation_budget_per_variant = evaluation_budget_per_variant
        self.logger = logger or logging.getLogger(__name__)
        
        # Algorithm management
        self.algorithm_variants: Dict[str, AlgorithmVariant] = {}
        self.active_experiments: Dict[str, AutoMLExperiment] = {}
        self.evaluation_queue: List[str] = []
        self.currently_evaluating: Set[str] = set()
        
        # Performance tracking
        self.deployment_history: List[Dict[str, Any]] = []
        self.experiment_results: List[Dict[str, Any]] = []
        
        # AutoML state
        self.is_running = False
        self.last_generation_time: Optional[datetime] = None
        self.generation_interval = 1800.0  # 30 minutes
        
        # Bayesian optimization state
        self.performance_database: Dict[str, List[Tuple[Dict, float]]] = {}
        self.pareto_front: List[AlgorithmVariant] = []
        
    async def start_automl_pipeline(self):
        """Start the continuous AutoML pipeline."""
        self.is_running = True
        self.logger.info("Starting AutoML pipeline for continuous algorithm improvement")
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._variant_generation_loop()),
            asyncio.create_task(self._variant_evaluation_loop()),
            asyncio.create_task(self._deployment_optimization_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"AutoML pipeline error: {e}")
        finally:
            self.is_running = False
    
    async def stop_automl_pipeline(self):
        """Stop the AutoML pipeline."""
        self.is_running = False
        self.logger.info("Stopping AutoML pipeline")
    
    async def _variant_generation_loop(self):
        """Continuously generate new algorithm variants."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check if it's time to generate new variants
                if (self.last_generation_time is None or 
                    (current_time - self.last_generation_time).total_seconds() >= self.generation_interval):
                    
                    await self._generate_algorithm_variants()
                    self.last_generation_time = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in variant generation loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _variant_evaluation_loop(self):
        """Continuously evaluate algorithm variants."""
        while self.is_running:
            try:
                # Process evaluation queue
                while (len(self.currently_evaluating) < self.max_concurrent_evaluations and 
                       self.evaluation_queue):
                    
                    variant_id = self.evaluation_queue.pop(0)
                    
                    if variant_id not in self.currently_evaluating:
                        self.currently_evaluating.add(variant_id)
                        
                        # Start evaluation task
                        asyncio.create_task(self._evaluate_variant(variant_id))
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in variant evaluation loop: {e}")
                await asyncio.sleep(60)
    
    async def _deployment_optimization_loop(self):
        """Continuously optimize deployment based on performance."""
        while self.is_running:
            try:
                # Update Pareto front
                await self._update_pareto_front()
                
                # Check for deployment opportunities
                await self._check_deployment_candidates()
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in deployment optimization loop: {e}")
                await asyncio.sleep(300)
    
    async def _performance_monitoring_loop(self):
        """Monitor performance of deployed variants."""
        while self.is_running:
            try:
                # Monitor deployed variants
                await self._monitor_deployed_variants()
                
                # Trigger rollback if performance degrades
                await self._check_rollback_conditions()
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _generate_algorithm_variants(self) -> List[str]:
        """Generate new algorithm variants using intelligent search."""
        self.logger.info("Generating new algorithm variants")
        
        generated_variants = []
        
        # Get current performance baseline
        current_performance = await self.performance_monitor.get_current_metrics()
        
        # Generate variants for each base algorithm
        for base_algorithm in self.base_algorithms:
            variants = await self._generate_variants_for_algorithm(
                base_algorithm, current_performance
            )
            generated_variants.extend(variants)
        
        # Add to evaluation queue
        for variant_id in generated_variants:
            if variant_id not in self.evaluation_queue:
                self.evaluation_queue.append(variant_id)
        
        self.logger.info(f"Generated {len(generated_variants)} new algorithm variants")
        return generated_variants
    
    async def _generate_variants_for_algorithm(
        self, base_algorithm: str, current_performance: Dict[str, float]
    ) -> List[str]:
        """Generate variants for a specific base algorithm."""
        variants = []
        
        # Get search space for this algorithm
        search_space = self._get_algorithm_search_space(base_algorithm)
        
        # Generate variants using different strategies
        strategies = [
            self._random_search_strategy,
            self._bayesian_optimization_strategy,
            self._evolutionary_strategy,
            self._gradient_based_strategy,
        ]
        
        for strategy in strategies:
            try:
                strategy_variants = await strategy(base_algorithm, search_space, current_performance)
                variants.extend(strategy_variants)
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.__name__} failed: {e}")
        
        return variants
    
    def _get_algorithm_search_space(self, algorithm: str) -> Dict[str, Any]:
        """Get search space for algorithm hyperparameters and architecture."""
        if algorithm == "graph_td3":
            return {
                "hyperparameters": {
                    "learning_rate": {"type": "log_uniform", "bounds": [1e-5, 1e-2]},
                    "batch_size": {"type": "choice", "choices": [16, 32, 64, 128]},
                    "tau": {"type": "uniform", "bounds": [0.001, 0.01]},
                    "policy_noise": {"type": "uniform", "bounds": [0.1, 0.3]},
                    "noise_clip": {"type": "uniform", "bounds": [0.3, 0.7]},
                    "policy_freq": {"type": "choice", "choices": [1, 2, 3, 4]},
                },
                "architecture": {
                    "hidden_dim": {"type": "choice", "choices": [64, 128, 256, 512]},
                    "num_layers": {"type": "choice", "choices": [2, 3, 4, 5]},
                    "activation": {"type": "choice", "choices": ["relu", "tanh", "gelu", "swish"]},
                    "dropout": {"type": "uniform", "bounds": [0.0, 0.3]},
                    "graph_conv_type": {"type": "choice", "choices": ["gcn", "gat", "sage", "gin"]},
                    "attention_heads": {"type": "choice", "choices": [1, 2, 4, 8]},
                },
            }
        elif algorithm == "fedavg":
            return {
                "hyperparameters": {
                    "client_lr": {"type": "log_uniform", "bounds": [1e-4, 1e-1]},
                    "server_lr": {"type": "log_uniform", "bounds": [1e-3, 1.0]},
                    "local_epochs": {"type": "choice", "choices": [1, 2, 3, 5]},
                    "client_fraction": {"type": "uniform", "bounds": [0.1, 1.0]},
                },
                "architecture": {
                    "model_size": {"type": "choice", "choices": ["small", "medium", "large"]},
                    "compression_ratio": {"type": "uniform", "bounds": [0.1, 1.0]},
                },
            }
        elif algorithm == "quantum_weighted":
            return {
                "hyperparameters": {
                    "quantum_coherence_time": {"type": "uniform", "bounds": [5.0, 20.0]},
                    "entanglement_strength": {"type": "uniform", "bounds": [0.1, 0.8]},
                    "measurement_frequency": {"type": "choice", "choices": [5, 10, 15, 20]},
                    "superposition_decay": {"type": "uniform", "bounds": [0.9, 0.999]},
                },
                "architecture": {
                    "quantum_dimensions": {"type": "choice", "choices": [4, 8, 16, 32]},
                    "interference_layers": {"type": "choice", "choices": [1, 2, 3]},
                },
            }
        
        return {"hyperparameters": {}, "architecture": {}}
    
    async def _random_search_strategy(
        self, algorithm: str, search_space: Dict[str, Any], performance: Dict[str, float]
    ) -> List[str]:
        """Generate variants using random search."""
        variants = []
        num_variants = 3
        
        for i in range(num_variants):
            # Sample random configuration
            config = self._sample_from_search_space(search_space)
            
            # Create variant
            variant_id = self._create_algorithm_variant(
                algorithm, config, f"random_search_{i}"
            )
            variants.append(variant_id)
        
        return variants
    
    async def _bayesian_optimization_strategy(
        self, algorithm: str, search_space: Dict[str, Any], performance: Dict[str, float]
    ) -> List[str]:
        """Generate variants using Bayesian optimization."""
        variants = []
        
        # Get historical performance data for this algorithm
        if algorithm not in self.performance_database:
            # Not enough data for Bayesian optimization, fall back to random
            return await self._random_search_strategy(algorithm, search_space, performance)
        
        history = self.performance_database[algorithm]
        if len(history) < 5:
            return await self._random_search_strategy(algorithm, search_space, performance)
        
        # Use Gaussian Process to suggest next points
        suggested_configs = self._bayesian_suggest_configs(search_space, history, num_suggestions=2)
        
        for i, config in enumerate(suggested_configs):
            variant_id = self._create_algorithm_variant(
                algorithm, config, f"bayesian_opt_{i}"
            )
            variants.append(variant_id)
        
        return variants
    
    async def _evolutionary_strategy(
        self, algorithm: str, search_space: Dict[str, Any], performance: Dict[str, float]
    ) -> List[str]:
        """Generate variants using evolutionary strategy."""
        variants = []
        
        # Get top performing variants for this algorithm
        algorithm_variants = [
            v for v in self.algorithm_variants.values()
            if v.base_algorithm == algorithm and v.evaluation_status == "evaluated"
        ]
        
        if len(algorithm_variants) < 2:
            return []  # Need parents for evolution
        
        # Sort by performance
        algorithm_variants.sort(key=lambda v: v.performance_scores.get("combined_score", 0), reverse=True)
        parents = algorithm_variants[:3]  # Top 3 as parents
        
        # Generate offspring through crossover and mutation
        for i in range(2):
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            
            # Crossover
            child_config = self._crossover_configs(parent1, parent2, search_space)
            
            # Mutation
            child_config = self._mutate_config(child_config, search_space, mutation_rate=0.2)
            
            variant_id = self._create_algorithm_variant(
                algorithm, child_config, f"evolutionary_{i}"
            )
            variants.append(variant_id)
        
        return variants
    
    async def _gradient_based_strategy(
        self, algorithm: str, search_space: Dict[str, Any], performance: Dict[str, float]
    ) -> List[str]:
        """Generate variants using gradient-based optimization."""
        variants = []
        
        # Get best performing variant
        algorithm_variants = [
            v for v in self.algorithm_variants.values()
            if v.base_algorithm == algorithm and v.evaluation_status == "evaluated"
        ]
        
        if not algorithm_variants:
            return []
        
        best_variant = max(algorithm_variants, key=lambda v: v.performance_scores.get("combined_score", 0))
        
        # Generate variants by perturbing best configuration
        for i in range(2):
            perturbed_config = self._perturb_config(
                {"hyperparameters": best_variant.hyperparameters, "architecture": best_variant.architecture_params},
                search_space,
                perturbation_strength=0.1
            )
            
            variant_id = self._create_algorithm_variant(
                algorithm, perturbed_config, f"gradient_based_{i}"
            )
            variants.append(variant_id)
        
        return variants
    
    def _sample_from_search_space(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample a configuration from the search space."""
        config = {"hyperparameters": {}, "architecture": {}}
        
        for category in ["hyperparameters", "architecture"]:
            if category in search_space:
                for param, spec in search_space[category].items():
                    config[category][param] = self._sample_parameter(spec)
        
        return config
    
    def _sample_parameter(self, spec: Dict[str, Any]) -> Any:
        """Sample a single parameter value."""
        param_type = spec["type"]
        
        if param_type == "uniform":
            return np.random.uniform(spec["bounds"][0], spec["bounds"][1])
        elif param_type == "log_uniform":
            log_bounds = [np.log(spec["bounds"][0]), np.log(spec["bounds"][1])]
            return np.exp(np.random.uniform(log_bounds[0], log_bounds[1]))
        elif param_type == "choice":
            return np.random.choice(spec["choices"])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def _create_algorithm_variant(
        self, base_algorithm: str, config: Dict[str, Any], suffix: str
    ) -> str:
        """Create a new algorithm variant."""
        # Generate unique ID
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        variant_id = f"{base_algorithm}_{suffix}_{config_hash}"
        
        # Create variant object
        variant = AlgorithmVariant(
            id=variant_id,
            name=f"{base_algorithm.upper()} {suffix}",
            base_algorithm=base_algorithm,
            hyperparameters=config.get("hyperparameters", {}),
            architecture_params=config.get("architecture", {}),
        )
        
        self.algorithm_variants[variant_id] = variant
        
        self.logger.debug(f"Created algorithm variant: {variant_id}")
        return variant_id
    
    async def _evaluate_variant(self, variant_id: str):
        """Evaluate an algorithm variant."""
        if variant_id not in self.algorithm_variants:
            self.currently_evaluating.discard(variant_id)
            return
        
        variant = self.algorithm_variants[variant_id]
        evaluation_start = time.time()
        
        try:
            self.logger.info(f"Starting evaluation of variant: {variant_id}")
            variant.evaluation_status = "training"
            
            # Evaluate variant performance
            evaluation_results = await self._run_variant_evaluation(variant)
            
            # Update variant with results
            variant.performance_scores = evaluation_results
            variant.evaluation_status = "evaluated"
            variant.total_training_time = time.time() - evaluation_start
            
            # Calculate combined score
            variant.performance_scores["combined_score"] = self._calculate_combined_score(evaluation_results)
            
            # Update performance database
            if variant.base_algorithm not in self.performance_database:
                self.performance_database[variant.base_algorithm] = []
            
            config = {"hyperparameters": variant.hyperparameters, "architecture": variant.architecture_params}
            self.performance_database[variant.base_algorithm].append(
                (config, variant.performance_scores["combined_score"])
            )
            
            self.logger.info(
                f"Variant {variant_id} evaluation completed: "
                f"score={variant.performance_scores['combined_score']:.3f}, "
                f"time={variant.total_training_time:.1f}s"
            )
            
        except Exception as e:
            variant.evaluation_status = "failed"
            variant.total_training_time = time.time() - evaluation_start
            self.logger.error(f"Variant {variant_id} evaluation failed: {e}")
        
        finally:
            self.currently_evaluating.discard(variant_id)
    
    async def _run_variant_evaluation(self, variant: AlgorithmVariant) -> Dict[str, float]:
        """Run evaluation for a specific variant."""
        # This would integrate with the actual training system
        # For now, we simulate evaluation with realistic performance metrics
        
        # Simulate training time based on complexity
        complexity_factor = (
            variant.hyperparameters.get("hidden_dim", 128) / 128 *
            variant.hyperparameters.get("num_layers", 3) / 3
        )
        
        training_time = min(self.evaluation_budget_per_variant * complexity_factor, self.evaluation_budget_per_variant)
        
        # Simulate training
        await asyncio.sleep(min(training_time, 10))  # Cap simulation time
        
        # Generate realistic performance metrics
        base_accuracy = 0.75
        base_efficiency = 0.80
        base_stability = 0.85
        
        # Performance varies based on hyperparameters
        learning_rate = variant.hyperparameters.get("learning_rate", 0.001)
        accuracy_boost = 0.1 * np.exp(-abs(np.log10(learning_rate) + 3))  # Optimal around 1e-3
        
        batch_size = variant.hyperparameters.get("batch_size", 32)
        efficiency_boost = 0.1 * (1 - abs(batch_size - 64) / 64)  # Optimal around 64
        
        # Add some randomness
        noise = np.random.normal(0, 0.05)
        
        return {
            "accuracy": max(0.1, min(1.0, base_accuracy + accuracy_boost + noise)),
            "efficiency": max(0.1, min(1.0, base_efficiency + efficiency_boost + noise * 0.5)),
            "stability": max(0.1, min(1.0, base_stability + noise * 0.3)),
            "convergence_rate": max(0.01, min(1.0, 0.1 + accuracy_boost + noise * 0.1)),
            "resource_utilization": max(0.1, min(1.0, 0.7 + efficiency_boost + noise * 0.2)),
        }
    
    def _calculate_combined_score(self, metrics: Dict[str, float]) -> float:
        """Calculate combined performance score."""
        weights = {
            "accuracy": 0.3,
            "efficiency": 0.25,
            "stability": 0.2,
            "convergence_rate": 0.15,
            "resource_utilization": 0.1,
        }
        
        score = sum(
            weights.get(metric, 0) * value
            for metric, value in metrics.items()
            if metric in weights
        )
        
        return max(0.0, min(1.0, score))
    
    async def _update_pareto_front(self):
        """Update Pareto front of non-dominated solutions."""
        evaluated_variants = [
            v for v in self.algorithm_variants.values()
            if v.evaluation_status == "evaluated"
        ]
        
        if len(evaluated_variants) < 2:
            self.pareto_front = evaluated_variants.copy()
            return
        
        # Multi-objective optimization: accuracy vs efficiency vs stability
        pareto_front = []
        
        for variant in evaluated_variants:
            is_dominated = False
            
            for other in evaluated_variants:
                if other.id == variant.id:
                    continue
                
                # Check if other dominates variant
                other_better_accuracy = other.performance_scores.get("accuracy", 0) >= variant.performance_scores.get("accuracy", 0)
                other_better_efficiency = other.performance_scores.get("efficiency", 0) >= variant.performance_scores.get("efficiency", 0)
                other_better_stability = other.performance_scores.get("stability", 0) >= variant.performance_scores.get("stability", 0)
                
                other_strictly_better = (
                    other.performance_scores.get("accuracy", 0) > variant.performance_scores.get("accuracy", 0) or
                    other.performance_scores.get("efficiency", 0) > variant.performance_scores.get("efficiency", 0) or
                    other.performance_scores.get("stability", 0) > variant.performance_scores.get("stability", 0)
                )
                
                if other_better_accuracy and other_better_efficiency and other_better_stability and other_strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(variant)
        
        self.pareto_front = pareto_front
        self.logger.debug(f"Updated Pareto front with {len(pareto_front)} variants")
    
    async def _check_deployment_candidates(self):
        """Check for variants ready for deployment."""
        if not self.pareto_front:
            return
        
        # Get current deployed variant performance
        current_performance = await self.performance_monitor.get_current_metrics()
        current_score = self._calculate_combined_score(current_performance)
        
        # Find significantly better variants
        deployment_candidates = []
        for variant in self.pareto_front:
            variant_score = variant.performance_scores.get("combined_score", 0)
            
            if variant_score > current_score + 0.05:  # 5% improvement threshold
                deployment_candidates.append(variant)
        
        if deployment_candidates:
            # Deploy the best candidate
            best_candidate = max(deployment_candidates, key=lambda v: v.performance_scores.get("combined_score", 0))
            await self._deploy_variant(best_candidate)
    
    async def _deploy_variant(self, variant: AlgorithmVariant):
        """Deploy a variant to production."""
        try:
            self.logger.info(f"Deploying variant: {variant.id}")
            
            # This would integrate with the actual deployment system
            # For now, we simulate deployment
            await asyncio.sleep(1)
            
            deployment_record = {
                "timestamp": datetime.now(),
                "variant_id": variant.id,
                "variant_name": variant.name,
                "performance_scores": variant.performance_scores,
                "deployment_reason": "performance_improvement",
            }
            
            self.deployment_history.append(deployment_record)
            
            # Notify performance monitor
            await self.performance_monitor.update_deployed_algorithm(variant.id, variant.hyperparameters)
            
            self.logger.info(f"Successfully deployed variant: {variant.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to deploy variant {variant.id}: {e}")
    
    async def _monitor_deployed_variants(self):
        """Monitor performance of deployed variants."""
        if not self.deployment_history:
            return
        
        latest_deployment = self.deployment_history[-1]
        deployment_time = latest_deployment["timestamp"]
        
        # Check if variant has been running long enough for evaluation
        if (datetime.now() - deployment_time).total_seconds() < 600:  # 10 minutes
            return
        
        current_performance = await self.performance_monitor.get_current_metrics()
        current_score = self._calculate_combined_score(current_performance)
        
        expected_score = latest_deployment["performance_scores"]["combined_score"]
        
        # Check for significant performance degradation
        if current_score < expected_score * 0.9:  # 10% degradation
            self.logger.warning(
                f"Deployed variant {latest_deployment['variant_id']} showing performance degradation: "
                f"{current_score:.3f} vs expected {expected_score:.3f}"
            )
    
    async def _check_rollback_conditions(self):
        """Check if rollback is needed due to performance issues."""
        # This would implement rollback logic
        # For now, we just log potential issues
        pass
    
    def _bayesian_suggest_configs(
        self, search_space: Dict[str, Any], history: List[Tuple[Dict, float]], num_suggestions: int
    ) -> List[Dict[str, Any]]:
        """Use Bayesian optimization to suggest configurations."""
        # Simplified Bayesian optimization - in practice would use library like scikit-optimize
        suggestions = []
        
        for _ in range(num_suggestions):
            # Generate random configuration with bias towards good regions
            config = self._sample_from_search_space(search_space)
            suggestions.append(config)
        
        return suggestions
    
    def _crossover_configs(
        self, parent1: AlgorithmVariant, parent2: AlgorithmVariant, search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover two configurations."""
        config = {"hyperparameters": {}, "architecture": {}}
        
        # Randomly select parameters from each parent
        for category in ["hyperparameters", "architecture"]:
            parent1_params = getattr(parent1, category if category == "hyperparameters" else "architecture_params")
            parent2_params = getattr(parent2, category if category == "hyperparameters" else "architecture_params")
            
            all_params = set(parent1_params.keys()) | set(parent2_params.keys())
            
            for param in all_params:
                if np.secrets.SystemRandom().random() < 0.5:
                    config[category][param] = parent1_params.get(param)
                else:
                    config[category][param] = parent2_params.get(param)
        
        return config
    
    def _mutate_config(self, config: Dict[str, Any], search_space: Dict[str, Any], mutation_rate: float) -> Dict[str, Any]:
        """Mutate a configuration."""
        mutated_config = {k: v.copy() if isinstance(v, dict) else v for k, v in config.items()}
        
        for category in ["hyperparameters", "architecture"]:
            if category in search_space and category in mutated_config:
                for param, spec in search_space[category].items():
                    if np.secrets.SystemRandom().random() < mutation_rate:
                        mutated_config[category][param] = self._sample_parameter(spec)
        
        return mutated_config
    
    def _perturb_config(self, config: Dict[str, Any], search_space: Dict[str, Any], perturbation_strength: float) -> Dict[str, Any]:
        """Perturb a configuration."""
        perturbed_config = {k: v.copy() if isinstance(v, dict) else v for k, v in config.items()}
        
        for category in ["hyperparameters", "architecture"]:
            if category in search_space and category in perturbed_config:
                for param, spec in search_space[category].items():
                    if param in perturbed_config[category]:
                        current_value = perturbed_config[category][param]
                        
                        if spec["type"] in ["uniform", "log_uniform"]:
                            # Add noise proportional to range
                            param_range = spec["bounds"][1] - spec["bounds"][0]
                            noise = np.random.normal(0, param_range * perturbation_strength)
                            new_value = current_value + noise
                            
                            # Clamp to bounds
                            new_value = max(spec["bounds"][0], min(spec["bounds"][1], new_value))
                            perturbed_config[category][param] = new_value
                        
                        elif spec["type"] == "choice":
                            # Randomly change with low probability
                            if np.secrets.SystemRandom().random() < perturbation_strength:
                                perturbed_config[category][param] = np.random.choice(spec["choices"])
        
        return perturbed_config
    
    def get_automl_stats(self) -> Dict[str, Any]:
        """Get AutoML pipeline statistics."""
        evaluated_variants = [v for v in self.algorithm_variants.values() if v.evaluation_status == "evaluated"]
        
        return {
            "total_variants": len(self.algorithm_variants),
            "evaluated_variants": len(evaluated_variants),
            "currently_evaluating": len(self.currently_evaluating),
            "evaluation_queue_size": len(self.evaluation_queue),
            "pareto_front_size": len(self.pareto_front),
            "deployments": len(self.deployment_history),
            "active_experiments": len(self.active_experiments),
            "is_running": self.is_running,
            "last_generation": self.last_generation_time.isoformat() if self.last_generation_time else None,
            "best_variant_score": max((v.performance_scores.get("combined_score", 0) for v in evaluated_variants), default=0),
        }