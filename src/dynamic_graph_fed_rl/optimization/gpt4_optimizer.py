"""
GPT-4 Integration for Dynamic Hyperparameter Optimization.

Uses GPT-4 to analyze real-time performance metrics and suggest optimal
hyperparameter adjustments for federated learning algorithms.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta
import openai
from ..quantum_planner.performance import PerformanceMonitor


@dataclass
class HyperparameterConfig:
    """Configuration for a hyperparameter."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    value_type: str  # "float", "int", "categorical"
    categories: Optional[List[str]] = None
    importance: float = 1.0
    last_updated: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    accuracy: float
    convergence_rate: float
    communication_efficiency: float
    resource_utilization: float
    latency: float
    throughput: float
    stability_score: float
    timestamp: datetime


class GPT4HyperparameterOptimizer:
    """
    GPT-4 powered hyperparameter optimization system.
    
    Analyzes real-time performance metrics and uses GPT-4's reasoning
    capabilities to suggest intelligent hyperparameter adjustments.
    """
    
    def __init__(
        self,
        openai_api_key: str,
        performance_monitor: PerformanceMonitor,
        optimization_interval: float = 300.0,  # 5 minutes
        logger: Optional[logging.Logger] = None,
    ):
        self.openai_api_key = openai_api_key
        self.performance_monitor = performance_monitor
        self.optimization_interval = optimization_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize OpenAI client
        openai.api_key = openai_api_key
        
        # Hyperparameter configuration
        self.hyperparams: Dict[str, HyperparameterConfig] = {}
        self._initialize_default_hyperparams()
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # GPT-4 optimization state
        self.last_optimization: Optional[datetime] = None
        self.optimization_context: List[Dict[str, str]] = []
        self.is_optimizing = False
        
        # Performance tracking
        self.successful_optimizations = 0
        self.total_optimizations = 0
        
    def _initialize_default_hyperparams(self):
        """Initialize default hyperparameter configurations."""
        defaults = [
            HyperparameterConfig("learning_rate", 0.001, 0.0001, 0.01, "float", importance=1.0),
            HyperparameterConfig("batch_size", 32, 8, 128, "int", importance=0.8),
            HyperparameterConfig("discount_factor", 0.95, 0.9, 0.99, "float", importance=0.9),
            HyperparameterConfig("exploration_rate", 0.3, 0.1, 0.5, "float", importance=0.7),
            HyperparameterConfig("communication_rounds", 10, 5, 20, "int", importance=0.6),
            HyperparameterConfig("aggregation_method", "fedavg", 0, 1, "categorical", 
                               categories=["fedavg", "fedprox", "quantum_weighted"], importance=0.8),
            HyperparameterConfig("quantum_coherence_time", 10.0, 5.0, 20.0, "float", importance=0.5),
        ]
        
        for config in defaults:
            self.hyperparams[config.name] = config
    
    async def start_optimization_loop(self):
        """Start continuous optimization loop."""
        self.logger.info("Starting GPT-4 hyperparameter optimization loop")
        
        while True:
            try:
                if not self.is_optimizing:
                    await self.optimize_hyperparameters()
                
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def optimize_hyperparameters(self) -> Dict[str, Any]:
        """Run GPT-4 powered hyperparameter optimization."""
        if self.is_optimizing:
            return {"status": "already_optimizing"}
        
        self.is_optimizing = True
        optimization_start = time.time()
        
        try:
            # Collect current performance metrics
            current_metrics = await self._collect_current_metrics()
            self.metrics_history.append(current_metrics)
            
            # Keep only recent metrics
            if len(self.metrics_history) > 50:
                self.metrics_history = self.metrics_history[-50:]
            
            # Generate optimization prompt for GPT-4
            prompt = self._generate_optimization_prompt(current_metrics)
            
            # Call GPT-4 for optimization suggestions
            optimization_response = await self._call_gpt4_for_optimization(prompt)
            
            # Parse and validate suggestions
            suggestions = self._parse_optimization_suggestions(optimization_response)
            
            # Apply validated suggestions
            applied_changes = await self._apply_hyperparameter_changes(suggestions)
            
            # Record optimization
            optimization_result = {
                "timestamp": datetime.now(),
                "current_metrics": current_metrics,
                "gpt4_response": optimization_response,
                "suggestions": suggestions,
                "applied_changes": applied_changes,
                "optimization_time": time.time() - optimization_start,
            }
            
            self.optimization_history.append(optimization_result)
            self.total_optimizations += 1
            
            if applied_changes:
                self.successful_optimizations += 1
                self.logger.info(f"GPT-4 optimization completed: {len(applied_changes)} changes applied")
            else:
                self.logger.info("GPT-4 optimization completed: no changes recommended")
            
            self.last_optimization = datetime.now()
            
            return {
                "status": "completed",
                "applied_changes": applied_changes,
                "optimization_time": time.time() - optimization_start,
                "suggestions_count": len(suggestions),
            }
            
        except Exception as e:
            self.logger.error(f"GPT-4 optimization failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "optimization_time": time.time() - optimization_start,
            }
        finally:
            self.is_optimizing = False
    
    async def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        # Get metrics from performance monitor
        perf_data = self.performance_monitor.get_current_metrics()
        
        return PerformanceMetrics(
            accuracy=perf_data.get("accuracy", 0.8),
            convergence_rate=perf_data.get("convergence_rate", 0.1),
            communication_efficiency=perf_data.get("communication_efficiency", 0.7),
            resource_utilization=perf_data.get("resource_utilization", 0.6),
            latency=perf_data.get("latency", 100.0),
            throughput=perf_data.get("throughput", 10.0),
            stability_score=perf_data.get("stability_score", 0.85),
            timestamp=datetime.now(),
        )
    
    def _generate_optimization_prompt(self, current_metrics: PerformanceMetrics) -> str:
        """Generate optimization prompt for GPT-4."""
        # Get recent performance trends
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        trend_analysis = self._analyze_performance_trends(recent_metrics)
        
        # Current hyperparameter values
        current_hyperparams = {
            name: config.current_value 
            for name, config in self.hyperparams.items()
        }
        
        prompt = f"""
You are an expert in federated learning and quantum-inspired optimization. Analyze the following performance metrics and hyperparameters to suggest optimal adjustments.

CURRENT PERFORMANCE METRICS:
- Accuracy: {current_metrics.accuracy:.3f}
- Convergence Rate: {current_metrics.convergence_rate:.3f}
- Communication Efficiency: {current_metrics.communication_efficiency:.3f}
- Resource Utilization: {current_metrics.resource_utilization:.3f}
- Latency: {current_metrics.latency:.1f}ms
- Throughput: {current_metrics.throughput:.1f} ops/sec
- Stability Score: {current_metrics.stability_score:.3f}

PERFORMANCE TRENDS:
{trend_analysis}

CURRENT HYPERPARAMETERS:
{json.dumps(current_hyperparams, indent=2)}

HYPERPARAMETER CONSTRAINTS:
{self._get_hyperparameter_constraints()}

OPTIMIZATION OBJECTIVES:
1. Maximize accuracy while maintaining stability
2. Improve convergence rate without sacrificing communication efficiency
3. Optimize resource utilization
4. Minimize latency while maximizing throughput

Please provide hyperparameter optimization suggestions in the following JSON format:
{{
    "analysis": "Brief analysis of current performance and bottlenecks",
    "suggestions": [
        {{
            "parameter": "parameter_name",
            "current_value": current_value,
            "suggested_value": new_value,
            "reasoning": "Explanation for this change",
            "expected_impact": "Expected performance improvement",
            "confidence": 0.8
        }}
    ],
    "overall_strategy": "High-level optimization strategy description"
}}

Focus on the most impactful changes based on the performance trends and theoretical understanding of federated learning dynamics.
"""
        return prompt
    
    def _analyze_performance_trends(self, recent_metrics: List[PerformanceMetrics]) -> str:
        """Analyze recent performance trends."""
        if len(recent_metrics) < 2:
            return "Insufficient data for trend analysis."
        
        # Calculate trends for key metrics
        accuracies = [m.accuracy for m in recent_metrics]
        convergence_rates = [m.convergence_rate for m in recent_metrics]
        latencies = [m.latency for m in recent_metrics]
        throughputs = [m.throughput for m in recent_metrics]
        
        trends = {
            "accuracy": "increasing" if accuracies[-1] > accuracies[0] else "decreasing",
            "convergence": "improving" if convergence_rates[-1] > convergence_rates[0] else "degrading",
            "latency": "increasing" if latencies[-1] > latencies[0] else "decreasing", 
            "throughput": "increasing" if throughputs[-1] > throughputs[0] else "decreasing",
        }
        
        return f"""
Recent trends over {len(recent_metrics)} measurements:
- Accuracy: {trends['accuracy']} (from {accuracies[0]:.3f} to {accuracies[-1]:.3f})
- Convergence rate: {trends['convergence']} (from {convergence_rates[0]:.3f} to {convergence_rates[-1]:.3f})
- Latency: {trends['latency']} (from {latencies[0]:.1f}ms to {latencies[-1]:.1f}ms)
- Throughput: {trends['throughput']} (from {throughputs[0]:.1f} to {throughputs[-1]:.1f} ops/sec)
"""
    
    def _get_hyperparameter_constraints(self) -> str:
        """Get hyperparameter constraints as string."""
        constraints = []
        for name, config in self.hyperparams.items():
            if config.value_type == "categorical":
                constraints.append(f"{name}: {config.categories}")
            else:
                constraints.append(f"{name}: [{config.min_value}, {config.max_value}]")
        
        return "\n".join(constraints)
    
    async def _call_gpt4_for_optimization(self, prompt: str) -> str:
        """Call GPT-4 API for optimization suggestions."""
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert AI system optimizer specializing in federated learning and quantum-inspired algorithms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"GPT-4 API call failed: {e}")
            raise
    
    def _parse_optimization_suggestions(self, gpt4_response: str) -> List[Dict[str, Any]]:
        """Parse GPT-4 response into optimization suggestions."""
        try:
            # Extract JSON from response
            json_start = gpt4_response.find('{')
            json_end = gpt4_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in GPT-4 response")
            
            json_str = gpt4_response[json_start:json_end]
            parsed_response = json.loads(json_str)
            
            suggestions = parsed_response.get("suggestions", [])
            
            # Validate and filter suggestions
            validated_suggestions = []
            for suggestion in suggestions:
                if self._validate_suggestion(suggestion):
                    validated_suggestions.append(suggestion)
                else:
                    self.logger.warning(f"Invalid suggestion filtered out: {suggestion}")
            
            return validated_suggestions
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse GPT-4 response: {e}")
            return []
    
    def _validate_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Validate a hyperparameter suggestion."""
        required_keys = ["parameter", "suggested_value", "confidence"]
        
        # Check required keys
        if not all(key in suggestion for key in required_keys):
            return False
        
        param_name = suggestion["parameter"]
        suggested_value = suggestion["suggested_value"]
        confidence = suggestion["confidence"]
        
        # Check if parameter exists
        if param_name not in self.hyperparams:
            return False
        
        config = self.hyperparams[param_name]
        
        # Check value bounds
        if config.value_type in ["float", "int"]:
            if not (config.min_value <= suggested_value <= config.max_value):
                return False
        elif config.value_type == "categorical":
            if suggested_value not in config.categories:
                return False
        
        # Check confidence threshold
        if confidence < 0.5:
            return False
        
        return True
    
    async def _apply_hyperparameter_changes(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply validated hyperparameter changes."""
        applied_changes = []
        
        for suggestion in suggestions:
            param_name = suggestion["parameter"]
            suggested_value = suggestion["suggested_value"]
            confidence = suggestion["confidence"]
            
            # Apply conservative threshold for automatic changes
            if confidence >= 0.7:
                config = self.hyperparams[param_name]
                old_value = config.current_value
                
                # Apply change
                config.current_value = suggested_value
                config.last_updated = datetime.now()
                
                # Notify performance monitor of change
                await self._notify_hyperparameter_change(param_name, old_value, suggested_value)
                
                applied_changes.append({
                    "parameter": param_name,
                    "old_value": old_value,
                    "new_value": suggested_value,
                    "confidence": confidence,
                    "reasoning": suggestion.get("reasoning", ""),
                })
                
                self.logger.info(f"Applied hyperparameter change: {param_name} = {suggested_value} (confidence: {confidence:.2f})")
        
        return applied_changes
    
    async def _notify_hyperparameter_change(self, param_name: str, old_value: Any, new_value: Any):
        """Notify system components of hyperparameter changes."""
        try:
            await self.performance_monitor.update_hyperparameter(param_name, new_value)
        except Exception as e:
            self.logger.warning(f"Failed to notify hyperparameter change: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        success_rate = self.successful_optimizations / max(1, self.total_optimizations)
        
        return {
            "total_optimizations": self.total_optimizations,
            "successful_optimizations": self.successful_optimizations,
            "success_rate": success_rate,
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "optimization_interval": self.optimization_interval,
            "metrics_history_size": len(self.metrics_history),
            "current_hyperparams": {
                name: config.current_value
                for name, config in self.hyperparams.items()
            },
            "is_optimizing": self.is_optimizing,
        }
    
    def get_recent_optimizations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent optimization history."""
        return self.optimization_history[-limit:] if self.optimization_history else []