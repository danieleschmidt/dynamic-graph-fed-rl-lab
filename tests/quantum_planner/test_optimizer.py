"""
Test suite for quantum task optimization components.

Tests quantum optimization algorithms and parameter tuning.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import jax.numpy as jnp

from src.dynamic_graph_fed_rl.quantum_planner.optimizer import (
    QuantumOptimizer,
    InterferenceOptimizer,
    ParameterOptimizer,
    OptimizationResult
)
from src.dynamic_graph_fed_rl.quantum_planner.core import QuantumTask, TaskState


class TestQuantumOptimizer:
    """Test quantum optimization functionality."""
    
    @pytest.fixture
    def optimizer(self):
        """Create test optimizer."""
        return QuantumOptimizer(
            optimization_rounds=10,
            learning_rate=0.01,
            convergence_threshold=1e-6
        )
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.optimization_rounds == 10
        assert optimizer.learning_rate == 0.01
        assert optimizer.convergence_threshold == 1e-6
    
    def test_optimize_schedule(self, optimizer):
        """Test schedule optimization."""
        # Create mock tasks
        tasks = [
            QuantumTask(id=f"task_{i}", name=f"Task {i}")
            for i in range(5)
        ]
        
        result = optimizer.optimize_schedule(tasks)
        
        assert isinstance(result, OptimizationResult)
        assert result.converged in [True, False]
        assert result.final_cost >= 0
        assert result.iterations >= 0
    
    def test_optimization_convergence(self, optimizer):
        """Test optimization convergence detection."""
        costs = [100, 50, 25, 12, 6, 3, 1.5, 1.4, 1.3, 1.25]
        
        converged = optimizer._check_convergence(costs)
        assert isinstance(converged, bool)


class TestInterferenceOptimizer:
    """Test interference optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create interference optimizer."""
        return InterferenceOptimizer(max_iterations=50)
    
    def test_minimize_interference(self, optimizer):
        """Test interference minimization."""
        # Mock task paths
        task_paths = [
            {"task_1": [0, 1, 2]},
            {"task_2": [1, 2, 3]}, 
            {"task_3": [2, 3, 4]}
        ]
        
        result = optimizer.minimize_interference(task_paths)
        
        assert isinstance(result, dict)
        assert "optimal_ordering" in result
        assert "interference_score" in result
    
    def test_calculate_interference_score(self, optimizer):
        """Test interference score calculation."""
        paths = [
            [0, 1, 2],
            [1, 2, 3],
            [0, 2, 4]
        ]
        
        score = optimizer.calculate_interference_score(paths)
        assert isinstance(score, float)
        assert score >= 0


class TestParameterOptimizer:
    """Test parameter optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create parameter optimizer."""
        return ParameterOptimizer()
    
    def test_optimize_quantum_parameters(self, optimizer):
        """Test quantum parameter optimization."""
        initial_params = {
            "coherence_time": 1.0,
            "entanglement_strength": 0.5,
            "measurement_accuracy": 0.95
        }
        
        # Mock performance function
        def mock_performance(params):
            return sum(params.values()) / len(params)
        
        result = optimizer.optimize_parameters(initial_params, mock_performance)
        
        assert isinstance(result, dict)
        assert "optimized_params" in result
        assert "performance_score" in result
    
    def test_parameter_bounds_validation(self, optimizer):
        """Test parameter bounds validation."""
        params = {
            "coherence_time": -1.0,  # Invalid
            "entanglement_strength": 1.5,  # Invalid  
            "measurement_accuracy": 0.5   # Valid
        }
        
        validated = optimizer.validate_parameter_bounds(params)
        
        assert validated["coherence_time"] >= 0
        assert validated["entanglement_strength"] <= 1.0
        assert validated["measurement_accuracy"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])