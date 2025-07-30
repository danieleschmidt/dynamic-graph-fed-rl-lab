"""Basic unit tests to verify package structure."""

import pytest
from dynamic_graph_fed_rl import __version__


def test_version():
    """Test that version is accessible."""
    assert __version__ == "0.1.0"


def test_imports():
    """Test that main modules can be imported."""
    from dynamic_graph_fed_rl import algorithms, environments, federation, models, utils
    
    # Test that modules exist
    assert algorithms is not None
    assert environments is not None  
    assert federation is not None
    assert models is not None
    assert utils is not None


def test_algorithm_imports():
    """Test algorithm module imports."""
    from dynamic_graph_fed_rl.algorithms import GraphTD3, GraphSAC, BaseGraphAgent
    
    # Just test that classes exist (implementation will be added later)
    assert GraphTD3 is not None
    assert GraphSAC is not None
    assert BaseGraphAgent is not None


def test_environment_imports():
    """Test environment module imports.""" 
    from dynamic_graph_fed_rl.environments import TrafficEnv, PowerGridEnv, DynamicGraphEnv
    
    assert TrafficEnv is not None
    assert PowerGridEnv is not None
    assert DynamicGraphEnv is not None


def test_federation_imports():
    """Test federation module imports."""
    from dynamic_graph_fed_rl.federation import AsyncGossipProtocol, FederatedHierarchy, FederatedOptimizer
    
    assert AsyncGossipProtocol is not None
    assert FederatedHierarchy is not None
    assert FederatedOptimizer is not None