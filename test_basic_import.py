#!/usr/bin/env python3
"""Basic import test for dynamic_graph_fed_rl package."""

import sys
sys.path.insert(0, '/root/repo/src')

try:
    import dynamic_graph_fed_rl as dgfrl
    print("✅ Successfully imported dynamic_graph_fed_rl")
    print(f"Version: {dgfrl.__version__}")
    print(f"Available modules: {[k for k in dir(dgfrl) if not k.startswith('_')]}")
    
    # Test core imports
    from dynamic_graph_fed_rl.environments.base import BaseGraphEnvironment, GraphState, GraphTransition
    from dynamic_graph_fed_rl.algorithms.base import BaseGraphAlgorithm
    print("✅ Core base classes imported successfully")
    
    # Test quantum planner
    from dynamic_graph_fed_rl.quantum_planner import QuantumTaskPlanner
    print("✅ Quantum planner imported successfully")
    
    print("\n🎯 Generation 1 (MAKE IT WORK) - Basic functionality verified")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)