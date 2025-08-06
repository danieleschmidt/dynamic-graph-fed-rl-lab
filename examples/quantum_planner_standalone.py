#!/usr/bin/env python3
"""
Standalone Quantum Task Planner Example

Demonstrates core quantum-inspired task planning without dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time

# Import core quantum planner directly without package dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'dynamic_graph_fed_rl', 'quantum_planner'))
from core import QuantumTaskPlanner


def task_a():
    """Simulate task A execution."""
    time.sleep(0.05)  # Reduced for demo
    return {"result": "Task A completed", "computation_time": 0.05}


def task_b():
    """Simulate task B execution.""" 
    time.sleep(0.08)  # Reduced for demo
    return {"result": "Task B completed", "data_processed": 1000}


def task_c():
    """Simulate task C execution."""
    time.sleep(0.04)  # Reduced for demo
    return {"result": "Task C completed", "files_processed": 42}


def task_d():
    """Simulate task D execution."""
    time.sleep(0.06)  # Reduced for demo
    return {"result": "Task D completed", "analysis_complete": True}


def main():
    """Run standalone quantum planner example."""
    print("🔬 Quantum Task Planner - Standalone Example")
    print("=" * 50)
    
    # Initialize quantum planner
    planner = QuantumTaskPlanner(
        max_parallel_tasks=2,
        quantum_coherence_time=5.0,
        interference_strength=0.1
    )
    
    # Add tasks with dependencies
    print("\n📋 Adding tasks to quantum planner...")
    
    # Task A - Independent
    planner.add_task(
        task_id="task_a",
        name="Initialize System",
        estimated_duration=0.05,
        priority=2.0,
        executor=task_a
    )
    
    # Task B - Depends on A
    planner.add_task(
        task_id="task_b", 
        name="Process Data",
        dependencies={"task_a"},
        estimated_duration=0.08,
        priority=1.5,
        resource_requirements={"cpu": 0.5, "memory": 0.3},
        executor=task_b
    )
    
    # Task C - Independent
    planner.add_task(
        task_id="task_c",
        name="File Operations",
        estimated_duration=0.04,
        priority=1.8,
        resource_requirements={"disk": 0.2},
        executor=task_c
    )
    
    # Task D - Depends on B and C
    planner.add_task(
        task_id="task_d",
        name="Final Analysis",
        dependencies={"task_b", "task_c"},
        estimated_duration=0.06,
        priority=3.0,
        resource_requirements={"cpu": 0.8, "memory": 0.5},
        executor=task_d
    )
    
    # Display initial quantum state
    print(f"Tasks added: {len(planner.tasks)}")
    print(f"Entanglements: {len(planner.entanglement_matrix)}")
    
    # Generate quantum superposition
    print("\n🌀 Generating quantum superposition of execution paths...")
    superposition = planner.generate_execution_paths()
    
    print(f"Generated {len(superposition.paths)} possible execution paths:")
    for i, path in enumerate(superposition.paths):
        amplitude = superposition.path_amplitudes[i]
        probability = abs(amplitude) ** 2
        print(f"  Path {i+1}: {' → '.join(path)} (P={probability:.3f})")
    
    # Quantum measurement and execution
    print("\n⚡ Executing quantum measurement and collapse...")
    start_time = time.time()
    
    result = planner.measure_and_execute()
    
    execution_time = time.time() - start_time
    
    # Display results
    print("\n📊 Execution Results:")
    print(f"Optimal path: {' → '.join(result['path'])}")
    print(f"Total duration: {result['total_duration']:.3f}s")
    print(f"Quantum efficiency: {result['quantum_efficiency']:.3f}")
    print(f"Real execution time: {execution_time:.3f}s")
    
    print("\nTask-specific results:")
    for task_id, task_result in result['task_results'].items():
        status = task_result['status']
        duration = task_result['duration']
        emoji = "✅" if status == "success" else "❌"
        print(f"  {emoji} {task_id}: {status} ({duration:.3f}s)")
        if status == "success" and "result" in task_result:
            print(f"     └─ {task_result['result']}")
    
    # Display final quantum state
    print("\n🔬 Final Quantum State:")
    state = planner.get_system_state()
    
    print(f"Coherence remaining: {state['quantum_coherence_remaining']:.1f}s")
    print(f"Execution history: {state['execution_history_length']} runs")
    
    print("\nTask state probabilities:")
    for task_id, states in state['task_states'].items():
        dominant_state = max(states.items(), key=lambda x: x[1])
        print(f"  {task_id}: {dominant_state[0]} ({dominant_state[1]:.3f})")
    
    print("\n🚀 Quantum planning complete!")
    
    # Run a second execution to demonstrate superposition evolution
    print("\n" + "=" * 50)
    print("🔄 Running second quantum execution...")
    
    # Add a new task to demonstrate dynamic behavior
    planner.add_task(
        task_id="task_e",
        name="Cleanup Operations",
        dependencies={"task_d"},
        estimated_duration=0.03,
        priority=1.0,
        executor=lambda: {"result": "Cleanup completed", "files_cleaned": 15}
    )
    
    result2 = planner.measure_and_execute()
    
    print(f"\nSecond execution path: {' → '.join(result2['path'])}")
    print(f"Quantum efficiency: {result2['quantum_efficiency']:.3f}")
    print(f"Total executions: {len(planner.execution_history)}")


if __name__ == "__main__":
    main()