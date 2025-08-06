#!/usr/bin/env python3
"""
Basic Quantum Task Planner Example

Demonstrates core quantum-inspired task planning functionality.
"""

import time
from dynamic_graph_fed_rl.quantum_planner import QuantumTaskPlanner


def task_a():
    """Simulate task A execution."""
    time.sleep(0.1)
    return {"result": "Task A completed", "computation_time": 0.1}


def task_b():
    """Simulate task B execution.""" 
    time.sleep(0.15)
    return {"result": "Task B completed", "data_processed": 1000}


def task_c():
    """Simulate task C execution."""
    time.sleep(0.08)
    return {"result": "Task C completed", "files_processed": 42}


def task_d():
    """Simulate task D execution."""
    time.sleep(0.12)
    return {"result": "Task D completed", "analysis_complete": True}


def main():
    """Run basic quantum planner example."""
    print("🔬 Quantum Task Planner - Basic Example")
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
        estimated_duration=0.1,
        priority=2.0,
        executor=task_a
    )
    
    # Task B - Depends on A
    planner.add_task(
        task_id="task_b", 
        name="Process Data",
        dependencies={"task_a"},
        estimated_duration=0.15,
        priority=1.5,
        resource_requirements={"cpu": 0.5, "memory": 0.3},
        executor=task_b
    )
    
    # Task C - Independent
    planner.add_task(
        task_id="task_c",
        name="File Operations",
        estimated_duration=0.08,
        priority=1.8,
        resource_requirements={"disk": 0.2},
        executor=task_c
    )
    
    # Task D - Depends on B and C
    planner.add_task(
        task_id="task_d",
        name="Final Analysis",
        dependencies={"task_b", "task_c"},
        estimated_duration=0.12,
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


if __name__ == "__main__":
    main()