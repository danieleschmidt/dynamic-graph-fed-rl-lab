#!/usr/bin/env python3
"""
Minimal Quantum Task Planner Example

Pure Python implementation without external dependencies.
Demonstrates quantum-inspired planning concepts.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable


class TaskState(Enum):
    """Task states in quantum superposition."""
    PENDING = "pending"
    ACTIVE = "active" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MinimalQuantumTask:
    """Simplified quantum task for demonstration."""
    id: str
    name: str
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 1.0
    priority: float = 1.0
    executor: Optional[Callable] = None
    
    # Simplified quantum properties
    state_probabilities: Dict[TaskState, float] = field(default_factory=dict)
    actual_duration: Optional[float] = None
    result: Any = None
    
    def __post_init__(self):
        """Initialize quantum state probabilities."""
        if not self.state_probabilities:
            self.state_probabilities = {
                TaskState.PENDING: 1.0,
                TaskState.ACTIVE: 0.0,
                TaskState.COMPLETED: 0.0,
                TaskState.FAILED: 0.0,
            }
    
    def collapse_to_state(self, state: TaskState):
        """Collapse quantum superposition to definite state."""
        self.state_probabilities = {s: 0.0 for s in TaskState}
        self.state_probabilities[state] = 1.0


class MinimalQuantumPlanner:
    """Simplified quantum-inspired task planner."""
    
    def __init__(self, max_parallel_tasks: int = 4):
        self.max_parallel_tasks = max_parallel_tasks
        self.tasks: Dict[str, MinimalQuantumTask] = {}
        self.execution_history: List[Dict] = []
        
    def add_task(
        self,
        task_id: str,
        name: str,
        dependencies: Optional[Set[str]] = None,
        estimated_duration: float = 1.0,
        priority: float = 1.0,
        executor: Optional[Callable] = None,
    ):
        """Add task to planner."""
        dependencies = dependencies or set()
        
        task = MinimalQuantumTask(
            id=task_id,
            name=name,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            priority=priority,
            executor=executor,
        )
        
        self.tasks[task_id] = task
        return task
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks ready for execution."""
        ready = []
        for task_id, task in self.tasks.items():
            if task.state_probabilities[TaskState.PENDING] > 0.5:
                # Check dependencies
                deps_completed = all(
                    self.tasks[dep_id].state_probabilities[TaskState.COMPLETED] > 0.5
                    for dep_id in task.dependencies
                    if dep_id in self.tasks
                )
                if deps_completed:
                    ready.append(task_id)
        return ready
    
    def generate_execution_paths(self) -> List[List[str]]:
        """Generate possible execution paths."""
        ready_tasks = self.get_ready_tasks()
        if not ready_tasks:
            return []
        
        paths = []
        
        # Path 1: Priority-based ordering
        priority_path = sorted(ready_tasks, key=lambda tid: self.tasks[tid].priority, reverse=True)
        paths.append(priority_path)
        
        # Path 2: Duration-based ordering (shortest first)
        duration_path = sorted(ready_tasks, key=lambda tid: self.tasks[tid].estimated_duration)
        paths.append(duration_path)
        
        # Path 3: Random exploration path
        random_path = ready_tasks.copy()
        random.shuffle(random_path)
        paths.append(random_path)
        
        return paths
    
    def quantum_measurement(self, paths: List[List[str]]) -> List[str]:
        """Simulate quantum measurement to select optimal path."""
        if not paths:
            return []
        
        # Calculate path weights (simplified quantum interference)
        path_weights = []
        for path in paths:
            weight = 0.0
            for task_id in path:
                task = self.tasks[task_id]
                # Weight by priority and inverse duration
                weight += task.priority * (1.0 / max(task.estimated_duration, 0.1))
            path_weights.append(weight)
        
        # Normalize weights to probabilities
        total_weight = sum(path_weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in path_weights]
        else:
            probabilities = [1.0 / len(paths)] * len(paths)
        
        # Quantum measurement (probabilistic selection)
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return paths[i]
        
        return paths[-1]  # Fallback
    
    def execute_tasks(self, task_sequence: List[str]) -> Dict[str, Any]:
        """Execute task sequence."""
        results = {}
        start_time = time.time()
        
        for task_id in task_sequence:
            if task_id not in self.tasks:
                continue
                
            task = self.tasks[task_id]
            
            # Collapse to active state
            task.collapse_to_state(TaskState.ACTIVE)
            
            task_start = time.time()
            
            try:
                # Execute task
                if task.executor:
                    task.result = task.executor()
                else:
                    # Simulate execution
                    sleep_time = min(task.estimated_duration, 0.1)  # Cap for demo
                    time.sleep(sleep_time)
                    task.result = {"status": "simulated", "task_id": task_id}
                
                task.actual_duration = time.time() - task_start
                
                # Collapse to completed state
                task.collapse_to_state(TaskState.COMPLETED)
                
                results[task_id] = {
                    "status": "success",
                    "result": task.result,
                    "duration": task.actual_duration,
                }
                
            except Exception as e:
                task.actual_duration = time.time() - task_start
                task.collapse_to_state(TaskState.FAILED)
                
                results[task_id] = {
                    "status": "failed",
                    "error": str(e),
                    "duration": task.actual_duration,
                }
        
        total_duration = time.time() - start_time
        
        return {
            "sequence": task_sequence,
            "task_results": results,
            "total_duration": total_duration,
            "success_rate": sum(1 for r in results.values() if r["status"] == "success") / len(results) if results else 0.0,
        }
    
    def plan_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute(self) -> Dict[str, Any]:
        """Full quantum planning and execution cycle."""
        # Generate superposition of paths
        paths = self.generate_execution_paths()
        
        # Quantum measurement to select path
        optimal_path = self.quantum_measurement(paths)
        
        # Execute selected path
        result = self.execute_tasks(optimal_path)
        
        # Record execution
        self.execution_history.append({
            "timestamp": time.time(),
            "paths_generated": len(paths),
            "selected_path": optimal_path,
            "result": result,
        })
        
        return result


# Example task functions
def task_initialize():
    """Initialize system."""
    time.sleep(0.05)
    return {"status": "System initialized", "memory_allocated": "512MB"}


def task_load_data():
    """Load data."""
    time.sleep(0.08)
    return {"status": "Data loaded", "records": 10000}


def task_process():
    """Process data."""
    time.sleep(0.06)
    return {"status": "Processing complete", "items_processed": 10000}


def task_analyze():
    """Analyze results."""
    time.sleep(0.04)
    return {"status": "Analysis complete", "insights": 42}


def task_cleanup():
    """Cleanup operations."""
    time.sleep(0.02)
    return {"status": "Cleanup complete", "temp_files_removed": 15}


def main():
    """Run minimal quantum planner demonstration."""
    print("🔬 Minimal Quantum Task Planner")
    print("=" * 40)
    
    # Initialize planner
    planner = MinimalQuantumPlanner(max_parallel_tasks=3)
    
    # Add tasks with dependencies
    print("\n📋 Adding tasks...")
    
    planner.add_task(
        task_id="init",
        name="Initialize System",
        estimated_duration=0.05,
        priority=3.0,
        executor=task_initialize
    )
    
    planner.add_task(
        task_id="load",
        name="Load Data",
        dependencies={"init"},
        estimated_duration=0.08,
        priority=2.5,
        executor=task_load_data
    )
    
    planner.add_task(
        task_id="process",
        name="Process Data",
        dependencies={"load"},
        estimated_duration=0.06,
        priority=2.0,
        executor=task_process
    )
    
    planner.add_task(
        task_id="analyze",
        name="Analyze Results",
        dependencies={"process"},
        estimated_duration=0.04,
        priority=2.8,
        executor=task_analyze
    )
    
    planner.add_task(
        task_id="cleanup",
        name="Cleanup",
        dependencies={"analyze"},
        estimated_duration=0.02,
        priority=1.0,
        executor=task_cleanup
    )
    
    print(f"Added {len(planner.tasks)} tasks")
    
    # Display task dependencies
    print("\nTask Dependencies:")
    for task_id, task in planner.tasks.items():
        deps = list(task.dependencies) if task.dependencies else ["None"]
        print(f"  {task_id}: depends on {deps}")
    
    # Execute quantum planning
    print("\n🌀 Generating quantum superposition...")
    paths = planner.generate_execution_paths()
    
    print(f"Generated {len(paths)} execution paths:")
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: {' → '.join(path)}")
    
    print("\n⚡ Quantum measurement and execution...")
    start_time = time.time()
    
    result = planner.plan_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute()
    
    execution_time = time.time() - start_time
    
    # Display results
    print("\n📊 Execution Results:")
    print(f"Selected path: {' → '.join(result['sequence'])}")
    print(f"Total duration: {result['total_duration']:.3f}s")
    print(f"Success rate: {result['success_rate']:.3f}")
    print(f"Real execution time: {execution_time:.3f}s")
    
    print("\nTask Results:")
    for task_id, task_result in result['task_results'].items():
        status = task_result['status']
        duration = task_result['duration']
        emoji = "✅" if status == "success" else "❌"
        print(f"  {emoji} {task_id}: {status} ({duration:.3f}s)")
        
        if status == "success" and "result" in task_result:
            task_data = task_result['result']
            if isinstance(task_data, dict) and 'status' in task_data:
                print(f"     └─ {task_data['status']}")
    
    # Run second execution to show different path selection
    print("\n" + "=" * 40)
    print("🔄 Second quantum execution...")
    
    result2 = planner.plan_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute()
    
    print(f"Second path: {' → '.join(result2['sequence'])}")
    print(f"Success rate: {result2['success_rate']:.3f}")
    
    print(f"\nTotal executions in history: {len(planner.execution_history)}")
    
    print("\n🚀 Quantum planning demonstration complete!")


if __name__ == "__main__":
    main()