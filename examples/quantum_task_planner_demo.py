#!/usr/bin/env python3
"""
Quantum Task Planner Demo

Demonstrates quantum-inspired task planning with real-world scenarios:
1. Software Development Pipeline
2. Infrastructure Deployment
3. Data Processing Pipeline
4. Multi-modal AI Training
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Import quantum planner components
from src.dynamic_graph_fed_rl.quantum_planner import (
    QuantumTaskPlanner,
    QuantumScheduler, 
    AdaptiveScheduler,
    QuantumOptimizer,
    InterferenceOptimizer,
    QuantumExecutor,
    ParallelExecutor
)


def create_software_pipeline_tasks(planner: QuantumTaskPlanner) -> List[str]:
    """Create tasks for a software development pipeline."""
    
    # Development tasks
    design_task = planner.add_task(
        "design_system",
        "System Design & Architecture",
        dependencies=set(),
        estimated_duration=2.0,
        priority=1.0,
        resource_requirements={"cpu": 0.2, "memory": 0.3},
        executor=lambda: {"status": "completed", "artifacts": ["architecture.md", "diagrams/"]}
    )
    
    frontend_task = planner.add_task(
        "develop_frontend", 
        "Frontend Development",
        dependencies={"design_system"},
        estimated_duration=5.0,
        priority=0.8,
        resource_requirements={"cpu": 0.4, "memory": 0.5},
        executor=lambda: {"status": "completed", "artifacts": ["dist/", "src/components/"]}
    )
    
    backend_task = planner.add_task(
        "develop_backend",
        "Backend API Development", 
        dependencies={"design_system"},
        estimated_duration=4.0,
        priority=0.9,
        resource_requirements={"cpu": 0.5, "memory": 0.6},
        executor=lambda: {"status": "completed", "artifacts": ["api/", "models/"]}
    )
    
    database_task = planner.add_task(
        "setup_database",
        "Database Schema & Setup",
        dependencies={"design_system"},
        estimated_duration=1.5,
        priority=0.7,
        resource_requirements={"cpu": 0.3, "memory": 0.4, "io": 0.8},
        executor=lambda: {"status": "completed", "artifacts": ["schema.sql", "migrations/"]}
    )
    
    # Testing tasks
    unit_tests_task = planner.add_task(
        "unit_tests",
        "Unit Testing",
        dependencies={"develop_frontend", "develop_backend"},
        estimated_duration=2.0,
        priority=0.6,
        resource_requirements={"cpu": 0.3, "memory": 0.3},
        executor=lambda: {"status": "completed", "coverage": 0.87}
    )
    
    integration_tests_task = planner.add_task(
        "integration_tests",
        "Integration Testing",
        dependencies={"unit_tests", "setup_database"},
        estimated_duration=3.0,
        priority=0.6,
        resource_requirements={"cpu": 0.4, "memory": 0.5, "io": 0.6},
        executor=lambda: {"status": "completed", "tests_passed": 42}
    )
    
    # Deployment tasks
    docker_task = planner.add_task(
        "containerization",
        "Docker Containerization",
        dependencies={"integration_tests"},
        estimated_duration=1.0,
        priority=0.5,
        resource_requirements={"cpu": 0.2, "memory": 0.3, "io": 0.4},
        executor=lambda: {"status": "completed", "images": ["app:latest", "db:latest"]}
    )
    
    deploy_task = planner.add_task(
        "deploy_production",
        "Production Deployment",
        dependencies={"containerization"},
        estimated_duration=2.0,
        priority=0.9,
        resource_requirements={"cpu": 0.3, "memory": 0.4, "io": 0.7},
        executor=lambda: {"status": "completed", "deployment_url": "https://app.example.com"}
    )
    
    return [
        "design_system", "develop_frontend", "develop_backend", "setup_database",
        "unit_tests", "integration_tests", "containerization", "deploy_production"
    ]


def create_infrastructure_tasks(planner: QuantumTaskPlanner) -> List[str]:
    """Create tasks for infrastructure deployment."""
    
    # Infrastructure provisioning
    vpc_task = planner.add_task(
        "provision_vpc",
        "VPC & Network Setup",
        dependencies=set(),
        estimated_duration=1.0,
        priority=1.0,
        resource_requirements={"cpu": 0.1, "memory": 0.2, "io": 0.3},
        executor=lambda: {"status": "completed", "vpc_id": "vpc-123456"}
    )
    
    security_task = planner.add_task(
        "security_groups",
        "Security Groups & Firewall",
        dependencies={"provision_vpc"},
        estimated_duration=0.5,
        priority=0.9,
        resource_requirements={"cpu": 0.1, "memory": 0.1, "io": 0.2},
        executor=lambda: {"status": "completed", "security_groups": ["web-sg", "db-sg"]}
    )
    
    load_balancer_task = planner.add_task(
        "setup_load_balancer",
        "Load Balancer Configuration",
        dependencies={"security_groups"},
        estimated_duration=1.5,
        priority=0.8,
        resource_requirements={"cpu": 0.2, "memory": 0.3, "io": 0.4},
        executor=lambda: {"status": "completed", "lb_dns": "lb-123.elb.amazonaws.com"}
    )
    
    database_cluster_task = planner.add_task(
        "database_cluster",
        "Database Cluster Setup",
        dependencies={"security_groups"},
        estimated_duration=3.0,
        priority=0.8,
        resource_requirements={"cpu": 0.4, "memory": 0.6, "io": 0.8},
        executor=lambda: {"status": "completed", "cluster_endpoint": "db-cluster.amazonaws.com"}
    )
    
    app_servers_task = planner.add_task(
        "application_servers",
        "Application Server Instances",
        dependencies={"setup_load_balancer"},
        estimated_duration=2.0,
        priority=0.7,
        resource_requirements={"cpu": 0.3, "memory": 0.4, "io": 0.5},
        executor=lambda: {"status": "completed", "instances": ["i-123", "i-456", "i-789"]}
    )
    
    monitoring_task = planner.add_task(
        "monitoring_setup",
        "Monitoring & Alerting",
        dependencies={"application_servers", "database_cluster"},
        estimated_duration=1.0,
        priority=0.6,
        resource_requirements={"cpu": 0.2, "memory": 0.3, "io": 0.3},
        executor=lambda: {"status": "completed", "dashboards": ["app-metrics", "db-metrics"]}
    )
    
    backup_task = planner.add_task(
        "backup_system",
        "Backup & Disaster Recovery",
        dependencies={"monitoring_setup"},
        estimated_duration=1.5,
        priority=0.5,
        resource_requirements={"cpu": 0.1, "memory": 0.2, "io": 0.6},
        executor=lambda: {"status": "completed", "backup_schedule": "daily_3am"}
    )
    
    return [
        "provision_vpc", "security_groups", "setup_load_balancer", 
        "database_cluster", "application_servers", "monitoring_setup", "backup_system"
    ]


def create_data_processing_tasks(planner: QuantumTaskPlanner) -> List[str]:
    """Create tasks for data processing pipeline."""
    
    # Data ingestion
    ingest_raw_task = planner.add_task(
        "ingest_raw_data",
        "Raw Data Ingestion",
        dependencies=set(),
        estimated_duration=2.0,
        priority=1.0,
        resource_requirements={"cpu": 0.3, "memory": 0.4, "io": 0.9},
        executor=lambda: {"status": "completed", "records": 1000000}
    )
    
    # Data validation and cleaning
    validate_task = planner.add_task(
        "validate_data",
        "Data Validation & Quality Check",
        dependencies={"ingest_raw_data"},
        estimated_duration=1.0,
        priority=0.9,
        resource_requirements={"cpu": 0.4, "memory": 0.5, "io": 0.3},
        executor=lambda: {"status": "completed", "quality_score": 0.94}
    )
    
    clean_task = planner.add_task(
        "clean_data",
        "Data Cleaning & Normalization",
        dependencies={"validate_data"},
        estimated_duration=3.0,
        priority=0.8,
        resource_requirements={"cpu": 0.6, "memory": 0.7, "io": 0.5},
        executor=lambda: {"status": "completed", "cleaned_records": 950000}
    )
    
    # Feature engineering
    features_task = planner.add_task(
        "feature_engineering",
        "Feature Engineering & Transformation",
        dependencies={"clean_data"},
        estimated_duration=4.0,
        priority=0.7,
        resource_requirements={"cpu": 0.8, "memory": 0.6, "io": 0.4},
        executor=lambda: {"status": "completed", "features": 127}
    )
    
    # Model training
    train_model_task = planner.add_task(
        "train_model",
        "ML Model Training",
        dependencies={"feature_engineering"},
        estimated_duration=6.0,
        priority=0.8,
        resource_requirements={"cpu": 0.9, "memory": 0.8, "io": 0.3},
        executor=lambda: {"status": "completed", "accuracy": 0.892}
    )
    
    # Model validation
    validate_model_task = planner.add_task(
        "validate_model",
        "Model Validation & Testing",
        dependencies={"train_model"},
        estimated_duration=2.0,
        priority=0.7,
        resource_requirements={"cpu": 0.5, "memory": 0.4, "io": 0.2},
        executor=lambda: {"status": "completed", "validation_score": 0.874}
    )
    
    # Results export
    export_results_task = planner.add_task(
        "export_results",
        "Export Results & Artifacts",
        dependencies={"validate_model"},
        estimated_duration=1.0,
        priority=0.6,
        resource_requirements={"cpu": 0.2, "memory": 0.3, "io": 0.8},
        executor=lambda: {"status": "completed", "artifacts": ["model.pkl", "metrics.json"]}
    )
    
    return [
        "ingest_raw_data", "validate_data", "clean_data", 
        "feature_engineering", "train_model", "validate_model", "export_results"
    ]


async def run_quantum_scheduler_demo(tasks: List[str], planner: QuantumTaskPlanner):
    """Demonstrate quantum scheduler capabilities."""
    print("\n🔬 QUANTUM SCHEDULER DEMO")
    print("=" * 50)
    
    scheduler = QuantumScheduler(
        max_concurrent_tasks=4,
        measurement_interval=0.5,
        interference_optimization=True
    )
    
    start_time = time.time()
    result = await scheduler.schedule(planner)
    execution_time = time.time() - start_time
    
    print(f"Scheduling Status: {result['status']}")
    print(f"Execution Time: {execution_time:.2f}s")
    print(f"Tasks Scheduled: {len(result['scheduled'])}")
    print(f"Quantum Paths Explored: {result.get('quantum_paths_explored', 0)}")
    
    # Show metrics
    metrics = scheduler.get_metrics()
    print(f"\nScheduler Metrics:")
    print(f"  Tasks Completed: {metrics.tasks_completed}")
    print(f"  Tasks Failed: {metrics.tasks_failed}")
    print(f"  Quantum Efficiency: {metrics.quantum_efficiency:.3f}")
    print(f"  Throughput: {metrics.throughput:.2f} tasks/sec")


async def run_adaptive_scheduler_demo(tasks: List[str], planner: QuantumTaskPlanner):
    """Demonstrate adaptive scheduler learning."""
    print("\n🧠 ADAPTIVE SCHEDULER DEMO")
    print("=" * 50)
    
    base_scheduler = QuantumScheduler(max_concurrent_tasks=3)
    adaptive_scheduler = AdaptiveScheduler(
        base_scheduler=base_scheduler,
        learning_rate=0.05,
        exploration_rate=0.1
    )
    
    # Run multiple rounds to show adaptation
    for round_num in range(3):
        print(f"\nAdaptation Round {round_num + 1}:")
        
        start_time = time.time()
        result = await adaptive_scheduler.schedule(planner)
        execution_time = time.time() - start_time
        
        print(f"  Execution Time: {execution_time:.2f}s")
        print(f"  Current Max Concurrent Tasks: {adaptive_scheduler.adaptive_params['max_concurrent_tasks']}")
        print(f"  Interference Strength: {adaptive_scheduler.adaptive_params['interference_strength']:.3f}")
        
        # Brief pause between rounds
        await asyncio.sleep(0.1)
    
    metrics = adaptive_scheduler.get_metrics()
    print(f"\nFinal Adaptive Metrics:")
    print(f"  Quantum Efficiency: {metrics.quantum_efficiency:.3f}")
    print(f"  Throughput: {metrics.throughput:.2f} tasks/sec")


async def run_optimizer_demo(tasks: List[str], planner: QuantumTaskPlanner):
    """Demonstrate quantum optimization."""
    print("\n⚡ QUANTUM OPTIMIZER DEMO")
    print("=" * 50)
    
    # Basic quantum optimizer
    basic_optimizer = QuantumOptimizer(max_iterations=50)
    
    print("Running basic quantum optimization...")
    result = basic_optimizer.optimize(planner.tasks)
    
    print(f"Basic Optimizer Results:")
    print(f"  Optimal Path Length: {len(result.optimal_path)}")
    print(f"  Optimization Score: {result.optimization_score:.4f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Converged: {result.convergence_achieved}")
    print(f"  Quantum Efficiency: {result.quantum_efficiency:.3f}")
    
    # Advanced interference optimizer
    print("\nRunning interference optimization...")
    interference_optimizer = InterferenceOptimizer(
        max_iterations=50,
        interference_strength=0.3,
        coherence_length=8
    )
    
    advanced_result = interference_optimizer.optimize(planner.tasks)
    
    print(f"\nInterference Optimizer Results:")
    print(f"  Optimal Path Length: {len(advanced_result.optimal_path)}")
    print(f"  Optimization Score: {advanced_result.optimization_score:.4f}")
    print(f"  Quantum Efficiency: {advanced_result.quantum_efficiency:.3f}")
    print(f"  Execution Time: {advanced_result.execution_time:.3f}s")


async def run_executor_demo(tasks: List[str], planner: QuantumTaskPlanner):
    """Demonstrate quantum execution."""
    print("\n⚙️ QUANTUM EXECUTOR DEMO")
    print("=" * 50)
    
    # Basic quantum executor
    executor = QuantumExecutor(
        max_concurrent_tasks=3,
        measurement_interval=0.5,
        error_probability=0.02
    )
    
    print("Executing tasks with quantum executor...")
    start_time = time.time()
    result = await executor.execute_tasks(planner.tasks, tasks[:5])  # First 5 tasks
    execution_time = time.time() - start_time
    
    print(f"Execution Results:")
    print(f"  Total Time: {execution_time:.2f}s")
    print(f"  Tasks Executed: {len(result['execution_results']['task_results'])}")
    print(f"  Quantum Measurements: {result['quantum_measurements']}")
    
    # Show task results
    for task_id, task_result in result['execution_results']['task_results'].items():
        status = task_result['status']
        duration = task_result['duration']
        print(f"    {task_id}: {status} ({duration:.2f}s)")
    
    # Parallel executor
    print("\nUsing parallel executor...")
    parallel_executor = ParallelExecutor(
        max_concurrent_tasks=4,
        max_worker_threads=3,
        adaptive_concurrency=True
    )
    
    start_time = time.time()
    parallel_result = await parallel_executor.execute_tasks(planner.tasks, tasks[:6])
    parallel_time = time.time() - start_time
    
    print(f"Parallel Execution:")
    print(f"  Total Time: {parallel_time:.2f}s")
    print(f"  Speedup: {execution_time/parallel_time:.2f}x")
    
    # Metrics comparison
    basic_metrics = executor.get_metrics()
    parallel_metrics = parallel_executor.get_metrics()
    
    print(f"\nMetrics Comparison:")
    print(f"  Basic Quantum Efficiency: {basic_metrics.quantum_efficiency:.3f}")
    print(f"  Parallel Quantum Efficiency: {parallel_metrics.quantum_efficiency:.3f}")
    print(f"  Parallel Utilization: {parallel_metrics.parallelism_achieved:.3f}")


async def comprehensive_workflow_demo():
    """Run comprehensive workflow combining all components."""
    print("\n🌟 COMPREHENSIVE QUANTUM WORKFLOW DEMO")
    print("=" * 60)
    
    # Create main planner
    main_planner = QuantumTaskPlanner(
        max_parallel_tasks=6,
        quantum_coherence_time=15.0,
        interference_strength=0.2
    )
    
    # Add multiple workflow types
    print("Creating multi-domain task workflows...")
    
    # Software development pipeline
    print("\n1. Software Development Pipeline:")
    sw_tasks = create_software_pipeline_tasks(main_planner)
    print(f"   Created {len(sw_tasks)} software development tasks")
    
    # Add fresh planner for infrastructure
    infra_planner = QuantumTaskPlanner()
    infra_tasks = create_infrastructure_tasks(infra_planner)
    print(f"2. Infrastructure Deployment:")
    print(f"   Created {len(infra_tasks)} infrastructure tasks")
    
    # Data processing pipeline
    data_planner = QuantumTaskPlanner()
    data_tasks = create_data_processing_tasks(data_planner)
    print(f"3. Data Processing Pipeline:")
    print(f"   Created {len(data_tasks)} data processing tasks")
    
    # Quantum system state
    print(f"\nQuantum System State:")
    system_state = main_planner.get_system_state()
    print(f"  Total Tasks: {system_state['num_tasks']}")
    print(f"  Quantum Entanglements: {system_state['entanglements']}")
    print(f"  Coherence Remaining: {system_state['quantum_coherence_remaining']:.1f}s")
    
    # Full quantum execution
    print(f"\n🚀 Executing Software Development Workflow:")
    
    # Use adaptive scheduler with parallel executor
    base_scheduler = QuantumScheduler(max_concurrent_tasks=4)
    adaptive_scheduler = AdaptiveScheduler(base_scheduler)
    parallel_executor = ParallelExecutor(max_concurrent_tasks=5, adaptive_concurrency=True)
    
    # Schedule and execute
    scheduling_result = await adaptive_scheduler.schedule(main_planner)
    if scheduling_result['status'] == 'success':
        scheduled_tasks = [task['task_id'] for task in scheduling_result['scheduled']]
        execution_result = await parallel_executor.execute_tasks(main_planner.tasks, scheduled_tasks)
        
        print(f"Workflow Execution Complete!")
        print(f"  Scheduled Tasks: {len(scheduled_tasks)}")
        print(f"  Execution Time: {execution_result['total_time']:.2f}s")
        print(f"  Success Rate: {len([r for r in execution_result['execution_results']['task_results'].values() if r['status'] == 'success'])}/{len(execution_result['execution_results']['task_results'])}")
    
    # Save comprehensive results
    results = {
        "workflow_type": "comprehensive_quantum_demo",
        "timestamp": time.time(),
        "software_tasks": len(sw_tasks),
        "infrastructure_tasks": len(infra_tasks), 
        "data_processing_tasks": len(data_tasks),
        "quantum_system_state": system_state,
        "scheduler_metrics": adaptive_scheduler.get_metrics().total_execution_time,
        "executor_metrics": parallel_executor.get_metrics().quantum_fidelity
    }
    
    # Save results
    results_dir = Path("experiments/quantum_planner_demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "comprehensive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {results_dir / 'comprehensive_results.json'}")


async def main():
    """Main demo execution."""
    print("🔮 QUANTUM-INSPIRED TASK PLANNER DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases quantum principles applied to task planning:")
    print("• Superposition: Multiple execution paths exist simultaneously") 
    print("• Entanglement: Dependencies create correlated task states")
    print("• Interference: Path optimization through quantum effects")
    print("• Measurement: Collapse to deterministic execution plans")
    print("=" * 80)
    
    # Create base planner for demos
    demo_planner = QuantumTaskPlanner()
    
    # Software pipeline demo
    print("\n📋 Creating Software Development Pipeline...")
    sw_tasks = create_software_pipeline_tasks(demo_planner)
    
    # Run individual component demos
    await run_quantum_scheduler_demo(sw_tasks, demo_planner)
    await run_adaptive_scheduler_demo(sw_tasks, demo_planner)  
    await run_optimizer_demo(sw_tasks, demo_planner)
    await run_executor_demo(sw_tasks, demo_planner)
    
    # Run comprehensive workflow
    await comprehensive_workflow_demo()
    
    print("\n" + "=" * 80)
    print("🎉 QUANTUM TASK PLANNER DEMO COMPLETE!")
    print("Key quantum advantages demonstrated:")
    print("• Parallel exploration of execution paths")
    print("• Adaptive optimization through quantum interference")
    print("• Self-tuning parameters via quantum learning")
    print("• Robust execution with quantum error handling")
    print("=" * 80)


if __name__ == "__main__":
    # Configure logging for demo
    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    # Run the demo
    asyncio.run(main())