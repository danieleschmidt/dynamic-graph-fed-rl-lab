"""
Quantum Hardware Integration Demo

Demonstrates real quantum computing hardware integration for federated learning
with IBM Quantum, Google Quantum, and AWS Braket platforms.
"""

import numpy as np
import jax.numpy as jnp
import time
from typing import Dict, Any

from dynamic_graph_fed_rl.quantum_hardware import (
    IBMQuantumBackend,
    GoogleQuantumBackend, 
    AWSBraketBackend,
    QuantumFederatedLearning,
    HybridClassicalQuantumOptimizer,
    QuantumAdvantageBenchmark,
    QuantumErrorCorrection
)
from dynamic_graph_fed_rl.quantum_hardware.quantum_fed_learning import (
    QuantumFederatedConfig,
    QuantumAggregationStrategy
)
from dynamic_graph_fed_rl.quantum_hardware.hybrid_optimizer import (
    HybridOptimizationConfig,
    OptimizationStrategy
)
from dynamic_graph_fed_rl.quantum_hardware.benchmarking import (
    BenchmarkConfig,
    BenchmarkType
)
from dynamic_graph_fed_rl.quantum_hardware.error_correction import (
    ErrorCorrectionConfig,
    ErrorCorrectionCode,
    NoiseMitigationTechnique
)


def setup_quantum_backends() -> Dict[str, Any]:
    """Set up quantum computing backends."""
    print("🔧 Setting up quantum computing backends...")
    
    backends = {}
    
    # IBM Quantum setup (requires real credentials)
    print("  Setting up IBM Quantum backend...")
    ibm_backend = IBMQuantumBackend()
    
    # For demo purposes, we'll use mock credentials
    # In real usage, replace with actual IBM Quantum token
    ibm_credentials = {
        "token": "YOUR_IBM_QUANTUM_TOKEN",  # Replace with real token
        "instance": "ibm-q/open/main"
    }
    
    # Try to connect (will fail without real credentials)
    try:
        if ibm_backend.connect(ibm_credentials):
            backends["ibm"] = ibm_backend
            print("    ✅ IBM Quantum connected")
        else:
            print("    ❌ IBM Quantum connection failed (credentials required)")
    except Exception as e:
        print(f"    ❌ IBM Quantum error: {e}")
    
    # Google Quantum setup (requires real credentials)
    print("  Setting up Google Quantum backend...")
    google_backend = GoogleQuantumBackend()
    
    google_credentials = {
        "project_id": "YOUR_GOOGLE_CLOUD_PROJECT"  # Replace with real project ID
    }
    
    try:
        if google_backend.connect(google_credentials):
            backends["google"] = google_backend
            print("    ✅ Google Quantum connected")
        else:
            print("    ❌ Google Quantum connection failed (credentials required)")
    except Exception as e:
        print(f"    ❌ Google Quantum error: {e}")
    
    # AWS Braket setup (requires real credentials)
    print("  Setting up AWS Braket backend...")
    braket_backend = AWSBraketBackend()
    
    braket_credentials = {
        "aws_access_key_id": "YOUR_ACCESS_KEY",        # Replace with real credentials
        "aws_secret_access_key": "YOUR_SECRET_KEY",    # Replace with real credentials
        "region": "us-east-1",
        "s3_bucket": "amazon-braket-quantum-results"
    }
    
    try:
        if braket_backend.connect(braket_credentials):
            backends["aws"] = braket_backend
            print("    ✅ AWS Braket connected")
        else:
            print("    ❌ AWS Braket connection failed (credentials required)")
    except Exception as e:
        print(f"    ❌ AWS Braket error: {e}")
    
    if not backends:
        print("    ⚠️  No real quantum backends available - using simulator for demo")
        # For demo purposes, create a mock backend
        from tests.quantum_hardware.test_quantum_backends import MockQuantumBackend
        mock_backend = MockQuantumBackend()
        mock_backend.connect({})
        backends["simulator"] = mock_backend
    
    return backends


def demonstrate_quantum_federated_aggregation(backends: Dict[str, Any]):
    """Demonstrate quantum federated parameter aggregation."""
    print("\n🧮 Demonstrating Quantum Federated Parameter Aggregation...")
    
    # Configuration for quantum federated learning
    config = QuantumFederatedConfig(
        aggregation_strategy=QuantumAggregationStrategy.QUANTUM_WEIGHTED_AVERAGE,
        num_qubits=6,
        circuit_depth=3,
        shots=1000,
        optimization_iterations=20,
        quantum_advantage_threshold=0.05,
        noise_mitigation=True,
        error_correction=False
    )
    
    # Create quantum federated learning system
    qfl = QuantumFederatedLearning(backends, config)
    
    print(f"  Initialized with {len(backends)} backend(s)")
    
    # Generate synthetic client parameters (simulating federated learning scenario)
    num_clients = 5
    parameter_dim = 20
    
    print(f"  Simulating {num_clients} clients with {parameter_dim}-dimensional parameters")
    
    client_parameters = []
    for i in range(num_clients):
        # Each client has slightly different parameters (representing local training)
        base_params = np.random.normal(0, 1, parameter_dim)
        noise = np.random.normal(0, 0.1, parameter_dim)  # Small client-specific variation
        client_params = jnp.array(base_params + noise)
        client_parameters.append(client_params)
    
    # Client weights (based on data size or quality)
    client_weights = jnp.array([0.25, 0.20, 0.20, 0.20, 0.15])
    
    # Perform quantum federated aggregation
    print("  Executing quantum federated rounds...")
    
    for round_num in range(3):
        print(f"    Round {round_num + 1}:")
        start_time = time.time()
        
        aggregated_params, round_info = qfl.federated_round(
            client_parameters, 
            client_weights, 
            round_number=round_num
        )
        
        execution_time = time.time() - start_time
        
        print(f"      Execution time: {execution_time:.3f}s")
        print(f"      Quantum backends used: {round_info['quantum_backends_used']}")
        print(f"      Aggregated parameter norm: {jnp.linalg.norm(aggregated_params):.3f}")
        
        # Show which backends succeeded
        for backend_name, result in round_info["results"].items():
            status = "✅" if result["success"] else "❌"
            print(f"        {backend_name}: {status}")
    
    # Get performance metrics
    metrics = qfl.get_performance_metrics()
    print(f"\n  Performance Summary:")
    print(f"    Total rounds: {metrics['total_rounds']}")
    print(f"    Quantum success rate: {metrics['quantum_success_rate']:.1%}")
    print(f"    Average execution time: {metrics['average_execution_time']:.3f}s")
    print(f"    Quantum advantage achieved: {metrics['quantum_advantage_achieved']}")


def demonstrate_hybrid_graph_optimization(backends: Dict[str, Any]):
    """Demonstrate hybrid classical-quantum graph optimization."""
    print("\n📊 Demonstrating Hybrid Classical-Quantum Graph Optimization...")
    
    # Configuration for hybrid optimization
    config = HybridOptimizationConfig(
        strategy=OptimizationStrategy.QAOA_CLASSICAL,
        max_classical_iterations=50,
        max_quantum_iterations=10,
        classical_learning_rate=0.1,
        quantum_learning_rate=0.05,
        convergence_threshold=1e-4,
        use_parameter_shift=True,
        noise_mitigation=True
    )
    
    # Create hybrid optimizer
    optimizer = HybridClassicalQuantumOptimizer(backends, config)
    
    # Generate test graphs representing dynamic federated learning topologies
    print("  Generating dynamic graph instances...")
    
    graph_snapshots = []
    
    # Time step 1: Dense connectivity
    graph_t1 = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
    graph_snapshots.append(graph_t1)
    
    # Time step 2: Changed topology
    graph_t2 = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4), (4, 0)]
    graph_snapshots.append(graph_t2)
    
    # Time step 3: Sparse connectivity
    graph_t3 = [(0, 2), (1, 3), (2, 4)]
    graph_snapshots.append(graph_t3)
    
    print(f"  Generated {len(graph_snapshots)} graph snapshots")
    for i, graph in enumerate(graph_snapshots):
        nodes = len(set([v for edge in graph for v in edge]))
        print(f"    T{i+1}: {nodes} nodes, {len(graph)} edges")
    
    # Temporal weights (recent snapshots more important)
    temporal_weights = jnp.array([0.2, 0.3, 0.5])
    
    print("  Optimizing dynamic graph partition using quantum algorithms...")
    start_time = time.time()
    
    optimization_result = optimizer.optimize_dynamic_graph_partition(
        graph_snapshots, 
        temporal_weights
    )
    
    execution_time = time.time() - start_time
    
    print(f"  Optimization completed in {execution_time:.3f}s")
    print(f"  Total optimization value: {optimization_result['optimization_value']:.3f}")
    print(f"  Quantum advantage achieved: {optimization_result['quantum_advantage']}")
    
    # Show backend-specific results
    print("  Backend results:")
    for backend_name, result in optimization_result["backend_results"].items():
        if result["success"]:
            print(f"    {backend_name}: ✅ Value = {result['weighted_total_value']:.3f}")
        else:
            print(f"    {backend_name}: ❌ {result.get('error', 'Failed')}")


def demonstrate_error_correction_and_mitigation(backends: Dict[str, Any]):
    """Demonstrate quantum error correction and noise mitigation."""
    print("\n🛡️  Demonstrating Quantum Error Correction & Noise Mitigation...")
    
    # Error correction configuration
    config = ErrorCorrectionConfig(
        code_type=ErrorCorrectionCode.REPETITION_CODE,
        code_distance=3,
        syndrome_extraction_rounds=2,
        error_threshold=0.1,
        mitigation_techniques=[
            NoiseMitigationTechnique.ZERO_NOISE_EXTRAPOLATION,
            NoiseMitigationTechnique.READOUT_ERROR_MITIGATION
        ],
        use_logical_qubits=True
    )
    
    # Create error correction system
    error_correction = QuantumErrorCorrection(config)
    
    print(f"  Using {config.code_type.value} with distance {config.code_distance}")
    print(f"  Mitigation techniques: {[t.value for t in config.mitigation_techniques]}")
    
    # Get resource overhead
    overhead = error_correction.get_error_correction_overhead()
    print(f"  Resource overhead: {overhead}")
    
    # Create a simple test circuit
    from dynamic_graph_fed_rl.quantum_hardware.base import QuantumCircuitBuilder
    
    logical_circuit = (QuantumCircuitBuilder(2)
                      .h(0)
                      .cnot(0, 1)
                      .measure_all()
                      .build())
    
    print("  Testing error correction on sample circuit...")
    
    # Apply error correction to each available backend
    for backend_name, backend in backends.items():
        print(f"    Testing with {backend_name} backend:")
        
        try:
            # Get available devices
            devices = backend.get_available_devices()
            if devices:
                device = devices[0]["name"]
                print(f"      Using device: {device}")
                
                # Apply error correction
                start_time = time.time()
                corrected_result = error_correction.apply_error_correction(
                    backend, logical_circuit, device, shots=100
                )
                execution_time = time.time() - start_time
                
                if corrected_result.success:
                    print(f"      ✅ Success in {execution_time:.3f}s")
                    print(f"      Measurement counts: {len(corrected_result.counts)} outcomes")
                else:
                    print(f"      ❌ Failed: {corrected_result.error_message}")
            else:
                print("      ❌ No devices available")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")


def run_quantum_advantage_benchmark(backends: Dict[str, Any]):
    """Run comprehensive quantum advantage benchmark."""
    print("\n🏆 Running Quantum Advantage Benchmark...")
    
    # Benchmark configuration
    config = BenchmarkConfig(
        benchmark_type=BenchmarkType.PARAMETER_AGGREGATION,
        num_trials=5,
        problem_sizes=[8, 16, 32],
        classical_baselines=["fedavg", "weighted_average", "median_aggregation"],
        quantum_algorithms=["quantum_weighted_average", "variational_quantum_eigensolver"],
        noise_levels=[0.0, 0.1, 0.2],
        time_limit_seconds=30.0,
        quality_threshold=1.05,  # 5% improvement threshold
        statistical_significance_level=0.05
    )
    
    # Create benchmark system
    benchmark = QuantumAdvantageBenchmark(config)
    
    print(f"  Benchmark type: {config.benchmark_type.value}")
    print(f"  Problem sizes: {config.problem_sizes}")
    print(f"  Number of trials per size: {config.num_trials}")
    
    # Set up quantum federated learning for benchmarking
    qfl_config = QuantumFederatedConfig(
        aggregation_strategy=QuantumAggregationStrategy.QUANTUM_WEIGHTED_AVERAGE,
        num_qubits=6,
        circuit_depth=2,
        shots=500,  # Reduced for faster benchmarking
        optimization_iterations=10,
        quantum_advantage_threshold=0.05,
        noise_mitigation=True,
        error_correction=False
    )
    
    qfl = QuantumFederatedLearning(backends, qfl_config)
    
    print("  Running comprehensive benchmark...")
    start_time = time.time()
    
    results = benchmark.run_comprehensive_benchmark(quantum_fed_learning=qfl)
    
    total_time = time.time() - start_time
    print(f"  Benchmark completed in {total_time:.1f}s")
    
    # Display results
    metrics = results["quantum_advantage_metrics"]
    print(f"\n  Quantum Advantage Analysis:")
    print(f"    Overall quantum advantage: {metrics['overall_quantum_advantage']}")
    
    if "average_advantage" in metrics:
        print(f"    Average advantage factor: {metrics['average_advantage']:.3f}")
        print(f"    Advantage std deviation: {metrics['advantage_std']:.3f}")
    
    print(f"\n  Performance by problem size:")
    for size, size_metrics in metrics["advantage_by_problem_size"].items():
        print(f"    Size {size}:")
        for metric_name, value in size_metrics.items():
            if isinstance(value, (int, float)):
                print(f"      {metric_name}: {value:.3f}")
            else:
                print(f"      {metric_name}: {value}")
    
    # Generate and display report
    print("\n" + "="*50)
    print(benchmark.generate_benchmark_report())
    print("="*50)
    
    return results


def main():
    """Main demo function."""
    print("🚀 Quantum Hardware Integration Demo")
    print("Moving from quantum-inspired to quantum-native implementations\n")
    
    # Setup quantum backends
    backends = setup_quantum_backends()
    
    if not backends:
        print("❌ No quantum backends available. Exiting.")
        return
    
    print(f"\n✅ Successfully initialized {len(backends)} quantum backend(s)")
    
    # List available devices
    print("\n📋 Available quantum devices:")
    for backend_name, backend in backends.items():
        print(f"  {backend_name}:")
        devices = backend.get_available_devices()
        for device in devices:
            device_type = "Simulator" if device.get("simulator", True) else "Hardware"
            print(f"    - {device['name']}: {device.get('qubits', 'Unknown')} qubits ({device_type})")
    
    try:
        # Demo 1: Quantum Federated Parameter Aggregation
        demonstrate_quantum_federated_aggregation(backends)
        
        # Demo 2: Hybrid Graph Optimization
        demonstrate_hybrid_graph_optimization(backends)
        
        # Demo 3: Error Correction and Mitigation
        demonstrate_error_correction_and_mitigation(backends)
        
        # Demo 4: Quantum Advantage Benchmarking
        benchmark_results = run_quantum_advantage_benchmark(backends)
        
        print(f"\n🎉 All demonstrations completed successfully!")
        print(f"Real quantum advantage demonstrated: {benchmark_results['quantum_advantage_metrics']['overall_quantum_advantage']}")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()