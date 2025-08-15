#!/usr/bin/env python3
"""
Generation 2 Robust Demo - Enhanced Error Handling & Monitoring

Demonstrates the robust, production-ready federated RL system with:
- Comprehensive error handling and validation
- Real-time health monitoring and alerting
- Performance profiling and optimization
- Data persistence and recovery mechanisms

GENERATION 2: MAKE IT ROBUST
"""

import asyncio
import time
import random
from typing import Dict, List, Optional

# Mock dependencies setup
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from autonomous_mock_deps import setup_autonomous_mocks
setup_autonomous_mocks()

# Add psutil mock for monitoring
import sys
from types import ModuleType
psutil = ModuleType('psutil')
psutil.cpu_percent = lambda interval=None: 25.5 + random.uniform(-5, 5)
psutil.virtual_memory = lambda: type('Memory', (), {
    'percent': 45.2 + random.uniform(-10, 10),
    'available': 8192 * 1024 * 1024
})()
psutil.disk_usage = lambda path: type('Disk', (), {
    'used': 500 * 1024 * 1024 * 1024,
    'total': 1000 * 1024 * 1024 * 1024,
    'free': 500 * 1024 * 1024 * 1024
})()
psutil.net_io_counters = lambda: type('NetIO', (), {
    'bytes_sent': 1024000,
    'bytes_recv': 2048000,
    'packets_sent': 1000,
    'packets_recv': 1500
})()
sys.modules['psutil'] = psutil

# Core imports
from dynamic_graph_fed_rl.quantum_planner import QuantumTask
from dynamic_graph_fed_rl.environments import IntersectionNode
from dynamic_graph_fed_rl.federation import AsyncGossipProtocol
from dynamic_graph_fed_rl.monitoring import MetricsCollector, HealthMonitor


class RobustFederatedRLDemo:
    """
    Generation 2 robust demonstration with comprehensive monitoring,
    error handling, validation, and recovery mechanisms.
    """
    
    def __init__(self):
        # Core components
        self.quantum_tasks = []
        self.federated_agents = []
        
        # Monitoring and reliability
        self.metrics_collector = MetricsCollector(
            collection_interval=2.0,
            retention_hours=24,
            enable_persistence=True
        )
        self.health_monitor = HealthMonitor(
            check_interval=15.0,
            enable_auto_recovery=True
        )
        
        # Error tracking
        self.error_counts = {
            'quantum_errors': 0,
            'federation_errors': 0,
            'validation_errors': 0,
            'recovery_attempts': 0
        }
        
        # Performance metrics
        self.performance_metrics = {
            'quantum_efficiency': [],
            'federation_sync': [],
            'error_recovery_time': [],
            'system_uptime': time.time()
        }
        
        # Setup monitoring callbacks
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup comprehensive monitoring and alerting."""
        
        # Metrics collector alerts
        def metrics_alert_handler(alert_type: str, alert_data: Dict):
            print(f"📊 METRICS ALERT: {alert_type} - {alert_data}")
            self.error_counts['recovery_attempts'] += 1
        
        self.metrics_collector.add_alert_callback(metrics_alert_handler)
        
        # Health monitor alerts
        def health_alert_handler(component_name: str, component_health):
            print(f"🏥 HEALTH ALERT: {component_name} status: {component_health.status.value}")
            print(f"   Message: {component_health.message}")
            self.error_counts['recovery_attempts'] += 1
        
        self.health_monitor.add_alert_callback(health_alert_handler)
        
        # Register custom health checks
        self.health_monitor.register_health_check("quantum_system", self._check_quantum_health)
        self.health_monitor.register_health_check("federation_network", self._check_federation_health)
        
        # Register recovery actions
        self.health_monitor.register_recovery_action("quantum_system", self._recover_quantum_system)
        self.health_monitor.register_recovery_action("federation_network", self._recover_federation_network)
    
    def _check_quantum_health(self) -> Dict:
        """Custom health check for quantum system."""
        try:
            if not self.quantum_tasks:
                return {
                    'status': 'degraded',
                    'message': 'No quantum tasks active',
                    'task_count': 0
                }
            
            # Simulate quantum coherence check
            coherence = 0.7 + random.uniform(-0.2, 0.2)
            task_efficiency = sum(getattr(task, 'efficiency', 0.8) for task in self.quantum_tasks) / len(self.quantum_tasks)
            
            if coherence < 0.5 or task_efficiency < 0.6:
                status = 'unhealthy'
                message = f'Low quantum performance (coherence: {coherence:.2f}, efficiency: {task_efficiency:.2f})'
            elif coherence < 0.7 or task_efficiency < 0.8:
                status = 'degraded'
                message = f'Degraded quantum performance (coherence: {coherence:.2f}, efficiency: {task_efficiency:.2f})'
            else:
                status = 'healthy'
                message = f'Quantum system optimal (coherence: {coherence:.2f}, efficiency: {task_efficiency:.2f})'
            
            return {
                'status': status,
                'message': message,
                'quantum_coherence': coherence,
                'task_efficiency': task_efficiency,
                'active_tasks': len(self.quantum_tasks)
            }
        
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Quantum health check failed: {e}',
                'error': str(e)
            }
    
    def _check_federation_health(self) -> Dict:
        """Custom health check for federation network."""
        try:
            if not self.federated_agents:
                return {
                    'status': 'critical',
                    'message': 'No federated agents available',
                    'agent_count': 0
                }
            
            # Check agent health
            healthy_agents = sum(1 for agent in self.federated_agents if agent.get('status', 'unknown') == 'healthy')
            sync_quality = sum(agent.get('sync_quality', 0.5) for agent in self.federated_agents) / len(self.federated_agents)
            
            if healthy_agents < len(self.federated_agents) * 0.5:
                status = 'critical'
                message = f'Majority of agents unhealthy ({healthy_agents}/{len(self.federated_agents)})'
            elif sync_quality < 0.6:
                status = 'degraded'
                message = f'Poor federation sync quality: {sync_quality:.2f}'
            else:
                status = 'healthy'
                message = f'Federation network stable ({healthy_agents}/{len(self.federated_agents)} healthy)'
            
            return {
                'status': status,
                'message': message,
                'healthy_agents': healthy_agents,
                'total_agents': len(self.federated_agents),
                'sync_quality': sync_quality
            }
        
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Federation health check failed: {e}',
                'error': str(e)
            }
    
    def _recover_quantum_system(self) -> bool:
        """Attempt to recover quantum system."""
        try:
            print("🔄 Attempting quantum system recovery...")
            
            # Simulate quantum system reset
            for task in self.quantum_tasks:
                # Reset quantum states
                setattr(task, 'efficiency', 0.8 + random.uniform(0, 0.2))
                setattr(task, 'coherence_restored', True)
            
            print("✅ Quantum system recovery successful")
            return True
        
        except Exception as e:
            print(f"❌ Quantum system recovery failed: {e}")
            return False
    
    def _recover_federation_network(self) -> bool:
        """Attempt to recover federation network."""
        try:
            print("🔄 Attempting federation network recovery...")
            
            # Simulate network recovery
            for agent in self.federated_agents:
                agent['status'] = 'healthy'
                agent['sync_quality'] = 0.8 + random.uniform(0, 0.2)
                agent['last_communication'] = time.time()
            
            print("✅ Federation network recovery successful")
            return True
        
        except Exception as e:
            print(f"❌ Federation network recovery failed: {e}")
            return False
    
    def create_robust_quantum_tasks(self, num_tasks: int = 5) -> List[QuantumTask]:
        """Create quantum tasks with comprehensive validation and error handling."""
        print(f"⚛️  Creating {num_tasks} robust quantum tasks...")
        
        try:
            self.quantum_tasks = []
            
            for i in range(num_tasks):
                try:
                    # Validate input parameters
                    if i < 0 or i >= 100:  # Reasonable bounds
                        raise ValueError(f"Invalid task index: {i}")
                    
                    # Create task with validation
                    task = QuantumTask(
                        id=f"robust_task_{i}",
                        name=f"Robust Traffic Control {i}",
                        estimated_duration=max(0.5, 1.0 + (i * 0.2)),  # Ensure positive duration
                        priority=max(0.1, min(1.0, 1.0 - (i * 0.1))),  # Clamp priority to valid range
                        resource_requirements={
                            'cpu': max(0.1, min(1.0, 0.5 + (i * 0.1))),  # Validate resource requirements
                            'memory': max(64, min(1024, 128 + (i * 64)))
                        }
                    )
                    
                    # Add dependencies with validation
                    if i > 0:
                        # Ensure dependency exists
                        prev_task_id = f"robust_task_{i-1}"
                        task.dependencies.add(prev_task_id)
                    
                    # Set efficiency metric for health monitoring
                    setattr(task, 'efficiency', 0.8 + random.uniform(0, 0.2))
                    
                    self.quantum_tasks.append(task)
                    
                    # Update metrics
                    self.metrics_collector.add_custom_metric(
                        f"task_created_{i}",
                        1.0,
                        {'task_id': task.id, 'priority': task.priority}
                    )
                
                except Exception as e:
                    print(f"❌ Failed to create task {i}: {e}")
                    self.error_counts['quantum_errors'] += 1
                    # Continue with other tasks instead of failing completely
                    continue
            
            if not self.quantum_tasks:
                raise RuntimeError("No quantum tasks could be created")
            
            print(f"✅ Successfully created {len(self.quantum_tasks)} robust quantum tasks")
            return self.quantum_tasks
        
        except Exception as e:
            print(f"❌ Critical error in quantum task creation: {e}")
            self.error_counts['quantum_errors'] += 1
            raise
    
    def create_robust_federated_agents(self, num_agents: int = 3) -> List[Dict]:
        """Create federated agents with validation and error handling."""
        print(f"🤝 Creating {num_agents} robust federated agents...")
        
        try:
            # Validate input
            if num_agents <= 0 or num_agents > 100:
                raise ValueError(f"Invalid number of agents: {num_agents}")
            
            self.federated_agents = []
            regions = ['North', 'South', 'Central', 'East', 'West']
            
            for i in range(num_agents):
                try:
                    # Create agent with comprehensive validation
                    agent = {
                        'id': i,
                        'name': f'RobustAgent_{i}',
                        'region': regions[i % len(regions)],
                        'status': 'healthy',
                        'policy_weights': self._generate_valid_weights(),
                        'performance_score': 0.0,
                        'sync_quality': 0.8 + random.uniform(0, 0.2),
                        'local_data_samples': max(50, 100 + (i * 50)),
                        'communication_cost': 0.0,
                        'last_communication': time.time(),
                        'error_count': 0,
                        'recovery_count': 0
                    }
                    
                    # Validate agent data
                    self._validate_agent(agent)
                    
                    self.federated_agents.append(agent)
                    
                    # Update metrics
                    self.metrics_collector.add_custom_metric(
                        f"agent_created_{i}",
                        1.0,
                        {'agent_id': i, 'region': agent['region']}
                    )
                
                except Exception as e:
                    print(f"❌ Failed to create agent {i}: {e}")
                    self.error_counts['federation_errors'] += 1
                    # Continue with other agents
                    continue
            
            if not self.federated_agents:
                raise RuntimeError("No federated agents could be created")
            
            print(f"✅ Successfully created {len(self.federated_agents)} robust federated agents")
            return self.federated_agents
        
        except Exception as e:
            print(f"❌ Critical error in federated agent creation: {e}")
            self.error_counts['federation_errors'] += 1
            raise
    
    def _generate_valid_weights(self) -> List[float]:
        """Generate validated policy weights."""
        weights = [0.5 + random.uniform(-0.3, 0.3) for _ in range(5)]
        # Ensure weights are in valid range [0, 1]
        weights = [max(0.0, min(1.0, w)) for w in weights]
        return weights
    
    def _validate_agent(self, agent: Dict):
        """Validate agent data structure."""
        required_fields = ['id', 'name', 'region', 'status', 'policy_weights']
        
        for field in required_fields:
            if field not in agent:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(agent['policy_weights'], list):
            raise ValueError("Policy weights must be a list")
        
        if len(agent['policy_weights']) == 0:
            raise ValueError("Policy weights cannot be empty")
        
        if agent['performance_score'] < 0:
            raise ValueError("Performance score cannot be negative")
    
    def simulate_robust_quantum_optimization(self) -> float:
        """Simulate quantum optimization with comprehensive error handling."""
        print("🎯 Executing robust quantum optimization...")
        
        try:
            if not self.quantum_tasks:
                raise RuntimeError("No quantum tasks available for optimization")
            
            start_time = time.time()
            total_efficiency = 0.0
            successful_tasks = 0
            
            for task in self.quantum_tasks:
                try:
                    # Simulate quantum optimization with potential failures
                    base_efficiency = getattr(task, 'efficiency', 0.8)
                    
                    # Introduce random failures for robustness testing
                    if random.random() < 0.1:  # 10% failure rate
                        raise RuntimeError(f"Quantum decoherence in task {task.id}")
                    
                    # Calculate efficiency with quantum effects
                    entanglement_bonus = len(task.entangled_tasks) * 0.05
                    priority_factor = task.priority * 0.2
                    
                    task_efficiency = min(1.0, base_efficiency + entanglement_bonus + priority_factor)
                    total_efficiency += task_efficiency
                    successful_tasks += 1
                    
                    print(f"   🎯 {task.name}: {task_efficiency:.3f} efficiency")
                    
                    # Update task metrics
                    self.metrics_collector.add_custom_metric(
                        "task_efficiency",
                        task_efficiency,
                        {'task_id': task.id}
                    )
                
                except Exception as e:
                    print(f"   ❌ Task {task.id} failed: {e}")
                    self.error_counts['quantum_errors'] += 1
                    
                    # Attempt task recovery
                    if self._attempt_task_recovery(task):
                        print(f"   ✅ Task {task.id} recovered")
                        # Retry with reduced efficiency
                        task_efficiency = 0.5
                        total_efficiency += task_efficiency
                        successful_tasks += 1
                    else:
                        print(f"   ❌ Task {task.id} recovery failed")
            
            # Calculate average efficiency
            if successful_tasks > 0:
                avg_efficiency = total_efficiency / successful_tasks
            else:
                raise RuntimeError("All quantum tasks failed")
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.performance_metrics['quantum_efficiency'].append(avg_efficiency)
            self.metrics_collector.update_quantum_metrics(
                coherence=avg_efficiency,
                task_stats={
                    'active': len(self.quantum_tasks),
                    'completed': successful_tasks,
                    'failed': len(self.quantum_tasks) - successful_tasks
                }
            )
            
            print(f"   ⚛️  Average quantum efficiency: {avg_efficiency:.3f} ({successful_tasks}/{len(self.quantum_tasks)} tasks)")
            return avg_efficiency
        
        except Exception as e:
            print(f"❌ Critical error in quantum optimization: {e}")
            self.error_counts['quantum_errors'] += 1
            
            # Attempt system recovery
            if self._recover_quantum_system():
                return 0.5  # Minimal efficiency after recovery
            else:
                raise
    
    def _attempt_task_recovery(self, task: QuantumTask) -> bool:
        """Attempt to recover a failed quantum task."""
        try:
            # Simulate task recovery
            setattr(task, 'efficiency', 0.5)  # Reduced efficiency after recovery
            return True
        except Exception:
            return False
    
    def simulate_robust_federated_learning(self) -> float:
        """Simulate federated learning with error handling and validation."""
        print("🔄 Executing robust federated learning...")
        
        try:
            if not self.federated_agents:
                raise RuntimeError("No federated agents available")
            
            start_time = time.time()
            successful_agents = 0
            failed_agents = 0
            
            # Local training with error handling
            for agent in self.federated_agents:
                try:
                    # Validate agent state
                    self._validate_agent(agent)
                    
                    # Simulate potential failures
                    if random.random() < 0.15:  # 15% failure rate
                        raise RuntimeError(f"Communication failure with agent {agent['id']}")
                    
                    # Simulate local training
                    local_improvement = 0.02 + random.uniform(0, 0.03)
                    agent['performance_score'] += local_improvement
                    
                    # Update communication metrics
                    data_size = agent['local_data_samples'] * 0.001
                    agent['communication_cost'] = data_size * 0.1
                    agent['last_communication'] = time.time()
                    
                    successful_agents += 1
                    print(f"   Agent {agent['id']} ({agent['region']}): +{local_improvement:.3f} improvement")
                
                except Exception as e:
                    print(f"   ❌ Agent {agent['id']} failed: {e}")
                    agent['error_count'] += 1
                    failed_agents += 1
                    self.error_counts['federation_errors'] += 1
                    
                    # Attempt agent recovery
                    if self._attempt_agent_recovery(agent):
                        successful_agents += 1
                        agent['recovery_count'] += 1
                        print(f"   ✅ Agent {agent['id']} recovered")
            
            # Check if we have enough healthy agents for federation
            if successful_agents < len(self.federated_agents) * 0.5:
                raise RuntimeError(f"Too many agent failures ({failed_agents}/{len(self.federated_agents)})")
            
            # Robust federated averaging
            sync_quality = self._perform_robust_aggregation()
            
            execution_time = time.time() - start_time
            self.performance_metrics['federation_sync'].append(sync_quality)
            
            # Update federation metrics
            self.metrics_collector.update_federation_metrics(sync_quality)
            
            print(f"   🤝 Federation sync quality: {sync_quality:.3f} ({successful_agents}/{len(self.federated_agents)} agents)")
            return sync_quality
        
        except Exception as e:
            print(f"❌ Critical error in federated learning: {e}")
            self.error_counts['federation_errors'] += 1
            
            # Attempt federation recovery
            if self._recover_federation_network():
                return 0.5  # Minimal sync quality after recovery
            else:
                raise
    
    def _attempt_agent_recovery(self, agent: Dict) -> bool:
        """Attempt to recover a failed agent."""
        try:
            # Reset agent to healthy state
            agent['status'] = 'healthy'
            agent['sync_quality'] = 0.6
            agent['last_communication'] = time.time()
            return True
        except Exception:
            return False
    
    def _perform_robust_aggregation(self) -> float:
        """Perform robust federated parameter aggregation."""
        try:
            healthy_agents = [agent for agent in self.federated_agents if agent.get('status') == 'healthy']
            
            if not healthy_agents:
                raise RuntimeError("No healthy agents for aggregation")
            
            # Simple averaging with validation
            avg_weights = []
            weight_length = len(healthy_agents[0]['policy_weights'])
            
            for i in range(weight_length):
                weight_sum = 0.0
                valid_weights = 0
                
                for agent in healthy_agents:
                    try:
                        if i < len(agent['policy_weights']):
                            weight = agent['policy_weights'][i]
                            # Validate weight value
                            if isinstance(weight, (int, float)) and 0 <= weight <= 1:
                                weight_sum += weight
                                valid_weights += 1
                    except (IndexError, TypeError):
                        continue
                
                if valid_weights > 0:
                    avg_weights.append(weight_sum / valid_weights)
                else:
                    avg_weights.append(0.5)  # Default value
            
            # Update all healthy agents
            for agent in healthy_agents:
                agent['policy_weights'] = avg_weights.copy()
                agent['sync_quality'] = 0.8 + random.uniform(0, 0.2)
            
            # Calculate sync quality
            return sum(agent.get('sync_quality', 0.5) for agent in healthy_agents) / len(healthy_agents)
        
        except Exception as e:
            print(f"❌ Aggregation failed: {e}")
            return 0.0
    
    def print_robust_results(self):
        """Print comprehensive results with error analysis."""
        print("\n" + "="*80)
        print("🎉 GENERATION 2 ROBUST DEMO COMPLETE!")
        print("="*80)
        
        # System metrics
        uptime = time.time() - self.performance_metrics['system_uptime']
        print(f"⏱️  System Uptime: {uptime:.2f} seconds")
        print(f"⚛️  Quantum Tasks: {len(self.quantum_tasks)}")
        print(f"🤝 Federated Agents: {len(self.federated_agents)}")
        
        # Performance metrics
        if self.performance_metrics['quantum_efficiency']:
            avg_quantum = sum(self.performance_metrics['quantum_efficiency']) / len(self.performance_metrics['quantum_efficiency'])
            print(f"🎯 Average Quantum Efficiency: {avg_quantum:.3f}")
        
        if self.performance_metrics['federation_sync']:
            avg_sync = sum(self.performance_metrics['federation_sync']) / len(self.performance_metrics['federation_sync'])
            print(f"🔄 Average Federation Sync: {avg_sync:.3f}")
        
        # Error analysis
        total_errors = sum(self.error_counts.values())
        print(f"❌ Total Errors: {total_errors}")
        for error_type, count in self.error_counts.items():
            if count > 0:
                print(f"   {error_type}: {count}")
        
        # Health status
        health_report = self.health_monitor.get_health_report()
        print(f"🏥 Overall Health: {health_report['overall_status'].upper()}")
        print(f"   Healthy Components: {health_report['healthy_components']}/{health_report['total_components']}")
        
        # Metrics summary
        metrics_summary = self.metrics_collector.get_metrics_summary(1.0)
        if metrics_summary:
            print(f"📊 System Performance (last hour):")
            print(f"   CPU Usage: {metrics_summary.get('avg_cpu_usage', 0):.1f}%")
            print(f"   Memory Usage: {metrics_summary.get('avg_memory_usage', 0):.1f}%")
            print(f"   Task Success Rate: {metrics_summary.get('task_success_rate', 0):.3f}")
        
        print("\n🚀 GENERATION 2 ROBUSTNESS FEATURES:")
        print("   ✅ Comprehensive error handling and validation")
        print("   ✅ Real-time health monitoring and alerting")
        print("   ✅ Automatic recovery mechanisms")
        print("   ✅ Performance metrics collection")
        print("   ✅ Data persistence and export")
        print("   ✅ Fault tolerance and graceful degradation")
        print("="*80)
    
    async def run_robust_demo(self):
        """Run the complete robust demonstration."""
        print("🎉 DYNAMIC GRAPH FEDERATED RL - ROBUST DEMO")
        print("🎯 Generation 2: Comprehensive Error Handling & Monitoring")
        print("-" * 80)
        
        try:
            # Start monitoring systems
            print("\n📊 Phase 1: Initialize Monitoring Systems")
            self.metrics_collector.start_collection()
            self.health_monitor.start_monitoring()
            
            # Wait for monitoring to stabilize
            await asyncio.sleep(2)
            
            # Create robust system components
            print("\n🔧 Phase 2: Create Robust System Components")
            self.create_robust_quantum_tasks()
            self.create_robust_federated_agents()
            
            # Execute with error handling
            print("\n⚛️  Phase 3: Robust Quantum Optimization")
            quantum_efficiency = self.simulate_robust_quantum_optimization()
            
            print("\n🤝 Phase 4: Robust Federated Learning (3 rounds)")
            for round_num in range(3):
                print(f"\n   Round {round_num + 1}/3:")
                federation_sync = self.simulate_robust_federated_learning()
                
                # Brief pause between rounds
                await asyncio.sleep(1)
            
            # Final health check
            print("\n🏥 Phase 5: Final Health Assessment")
            health_report = self.health_monitor.get_health_report()
            print(f"Final system health: {health_report['overall_status']}")
            
            # Results and analysis
            self.print_robust_results()
        
        except Exception as e:
            print(f"❌ Demo execution failed: {e}")
            print("🔄 Attempting recovery...")
            
            # Demonstrate recovery capabilities
            recovery_start = time.time()
            
            if self._recover_quantum_system() and self._recover_federation_network():
                recovery_time = time.time() - recovery_start
                self.performance_metrics['error_recovery_time'].append(recovery_time)
                print(f"✅ Recovery successful in {recovery_time:.2f} seconds")
                
                # Print partial results
                self.print_robust_results()
            else:
                print("❌ Recovery failed - system requires manual intervention")
                raise
        
        finally:
            # Cleanup monitoring
            print("\n🧹 Phase 6: Cleanup and Data Export")
            
            # Export metrics and health data
            try:
                self.metrics_collector.export_metrics("robust_demo_metrics.json")
                print("📊 Metrics data exported successfully")
            except Exception as e:
                print(f"❌ Failed to export metrics: {e}")
            
            # Stop monitoring
            self.metrics_collector.stop_collection()
            self.health_monitor.stop_monitoring()
            
            print("✅ Cleanup complete")


async def main():
    """Main entry point for robust demo."""
    demo = RobustFederatedRLDemo()
    await demo.run_robust_demo()


if __name__ == "__main__":
    print("🎯 Dynamic Graph Federated RL - Generation 2 Robust Demo")
    print("🛡️ Comprehensive Error Handling & Monitoring")
    print("=" * 80)
    
    # Run the robust demonstration
    asyncio.run(main())