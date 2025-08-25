#!/usr/bin/env python3
"""
Dynamic Graph Fed-RL CLI

Command-line interface for quantum-inspired task planning and federated RL.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

from .quantum_planner import QuantumTaskPlanner


class QuantumPlannerCLI:
    """CLI for quantum task planner operations."""
    
    def __init__(self):
        self.planner = None
        
    def create_planner(self, config: Dict[str, Any]) -> QuantumTaskPlanner:
        """Create quantum planner with configuration."""
        return QuantumTaskPlanner(
            max_parallel_tasks=config.get("max_parallel_tasks", 4),
            quantum_coherence_time=config.get("quantum_coherence_time", 10.0),
            interference_strength=config.get("interference_strength", 0.1)
        )
    
    def load_tasks_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load tasks from JSON configuration file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def run_planner(self, args):
        """Run quantum task planner."""
        print("🔬 Dynamic Graph Fed-RL - Quantum Task Planner")
        print("=" * 50)
        
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Create planner
        self.planner = self.create_planner(config.get("planner", {}))
        
        # Load tasks
        if args.tasks:
            task_config = self.load_tasks_from_file(args.tasks)
            
            print(f"\n📋 Loading {len(task_config.get('tasks', []))} tasks...")
            
            for task_def in task_config.get("tasks", []):
                self.planner.add_task(
                    task_id=task_def["id"],
                    name=task_def["name"],
                    dependencies=set(task_def.get("dependencies", [])),
                    estimated_duration=task_def.get("estimated_duration", 1.0),
                    priority=task_def.get("priority", 1.0),
                    resource_requirements=task_def.get("resource_requirements", {})
                )
        
        # Generate and display superposition
        print("\n🌀 Generating quantum superposition...")
        superposition = self.planner.generate_execution_paths()
        
        if args.verbose:
            print(f"Generated {len(superposition.paths)} execution paths:")
            for i, path in enumerate(superposition.paths):
                amplitude = superposition.path_amplitudes[i]
                probability = abs(amplitude) ** 2
                print(f"  Path {i+1}: {' → '.join(path)} (P={probability:.3f})")
        
        # Execute quantum measurement
        print("\n⚡ Executing quantum measurement...")
        result = self.planner.measure_and_# SECURITY WARNING: Potential SQL injection - use parameterized queries
execute()
        
        # Display results
        print(f"\n📊 Execution Results:")
        print(f"Optimal path: {' → '.join(result['path'])}")
        print(f"Total duration: {result['total_duration']:.3f}s")
        print(f"Quantum efficiency: {result['quantum_efficiency']:.3f}")
        
        if args.verbose:
            for task_id, task_result in result['task_results'].items():
                status = task_result['status']
                duration = task_result['duration']
                emoji = "✅" if status == "success" else "❌"
                print(f"  {emoji} {task_id}: {status} ({duration:.3f}s)")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")
    
    def status(self, args):
        """Display quantum planner status."""
        if not self.planner:
            print("❌ No active planner session")
            return
        
        state = self.planner.get_system_state()
        
        print("🔬 Quantum Planner Status:")
        print(f"Tasks: {state['num_tasks']}")
        print(f"Entanglements: {state['entanglements']}")
        print(f"Superposition paths: {state['superposition_paths']}")
        print(f"Execution history: {state['execution_history_length']}")
        print(f"Coherence remaining: {state['quantum_coherence_remaining']:.1f}s")


def create_sample_config():
    """Create sample configuration files."""
    planner_config = {
        "planner": {
            "max_parallel_tasks": 4,
            "quantum_coherence_time": 10.0,
            "interference_strength": 0.1
        }
    }
    
    tasks_config = {
        "tasks": [
            {
                "id": "init",
                "name": "Initialize System",
                "estimated_duration": 1.0,
                "priority": 2.0
            },
            {
                "id": "process",
                "name": "Process Data", 
                "dependencies": ["init"],
                "estimated_duration": 2.0,
                "priority": 1.5,
                "resource_requirements": {"cpu": 0.5, "memory": 0.3}
            },
            {
                "id": "analyze",
                "name": "Analyze Results",
                "dependencies": ["process"],
                "estimated_duration": 1.5,
                "priority": 3.0,
                "resource_requirements": {"cpu": 0.8}
            }
        ]
    }
    
    with open("planner_config.json", 'w') as f:
        json.dump(planner_config, f, indent=2)
    
    with open("tasks_config.json", 'w') as f:
        json.dump(tasks_config, f, indent=2)
    
    print("📄 Created sample configuration files:")
    print("  - planner_config.json")
    print("  - tasks_config.json")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dynamic Graph Fed-RL Quantum Task Planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample configuration
  dgfrl plan --tasks tasks_config.json --config planner_config.json
  
  # Run with verbose output
  dgfrl plan --tasks tasks_config.json --verbose
  
  # Save results to file
  dgfrl plan --tasks tasks_config.json --output results.json
  
  # Create sample configuration
  dgfrl init-config
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Plan command
    plan_parser = subparsers.add_parser('plan', help='Execute quantum task planning')
    plan_parser.add_argument('--tasks', '-t', help='Tasks configuration JSON file')
    plan_parser.add_argument('--config', '-c', help='Planner configuration JSON file')
    plan_parser.add_argument('--output', '-o', help='Output file for results')
    plan_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show planner status')
    
    # Init config command
    init_parser = subparsers.add_parser('init-config', help='Create sample configuration files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = QuantumPlannerCLI()
    
    try:
        if args.command == 'plan':
            cli.run_planner(args)
        elif args.command == 'status':
            cli.status(args)
        elif args.command == 'init-config':
            create_sample_config()
        
    except KeyboardInterrupt:
        print("\n\n👋 Quantum planning interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()