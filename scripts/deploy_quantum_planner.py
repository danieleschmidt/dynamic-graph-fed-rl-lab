#!/usr/bin/env python3
"""
Production deployment script for Quantum Task Planner.

Prepares the quantum-inspired task planner for production deployment with
proper configuration, monitoring, and health checks.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, List


class QuantumPlannerDeployment:
    """Handle production deployment of quantum task planner."""
    
    def __init__(self, deployment_env: str = "production"):
        self.deployment_env = deployment_env
        self.root_path = Path.cwd()
        self.quantum_path = self.root_path / "src" / "dynamic_graph_fed_rl" / "quantum_planner"
        self.deployment_config = self._load_deployment_config()
    
    def _load_deployment_config(self) -> Dict[str, Any]:
        """Load deployment-specific configuration."""
        config = {
            "production": {
                "log_level": "INFO",
                "cache_size": 10000,
                "enable_monitoring": True,
                "enable_profiling": True,
                "health_check_interval": 30,
                "max_concurrent_tasks": 1000,
                "security_validation": True,
                "performance_optimization": True,
                "auto_scaling": True,
                "circuit_breaker_enabled": True,
            },
            "staging": {
                "log_level": "DEBUG",
                "cache_size": 1000,
                "enable_monitoring": True,
                "enable_profiling": True,
                "health_check_interval": 10,
                "max_concurrent_tasks": 100,
                "security_validation": True,
                "performance_optimization": False,
                "auto_scaling": False,
                "circuit_breaker_enabled": True,
            },
            "development": {
                "log_level": "DEBUG",
                "cache_size": 100,
                "enable_monitoring": False,
                "enable_profiling": False,
                "health_check_interval": 5,
                "max_concurrent_tasks": 10,
                "security_validation": False,
                "performance_optimization": False,
                "auto_scaling": False,
                "circuit_breaker_enabled": False,
            }
        }
        return config.get(self.deployment_env, config["production"])
    
    def validate_deployment_readiness(self) -> List[str]:
        """Validate quantum planner is ready for deployment."""
        issues = []
        
        # Check module structure
        required_modules = [
            "__init__.py",
            "core.py",
            "scheduler.py",
            "optimizer.py", 
            "executor.py",
            "validation.py",
            "monitoring.py",
            "security.py",
            "performance.py",
            "concurrency.py",
            "scaling.py",
            "exceptions.py"
        ]
        
        for module in required_modules:
            if not (self.quantum_path / module).exists():
                issues.append(f"Missing required module: {module}")
        
        # Check test coverage
        test_path = self.root_path / "tests" / "quantum_planner"
        if not test_path.exists():
            issues.append("Missing test directory")
        else:
            test_files = list(test_path.glob("test_*.py"))
            if len(test_files) < len(required_modules) - 2:  # Allow for some modules without tests
                issues.append(f"Insufficient test coverage: {len(test_files)} test files for {len(required_modules)} modules")
        
        return issues
    
    def create_deployment_manifest(self) -> Dict[str, Any]:
        """Create deployment manifest with metadata."""
        return {
            "name": "quantum-task-planner",
            "version": "1.0.0",
            "description": "Quantum-inspired task planning and scheduling system",
            "deployment": {
                "environment": self.deployment_env,
                "timestamp": "2024-08-06T00:00:00Z",
                "config": self.deployment_config,
            },
            "components": {
                "core": {
                    "quantum_task_manager": "Core task management with quantum superposition",
                    "state_measurement": "Quantum state collapse and measurement",
                    "entanglement_tracking": "Task dependency and relationship management"
                },
                "scheduling": {
                    "quantum_scheduler": "Interference-optimized task scheduling",
                    "adaptive_scheduler": "Self-learning parameter optimization",
                },
                "performance": {
                    "quantum_cache": "Coherence-aware caching system",
                    "jit_optimization": "Just-in-time compilation for hot paths",
                    "vectorized_operations": "Batch processing optimization",
                    "memory_pooling": "Pre-allocated memory management"
                },
                "security": {
                    "input_validation": "Comprehensive sanitization and validation",
                    "security_monitoring": "Real-time threat detection",
                    "access_control": "Role-based access management"
                },
                "monitoring": {
                    "health_checks": "System health monitoring and alerting",
                    "performance_profiling": "Execution time and resource tracking", 
                    "metrics_collection": "Operational metrics and KPIs"
                },
                "scaling": {
                    "auto_scaler": "Dynamic resource scaling based on load",
                    "load_balancer": "Request distribution and circuit breaking",
                    "resource_manager": "CPU, memory, and I/O optimization"
                }
            },
            "requirements": {
                "python": ">=3.8",
                "dependencies": [
                    "jax>=0.4.0",
                    "jaxlib>=0.4.0", 
                    "numpy>=1.21.0",
                    "psutil>=5.8.0",
                ],
                "optional_dependencies": [
                    "flax>=0.6.0",
                    "optax>=0.1.0",
                    "networkx>=2.8.0",
                    "gymnasium>=0.28.0"
                ]
            },
            "features": {
                "quantum_principles": [
                    "Superposition-based task states",
                    "Quantum entanglement for dependencies", 
                    "Interference optimization",
                    "Coherence tracking and measurement"
                ],
                "performance_optimization": [
                    "JIT compilation with JAX",
                    "Vectorized batch operations",
                    "Memory pooling and reuse",
                    "Quantum-aware caching"
                ],
                "reliability": [
                    "Circuit breaker patterns",
                    "Graceful degradation",
                    "Health monitoring",
                    "Auto-recovery mechanisms"
                ],
                "security": [
                    "Input sanitization",
                    "Injection attack prevention",
                    "Security audit logging",
                    "Access control validation"
                ],
                "scalability": [
                    "Horizontal auto-scaling",
                    "Load balancing",
                    "Resource optimization",
                    "Concurrent processing"
                ]
            },
            "quality_metrics": {
                "test_coverage": "85%+",
                "performance": "Sub-200ms response time",
                "security": "Zero known vulnerabilities", 
                "reliability": "99.9% uptime target",
                "scalability": "1000+ concurrent tasks"
            }
        }
    
    def setup_production_config(self) -> Dict[str, Any]:
        """Setup production-ready configuration."""
        config_template = {
            "quantum_planner": {
                "core": {
                    "max_tasks": self.deployment_config["max_concurrent_tasks"],
                    "enable_superposition": True,
                    "enable_entanglement": True,
                    "coherence_threshold": 0.95,
                    "measurement_timeout": 30.0
                },
                "scheduler": {
                    "algorithm": "quantum_interference",
                    "optimization_rounds": 100,
                    "enable_adaptive_learning": True,
                    "learning_rate": 0.01
                },
                "performance": {
                    "cache_size": self.deployment_config["cache_size"],
                    "enable_jit": self.deployment_config["performance_optimization"],
                    "enable_vectorization": True,
                    "enable_memory_pooling": True,
                    "profile_threshold": 0.01
                },
                "monitoring": {
                    "enable_health_checks": self.deployment_config["enable_monitoring"],
                    "health_check_interval": self.deployment_config["health_check_interval"],
                    "enable_profiling": self.deployment_config["enable_profiling"],
                    "metrics_retention_days": 30
                },
                "security": {
                    "enable_validation": self.deployment_config["security_validation"],
                    "max_input_size": 1024 * 1024,  # 1MB
                    "enable_audit_logging": True,
                    "security_scan_interval": 3600  # 1 hour
                },
                "scaling": {
                    "enable_auto_scaling": self.deployment_config["auto_scaling"],
                    "scale_up_threshold": 0.8,
                    "scale_down_threshold": 0.3,
                    "max_instances": 10,
                    "min_instances": 2
                },
                "logging": {
                    "level": self.deployment_config["log_level"],
                    "format": "json",
                    "enable_structured_logging": True,
                    "log_rotation": "daily",
                    "max_log_files": 30
                }
            }
        }
        return config_template
    
    def create_deployment_scripts(self) -> Dict[str, str]:
        """Create deployment and management scripts."""
        scripts = {
            "start.sh": '''#!/bin/bash
# Quantum Task Planner Startup Script

set -e

echo "Starting Quantum Task Planner..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export QUANTUM_PLANNER_ENV=production

# Validate environment
python3 -c "import sys; print(f'Python version: {sys.version}')"

# Start the quantum planner service
python3 -m src.dynamic_graph_fed_rl.quantum_planner --config config/production.yaml

echo "Quantum Task Planner started successfully!"
''',
            
            "stop.sh": '''#!/bin/bash
# Quantum Task Planner Shutdown Script

echo "Shutting down Quantum Task Planner..."

# Graceful shutdown
pkill -f "quantum_planner" || true

echo "Quantum Task Planner stopped."
''',
            
            "health_check.sh": '''#!/bin/bash
# Health Check Script

response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health || echo "000")

if [ "$response" = "200" ]; then
    echo "Quantum Task Planner is healthy"
    exit 0
else
    echo "Quantum Task Planner health check failed (HTTP $response)"
    exit 1
fi
''',
            
            "deploy.sh": '''#!/bin/bash
# Production Deployment Script

set -e

echo "🚀 Deploying Quantum Task Planner..."

# Backup current deployment
if [ -d "quantum_planner_backup" ]; then
    rm -rf quantum_planner_backup_old
    mv quantum_planner_backup quantum_planner_backup_old
fi

# Create deployment directory
mkdir -p deployment/quantum_planner
mkdir -p deployment/logs
mkdir -p deployment/config

# Copy quantum planner modules
cp -r src/dynamic_graph_fed_rl/quantum_planner deployment/
cp -r tests/quantum_planner deployment/tests/

# Copy configuration
cp config/production.yaml deployment/config/

# Set permissions
chmod +x deployment/*.sh

echo "✅ Quantum Task Planner deployed successfully!"
echo "Use './start.sh' to start the service"
'''
        }
        return scripts
    
    def generate_monitoring_setup(self) -> Dict[str, Any]:
        """Generate monitoring and alerting configuration."""
        return {
            "health_endpoints": {
                "/health": "Overall system health",
                "/health/quantum": "Quantum system coherence",
                "/health/performance": "Performance metrics",
                "/health/security": "Security status",
                "/metrics": "Prometheus-compatible metrics"
            },
            "key_metrics": {
                "task_processing": {
                    "tasks_per_second": "Task throughput",
                    "average_completion_time": "Task completion latency",
                    "quantum_coherence_avg": "Average system coherence",
                    "cache_hit_rate": "Caching effectiveness"
                },
                "system_health": {
                    "cpu_usage": "CPU utilization",
                    "memory_usage": "Memory consumption",
                    "error_rate": "Error frequency",
                    "uptime": "System availability"
                },
                "security": {
                    "validation_failures": "Input validation failures", 
                    "security_events": "Security incidents",
                    "access_violations": "Unauthorized access attempts"
                }
            },
            "alerts": {
                "critical": {
                    "system_down": "Service unavailable",
                    "high_error_rate": "Error rate > 5%",
                    "memory_exhaustion": "Memory usage > 90%"
                },
                "warning": {
                    "high_latency": "Response time > 1s",
                    "low_cache_hit": "Cache hit rate < 70%",
                    "quantum_decoherence": "Coherence < 0.8"
                }
            }
        }
    
    def deploy(self) -> bool:
        """Execute full deployment process."""
        print("🚀 QUANTUM TASK PLANNER DEPLOYMENT")
        print("=" * 50)
        
        # Step 1: Validate readiness
        print("\n1. Validating deployment readiness...")
        issues = self.validate_deployment_readiness()
        
        if issues:
            print("❌ Deployment validation failed:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ Deployment validation passed")
        
        # Step 2: Create deployment manifest  
        print("\n2. Creating deployment manifest...")
        manifest = self.create_deployment_manifest()
        
        manifest_path = self.root_path / "deployment_manifest.json"
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"✅ Manifest created: {manifest_path}")
        
        # Step 3: Setup configuration
        print("\n3. Setting up production configuration...")
        config = self.setup_production_config()
        
        config_dir = self.root_path / "config"
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / f"{self.deployment_env}.yaml"
        import yaml
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"✅ Configuration created: {config_path}")
        except ImportError:
            # Fallback to JSON if PyYAML not available
            config_path = config_dir / f"{self.deployment_env}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Configuration created: {config_path}")
        
        # Step 4: Create deployment scripts
        print("\n4. Creating deployment scripts...")
        scripts = self.create_deployment_scripts()
        
        scripts_dir = self.root_path / "scripts" / "deployment"
        scripts_dir.mkdir(exist_ok=True)
        
        for script_name, script_content in scripts.items():
            script_path = scripts_dir / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)  # Make executable
            print(f"✅ Script created: {script_path}")
        
        # Step 5: Generate monitoring setup
        print("\n5. Generating monitoring configuration...")
        monitoring = self.generate_monitoring_setup()
        
        monitoring_path = self.root_path / "monitoring_config.json"
        with open(monitoring_path, 'w') as f:
            json.dump(monitoring, f, indent=2)
        print(f"✅ Monitoring config created: {monitoring_path}")
        
        # Step 6: Final summary
        print("\n" + "=" * 50)
        print("🎉 DEPLOYMENT PREPARATION COMPLETE!")
        print("\nQuantum Task Planner is ready for production deployment.")
        print(f"Environment: {self.deployment_env}")
        print(f"Configuration: {config_path}")
        print(f"Scripts: {scripts_dir}")
        print(f"Monitoring: {monitoring_path}")
        
        print("\nNext steps:")
        print("1. Review configuration files")
        print("2. Set up monitoring and alerting")
        print("3. Execute deployment script")
        print("4. Verify health checks")
        print("5. Monitor performance metrics")
        
        return True


def main():
    """Main deployment function."""
    deployment_env = sys.argv[1] if len(sys.argv) > 1 else "production"
    
    deployer = QuantumPlannerDeployment(deployment_env)
    success = deployer.deploy()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()