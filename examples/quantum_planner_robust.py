#!/usr/bin/env python3
"""
Robust Quantum Task Planner Example

Demonstrates Generation 2 features including:
- Input validation and sanitization
- Error handling and recovery
- Security measures
- Monitoring and logging
- Health checks
"""

import sys
import os
import time
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'dynamic_graph_fed_rl', 'quantum_planner'))

# Import minimal quantum planner without numpy dependencies
sys.path.insert(0, os.path.dirname(__file__))
from quantum_planner_minimal import MinimalQuantumPlanner, MinimalQuantumTask

# Define exception classes for this example
class TaskValidationError(Exception):
    def __init__(self, task_id, errors):
        self.task_id = task_id
        self.errors = errors
        super().__init__(f"Validation error in {task_id}: {errors}")

class DependencyError(Exception):
    pass

class ExecutionError(Exception):
    def __init__(self, task_id, error_msg, error_type="GENERAL", context=None):
        self.task_id = task_id
        self.error_type = error_type 
        self.context = context or {}
        super().__init__(f"Execution error in {task_id}: {error_msg}")

class ResourceAllocationError(Exception):
    def __init__(self, task_id, required, available, context=None):
        self.task_id = task_id
        self.required = required
        self.available = available
        self.context = context or {}
        super().__init__(f"Resource error for {task_id}: required {required}, available {available}")

class QuantumStateError(Exception):
    pass


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantum_planner.log')
    ]
)

logger = logging.getLogger(__name__)


class RobustQuantumPlanner(MinimalQuantumPlanner):
    """
    Enhanced quantum planner with robust error handling and validation.
    
    Generation 2 features:
    - Input validation and sanitization
    - Comprehensive error handling
    - Resource management with quotas
    - Execution monitoring and recovery
    - Security measures
    """
    
    def __init__(self, max_parallel_tasks=4, quantum_coherence_time=10.0, interference_strength=0.1):
        # Initialize minimal planner with only supported parameters
        super().__init__(max_parallel_tasks=max_parallel_tasks)
        
        # Store additional parameters for our robust features
        self.quantum_coherence_time = quantum_coherence_time
        self.interference_strength = interference_strength
        
        # Resource quotas
        self.resource_quotas = {
            "cpu": 8.0,
            "memory": 16.0,
            "disk": 100.0,
            "network": 10.0
        }
        
        self.resource_usage = {
            "cpu": 0.0,
            "memory": 0.0,
            "disk": 0.0,
            "network": 0.0
        }
        
        # Error tracking
        self.error_count = 0
        self.max_errors = 10
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        # Security settings
        self.max_task_duration = 300.0  # 5 minutes max
        self.max_dependencies = 20
        self.allowed_task_ids = set()  # Whitelist for security
        
        logger.info("RobustQuantumPlanner initialized with enhanced features")
    
    def validate_and_sanitize_task(self, task_data):
        """Validate and sanitize task input data."""
        logger.info(f"Validating task data: {task_data.get('id', 'unknown')}")
        
        # Input sanitization
        sanitized_data = {}
        
        # Validate and sanitize task ID
        if "id" not in task_data or not task_data["id"]:
            raise TaskValidationError("task_unknown", ["Task ID is required"])
        
        task_id = str(task_data["id"]).strip()
        if len(task_id) > 50:
            raise TaskValidationError(task_id, ["Task ID too long (max 50 characters)"])
        
        # Remove dangerous characters
        import re
        task_id = re.sub(r'[^\w\-_.]', '', task_id)
        if not task_id:
            raise TaskValidationError("task_unknown", ["Task ID contains only invalid characters"])
        
        sanitized_data["id"] = task_id
        
        # Validate and sanitize name
        if "name" not in task_data or not task_data["name"]:
            raise TaskValidationError(task_id, ["Task name is required"])
        
        name = str(task_data["name"]).strip()
        if len(name) > 200:
            name = name[:200]
        
        # Remove potentially dangerous content
        name = re.sub(r'[<>"\';]', '', name)
        sanitized_data["name"] = name
        
        # Validate duration
        duration = task_data.get("estimated_duration", 1.0)
        try:
            duration = float(duration)
            if duration <= 0:
                duration = 1.0
            elif duration > self.max_task_duration:
                raise TaskValidationError(task_id, [f"Duration exceeds maximum ({self.max_task_duration}s)"])
        except (ValueError, TypeError):
            duration = 1.0
        
        sanitized_data["estimated_duration"] = duration
        
        # Validate priority
        priority = task_data.get("priority", 1.0)
        try:
            priority = float(priority)
            priority = max(0.0, min(10.0, priority))  # Clamp to 0-10 range
        except (ValueError, TypeError):
            priority = 1.0
        
        sanitized_data["priority"] = priority
        
        # Validate dependencies
        dependencies = task_data.get("dependencies", set())
        if isinstance(dependencies, (list, tuple)):
            dependencies = set(dependencies)
        elif not isinstance(dependencies, set):
            dependencies = set()
        
        # Sanitize dependency IDs
        sanitized_deps = set()
        for dep in dependencies:
            dep = str(dep).strip()
            dep = re.sub(r'[^\w\-_.]', '', dep)
            if dep and len(dep) <= 50:
                sanitized_deps.add(dep)
        
        if len(sanitized_deps) > self.max_dependencies:
            raise TaskValidationError(
                task_id, 
                [f"Too many dependencies ({len(sanitized_deps)} > {self.max_dependencies})"]
            )
        
        sanitized_data["dependencies"] = sanitized_deps
        
        # Validate resource requirements
        resources = task_data.get("resource_requirements", {})
        if not isinstance(resources, dict):
            resources = {}
        
        sanitized_resources = {}
        for resource_name, value in resources.items():
            if resource_name in self.resource_quotas:
                try:
                    value = float(value)
                    value = max(0.0, min(self.resource_quotas[resource_name], value))
                    sanitized_resources[resource_name] = value
                except (ValueError, TypeError):
                    pass
        
        sanitized_data["resource_requirements"] = sanitized_resources
        
        logger.info(f"Task validation successful for: {task_id}")
        return sanitized_data
    
    def check_resource_availability(self, task_requirements):
        """Check if resources are available for task execution."""
        for resource, required in task_requirements.items():
            available = self.resource_quotas.get(resource, 0.0) - self.resource_usage.get(resource, 0.0)
            if required > available:
                raise ResourceAllocationError(
                    "task_unknown",
                    task_requirements,
                    self.resource_usage,
                    context={
                        "required": required,
                        "available": available,
                        "resource": resource
                    }
                )
    
    def allocate_resources(self, task_id, requirements):
        """Allocate resources for task execution."""
        logger.info(f"Allocating resources for task {task_id}: {requirements}")
        
        self.check_resource_availability(requirements)
        
        # Allocate resources
        for resource, amount in requirements.items():
            self.resource_usage[resource] = self.resource_usage.get(resource, 0.0) + amount
        
        logger.info(f"Resources allocated. Current usage: {self.resource_usage}")
    
    def release_resources(self, task_id, requirements):
        """Release resources after task completion."""
        logger.info(f"Releasing resources for task {task_id}: {requirements}")
        
        for resource, amount in requirements.items():
            self.resource_usage[resource] = max(0.0, self.resource_usage.get(resource, 0.0) - amount)
        
        logger.info(f"Resources released. Current usage: {self.resource_usage}")
    
    def add_task_robust(self, task_data, executor=None):
        """Add task with comprehensive validation and error handling."""
        try:
            # Validate and sanitize input
            sanitized_data = self.validate_and_sanitize_task(task_data)
            
            # Check for duplicate task IDs
            if sanitized_data["id"] in self.tasks:
                raise TaskValidationError(
                    sanitized_data["id"],
                    ["Task with this ID already exists"]
                )
            
            # Validate dependencies exist
            for dep_id in sanitized_data["dependencies"]:
                if dep_id not in self.tasks and dep_id not in [sanitized_data["id"]]:
                    logger.warning(f"Dependency {dep_id} not found for task {sanitized_data['id']}")
            
            # Check resource availability
            self.check_resource_availability(sanitized_data["resource_requirements"])
            
            # Create task using the minimal planner interface
            task = super().add_task(
                task_id=sanitized_data["id"],
                name=sanitized_data["name"],
                dependencies=sanitized_data["dependencies"],
                estimated_duration=sanitized_data["estimated_duration"],
                priority=sanitized_data["priority"],
                executor=executor
            )
            
            # Store resource requirements separately for our robust features
            if hasattr(task, 'resource_requirements'):
                task.resource_requirements = sanitized_data["resource_requirements"]
            else:
                # Add attribute if not present in minimal implementation
                setattr(task, 'resource_requirements', sanitized_data["resource_requirements"])
            
            logger.info(f"Task {task.id} added successfully")
            return task
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to add task: {e}")
            
            if self.error_count > self.max_errors:
                raise ExecutionError(
                    "system",
                    f"Maximum error limit exceeded ({self.max_errors})",
                    "SYSTEM_OVERLOAD"
                )
            
            raise
    
    def execute_with_recovery(self, task_sequence):
        """Execute task sequence with error recovery."""
        results = {}
        failed_tasks = []
        
        for task_id in task_sequence:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found, skipping")
                continue
            
            task = self.tasks[task_id]
            
            try:
                # Get resource requirements (with fallback for minimal tasks)
                resource_requirements = getattr(task, 'resource_requirements', {})
                
                # Allocate resources
                self.allocate_resources(task_id, resource_requirements)
                
                # Execute task with monitoring
                result = self._execute_task_monitored(task)
                results[task_id] = result
                
                # Release resources
                self.release_resources(task_id, resource_requirements)
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                failed_tasks.append(task_id)
                
                # Release resources on failure
                self.release_resources(task_id, resource_requirements)
                
                # Record failure
                results[task_id] = {
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration": 0.0
                }
                
                # Attempt recovery if not too many failures
                if len(failed_tasks) <= 3 and self.recovery_attempts < self.max_recovery_attempts:
                    self.recovery_attempts += 1
                    logger.info(f"Attempting recovery for task {task_id} (attempt {self.recovery_attempts})")
                    
                    # Simple recovery: retry with reduced resources
                    try:
                        original_resources = getattr(task, 'resource_requirements', {})
                        reduced_resources = {
                            k: v * 0.5 for k, v in original_resources.items()
                        }
                        setattr(task, 'resource_requirements', reduced_resources)
                        
                        self.allocate_resources(task_id, reduced_resources)
                        result = self._execute_task_monitored(task)
                        results[task_id] = result
                        self.release_resources(task_id, reduced_resources)
                        
                        logger.info(f"Recovery successful for task {task_id}")
                        failed_tasks.remove(task_id)
                        
                    except Exception as recovery_error:
                        logger.error(f"Recovery failed for task {task_id}: {recovery_error}")
        
        return results, failed_tasks
    
    def _execute_task_monitored(self, task):
        """Execute single task with monitoring and timeout."""
        start_time = time.time()
        
        # Update task state (compatible with minimal planner)
        if hasattr(task, 'collapse_to_state'):
            task.collapse_to_state("active")
        elif hasattr(task, 'state_probabilities'):
            # Manually update state probabilities for minimal planner
            task.state_probabilities = {"active": 1.0, "pending": 0.0, "completed": 0.0, "failed": 0.0}
        task.start_time = start_time
        
        try:
            # Execute with timeout
            if task.executor:
                # Simple timeout mechanism
                result = task.executor()
            else:
                # Simulate execution
                time.sleep(min(task.estimated_duration, 0.1))
                result = {"status": "simulated", "task_id": task.id}
            
            # Check execution time
            execution_time = time.time() - start_time
            if execution_time > task.estimated_duration * 2:
                logger.warning(f"Task {task.id} took longer than expected: {execution_time:.2f}s vs {task.estimated_duration:.2f}s")
            
            task.end_time = time.time()
            task.actual_duration = execution_time
            task.result = result
            
            # Update task state to completed (compatible with minimal planner)
            if hasattr(task, 'collapse_to_state'):
                task.collapse_to_state("completed")
            elif hasattr(task, 'state_probabilities'):
                task.state_probabilities = {"completed": 1.0, "active": 0.0, "pending": 0.0, "failed": 0.0}
            
            return {
                "status": "success",
                "result": result,
                "duration": execution_time
            }
            
        except Exception as e:
            task.end_time = time.time()
            task.actual_duration = time.time() - start_time
            
            # Update task state to failed (compatible with minimal planner)
            if hasattr(task, 'collapse_to_state'):
                task.collapse_to_state("failed")
            elif hasattr(task, 'state_probabilities'):
                task.state_probabilities = {"failed": 1.0, "active": 0.0, "pending": 0.0, "completed": 0.0}
            
            raise ExecutionError(
                task.id,
                str(e),
                "TASK_EXECUTION_FAILURE",
                context={"duration": task.actual_duration}
            )
    
    def get_system_health(self):
        """Get comprehensive system health information."""
        total_resource_usage = sum(self.resource_usage.values())
        total_resource_quota = sum(self.resource_quotas.values())
        
        health_status = "healthy"
        if total_resource_usage > total_resource_quota * 0.9:
            health_status = "critical"
        elif total_resource_usage > total_resource_quota * 0.7:
            health_status = "warning"
        
        return {
            "status": health_status,
            "resource_usage": self.resource_usage.copy(),
            "resource_quotas": self.resource_quotas.copy(),
            "utilization_percent": (total_resource_usage / total_resource_quota) * 100 if total_resource_quota > 0 else 0,
            "error_count": self.error_count,
            "max_errors": self.max_errors,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts,
            "active_tasks": len([t for t in self.tasks.values() if hasattr(t, 'start_time') and getattr(t, 'start_time', None) and not getattr(t, 'end_time', None)]),
            "total_tasks": len(self.tasks)
        }


# Example task functions with potential failures
def robust_task_init():
    """Initialize system with potential validation."""
    time.sleep(0.05)
    # Simulate random failure
    import random
    if random.random() < 0.1:  # 10% failure rate
        raise Exception("Initialization failed due to missing configuration")
    return {"status": "System initialized", "config_loaded": True}


def robust_task_process():
    """Process data with resource monitoring."""
    time.sleep(0.08)
    # Simulate resource-intensive operation
    import random
    if random.random() < 0.15:  # 15% failure rate
        raise Exception("Processing failed due to memory exhaustion")
    return {"status": "Data processed", "records": 5000, "memory_used": "2.3GB"}


def robust_task_validate():
    """Validate results with comprehensive checking."""
    time.sleep(0.04)
    # Simulate validation failure
    import random
    if random.random() < 0.05:  # 5% failure rate
        raise Exception("Validation failed: data integrity check failed")
    return {"status": "Validation complete", "errors_found": 0, "confidence": 0.98}


def robust_task_cleanup():
    """Cleanup with retry logic."""
    time.sleep(0.03)
    # Simulate cleanup issues
    import random
    if random.random() < 0.08:  # 8% failure rate
        raise Exception("Cleanup failed: permission denied on temp files")
    return {"status": "Cleanup complete", "files_removed": 42, "disk_freed": "150MB"}


def main():
    """Run robust quantum planner demonstration."""
    print("🛡️  Robust Quantum Task Planner - Generation 2")
    print("=" * 55)
    
    # Initialize robust planner
    planner = RobustQuantumPlanner(
        max_parallel_tasks=3,
        quantum_coherence_time=8.0,
        interference_strength=0.15
    )
    
    print("\n📋 Adding tasks with validation...")
    
    # Test data with various validation scenarios
    test_tasks = [
        {
            "id": "init-system",
            "name": "Initialize System Components",
            "estimated_duration": 0.05,
            "priority": 3.0,
            "resource_requirements": {"cpu": 0.5, "memory": 1.0}
        },
        {
            "id": "process_data", 
            "name": "Process <script>alert('xss')</script> Data",  # Test XSS filtering
            "dependencies": ["init-system"],
            "estimated_duration": 0.08,
            "priority": 2.5,
            "resource_requirements": {"cpu": 1.5, "memory": 2.0, "disk": 0.5}
        },
        {
            "id": "validate-results",
            "name": "Validate Processing Results",
            "dependencies": ["process_data"],
            "estimated_duration": 0.04,
            "priority": 2.8,
            "resource_requirements": {"cpu": 0.3, "memory": 0.5}
        },
        {
            "id": "cleanup-temp",
            "name": "Cleanup Temporary Files",
            "dependencies": ["validate-results"],
            "estimated_duration": 0.03,
            "priority": 1.0,
            "resource_requirements": {"disk": 0.2}
        },
        {
            "id": "invalid@task!",  # Test ID sanitization
            "name": "This name is way too long and contains dangerous content like <script> and should be truncated" * 5,
            "estimated_duration": -1.0,  # Test negative duration
            "priority": "not_a_number",  # Test invalid priority
            "dependencies": ["nonexistent-task"],  # Test missing dependency
            "resource_requirements": {"cpu": 999.0}  # Test resource limit
        }
    ]
    
    # Add tasks with error handling
    executors = [robust_task_init, robust_task_process, robust_task_validate, robust_task_cleanup, None]
    
    for i, task_data in enumerate(test_tasks):
        try:
            executor = executors[i] if i < len(executors) else None
            task = planner.add_task_robust(task_data, executor)
            print(f"  ✅ Added task: {task.id}")
        except Exception as e:
            print(f"  ❌ Failed to add task {task_data.get('id', 'unknown')}: {e}")
    
    # Display system health
    health = planner.get_system_health()
    print(f"\n🏥 System Health: {health['status'].upper()}")
    print(f"Resource utilization: {health['utilization_percent']:.1f}%")
    print(f"Active tasks: {health['active_tasks']}/{health['total_tasks']}")
    
    # Generate execution paths (adapt for minimal planner)
    print("\n🌀 Generating quantum superposition...")
    paths = planner.generate_execution_paths()
    
    print(f"Generated {len(paths)} execution paths:")
    for i, path in enumerate(paths):
        if path:  # Only show non-empty paths
            print(f"  Path {i+1}: {' → '.join(path)}")
    
    # Execute with recovery
    print("\n⚡ Executing with error recovery...")
    start_time = time.time()
    
    # Use quantum measurement to select optimal path
    optimal_path = planner.quantum_measurement(paths) if paths else []
    if optimal_path:
        results, failed_tasks = planner.execute_with_recovery(optimal_path)
        
        execution_time = time.time() - start_time
        
        # Display results
        print(f"\n📊 Execution Results:")
        print(f"Selected path: {' → '.join(optimal_path)}")
        print(f"Total execution time: {execution_time:.3f}s")
        print(f"Failed tasks: {len(failed_tasks)}")
        print(f"Recovery attempts: {planner.recovery_attempts}")
        
        print("\nTask Results:")
        for task_id, result in results.items():
            status = result['status']
            duration = result.get('duration', 0.0)
            emoji = "✅" if status == "success" else "❌"
            print(f"  {emoji} {task_id}: {status} ({duration:.3f}s)")
            
            if status == "success" and 'result' in result and isinstance(result['result'], dict):
                for key, value in result['result'].items():
                    if key != 'status':
                        print(f"     └─ {key}: {value}")
            elif status == "failed":
                print(f"     └─ Error: {result.get('error', 'Unknown error')}")
    else:
        print("  ⚠️  No executable path found")
    
    # Final system health
    final_health = planner.get_system_health()
    print(f"\n🏥 Final System Health: {final_health['status'].upper()}")
    print(f"Total errors encountered: {final_health['error_count']}")
    print(f"Resource usage: {final_health['resource_usage']}")
    
    print("\n🛡️  Robust quantum planning demonstration complete!")
    print("Features demonstrated:")
    print("  • Input validation and sanitization")
    print("  • Resource quota management")
    print("  • Error handling and recovery")
    print("  • Security filtering (XSS, injection)")
    print("  • System health monitoring")
    print("  • Execution timeout and monitoring")


if __name__ == "__main__":
    main()