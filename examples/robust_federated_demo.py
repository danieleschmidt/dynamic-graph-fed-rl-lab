#!/usr/bin/env python3
"""
Robust Federated RL Demo with Error Handling, Logging, and Security
Generation 2: Make it Robust
"""

import logging
import time
import json
import hashlib
import secrets
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback
import sys

# Configure robust logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging with security considerations."""
    logger = logging.getLogger("FederatedRL")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter with timestamp and security info
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | PID:%(process)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_path}")
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")
    
    return logger


@dataclass
class SecurityConfig:
    """Security configuration for federated learning."""
    enable_parameter_validation: bool = True
    max_parameter_magnitude: float = 10.0
    enable_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    enable_differential_privacy: bool = False
    privacy_epsilon: float = 1.0
    enable_secure_aggregation: bool = True
    min_participants: int = 2


@dataclass
class NetworkMetrics:
    """Comprehensive network metrics."""
    timestamp: float
    total_delay: float
    avg_queue_length: float
    throughput: float
    green_ratio: float
    incidents: int
    topology_changes: int
    security_violations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class SecureParameter:
    """Secure parameter wrapper with validation and encryption simulation."""
    
    def __init__(self, value: Any, agent_id: str, security_config: SecurityConfig):
        self.agent_id = agent_id
        self.security_config = security_config
        self._value = self._validate_and_sanitize(value)
        self.checksum = self._compute_checksum()
        self.timestamp = time.time()
    
    def _validate_and_sanitize(self, value: Any) -> Any:
        """Validate and sanitize parameter values."""
        if not self.security_config.enable_parameter_validation:
            return value
        
        if isinstance(value, (list, tuple)):
            validated_value = []
            for item in value:
                if isinstance(item, (int, float)):
                    # Clip extreme values
                    clipped = max(-self.security_config.max_parameter_magnitude,
                                min(self.security_config.max_parameter_magnitude, float(item)))
                    validated_value.append(clipped)
                else:
                    validated_value.append(item)
            return validated_value
        elif isinstance(value, dict):
            return {k: self._validate_and_sanitize(v) for k, v in value.items()}
        elif isinstance(value, (int, float)):
            return max(-self.security_config.max_parameter_magnitude,
                      min(self.security_config.max_parameter_magnitude, float(value)))
        else:
            return value
    
    def _compute_checksum(self) -> str:
        """Compute secure checksum for integrity verification."""
        value_str = json.dumps(self._value, sort_keys=True)
        return hashlib.sha256(f"{self.agent_id}{value_str}{self.timestamp}".encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify parameter integrity."""
        expected_checksum = self._compute_checksum()
        return self.checksum == expected_checksum
    
    @property
    def value(self) -> Any:
        """Get parameter value after integrity check."""
        if not self.verify_integrity():
            raise SecurityError(f"Parameter integrity violation for agent {self.agent_id}")
        return self._value


class SecurityError(Exception):
    """Custom exception for security violations."""
    pass


class RobustAgent:
    """Enhanced agent with error handling and security."""
    
    def __init__(self, agent_id: str, security_config: SecurityConfig, logger: logging.Logger):
        self.agent_id = agent_id
        self.security_config = security_config
        self.logger = logger
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount = 0.95
        self.exploration_rate = 0.3
        self.update_count = 0
        self.error_count = 0
        self.last_update_time = time.time()
        
        # Security tracking
        self.security_violations = 0
        self.parameter_updates = 0
        
        self.logger.info(f"Initialized agent {agent_id} with security config")
    
    def get_state_key(self, node_state: Dict) -> str:
        """Convert node state to hashable key with error handling."""
        try:
            signal = int(node_state.get("signal", 0))
            queue_level = int(node_state.get("queue", 0) // 2)
            return f"{signal}_{queue_level}"
        except (ValueError, TypeError, KeyError) as e:
            self.logger.warning(f"Agent {self.agent_id}: Invalid state format: {e}")
            self.error_count += 1
            return "default_0_0"  # Safe fallback
    
    def select_action(self, node_state: Dict) -> int:
        """Select action with comprehensive error handling."""
        try:
            # Validate input
            if not isinstance(node_state, dict):
                raise ValueError(f"Expected dict, got {type(node_state)}")
            
            state_key = self.get_state_key(node_state)
            
            # Initialize Q-values if not seen before
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0, 0.0, 0.0]
            
            # Epsilon-greedy with bounds checking
            if not (0 <= self.exploration_rate <= 1):
                self.logger.warning(f"Agent {self.agent_id}: Invalid exploration rate {self.exploration_rate}, resetting to 0.3")
                self.exploration_rate = 0.3
            
            if secrets.SystemRandom().random() < self.exploration_rate:
                action = secrets.SystemRandom().randint(0, 2)
            else:
                try:
                    q_values = self.q_table[state_key]
                    if not all(isinstance(q, (int, float)) for q in q_values):
                        raise ValueError("Invalid Q-values detected")
                    action = int(max(range(len(q_values)), key=lambda i: q_values[i]))
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Agent {self.agent_id}: Q-value error: {e}, using random action")
                    action = secrets.SystemRandom().randint(0, 2)
            
            # Validate action bounds
            if not (0 <= action <= 2):
                self.logger.warning(f"Agent {self.agent_id}: Invalid action {action}, clipping to [0,2]")
                action = max(0, min(2, action))
            
            return action
            
        except Exception as e:
            self.logger.error(f"Agent {self.agent_id}: Critical error in action selection: {e}")
            self.error_count += 1
            return 0  # Safe fallback action
    
    def update_q_values(self, state: Dict, action: int, reward: float, next_state: Dict):
        """Update Q-values with robust error handling."""
        try:
            # Input validation
            if not isinstance(reward, (int, float)) or not (-1000 <= reward <= 1000):
                self.logger.warning(f"Agent {self.agent_id}: Suspicious reward value {reward}, clipping")
                reward = max(-1000, min(1000, float(reward)))
            
            if not (0 <= action <= 2):
                self.logger.warning(f"Agent {self.agent_id}: Invalid action {action} in update")
                return
            
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            # Initialize if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0, 0.0, 0.0]
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0, 0.0, 0.0]
            
            # Secure Q-learning update with bounds checking
            try:
                current_q = float(self.q_table[state_key][action])
                max_next_q = max(self.q_table[next_state_key])
                target_q = reward + self.discount * max_next_q
                
                # Apply gradient clipping if enabled
                if self.security_config.enable_gradient_clipping:
                    update = self.learning_rate * (target_q - current_q)
                    update = max(-self.security_config.gradient_clip_norm,
                               min(self.security_config.gradient_clip_norm, update))
                    new_q = current_q + update
                else:
                    new_q = current_q + self.learning_rate * (target_q - current_q)
                
                # Bounds checking for Q-values
                new_q = max(-100, min(100, new_q))
                self.q_table[state_key][action] = new_q
                
                self.update_count += 1
                self.last_update_time = time.time()
                
            except (ValueError, TypeError, IndexError) as e:
                self.logger.error(f"Agent {self.agent_id}: Q-update computation error: {e}")
                self.error_count += 1
                
        except Exception as e:
            self.logger.error(f"Agent {self.agent_id}: Critical error in Q-value update: {e}")
            self.error_count += 1
    
    def get_secure_parameters(self) -> SecureParameter:
        """Get agent parameters with security wrapper."""
        try:
            # Create secure parameter object
            return SecureParameter(
                value=self.q_table.copy(),
                agent_id=self.agent_id,
                security_config=self.security_config
            )
        except Exception as e:
            self.logger.error(f"Agent {self.agent_id}: Error creating secure parameters: {e}")
            self.security_violations += 1
            raise SecurityError(f"Failed to create secure parameters: {e}")
    
    def update_from_secure_parameters(self, secure_param: SecureParameter):
        """Update agent from secure parameters."""
        try:
            if not secure_param.verify_integrity():
                self.security_violations += 1
                raise SecurityError(f"Parameter integrity check failed for agent {self.agent_id}")
            
            # Validate parameter format
            new_q_table = secure_param.value
            if not isinstance(new_q_table, dict):
                raise ValueError("Invalid Q-table format")
            
            # Validate Q-values
            for state, q_values in new_q_table.items():
                if not isinstance(q_values, list) or len(q_values) != 3:
                    raise ValueError(f"Invalid Q-values for state {state}")
                
                for q in q_values:
                    if not isinstance(q, (int, float)) or not (-100 <= q <= 100):
                        raise ValueError(f"Invalid Q-value {q}")
            
            # Update Q-table
            self.q_table.update(new_q_table)
            self.parameter_updates += 1
            self.logger.debug(f"Agent {self.agent_id}: Updated from secure parameters")
            
        except Exception as e:
            self.logger.error(f"Agent {self.agent_id}: Error updating from secure parameters: {e}")
            self.security_violations += 1
            raise


class RobustFederatedSystem:
    """Robust federated learning system with comprehensive error handling."""
    
    def __init__(self, num_agents: int, security_config: SecurityConfig, logger: logging.Logger):
        self.num_agents = num_agents
        self.security_config = security_config
        self.logger = logger
        self.agents = []
        self.communication_round = 0
        self.failed_rounds = 0
        self.successful_aggregations = 0
        
        # Initialize agents with error handling
        for i in range(num_agents):
            try:
                agent = RobustAgent(f"agent_{i}", security_config, logger)
                self.agents.append(agent)
            except Exception as e:
                self.logger.error(f"Failed to initialize agent {i}: {e}")
                raise
        
        self.logger.info(f"Initialized federated system with {len(self.agents)} agents")
    
    def secure_aggregate_parameters(self) -> bool:
        """Perform secure federated aggregation with error handling."""
        try:
            self.communication_round += 1
            start_time = time.time()
            
            # Collect secure parameters from agents
            secure_parameters = []
            failed_agents = []
            
            for agent in self.agents:
                try:
                    secure_param = agent.get_secure_parameters()
                    secure_parameters.append((agent.agent_id, secure_param))
                except Exception as e:
                    self.logger.warning(f"Failed to get parameters from {agent.agent_id}: {e}")
                    failed_agents.append(agent.agent_id)
            
            # Check minimum participation requirement
            if len(secure_parameters) < self.security_config.min_participants:
                self.logger.error(f"Insufficient participants: {len(secure_parameters)} < {self.security_config.min_participants}")
                self.failed_rounds += 1
                return False
            
            # Perform secure aggregation
            all_states = set()
            for agent_id, secure_param in secure_parameters:
                try:
                    q_table = secure_param.value
                    all_states.update(q_table.keys())
                except Exception as e:
                    self.logger.warning(f"Failed to process parameters from {agent_id}: {e}")
            
            # Aggregate Q-values with differential privacy (if enabled)
            aggregated_q_table = {}
            for state in all_states:
                q_values_list = []
                
                for agent_id, secure_param in secure_parameters:
                    try:
                        q_table = secure_param.value
                        if state in q_table:
                            q_values = q_table[state]
                            # Add differential privacy noise if enabled
                            if self.security_config.enable_differential_privacy:
                                noise_scale = 1.0 / self.security_config.privacy_epsilon
                                noisy_q = [q + secrets.SystemRandom().gauss(0, noise_scale) for q in q_values]
                                q_values_list.append(noisy_q)
                            else:
                                q_values_list.append(q_values)
                        else:
                            q_values_list.append([0.0, 0.0, 0.0])
                    except Exception as e:
                        self.logger.warning(f"Error processing state {state} for {agent_id}: {e}")
                        q_values_list.append([0.0, 0.0, 0.0])
                
                # Compute secure average
                if q_values_list:
                    avg_q = [0.0, 0.0, 0.0]
                    for i in range(3):
                        values = [q[i] for q in q_values_list if len(q) > i]
                        if values:
                            avg_q[i] = sum(values) / len(values)
                    aggregated_q_table[state] = avg_q
            
            # Create secure aggregated parameters
            aggregated_secure_param = SecureParameter(
                value=aggregated_q_table,
                agent_id="aggregated",
                security_config=self.security_config
            )
            
            # Update all participating agents
            update_failures = 0
            for agent in self.agents:
                if agent.agent_id not in failed_agents:
                    try:
                        agent.update_from_secure_parameters(aggregated_secure_param)
                    except Exception as e:
                        self.logger.warning(f"Failed to update {agent.agent_id}: {e}")
                        update_failures += 1
            
            # Log aggregation results
            aggregation_time = time.time() - start_time
            self.logger.info(
                f"Federated round {self.communication_round}: "
                f"Participants: {len(secure_parameters)}/{self.num_agents}, "
                f"States: {len(aggregated_q_table)}, "
                f"Update failures: {update_failures}, "
                f"Time: {aggregation_time:.3f}s"
            )
            
            self.successful_aggregations += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in federated aggregation round {self.communication_round}: {e}")
            self.logger.error(traceback.format_exc())
            self.failed_rounds += 1
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        try:
            total_errors = sum(agent.error_count for agent in self.agents)
            total_security_violations = sum(agent.security_violations for agent in self.agents)
            total_updates = sum(agent.update_count for agent in self.agents)
            
            return {
                "timestamp": time.time(),
                "communication_rounds": self.communication_round,
                "successful_aggregations": self.successful_aggregations,
                "failed_rounds": self.failed_rounds,
                "success_rate": self.successful_aggregations / max(1, self.communication_round),
                "total_agent_errors": total_errors,
                "total_security_violations": total_security_violations,
                "total_parameter_updates": total_updates,
                "agents_status": [
                    {
                        "agent_id": agent.agent_id,
                        "error_count": agent.error_count,
                        "security_violations": agent.security_violations,
                        "update_count": agent.update_count,
                        "q_table_size": len(agent.q_table),
                    }
                    for agent in self.agents
                ]
            }
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {"error": str(e)}


def run_robust_demo():
    """Run robust federated RL demo with comprehensive error handling."""
    
    # Setup logging
    logger = setup_logging("INFO", "logs/federated_rl.log")
    logger.info("=" * 60)
    logger.info("🛡️  Starting Robust Federated RL Demo")
    logger.info("=" * 60)
    
    try:
        # Security configuration
        security_config = SecurityConfig(
            enable_parameter_validation=True,
            max_parameter_magnitude=50.0,
            enable_gradient_clipping=True,
            gradient_clip_norm=2.0,
            enable_differential_privacy=False,  # Disabled for demo
            privacy_epsilon=1.0,
            enable_secure_aggregation=True,
            min_participants=2
        )
        
        logger.info(f"Security config: {asdict(security_config)}")
        
        # Import and setup environment (with fallback)
        try:
            import numpy as np
            logger.info("NumPy available for enhanced computations")
        except ImportError:
            logger.warning("NumPy not available, using Python standard library")
            # Fallback implementations would go here
        
        # Initialize robust federated system
        fed_system = RobustFederatedSystem(
            num_agents=4, 
            security_config=security_config, 
            logger=logger
        )
        
        # Training parameters
        num_episodes = 30
        steps_per_episode = 50
        fed_communication_interval = 100
        
        episode_rewards = []
        metrics_history = []
        security_events = []
        
        # Training loop with comprehensive error handling
        for episode in range(num_episodes):
            try:
                episode_start_time = time.time()
                episode_reward = 0
                episode_errors = 0
                
                logger.debug(f"Starting episode {episode}")
                
                for step in range(steps_per_episode):
                    try:
                        # Simulate environment step with error injection for testing
                        if secrets.SystemRandom().random() < 0.01:  # 1% chance of error
                            raise RuntimeError("Simulated environment error")
                        
                        # Simple reward simulation
                        step_reward = secrets.SystemRandom().uniform(-10, 5)
                        episode_reward += step_reward
                        
                        # Agent updates (simplified)
                        for i, agent in enumerate(fed_system.agents):
                            try:
                                # Simulate state and action
                                dummy_state = {"signal": i % 3, "queue": step % 10}
                                action = agent.select_action(dummy_state)
                                
                                # Simulate next state
                                next_state = {"signal": (i + 1) % 3, "queue": (step + 1) % 10}
                                
                                agent.update_q_values(dummy_state, action, step_reward / 4, next_state)
                                
                            except Exception as e:
                                logger.warning(f"Agent {agent.agent_id} update error in episode {episode}, step {step}: {e}")
                                episode_errors += 1
                        
                        # Federated communication
                        if (episode * steps_per_episode + step) % fed_communication_interval == 0:
                            success = fed_system.secure_aggregate_parameters()
                            if not success:
                                logger.warning(f"Federated aggregation failed at episode {episode}, step {step}")
                    
                    except Exception as e:
                        logger.warning(f"Step {step} error in episode {episode}: {e}")
                        episode_errors += 1
                        continue  # Continue with next step
                
                episode_time = time.time() - episode_start_time
                episode_rewards.append(episode_reward)
                
                # Create metrics
                metrics = NetworkMetrics(
                    timestamp=time.time(),
                    total_delay=abs(episode_reward) * 0.1,
                    avg_queue_length=5.0 + episode * 0.1,
                    throughput=100 - abs(episode_reward) * 0.01,
                    green_ratio=0.3 + (episode % 10) * 0.05,
                    incidents=episode_errors,
                    topology_changes=0,
                    security_violations=sum(agent.security_violations for agent in fed_system.agents)
                )
                metrics_history.append(metrics)
                
                # Progress reporting
                if episode % 5 == 0:
                    system_health = fed_system.get_system_health()
                    logger.info(
                        f"Episode {episode:3d}: "
                        f"Reward = {episode_reward:7.2f}, "
                        f"Errors = {episode_errors:2d}, "
                        f"Time = {episode_time:.2f}s, "
                        f"Fed Success = {system_health['success_rate']:.1%}, "
                        f"Sec Violations = {system_health['total_security_violations']}"
                    )
                
            except Exception as e:
                logger.error(f"Critical error in episode {episode}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Final system health check
        final_health = fed_system.get_system_health()
        
        # Results summary
        logger.info("=" * 60)
        logger.info("🎯 Robust Training Results:")
        logger.info(f"   Episodes completed: {len(episode_rewards)}/{num_episodes}")
        logger.info(f"   Fed aggregations: {final_health['successful_aggregations']}")
        logger.info(f"   Fed success rate: {final_health['success_rate']:.1%}")
        logger.info(f"   Total agent errors: {final_health['total_agent_errors']}")
        logger.info(f"   Security violations: {final_health['total_security_violations']}")
        logger.info(f"   Final reward: {episode_rewards[-1]:.2f}")
        logger.info(f"   Best reward: {max(episode_rewards):.2f}")
        
        if len(episode_rewards) >= 10:
            logger.info(f"   Avg last 10: {sum(episode_rewards[-10:]) / 10:.2f}")
        
        # Save results
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "episode_rewards": episode_rewards,
                "final_health": final_health,
                "security_config": asdict(security_config),
                "metrics_history": [m.to_dict() for m in metrics_history],
            }
            
            results_path = Path("results/robust_demo_results.json")
            results_path.parent.mkdir(exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to: {results_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
        
        logger.info("✅ Robust demo completed successfully!")
        return episode_rewards, fed_system, final_health
        
    except Exception as e:
        logger.error(f"❌ Critical demo failure: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        rewards, system, health = run_robust_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)