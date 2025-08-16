#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Enhanced federated RL with comprehensive error handling,
validation, logging, monitoring, health checks, and security measures.
"""

import random
import json
import time
import math
import hashlib
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/repo/federated_rl.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SecurityManager:
    """Security measures for federated learning."""
    
    @staticmethod
    def hash_parameters(parameters: List[float]) -> str:
        """Create secure hash of parameters."""
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()
    
    @staticmethod
    def validate_parameter_bounds(parameters: List[float], max_norm: float = 10.0) -> bool:
        """Validate parameter bounds to prevent adversarial attacks."""
        norm = math.sqrt(sum(x*x for x in parameters))
        return norm <= max_norm
    
    @staticmethod
    def sanitize_input(value: Union[float, int], min_val: float = -100.0, max_val: float = 100.0) -> float:
        """Sanitize numerical input."""
        try:
            val = float(value)
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            logger.warning(f"Invalid input value: {value}, using 0.0")
            return 0.0


class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
        self.check_interval = 10  # seconds
        self.last_check = time.time()
    
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        try:
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append({
                'value': SecurityManager.sanitize_input(value),
                'timestamp': time.time()
            })
            
            # Keep only recent metrics (last 100 values)
            if len(self.metrics[name]) > 100:
                self.metrics[name] = self.metrics[name][-100:]
                
        except Exception as e:
            logger.error(f"Error recording metric {name}: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        health = {
            'status': 'healthy',
            'uptime_seconds': uptime,
            'last_check': self.last_check,
            'metrics_summary': {}
        }
        
        try:
            for metric_name, values in self.metrics.items():
                if values:
                    recent_values = [v['value'] for v in values[-10:]]  # Last 10 values
                    health['metrics_summary'][metric_name] = {
                        'mean': sum(recent_values) / len(recent_values),
                        'count': len(values),
                        'latest': values[-1]['value']
                    }
            
            # Health check logic
            if uptime < 5:  # System just started
                health['status'] = 'starting'
            elif any(abs(summary.get('latest', 0)) > 1000 for summary in health['metrics_summary'].values()):
                health['status'] = 'warning'
                
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            health['status'] = 'error'
        
        self.last_check = current_time
        return health


@dataclass
class RobustGraphState:
    """Enhanced graph state with validation."""
    node_features: List[List[float]]
    edges: List[Tuple[int, int]]
    timestamp: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate state after creation."""
        if self.metadata is None:
            self.metadata = {}
        
        # Validate node features
        if not self.node_features:
            raise ValueError("Node features cannot be empty")
        
        feature_dim = len(self.node_features[0]) if self.node_features else 0
        for i, features in enumerate(self.node_features):
            if len(features) != feature_dim:
                raise ValueError(f"Inconsistent feature dimensions at node {i}")
        
        # Validate edges
        max_node_id = len(self.node_features) - 1
        for edge in self.edges:
            if len(edge) != 2:
                raise ValueError(f"Invalid edge format: {edge}")
            if not (0 <= edge[0] <= max_node_id and 0 <= edge[1] <= max_node_id):
                raise ValueError(f"Edge references non-existent node: {edge}")
    
    @property
    def num_nodes(self) -> int:
        return len(self.node_features)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)
    
    def get_node_feature_sum(self, node_id: int) -> float:
        """Get sum of features for a node with bounds checking."""
        if not (0 <= node_id < len(self.node_features)):
            logger.warning(f"Invalid node ID: {node_id}")
            return 0.0
        return sum(self.node_features[node_id])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class RobustFederatedAgent:
    """Enhanced federated agent with error handling and validation."""
    
    def __init__(self, agent_id: int, learning_rate: float = 0.01, max_param_norm: float = 10.0):
        self.agent_id = int(SecurityManager.sanitize_input(agent_id, 0, 1000))
        self.learning_rate = SecurityManager.sanitize_input(learning_rate, 0.0001, 1.0)
        self.max_param_norm = max_param_norm
        
        # Initialize parameters with validation
        self.parameters = [random.gauss(0, 0.1) for _ in range(10)]
        self._validate_parameters()
        
        # Tracking and statistics
        self.local_rewards = []
        self.training_steps = 0
        self.parameter_history = []
        self.error_count = 0
        self.last_update_time = time.time()
        
        # Health monitoring
        self.health_monitor = HealthMonitor()
        
        logger.info(f"Initialized agent {self.agent_id} with learning rate {self.learning_rate}")
    
    def _validate_parameters(self) -> bool:
        """Validate parameter vector."""
        try:
            if not SecurityManager.validate_parameter_bounds(self.parameters, self.max_param_norm):
                logger.warning(f"Agent {self.agent_id}: Parameter norm exceeds bounds, clipping")
                # Clip parameters
                norm = math.sqrt(sum(x*x for x in self.parameters))
                if norm > 0:
                    scale = self.max_param_norm / norm
                    self.parameters = [x * scale for x in self.parameters]
                return False
            return True
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error validating parameters: {e}")
            self.error_count += 1
            return False
    
    @contextmanager
    def error_handler(self, operation: str):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            self.error_count += 1
            logger.error(f"Agent {self.agent_id}: Error in {operation}: {e}")
            # Record error metric
            self.health_monitor.record_metric('errors', 1)
    
    def select_action(self, state: RobustGraphState) -> float:
        """Select action with error handling and validation."""
        with self.error_handler("select_action"):
            # Use agent's node if available, otherwise use global state
            if self.agent_id < state.num_nodes:
                node_sum = state.get_node_feature_sum(self.agent_id)
            else:
                node_sum = sum(sum(features) for features in state.node_features)
            
            # Sanitize input
            node_sum = SecurityManager.sanitize_input(node_sum)
            
            # Simple linear policy with tanh activation
            action = math.tanh(node_sum * self.parameters[0])
            action = SecurityManager.sanitize_input(action, -1.0, 1.0)
            
            # Record action metric
            self.health_monitor.record_metric('action', action)
            
            return action
        
        # Fallback action on error
        return 0.0
    
    def compute_reward(self, state: RobustGraphState, action: float) -> float:
        """Compute reward with validation."""
        with self.error_handler("compute_reward"):
            # Validate inputs
            action = SecurityManager.sanitize_input(action)
            
            # Reward based on graph connectivity and action
            connectivity = state.num_edges / max(state.num_nodes, 1)
            action_penalty = 0.1 * (action ** 2)
            reward = connectivity - action_penalty
            
            # Sanitize output
            reward = SecurityManager.sanitize_input(reward)
            
            # Record reward metric
            self.health_monitor.record_metric('reward', reward)
            
            return reward
        
        # Fallback reward on error
        return 0.0
    
    def local_update(self, state: RobustGraphState, action: float, reward: float):
        """Enhanced local parameter update with validation."""
        with self.error_handler("local_update"):
            # Validate inputs
            action = SecurityManager.sanitize_input(action)
            reward = SecurityManager.sanitize_input(reward)
            
            # Store parameter state before update
            old_params = self.parameters.copy()
            
            # Gradient ascent approximation with clipping
            for i in range(len(self.parameters)):
                gradient = reward * random.gauss(0, 0.1)
                gradient = SecurityManager.sanitize_input(gradient, -1.0, 1.0)
                self.parameters[i] += self.learning_rate * gradient
            
            # Validate updated parameters
            if not self._validate_parameters():
                logger.warning(f"Agent {self.agent_id}: Parameter update failed validation, reverting")
                self.parameters = old_params
            else:
                # Store parameter history
                param_hash = SecurityManager.hash_parameters(self.parameters)
                self.parameter_history.append({
                    'timestamp': time.time(),
                    'hash': param_hash,
                    'reward': reward
                })
                
                # Keep only recent history
                if len(self.parameter_history) > 50:
                    self.parameter_history = self.parameter_history[-50:]
            
            self.local_rewards.append(reward)
            self.training_steps += 1
            self.last_update_time = time.time()
            
            # Record update metrics
            self.health_monitor.record_metric('training_steps', self.training_steps)
            self.health_monitor.record_metric('parameter_norm', math.sqrt(sum(x*x for x in self.parameters)))
    
    def get_parameters(self) -> List[float]:
        """Get current parameters with validation."""
        if self._validate_parameters():
            return self.parameters.copy()
        else:
            logger.warning(f"Agent {self.agent_id}: Parameters invalid, returning safe defaults")
            return [0.0] * 10
    
    def set_parameters(self, parameters: List[float]):
        """Set parameters with validation."""
        with self.error_handler("set_parameters"):
            # Validate input parameters
            if len(parameters) != len(self.parameters):
                raise ValueError(f"Parameter dimension mismatch: {len(parameters)} vs {len(self.parameters)}")
            
            # Sanitize parameters
            sanitized_params = [SecurityManager.sanitize_input(p) for p in parameters]
            
            if SecurityManager.validate_parameter_bounds(sanitized_params, self.max_param_norm):
                self.parameters = sanitized_params
                logger.debug(f"Agent {self.agent_id}: Parameters updated successfully")
            else:
                logger.warning(f"Agent {self.agent_id}: Received invalid parameters, ignoring update")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        current_time = time.time()
        time_since_update = current_time - self.last_update_time
        
        recent_rewards = self.local_rewards[-10:] if self.local_rewards else [0]
        
        return {
            'agent_id': self.agent_id,
            'training_steps': self.training_steps,
            'error_count': self.error_count,
            'time_since_last_update': time_since_update,
            'parameter_norm': math.sqrt(sum(x*x for x in self.parameters)),
            'parameter_hash': SecurityManager.hash_parameters(self.parameters),
            'avg_recent_reward': sum(recent_rewards) / len(recent_rewards),
            'total_rewards': len(self.local_rewards),
            'health_status': self.health_monitor.get_health_status()
        }


class RobustFederationProtocol:
    """Enhanced federation protocol with security and robustness."""
    
    def __init__(self, agents: List[RobustFederatedAgent], min_agents: int = 2):
        self.agents = agents
        self.min_agents = max(1, min_agents)
        self.global_parameters = [0.0] * 10
        self.federation_rounds = 0
        self.failed_aggregations = 0
        self.parameter_hashes = []
        self.health_monitor = HealthMonitor()
        
        logger.info(f"Initialized federation with {len(agents)} agents")
    
    def federated_averaging(self) -> bool:
        """Enhanced federated averaging with validation."""
        try:
            # Collect parameters from healthy agents
            valid_params = []
            valid_agents = []
            
            for agent in self.agents:
                agent_params = agent.get_parameters()
                if SecurityManager.validate_parameter_bounds(agent_params):
                    valid_params.append(agent_params)
                    valid_agents.append(agent.agent_id)
                else:
                    logger.warning(f"Excluding agent {agent.agent_id} from aggregation (invalid parameters)")
            
            # Check if we have enough valid agents
            if len(valid_params) < self.min_agents:
                logger.error(f"Insufficient valid agents for aggregation: {len(valid_params)} < {self.min_agents}")
                self.failed_aggregations += 1
                return False
            
            # Perform weighted averaging (equal weights for now)
            num_params = len(self.global_parameters)
            new_global_params = [0.0] * num_params
            
            for i in range(num_params):
                param_sum = sum(params[i] for params in valid_params)
                new_global_params[i] = param_sum / len(valid_params)
            
            # Validate aggregated parameters
            if not SecurityManager.validate_parameter_bounds(new_global_params):
                logger.error("Aggregated parameters failed validation")
                self.failed_aggregations += 1
                return False
            
            # Store previous state for rollback if needed
            old_global_params = self.global_parameters.copy()
            self.global_parameters = new_global_params
            
            # Distribute to agents with error handling
            successful_distributions = 0
            for agent in self.agents:
                try:
                    agent.set_parameters(self.global_parameters)
                    successful_distributions += 1
                except Exception as e:
                    logger.error(f"Failed to distribute parameters to agent {agent.agent_id}: {e}")
            
            # Check if distribution was successful enough
            if successful_distributions < self.min_agents:
                logger.error("Parameter distribution failed, rolling back")
                self.global_parameters = old_global_params
                self.failed_aggregations += 1
                return False
            
            # Record successful aggregation
            self.federation_rounds += 1
            param_hash = SecurityManager.hash_parameters(self.global_parameters)
            self.parameter_hashes.append({
                'round': self.federation_rounds,
                'hash': param_hash,
                'timestamp': time.time(),
                'participating_agents': valid_agents
            })
            
            # Record metrics
            self.health_monitor.record_metric('federation_rounds', self.federation_rounds)
            self.health_monitor.record_metric('participating_agents', len(valid_agents))
            
            logger.info(f"Federation round {self.federation_rounds} completed with {len(valid_agents)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Critical error in federated averaging: {e}")
            self.failed_aggregations += 1
            return False
    
    def get_federation_stats(self) -> Dict[str, Any]:
        """Get comprehensive federation statistics."""
        recent_rewards = []
        for agent in self.agents:
            if agent.local_rewards:
                recent_rewards.extend(agent.local_rewards[-5:])  # Last 5 rewards per agent
        
        total_errors = sum(agent.error_count for agent in self.agents)
        total_training_steps = sum(agent.training_steps for agent in self.agents)
        
        return {
            "federation_rounds": self.federation_rounds,
            "failed_aggregations": self.failed_aggregations,
            "num_agents": len(self.agents),
            "avg_reward": sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0,
            "parameter_norm": math.sqrt(sum(x*x for x in self.global_parameters)),
            "parameter_hash": SecurityManager.hash_parameters(self.global_parameters),
            "total_errors": total_errors,
            "total_training_steps": total_training_steps,
            "success_rate": (self.federation_rounds / max(1, self.federation_rounds + self.failed_aggregations)),
            "health_status": self.health_monitor.get_health_status()
        }


class RobustTrafficEnvironment:
    """Enhanced traffic environment with validation and error handling."""
    
    def __init__(self, num_intersections: int = 10):
        self.num_intersections = max(1, min(100, num_intersections))  # Bounds checking
        self.health_monitor = HealthMonitor()
        self.reset_count = 0
        self.step_count = 0
        self.reset()
        
        logger.info(f"Initialized traffic environment with {self.num_intersections} intersections")
    
    def reset(self) -> RobustGraphState:
        """Reset environment with validation."""
        try:
            # Create simple grid topology with random traffic features
            node_features = []
            for i in range(self.num_intersections):
                features = [SecurityManager.sanitize_input(random.uniform(0, 1), 0, 2) for _ in range(4)]
                node_features.append(features)
            
            # Simple edge connectivity (chain topology)
            edges = []
            for i in range(self.num_intersections - 1):
                edges.append((i, i+1))  # Chain connection
                edges.append((i+1, i))  # Bidirectional
            
            # Create state with validation
            self.current_state = RobustGraphState(
                node_features=node_features,
                edges=edges,
                timestamp=time.time(),
                metadata={'reset_count': self.reset_count}
            )
            
            self.reset_count += 1
            self.step_count = 0
            
            # Record reset metric
            self.health_monitor.record_metric('resets', self.reset_count)
            
            logger.debug(f"Environment reset #{self.reset_count}")
            return self.current_state
            
        except Exception as e:
            logger.error(f"Error in environment reset: {e}")
            # Create minimal safe state
            safe_features = [[0.1] * 4 for _ in range(min(3, self.num_intersections))]
            safe_edges = [(0, 1), (1, 0)] if len(safe_features) > 1 else []
            return RobustGraphState(
                node_features=safe_features,
                edges=safe_edges,
                timestamp=time.time(),
                metadata={'error_reset': True}
            )
    
    def step(self, actions: Dict[int, float]) -> Tuple[RobustGraphState, Dict[int, float], bool]:
        """Enhanced environment step with validation."""
        try:
            self.step_count += 1
            
            # Validate actions
            validated_actions = {}
            for agent_id, action in actions.items():
                if 0 <= agent_id < self.num_intersections:
                    validated_actions[agent_id] = SecurityManager.sanitize_input(action, -1.0, 1.0)
                else:
                    logger.warning(f"Invalid agent ID in actions: {agent_id}")
            
            # Simple dynamics: traffic flow changes based on actions
            new_features = []
            for i in range(self.num_intersections):
                new_node_features = self.current_state.node_features[i].copy()
                
                if i in validated_actions:
                    # Action affects traffic flow
                    for j in range(len(new_node_features)):
                        new_node_features[j] += 0.1 * validated_actions[i]
                
                # Add controlled noise
                for j in range(len(new_node_features)):
                    noise = random.gauss(0, 0.05)
                    new_node_features[j] += noise
                    new_node_features[j] = SecurityManager.sanitize_input(new_node_features[j], 0.0, 2.0)
                
                new_features.append(new_node_features)
            
            # Update state
            self.current_state = RobustGraphState(
                node_features=new_features,
                edges=self.current_state.edges,
                timestamp=time.time(),
                metadata={
                    'step_count': self.step_count,
                    'actions_received': len(validated_actions)
                }
            )
            
            # Compute rewards for all agents
            rewards = {}
            for agent_id in validated_actions:
                if agent_id < self.num_intersections:
                    # Reward based on local traffic efficiency (minimize congestion)
                    local_traffic = self.current_state.node_features[agent_id]
                    congestion = sum(feature ** 2 for feature in local_traffic)
                    reward = SecurityManager.sanitize_input(-congestion)  # Negative to minimize
                    rewards[agent_id] = reward
            
            # Simple termination condition
            done = self.step_count >= 1000  # Maximum episode length
            
            # Record metrics
            self.health_monitor.record_metric('steps', self.step_count)
            if rewards:
                avg_reward = sum(rewards.values()) / len(rewards)
                self.health_monitor.record_metric('avg_reward', avg_reward)
            
            return self.current_state, rewards, done
            
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            # Return safe fallback
            return self.current_state, {}, True
    
    def get_environment_health(self) -> Dict[str, Any]:
        """Get environment health status."""
        return {
            'step_count': self.step_count,
            'reset_count': self.reset_count,
            'num_intersections': self.num_intersections,
            'current_state_valid': self.current_state is not None,
            'health_monitor': self.health_monitor.get_health_status()
        }


def run_generation2_demo():
    """Run Generation 2 demo: Robust federated learning with comprehensive error handling."""
    print("🛡️ Generation 2: MAKE IT ROBUST - Enhanced Federated RL Demo")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Initialize environment with validation
        env = RobustTrafficEnvironment(num_intersections=5)
        
        # Create federated agents
        agents = [RobustFederatedAgent(agent_id=i, learning_rate=0.01) for i in range(3)]
        federation = RobustFederationProtocol(agents, min_agents=2)
        
        # Training parameters
        episodes = 25
        steps_per_episode = 60
        federation_interval = 15
        health_check_interval = 5
        
        results = []
        
        logger.info(f"Starting robust training: {len(agents)} agents, {episodes} episodes")
        print(f"Training {len(agents)} agents for {episodes} episodes with enhanced robustness...")
        
        for episode in range(episodes):
            try:
                state = env.reset()
                episode_rewards = {i: [] for i in range(len(agents))}
                episode_start_time = time.time()
                
                for step in range(steps_per_episode):
                    # Each agent takes action with error handling
                    actions = {}
                    for agent in agents:
                        try:
                            action = agent.select_action(state)
                            actions[agent.agent_id] = action
                        except Exception as e:
                            logger.error(f"Agent {agent.agent_id} action selection failed: {e}")
                            actions[agent.agent_id] = 0.0  # Safe fallback
                    
                    # Environment step with validation
                    next_state, rewards, done = env.step(actions)
                    
                    # Local learning with error handling
                    for agent in agents:
                        if agent.agent_id in rewards:
                            try:
                                agent.local_update(state, actions[agent.agent_id], rewards[agent.agent_id])
                                episode_rewards[agent.agent_id].append(rewards[agent.agent_id])
                            except Exception as e:
                                logger.error(f"Agent {agent.agent_id} local update failed: {e}")
                    
                    # Federated aggregation with error handling
                    if step % federation_interval == 0:
                        success = federation.federated_averaging()
                        if not success:
                            logger.warning(f"Federation failed at episode {episode}, step {step}")
                    
                    # Periodic health checks
                    if step % health_check_interval == 0:
                        env_health = env.get_environment_health()
                        if not env_health['current_state_valid']:
                            logger.error("Environment state invalid, resetting")
                            state = env.reset()
                        else:
                            state = next_state
                    else:
                        state = next_state
                    
                    if done:
                        logger.info(f"Episode {episode} terminated early at step {step}")
                        break
                
                # Episode statistics with error handling
                avg_rewards = {}
                for i in range(len(agents)):
                    if episode_rewards[i]:
                        avg_rewards[i] = sum(episode_rewards[i]) / len(episode_rewards[i])
                    else:
                        avg_rewards[i] = 0.0
                
                fed_stats = federation.get_federation_stats()
                episode_time = time.time() - episode_start_time
                
                # Comprehensive result tracking
                result = {
                    "episode": episode,
                    "agent_rewards": avg_rewards,
                    "federation_stats": fed_stats,
                    "environment_health": env.get_environment_health(),
                    "agent_stats": [agent.get_agent_stats() for agent in agents],
                    "episode_time": episode_time,
                    "timestamp": time.time()
                }
                results.append(result)
                
                # Progress reporting
                if episode % 5 == 0:
                    success_rate = fed_stats['success_rate']
                    total_errors = fed_stats['total_errors']
                    print(f"Episode {episode:2d}: Avg Reward = {fed_stats['avg_reward']:.3f}, "
                          f"Success Rate = {success_rate:.1%}, Errors = {total_errors}")
                
            except Exception as e:
                logger.error(f"Critical error in episode {episode}: {e}")
                # Continue with next episode
                continue
        
        total_time = time.time() - start_time
        
        # Save results with error handling
        try:
            with open('/root/repo/gen2_robust_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            logger.info("Results saved successfully")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        # Final statistics
        if results:
            final_stats = results[-1]['federation_stats']
            print("\n🛡️ Generation 2 Complete!")
            print(f"Final average reward: {final_stats['avg_reward']:.3f}")
            print(f"Total federation rounds: {final_stats['federation_rounds']}")
            print(f"Failed aggregations: {final_stats['failed_aggregations']}")
            print(f"Success rate: {final_stats['success_rate']:.1%}")
            print(f"Total errors: {final_stats['total_errors']}")
            print(f"Training time: {total_time:.2f} seconds")
            print("Results saved to gen2_robust_results.json")
            
            # Comprehensive validation
            assert len(results) > 0, "No results recorded"
            assert final_stats['federation_rounds'] > 0, "No successful federation occurred"
            assert final_stats['success_rate'] >= 0.5, f"Success rate too low: {final_stats['success_rate']}"
            print("✅ Robustness validation passed")
        else:
            print("❌ No results recorded - critical system failure")
            return None
        
        return results
        
    except Exception as e:
        logger.critical(f"Critical system failure: {e}")
        print(f"❌ System failure: {e}")
        return None


if __name__ == "__main__":
    results = run_generation2_demo()