#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Pure Python implementation without dependencies.
This demonstrates the basic federated RL structure with minimal viable features.
"""

import random
import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple


class SimpleMath:
    """Simple math utilities without numpy."""
    
    @staticmethod
    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def norm(vector: List[float]) -> float:
        return math.sqrt(sum(x*x for x in vector))
    
    @staticmethod
    def tanh(x: float) -> float:
        return math.tanh(x)
    
    @staticmethod
    def clip(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))


class MockGraphState:
    """Simplified graph state for Generation 1."""
    
    def __init__(self, node_features: List[List[float]], edges: List[Tuple[int, int]], timestamp: float = 0.0):
        self.node_features = node_features
        self.edges = edges
        self.timestamp = timestamp
    
    @property
    def num_nodes(self) -> int:
        return len(self.node_features)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)
    
    def get_node_feature_sum(self, node_id: int) -> float:
        """Get sum of features for a node."""
        if 0 <= node_id < len(self.node_features):
            return sum(self.node_features[node_id])
        return 0.0


class SimpleFederatedAgent:
    """Basic federated agent for Generation 1."""
    
    def __init__(self, agent_id: int, learning_rate: float = 0.01):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        # Simple parameter vector (10 parameters)
        self.parameters = [random.gauss(0, 0.1) for _ in range(10)]
        self.local_rewards = []
        self.training_steps = 0
    
    def select_action(self, state: MockGraphState) -> float:
        """Select action based on simple policy."""
        # Use agent's node if available, otherwise use global state
        if self.agent_id < state.num_nodes:
            node_sum = state.get_node_feature_sum(self.agent_id)
        else:
            node_sum = sum(sum(features) for features in state.node_features)
        
        # Simple linear policy with tanh activation
        action = SimpleMath.tanh(node_sum * self.parameters[0])
        return action
    
    def compute_reward(self, state: MockGraphState, action: float) -> float:
        """Compute reward (simplified)."""
        # Reward based on graph connectivity and action
        connectivity = state.num_edges / max(state.num_nodes, 1)
        action_penalty = 0.1 * (action ** 2)
        reward = connectivity - action_penalty
        return reward
    
    def local_update(self, state: MockGraphState, action: float, reward: float):
        """Simple local parameter update."""
        # Gradient ascent approximation
        for i in range(len(self.parameters)):
            gradient = reward * random.gauss(0, 0.1)
            self.parameters[i] += self.learning_rate * gradient
        
        self.local_rewards.append(reward)
        self.training_steps += 1
    
    def get_parameters(self) -> List[float]:
        """Get current parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: List[float]):
        """Set parameters from federation."""
        self.parameters = parameters.copy()


class SimpleFederationProtocol:
    """Basic federation protocol for Generation 1."""
    
    def __init__(self, agents: List[SimpleFederatedAgent]):
        self.agents = agents
        self.global_parameters = [0.0] * 10
        self.federation_rounds = 0
    
    def federated_averaging(self):
        """Simple federated averaging."""
        # Collect parameters from all agents
        all_params = [agent.get_parameters() for agent in self.agents]
        
        # Average parameters
        num_params = len(self.global_parameters)
        for i in range(num_params):
            param_sum = sum(params[i] for params in all_params)
            self.global_parameters[i] = param_sum / len(all_params)
        
        # Distribute back to agents
        for agent in self.agents:
            agent.set_parameters(self.global_parameters)
        
        self.federation_rounds += 1
    
    def get_federation_stats(self) -> Dict[str, Any]:
        """Get federation statistics."""
        recent_rewards = []
        for agent in self.agents:
            if agent.local_rewards:
                recent_rewards.extend(agent.local_rewards[-10:])  # Last 10 rewards
        
        return {
            "federation_rounds": self.federation_rounds,
            "num_agents": len(self.agents),
            "avg_reward": SimpleMath.mean(recent_rewards),
            "parameter_norm": SimpleMath.norm(self.global_parameters),
        }


class SimpleTrafficEnvironment:
    """Simplified traffic network for Generation 1."""
    
    def __init__(self, num_intersections: int = 10):
        self.num_intersections = num_intersections
        self.reset()
    
    def reset(self) -> MockGraphState:
        """Reset environment."""
        # Create simple grid topology with random traffic features
        node_features = []
        for i in range(self.num_intersections):
            features = [random.uniform(0, 1) for _ in range(4)]  # Traffic flow features
            node_features.append(features)
        
        # Simple edge connectivity (chain topology)
        edges = []
        for i in range(self.num_intersections - 1):
            edges.append((i, i+1))  # Chain connection
            edges.append((i+1, i))  # Bidirectional
        
        self.current_state = MockGraphState(
            node_features=node_features,
            edges=edges,
            timestamp=time.time()
        )
        return self.current_state
    
    def step(self, actions: Dict[int, float]) -> Tuple[MockGraphState, Dict[int, float], bool]:
        """Environment step."""
        # Simple dynamics: traffic flow changes based on actions
        for i in range(self.num_intersections):
            if i in actions:
                # Action affects traffic flow
                for j in range(len(self.current_state.node_features[i])):
                    self.current_state.node_features[i][j] += 0.1 * actions[i]
        
        # Add some noise and clip values
        for i in range(self.num_intersections):
            for j in range(len(self.current_state.node_features[i])):
                noise = random.gauss(0, 0.05)
                self.current_state.node_features[i][j] += noise
                self.current_state.node_features[i][j] = SimpleMath.clip(
                    self.current_state.node_features[i][j], 0.0, 2.0
                )
        
        # Compute rewards for all agents
        rewards = {}
        for agent_id in actions:
            if agent_id < self.num_intersections:
                # Reward based on local traffic efficiency (minimize congestion)
                local_traffic = self.current_state.node_features[agent_id]
                congestion = sum(feature ** 2 for feature in local_traffic)
                rewards[agent_id] = -congestion  # Negative to minimize
        
        self.current_state.timestamp = time.time()
        done = False  # Simple version doesn't terminate
        
        return self.current_state, rewards, done


def run_generation1_demo():
    """Run Generation 1 demo: Basic federated learning."""
    print("🚀 Generation 1: MAKE IT WORK - Basic Federated RL Demo")
    print("=" * 60)
    
    # Initialize environment
    env = SimpleTrafficEnvironment(num_intersections=5)
    
    # Create federated agents
    agents = [SimpleFederatedAgent(agent_id=i) for i in range(3)]
    federation = SimpleFederationProtocol(agents)
    
    # Training parameters
    episodes = 20
    steps_per_episode = 50
    federation_interval = 10
    
    results = []
    
    print(f"Training {len(agents)} agents for {episodes} episodes...")
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        episode_rewards = {i: [] for i in range(len(agents))}
        
        for step in range(steps_per_episode):
            # Each agent takes action
            actions = {}
            for agent in agents:
                action = agent.select_action(state)
                actions[agent.agent_id] = action
            
            # Environment step
            next_state, rewards, done = env.step(actions)
            
            # Local learning
            for agent in agents:
                if agent.agent_id in rewards:
                    agent.local_update(state, actions[agent.agent_id], rewards[agent.agent_id])
                    episode_rewards[agent.agent_id].append(rewards[agent.agent_id])
            
            # Federated aggregation
            if step % federation_interval == 0:
                federation.federated_averaging()
            
            state = next_state
            
            if done:
                break
        
        # Episode statistics
        avg_rewards = {}
        for i in range(len(agents)):
            if episode_rewards[i]:
                avg_rewards[i] = SimpleMath.mean(episode_rewards[i])
            else:
                avg_rewards[i] = 0.0
        
        fed_stats = federation.get_federation_stats()
        
        result = {
            "episode": episode,
            "agent_rewards": avg_rewards,
            "federation_stats": fed_stats,
            "timestamp": time.time()
        }
        results.append(result)
        
        if episode % 5 == 0:
            print(f"Episode {episode:2d}: Avg Reward = {fed_stats['avg_reward']:.3f}, "
                  f"Fed Rounds = {fed_stats['federation_rounds']}")
    
    total_time = time.time() - start_time
    
    # Save results
    with open('/root/repo/gen1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Generation 1 Complete!")
    print(f"Final average reward: {results[-1]['federation_stats']['avg_reward']:.3f}")
    print(f"Total federation rounds: {results[-1]['federation_stats']['federation_rounds']}")
    print(f"Training time: {total_time:.2f} seconds")
    print("Results saved to gen1_results.json")
    
    # Basic validation
    assert len(results) == episodes, "Incorrect number of results"
    assert results[-1]['federation_stats']['federation_rounds'] > 0, "No federation occurred"
    print("✅ Basic validation passed")
    
    return results


if __name__ == "__main__":
    results = run_generation1_demo()