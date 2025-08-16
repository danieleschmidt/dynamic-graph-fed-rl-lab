#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Simple mock demo without heavy dependencies.
This demonstrates the basic federated RL structure with minimal viable features.
"""

import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

@dataclass
class MockGraphState:
    """Simplified graph state for Generation 1."""
    node_features: np.ndarray
    edge_index: np.ndarray
    timestamp: float = 0.0
    
    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]
    
    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index.size > 0 else 0


class SimpleFederatedAgent:
    """Basic federated agent for Generation 1."""
    
    def __init__(self, agent_id: int, learning_rate: float = 0.01):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.parameters = np.random.normal(0, 0.1, size=(10,))  # Simple parameter vector
        self.local_rewards = []
        self.training_steps = 0
    
    def select_action(self, state: MockGraphState) -> np.ndarray:
        """Select action based on simple policy."""
        # Simple linear policy
        node_sum = np.sum(state.node_features)
        action = np.tanh(node_sum * self.parameters[0])
        return np.array([action])
    
    def compute_reward(self, state: MockGraphState, action: np.ndarray) -> float:
        """Compute reward (simplified)."""
        # Reward based on graph connectivity and action
        connectivity = state.num_edges / max(state.num_nodes, 1)
        action_penalty = 0.1 * np.sum(action ** 2)
        reward = connectivity - action_penalty
        return float(reward)
    
    def local_update(self, state: MockGraphState, action: np.ndarray, reward: float):
        """Simple local parameter update."""
        # Gradient ascent approximation
        gradient = reward * np.random.normal(0, 0.1, size=self.parameters.shape)
        self.parameters += self.learning_rate * gradient
        self.local_rewards.append(reward)
        self.training_steps += 1
    
    def get_parameters(self) -> np.ndarray:
        """Get current parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: np.ndarray):
        """Set parameters from federation."""
        self.parameters = parameters.copy()


class SimpleFederationProtocol:
    """Basic federation protocol for Generation 1."""
    
    def __init__(self, agents: List[SimpleFederatedAgent]):
        self.agents = agents
        self.global_parameters = np.zeros((10,))
        self.federation_rounds = 0
    
    def federated_averaging(self):
        """Simple federated averaging."""
        # Collect parameters from all agents
        all_params = np.array([agent.get_parameters() for agent in self.agents])
        
        # Average parameters
        self.global_parameters = np.mean(all_params, axis=0)
        
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
            "avg_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
            "parameter_norm": np.linalg.norm(self.global_parameters),
        }


class SimpleTrafficEnvironment:
    """Simplified traffic network for Generation 1."""
    
    def __init__(self, num_intersections: int = 10):
        self.num_intersections = num_intersections
        self.reset()
    
    def reset(self) -> MockGraphState:
        """Reset environment."""
        # Create simple grid topology
        node_features = np.random.uniform(0, 1, size=(self.num_intersections, 4))  # Traffic flow features
        
        # Simple edge connectivity (chain topology)
        edges = []
        for i in range(self.num_intersections - 1):
            edges.extend([[i, i+1], [i+1, i]])  # Bidirectional
        
        edge_index = np.array(edges).T if edges else np.zeros((2, 0))
        
        self.current_state = MockGraphState(
            node_features=node_features,
            edge_index=edge_index,
            timestamp=time.time()
        )
        return self.current_state
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[MockGraphState, Dict[int, float], bool]:
        """Environment step."""
        # Simple dynamics: traffic flow changes
        for i in range(self.num_intersections):
            if i in actions:
                # Action affects traffic flow
                self.current_state.node_features[i] += 0.1 * actions[i]
        
        # Add some noise
        self.current_state.node_features += np.random.normal(0, 0.05, self.current_state.node_features.shape)
        self.current_state.node_features = np.clip(self.current_state.node_features, 0, 2)
        
        # Compute rewards for all agents
        rewards = {}
        for agent_id in actions:
            if agent_id < self.num_intersections:
                # Reward based on local traffic efficiency
                local_traffic = self.current_state.node_features[agent_id]
                rewards[agent_id] = -np.sum(local_traffic ** 2)  # Minimize congestion
        
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
        avg_rewards = {i: np.mean(episode_rewards[i]) if episode_rewards[i] else 0 
                      for i in range(len(agents))}
        
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
    
    # Save results
    with open('/root/repo/gen1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Generation 1 Complete!")
    print(f"Final average reward: {results[-1]['federation_stats']['avg_reward']:.3f}")
    print(f"Total federation rounds: {results[-1]['federation_stats']['federation_rounds']}")
    print("Results saved to gen1_results.json")
    
    return results


if __name__ == "__main__":
    results = run_generation1_demo()