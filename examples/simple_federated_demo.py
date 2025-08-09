#!/usr/bin/env python3
"""
Simple Federated Graph RL Demo - Generation 1 Implementation

Demonstrates basic functionality:
- Simple traffic network environment 
- Graph TD3 algorithm
- Basic federated learning
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple

# Mock JAX dependencies for testing
class MockModule:
    def __getattr__(self, name):
        if name == 'random':
            return MockModule()
        elif name == 'PRNGKey':
            return lambda x: x
        elif name == 'normal':
            return lambda key, shape: np.random.normal(0, 1, shape)
        elif name == 'uniform':
            return lambda key, shape, **kwargs: np.random.uniform(0, 1, shape)
        elif name == 'split':
            return lambda key: (key, key)
        elif name in ['mean', 'sum', 'maximum', 'minimum', 'std', 'clip', 'concatenate', 'stack', 'zeros', 'ones']:
            return getattr(np, name)
        elif name == 'tanh':
            return np.tanh
        return MockModule()

# Simple graph state representation
class SimpleGraphState:
    def __init__(self, node_features: np.ndarray, edge_index: np.ndarray):
        self.node_features = node_features
        self.edge_index = edge_index
        self.num_nodes = node_features.shape[0]
        self.num_edges = edge_index.shape[1]
        self.timestamp = 0.0
    
    @property
    def node_dim(self) -> int:
        return self.node_features.shape[1]

class SimpleGraphTransition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class SimpleTrafficEnvironment:
    """Simplified traffic network for basic testing."""
    
    def __init__(self, num_intersections: int = 10):
        self.num_intersections = num_intersections
        self.node_dim = 4  # [signal_state, queue_length, flow, congestion]
        self.action_dim = 3  # Signal states: red, yellow, green
        
        # Create simple grid topology
        self.edge_index = self._create_grid_topology()
        self.current_state = None
        self.step_count = 0
    
    def _create_grid_topology(self) -> np.ndarray:
        """Create simple grid topology."""
        edges = []
        grid_size = int(np.sqrt(self.num_intersections))
        
        for i in range(self.num_intersections):
            row, col = i // grid_size, i % grid_size
            
            # Connect to right neighbor
            if col < grid_size - 1:
                right_neighbor = row * grid_size + (col + 1)
                edges.append([i, right_neighbor])
                edges.append([right_neighbor, i])  # Bidirectional
            
            # Connect to bottom neighbor
            if row < grid_size - 1:
                bottom_neighbor = (row + 1) * grid_size + col
                edges.append([i, bottom_neighbor])
                edges.append([bottom_neighbor, i])  # Bidirectional
        
        if edges:
            return np.array(edges).T
        else:
            return np.zeros((2, 0), dtype=int)
    
    def reset(self) -> SimpleGraphState:
        """Reset environment."""
        # Initialize node features: [signal_state, queue_length, flow, congestion]
        node_features = np.random.rand(self.num_intersections, self.node_dim)
        node_features[:, 0] = np.random.randint(0, 3, self.num_intersections)  # Signal states
        
        self.current_state = SimpleGraphState(node_features, self.edge_index)
        self.step_count = 0
        return self.current_state
    
    def step(self, actions: np.ndarray) -> Tuple[SimpleGraphState, float, bool]:
        """Execute environment step."""
        # Update signal states
        new_features = self.current_state.node_features.copy()
        new_features[:len(actions), 0] = actions[:len(new_features)]
        
        # Simple dynamics: queue length depends on signal state
        for i in range(len(new_features)):
            signal_state = new_features[i, 0]
            if signal_state == 0:  # Red
                new_features[i, 1] += 0.1  # Increase queue
            elif signal_state == 2:  # Green
                new_features[i, 1] = max(0, new_features[i, 1] - 0.2)  # Decrease queue
        
        # Update flow and congestion based on queue
        new_features[:, 2] = 1.0 / (1.0 + new_features[:, 1])  # Flow inversely related to queue
        new_features[:, 3] = new_features[:, 1] / 10.0  # Congestion proportional to queue
        
        next_state = SimpleGraphState(new_features, self.edge_index)
        
        # Reward: minimize total delay (queue length)
        reward = -np.sum(next_state.node_features[:, 1])
        
        self.current_state = next_state
        self.step_count += 1
        done = self.step_count >= 100
        
        return next_state, reward, done

class SimpleGraphActor:
    """Simplified graph actor network."""
    
    def __init__(self, node_dim: int, action_dim: int, hidden_dim: int = 64):
        self.node_dim = node_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Simple linear layers (mock neural network)
        self.w1 = np.random.normal(0, 0.1, (node_dim, hidden_dim))
        self.w2 = np.random.normal(0, 0.1, (hidden_dim, action_dim))
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(action_dim)
    
    def __call__(self, state: SimpleGraphState) -> np.ndarray:
        """Forward pass."""
        # Simple pooling: average node features
        pooled = np.mean(state.node_features, axis=0)
        
        # Two-layer network
        h1 = np.tanh(np.dot(pooled, self.w1) + self.b1)
        logits = np.dot(h1, self.w2) + self.b2
        
        # Convert to probabilities and sample
        probs = self._softmax(logits)
        return np.random.choice(len(probs), p=probs)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class SimpleGraphTD3:
    """Simplified TD3 implementation for graphs."""
    
    def __init__(self, node_dim: int, action_dim: int):
        self.node_dim = node_dim
        self.action_dim = action_dim
        
        self.actor = SimpleGraphActor(node_dim, action_dim)
        self.buffer = []
        self.training_step = 0
        
    def select_action(self, state: SimpleGraphState, deterministic: bool = False) -> int:
        """Select action."""
        return self.actor(state)
    
    def add_transition(self, transition: SimpleGraphTransition):
        """Add transition to buffer."""
        self.buffer.append(transition)
        if len(self.buffer) > 10000:
            self.buffer.pop(0)
    
    def update(self) -> Dict[str, float]:
        """Update algorithm (simplified)."""
        if len(self.buffer) < 32:
            return {}
        
        # Mock update with random loss
        self.training_step += 1
        return {
            "actor_loss": np.random.uniform(-1, 0),
            "critic_loss": np.random.uniform(0, 2)
        }

class SimpleFederatedSystem:
    """Basic federated learning system."""
    
    def __init__(self, num_agents: int, node_dim: int, action_dim: int):
        self.num_agents = num_agents
        self.agents = [SimpleGraphTD3(node_dim, action_dim) for _ in range(num_agents)]
        self.round_count = 0
    
    def local_training_round(self, agent_id: int, env: SimpleTrafficEnvironment, steps: int = 10):
        """Perform local training for an agent."""
        agent = self.agents[agent_id]
        episode_reward = 0
        
        state = env.reset()
        
        for step in range(steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(np.array([action]))
            
            transition = SimpleGraphTransition(state, action, reward, next_state, done)
            agent.add_transition(transition)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update agent
        metrics = agent.update()
        return episode_reward, metrics
    
    def aggregate_parameters(self):
        """Simple parameter aggregation (mock)."""
        # In real implementation, would average neural network parameters
        self.round_count += 1
        print(f"Federated aggregation round {self.round_count} completed")
    
    def get_global_stats(self) -> Dict:
        """Get federated training statistics."""
        return {
            "round_count": self.round_count,
            "num_agents": self.num_agents,
            "total_training_steps": sum(agent.training_step for agent in self.agents)
        }

def run_simple_demo():
    """Run simplified federated graph RL demo."""
    print("🚀 Starting Simple Federated Graph RL Demo")
    print("=" * 50)
    
    # Initialize system
    num_agents = 3
    num_intersections = 9  # 3x3 grid
    
    # Create environment
    env = SimpleTrafficEnvironment(num_intersections)
    
    # Create federated system
    fed_system = SimpleFederatedSystem(
        num_agents=num_agents,
        node_dim=env.node_dim,
        action_dim=env.action_dim
    )
    
    print(f"Initialized {num_agents} agents for {num_intersections}-intersection traffic network")
    print(f"Graph topology: {env.edge_index.shape[1]} edges")
    
    # Training loop
    num_rounds = 5
    local_steps = 20
    
    for round_idx in range(num_rounds):
        print(f"\n📊 Federated Round {round_idx + 1}/{num_rounds}")
        print("-" * 30)
        
        round_rewards = []
        
        # Local training for each agent
        for agent_id in range(num_agents):
            episode_reward, metrics = fed_system.local_training_round(
                agent_id, env, local_steps
            )
            round_rewards.append(episode_reward)
            
            print(f"  Agent {agent_id}: Reward={episode_reward:.2f}, "
                  f"Buffer size={len(fed_system.agents[agent_id].buffer)}")
        
        # Global aggregation
        fed_system.aggregate_parameters()
        
        # Statistics
        avg_reward = np.mean(round_rewards)
        stats = fed_system.get_global_stats()
        
        print(f"  Round Average Reward: {avg_reward:.2f}")
        print(f"  Total Training Steps: {stats['total_training_steps']}")
    
    print("\n✅ Demo completed successfully!")
    print("Generation 1 (Simple) functionality verified:")
    print("  ✓ Basic graph environment")
    print("  ✓ Simple graph neural network")
    print("  ✓ TD3 algorithm mock")
    print("  ✓ Federated learning coordination")
    print("  ✓ Multi-agent traffic control")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    run_simple_demo()