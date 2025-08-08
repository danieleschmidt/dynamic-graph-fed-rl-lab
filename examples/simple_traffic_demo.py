#!/usr/bin/env python3
"""Simple traffic network demo without heavy dependencies."""

import random
import time
import numpy as np
from typing import Dict, List, Tuple, Optional


class SimpleNode:
    """Simple traffic intersection representation."""
    
    def __init__(self, node_id: int, x: float, y: float):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.signal_state = random.randint(0, 2)  # 0=red, 1=yellow, 2=green
        self.queue_length = random.uniform(0, 10)
        self.flow_capacity = random.uniform(30, 60)
    
    def update_signal(self, action: int):
        """Update traffic signal based on action."""
        if action in [0, 1, 2]:
            self.signal_state = action
    
    def compute_delay(self) -> float:
        """Compute traffic delay at intersection."""
        base_delay = self.queue_length / max(self.flow_capacity, 1e-6)
        if self.signal_state == 0:  # Red
            base_delay *= 2.0
        elif self.signal_state == 1:  # Yellow
            base_delay *= 1.5
        return base_delay


class SimpleTrafficNetwork:
    """Simple traffic network without graph dependencies."""
    
    def __init__(self, num_intersections: int = 20):
        self.num_intersections = num_intersections
        self.nodes = self._create_nodes()
        self.edges = self._create_edges()
        self.time_step = 0
        
    def _create_nodes(self) -> List[SimpleNode]:
        """Create traffic intersections."""
        nodes = []
        for i in range(self.num_intersections):
            x = random.uniform(0, 10)
            y = random.uniform(0, 10)
            node = SimpleNode(i, x, y)
            nodes.append(node)
        return nodes
    
    def _create_edges(self) -> List[Tuple[int, int]]:
        """Create simple grid-like connections."""
        edges = []
        grid_size = int(np.sqrt(self.num_intersections))
        
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                if node_id >= self.num_intersections:
                    break
                
                # Connect to right neighbor
                if j < grid_size - 1:
                    right_neighbor = i * grid_size + (j + 1)
                    if right_neighbor < self.num_intersections:
                        edges.append((node_id, right_neighbor))
                
                # Connect to bottom neighbor
                if i < grid_size - 1:
                    bottom_neighbor = (i + 1) * grid_size + j
                    if bottom_neighbor < self.num_intersections:
                        edges.append((node_id, bottom_neighbor))
        
        return edges
    
    def step(self, actions: Dict[int, int]) -> Tuple[float, Dict[str, float]]:
        """Execute one simulation step."""
        # Update traffic signals
        for node_id, action in actions.items():
            if 0 <= node_id < len(self.nodes):
                self.nodes[node_id].update_signal(action)
        
        # Update traffic dynamics
        self._update_traffic_flow()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        self.time_step += 1
        return reward, metrics
    
    def _update_traffic_flow(self):
        """Update traffic flow and queue lengths."""
        for node in self.nodes:
            # Simple queue length update
            if node.signal_state == 2:  # Green
                node.queue_length = max(0, node.queue_length - 2.0)
            else:
                node.queue_length += random.uniform(0, 1.5)
            
            # Add some randomness
            node.queue_length += random.uniform(-0.5, 0.5)
            node.queue_length = max(0, node.queue_length)
    
    def _compute_reward(self) -> float:
        """Compute global reward."""
        total_delay = sum(node.compute_delay() for node in self.nodes)
        avg_queue = sum(node.queue_length for node in self.nodes) / len(self.nodes)
        
        # Minimize delay and queue length
        reward = -(total_delay + avg_queue * 0.1)
        return reward
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute performance metrics."""
        total_delay = sum(node.compute_delay() for node in self.nodes)
        avg_queue = sum(node.queue_length for node in self.nodes) / len(self.nodes)
        green_signals = sum(1 for node in self.nodes if node.signal_state == 2)
        
        return {
            "total_delay": total_delay,
            "avg_queue_length": avg_queue,
            "green_signals": green_signals,
            "green_ratio": green_signals / len(self.nodes),
        }
    
    def get_state(self) -> Dict[str, any]:
        """Get current network state."""
        node_states = []
        for node in self.nodes:
            node_states.append({
                "id": node.node_id,
                "x": node.x,
                "y": node.y,
                "signal": node.signal_state,
                "queue": node.queue_length,
                "capacity": node.flow_capacity,
                "delay": node.compute_delay(),
            })
        
        return {
            "nodes": node_states,
            "edges": self.edges,
            "time_step": self.time_step,
        }


class SimpleAgent:
    """Simple learning agent for traffic control."""
    
    def __init__(self, agent_id: int, exploration_rate: float = 0.3):
        self.agent_id = agent_id
        self.exploration_rate = exploration_rate
        self.q_table = {}  # Simple Q-learning table
        self.learning_rate = 0.1
        self.discount = 0.95
        
    def get_state_key(self, node_state: Dict) -> str:
        """Convert node state to a hashable key."""
        signal = node_state["signal"]
        queue_level = int(node_state["queue"] // 2)  # Discretize
        return f"{signal}_{queue_level}"
    
    def select_action(self, node_state: Dict) -> int:
        """Select action using epsilon-greedy policy."""
        state_key = self.get_state_key(node_state)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0]  # 3 actions
        
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.randint(0, 2)
        else:
            return int(np.argmax(self.q_table[state_key]))
    
    def update_q_values(self, state: Dict, action: int, reward: float, next_state: Dict):
        """Update Q-values using Q-learning."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0, 0.0, 0.0]
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])
        target_q = reward + self.discount * max_next_q
        
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)


class SimpleFederatedSystem:
    """Simple federated learning system."""
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agents = [SimpleAgent(i) for i in range(num_agents)]
        self.communication_round = 0
        
    def aggregate_q_tables(self):
        """Simple federated averaging of Q-tables."""
        # Collect all unique states
        all_states = set()
        for agent in self.agents:
            all_states.update(agent.q_table.keys())
        
        # Average Q-values for each state
        aggregated_q_table = {}
        for state in all_states:
            q_values = []
            for agent in self.agents:
                if state in agent.q_table:
                    q_values.append(agent.q_table[state])
                else:
                    q_values.append([0.0, 0.0, 0.0])
            
            # Compute average
            avg_q = [0.0, 0.0, 0.0]
            for i in range(3):
                avg_q[i] = sum(q[i] for q in q_values) / len(q_values)
            
            aggregated_q_table[state] = avg_q
        
        # Update all agents with aggregated Q-table
        for agent in self.agents:
            agent.q_table.update(aggregated_q_table)
        
        self.communication_round += 1
        print(f"Federated aggregation round {self.communication_round} completed")


def run_simple_demo():
    """Run a simple traffic control demo."""
    print("🚦 Starting Simple Traffic Control Demo")
    print("=" * 50)
    
    # Create environment and federated system
    env = SimpleTrafficNetwork(num_intersections=16)
    fed_system = SimpleFederatedSystem(num_agents=4)
    
    # Training parameters
    num_episodes = 50
    steps_per_episode = 100
    fed_communication_interval = 200
    
    episode_rewards = []
    step_count = 0
    
    for episode in range(num_episodes):
        episode_reward = 0
        
        # Store previous states for learning
        prev_states = {}
        prev_actions = {}
        
        for step in range(steps_per_episode):
            # Get current state
            current_state = env.get_state()
            
            # Each agent selects actions for their assigned intersections
            actions = {}
            for i, agent in enumerate(fed_system.agents):
                # Simple assignment: each agent controls 4 intersections
                agent_intersections = list(range(i * 4, min((i + 1) * 4, env.num_intersections)))
                
                for intersection_id in agent_intersections:
                    if intersection_id < len(current_state["nodes"]):
                        node_state = current_state["nodes"][intersection_id]
                        action = agent.select_action(node_state)
                        actions[intersection_id] = action
            
            # Execute step in environment
            reward, metrics = env.step(actions)
            episode_reward += reward
            
            # Update Q-values for agents
            if step > 0:  # Need previous state for learning
                for i, agent in enumerate(fed_system.agents):
                    agent_intersections = list(range(i * 4, min((i + 1) * 4, env.num_intersections)))
                    
                    # Simple reward shaping: agents get global reward
                    for intersection_id in agent_intersections:
                        if (intersection_id in prev_states and 
                            intersection_id in prev_actions and
                            intersection_id < len(current_state["nodes"])):
                            
                            agent.update_q_values(
                                prev_states[intersection_id],
                                prev_actions[intersection_id],
                                reward / len(agent_intersections),  # Share reward
                                current_state["nodes"][intersection_id]
                            )
            
            # Store states and actions for next iteration
            prev_states = {i: node for i, node in enumerate(current_state["nodes"])}
            prev_actions = actions.copy()
            
            step_count += 1
            
            # Federated communication
            if step_count % fed_communication_interval == 0:
                fed_system.aggregate_q_tables()
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode >= 10 else episode_reward
            print(f"Episode {episode:3d}: Reward = {episode_reward:7.2f}, "
                  f"Avg(10) = {avg_reward:7.2f}, "
                  f"Delay = {metrics['total_delay']:6.2f}, "
                  f"Green% = {metrics['green_ratio']:.1%}")
    
    # Final results
    print("\n" + "=" * 50)
    print("🎯 Training Results:")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Total steps: {step_count}")
    print(f"   Fed rounds: {fed_system.communication_round}")
    print(f"   Final reward: {episode_rewards[-1]:.2f}")
    print(f"   Best reward: {max(episode_rewards):.2f}")
    print(f"   Avg last 10: {np.mean(episode_rewards[-10:]):.2f}")
    
    # Show learned Q-table size
    total_states = sum(len(agent.q_table) for agent in fed_system.agents)
    print(f"   Total Q-states learned: {total_states}")
    
    print("\n✅ Simple demo completed successfully!")
    return episode_rewards, fed_system


if __name__ == "__main__":
    try:
        rewards, system = run_simple_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()