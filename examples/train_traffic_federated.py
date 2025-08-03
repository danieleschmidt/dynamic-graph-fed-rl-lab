#!/usr/bin/env python3
"""
Example training script for federated Graph TD3 on traffic networks.

This script demonstrates:
1. Creating a dynamic traffic environment
2. Setting up federated Graph TD3 agents
3. Running distributed training with gossip protocol
4. Monitoring performance and convergence
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

# Import our federated RL components
from src.dynamic_graph_fed_rl.algorithms.graph_td3 import GraphTD3, FederatedGraphTD3
from src.dynamic_graph_fed_rl.environments.traffic_network import TrafficNetworkEnv
from src.dynamic_graph_fed_rl.federation.gossip import AsyncGossipProtocol
from src.dynamic_graph_fed_rl.utils.data_loader import GraphDataGenerator


class FederatedTrafficTraining:
    """Main training coordinator for federated traffic optimization."""
    
    def __init__(
        self,
        num_agents: int = 5,
        num_intersections: int = 50,
        max_episodes: int = 1000,
        communication_round: int = 50,
        save_dir: str = "experiments/traffic_federated",
    ):
        self.num_agents = num_agents
        self.num_intersections = num_intersections
        self.max_episodes = max_episodes
        self.communication_round = communication_round
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.convergence_metrics = []
        self.communication_stats = []
        
        # Initialize environment
        self.env = TrafficNetworkEnv(
            num_intersections=num_intersections,
            scenario="city_grid",
            rush_hour_dynamics=True,
            incident_probability=0.001,
            weather_enabled=True,
        )
        
        # Initialize federated system
        self._setup_federated_system()
        
        print(f"Initialized federated training with {num_agents} agents")
        print(f"Traffic network: {num_intersections} intersections")
        print(f"Save directory: {self.save_dir}")
    
    def _setup_federated_system(self) -> None:
        """Setup federated learning system."""
        # Create agent configurations
        agent_configs = []
        for i in range(self.num_agents):
            config = {
                "state_dim": self.env.node_feature_dim,
                "action_dim": 1,  # Each agent controls one action dimension
                "edge_dim": self.env.edge_feature_dim,
                "hidden_dim": 128,
                "gnn_type": "temporal_attention",
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_freq": 2,
                "buffer_size": 50000,
                "batch_size": 256,
                "exploration_noise": 0.1,
                "seed": 42 + i,  # Different seed for each agent
            }
            agent_configs.append(config)
        
        # Initialize federated TD3 system
        self.fed_system = FederatedGraphTD3(
            num_agents=self.num_agents,
            agent_configs=agent_configs,
            communication_round=self.communication_round,
            aggregation_method="fedavg",
        )
        
        # Initialize gossip protocol
        self.gossip_protocol = AsyncGossipProtocol(
            num_agents=self.num_agents,
            topology="small_world",
            fanout=3,
            gossip_interval=1.0,
            compression_ratio=0.1,
            byzantine_tolerance=True,
        )
    
    async def train(self) -> None:
        """Main training loop."""
        print("Starting federated training...")
        start_time = time.time()
        
        for episode in range(self.max_episodes):
            episode_start = time.time()
            
            # Run episode for each agent
            episode_metrics = await self._run_episode(episode)
            
            # Federated aggregation
            if episode % self.communication_round == 0:
                await self._federated_round(episode)
            
            # Track metrics
            self.episode_rewards.append(episode_metrics["total_reward"])
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                episode_time = time.time() - episode_start
                
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Time: {episode_time:6.2f}s")
                
                # Save checkpoint
                if episode % 100 == 0:
                    self._save_checkpoint(episode)
        
        # Final save
        self._save_results()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
    
    async def _run_episode(self, episode: int) -> Dict[str, float]:
        """Run one episode of training."""
        # Reset environment
        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        max_steps = 200
        
        while episode_steps < max_steps:
            # Get actions from all agents
            actions = {}
            
            for agent_id, agent in enumerate(self.fed_system.agents):
                if agent_id < self.num_intersections:
                    # Each agent controls a subset of intersections
                    agent_intersections = self._get_agent_intersections(agent_id)
                    
                    # Get local view of the graph
                    local_state = self._get_local_state(state, agent_intersections)
                    
                    # Select action
                    action = agent.select_action(local_state, deterministic=False)
                    
                    # Map to intersection actions
                    for i, intersection_id in enumerate(agent_intersections):
                        if i < len(action):
                            # Convert continuous action to discrete signal
                            discrete_action = self._continuous_to_discrete_action(action[i])
                            actions[intersection_id] = discrete_action
            
            # Fill in any missing actions with default (green)
            for i in range(self.num_intersections):
                if i not in actions:
                    actions[i] = 2  # Green light
            
            # Environment step
            next_state, rewards, done, info = self.env.step(actions)
            
            # Store transitions for each agent
            for agent_id, agent in enumerate(self.fed_system.agents):
                agent_intersections = self._get_agent_intersections(agent_id)
                
                if agent_intersections:
                    # Local reward (average of controlled intersections)
                    local_reward = np.mean([rewards.get(i, 0.0) for i in agent_intersections])
                    
                    # Create transition
                    local_state = self._get_local_state(state, agent_intersections)
                    local_next_state = self._get_local_state(next_state, agent_intersections)
                    
                    # Get agent's actions
                    agent_actions = jnp.array([actions.get(i, 2) for i in agent_intersections])
                    
                    # Add to agent's buffer
                    from src.dynamic_graph_fed_rl.algorithms.buffers import GraphTransition
                    transition = GraphTransition(
                        state=local_state,
                        action=agent_actions,
                        reward=local_reward,
                        next_state=local_next_state,
                        done=done,
                        info=info,
                    )
                    agent.add_transition(transition)
                    
                    # Local update if enough data
                    if agent.can_update():
                        agent.update({})
            
            # Update for next step
            state = next_state
            episode_reward += sum(rewards.values())
            episode_steps += 1
            
            if done:
                break
        
        # End episode for all agents
        for agent in self.fed_system.agents:
            agent.reset_episode()
            if hasattr(agent, 'buffer'):
                agent.buffer.end_episode()
        
        return {
            "total_reward": episode_reward,
            "steps": episode_steps,
            "avg_reward_per_step": episode_reward / max(episode_steps, 1),
        }
    
    def _get_agent_intersections(self, agent_id: int) -> List[int]:
        """Get intersections controlled by agent."""
        # Simple partitioning: divide intersections among agents
        intersections_per_agent = max(1, self.num_intersections // self.num_agents)
        start_idx = agent_id * intersections_per_agent
        end_idx = min((agent_id + 1) * intersections_per_agent, self.num_intersections)
        
        return list(range(start_idx, end_idx))
    
    def _get_local_state(self, state, agent_intersections: List[int]):
        """Get local state view for agent."""
        # For simplicity, return full state
        # In practice, you'd extract subgraph around agent's intersections
        return state
    
    def _continuous_to_discrete_action(self, continuous_action: float) -> int:
        """Convert continuous action to discrete traffic signal."""
        # Map [-1, 1] to [0, 1, 2] (red, yellow, green)
        if continuous_action < -0.33:
            return 0  # Red
        elif continuous_action < 0.33:
            return 1  # Yellow
        else:
            return 2  # Green
    
    async def _federated_round(self, episode: int) -> None:
        """Execute federated aggregation round."""
        print(f"  Federated round at episode {episode}")
        
        # Aggregate parameters
        aggregation_metrics = await self.fed_system.federated_round()
        
        # Track communication metrics
        comm_stats = self.gossip_protocol.get_gossip_metrics()
        self.communication_stats.append({
            "episode": episode,
            "aggregation_metrics": aggregation_metrics,
            "gossip_metrics": comm_stats,
        })
        
        # Log aggregation results
        if aggregation_metrics.get("convergence", False):
            print(f"    Convergence detected at episode {episode}")
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir = self.save_dir / f"checkpoint_episode_{episode}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save each agent
        for i, agent in enumerate(self.fed_system.agents):
            agent.save_checkpoint(checkpoint_dir / f"agent_{i}.pkl")
        
        # Save metrics
        metrics = {
            "episode": episode,
            "episode_rewards": self.episode_rewards,
            "communication_stats": self.communication_stats,
        }
        
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"    Checkpoint saved at episode {episode}")
    
    def _save_results(self) -> None:
        """Save final training results."""
        results = {
            "num_agents": self.num_agents,
            "num_intersections": self.num_intersections,
            "max_episodes": self.max_episodes,
            "communication_round": self.communication_round,
            "episode_rewards": self.episode_rewards,
            "communication_stats": self.communication_stats,
            "final_performance": {
                "avg_reward_last_100": float(np.mean(self.episode_rewards[-100:])),
                "best_reward": float(max(self.episode_rewards)) if self.episode_rewards else 0.0,
                "convergence_achieved": len(self.communication_stats) > 0 and 
                                       any(stats["aggregation_metrics"].get("convergence", False) 
                                           for stats in self.communication_stats),
            }
        }
        
        with open(self.save_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.save_dir / 'final_results.json'}")


def create_baseline_experiment() -> None:
    """Create baseline single-agent experiment for comparison."""
    print("Running baseline single-agent experiment...")
    
    env = TrafficNetworkEnv(
        num_intersections=50,
        scenario="city_grid",
        rush_hour_dynamics=True,
    )
    
    # Single TD3 agent
    agent = GraphTD3(
        state_dim=env.node_feature_dim,
        action_dim=env.num_intersections,  # Control all intersections
        edge_dim=env.edge_feature_dim,
        hidden_dim=128,
        seed=42,
    )
    
    episode_rewards = []
    max_episodes = 200
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(200):
            # Get action for all intersections
            action = agent.select_action(state, deterministic=False)
            
            # Convert to discrete actions
            discrete_actions = {}
            for i in range(env.num_intersections):
                if i < len(action):
                    if action[i] < -0.33:
                        discrete_actions[i] = 0
                    elif action[i] < 0.33:
                        discrete_actions[i] = 1
                    else:
                        discrete_actions[i] = 2
                else:
                    discrete_actions[i] = 2
            
            next_state, rewards, done, info = env.step(discrete_actions)
            
            # Store transition
            from src.dynamic_graph_fed_rl.algorithms.buffers import GraphTransition
            transition = GraphTransition(
                state=state,
                action=action,
                reward=sum(rewards.values()),
                next_state=next_state,
                done=done,
                info=info,
            )
            agent.add_transition(transition)
            
            # Update agent
            if agent.can_update():
                agent.update({})
            
            state = next_state
            episode_reward += sum(rewards.values())
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        agent.reset_episode()
        
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Baseline Episode {episode:3d} | Avg Reward: {avg_reward:8.2f}")
    
    # Save baseline results
    baseline_results = {
        "type": "baseline_single_agent",
        "episode_rewards": episode_rewards,
        "final_performance": {
            "avg_reward_last_50": float(np.mean(episode_rewards[-50:])),
            "best_reward": float(max(episode_rewards)),
        }
    }
    
    save_dir = Path("experiments/baseline")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"Baseline results saved to {save_dir / 'baseline_results.json'}")


async def main():
    """Main execution function."""
    print("Dynamic Graph Federated RL - Traffic Network Example")
    print("=" * 60)
    
    # Run baseline experiment first
    create_baseline_experiment()
    
    print("\n" + "=" * 60)
    
    # Run federated experiment
    trainer = FederatedTrafficTraining(
        num_agents=5,
        num_intersections=50,
        max_episodes=500,
        communication_round=25,
        save_dir="experiments/traffic_federated",
    )
    
    await trainer.train()
    
    print("\n" + "=" * 60)
    print("Training completed! Check the experiments/ directory for results.")


if __name__ == "__main__":
    # Set JAX to use CPU for this example (comment out for GPU)
    jax.config.update('jax_platform_name', 'cpu')
    
    # Run the training
    asyncio.run(main())