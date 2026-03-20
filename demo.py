"""Demo: 3 agents navigate a changing graph using federated Q-learning."""

import numpy as np

from fedrl.env import GraphEnv
from fedrl.agent import FedAgent
from fedrl.aggregator import FederatedAggregator

NUM_AGENTS = 3
NUM_EPISODES = 100
EPSILON_START = 1.0
EPSILON_END = 0.05
WINDOW = 10  # for printing running averages


def linear_epsilon(episode: int, total: int) -> float:
    """Linearly decay epsilon from EPSILON_START to EPSILON_END."""
    return EPSILON_START + (EPSILON_END - EPSILON_START) * episode / total


def main():
    env = GraphEnv(max_nodes=20, max_actions=4, max_steps=50, seed=42)
    agents = [FedAgent(max_nodes=20, max_actions=4) for _ in range(NUM_AGENTS)]
    agg = FederatedAggregator(agents)

    all_rewards = []  # mean reward across agents per episode

    print(f"Training {NUM_AGENTS} agents for {NUM_EPISODES} episodes…\n")

    for ep in range(NUM_EPISODES):
        epsilon = linear_epsilon(ep, NUM_EPISODES)
        rewards = agg.run_episode(env, epsilon=epsilon)
        mean_reward = float(np.mean(rewards))
        all_rewards.append(mean_reward)

        if (ep + 1) % WINDOW == 0:
            avg = float(np.mean(all_rewards[-WINDOW:]))
            print(f"  Episode {ep + 1:4d} | ε={epsilon:.3f} | "
                  f"mean_reward={mean_reward:+.3f} | "
                  f"last{WINDOW}_avg={avg:+.3f}")

    # Evaluate greedily
    print("\nGreedy evaluation (10 episodes, ε=0.0):")
    eval_rewards = agg.run_episode(env, epsilon=0.0)
    print(f"  Agent rewards: {[f'{r:+.3f}' for r in eval_rewards]}")
    print(f"  Mean: {float(np.mean(eval_rewards)):+.3f}")

    # Check improvement
    early = float(np.mean(all_rewards[:10]))
    late = float(np.mean(all_rewards[-10:]))
    improved = late > early
    print(f"\nEarly avg (eps 1-10):  {early:+.4f}")
    print(f"Late  avg (eps 91-100): {late:+.4f}")
    print(f"Improved: {'✓' if improved else '✗'}")


if __name__ == "__main__":
    main()
