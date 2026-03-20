"""FederatedAggregator: federated averaging of Q-tables across agents."""

import numpy as np
from fedrl.agent import FedAgent
from fedrl.env import GraphEnv
from fedrl.utils import epsilon_greedy


class FederatedAggregator:
    """
    Coordinates a set of FedAgents using Federated Averaging.

    Each round:
    1. Each agent runs one full episode independently.
    2. Q-tables are averaged and broadcast back to all agents.

    Args:
        agents (list[FedAgent]): The list of local agents.
    """

    def __init__(self, agents: list[FedAgent]):
        if not agents:
            raise ValueError("agents list must be non-empty")
        self.agents = agents

    # ------------------------------------------------------------------

    def aggregate(self) -> np.ndarray:
        """
        Average Q-tables across all agents and update each agent in-place.

        Returns:
            The averaged Q-table (numpy array).
        """
        mean_q = np.mean([a.q_table for a in self.agents], axis=0)
        for agent in self.agents:
            agent.q_table = mean_q.copy()
        return mean_q

    def run_episode(self, env: GraphEnv, epsilon: float = 0.1) -> list[float]:
        """
        Run one episode per agent, then aggregate.

        Args:
            env: The shared environment (each agent resets it independently).
            epsilon: Exploration rate passed to each agent.

        Returns:
            List of total rewards, one per agent.
        """
        rewards = []
        for agent in self.agents:
            total_reward = _run_single_episode(agent, env, epsilon)
            rewards.append(total_reward)

        self.aggregate()
        return rewards


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _run_single_episode(agent: FedAgent, env: GraphEnv, epsilon: float) -> float:
    """Run one episode for a single agent and return total reward."""
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = epsilon_greedy(agent.q_table, state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.train_step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    return total_reward
