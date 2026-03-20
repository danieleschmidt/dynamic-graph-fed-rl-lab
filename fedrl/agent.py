"""FedAgent: local Q-learning agent for graph navigation."""

import numpy as np


class FedAgent:
    """
    A tabular Q-learning agent for graph navigation.

    The Q-table has shape (max_nodes, max_actions) and is initialized to zeros.

    Args:
        max_nodes (int): Size of the state space.
        max_actions (int): Size of the action space.
        lr (float): Learning rate. Default 0.1.
        gamma (float): Discount factor. Default 0.95.
    """

    def __init__(
        self,
        max_nodes: int = 20,
        max_actions: int = 4,
        lr: float = 0.1,
        gamma: float = 0.95,
    ):
        self.max_nodes = max_nodes
        self.max_actions = max_actions
        self.lr = lr
        self.gamma = gamma

        self.q_table: np.ndarray = np.zeros((max_nodes, max_actions), dtype=np.float64)

    # ------------------------------------------------------------------

    def get_action(self, state: int, epsilon: float) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Current node id.
            epsilon: Exploration probability (0 = greedy, 1 = random).

        Returns:
            Action index in [0, max_actions).
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.max_actions)
        return int(np.argmax(self.q_table[state]))

    def train_step(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        Apply one Q-learning update.

        Q[s,a] += lr * (reward + gamma * max(Q[s',:]) * (1-done) - Q[s,a])
        """
        target = reward + self.gamma * np.max(self.q_table[next_state]) * (1 - int(done))
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
