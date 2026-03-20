"""Utility functions for federated RL."""

import numpy as np


def epsilon_greedy(q_table: np.ndarray, state: int, epsilon: float) -> int:
    """
    Epsilon-greedy action selection from a Q-table row.

    Args:
        q_table: Q-table of shape (num_states, num_actions).
        state: Current state index.
        epsilon: Probability of random action.

    Returns:
        Selected action index.
    """
    if np.random.random() < epsilon:
        return np.random.randint(0, q_table.shape[1])
    return int(np.argmax(q_table[state]))


def moving_average(values: list[float], window: int = 10) -> list[float]:
    """
    Compute a simple moving average over a list of values.

    Args:
        values: Input sequence.
        window: Window size.

    Returns:
        Smoothed list (same length; early values use a smaller window).
    """
    result = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start : i + 1])))
    return result
