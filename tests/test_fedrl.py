"""Tests for the federated RL package."""

import numpy as np
import pytest

from fedrl.env import GraphEnv
from fedrl.agent import FedAgent
from fedrl.aggregator import FederatedAggregator
from fedrl.utils import epsilon_greedy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return GraphEnv(max_nodes=20, max_actions=4, max_steps=50, seed=0)


@pytest.fixture
def agent():
    return FedAgent(max_nodes=20, max_actions=4)


@pytest.fixture
def three_agents():
    return [FedAgent(max_nodes=20, max_actions=4) for _ in range(3)]


@pytest.fixture
def aggregator(three_agents):
    return FederatedAggregator(three_agents)


# ---------------------------------------------------------------------------
# GraphEnv tests
# ---------------------------------------------------------------------------


def test_env_initial_nodes(env):
    """GraphEnv initializes with nodes 0-4 (ring graph)."""
    assert set(env.graph.keys()) == {0, 1, 2, 3, 4}


def test_env_reset_returns_start(env):
    """reset() must return the start node (0)."""
    state = env.reset()
    assert state == 0


def test_env_reset_valid_state(env):
    """reset() return value must be a valid node."""
    state = env.reset()
    assert state in env.graph


def test_env_step_returns_valid_tuple(env):
    """step() must return (state, reward, done, info) with correct types."""
    env.reset()
    next_state, reward, done, info = env.step(0)
    assert isinstance(next_state, int)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_env_step_valid_state(env):
    """step() next_state must be a valid node."""
    env.reset()
    next_state, _, _, _ = env.step(0)
    assert next_state in env.graph


def test_env_goal_reward_positive(env):
    """Reaching the goal node yields a positive reward."""
    # Manually place agent at node 3 — action 1 leads to node 4 (goal)
    env.reset()
    env._current_node = 3
    neighbors = sorted(env.graph[3])
    goal_action = neighbors.index(env.goal_node)
    _, reward, done, _ = env.step(goal_action)
    assert reward > 0
    assert done is True


def test_env_non_goal_reward_negative(env):
    """Non-goal steps yield a small negative reward."""
    env.reset()
    # Start at node 0; any action that doesn't reach node 4 immediately
    # The ring is 0-1, so action 0 goes to node 1
    neighbors = sorted(env.graph[0])
    non_goal_action = next(
        i for i, nb in enumerate(neighbors) if nb != env.goal_node
    )
    _, reward, done, _ = env.step(non_goal_action)
    assert reward < 0


def test_env_dynamic_changes(env):
    """Graph should change between episodes (either nodes or edges differ)."""
    before = {k: sorted(v) for k, v in env.graph.items()}
    env.reset()
    after = {k: sorted(v) for k, v in env.graph.items()}
    assert before != after


def test_env_max_steps_terminates(env):
    """Episode must terminate after max_steps even if goal is never reached."""
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(0)  # always try first action
        steps += 1
        assert steps <= env.max_steps + 1, "Episode ran beyond max_steps"


# ---------------------------------------------------------------------------
# FedAgent tests
# ---------------------------------------------------------------------------


def test_agent_q_table_shape(agent):
    """Q-table must have shape (max_nodes, max_actions)."""
    assert agent.q_table.shape == (20, 4)


def test_agent_q_table_initialized_zero(agent):
    """Q-table starts at zero."""
    assert np.all(agent.q_table == 0.0)


def test_agent_get_action_valid(agent):
    """get_action() must return an action in [0, max_actions)."""
    for epsilon in [0.0, 0.5, 1.0]:
        action = agent.get_action(state=0, epsilon=epsilon)
        assert 0 <= action < agent.max_actions


def test_agent_q_values_update_after_train_step(agent):
    """Q-table values must change after a training step."""
    before = agent.q_table.copy()
    agent.train_step(state=0, action=0, reward=1.0, next_state=1, done=False)
    assert not np.array_equal(agent.q_table, before)


def test_agent_q_learning_formula(agent):
    """Verify the Q-update matches the expected formula."""
    s, a, r, ns = 2, 1, 0.5, 3
    agent.q_table[ns, :] = [0.2, 0.3, 0.1, 0.0]  # max = 0.3
    old_q = agent.q_table[s, a]
    agent.train_step(s, a, r, ns, done=False)
    expected = old_q + agent.lr * (r + agent.gamma * 0.3 - old_q)
    assert abs(agent.q_table[s, a] - expected) < 1e-9


# ---------------------------------------------------------------------------
# FederatedAggregator tests
# ---------------------------------------------------------------------------


def test_aggregator_needs_agents():
    """FederatedAggregator must reject an empty agent list."""
    with pytest.raises(ValueError):
        FederatedAggregator([])


def test_aggregator_holds_three_agents(aggregator, three_agents):
    """Aggregator holds exactly the agents passed in."""
    assert len(aggregator.agents) == 3
    assert aggregator.agents is three_agents


def test_aggregation_averages_q_tables(three_agents, aggregator):
    """aggregate() must set each agent's Q-table to the mean."""
    np.random.seed(7)
    for agent in three_agents:
        agent.q_table = np.random.rand(20, 4)

    expected_mean = np.mean([a.q_table for a in three_agents], axis=0)
    result = aggregator.aggregate()

    np.testing.assert_allclose(result, expected_mean)
    for agent in three_agents:
        np.testing.assert_allclose(agent.q_table, expected_mean)


def test_run_episode_returns_reward_list(aggregator, env):
    """run_episode() must return a list of floats, one per agent."""
    rewards = aggregator.run_episode(env, epsilon=0.5)
    assert isinstance(rewards, list)
    assert len(rewards) == 3
    assert all(isinstance(r, float) for r in rewards)


def test_run_episode_updates_q_tables(aggregator, env):
    """run_episode() must modify Q-tables (they start at 0)."""
    before = aggregator.agents[0].q_table.copy()
    aggregator.run_episode(env, epsilon=0.5)
    assert not np.array_equal(aggregator.agents[0].q_table, before)


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------


def test_epsilon_greedy_random_with_epsilon_one():
    """With epsilon=1.0, actions should be uniformly random."""
    q = np.zeros((10, 4))
    np.random.seed(0)
    actions = [epsilon_greedy(q, state=0, epsilon=1.0) for _ in range(200)]
    unique = set(actions)
    # Should hit more than one action
    assert len(unique) > 1


def test_epsilon_greedy_greedy_with_epsilon_zero():
    """With epsilon=0.0, always pick the argmax action."""
    q = np.zeros((10, 4))
    q[0, 2] = 1.0  # best action is 2
    for _ in range(20):
        assert epsilon_greedy(q, state=0, epsilon=0.0) == 2


# ---------------------------------------------------------------------------
# Demo / integration test
# ---------------------------------------------------------------------------


def test_demo_training_improves():
    """After training, late-episode mean reward should exceed early-episode mean."""
    np.random.seed(42)
    env = GraphEnv(max_nodes=20, max_actions=4, max_steps=50, seed=42)
    agents = [FedAgent(max_nodes=20, max_actions=4) for _ in range(3)]
    agg = FederatedAggregator(agents)

    rewards = []
    for ep in range(120):
        epsilon = max(0.05, 1.0 - ep / 80)
        ep_rewards = agg.run_episode(env, epsilon=epsilon)
        rewards.append(float(np.mean(ep_rewards)))

    early_avg = float(np.mean(rewards[:10]))
    late_avg = float(np.mean(rewards[-10:]))
    assert late_avg > early_avg, (
        f"Expected late avg ({late_avg:.4f}) > early avg ({early_avg:.4f})"
    )
