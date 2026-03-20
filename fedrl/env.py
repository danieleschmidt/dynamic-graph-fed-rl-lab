"""GraphEnv: a graph environment with time-evolving topology."""

import random
import numpy as np


class GraphEnv:
    """
    A graph navigation environment with dynamic topology changes.

    The agent starts at node 0 and must reach the goal node (node 4).
    Each episode, the graph changes: either a new node+edge is added,
    or a non-critical edge is removed.

    Args:
        max_nodes (int): Maximum number of nodes the graph can grow to.
        max_actions (int): Maximum degree / number of neighbors per node.
        max_steps (int): Maximum steps per episode.
        goal_node (int): Target node id. Default 4.
        start_node (int): Starting node id. Default 0.
        seed (int | None): Optional random seed for reproducibility.
    """

    def __init__(
        self,
        max_nodes: int = 20,
        max_actions: int = 4,
        max_steps: int = 50,
        goal_node: int = 4,
        start_node: int = 0,
        seed: int | None = None,
    ):
        self.max_nodes = max_nodes
        self.max_actions = max_actions
        self.max_steps = max_steps
        self.goal_node = goal_node
        self.start_node = start_node

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Build initial ring graph: 0-1-2-3-4-0
        self._initial_graph = {
            0: [1, 4],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3, 0],
        }
        self.graph: dict[int, list[int]] = {}
        self._next_node_id: int = 5
        self._current_step: int = 0
        self._current_node: int = start_node

        self._reset_graph()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self.graph)

    @property
    def observation_space_n(self) -> int:
        return self.max_nodes

    @property
    def action_space_n(self) -> int:
        return self.max_actions

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> int:
        """Reset the environment to the start node and apply one graph change."""
        self._apply_graph_change()
        self._current_step = 0
        self._current_node = self.start_node
        return self._current_node

    def step(self, action: int):
        """
        Take an action (move to a neighbor by index).

        Args:
            action: Index into the sorted neighbor list of the current node.
                    Out-of-range actions stay in place.

        Returns:
            (next_state, reward, done, info)
        """
        neighbors = sorted(self.graph.get(self._current_node, []))
        if action < len(neighbors):
            self._current_node = neighbors[action]

        self._current_step += 1

        reached_goal = self._current_node == self.goal_node
        reward = 1.0 if reached_goal else -0.01
        done = reached_goal or (self._current_step >= self.max_steps)

        info = {
            "step": self._current_step,
            "node": self._current_node,
            "num_nodes": self.num_nodes,
        }
        return self._current_node, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_graph(self):
        """Copy the initial graph and reset the node counter."""
        self.graph = {k: list(v) for k, v in self._initial_graph.items()}
        self._next_node_id = max(self.graph.keys()) + 1

    def _apply_graph_change(self):
        """Randomly add a node+edge or remove a non-critical edge."""
        if self._rng.random() < 0.5 and self.num_nodes < self.max_nodes:
            self._add_node()
        else:
            self._remove_edge()

    def _add_node(self):
        """Add a new node connected to a random existing node."""
        new_id = self._next_node_id
        existing = list(self.graph.keys())
        attach_to = self._rng.choice(existing)

        # Respect max_actions degree limit
        if len(self.graph[attach_to]) >= self.max_actions:
            return

        self.graph[new_id] = [attach_to]
        self.graph[attach_to].append(new_id)
        self._next_node_id += 1

    def _remove_edge(self):
        """Remove one edge that does not disconnect the goal or start node critically."""
        # Collect removable edges: neither endpoint is goal/start, and removing won't
        # leave a node isolated.
        candidates = []
        for node, neighbors in self.graph.items():
            if node in (self.goal_node, self.start_node):
                continue
            for nb in neighbors:
                if nb in (self.goal_node, self.start_node):
                    continue
                if len(self.graph[node]) > 1 and len(self.graph[nb]) > 1:
                    edge = tuple(sorted((node, nb)))
                    candidates.append(edge)

        if not candidates:
            return

        u, v = self._rng.choice(candidates)
        if v in self.graph[u]:
            self.graph[u].remove(v)
        if u in self.graph[v]:
            self.graph[v].remove(u)
