"""Replay buffers for graph-based reinforcement learning."""

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from ..environments.base import GraphState, GraphTransition


# Remove duplicate GraphTransition - already defined in base.py


class GraphReplayBuffer:
    """Standard replay buffer for graph transitions."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def add(self, transition: GraphTransition) -> None:
        """Add transition to buffer."""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[GraphTransition]:
        """Sample random batch of transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} transitions, need {batch_size}")
        
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.position = 0


class GraphTemporalBuffer:
    """Temporal replay buffer for sequences of graph transitions."""
    
    def __init__(
        self,
        capacity: int,
        sequence_length: int = 10,
        overlap: int = 5,
    ):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.overlap = overlap
        
        # Store transitions in episodes
        self.episodes: List[List[GraphTransition]] = []
        self.current_episode: List[GraphTransition] = []
        
        # Storage for graph topology hashes
        self.topology_hashes: List[str] = []
        
        # Temporal indexing
        self.temporal_index = {}
        
    def add(self, transition: GraphTransition) -> None:
        """Add transition to current episode."""
        self.current_episode.append(transition)
        
        # Store topology hash for tracking changes
        topology_hash = self._compute_topology_hash(transition.state)
        self.topology_hashes.append(topology_hash)
        
        # End episode if done
        if transition.done:
            self.end_episode()
    
    def end_episode(self) -> None:
        """End current episode and start new one."""
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self._update_temporal_index()
            
            # Remove old episodes if capacity exceeded
            while len(self.episodes) > self.capacity:
                self.episodes.pop(0)
            
            self.current_episode = []
    
    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: Optional[int] = None,
    ) -> List[List[GraphTransition]]:
        """Sample sequences of transitions."""
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        sequences = []
        
        for _ in range(batch_size):
            # Choose random episode
            if not self.episodes:
                continue
            
            episode = random.choice(self.episodes)
            
            # Choose random starting point
            if len(episode) >= sequence_length:
                start_idx = random.randint(0, len(episode) - sequence_length)
                sequence = episode[start_idx:start_idx + sequence_length]
                sequences.append(sequence)
        
        return sequences
    
    def sample_temporal(
        self,
        batch_size: int,
        lookback: int = 5,
        respect_topology_changes: bool = True,
    ) -> List[List[GraphTransition]]:
        """Sample temporal sequences respecting topology changes."""
        sequences = []
        
        for _ in range(batch_size):
            if not self.episodes:
                continue
            
            episode = random.choice(self.episodes)
            
            if len(episode) < lookback:
                continue
            
            # Random starting point
            start_idx = random.randint(lookback, len(episode))
            
            # Build sequence backwards
            sequence = []
            current_idx = start_idx
            
            for step in range(lookback):
                if current_idx - step >= 0:
                    transition = episode[current_idx - step]
                    
                    # Check topology consistency
                    if respect_topology_changes and step > 0:
                        current_hash = self._compute_topology_hash(transition.state)
                        prev_hash = self._compute_topology_hash(sequence[-1].state)
                        
                        # Break sequence if topology changed significantly
                        if current_hash != prev_hash:
                            break
                    
                    sequence.append(transition)
                else:
                    break
            
            # Reverse to get chronological order
            sequence.reverse()
            
            if len(sequence) >= 2:  # Minimum useful sequence
                sequences.append(sequence)
        
        return sequences
    
    def get_topology_consistent_batch(
        self,
        batch_size: int,
        max_topology_changes: int = 1,
    ) -> List[GraphTransition]:
        """Sample batch with limited topology changes."""
        consistent_transitions = []
        
        for episode in self.episodes:
            if not episode:
                continue
            
            # Find segments with consistent topology
            current_segment = []
            topology_changes = 0
            current_topology = None
            
            for transition in episode:
                topology_hash = self._compute_topology_hash(transition.state)
                
                if current_topology is None:
                    current_topology = topology_hash
                    current_segment = [transition]
                elif topology_hash == current_topology:
                    current_segment.append(transition)
                else:
                    # Topology change detected
                    if topology_changes < max_topology_changes:
                        topology_changes += 1
                        current_topology = topology_hash
                        current_segment.append(transition)
                    else:
                        # Add current segment and start new one
                        consistent_transitions.extend(current_segment)
                        current_segment = [transition]
                        topology_changes = 0
                        current_topology = topology_hash
            
            # Add final segment
            consistent_transitions.extend(current_segment)
        
        # Sample from consistent transitions
        if len(consistent_transitions) >= batch_size:
            return random.sample(consistent_transitions, batch_size)
        else:
            return consistent_transitions
    
    def _compute_topology_hash(self, state: GraphState) -> str:
        """Compute hash of graph topology."""
        # Sort edges for consistent hashing
        edges = state.edge_index.T
        sorted_edges = edges[jnp.lexsort((edges[:, 1], edges[:, 0]))]
        
        # Create hash from sorted edge list
        edge_str = str(sorted_edges.tolist())
        return str(hash(edge_str))
    
    def _update_temporal_index(self) -> None:
        """Update temporal index for efficient querying."""
        if not self.episodes:
            return
        
        latest_episode = self.episodes[-1]
        episode_idx = len(self.episodes) - 1
        
        for step_idx, transition in enumerate(latest_episode):
            timestamp = transition.state.timestamp
            topology_hash = self._compute_topology_hash(transition.state)
            
            if timestamp not in self.temporal_index:
                self.temporal_index[timestamp] = []
            
            self.temporal_index[timestamp].append({
                "episode_idx": episode_idx,
                "step_idx": step_idx,
                "topology_hash": topology_hash,
            })
    
    def get_transitions_by_time(
        self,
        start_time: float,
        end_time: float,
    ) -> List[GraphTransition]:
        """Get transitions within time range."""
        transitions = []
        
        for timestamp, entries in self.temporal_index.items():
            if start_time <= timestamp <= end_time:
                for entry in entries:
                    episode_idx = entry["episode_idx"]
                    step_idx = entry["step_idx"]
                    
                    if episode_idx < len(self.episodes):
                        episode = self.episodes[episode_idx]
                        if step_idx < len(episode):
                            transitions.append(episode[step_idx])
        
        return transitions
    
    def __len__(self) -> int:
        """Return total number of transitions."""
        total = len(self.current_episode)
        for episode in self.episodes:
            total += len(episode)
        return total
    
    def clear(self) -> None:
        """Clear buffer."""
        self.episodes.clear()
        self.current_episode.clear()
        self.topology_hashes.clear()
        self.temporal_index.clear()


class PrioritizedGraphBuffer:
    """Prioritized experience replay for graph transitions."""
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.frame_count = 0
        
        # Sum tree for efficient sampling
        self._max_priority = 1.0
    
    def add(self, transition: GraphTransition) -> None:
        """Add transition with maximum priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(self._max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = self._max_priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[List[GraphTransition], jnp.ndarray, jnp.ndarray]:
        """Sample batch with priorities."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} transitions, need {batch_size}")
        
        # Compute sampling probabilities
        priorities = jnp.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities = probabilities / jnp.sum(probabilities)
        
        # Sample indices (using numpy for compatibility)
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probabilities,
        )
        indices = jnp.array(indices)
        
        # Get transitions
        transitions = [self.buffer[int(idx)] for idx in indices]
        
        # Compute importance sampling weights
        beta = self._compute_beta()
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / jnp.max(weights)  # Normalize
        
        return transitions, indices, weights
    
    def update_priorities(self, indices: jnp.ndarray, priorities: jnp.ndarray) -> None:
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            idx = int(idx)
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = float(priority)
                self._max_priority = max(self._max_priority, float(priority))
    
    def _compute_beta(self) -> float:
        """Compute current beta value."""
        progress = min(1.0, self.frame_count / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)


def collate_graph_transitions(transitions: List[GraphTransition]) -> Dict[str, jnp.ndarray]:
    """Collate list of graph transitions into batched arrays."""
    if not transitions:
        return {}
    
    # Determine maximum graph sizes
    max_nodes = max(t.state.num_nodes for t in transitions)
    max_edges = max(t.state.num_edges for t in transitions)
    
    batch_size = len(transitions)
    node_dim = transitions[0].state.node_dim
    edge_dim = transitions[0].state.edge_dim if transitions[0].state.edge_features is not None else 0
    
    # Initialize batched arrays
    states = jnp.zeros((batch_size, max_nodes, node_dim))
    next_states = jnp.zeros((batch_size, max_nodes, node_dim))
    edge_indices = jnp.zeros((batch_size, 2, max_edges), dtype=int)
    next_edge_indices = jnp.zeros((batch_size, 2, max_edges), dtype=int)
    
    if edge_dim > 0:
        edge_features = jnp.zeros((batch_size, max_edges, edge_dim))
        next_edge_features = jnp.zeros((batch_size, max_edges, edge_dim))
    else:
        edge_features = None
        next_edge_features = None
    
    actions = []
    rewards = []
    dones = []
    
    # Fill batched arrays
    for i, transition in enumerate(transitions):
        # States
        state = transition.state
        next_state = transition.next_state
        
        states = states.at[i, :state.num_nodes, :].set(state.node_features)
        next_states = next_states.at[i, :next_state.num_nodes, :].set(next_state.node_features)
        
        # Edge indices (pad with -1 for unused edges)
        edge_indices = edge_indices.at[i, :, :state.num_edges].set(state.edge_index)
        next_edge_indices = next_edge_indices.at[i, :, :next_state.num_edges].set(next_state.edge_index)
        
        # Edge features
        if edge_features is not None and state.edge_features is not None:
            edge_features = edge_features.at[i, :state.num_edges, :].set(state.edge_features)
        if next_edge_features is not None and next_state.edge_features is not None:
            next_edge_features = next_edge_features.at[i, :next_state.num_edges, :].set(next_state.edge_features)
        
        # Actions, rewards, dones
        if isinstance(transition.action, dict):
            # Multi-agent case - convert to array
            action_array = jnp.zeros(max_nodes)
            for agent_id, action in transition.action.items():
                if agent_id < max_nodes:
                    action_array = action_array.at[agent_id].set(action)
            actions.append(action_array)
        else:
            actions.append(transition.action)
        
        if isinstance(transition.reward, dict):
            # Multi-agent case - use sum or mean
            total_reward = sum(transition.reward.values())
            rewards.append(total_reward)
        else:
            rewards.append(transition.reward)
        
        dones.append(float(transition.done))
    
    # Convert lists to arrays
    actions = jnp.array(actions)
    rewards = jnp.array(rewards)
    dones = jnp.array(dones)
    
    batch = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "edge_indices": edge_indices,
        "next_edge_indices": next_edge_indices,
    }
    
    if edge_features is not None:
        batch["edge_features"] = edge_features
        batch["next_edge_features"] = next_edge_features
    
    return batch