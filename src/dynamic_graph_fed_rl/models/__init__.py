"""Neural network models for graph RL."""

from .temporal_gnn import MultiScaleTemporalGNN
from .graph_attention import GraphAttentionNetwork

__all__ = ["MultiScaleTemporalGNN", "GraphAttentionNetwork"]