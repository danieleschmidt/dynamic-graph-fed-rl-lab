"""
Dynamic Graph Federated Reinforcement Learning Lab

A cutting-edge federated reinforcement learning framework for controlling 
systems with time-evolving graph structures.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

from . import algorithms
from . import environments
from . import federation
from . import models
from . import utils

__all__ = [
    "algorithms",
    "environments", 
    "federation",
    "models",
    "utils",
]