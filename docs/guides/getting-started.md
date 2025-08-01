# Getting Started with Dynamic Graph Fed-RL

Welcome to the Dynamic Graph Federated Reinforcement Learning Laboratory! This guide will help you get up and running with the framework in just a few minutes.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended)
- Docker and Docker Compose
- Git

### Installation

#### Option 1: PyPI Installation (Recommended)

```bash
# Install the stable release
pip install dynamic-graph-fed-rl

# For GPU support
pip install dynamic-graph-fed-rl[gpu]

# For full development environment
pip install dynamic-graph-fed-rl[dev]
```

#### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/dynamic-graph-fed-rl-lab.git
cd dynamic-graph-fed-rl-lab

# Install in development mode
pip install -e ".[dev]"
```

#### Option 3: Docker

```bash
# Pull the pre-built image
docker pull dgfrl/dynamic-graph-fed-rl:latest

# Or build from source
git clone https://github.com/danieleschmidt/dynamic-graph-fed-rl-lab.git
cd dynamic-graph-fed-rl-lab
docker build -t dgfrl/dynamic-graph-fed-rl .
```

### Verify Installation

```bash
# Test the installation
python -c "import dynamic_graph_fed_rl; print('Installation successful!')"

# Check GPU support (if available)
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
```

## Your First Experiment

Let's run a simple traffic control experiment with federated learning:

```python
import jax
import numpy as np
from dynamic_graph_fed_rl import DynamicGraphEnv, FederatedActorCritic
from dynamic_graph_fed_rl.algorithms import GraphTD3

# Create a simple traffic network
env = DynamicGraphEnv(
    scenario="small_city_traffic",
    num_intersections=20,
    time_varying_topology=True,
    render_mode="human"  # Optional: visualize the environment
)

# Set up federated learning with 4 agents
fed_system = FederatedActorCritic(
    num_agents=4,
    communication="sync_averaging",  # Start with synchronous for simplicity
    aggregation_interval=50
)

# Create and initialize agents
agents = []
for i in range(fed_system.num_agents):
    agent = GraphTD3(
        state_dim=env.node_feature_dim,
        action_dim=env.action_dim,
        edge_dim=env.edge_feature_dim,
        learning_rate=3e-4,
        buffer_size=10000
    )
    agents.append(agent)

print("Starting federated training...")

# Training loop
total_episodes = 100
for episode in range(total_episodes):
    graph_state = env.reset()
    episode_reward = 0
    
    for step in range(env.max_episode_steps):
        # Each agent selects actions for their assigned intersections
        actions = {}
        for agent_id, agent in enumerate(agents):
            # Get local observation for this agent
            local_graph = env.get_agent_observation(agent_id)
            action = agent.select_action(local_graph)
            actions[agent_id] = action
        
        # Environment step
        next_graph_state, rewards, done, info = env.step(actions)
        
        # Store experiences in agent buffers
        for agent_id, agent in enumerate(agents):
            agent.store_transition(
                state=env.get_agent_observation(agent_id, graph_state),
                action=actions[agent_id],
                reward=rewards[agent_id],
                next_state=env.get_agent_observation(agent_id, next_graph_state),
                done=done
            )
        
        # Train agents
        if step % 10 == 0:  # Train every 10 steps
            for agent in agents:
                if len(agent.buffer) > 1000:  # Ensure sufficient data
                    agent.train()
        
        # Federated parameter sharing
        if step % fed_system.aggregation_interval == 0:
            fed_system.aggregate_parameters(agents)
        
        graph_state = next_graph_state
        episode_reward += sum(rewards.values())
        
        if done:
            break
    
    # Log progress
    if episode % 10 == 0:
        avg_reward = episode_reward / len(agents)
        print(f"Episode {episode}: Average reward = {avg_reward:.2f}")

print("Training completed!")

# Evaluate the trained system
evaluation_rewards = []
for eval_episode in range(10):
    graph_state = env.reset()
    episode_reward = 0
    
    for step in range(env.max_episode_steps):
        actions = {}
        for agent_id, agent in enumerate(agents):
            local_graph = env.get_agent_observation(agent_id)
            action = agent.select_action(local_graph, explore=False)  # No exploration
            actions[agent_id] = action
        
        next_graph_state, rewards, done, info = env.step(actions)
        episode_reward += sum(rewards.values())
        graph_state = next_graph_state
        
        if done:
            break
    
    evaluation_rewards.append(episode_reward)

avg_eval_reward = np.mean(evaluation_rewards)
std_eval_reward = np.std(evaluation_rewards)

print(f"Evaluation Results:")
print(f"Average reward: {avg_eval_reward:.2f} ± {std_eval_reward:.2f}")
print(f"Traffic flow improvement: {info.get('flow_improvement', 0):.1%}")
```

## Understanding the Code

### Key Components

1. **DynamicGraphEnv**: The environment simulator that models changing network topologies
2. **FederatedActorCritic**: Coordinates parameter sharing between agents
3. **GraphTD3**: The reinforcement learning algorithm adapted for graph structures
4. **Agent Observations**: Local views of the global graph state

### Configuration Options

```python
# Environment configuration
env_config = {
    "scenario": "rush_hour_dynamics",  # Pre-defined scenarios
    "num_intersections": 100,          # Scale of the network
    "time_varying_topology": True,     # Enable dynamic changes
    "edge_failure_rate": 0.01,         # Random link failures
    "demand_pattern": "realistic"      # Traffic demand model
}

# Federated learning configuration
fed_config = {
    "communication": "async_gossip",   # Communication protocol
    "aggregation_interval": 100,       # Steps between sharing
    "compression": "top_k",            # Gradient compression
    "byzantine_robustness": True       # Handle malicious agents
}

# Algorithm configuration
algo_config = {
    "learning_rate": 3e-4,             # Learning rate
    "buffer_size": 100000,             # Experience replay size
    "batch_size": 256,                 # Training batch size
    "target_update_freq": 100,         # Target network updates
    "exploration_noise": 0.1           # Action exploration noise
}
```

## Next Steps

### Tutorials
1. [Dynamic Graphs in RL](../tutorials/01_dynamic_graphs.md) - Understanding time-varying graphs
2. [Federated Learning Setup](../tutorials/02_federation.md) - Advanced federation strategies
3. [Custom Environments](../tutorials/03_custom_environments.md) - Creating your own scenarios
4. [Real-World Deployment](../tutorials/04_deployment.md) - Production deployment guide

### Example Applications
- **Smart Traffic Control**: City-wide traffic optimization
- **Power Grid Management**: Renewable energy integration
- **Supply Chain Optimization**: Multi-tier logistics coordination
- **Telecommunications**: Dynamic bandwidth allocation

### Advanced Features
- **Multi-scale Temporal Modeling**: Different time horizons
- **Hierarchical Federation**: Multi-level agent organization
- **Privacy-Preserving Learning**: Differential privacy support
- **Real-time Monitoring**: Grafana dashboard integration

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or buffer size
export DGFRL_BATCH_SIZE=128
export DGFRL_BUFFER_SIZE=50000
```

**Slow Training**
```bash
# Enable XLA compilation for JAX
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
```

**Connection Issues**
```bash
# Check network connectivity for distributed training
python -m dynamic_graph_fed_rl.utils.network_test
```

### Performance Optimization

```python
# Use mixed precision training
import jax
jax.config.update('jax_enable_x64', False)

# Enable memory pre-allocation
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Use multiple GPUs
from dynamic_graph_fed_rl.distributed import MultiGPUTrainer
trainer = MultiGPUTrainer(num_devices=4)
```

## Getting Help

- **Documentation**: [https://dynamic-graph-fed-rl.readthedocs.io](https://dynamic-graph-fed-rl.readthedocs.io)
- **GitHub Issues**: [Report bugs and request features](https://github.com/danieleschmidt/dynamic-graph-fed-rl-lab/issues)
- **Community Forum**: [Join discussions](https://github.com/danieleschmidt/dynamic-graph-fed-rl-lab/discussions)
- **Discord**: [Real-time chat support](https://discord.gg/dgfrl)

## Contributing

We welcome contributions! See our [Contributing Guide](../../CONTRIBUTING.md) for:
- Code contribution guidelines
- Development environment setup
- Testing and documentation standards
- Community guidelines

Happy learning with Dynamic Graph Fed-RL! 🚀