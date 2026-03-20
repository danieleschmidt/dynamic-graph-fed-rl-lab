# dynamic-graph-fed-rl-lab

Federated Reinforcement Learning for time-evolving graph structures.

## What it does

Three agents independently learn to navigate a graph that changes every episode (nodes and edges are added or removed). After each episode, their Q-tables are averaged via **Federated Averaging** — no raw observations are shared, only model weights.

## Architecture

```
fedrl/
├── env.py          # GraphEnv  — ring graph that mutates each episode
├── agent.py        # FedAgent  — tabular Q-learning with ε-greedy exploration
├── aggregator.py   # FederatedAggregator — fedavg coordinator
└── utils.py        # epsilon_greedy, moving_average helpers
```

### GraphEnv

- **Initial topology:** 5-node ring `0 → 1 → 2 → 3 → 4 → 0`
- **Goal:** reach node 4 (reward +1); all other steps cost −0.01
- **Dynamic changes:** each `reset()` either adds a new node+edge or removes a non-critical edge
- **Episode length:** up to 50 steps

### FedAgent

- Q-table: `(max_nodes=20, max_actions=4)` zeros
- Update: `Q[s,a] += lr * (r + γ · max Q[s'] · (1−done) − Q[s,a])`
- Defaults: `lr=0.1`, `γ=0.95`

### FederatedAggregator

- Runs one episode per agent, then broadcasts the mean Q-table back to all agents

## Quick start

```bash
pip install -r requirements.txt
python demo.py
```

## Tests

```bash
pytest tests/ -v
```

Requires Python ≥ 3.10 and NumPy.

## License

MIT
