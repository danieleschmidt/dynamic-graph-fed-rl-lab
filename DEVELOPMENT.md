# Development Guide

This guide covers the development setup and workflow for Dynamic Graph Fed-RL Lab.

## 🔧 Environment Setup

### Prerequisites
- Python 3.9+ 
- Git
- CUDA-compatible GPU (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dynamic-graph-fed-rl-lab.git
   cd dynamic-graph-fed-rl-lab
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate    # Windows
   ```

3. **Install in development mode**
   ```bash
   # Basic installation
   pip install -e \".[dev]\"
   
   # With GPU support
   pip install -e \".[dev,gpu]\"
   
   # With distributed training support
   pip install -e \".[dev,distributed]\"
   ```

4. **Setup pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🏗️ Project Structure

```
src/dynamic_graph_fed_rl/
├── algorithms/          # RL algorithms (TD3, SAC, etc.)
├── environments/        # Simulation environments
├── buffers/            # Replay buffers for graph data
├── federation/         # Federated learning coordination
├── models/             # Neural network architectures
├── augmentation/       # Graph augmentation techniques
├── profiling/          # Performance monitoring
└── benchmarks/         # Evaluation framework

tests/
├── unit/               # Unit tests
├── integration/        # Integration tests
└── benchmarks/         # Performance benchmarks

docs/
├── tutorials/          # Step-by-step guides
└── api/               # API documentation
```

## 🧪 Testing

### Running Tests
```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_algorithms.py
```

### Adding Tests
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Follow naming convention: `test_*.py`
- Use descriptive test names: `test_graph_td3_learns_traffic_control`

## 🎯 Code Quality

### Linting and Formatting
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/

# Sort imports
isort src/ tests/
```

### Pre-commit Hooks
Automatically run on commit:
- Black (code formatting)
- Flake8 (style checking)
- MyPy (type checking)
- YAML validation

## 📊 Benchmarking

### Running Benchmarks
```bash
# Standard benchmarks
python scripts/run_benchmarks.py

# Specific environment
python scripts/run_benchmarks.py --env traffic_network

# Performance profiling
python scripts/profile_training.py
```

## 🔍 Debugging

### Common Issues
1. **JAX GPU not found**: Install `jax[cuda12_pip]` for CUDA support
2. **Import errors**: Ensure package installed with `pip install -e .`
3. **Test failures**: Check virtual environment is activated

### Debugging Tools
- Use `breakpoint()` for interactive debugging
- Enable JAX debugging: `export JAX_DEBUG_NANS=True`
- Profile with: `python -m cProfile script.py`

## 📈 Performance Optimization

### JAX Best Practices
- Use `jax.jit` for performance-critical functions
- Prefer JAX arrays over NumPy arrays
- Use `vmap` for batch operations
- Avoid Python loops in hot paths

### Memory Management
- Monitor GPU memory with `nvidia-smi`
- Use gradient checkpointing for large models
- Clear JAX cache: `jax.clear_backends()`

## 🐛 Troubleshooting

### Common Solutions
- **Slow imports**: Use `python -O` to disable assertions
- **CUDA out of memory**: Reduce batch size or use gradient accumulation
- **Numerical instability**: Enable mixed precision training

For additional help, check the [troubleshooting guide](docs/troubleshooting.md) or open an issue.