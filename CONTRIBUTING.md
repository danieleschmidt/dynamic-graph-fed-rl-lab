# Contributing to Dynamic Graph Fed-RL Lab

We welcome contributions to the Dynamic Graph Federated Reinforcement Learning framework! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/dynamic-graph-fed-rl-lab.git
   cd dynamic-graph-fed-rl-lab
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🔧 Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test**
   ```bash
   # Run tests
   pytest tests/
   
   # Run linting
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Contribution Areas

- **Algorithms**: New RL algorithms for dynamic graphs
- **Environments**: Additional simulation environments
- **Federation**: Communication protocols and aggregation methods
- **Performance**: Optimization and scalability improvements
- **Documentation**: Tutorials, examples, and API docs
- **Testing**: Unit tests, integration tests, benchmarks

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📝 Code Style

- Follow PEP 8 with line length of 88 characters
- Use Black for code formatting
- Include type hints for all public functions
- Write docstrings in Google style

## 📚 Documentation

- Update docstrings for new/modified functions
- Add examples to demonstrate usage
- Update README.md if needed

## 🐛 Bug Reports

Use GitHub Issues with:
- Clear description of the problem
- Minimal reproduction example
- Environment details (Python version, dependencies)

## 💡 Feature Requests

Open an issue with:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach

Thank you for contributing to Dynamic Graph Fed-RL Lab!