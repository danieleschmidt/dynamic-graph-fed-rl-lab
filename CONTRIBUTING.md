# Contributing to Dynamic Graph Federated RL Lab

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.9+
- Git
- CUDA-compatible GPU (optional, for GPU acceleration)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/dynamic-graph-fed-rl-lab.git
cd dynamic-graph-fed-rl-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and quality checks**
   ```bash
   # Run tests
   pytest

   # Check code formatting
   black --check src/ tests/
   ruff check src/ tests/

   # Type checking
   mypy src/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- **Python**: Follow PEP 8, enforced by Black and Ruff
- **Type Hints**: Required for all public APIs
- **Docstrings**: Use Google-style docstrings
- **Line Length**: 88 characters (Black default)

## Testing

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete workflows
- **Coverage**: Maintain >80% test coverage

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
└── fixtures/      # Test data and fixtures
```

## Documentation

- Update relevant documentation for any changes
- Add docstrings to all public functions/classes
- Include examples in docstrings where helpful
- Update README.md if adding major features

## Pull Request Guidelines

### PR Title Format
Use conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `perf:` for performance improvements

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass
- [ ] New tests added
- [ ] Documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
```

## Priority Areas for Contributions

1. **Environments**: New dynamic graph environments
2. **Algorithms**: Novel federated RL algorithms
3. **Models**: Advanced graph neural architectures
4. **Protocols**: Communication-efficient federation methods
5. **Benchmarks**: Evaluation frameworks and metrics
6. **Documentation**: Tutorials and examples

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

- Create an issue for bugs or feature requests
- Discussions for general questions
- Email maintainers for security issues

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes for major contributions
- Project documentation

Thank you for contributing to advancing federated reinforcement learning research!