#!/bin/bash
set -e

echo "🚀 Setting up Dynamic Graph Fed-RL development environment..."

# Update package lists
sudo apt-get update

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    htop \
    tree \
    jq \
    vim

# Install Python development tools
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip setuptools wheel

# Install the project in development mode
echo "🔧 Installing project dependencies..."
if [ -f "pyproject.toml" ]; then
    pip install -e ".[dev,docs,monitoring,distributed]"
else
    echo "⚠️  pyproject.toml not found, installing basic dependencies..."
    pip install -r requirements.txt
fi

# Install pre-commit hooks
echo "🔐 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not available, skipping hooks setup"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p \
    logs \
    data \
    models \
    experiments \
    .cache

# Set up Git configuration (if not already configured)
if [ -z "$(git config --global user.name)" ]; then
    echo "📝 Setting up Git configuration..."
    git config --global user.name "Dev Container User"
    git config --global user.email "dev@example.com"
    git config --global init.defaultBranch main
    git config --global pull.rebase false
fi

# Install Jupyter extensions
echo "📊 Setting up Jupyter extensions..."
if command -v jupyter &> /dev/null; then
    jupyter lab build
    echo "✅ Jupyter Lab built successfully"
fi

# Set up environment variables
echo "🌍 Setting up environment variables..."
cat >> ~/.bashrc << EOF

# Dynamic Graph Fed-RL environment
export PYTHONPATH="/workspaces/dynamic-graph-fed-rl-lab/src:\$PYTHONPATH"
export DGFRL_ENV="development"
export JAX_PLATFORM_NAME="cpu"
export WANDB_MODE="offline"

# JAX configuration for development
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Useful aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Project-specific aliases
alias dgfrl-test='python -m pytest tests/ -v'
alias dgfrl-lint='ruff check src/ tests/ && black --check src/ tests/'
alias dgfrl-format='black src/ tests/ && ruff --fix src/ tests/'
alias dgfrl-docs='cd docs && make html'
alias dgfrl-serve='uvicorn src.dynamic_graph_fed_rl.server.app:app --reload --host 0.0.0.0 --port 8000'

EOF

# Create a welcome message
cat > ~/.welcome << 'EOF'

🌟 Welcome to Dynamic Graph Fed-RL Development Environment! 🌟

Quick Start:
  dgfrl-test     - Run tests
  dgfrl-lint     - Check code quality
  dgfrl-format   - Format code
  dgfrl-docs     - Build documentation
  dgfrl-serve    - Start API server

Useful commands:
  make dev-setup - Full development setup
  make test      - Run test suite
  make lint      - Lint and format code
  make docs      - Build documentation
  make clean     - Clean build artifacts

Monitoring:
  - Grafana:    http://localhost:3000
  - Prometheus: http://localhost:9090
  - Jupyter:    http://localhost:8888
  - API Docs:   http://localhost:8000/docs

Happy coding! 🚀

EOF

# Add welcome message to shell
echo 'cat ~/.welcome' >> ~/.bashrc

# Create useful development scripts
mkdir -p ~/bin

# Script for quick testing
cat > ~/bin/quick-test << 'EOF'
#!/bin/bash
cd /workspaces/dynamic-graph-fed-rl-lab
echo "🧪 Running quick tests..."
python -m pytest tests/unit/ -x -v --tb=short
EOF

# Script for environment check
cat > ~/bin/env-check << 'EOF'
#!/bin/bash
echo "🔍 Environment Check:"
echo "Python: $(python --version)"
echo "JAX: $(python -c 'import jax; print(jax.__version__)')"
echo "JAX devices: $(python -c 'import jax; print(jax.devices())')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Working directory: $(pwd)"
echo "Git branch: $(git branch --show-current 2>/dev/null || echo 'N/A')"
echo "Git status: $(git status --porcelain | wc -l) modified file(s)"
EOF

chmod +x ~/bin/*
export PATH="$HOME/bin:$PATH"

# Verify installation
echo "🔍 Verifying installation..."

# Check Python imports
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import jax
    print(f'✅ JAX {jax.__version__} installed')
    print(f'JAX devices: {jax.devices()}')
except ImportError:
    print('❌ JAX not available')

try:
    import torch
    print(f'✅ PyTorch {torch.__version__} installed')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('❌ PyTorch not available')

try:
    import networkx as nx
    print(f'✅ NetworkX {nx.__version__} installed')
except ImportError:
    print('❌ NetworkX not available')
"

echo ""
echo "✅ Development environment setup complete!"
echo "🔧 Run 'source ~/.bashrc' to refresh your shell"
echo "📖 See ~/.welcome for quick start guide"
echo ""