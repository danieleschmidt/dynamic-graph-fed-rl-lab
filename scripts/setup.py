#!/usr/bin/env python3
"""
Development environment setup script.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.9+."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def main():
    """Main setup function."""
    print("🚀 Setting up Dynamic Graph Federated RL Lab development environment\n")
    
    # Check Python version
    check_python_version()
    
    # Get repository root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    # Setup steps
    steps = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -e .[dev]", "Installing package in development mode"),
        ("pre-commit install", "Installing pre-commit hooks"),
    ]
    
    # Optional steps
    optional_steps = [
        ("pip install -e .[gpu]", "Installing GPU dependencies (optional)"),
        ("pip install -e .[monitoring]", "Installing monitoring dependencies (optional)"),
    ]
    
    # Run required steps
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"\n❌ Setup failed at: {desc}")
            sys.exit(1)
    
    print("\n🎯 Optional installations:")
    for cmd, desc in optional_steps:
        choice = input(f"Install {desc.lower()}? (y/N): ").lower().strip()
        if choice in ['y', 'yes']:
            run_command(cmd, desc)
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    verification_steps = [
        ("python -c 'import dynamic_graph_fed_rl; print(f\"Version: {dynamic_graph_fed_rl.__version__}\")'", "Package import"),
        ("pytest --version", "pytest"),
        ("black --version", "black"),
        ("ruff --version", "ruff"),
        ("mypy --version", "mypy"),
    ]
    
    for cmd, desc in verification_steps:
        run_command(cmd, f"Verifying {desc}")
    
    print("\n🎉 Development environment setup complete!")
    print("\n📋 Next steps:")
    print("1. Run tests: make test")
    print("2. Check code quality: make lint")
    print("3. Format code: make format")
    print("4. See all commands: make help")
    print("\n📖 Documentation: docs/DEVELOPMENT.md")


if __name__ == "__main__":
    main()