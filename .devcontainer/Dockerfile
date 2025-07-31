# Development container for Dynamic Graph Fed-RL
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        vim \
        htop \
        tree \
        jq \
        unzip \
        wget \
        ca-certificates \
        gnupg \
        lsb-release \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

# Install Terraform
RUN wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | tee /usr/share/keyrings/hashicorp-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/hashicorp.list \
    && apt update && apt install terraform -y

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Switch to vscode user
USER vscode

# Install Python tools
RUN pip install --user --upgrade pip setuptools wheel

# Pre-install common development packages
RUN pip install --user \
    ipython \
    jupyter \
    jupyterlab \
    black \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pre-commit \
    safety \
    bandit \
    locust

# Set up shell
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc \
    && echo 'export PYTHONPATH="/workspaces/dynamic-graph-fed-rl-lab/src:$PYTHONPATH"' >> ~/.bashrc

# Configure git (will be overridden by user settings)
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false

WORKDIR /workspaces/dynamic-graph-fed-rl-lab