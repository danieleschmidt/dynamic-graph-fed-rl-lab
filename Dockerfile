# Multi-stage build for efficient container images
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements first for better caching
COPY --chown=app:app requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --user -r requirements.txt && \
    pip install --user -e .

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --user -e ".[dev,docs,monitoring]"

# Copy source code
COPY --chown=app:app . .

# Install pre-commit hooks
RUN git config --global --add safe.directory /home/app && \
    pre-commit install 2>/dev/null || true

EXPOSE 8000
CMD ["python", "-m", "dynamic_graph_fed_rl.server"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=app:app src/ ./src/
COPY --chown=app:app README.md LICENSE ./

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "-m", "dynamic_graph_fed_rl.server", "--production"]

# GPU-enabled stage for training
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

COPY --chown=app:app requirements.txt pyproject.toml ./
RUN pip3.11 install --user -r requirements.txt && \
    pip3.11 install --user -e ".[gpu,distributed]"

COPY --chown=app:app src/ ./src/

CMD ["python3.11", "-m", "dynamic_graph_fed_rl.training"]