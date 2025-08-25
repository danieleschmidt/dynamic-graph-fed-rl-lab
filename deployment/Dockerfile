# Production Dockerfile for Universal Quantum Consciousness
FROM python:3.9-slim

# Set production environment
ENV ENVIRONMENT=production
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY deployment/production.env .env
COPY README.md .
COPY SECURITY.md .
COPY LICENSE .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash consciousness
RUN chown -R consciousness:consciousness /app
USER consciousness

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from dynamic_graph_fed_rl.consciousness.universal_quantum_consciousness import UniversalQuantumConsciousness; print('Healthy')"

# Expose port
EXPOSE 8000

# Production startup command
CMD ["python", "-m", "dynamic_graph_fed_rl.consciousness.universal_quantum_consciousness"]
