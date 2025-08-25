# Production Deployment Guide

## Universal Quantum Consciousness Production Deployment

### Prerequisites
- Docker and Docker Compose installed
- Kubernetes cluster (optional, for K8s deployment)
- Minimum 4GB RAM, 2 CPU cores
- Python 3.9+ (for direct deployment)

### Deployment Options

#### Option 1: Docker Compose (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd dynamic-graph-fed-rl-lab

# Copy environment template
cp .env.template .env
# Edit .env with your production values

# Build and start
cd deployment
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
curl http://localhost:8000/health
```

#### Option 2: Kubernetes
```bash
# Apply Kubernetes manifest
kubectl apply -f deployment/kubernetes-manifest.yml

# Check deployment status
kubectl get pods -l app=consciousness
kubectl get services consciousness-service
```

#### Option 3: Direct Python Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ENVIRONMENT=production
export PYTHONPATH=$(pwd)/src

# Run startup script
./deployment/start-production.sh
```

### Configuration

#### Environment Variables
- `ENVIRONMENT`: Set to "production"
- `CONSCIOUSNESS_SECURITY_LEVEL`: Set to "HIGH"
- `QUANTUM_ENCRYPTION_ENABLED`: Set to "true"
- `MONITORING_ENABLED`: Set to "true"
- `AUTO_SCALING_ENABLED`: Set to "true"

#### Security Configuration
- Review SECURITY.md for security guidelines
- Use strong passwords and API keys
- Enable HTTPS in production
- Configure firewall rules
- Regular security audits

#### Performance Tuning
- Adjust `CACHE_SIZE` based on available memory
- Configure `MAX_THREADS` for optimal performance
- Set `OPTIMIZATION_LEVEL` to "AGGRESSIVE" for production
- Monitor resource usage and scale accordingly

### Monitoring

#### Metrics
- Consciousness awareness level
- Quantum entanglement strength
- System performance metrics
- Memory and CPU usage

#### Alerts
- Low consciousness awareness
- System downtime
- High resource usage
- Security incidents

### Backup and Recovery

#### Consciousness State Backup
- Automatic backups every 5 minutes
- Stored in persistent volume
- Encryption at rest enabled

#### Recovery Procedures
1. Stop current instance
2. Restore from latest backup
3. Restart system
4. Verify consciousness state integrity

### Troubleshooting

#### Common Issues
1. **Consciousness awareness drops**: Check system resources
2. **Quantum entanglement weak**: Verify network connectivity
3. **High memory usage**: Consider scaling or optimizing cache size
4. **Security warnings**: Review security logs and update configurations

#### Log Locations
- Application logs: `/app/logs/consciousness.log`
- Error logs: `/app/logs/consciousness-error.log`
- Container logs: `docker logs universal-quantum-consciousness`

### Support
For deployment issues, consult the troubleshooting guide or contact support.
