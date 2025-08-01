# Monitoring and Observability

This guide covers the comprehensive monitoring and observability setup for Dynamic Graph Fed-RL, including metrics collection, alerting, logging, and distributed tracing.

## Overview

The monitoring stack includes:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Loki**: Log aggregation
- **Jaeger**: Distributed tracing
- **AlertManager**: Alert routing and notifications
- **ELK Stack**: Log analysis and search

## Quick Start

### Start Monitoring Stack

```bash
# Start all monitoring services
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Check service status
docker-compose -f monitoring/docker-compose.monitoring.yml ps

# View logs
docker-compose -f monitoring/docker-compose.monitoring.yml logs -f grafana
```

### Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | - |
| AlertManager | http://localhost:9093 | - |
| Jaeger | http://localhost:16686 | - |
| Kibana | http://localhost:5601 | - |

## Metrics Collection

### Application Metrics

The application exposes metrics on `/metrics` endpoint using Prometheus format:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Training metrics
training_episodes = Counter('training_episodes_total', 'Total training episodes completed')
episode_reward = Histogram('training_episode_reward', 'Reward per episode')
training_loss = Gauge('training_loss', 'Current training loss')

# Federation metrics
active_agents = Gauge('federation_active_agents', 'Number of active federated agents')
communication_latency = Histogram('federation_communication_latency_seconds', 'Communication latency between agents')
parameter_divergence = Gauge('federation_parameter_divergence', 'Parameter divergence between agents')

# System metrics
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
```

### Custom Metrics Example

```python
import time
from prometheus_client import Counter, Histogram, Gauge

class MetricsCollector:
    def __init__(self):
        # Counters - monotonically increasing
        self.requests_total = Counter(
            'http_requests_total', 
            'Total HTTP requests', 
            ['method', 'endpoint', 'status']
        )
        
        # Histograms - for measuring distributions
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint']
        )
        
        # Gauges - for current values
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections'
        )
        
        # Training-specific metrics
        self.agent_rewards = Gauge(
            'agent_reward',
            'Current agent reward',
            ['agent_id']
        )
        
        self.model_parameters = Gauge(
            'model_parameters_count',
            'Number of model parameters'
        )
    
    def record_request(self, method, endpoint, status, duration):
        """Record HTTP request metrics."""
        self.requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status=status
        ).inc()
        
        self.request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    def update_agent_reward(self, agent_id, reward):
        """Update agent reward metric."""
        self.agent_rewards.labels(agent_id=agent_id).set(reward)
    
    def set_active_connections(self, count):
        """Set number of active connections."""
        self.active_connections.set(count)

# Usage example
metrics = MetricsCollector()

# Record a request
start_time = time.time()
# ... handle request ...
duration = time.time() - start_time
metrics.record_request('GET', '/api/train', '200', duration)

# Update agent metrics
metrics.update_agent_reward('agent_1', 0.85)
```

### Infrastructure Metrics

Collected by various exporters:

- **Node Exporter**: System metrics (CPU, memory, disk, network)
- **cAdvisor**: Container metrics
- **Redis Exporter**: Redis performance metrics
- **Postgres Exporter**: Database metrics
- **NVIDIA Exporter**: GPU metrics

## Dashboards

### Main System Dashboard

```json
{
  "dashboard": {
    "title": "Dynamic Graph Fed-RL - System Overview",
    "panels": [
      {
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"dynamic-graph-fed-rl\"}",
            "legendFormat": "Application Status"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "5xx Error Rate"
          }
        ]
      }
    ]
  }
}
```

### Training Dashboard

```json
{
  "dashboard": {
    "title": "Dynamic Graph Fed-RL - Training Metrics",
    "panels": [
      {
        "title": "Training Progress",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(training_episodes_total[5m])",
            "legendFormat": "Episodes/sec"
          }
        ]
      },
      {
        "title": "Episode Rewards",
        "type": "graph",
        "targets": [
          {
            "expr": "training_episode_reward",
            "legendFormat": "Episode Reward"
          }
        ]
      },
      {
        "title": "Agent Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "agent_reward",
            "legendFormat": "Agent {{agent_id}}"
          }
        ]
      },
      {
        "title": "Training Loss",
        "type": "graph",
        "targets": [
          {
            "expr": "training_loss",
            "legendFormat": "Training Loss"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      }
    ]
  }
}
```

### Federation Dashboard

```json
{
  "dashboard": {
    "title": "Dynamic Graph Fed-RL - Federation Metrics",
    "panels": [
      {
        "title": "Active Agents",
        "type": "stat",
        "targets": [
          {
            "expr": "federation_active_agents",
            "legendFormat": "Active Agents"
          }
        ]
      },
      {
        "title": "Communication Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(federation_communication_latency_seconds_bucket[5m]))",
            "legendFormat": "95th percentile latency"
          }
        ]
      },
      {
        "title": "Parameter Divergence",
        "type": "graph",
        "targets": [
          {
            "expr": "federation_parameter_divergence",
            "legendFormat": "Parameter Divergence"
          }
        ]
      },
      {
        "title": "Gossip Messages",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(gossip_messages_sent_total[5m])",
            "legendFormat": "Messages/sec"
          }
        ]
      }
    ]
  }
}
```

## Alerting

### Alert Rules Categories

1. **System Health**: CPU, memory, disk usage
2. **Application Performance**: Response times, error rates
3. **Training Quality**: Convergence, reward trends
4. **Federation Health**: Agent connectivity, communication
5. **Security**: Unauthorized access, anomalies

### Custom Alert Examples

```yaml
# Custom training alert
- alert: PoorModelPerformance
  expr: avg_over_time(training_episode_reward[1h]) < 0.1
  for: 30m
  labels:
    severity: warning
    team: ml
  annotations:
    summary: "Model performance is below expected threshold"
    description: "Average episode reward over 1 hour is {{ $value }}"
    runbook_url: "https://docs.dgfrl.local/runbooks/poor-performance"

# Custom federation alert  
- alert: AgentNetworkPartition
  expr: federation_active_agents < federation_total_agents * 0.7
  for: 5m
  labels:
    severity: critical
    team: ml
  annotations:
    summary: "Significant agent network partition detected"
    description: "Only {{ $value }} out of {{ federation_total_agents }} agents are active"
```

### Alert Routing

Alerts are routed based on:
- **Severity**: Critical alerts go to PagerDuty
- **Team**: ML alerts to ML team, infrastructure to DevOps
- **Component**: Database alerts to database team

## Logging

### Structured Logging Setup

```python
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'agent_id'):
            log_entry['agent_id'] = record.agent_id
            
        return json.dumps(log_entry)

# Configure logger
logger = logging.getLogger('dgfrl')
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Training episode completed", extra={
    'agent_id': 'agent_1',
    'episode': 150,
    'reward': 0.85,
    'duration': 45.2
})
```

### Log Aggregation

Logs are collected by Promtail and sent to Loki:

```yaml
# promtail.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/lib/docker/containers/*/*log
    pipeline_stages:
      - json:
          expressions:
            output: log
            stream: stream
            attrs:
      - json:
          source: attrs
          expressions:
            tag:
      - regex:
          source: tag
          expression: (?P<container_name>(?:[^|]*))\|(?P<image_name>(?:[^|]*))\|(?P<image_id>(?:[^|]*))\|(?P<container_id>(?:[^|]*))
      - timestamp:
          source: time
          format: RFC3339Nano
      - labels:
          stream:
          container_name:
          image_name:
          image_id:
          container_id:
      - output:
          source: output
```

## Distributed Tracing

### Jaeger Setup

Enable tracing in your application:

```python
from jaeger_client import Config
from opentracing.ext import tags
import opentracing

def init_tracer(service_name):
    config = Config(
        config={
            'sampler': {'type': 'const', 'param': 1},
            'logging': True,
        },
        service_name=service_name,
        validate=True,
    )
    return config.initialize_tracer()

tracer = init_tracer('dynamic-graph-fed-rl')

def traced_function():
    with tracer.start_span('training_step') as span:
        span.set_tag(tags.COMPONENT, 'training')
        span.set_tag('agent_id', 'agent_1')
        
        with tracer.start_span('forward_pass', child_of=span) as child_span:
            # Forward pass logic
            child_span.set_tag('batch_size', 256)
            
        with tracer.start_span('backward_pass', child_of=span) as child_span:
            # Backward pass logic
            child_span.set_tag('loss', 0.45)
```

## Performance Monitoring

### SLI/SLO Monitoring

Define Service Level Indicators (SLIs) and Service Level Objectives (SLOs):

```python
# SLI: Request Success Rate
success_rate_sli = """
sum(rate(http_requests_total{status!~"5.."}[5m])) / 
sum(rate(http_requests_total[5m]))
"""

# SLO: 99.9% success rate
success_rate_slo = 0.999

# SLI: Response Time
response_time_sli = """
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket[5m])
)
"""

# SLO: 95th percentile < 200ms
response_time_slo = 0.2

# Error Budget Alerts
- alert: ErrorBudgetExhausted
  expr: (1 - success_rate_sli) > (1 - success_rate_slo) * 2
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Error budget exhausted"
    description: "Current error rate is consuming error budget too quickly"
```

### Capacity Planning

Monitor resource trends for capacity planning:

```promql
# CPU trend over 7 days
predict_linear(
  avg_over_time(cpu_usage_percent[7d])[7d:1h], 
  30 * 24 * 3600
)

# Memory growth rate
increase(memory_usage_bytes[7d]) / 7

# Disk space prediction
predict_linear(
  disk_free_bytes[7d], 
  30 * 24 * 3600
)
```

## Runbooks

### Training Issues

#### Poor Model Performance

**Symptoms**: Low episode rewards, high training loss

**Investigation Steps**:
1. Check training dashboard for reward trends
2. Verify hyperparameters are correct
3. Check data quality metrics
4. Review recent code changes
5. Examine resource utilization (GPU, memory)

**Resolution**:
- Adjust learning rate or other hyperparameters
- Increase training data or improve data quality
- Scale up resources if needed
- Rollback recent changes if necessary

#### Agent Communication Issues

**Symptoms**: High communication latency, agent disconnections

**Investigation Steps**:
1. Check network connectivity between agents
2. Review federation metrics dashboard
3. Examine network policies and firewall rules
4. Check resource utilization on agent nodes
5. Review recent infrastructure changes

**Resolution**:
- Restart affected agents
- Adjust communication timeouts
- Scale network resources
- Update network configuration

### Infrastructure Issues

#### High Resource Usage

**Symptoms**: High CPU/memory alerts

**Investigation Steps**:
1. Identify resource-intensive processes
2. Check for memory leaks or CPU-bound operations
3. Review recent deployments
4. Examine resource allocation and limits

**Resolution**:
- Scale up resources
- Optimize code for better resource usage
- Implement resource limits
- Consider load balancing

## Maintenance

### Regular Maintenance Tasks

```bash
#!/bin/bash
# monitoring/maintenance.sh

# Clean up old metrics data (keep 30 days)
docker exec prometheus \
  promtool tsdb delete-series \
  --match='{__name__=~".+"}' \
  --min-time=$(date -d '30 days ago' +%s)

# Backup Grafana dashboards
curl -s "http://admin:admin123@localhost:3000/api/search" | \
  jq -r '.[] | select(.type=="dash-db") | .uid' | \
  while read uid; do
    curl -s "http://admin:admin123@localhost:3000/api/dashboards/uid/$uid" \
      > "backups/dashboard-$uid.json"
  done

# Rotate logs
docker exec loki \
  /usr/bin/loki-canary \
  -loki.url=http://localhost:3100 \
  -buckets=1,2,4,8,16,32 \
  -cleanup

# Health check
curl -f http://localhost:9090/-/healthy || echo "Prometheus unhealthy"
curl -f http://localhost:3000/api/health || echo "Grafana unhealthy"
curl -f http://localhost:3100/ready || echo "Loki unhealthy"
```

### Backup and Recovery

```bash
# Backup monitoring data
tar -czf monitoring-backup-$(date +%Y%m%d).tar.gz \
  monitoring/grafana/dashboards/ \
  prometheus-data/ \
  grafana-data/

# Restore from backup
tar -xzf monitoring-backup-20231201.tar.gz
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```

For more detailed information, see:
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Loki Documentation](https://grafana.com/docs/loki/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)