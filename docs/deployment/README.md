# Deployment Guide

This guide covers deploying the Dynamic Graph Fed-RL system in various environments, from local development to production Kubernetes clusters.

## Quick Start Deployment

### Local Development

```bash
# 1. Clone and setup
git clone https://github.com/danieleschmidt/dynamic-graph-fed-rl-lab.git
cd dynamic-graph-fed-rl-lab
make dev-setup

# 2. Start services
docker-compose up -d

# 3. Run a test experiment
python examples/quick_start.py
```

### Docker Deployment

```bash
# Build images
make docker-build

# Run production stack
docker-compose -f docker-compose.prod.yml up -d

# Check health
curl http://localhost:8000/health
```

### Kubernetes Deployment

```bash
# Deploy to cluster
kubectl apply -f infrastructure/kubernetes/

# Check status
kubectl get pods -n dgfrl

# Access dashboard
kubectl port-forward svc/dgfrl-web 3000:3000
```

## Container Images

### Available Images

| Image | Purpose | Size | Base |
|-------|---------|------|------|
| `dgfrl:dev` | Development & testing | ~2GB | python:3.11-slim |
| `dgfrl:prod` | Production deployment | ~800MB | python:3.11-slim |
| `dgfrl:gpu` | GPU-accelerated training | ~4GB | nvidia/cuda:12.1 |

### Building Images

```bash
# Build all images
./scripts/build.sh --production --gpu

# Build specific target
docker build --target production -t dgfrl:prod .

# Build with version
./scripts/build.sh --version v1.0.0 --push
```

### Image Security

```bash
# Security scan
make docker-scan

# Generate SBOM
syft dgfrl:prod -o spdx-json

# Vulnerability assessment
trivy image dgfrl:prod
```

## Docker Compose Configurations

### Development Stack (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      target: development
    ports:
      - "8000:8000"
    volumes:
      - .:/home/app
      - pip-cache:/home/app/.cache/pip
    environment:
      - DGFRL_ENV=development
    depends_on:
      - redis
      - prometheus
      - grafana

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning

volumes:
  pip-cache:
```

### Production Stack (`docker-compose.prod.yml`)

```yaml
version: '3.8'

services:
  app:
    image: dgfrl:prod
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    ports:
      - "8000:8000"
    environment:
      - DGFRL_ENV=production
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    image: dgfrl:gpu
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - DGFRL_ENV=production
      - WORKER_TYPE=training
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    deploy:
      resources:
        limits:
          memory: 1G
    volumes:
      - redis-data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=dgfrl
      - POSTGRES_USER=dgfrl
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres-data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app

volumes:
  redis-data:
  postgres-data:

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
```

## Kubernetes Deployment

### Namespace

```yaml
# infrastructure/kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: dgfrl
  labels:
    name: dgfrl
    app.kubernetes.io/name: dynamic-graph-fed-rl
    app.kubernetes.io/version: "1.0.0"
```

### ConfigMap

```yaml
# infrastructure/kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dgfrl-config
  namespace: dgfrl
data:
  app-config.yaml: |
    environment: production
    logging:
      level: INFO
      format: json
    federation:
      protocol: async_gossip
      agents: 20
      aggregation_interval: 100
    training:
      batch_size: 256
      learning_rate: 0.0003
      buffer_size: 1000000
  redis.conf: |
    maxmemory 1gb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
```

### Secret

```yaml
# infrastructure/kubernetes/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: dgfrl-secrets
  namespace: dgfrl
type: Opaque
data:
  postgres-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
  jwt-secret: <base64-encoded-jwt-secret>
  wandb-api-key: <base64-encoded-wandb-key>
```

### Deployment

```yaml
# infrastructure/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dgfrl-api
  namespace: dgfrl
  labels:
    app: dgfrl-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dgfrl-api
  template:
    metadata:
      labels:
        app: dgfrl-api
    spec:
      containers:
      - name: api
        image: ghcr.io/danieleschmidt/dynamic-graph-fed-rl:latest
        ports:
        - containerPort: 8000
        env:
        - name: DGFRL_ENV
          value: "production"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: dgfrl-secrets
              key: postgres-password
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /etc/dgfrl
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: dgfrl-config
```

### GPU Training Deployment

```yaml
# infrastructure/kubernetes/gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dgfrl-training
  namespace: dgfrl
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dgfrl-training
  template:
    metadata:
      labels:
        app: dgfrl-training
    spec:
      containers:
      - name: trainer
        image: ghcr.io/danieleschmidt/dynamic-graph-fed-rl:gpu
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2000m"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4000m"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: WORKER_TYPE
          value: "training"
      nodeSelector:
        kubernetes.io/arch: amd64
        node.kubernetes.io/instance-type: g5.xlarge
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Service

```yaml
# infrastructure/kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: dgfrl-api-service
  namespace: dgfrl
spec:
  selector:
    app: dgfrl-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress

```yaml
# infrastructure/kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dgfrl-ingress
  namespace: dgfrl
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.dgfrl.example.com
    secretName: dgfrl-tls
  rules:
  - host: api.dgfrl.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dgfrl-api-service
            port:
              number: 80
```

## Cloud Deployment

### AWS EKS

```bash
# 1. Create EKS cluster
eksctl create cluster --name dgfrl-cluster --region us-west-2 \
  --nodegroup-name standard-workers --node-type m5.large \
  --nodes 3 --nodes-min 1 --nodes-max 10 --managed

# 2. Create GPU node group
eksctl create nodegroup --cluster dgfrl-cluster \
  --name gpu-workers --node-type g4dn.xlarge \
  --nodes 2 --nodes-min 0 --nodes-max 5 \
  --node-ami-family AmazonLinux2 --managed

# 3. Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml

# 4. Deploy application
kubectl apply -f infrastructure/kubernetes/
```

### Google GKE

```bash
# 1. Create cluster with GPU support
gcloud container clusters create dgfrl-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# 2. Create GPU node pool
gcloud container node-pools create gpu-pool \
  --accelerator type=nvidia-tesla-k80,count=1 \
  --zone us-central1-a \
  --cluster dgfrl-cluster \
  --num-nodes 2 \
  --min-nodes 0 \
  --max-nodes 5 \
  --enable-autoscaling

# 3. Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# 4. Deploy application
kubectl apply -f infrastructure/kubernetes/
```

### Azure AKS

```bash
# 1. Create resource group
az group create --name dgfrl-rg --location eastus

# 2. Create AKS cluster
az aks create \
  --resource-group dgfrl-rg \
  --name dgfrl-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# 3. Add GPU node pool
az aks nodepool add \
  --resource-group dgfrl-rg \
  --cluster-name dgfrl-cluster \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 0 \
  --max-count 5

# 4. Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml

# 5. Deploy application
kubectl apply -f infrastructure/kubernetes/
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'dgfrl-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'dgfrl-training'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Dynamic Graph Fed-RL Monitoring",
    "panels": [
      {
        "title": "Training Progress",
        "targets": [
          {
            "expr": "rate(training_episodes_total[5m])",
            "legendFormat": "Episodes/sec"
          }
        ]
      },
      {
        "title": "Agent Performance",
        "targets": [
          {
            "expr": "avg(agent_reward) by (agent_id)",
            "legendFormat": "Agent {{agent_id}}"
          }
        ]
      },
      {
        "title": "System Resources",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m])",
            "legendFormat": "CPU Usage"
          },
          {
            "expr": "container_memory_usage_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)"
          }
        ]
      }
    ]
  }
}
```

## Scaling and High Availability

### Horizontal Pod Autoscaler

```yaml
# infrastructure/kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dgfrl-api-hpa
  namespace: dgfrl
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dgfrl-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaler

```yaml
# infrastructure/kubernetes/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: dgfrl-api-vpa
  namespace: dgfrl
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dgfrl-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
```

### Pod Disruption Budget

```yaml
# infrastructure/kubernetes/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: dgfrl-api-pdb
  namespace: dgfrl
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: dgfrl-api
```

## Security

### RBAC Configuration

```yaml
# infrastructure/kubernetes/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: dgfrl-service-account
  namespace: dgfrl

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: dgfrl-role
  namespace: dgfrl
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: dgfrl-role-binding
  namespace: dgfrl
subjects:
- kind: ServiceAccount
  name: dgfrl-service-account
  namespace: dgfrl
roleRef:
  kind: Role
  name: dgfrl-role
  apiGroup: rbac.authorization.k8s.io
```

### Network Policies

```yaml
# infrastructure/kubernetes/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: dgfrl-network-policy
  namespace: dgfrl
spec:
  podSelector:
    matchLabels:
      app: dgfrl-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
```

## Backup and Disaster Recovery

### Database Backup

```bash
#!/bin/bash
# scripts/backup-db.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="dgfrl"

# Create backup
kubectl exec -n dgfrl deployment/postgres -- \
  pg_dump -U dgfrl $DB_NAME | \
  gzip > "${BACKUP_DIR}/dgfrl_${DATE}.sql.gz"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/dgfrl_${DATE}.sql.gz" \
  s3://dgfrl-backups/database/

# Cleanup old local backups (keep last 7 days)
find $BACKUP_DIR -name "dgfrl_*.sql.gz" -mtime +7 -delete
```

### Redis Backup

```bash
#!/bin/bash
# scripts/backup-redis.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Create Redis backup
kubectl exec -n dgfrl deployment/redis -- \
  redis-cli BGSAVE

# Wait for backup to complete
kubectl exec -n dgfrl deployment/redis -- \
  redis-cli SAVE

# Copy backup file
kubectl cp dgfrl/redis-pod:/data/dump.rdb \
  "${BACKUP_DIR}/redis_${DATE}.rdb"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/redis_${DATE}.rdb" \
  s3://dgfrl-backups/redis/
```

## Troubleshooting

### Common Issues

#### Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n dgfrl

# Check logs
kubectl logs <pod-name> -n dgfrl --previous

# Check resource constraints
kubectl top nodes
kubectl top pods -n dgfrl
```

#### GPU Not Available

```bash
# Check GPU nodes
kubectl describe nodes | grep nvidia.com/gpu

# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia

# Test GPU access
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:11.8-base-ubuntu20.04 -- nvidia-smi
```

#### Network Connectivity Issues

```bash
# Test service connectivity
kubectl run network-test --rm -it --restart=Never \
  --image=busybox -- nslookup dgfrl-api-service.dgfrl.svc.cluster.local

# Check network policies
kubectl describe networkpolicy -n dgfrl

# Test external connectivity
kubectl run external-test --rm -it --restart=Never \
  --image=curlimages/curl -- curl -I https://api.github.com
```

### Performance Tuning

#### Resource Optimization

```yaml
# Optimized resource requests/limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

#### JVM Tuning (if using Java components)

```bash
export JAVA_OPTS="-Xms2g -Xmx4g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

#### GPU Memory Management

```python
# In Python training code
import jax
jax.config.update('jax_gpu_memory_fraction', 0.8)
```

For additional help:
- **Documentation**: See `/docs` directory
- **Issues**: https://github.com/danieleschmidt/dynamic-graph-fed-rl-lab/issues
- **Discussions**: https://github.com/danieleschmidt/dynamic-graph-fed-rl-lab/discussions
- **Community**: Join our Discord server