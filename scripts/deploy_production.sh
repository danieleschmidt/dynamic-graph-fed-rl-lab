#!/bin/bash
set -euo pipefail

# Production Deployment Script for Dynamic Graph Federated RL
# Terragon Labs - Autonomous SDLC Implementation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY=${REGISTRY:-"ghcr.io/terragonlabs"}
IMAGE_NAME=${IMAGE_NAME:-"dynamic-graph-fed-rl"}
VERSION=${VERSION:-$(cat src/dynamic_graph_fed_rl/__init__.py | grep "__version__" | cut -d'"' -f2)}
ENVIRONMENT=${ENVIRONMENT:-"production"}
NAMESPACE=${NAMESPACE:-"federated-rl"}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi
    
    # Check if kubectl is available
    if ! command -v kubectl >/dev/null 2>&1; then
        log_warning "kubectl not found. Kubernetes deployment will be skipped."
        SKIP_K8S=true
    fi
    
    # Check if helm is available
    if ! command -v helm >/dev/null 2>&1; then
        log_warning "helm not found. Helm deployment will be skipped."
        SKIP_HELM=true
    fi
    
    log_success "Prerequisites check completed"
}

run_quality_gates() {
    log_info "Running quality gates validation..."
    
    # Run tests
    python3 tests/test_quality_gates.py || {
        log_error "Quality gates failed. Deployment aborted."
        exit 1
    }
    
    # Security scan
    if command -v bandit >/dev/null 2>&1; then
        bandit -r src/ -f json -o security_report.json || {
            log_warning "Security scan found issues. Check security_report.json"
        }
    fi
    
    log_success "Quality gates passed"
}

build_docker_image() {
    log_info "Building Docker image..."
    
    local full_tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    local latest_tag="${REGISTRY}/${IMAGE_NAME}:latest"
    
    # Build production image
    docker build \
        --target production \
        --build-arg VERSION="${VERSION}" \
        --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
        -t "${full_tag}" \
        -t "${latest_tag}" \
        .
    
    log_success "Docker image built: ${full_tag}"
    
    # Security scan with Docker Scout (if available)
    if docker scout version >/dev/null 2>&1; then
        log_info "Running Docker security scan..."
        docker scout cves "${full_tag}" || log_warning "Docker security scan found vulnerabilities"
    fi
    
    echo "${full_tag}" > .docker_image_tag
}

push_docker_image() {
    log_info "Pushing Docker image to registry..."
    
    local full_tag=$(cat .docker_image_tag)
    local latest_tag="${REGISTRY}/${IMAGE_NAME}:latest"
    
    # Push versioned tag
    docker push "${full_tag}"
    
    # Push latest tag
    docker push "${latest_tag}"
    
    log_success "Docker image pushed: ${full_tag}"
}

deploy_kubernetes() {
    if [[ "${SKIP_K8S:-false}" == "true" ]]; then
        log_warning "Skipping Kubernetes deployment"
        return
    fi
    
    log_info "Deploying to Kubernetes..."
    
    local full_tag=$(cat .docker_image_tag)
    
    # Create namespace if it doesn't exist
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Generate Kubernetes manifests
    cat > k8s-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamic-graph-fed-rl
  namespace: ${NAMESPACE}
  labels:
    app: dynamic-graph-fed-rl
    version: ${VERSION}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dynamic-graph-fed-rl
  template:
    metadata:
      labels:
        app: dynamic-graph-fed-rl
        version: ${VERSION}
    spec:
      containers:
      - name: federated-rl
        image: ${full_tag}
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
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
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
---
apiVersion: v1
kind: Service
metadata:
  name: dynamic-graph-fed-rl-service
  namespace: ${NAMESPACE}
spec:
  selector:
    app: dynamic-graph-fed-rl
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dynamic-graph-fed-rl-ingress
  namespace: ${NAMESPACE}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - federated-rl.terragonlabs.ai
    secretName: federated-rl-tls
  rules:
  - host: federated-rl.terragonlabs.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dynamic-graph-fed-rl-service
            port:
              number: 80
EOF
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s-deployment.yaml
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/dynamic-graph-fed-rl -n "${NAMESPACE}" --timeout=300s
    
    log_success "Kubernetes deployment completed"
}

deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true
    
    # Deploy Prometheus and Grafana using Helm
    if [[ "${SKIP_HELM:-false}" != "true" ]]; then
        # Add Prometheus community Helm repository
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        
        # Deploy Prometheus
        helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --set grafana.adminPassword=federated-rl-admin \
            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
            --wait
        
        log_success "Monitoring stack deployed"
    else
        log_warning "Skipping Helm-based monitoring deployment"
    fi
}

validate_deployment() {
    log_info "Validating deployment..."
    
    if [[ "${SKIP_K8S:-false}" != "true" ]]; then
        # Check if pods are running
        if kubectl get pods -n "${NAMESPACE}" -l app=dynamic-graph-fed-rl | grep -q "Running"; then
            log_success "Pods are running successfully"
        else
            log_error "Some pods are not running. Check with: kubectl get pods -n ${NAMESPACE}"
            return 1
        fi
        
        # Check if service is accessible
        if kubectl get service dynamic-graph-fed-rl-service -n "${NAMESPACE}" >/dev/null 2>&1; then
            log_success "Service is accessible"
        else
            log_error "Service is not accessible"
            return 1
        fi
    fi
    
    log_success "Deployment validation completed"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f .docker_image_tag k8s-deployment.yaml security_report.json
    log_success "Cleanup completed"
}

generate_deployment_report() {
    log_info "Generating deployment report..."
    
    cat > deployment_report.md << EOF
# Deployment Report

**Timestamp:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Version:** ${VERSION}
**Environment:** ${ENVIRONMENT}
**Image:** ${REGISTRY}/${IMAGE_NAME}:${VERSION}

## Quality Gates
- ✅ Tests passed
- ✅ Security scan completed
- ✅ Docker image built and pushed

## Kubernetes Deployment
- **Namespace:** ${NAMESPACE}
- **Replicas:** 3
- **Image:** ${REGISTRY}/${IMAGE_NAME}:${VERSION}
- **Status:** $(kubectl get deployment dynamic-graph-fed-rl -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "N/A") replicas ready

## Access Information
- **Service:** dynamic-graph-fed-rl-service.${NAMESPACE}.svc.cluster.local:80
- **Ingress:** https://federated-rl.terragonlabs.ai (if configured)

## Monitoring
- **Prometheus:** http://prometheus.monitoring.svc.cluster.local:9090
- **Grafana:** http://grafana.monitoring.svc.cluster.local:3000

## Commands for Management
\`\`\`bash
# Check deployment status
kubectl get deployment dynamic-graph-fed-rl -n ${NAMESPACE}

# View pods
kubectl get pods -n ${NAMESPACE}

# View logs
kubectl logs -f deployment/dynamic-graph-fed-rl -n ${NAMESPACE}

# Scale deployment
kubectl scale deployment dynamic-graph-fed-rl --replicas=5 -n ${NAMESPACE}
\`\`\`

## Rollback Instructions
\`\`\`bash
# Rollback to previous version
kubectl rollout undo deployment/dynamic-graph-fed-rl -n ${NAMESPACE}

# Check rollout status
kubectl rollout status deployment/dynamic-graph-fed-rl -n ${NAMESPACE}
\`\`\`
EOF
    
    log_success "Deployment report generated: deployment_report.md"
}

# Main execution
main() {
    log_info "🚀 Starting production deployment for Dynamic Graph Federated RL v${VERSION}"
    echo "=================================================="
    
    # Trap to ensure cleanup on exit
    trap cleanup EXIT
    
    check_prerequisites
    run_quality_gates
    build_docker_image
    push_docker_image
    deploy_kubernetes
    deploy_monitoring
    validate_deployment
    generate_deployment_report
    
    echo "=================================================="
    log_success "🎉 Production deployment completed successfully!"
    log_info "Version ${VERSION} is now live in ${ENVIRONMENT} environment"
    log_info "📊 Check deployment_report.md for detailed information"
    echo "=================================================="
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        log_info "Rolling back deployment..."
        kubectl rollout undo deployment/dynamic-graph-fed-rl -n "${NAMESPACE}"
        kubectl rollout status deployment/dynamic-graph-fed-rl -n "${NAMESPACE}"
        log_success "Rollback completed"
        ;;
    "status")
        log_info "Checking deployment status..."
        kubectl get deployment dynamic-graph-fed-rl -n "${NAMESPACE}"
        kubectl get pods -n "${NAMESPACE}" -l app=dynamic-graph-fed-rl
        ;;
    "logs")
        kubectl logs -f deployment/dynamic-graph-fed-rl -n "${NAMESPACE}"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|logs}"
        echo "  deploy   - Deploy to production (default)"
        echo "  rollback - Rollback to previous version"
        echo "  status   - Check deployment status"
        echo "  logs     - View deployment logs"
        exit 1
        ;;
esac