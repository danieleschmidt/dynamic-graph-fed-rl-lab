#!/bin/bash
set -e

# Build script for Dynamic Graph Fed-RL
# Usage: ./scripts/build.sh [--production] [--gpu] [--push] [--version VERSION]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
BUILD_PRODUCTION=false
BUILD_GPU=false
PUSH_IMAGES=false
VERSION="${VERSION:-dev}"
REGISTRY="${REGISTRY:-ghcr.io/danieleschmidt}"
IMAGE_NAME="dynamic-graph-fed-rl"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --production)
            BUILD_PRODUCTION=true
            shift
            ;;
        --gpu)
            BUILD_GPU=true
            shift
            ;;
        --push)
            PUSH_IMAGES=true
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --production    Build production images"
            echo "  --gpu          Build GPU-enabled images"
            echo "  --push         Push images to registry"
            echo "  --version      Set image version (default: dev)"
            echo "  --registry     Set registry URL (default: ghcr.io/danieleschmidt)"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required tools
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "All dependencies found"
}

# Get build metadata
get_build_metadata() {
    log_info "Gathering build metadata..."
    
    # Git information
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    GIT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "")
    
    # Build timestamp
    BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # If we have a git tag and no version specified, use the tag
    if [[ -n "$GIT_TAG" && "$VERSION" == "dev" ]]; then
        VERSION="$GIT_TAG"
    fi
    
    log_info "Build metadata:"
    echo "  Version: $VERSION"
    echo "  Git Commit: $GIT_COMMIT"
    echo "  Git Branch: $GIT_BRANCH"
    echo "  Git Tag: $GIT_TAG"
    echo "  Build Date: $BUILD_DATE"
}

# Build Docker images
build_images() {
    local build_args=(
        --build-arg "VERSION=$VERSION"
        --build-arg "GIT_COMMIT=$GIT_COMMIT"
        --build-arg "GIT_BRANCH=$GIT_BRANCH"
        --build-arg "BUILD_DATE=$BUILD_DATE"
    )
    
    # Always build development image
    log_info "Building development image..."
    docker build "${build_args[@]}" \
        --target development \
        -t "${IMAGE_NAME}:dev" \
        -t "${IMAGE_NAME}:${VERSION}-dev" \
        .
    log_success "Development image built: ${IMAGE_NAME}:dev"
    
    # Build production image if requested
    if [[ "$BUILD_PRODUCTION" == "true" ]]; then
        log_info "Building production image..."
        docker build "${build_args[@]}" \
            --target production \
            -t "${IMAGE_NAME}:prod" \
            -t "${IMAGE_NAME}:${VERSION}" \
            -t "${IMAGE_NAME}:latest" \
            .
        log_success "Production image built: ${IMAGE_NAME}:prod"
    fi
    
    # Build GPU image if requested
    if [[ "$BUILD_GPU" == "true" ]]; then
        log_info "Building GPU image..."
        docker build "${build_args[@]}" \
            --target gpu \
            -t "${IMAGE_NAME}:gpu" \
            -t "${IMAGE_NAME}:${VERSION}-gpu" \
            .
        log_success "GPU image built: ${IMAGE_NAME}:gpu"
    fi
}

# Tag images for registry
tag_images() {
    if [[ "$PUSH_IMAGES" != "true" ]]; then
        return
    fi
    
    log_info "Tagging images for registry: $REGISTRY"
    
    # Tag development image
    docker tag "${IMAGE_NAME}:dev" "${REGISTRY}/${IMAGE_NAME}:dev"
    docker tag "${IMAGE_NAME}:${VERSION}-dev" "${REGISTRY}/${IMAGE_NAME}:${VERSION}-dev"
    
    if [[ "$BUILD_PRODUCTION" == "true" ]]; then
        docker tag "${IMAGE_NAME}:prod" "${REGISTRY}/${IMAGE_NAME}:prod"
        docker tag "${IMAGE_NAME}:${VERSION}" "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
        docker tag "${IMAGE_NAME}:latest" "${REGISTRY}/${IMAGE_NAME}:latest"
    fi
    
    if [[ "$BUILD_GPU" == "true" ]]; then
        docker tag "${IMAGE_NAME}:gpu" "${REGISTRY}/${IMAGE_NAME}:gpu"
        docker tag "${IMAGE_NAME}:${VERSION}-gpu" "${REGISTRY}/${IMAGE_NAME}:${VERSION}-gpu"
    fi
    
    log_success "Images tagged for registry"
}

# Push images to registry
push_images() {
    if [[ "$PUSH_IMAGES" != "true" ]]; then
        return
    fi
    
    log_info "Pushing images to registry..."
    
    # Push development images
    docker push "${REGISTRY}/${IMAGE_NAME}:dev"
    docker push "${REGISTRY}/${IMAGE_NAME}:${VERSION}-dev"
    
    if [[ "$BUILD_PRODUCTION" == "true" ]]; then
        docker push "${REGISTRY}/${IMAGE_NAME}:prod"
        docker push "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
        docker push "${REGISTRY}/${IMAGE_NAME}:latest"
    fi
    
    if [[ "$BUILD_GPU" == "true" ]]; then
        docker push "${REGISTRY}/${IMAGE_NAME}:gpu"
        docker push "${REGISTRY}/${IMAGE_NAME}:${VERSION}-gpu"
    fi
    
    log_success "Images pushed to registry"
}

# Clean up old images
cleanup_images() {
    log_info "Cleaning up old images..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old versions (keep last 5)
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}" | \
        grep -v "latest\|dev\|gpu\|prod" | \
        sort -k2 -r | \
        tail -n +6 | \
        awk '{print $1}' | \
        xargs -r docker rmi
    
    log_success "Cleanup completed"
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    if command -v syft &> /dev/null; then
        log_info "Generating SBOM..."
        
        mkdir -p build/sbom
        
        syft "${IMAGE_NAME}:dev" -o spdx-json > "build/sbom/${IMAGE_NAME}-dev-${VERSION}-sbom.json"
        
        if [[ "$BUILD_PRODUCTION" == "true" ]]; then
            syft "${IMAGE_NAME}:prod" -o spdx-json > "build/sbom/${IMAGE_NAME}-prod-${VERSION}-sbom.json"
        fi
        
        if [[ "$BUILD_GPU" == "true" ]]; then
            syft "${IMAGE_NAME}:gpu" -o spdx-json > "build/sbom/${IMAGE_NAME}-gpu-${VERSION}-sbom.json"
        fi
        
        log_success "SBOM generated in build/sbom/"
    else
        log_warning "Syft not found, skipping SBOM generation"
    fi
}

# Security scan
security_scan() {
    if command -v trivy &> /dev/null; then
        log_info "Running security scan..."
        
        mkdir -p build/security
        
        trivy image --format json --output "build/security/${IMAGE_NAME}-dev-${VERSION}-scan.json" "${IMAGE_NAME}:dev"
        
        if [[ "$BUILD_PRODUCTION" == "true" ]]; then
            trivy image --format json --output "build/security/${IMAGE_NAME}-prod-${VERSION}-scan.json" "${IMAGE_NAME}:prod"
        fi
        
        if [[ "$BUILD_GPU" == "true" ]]; then
            trivy image --format json --output "build/security/${IMAGE_NAME}-gpu-${VERSION}-scan.json" "${IMAGE_NAME}:gpu"
        fi
        
        log_success "Security scan completed in build/security/"
    else
        log_warning "Trivy not found, skipping security scan"
    fi
}

# Main execution
main() {
    log_info "Starting build process for Dynamic Graph Fed-RL"
    log_info "Configuration:"
    echo "  Production: $BUILD_PRODUCTION"
    echo "  GPU: $BUILD_GPU"
    echo "  Push: $PUSH_IMAGES"
    echo "  Version: $VERSION"
    echo "  Registry: $REGISTRY"
    echo ""
    
    check_dependencies
    get_build_metadata
    build_images
    tag_images
    push_images
    generate_sbom
    security_scan
    cleanup_images
    
    log_success "Build process completed successfully!"
    
    # Print summary
    echo ""
    log_info "Built images:"
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
}

# Run main function
main "$@"