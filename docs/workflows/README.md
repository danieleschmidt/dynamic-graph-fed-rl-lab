# CI/CD Workflow Documentation

This directory contains documentation and templates for GitHub Actions workflows. Since GitHub Actions workflows must be manually created in `.github/workflows/`, this documentation provides templates and setup instructions.

## Required Workflows

### 1. Main CI Pipeline
**File**: `.github/workflows/ci.yml`
**Purpose**: Run tests, linting, and quality checks on every PR and push

**Template**: See [ci-template.yml](ci-template.yml)

**Key features**:
- Multi-Python version testing (3.9, 3.10, 3.11)
- JAX CPU/GPU testing matrix
- Code coverage reporting
- Security scanning with bandit
- Dependency vulnerability checks

### 2. Release Pipeline
**File**: `.github/workflows/release.yml`
**Purpose**: Automated package publishing to PyPI

**Template**: See [release-template.yml](release-template.yml)

**Key features**:
- Triggered on version tags
- Build and publish to PyPI
- Create GitHub releases
- Update documentation

### 3. Documentation Build
**File**: `.github/workflows/docs.yml`
**Purpose**: Build and deploy documentation

**Template**: See [docs-template.yml](docs-template.yml)

**Key features**:
- Sphinx documentation building
- GitHub Pages deployment
- API documentation generation

### 4. Security Scanning
**File**: `.github/workflows/security.yml`
**Purpose**: Regular security scans and dependency updates

**Template**: See [security-template.yml](security-template.yml)

**Key features**:
- Daily dependency scanning
- Container security analysis
- SBOM generation
- Automated security updates

## Setup Instructions

### 1. Create Workflow Files
```bash
# Create .github/workflows directory
mkdir -p .github/workflows

# Copy templates to actual workflow files
cp docs/workflows/ci-template.yml .github/workflows/ci.yml
cp docs/workflows/release-template.yml .github/workflows/release.yml
cp docs/workflows/docs-template.yml .github/workflows/docs.yml
cp docs/workflows/security-template.yml .github/workflows/security.yml
```

### 2. Configure Repository Secrets
Add these secrets in GitHub repository settings:

**Required for PyPI publishing**:
- `PYPI_API_TOKEN`: PyPI API token for package publishing

**Optional for enhanced features**:
- `CODECOV_TOKEN`: For code coverage reporting
- `SLACK_WEBHOOK`: For build notifications

### 3. Configure Repository Settings

**Branch Protection**:
- Require PR reviews before merging
- Require status checks to pass
- Require up-to-date branches

**Pages**:
- Enable GitHub Pages for documentation
- Set source to GitHub Actions

### 4. Customize Workflows

Edit the workflow files to match your specific needs:
- Adjust Python versions
- Modify test commands
- Add additional quality checks
- Configure notification preferences

## Workflow Triggers

### Continuous Integration (ci.yml)
- **Push**: All branches
- **Pull Request**: All branches
- **Schedule**: Daily dependency checks

### Release (release.yml)
- **Tags**: Version tags (v*.*.*)
- **Manual**: Workflow dispatch

### Documentation (docs.yml)
- **Push**: Main branch
- **Pull Request**: Documentation changes
- **Manual**: Workflow dispatch

### Security (security.yml)
- **Schedule**: Daily at 2 AM UTC
- **Push**: Security-related files
- **Manual**: Workflow dispatch

## Quality Gates

### Pre-merge Checks
- [ ] All tests pass
- [ ] Code coverage ≥ 80%
- [ ] Linting passes (black, ruff)
- [ ] Type checking passes (mypy)
- [ ] Security scan clean
- [ ] No high-severity vulnerabilities

### Release Checks
- [ ] Version bump committed
- [ ] Changelog updated
- [ ] All CI checks pass
- [ ] Documentation builds successfully
- [ ] Manual testing completed

## Monitoring and Notifications

### Build Status
- Status badges in README
- Slack notifications for failures
- Email notifications for security issues

### Performance Monitoring
- Build time tracking
- Test execution time
- Package size monitoring

## Troubleshooting

### Common Issues

1. **JAX Installation Failures**
   - Use appropriate JAX version for Python/CUDA
   - Separate CPU/GPU installation steps

2. **Memory Issues in CI**
   - Limit test parallelization
   - Use lighter test datasets
   - Optimize memory usage in tests

3. **Timeout Issues**
   - Increase workflow timeouts
   - Optimize slow tests
   - Use test result caching

### Debug Steps
1. Check workflow logs in GitHub Actions
2. Reproduce locally with same environment
3. Use workflow debugging features
4. Review dependency conflicts

## Best Practices

### Workflow Design
- Keep workflows fast and focused
- Use caching for dependencies
- Parallelize independent steps
- Fail fast on critical errors

### Security
- Never commit secrets to repository
- Use least-privilege tokens
- Regularly update action versions
- Scan for vulnerabilities

### Maintenance
- Regular workflow updates
- Monitor execution costs
- Optimize for performance
- Document changes thoroughly

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python CI/CD Best Practices](https://docs.python.org/3/using/unix.html)
- [JAX Installation Guide](https://github.com/google/jax#installation)
- [PyPI Publishing Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)