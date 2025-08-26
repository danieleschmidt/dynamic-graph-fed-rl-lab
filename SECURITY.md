# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

To report a security vulnerability, please email security@terragon.ai with:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if available)

We will acknowledge receipt within 24 hours and provide a detailed response within 72 hours.

## Security Measures

### Encryption
- All data in transit is encrypted using TLS 1.3
- Data at rest uses AES-256 encryption
- Post-quantum cryptography for future-proofing

### Authentication
- Multi-factor authentication required
- Zero-trust security model
- Regular security audits

### Code Security
- Static code analysis on every commit
- Dependency vulnerability scanning
- Regular security updates

### Privacy
- Differential privacy for federated learning
- Homomorphic encryption for sensitive computations
- GDPR and CCPA compliance
