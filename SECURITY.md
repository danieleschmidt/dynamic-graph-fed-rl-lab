# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Dynamic Graph Fed-RL Lab seriously. If you discover a security vulnerability, please follow these steps:

1. **Do not** open a public GitHub issue
2. Email security concerns to: security@example.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

## Security Considerations

This framework handles distributed training and communication between federated agents. Key security areas:

- **Model poisoning**: Malicious agents submitting harmful updates
- **Data privacy**: Ensuring training data remains local to agents
- **Communication security**: Protecting parameter exchanges
- **Resource exhaustion**: Preventing DoS attacks on coordination

## Best Practices

- Use secure communication channels for parameter exchange
- Implement model validation before aggregation
- Monitor for anomalous agent behavior
- Regularly update dependencies for security patches

We will respond to security reports within 48 hours and provide updates on the resolution timeline.