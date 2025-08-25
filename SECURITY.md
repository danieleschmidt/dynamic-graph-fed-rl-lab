# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability, please report it responsibly:

1. **Do not** create a public GitHub issue
2. Email security@terragon.ai with details
3. Include steps to reproduce if possible
4. Allow time for fix before public disclosure

## Security Measures

### Code Security
- No hardcoded secrets or passwords
- Input validation for all user data
- Parameterized queries to prevent SQL injection
- Avoid eval() and similar dynamic execution

### Data Security
- Encrypt sensitive data at rest and in transit
- Use environment variables for secrets
- Implement proper access controls
- Regular security audits

### Deployment Security
- Use HTTPS for all communications
- Keep dependencies updated
- Monitor for vulnerabilities
- Implement proper logging

## Security Checklist

- [ ] No hardcoded credentials
- [ ] Input validation implemented
- [ ] SQL injection protection
- [ ] XSS prevention
- [ ] CSRF protection
- [ ] Access control implemented
- [ ] Logging and monitoring
- [ ] Dependencies updated

## Contact

For security-related questions: security@terragon.ai
