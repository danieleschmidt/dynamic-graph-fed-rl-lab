"""
Advanced Security Scanner

Implements breakthrough security scanning with autonomous vulnerability detection,
threat modeling, and intelligent security validation.
"""

import asyncio
import json
import logging
import time
import hashlib
import re
import ast
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from collections import defaultdict, deque
import tempfile
import concurrent.futures
import base64
import secrets

import jax
import jax.numpy as jnp
import numpy as np


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XML_EXTERNAL = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    KNOWN_VULNERABILITIES = "known_vulnerabilities"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    CRYPTOGRAPHIC_FAILURE = "cryptographic_failure"
    SSRF = "server_side_request_forgery"


class SeverityLevel(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreatLevel(Enum):
    """Threat assessment levels."""
    IMMINENT = "imminent"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class SecurityVulnerability:
    """Comprehensive vulnerability representation."""
    vuln_id: str
    vuln_type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    threat_level: ThreatLevel = ThreatLevel.MODERATE
    exploitability: float = 0.5
    impact_score: float = 0.5
    detection_confidence: float = 1.0
    remediation_effort: str = "medium"
    remediation_steps: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    detected_timestamp: float = field(default_factory=time.time)


@dataclass
class SecurityScanResult:
    """Security scan execution result."""
    scan_id: str
    scan_type: str
    start_time: float
    end_time: float
    vulnerabilities: List[SecurityVulnerability]
    files_scanned: int
    lines_analyzed: int
    scan_coverage: float
    execution_time: float
    scan_metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class StaticCodeAnalyzer:
    """Advanced static code analysis for security vulnerabilities."""
    
    def __init__(self):
        self.vulnerability_patterns = self._initialize_vulnerability_patterns()
        self.analysis_cache = {}
        self.logger = logging.getLogger(__name__)
    
    def _initialize_vulnerability_patterns(self) -> Dict[VulnerabilityType, List[Dict[str, Any]]]:
        """Initialize comprehensive vulnerability detection patterns."""
        
        patterns = {
            VulnerabilityType.INJECTION: [
                {
                    "pattern": r'execute\s*\(\s*["\'].*%.*["\']',
                    "description": "SQL injection via string formatting",
                    "severity": SeverityLevel.HIGH,
                    "cwe": "CWE-89"
                },
                {
                    "pattern": r'cursor\.execute\s*\(\s*f["\']',
                    "description": "SQL injection via f-string",
                    "severity": SeverityLevel.HIGH,
                    "cwe": "CWE-89"
                },
                {
                    "pattern": r'os\.system\s*\(\s*["\'].*\+.*["\']',
                    "description": "Command injection via string concatenation",
                    "severity": SeverityLevel.HIGH,
                    "cwe": "CWE-78"
                },
                {
                    "pattern": r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True',
                    "description": "Command injection via shell=True",
                    "severity": SeverityLevel.HIGH,
                    "cwe": "CWE-78"
                }
            ],
            
            VulnerabilityType.SENSITIVE_DATA: [
                {
                    "pattern": r'(password|pwd|secret|key|token)\s*=\s*["\'][^"\']{3,}["\']',
                    "description": "Hardcoded sensitive credentials",
                    "severity": SeverityLevel.CRITICAL,
                    "cwe": "CWE-798"
                },
                {
                    "pattern": r'api_key\s*=\s*["\'][A-Za-z0-9]{20,}["\']',
                    "description": "Hardcoded API key",
                    "severity": SeverityLevel.CRITICAL,
                    "cwe": "CWE-798"
                },
                {
                    "pattern": r'print\s*\([^)]*password[^)]*\)',
                    "description": "Sensitive data in logs",
                    "severity": SeverityLevel.MEDIUM,
                    "cwe": "CWE-532"
                },
                {
                    "pattern": r'logging\.(info|debug|warning)\s*\([^)]*password[^)]*\)',
                    "description": "Password logged in application logs",
                    "severity": SeverityLevel.MEDIUM,
                    "cwe": "CWE-532"
                }
            ],
            
            VulnerabilityType.CRYPTOGRAPHIC_FAILURE: [
                {
                    "pattern": r'hashlib\.md5\s*\(',
                    "description": "Use of weak MD5 hash function",
                    "severity": SeverityLevel.MEDIUM,
                    "cwe": "CWE-327"
                },
                {
                    "pattern": r'hashlib\.sha1\s*\(',
                    "description": "Use of weak SHA1 hash function",
                    "severity": SeverityLevel.MEDIUM,
                    "cwe": "CWE-327"
                },
                {
                    "pattern": r'random\.random\s*\(',
                    "description": "Use of non-cryptographic random generator",
                    "severity": SeverityLevel.LOW,
                    "cwe": "CWE-338"
                },
                {
                    "pattern": r'ssl\.create_default_context\s*\([^)]*check_hostname\s*=\s*False',
                    "description": "Disabled SSL hostname verification",
                    "severity": SeverityLevel.HIGH,
                    "cwe": "CWE-295"
                }
            ],
            
            VulnerabilityType.INSECURE_DESERIALIZATION: [
                {
                    "pattern": r'pickle\.loads?\s*\(',
                    "description": "Unsafe pickle deserialization",
                    "severity": SeverityLevel.HIGH,
                    "cwe": "CWE-502"
                },
                {
                    "pattern": r'yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader',
                    "description": "Unsafe YAML deserialization",
                    "severity": SeverityLevel.HIGH,
                    "cwe": "CWE-502"
                },
                {
                    "pattern": r'eval\s*\(',
                    "description": "Code injection via eval",
                    "severity": SeverityLevel.HIGH,
                    "cwe": "CWE-95"
                },
                {
                    "pattern": r'exec\s*\(',
                    "description": "Code injection via exec",
                    "severity": SeverityLevel.HIGH,
                    "cwe": "CWE-95"
                }
            ],
            
            VulnerabilityType.BROKEN_ACCESS: [
                {
                    "pattern": r'@app\.route\s*\([^)]*methods\s*=\s*\[["\']GET["\'].*["\']POST["\']',
                    "description": "Mixed HTTP methods without proper access control",
                    "severity": SeverityLevel.MEDIUM,
                    "cwe": "CWE-285"
                },
                {
                    "pattern": r'os\.chmod\s*\([^)]*0o777',
                    "description": "Overly permissive file permissions",
                    "severity": SeverityLevel.MEDIUM,
                    "cwe": "CWE-732"
                }
            ],
            
            VulnerabilityType.INSUFFICIENT_LOGGING: [
                {
                    "pattern": r'except\s+\w+:\s*pass',
                    "description": "Silent exception handling without logging",
                    "severity": SeverityLevel.LOW,
                    "cwe": "CWE-778"
                },
                {
                    "pattern": r'try:\s*[^}]*except:\s*[^}]*(?!log)',
                    "description": "Exception without proper logging",
                    "severity": SeverityLevel.LOW,
                    "cwe": "CWE-778"
                }
            ]
        }
        
        return patterns
    
    async def scan_codebase(
        self,
        project_path: Path,
        scan_types: List[VulnerabilityType] = None
    ) -> SecurityScanResult:
        """Perform comprehensive static code analysis."""
        
        if scan_types is None:
            scan_types = list(VulnerabilityType)
        
        self.logger.info(f"Starting static code analysis for {len(scan_types)} vulnerability types")
        
        scan_id = f"static_scan_{int(time.time())}"
        start_time = time.time()
        
        vulnerabilities = []
        files_scanned = 0
        lines_analyzed = 0
        
        # Scan source code
        src_path = project_path / "src"
        if src_path.exists():
            for py_file in src_path.rglob("*.py"):
                file_vulnerabilities, file_lines = await self._scan_file(py_file, scan_types)
                vulnerabilities.extend(file_vulnerabilities)
                files_scanned += 1
                lines_analyzed += file_lines
        
        # Scan configuration files
        config_vulnerabilities = await self._scan_configuration_files(project_path)
        vulnerabilities.extend(config_vulnerabilities)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate scan coverage
        scan_coverage = self._calculate_scan_coverage(project_path, files_scanned)
        
        self.logger.info(
            f"Static analysis complete: {len(vulnerabilities)} vulnerabilities found "
            f"in {files_scanned} files ({lines_analyzed} lines) in {execution_time:.1f}s"
        )
        
        return SecurityScanResult(
            scan_id=scan_id,
            scan_type="static_code_analysis",
            start_time=start_time,
            end_time=end_time,
            vulnerabilities=vulnerabilities,
            files_scanned=files_scanned,
            lines_analyzed=lines_analyzed,
            scan_coverage=scan_coverage,
            execution_time=execution_time,
            scan_metadata={
                "vulnerability_types_scanned": [vt.value for vt in scan_types],
                "patterns_checked": sum(len(patterns) for patterns in self.vulnerability_patterns.values())
            }
        )
    
    async def _scan_file(
        self,
        file_path: Path,
        scan_types: List[VulnerabilityType]
    ) -> Tuple[List[SecurityVulnerability], int]:
        """Scan individual file for vulnerabilities."""
        
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Check each vulnerability type
            for vuln_type in scan_types:
                if vuln_type in self.vulnerability_patterns:
                    patterns = self.vulnerability_patterns[vuln_type]
                    
                    for pattern_config in patterns:
                        pattern = pattern_config["pattern"]
                        description = pattern_config["description"]
                        severity = pattern_config["severity"]
                        cwe = pattern_config.get("cwe", "")
                        
                        # Find matches
                        matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
                        
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            
                            vulnerability = SecurityVulnerability(
                                vuln_id=self._generate_vuln_id(file_path, line_num, pattern),
                                vuln_type=vuln_type,
                                severity=severity,
                                title=description,
                                description=f"{description} detected in {file_path.name}",
                                file_path=str(file_path),
                                line_number=line_num,
                                code_snippet=lines[line_num - 1] if line_num <= len(lines) else "",
                                cwe_id=cwe,
                                exploitability=self._calculate_exploitability(vuln_type, severity),
                                impact_score=self._calculate_impact_score(vuln_type, severity),
                                detection_confidence=0.85,
                                remediation_steps=self._get_remediation_steps(vuln_type),
                                tags={vuln_type.value, severity.value, "static_analysis"}
                            )
                            
                            vulnerabilities.append(vulnerability)
            
            return vulnerabilities, len(lines)
            
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
            return [], 0
    
    def _generate_vuln_id(self, file_path: Path, line_num: int, pattern: str) -> str:
        """Generate unique vulnerability ID."""
        data = f"{file_path}:{line_num}:{pattern}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]
    
    def _calculate_exploitability(self, vuln_type: VulnerabilityType, severity: SeverityLevel) -> float:
        """Calculate exploitability score."""
        
        # Base exploitability by type
        type_exploitability = {
            VulnerabilityType.INJECTION: 0.9,
            VulnerabilityType.SENSITIVE_DATA: 0.8,
            VulnerabilityType.XSS: 0.8,
            VulnerabilityType.INSECURE_DESERIALIZATION: 0.7,
            VulnerabilityType.BROKEN_AUTH: 0.7,
            VulnerabilityType.CRYPTOGRAPHIC_FAILURE: 0.6,
            VulnerabilityType.BROKEN_ACCESS: 0.6,
            VulnerabilityType.SECURITY_MISCONFIG: 0.5,
            VulnerabilityType.INSUFFICIENT_LOGGING: 0.3
        }
        
        base_score = type_exploitability.get(vuln_type, 0.5)
        
        # Adjust by severity
        severity_multipliers = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.6,
            SeverityLevel.LOW: 0.4,
            SeverityLevel.INFO: 0.2
        }
        
        return base_score * severity_multipliers.get(severity, 0.5)
    
    def _calculate_impact_score(self, vuln_type: VulnerabilityType, severity: SeverityLevel) -> float:
        """Calculate impact score."""
        
        # Base impact by type
        type_impact = {
            VulnerabilityType.INJECTION: 0.9,
            VulnerabilityType.BROKEN_AUTH: 0.9,
            VulnerabilityType.SENSITIVE_DATA: 0.8,
            VulnerabilityType.BROKEN_ACCESS: 0.8,
            VulnerabilityType.INSECURE_DESERIALIZATION: 0.7,
            VulnerabilityType.CRYPTOGRAPHIC_FAILURE: 0.7,
            VulnerabilityType.XSS: 0.6,
            VulnerabilityType.SECURITY_MISCONFIG: 0.5,
            VulnerabilityType.INSUFFICIENT_LOGGING: 0.3
        }
        
        base_score = type_impact.get(vuln_type, 0.5)
        
        # Adjust by severity
        severity_multipliers = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.6,
            SeverityLevel.LOW: 0.4,
            SeverityLevel.INFO: 0.2
        }
        
        return base_score * severity_multipliers.get(severity, 0.5)
    
    def _get_remediation_steps(self, vuln_type: VulnerabilityType) -> List[str]:
        """Get remediation steps for vulnerability type."""
        
        remediation_map = {
            VulnerabilityType.INJECTION: [
                "Use parameterized queries or prepared statements",
                "Validate and sanitize all user inputs",
                "Implement input validation frameworks",
                "Use ORM frameworks with built-in protection"
            ],
            VulnerabilityType.SENSITIVE_DATA: [
                "Remove hardcoded credentials from source code",
                "Use environment variables or secure vault systems",
                "Implement proper secret management",
                "Add credential scanning to CI/CD pipeline"
            ],
            VulnerabilityType.CRYPTOGRAPHIC_FAILURE: [
                "Use strong cryptographic algorithms (AES-256, SHA-256+)",
                "Implement proper key management",
                "Use secure random number generators",
                "Enable proper SSL/TLS configuration"
            ],
            VulnerabilityType.INSECURE_DESERIALIZATION: [
                "Avoid deserializing untrusted data",
                "Use safe serialization formats (JSON)",
                "Implement integrity checks for serialized data",
                "Use allow-lists for deserialization"
            ],
            VulnerabilityType.BROKEN_ACCESS: [
                "Implement proper authorization checks",
                "Use principle of least privilege",
                "Add access control testing",
                "Review and update permission models"
            ]
        }
        
        return remediation_map.get(vuln_type, ["Review and address security issue"])
    
    async def _scan_configuration_files(self, project_path: Path) -> List[SecurityVulnerability]:
        """Scan configuration files for security issues."""
        
        vulnerabilities = []
        
        # Configuration file patterns
        config_patterns = [
            "*.yml", "*.yaml", "*.json", "*.ini", "*.cfg", "*.conf",
            ".env", "*.env", "docker-compose.yml", "Dockerfile"
        ]
        
        for pattern in config_patterns:
            for config_file in project_path.rglob(pattern):
                try:
                    file_vulns = await self._scan_config_file(config_file)
                    vulnerabilities.extend(file_vulns)
                except Exception as e:
                    self.logger.warning(f"Error scanning config file {config_file}: {e}")
        
        return vulnerabilities
    
    async def _scan_config_file(self, config_file: Path) -> List[SecurityVulnerability]:
        """Scan individual configuration file."""
        
        vulnerabilities = []
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Check for sensitive data in config
            sensitive_patterns = [
                (r'password\s*[:=]\s*["\'][^"\']{3,}["\']', "Hardcoded password in config"),
                (r'secret\s*[:=]\s*["\'][^"\']{5,}["\']', "Hardcoded secret in config"),
                (r'api_key\s*[:=]\s*["\'][^"\']{10,}["\']', "Hardcoded API key in config"),
                (r'token\s*[:=]\s*["\'][^"\']{20,}["\']', "Hardcoded token in config"),
                (r'debug\s*[:=]\s*true', "Debug mode enabled"),
                (r'ssl_verify\s*[:=]\s*false', "SSL verification disabled")
            ]
            
            for pattern, description in sensitive_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # Determine severity
                    if "password" in description.lower() or "secret" in description.lower():
                        severity = SeverityLevel.CRITICAL
                    elif "debug" in description.lower() or "ssl" in description.lower():
                        severity = SeverityLevel.HIGH
                    else:
                        severity = SeverityLevel.MEDIUM
                    
                    vulnerability = SecurityVulnerability(
                        vuln_id=self._generate_vuln_id(config_file, line_num, pattern),
                        vuln_type=VulnerabilityType.SENSITIVE_DATA,
                        severity=severity,
                        title=description,
                        description=f"{description} found in configuration file",
                        file_path=str(config_file),
                        line_number=line_num,
                        code_snippet=lines[line_num - 1] if line_num <= len(lines) else "",
                        cwe_id="CWE-798",
                        detection_confidence=0.9,
                        remediation_steps=[
                            "Move sensitive data to environment variables",
                            "Use secure configuration management",
                            "Implement configuration encryption"
                        ],
                        tags={"configuration", "sensitive_data", severity.value}
                    )
                    
                    vulnerabilities.append(vulnerability)
        
        except Exception as e:
            self.logger.error(f"Error scanning config file {config_file}: {e}")
        
        return vulnerabilities
    
    def _calculate_scan_coverage(self, project_path: Path, files_scanned: int) -> float:
        """Calculate scan coverage percentage."""
        
        total_files = 0
        
        # Count Python files
        src_path = project_path / "src"
        if src_path.exists():
            total_files += len(list(src_path.rglob("*.py")))
        
        # Count configuration files
        config_patterns = ["*.yml", "*.yaml", "*.json", "*.ini", "*.cfg"]
        for pattern in config_patterns:
            total_files += len(list(project_path.rglob(pattern)))
        
        return files_scanned / max(1, total_files)


class DependencyScanner:
    """Scans dependencies for known vulnerabilities."""
    
    def __init__(self):
        self.vulnerability_database = self._load_vulnerability_database()
        self.logger = logging.getLogger(__name__)
    
    def _load_vulnerability_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load vulnerability database (simplified mock)."""
        
        # Mock vulnerability database
        return {
            "numpy": [
                {
                    "cve": "CVE-2021-33430",
                    "versions": "< 1.21.1",
                    "severity": "HIGH",
                    "description": "Buffer overflow in numpy array handling",
                    "cvss_score": 7.5
                }
            ],
            "requests": [
                {
                    "cve": "CVE-2023-32681",
                    "versions": "< 2.31.0",
                    "severity": "MEDIUM",
                    "description": "Proxy-Authorization header leak",
                    "cvss_score": 6.1
                }
            ],
            "pillow": [
                {
                    "cve": "CVE-2023-50447",
                    "versions": "< 10.2.0",
                    "severity": "HIGH",
                    "description": "Arbitrary code execution via crafted image",
                    "cvss_score": 8.1
                }
            ]
        }
    
    async def scan_dependencies(self, project_path: Path) -> SecurityScanResult:
        """Scan project dependencies for known vulnerabilities."""
        
        self.logger.info("Starting dependency vulnerability scan")
        
        scan_id = f"dep_scan_{int(time.time())}"
        start_time = time.time()
        
        vulnerabilities = []
        
        # Check requirements.txt
        requirements_file = project_path / "requirements.txt"
        if requirements_file.exists():
            req_vulnerabilities = await self._scan_requirements_file(requirements_file)
            vulnerabilities.extend(req_vulnerabilities)
        
        # Check pyproject.toml
        pyproject_file = project_path / "pyproject.toml"
        if pyproject_file.exists():
            pyproject_vulnerabilities = await self._scan_pyproject_file(pyproject_file)
            vulnerabilities.extend(pyproject_vulnerabilities)
        
        end_time = time.time()
        
        return SecurityScanResult(
            scan_id=scan_id,
            scan_type="dependency_scan",
            start_time=start_time,
            end_time=end_time,
            vulnerabilities=vulnerabilities,
            files_scanned=2,  # requirements.txt and pyproject.toml
            lines_analyzed=0,
            scan_coverage=1.0,
            execution_time=end_time - start_time,
            scan_metadata={
                "database_version": "mock_v1.0",
                "packages_checked": len(self.vulnerability_database)
            }
        )
    
    async def _scan_requirements_file(self, requirements_file: Path) -> List[SecurityVulnerability]:
        """Scan requirements.txt for vulnerable dependencies."""
        
        vulnerabilities = []
        
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    package_name, version = self._parse_requirement(line)
                    
                    if package_name in self.vulnerability_database:
                        package_vulns = self.vulnerability_database[package_name]
                        
                        for vuln_data in package_vulns:
                            if self._is_vulnerable_version(version, vuln_data["versions"]):
                                vulnerability = SecurityVulnerability(
                                    vuln_id=f"dep_{package_name}_{vuln_data['cve']}",
                                    vuln_type=VulnerabilityType.KNOWN_VULNERABILITIES,
                                    severity=SeverityLevel(vuln_data["severity"].lower()),
                                    title=f"Vulnerable dependency: {package_name}",
                                    description=vuln_data["description"],
                                    file_path=str(requirements_file),
                                    line_number=line_num,
                                    code_snippet=line,
                                    cve_id=vuln_data["cve"],
                                    cvss_score=vuln_data["cvss_score"],
                                    detection_confidence=0.95,
                                    remediation_steps=[
                                        f"Update {package_name} to latest secure version",
                                        "Review dependency security advisories",
                                        "Implement dependency scanning in CI/CD"
                                    ],
                                    tags={"dependency", "known_vulnerability", vuln_data["severity"].lower()}
                                )
                                
                                vulnerabilities.append(vulnerability)
        
        except Exception as e:
            self.logger.error(f"Error scanning requirements file: {e}")
        
        return vulnerabilities
    
    async def _scan_pyproject_file(self, pyproject_file: Path) -> List[SecurityVulnerability]:
        """Scan pyproject.toml for vulnerable dependencies."""
        
        # Similar implementation to requirements scanning
        # For brevity, returning empty list in this demo
        return []
    
    def _parse_requirement(self, requirement_line: str) -> Tuple[str, Optional[str]]:
        """Parse requirement line to extract package name and version."""
        
        # Simple parsing - would be more sophisticated in production
        if ">=" in requirement_line:
            parts = requirement_line.split(">=")
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else None
        elif "==" in requirement_line:
            parts = requirement_line.split("==")
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else None
        else:
            return requirement_line.strip(), None
    
    def _is_vulnerable_version(self, current_version: Optional[str], vulnerable_versions: str) -> bool:
        """Check if current version is vulnerable."""
        
        if not current_version:
            return True  # Unknown version, assume vulnerable
        
        # Simplified version comparison
        if "< " in vulnerable_versions:
            threshold = vulnerable_versions.replace("< ", "").strip()
            return self._version_less_than(current_version, threshold)
        
        return False
    
    def _version_less_than(self, version1: str, version2: str) -> bool:
        """Simple version comparison."""
        
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad to same length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            return v1_parts < v2_parts
            
        except Exception:
            return True  # Assume vulnerable if can't parse


class ThreatModelingEngine:
    """Advanced threat modeling and risk assessment."""
    
    def __init__(self):
        self.threat_models = {}
        self.attack_vectors = self._initialize_attack_vectors()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_attack_vectors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize attack vector database."""
        
        return {
            "code_injection": {
                "description": "Injection of malicious code through user inputs",
                "likelihood": 0.7,
                "impact": 0.9,
                "detection_methods": ["static_analysis", "dynamic_testing"],
                "mitigation_strategies": ["input_validation", "parameterized_queries", "sandboxing"]
            },
            "data_exfiltration": {
                "description": "Unauthorized access and extraction of sensitive data",
                "likelihood": 0.6,
                "impact": 0.8,
                "detection_methods": ["access_monitoring", "data_flow_analysis"],
                "mitigation_strategies": ["access_controls", "encryption", "monitoring"]
            },
            "privilege_escalation": {
                "description": "Gaining higher privileges than intended",
                "likelihood": 0.5,
                "impact": 0.9,
                "detection_methods": ["permission_analysis", "runtime_monitoring"],
                "mitigation_strategies": ["least_privilege", "role_based_access", "audit_logging"]
            },
            "denial_of_service": {
                "description": "Making system unavailable to legitimate users",
                "likelihood": 0.8,
                "impact": 0.6,
                "detection_methods": ["performance_monitoring", "resource_analysis"],
                "mitigation_strategies": ["rate_limiting", "resource_limits", "load_balancing"]
            }
        }
    
    async def generate_threat_model(
        self,
        project_path: Path,
        vulnerabilities: List[SecurityVulnerability]
    ) -> Dict[str, Any]:
        """Generate comprehensive threat model."""
        
        self.logger.info("Generating threat model based on discovered vulnerabilities")
        
        threat_model = {
            "model_id": f"threat_model_{int(time.time())}",
            "project_path": str(project_path),
            "generation_time": time.time(),
            "attack_scenarios": [],
            "risk_assessment": {},
            "mitigation_roadmap": [],
            "security_posture": {}
        }
        
        # Analyze attack scenarios
        attack_scenarios = await self._analyze_attack_scenarios(vulnerabilities)
        threat_model["attack_scenarios"] = attack_scenarios
        
        # Calculate risk assessment
        risk_assessment = await self._calculate_risk_assessment(vulnerabilities, attack_scenarios)
        threat_model["risk_assessment"] = risk_assessment
        
        # Generate mitigation roadmap
        mitigation_roadmap = await self._generate_mitigation_roadmap(vulnerabilities, risk_assessment)
        threat_model["mitigation_roadmap"] = mitigation_roadmap
        
        # Assess overall security posture
        security_posture = await self._assess_security_posture(vulnerabilities, risk_assessment)
        threat_model["security_posture"] = security_posture
        
        return threat_model
    
    async def _analyze_attack_scenarios(
        self,
        vulnerabilities: List[SecurityVulnerability]
    ) -> List[Dict[str, Any]]:
        """Analyze potential attack scenarios."""
        
        scenarios = []
        
        # Group vulnerabilities by type
        vuln_groups = defaultdict(list)
        for vuln in vulnerabilities:
            vuln_groups[vuln.vuln_type].append(vuln)
        
        # Generate scenarios for each attack vector
        for attack_vector, vector_data in self.attack_vectors.items():
            # Check if vulnerabilities enable this attack vector
            enabling_vulns = []
            
            for vuln_type, vulns in vuln_groups.items():
                if self._enables_attack_vector(vuln_type, attack_vector):
                    enabling_vulns.extend(vulns)
            
            if enabling_vulns:
                scenario = {
                    "attack_vector": attack_vector,
                    "description": vector_data["description"],
                    "likelihood": self._calculate_scenario_likelihood(enabling_vulns, vector_data),
                    "impact": vector_data["impact"],
                    "risk_score": 0.0,
                    "enabling_vulnerabilities": [v.vuln_id for v in enabling_vulns],
                    "attack_path": self._generate_attack_path(attack_vector, enabling_vulns),
                    "mitigation_priority": "high" if vector_data["impact"] > 0.8 else "medium"
                }
                
                scenario["risk_score"] = scenario["likelihood"] * scenario["impact"]
                scenarios.append(scenario)
        
        # Sort by risk score
        scenarios.sort(key=lambda s: s["risk_score"], reverse=True)
        
        return scenarios
    
    def _enables_attack_vector(self, vuln_type: VulnerabilityType, attack_vector: str) -> bool:
        """Check if vulnerability type enables attack vector."""
        
        enabling_map = {
            "code_injection": [VulnerabilityType.INJECTION, VulnerabilityType.INSECURE_DESERIALIZATION],
            "data_exfiltration": [VulnerabilityType.SENSITIVE_DATA, VulnerabilityType.BROKEN_ACCESS],
            "privilege_escalation": [VulnerabilityType.BROKEN_AUTH, VulnerabilityType.BROKEN_ACCESS],
            "denial_of_service": [VulnerabilityType.INJECTION, VulnerabilityType.SECURITY_MISCONFIG]
        }
        
        return vuln_type in enabling_map.get(attack_vector, [])
    
    def _calculate_scenario_likelihood(
        self,
        enabling_vulns: List[SecurityVulnerability],
        vector_data: Dict[str, Any]
    ) -> float:
        """Calculate likelihood of attack scenario."""
        
        base_likelihood = vector_data["likelihood"]
        
        # Adjust based on vulnerability severity
        severity_weights = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.6,
            SeverityLevel.LOW: 0.4,
            SeverityLevel.INFO: 0.2
        }
        
        vuln_factor = max([severity_weights.get(v.severity, 0.5) for v in enabling_vulns])
        
        return min(1.0, base_likelihood * vuln_factor)
    
    def _generate_attack_path(
        self,
        attack_vector: str,
        enabling_vulns: List[SecurityVulnerability]
    ) -> List[str]:
        """Generate step-by-step attack path."""
        
        attack_paths = {
            "code_injection": [
                "Identify input validation vulnerabilities",
                "Craft malicious payload",
                "Inject payload through vulnerable endpoint",
                "Execute arbitrary code on system"
            ],
            "data_exfiltration": [
                "Identify access control weaknesses",
                "Gain unauthorized access to sensitive data",
                "Extract and transfer data externally",
                "Cover tracks and maintain persistence"
            ],
            "privilege_escalation": [
                "Identify authentication bypass",
                "Gain initial system access",
                "Exploit privilege escalation vulnerability",
                "Achieve administrative privileges"
            ],
            "denial_of_service": [
                "Identify resource consumption vulnerabilities",
                "Craft resource exhaustion attack",
                "Overwhelm system resources",
                "Cause service unavailability"
            ]
        }
        
        return attack_paths.get(attack_vector, ["Generic attack path"])
    
    async def _calculate_risk_assessment(
        self,
        vulnerabilities: List[SecurityVulnerability],
        attack_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment."""
        
        # Vulnerability risk distribution
        severity_counts = defaultdict(int)
        for vuln in vulnerabilities:
            severity_counts[vuln.severity.value] += 1
        
        # Calculate overall risk score
        severity_weights = {
            "critical": 10,
            "high": 7,
            "medium": 4,
            "low": 2,
            "info": 1
        }
        
        risk_score = sum(
            count * severity_weights.get(severity, 0)
            for severity, count in severity_counts.items()
        )
        
        # Normalize to 0-100 scale
        normalized_risk = min(100, risk_score)
        
        # Calculate threat level
        if normalized_risk >= 80:
            threat_level = ThreatLevel.IMMINENT
        elif normalized_risk >= 60:
            threat_level = ThreatLevel.HIGH
        elif normalized_risk >= 40:
            threat_level = ThreatLevel.MODERATE
        elif normalized_risk >= 20:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.MINIMAL
        
        return {
            "overall_risk_score": normalized_risk,
            "threat_level": threat_level.value,
            "vulnerability_distribution": dict(severity_counts),
            "top_attack_scenarios": sorted(attack_scenarios, key=lambda s: s["risk_score"], reverse=True)[:5],
            "risk_factors": [
                f"{count} {severity} severity vulnerabilities"
                for severity, count in severity_counts.items() if count > 0
            ],
            "immediate_threats": len([s for s in attack_scenarios if s["risk_score"] > 0.7]),
            "risk_trend": "stable"  # Would analyze trend over time
        }
    
    async def _generate_mitigation_roadmap(
        self,
        vulnerabilities: List[SecurityVulnerability],
        risk_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate prioritized mitigation roadmap."""
        
        roadmap = []
        
        # Group vulnerabilities by priority
        critical_vulns = [v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL]
        high_vulns = [v for v in vulnerabilities if v.severity == SeverityLevel.HIGH]
        medium_vulns = [v for v in vulnerabilities if v.severity == SeverityLevel.MEDIUM]
        
        # Critical vulnerabilities - immediate action
        if critical_vulns:
            roadmap.append({
                "phase": "immediate",
                "timeline": "0-24 hours",
                "priority": "critical",
                "actions": [
                    "Address all critical vulnerabilities immediately",
                    "Implement emergency patches",
                    "Consider system isolation if necessary"
                ],
                "vulnerabilities": len(critical_vulns),
                "estimated_effort": "high"
            })
        
        # High vulnerabilities - short term
        if high_vulns:
            roadmap.append({
                "phase": "short_term",
                "timeline": "1-7 days",
                "priority": "high",
                "actions": [
                    "Fix high-severity vulnerabilities",
                    "Implement additional security controls",
                    "Update security monitoring"
                ],
                "vulnerabilities": len(high_vulns),
                "estimated_effort": "medium"
            })
        
        # Medium vulnerabilities - medium term
        if medium_vulns:
            roadmap.append({
                "phase": "medium_term",
                "timeline": "1-4 weeks",
                "priority": "medium",
                "actions": [
                    "Address medium-severity vulnerabilities",
                    "Implement security best practices",
                    "Enhance security testing"
                ],
                "vulnerabilities": len(medium_vulns),
                "estimated_effort": "medium"
            })
        
        # Long-term security improvements
        roadmap.append({
            "phase": "long_term",
            "timeline": "1-3 months",
            "priority": "improvement",
            "actions": [
                "Implement comprehensive security framework",
                "Establish security development lifecycle",
                "Deploy advanced threat detection",
                "Conduct security awareness training"
            ],
            "vulnerabilities": 0,
            "estimated_effort": "high"
        })
        
        return roadmap
    
    async def _assess_security_posture(
        self,
        vulnerabilities: List[SecurityVulnerability],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall security posture."""
        
        # Calculate security metrics
        total_vulns = len(vulnerabilities)
        critical_vulns = len([v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL])
        high_vulns = len([v for v in vulnerabilities if v.severity == SeverityLevel.HIGH])
        
        # Security posture score (0-100)
        if critical_vulns > 0:
            posture_score = 20
        elif high_vulns > 5:
            posture_score = 40
        elif high_vulns > 0:
            posture_score = 60
        elif total_vulns > 10:
            posture_score = 70
        elif total_vulns > 0:
            posture_score = 80
        else:
            posture_score = 95
        
        # Security maturity assessment
        maturity_indicators = {
            "vulnerability_management": 0.7,
            "secure_coding_practices": 0.6,
            "security_testing": 0.8,
            "incident_response": 0.5,
            "security_monitoring": 0.6
        }
        
        maturity_score = statistics.mean(maturity_indicators.values())
        
        # Overall security grade
        if posture_score >= 90 and maturity_score >= 0.8:
            security_grade = "A"
        elif posture_score >= 80 and maturity_score >= 0.7:
            security_grade = "B"
        elif posture_score >= 70 and maturity_score >= 0.6:
            security_grade = "C"
        elif posture_score >= 60:
            security_grade = "D"
        else:
            security_grade = "F"
        
        return {
            "security_score": posture_score,
            "security_grade": security_grade,
            "maturity_score": maturity_score,
            "maturity_indicators": maturity_indicators,
            "risk_level": risk_assessment["threat_level"],
            "improvement_areas": [
                area for area, score in maturity_indicators.items() if score < 0.7
            ],
            "strengths": [
                area for area, score in maturity_indicators.items() if score >= 0.8
            ],
            "recommendations": [
                "Implement automated vulnerability scanning",
                "Establish security code review process",
                "Deploy security monitoring and alerting",
                "Conduct regular penetration testing",
                "Implement security awareness training"
            ]
        }


class AdvancedSecurityScanner:
    """Master security scanner orchestrating all security validation."""
    
    def __init__(
        self,
        project_path: Path,
        enable_static_analysis: bool = True,
        enable_dependency_scanning: bool = True,
        enable_threat_modeling: bool = True,
        parallel_scanning: bool = True
    ):
        self.project_path = Path(project_path)
        self.enable_static_analysis = enable_static_analysis
        self.enable_dependency_scanning = enable_dependency_scanning
        self.enable_threat_modeling = enable_threat_modeling
        self.parallel_scanning = parallel_scanning
        
        # Initialize scanners
        self.static_analyzer = StaticCodeAnalyzer() if enable_static_analysis else None
        self.dependency_scanner = DependencyScanner() if enable_dependency_scanning else None
        self.threat_modeler = ThreatModelingEngine() if enable_threat_modeling else None
        
        # Scan tracking
        self.scan_history = deque(maxlen=100)
        self.vulnerability_database = defaultdict(list)
        self.security_metrics = {
            "total_scans": 0,
            "vulnerabilities_found": 0,
            "vulnerabilities_fixed": 0,
            "average_scan_time": 0.0,
            "security_score_trend": []
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_security_scan(
        self,
        scan_types: Optional[List[str]] = None,
        include_threat_modeling: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive security scan with all available methods."""
        
        self.logger.info("🛡️ Starting comprehensive security scan")
        
        start_time = time.time()
        scan_id = f"comprehensive_scan_{int(start_time)}"
        
        # Initialize scan results
        scan_results = {
            "scan_id": scan_id,
            "start_time": start_time,
            "scan_types_requested": scan_types or ["all"],
            "static_analysis": None,
            "dependency_scan": None,
            "threat_model": None,
            "overall_summary": {},
            "remediation_plan": {}
        }
        
        # Execute scans
        scan_tasks = []
        
        if self.enable_static_analysis and self.static_analyzer:
            if self.parallel_scanning:
                scan_tasks.append(self._run_static_analysis())
            else:
                scan_results["static_analysis"] = await self._run_static_analysis()
        
        if self.enable_dependency_scanning and self.dependency_scanner:
            if self.parallel_scanning:
                scan_tasks.append(self._run_dependency_scan())
            else:
                scan_results["dependency_scan"] = await self._run_dependency_scan()
        
        # Execute parallel scans
        if scan_tasks:
            scan_task_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
            
            # Process results
            task_index = 0
            if self.enable_static_analysis:
                result = scan_task_results[task_index]
                scan_results["static_analysis"] = result if not isinstance(result, Exception) else {"error": str(result)}
                task_index += 1
            
            if self.enable_dependency_scanning:
                result = scan_task_results[task_index]
                scan_results["dependency_scan"] = result if not isinstance(result, Exception) else {"error": str(result)}
        
        # Collect all vulnerabilities
        all_vulnerabilities = []
        
        if scan_results["static_analysis"] and "vulnerabilities" in scan_results["static_analysis"]:
            all_vulnerabilities.extend(scan_results["static_analysis"]["vulnerabilities"])
        
        if scan_results["dependency_scan"] and "vulnerabilities" in scan_results["dependency_scan"]:
            all_vulnerabilities.extend(scan_results["dependency_scan"]["vulnerabilities"])
        
        # Generate threat model
        if self.enable_threat_modeling and self.threat_modeler and include_threat_modeling:
            threat_model = await self.threat_modeler.generate_threat_model(
                self.project_path, all_vulnerabilities
            )
            scan_results["threat_model"] = threat_model
        
        # Calculate overall summary
        scan_results["overall_summary"] = await self._calculate_overall_security_summary(
            scan_results, all_vulnerabilities
        )
        
        # Generate remediation plan
        scan_results["remediation_plan"] = await self._generate_comprehensive_remediation_plan(
            all_vulnerabilities, scan_results["threat_model"]
        )
        
        # Update metrics
        self._update_security_metrics(scan_results)
        
        # Store scan history
        self.scan_history.append(scan_results)
        
        execution_time = time.time() - start_time
        self.logger.info(
            f"✅ Comprehensive security scan complete: {len(all_vulnerabilities)} vulnerabilities "
            f"found in {execution_time:.1f}s"
        )
        
        scan_results["end_time"] = time.time()
        scan_results["execution_time"] = execution_time
        
        return scan_results
    
    async def _run_static_analysis(self) -> SecurityScanResult:
        """Run static code analysis."""
        return await self.static_analyzer.scan_codebase(self.project_path)
    
    async def _run_dependency_scan(self) -> SecurityScanResult:
        """Run dependency vulnerability scan."""
        return await self.dependency_scanner.scan_dependencies(self.project_path)
    
    async def _calculate_overall_security_summary(
        self,
        scan_results: Dict[str, Any],
        all_vulnerabilities: List[SecurityVulnerability]
    ) -> Dict[str, Any]:
        """Calculate overall security summary."""
        
        # Vulnerability statistics
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for vuln in all_vulnerabilities:
            severity_counts[vuln.severity.value] += 1
            type_counts[vuln.vuln_type.value] += 1
        
        # Calculate security score (0-100)
        score_penalties = {
            "critical": 25,
            "high": 15,
            "medium": 5,
            "low": 2,
            "info": 1
        }
        
        total_penalty = sum(
            count * score_penalties.get(severity, 0)
            for severity, count in severity_counts.items()
        )
        
        security_score = max(0, 100 - total_penalty)
        
        # Determine security status
        if security_score >= 90:
            security_status = "excellent"
        elif security_score >= 80:
            security_status = "good"
        elif security_score >= 60:
            security_status = "fair"
        elif security_score >= 40:
            security_status = "poor"
        else:
            security_status = "critical"
        
        # Calculate scan efficiency
        total_files_scanned = 0
        total_execution_time = 0.0
        
        for scan_key in ["static_analysis", "dependency_scan"]:
            if scan_results.get(scan_key) and not isinstance(scan_results[scan_key], dict) or "error" not in scan_results[scan_key]:
                scan_result = scan_results[scan_key]
                total_files_scanned += scan_result.files_scanned
                total_execution_time += scan_result.execution_time
        
        return {
            "security_score": security_score,
            "security_status": security_status,
            "total_vulnerabilities": len(all_vulnerabilities),
            "vulnerability_distribution": dict(severity_counts),
            "vulnerability_types": dict(type_counts),
            "files_scanned": total_files_scanned,
            "scan_efficiency": total_files_scanned / max(1, total_execution_time),
            "critical_issues": severity_counts["critical"],
            "high_issues": severity_counts["high"],
            "requires_immediate_action": severity_counts["critical"] > 0 or severity_counts["high"] > 3,
            "compliance_status": self._assess_compliance_status(severity_counts),
            "scan_completeness": self._calculate_scan_completeness(scan_results)
        }
    
    def _assess_compliance_status(self, severity_counts: Dict[str, int]) -> Dict[str, Any]:
        """Assess compliance with security standards."""
        
        # Mock compliance assessment
        compliance_standards = {
            "OWASP_Top_10": {
                "compliant": severity_counts["critical"] == 0 and severity_counts["high"] <= 2,
                "score": max(0, 100 - severity_counts["critical"] * 50 - severity_counts["high"] * 20)
            },
            "CWE_Top_25": {
                "compliant": severity_counts["critical"] == 0 and severity_counts["high"] <= 1,
                "score": max(0, 100 - severity_counts["critical"] * 60 - severity_counts["high"] * 25)
            },
            "NIST_Cybersecurity": {
                "compliant": severity_counts["critical"] == 0,
                "score": max(0, 100 - severity_counts["critical"] * 100)
            }
        }
        
        overall_compliance = all(std["compliant"] for std in compliance_standards.values())
        average_score = statistics.mean([std["score"] for std in compliance_standards.values()])
        
        return {
            "overall_compliant": overall_compliance,
            "average_compliance_score": average_score,
            "standards": compliance_standards,
            "compliance_grade": "Pass" if overall_compliance else "Fail"
        }
    
    def _calculate_scan_completeness(self, scan_results: Dict[str, Any]) -> float:
        """Calculate how complete the security scan was."""
        
        completeness_factors = []
        
        # Static analysis completeness
        if scan_results.get("static_analysis"):
            static_result = scan_results["static_analysis"]
            if isinstance(static_result, SecurityScanResult):
                completeness_factors.append(static_result.scan_coverage)
            else:
                completeness_factors.append(1.0)  # Assume complete if successful
        
        # Dependency scan completeness
        if scan_results.get("dependency_scan"):
            completeness_factors.append(1.0)  # Dependencies are either scanned or not
        
        # Threat modeling completeness
        if scan_results.get("threat_model"):
            completeness_factors.append(1.0)  # Threat model is comprehensive when present
        
        return statistics.mean(completeness_factors) if completeness_factors else 0.0
    
    async def _generate_comprehensive_remediation_plan(
        self,
        vulnerabilities: List[SecurityVulnerability],
        threat_model: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive remediation plan."""
        
        remediation_plan = {
            "plan_id": f"remediation_{int(time.time())}",
            "total_vulnerabilities": len(vulnerabilities),
            "immediate_actions": [],
            "short_term_actions": [],
            "long_term_actions": [],
            "estimated_effort": {},
            "priority_matrix": {},
            "automation_opportunities": []
        }
        
        # Categorize vulnerabilities by urgency
        immediate = [v for v in vulnerabilities if v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]
        short_term = [v for v in vulnerabilities if v.severity == SeverityLevel.MEDIUM]
        long_term = [v for v in vulnerabilities if v.severity == SeverityLevel.LOW]
        
        # Generate immediate actions
        for vuln in immediate:
            remediation_plan["immediate_actions"].extend([
                f"Fix {vuln.vuln_type.value} in {vuln.file_path}:{vuln.line_number}",
                *vuln.remediation_steps
            ])
        
        # Generate short-term actions
        for vuln in short_term:
            remediation_plan["short_term_actions"].extend(vuln.remediation_steps)
        
        # Generate long-term actions
        remediation_plan["long_term_actions"] = [
            "Implement comprehensive security framework",
            "Establish security code review process",
            "Deploy automated security testing",
            "Conduct regular security assessments",
            "Implement security awareness training"
        ]
        
        # Estimate effort
        remediation_plan["estimated_effort"] = {
            "immediate": f"{len(immediate)} issues, ~{len(immediate) * 2} hours",
            "short_term": f"{len(short_term)} issues, ~{len(short_term) * 1} hours",
            "long_term": "Framework implementation, ~40-80 hours"
        }
        
        # Identify automation opportunities
        remediation_plan["automation_opportunities"] = [
            "Automated dependency vulnerability scanning",
            "Pre-commit security hooks",
            "Automated security testing in CI/CD",
            "Dynamic security monitoring",
            "Automated threat intelligence updates"
        ]
        
        return remediation_plan
    
    def _update_security_metrics(self, scan_results: Dict[str, Any]):
        """Update security metrics from scan results."""
        
        self.security_metrics["total_scans"] += 1
        
        # Count vulnerabilities
        all_vulns = []
        if scan_results.get("static_analysis") and isinstance(scan_results["static_analysis"], SecurityScanResult):
            all_vulns.extend(scan_results["static_analysis"].vulnerabilities)
        if scan_results.get("dependency_scan") and isinstance(scan_results["dependency_scan"], SecurityScanResult):
            all_vulns.extend(scan_results["dependency_scan"].vulnerabilities)
        
        self.security_metrics["vulnerabilities_found"] += len(all_vulns)
        
        # Update average scan time
        execution_time = scan_results.get("execution_time", 0.0)
        total_scans = self.security_metrics["total_scans"]
        current_avg = self.security_metrics["average_scan_time"]
        self.security_metrics["average_scan_time"] = (
            (current_avg * (total_scans - 1) + execution_time) / total_scans
        )
        
        # Track security score trend
        security_score = scan_results.get("overall_summary", {}).get("security_score", 0)
        self.security_metrics["security_score_trend"].append(security_score)
        
        # Keep only last 50 scores
        if len(self.security_metrics["security_score_trend"]) > 50:
            self.security_metrics["security_score_trend"] = self.security_metrics["security_score_trend"][-50:]
    
    async def run_continuous_security_monitoring(
        self,
        monitoring_interval: float = 3600.0,  # 1 hour
        quick_scan_only: bool = True
    ):
        """Run continuous security monitoring."""
        
        self.logger.info("Starting continuous security monitoring")
        
        while True:
            try:
                # Run security scan
                if quick_scan_only:
                    # Quick scan - static analysis only
                    scan_result = await self._run_static_analysis()
                    vulnerabilities = scan_result.vulnerabilities
                else:
                    # Full comprehensive scan
                    scan_result = await self.run_comprehensive_security_scan(
                        include_threat_modeling=False
                    )
                    vulnerabilities = []
                    if scan_result.get("static_analysis"):
                        vulnerabilities.extend(scan_result["static_analysis"].vulnerabilities)
                    if scan_result.get("dependency_scan"):
                        vulnerabilities.extend(scan_result["dependency_scan"].vulnerabilities)
                
                # Check for critical issues
                critical_vulns = [v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL]
                high_vulns = [v for v in vulnerabilities if v.severity == SeverityLevel.HIGH]
                
                if critical_vulns:
                    self.logger.critical(f"CRITICAL: {len(critical_vulns)} critical vulnerabilities detected!")
                
                if high_vulns:
                    self.logger.warning(f"WARNING: {len(high_vulns)} high-severity vulnerabilities detected")
                
                # Adaptive monitoring interval
                if critical_vulns:
                    monitoring_interval = 900  # 15 minutes for critical issues
                elif high_vulns:
                    monitoring_interval = 1800  # 30 minutes for high issues
                else:
                    monitoring_interval = min(7200, monitoring_interval * 1.1)  # Increase interval if clean
                
                await asyncio.sleep(monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Continuous security monitoring error: {e}")
                await asyncio.sleep(300)  # 5 minutes before retry
    
    def export_security_report(
        self,
        output_path: Path,
        include_threat_model: bool = True,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export comprehensive security report."""
        
        if not self.scan_history:
            return {"error": "No scan history available"}
        
        latest_scan = self.scan_history[-1]
        
        report_data = {
            "report_metadata": {
                "project_path": str(self.project_path),
                "generation_time": time.time(),
                "report_version": "1.0.0",
                "scanner_version": "advanced_v1.0"
            },
            "executive_summary": self._generate_executive_summary(latest_scan),
            "latest_scan_results": latest_scan,
            "security_metrics": self.security_metrics.copy(),
            "historical_analysis": self._analyze_security_history(),
            "recommendations": self._generate_security_recommendations(latest_scan)
        }
        
        # Export to file
        timestamp = int(time.time())
        if format.lower() == "json":
            output_file = output_path / f"security_report_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        return {
            "status": "exported",
            "output_file": str(output_file),
            "report_size": len(json.dumps(report_data, default=str)),
            "vulnerabilities_reported": latest_scan.get("overall_summary", {}).get("total_vulnerabilities", 0)
        }
    
    def _generate_executive_summary(self, latest_scan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of security posture."""
        
        overall_summary = latest_scan.get("overall_summary", {})
        
        return {
            "security_posture": overall_summary.get("security_status", "unknown"),
            "overall_score": overall_summary.get("security_score", 0),
            "critical_findings": overall_summary.get("critical_issues", 0),
            "high_priority_findings": overall_summary.get("high_issues", 0),
            "immediate_action_required": overall_summary.get("requires_immediate_action", False),
            "compliance_status": overall_summary.get("compliance_status", {}).get("overall_compliant", False),
            "key_risks": [
                "Critical vulnerabilities require immediate attention",
                "Dependency vulnerabilities present supply chain risks",
                "Configuration issues may expose sensitive data"
            ] if overall_summary.get("total_vulnerabilities", 0) > 0 else [
                "No critical security issues identified",
                "Security posture is acceptable",
                "Continue regular security monitoring"
            ]
        }
    
    def _analyze_security_history(self) -> Dict[str, Any]:
        """Analyze security history and trends."""
        
        if len(self.scan_history) < 2:
            return {"message": "Insufficient history for trend analysis"}
        
        # Calculate trends
        recent_scans = list(self.scan_history)[-10:]
        security_scores = [
            scan.get("overall_summary", {}).get("security_score", 0)
            for scan in recent_scans
        ]
        
        vulnerability_counts = [
            scan.get("overall_summary", {}).get("total_vulnerabilities", 0)
            for scan in recent_scans
        ]
        
        # Trend analysis
        if len(security_scores) >= 3:
            score_trend = "improving" if security_scores[-1] > security_scores[0] else "declining" if security_scores[-1] < security_scores[0] else "stable"
        else:
            score_trend = "unknown"
        
        return {
            "scan_count": len(self.scan_history),
            "recent_scans_analyzed": len(recent_scans),
            "average_security_score": statistics.mean(security_scores),
            "security_score_trend": score_trend,
            "average_vulnerabilities": statistics.mean(vulnerability_counts),
            "vulnerability_trend": "decreasing" if len(vulnerability_counts) >= 2 and vulnerability_counts[-1] < vulnerability_counts[0] else "stable",
            "best_security_score": max(security_scores),
            "worst_security_score": min(security_scores),
            "improvement_rate": (security_scores[-1] - security_scores[0]) / len(security_scores) if len(security_scores) > 1 else 0.0
        }
    
    def _generate_security_recommendations(self, latest_scan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate security improvement recommendations."""
        
        recommendations = []
        
        overall_summary = latest_scan.get("overall_summary", {})
        remediation_plan = latest_scan.get("remediation_plan", {})
        
        # High-priority recommendations
        if overall_summary.get("critical_issues", 0) > 0:
            recommendations.append({
                "priority": "critical",
                "category": "vulnerability_management",
                "title": "Address Critical Vulnerabilities",
                "description": "Critical vulnerabilities require immediate attention",
                "actions": remediation_plan.get("immediate_actions", []),
                "timeline": "0-24 hours",
                "effort": "high"
            })
        
        if overall_summary.get("high_issues", 0) > 3:
            recommendations.append({
                "priority": "high",
                "category": "security_posture",
                "title": "Improve Overall Security Posture",
                "description": "Multiple high-severity issues indicate systemic security problems",
                "actions": [
                    "Implement security code review process",
                    "Add automated security scanning to CI/CD",
                    "Conduct security training for development team"
                ],
                "timeline": "1-2 weeks",
                "effort": "medium"
            })
        
        # Framework recommendations
        recommendations.append({
            "priority": "medium",
            "category": "security_framework",
            "title": "Establish Security Framework",
            "description": "Implement comprehensive security development lifecycle",
            "actions": [
                "Define security policies and standards",
                "Implement security testing automation",
                "Establish incident response procedures",
                "Deploy security monitoring and alerting"
            ],
            "timeline": "1-3 months",
            "effort": "high"
        })
        
        return recommendations


async def main():
    """Demonstration of advanced security scanner."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Initializing Advanced Security Scanner")
    
    # Initialize scanner
    scanner = AdvancedSecurityScanner(
        project_path=Path.cwd(),
        enable_static_analysis=True,
        enable_dependency_scanning=True,
        enable_threat_modeling=True,
        parallel_scanning=True
    )
    
    # Run comprehensive scan
    scan_result = await scanner.run_comprehensive_security_scan()
    
    # Display results
    summary = scan_result["overall_summary"]
    logger.info(f"✅ Security scan complete:")
    logger.info(f"  Security score: {summary.get('security_score', 0)}/100")
    logger.info(f"  Total vulnerabilities: {summary.get('total_vulnerabilities', 0)}")
    logger.info(f"  Critical issues: {summary.get('critical_issues', 0)}")
    logger.info(f"  High issues: {summary.get('high_issues', 0)}")
    
    # Export report
    export_result = scanner.export_security_report(
        output_path=Path.cwd() / "security_reports"
    )
    
    logger.info(f"📊 Security report exported to: {export_result.get('output_file', 'N/A')}")
    
    return scan_result


if __name__ == "__main__":
    asyncio.run(main())