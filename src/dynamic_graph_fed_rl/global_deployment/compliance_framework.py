"""Compliance framework for global regulatory adherence in federated RL systems."""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
import hashlib
import threading


class ComplianceLevel(Enum):
    """Compliance requirement levels."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class ComplianceStatus(Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL_COMPLIANT = "partial_compliant"
    PENDING_REVIEW = "pending_review"
    EXEMPT = "exempt"


@dataclass
class ComplianceStandard:
    """Definition of a compliance standard."""
    
    id: str
    name: str
    description: str
    jurisdiction: str  # e.g., "EU", "US", "Global"
    version: str
    effective_date: str
    requirements: List[str] = field(default_factory=list)
    level: ComplianceLevel = ComplianceLevel.MANDATORY
    audit_frequency_days: int = 90
    data_retention_days: int = 2555  # 7 years default
    encryption_required: bool = True
    data_residency_required: bool = False
    consent_required: bool = False
    documentation_required: List[str] = field(default_factory=list)


@dataclass
class ComplianceViolation:
    """Record of a compliance violation."""
    
    id: str
    standard_id: str
    violation_type: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: float
    region: Optional[str] = None
    component: Optional[str] = None
    data_affected: Optional[str] = None
    remediation_required: bool = True
    remediation_deadline: Optional[float] = None
    status: str = "open"


class ComplianceFramework:
    """
    Comprehensive compliance framework for global federated RL systems.
    
    Features:
    - Multi-standard compliance monitoring (GDPR, CCPA, HIPAA, SOC2, etc.)
    - Automated compliance checking and reporting
    - Data governance and audit trails
    - Privacy impact assessments
    - Right to deletion and data portability
    - Consent management
    - Cross-border data transfer compliance
    """
    
    def __init__(self):
        self.standards: Dict[str, ComplianceStandard] = {}
        self.violations: List[ComplianceViolation] = []
        self.compliance_status: Dict[str, ComplianceStatus] = {}
        self.audit_logs: List[Dict[str, Any]] = []
        
        # Data governance
        self.data_processing_records: List[Dict[str, Any]] = []
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_subject_requests: List[Dict[str, Any]] = []
        
        # Privacy and encryption
        self.encryption_keys: Dict[str, str] = {}
        self.data_anonymization_records: List[Dict[str, Any]] = []
        
        # Compliance monitoring
        self.compliance_checks: Dict[str, Callable] = {}
        self.compliance_lock = threading.RLock()
        
        # Metrics
        self.compliance_metrics = {
            "total_checks": 0,
            "violations_found": 0,
            "last_audit": None,
            "compliance_score": 0.0
        }
        
        print("🔒 Compliance framework initialized")
        
        # Setup standard compliance frameworks
        self._setup_compliance_standards()
        self._setup_compliance_checks()
    
    def _setup_compliance_standards(self):
        """Setup standard compliance frameworks."""
        standards = [
            ComplianceStandard(
                id="gdpr",
                name="General Data Protection Regulation",
                description="EU data protection regulation",
                jurisdiction="EU",
                version="2018",
                effective_date="2018-05-25",
                requirements=[
                    "lawful_basis_for_processing",
                    "data_subject_consent",
                    "right_to_erasure",
                    "data_portability",
                    "privacy_by_design",
                    "data_protection_impact_assessment",
                    "breach_notification_72h"
                ],
                level=ComplianceLevel.MANDATORY,
                data_residency_required=True,
                consent_required=True,
                documentation_required=["privacy_policy", "consent_forms", "dpia"]
            ),
            ComplianceStandard(
                id="ccpa",
                name="California Consumer Privacy Act",
                description="California privacy rights for consumers",
                jurisdiction="US-CA",
                version="2020",
                effective_date="2020-01-01",
                requirements=[
                    "right_to_know",
                    "right_to_delete",
                    "right_to_opt_out",
                    "non_discrimination",
                    "privacy_notice"
                ],
                level=ComplianceLevel.MANDATORY,
                documentation_required=["privacy_notice", "opt_out_process"]
            ),
            ComplianceStandard(
                id="hipaa",
                name="Health Insurance Portability and Accountability Act",
                description="US healthcare data protection",
                jurisdiction="US",
                version="1996",
                effective_date="1996-08-21",
                requirements=[
                    "minimum_necessary_standard",
                    "safeguards_rule",
                    "breach_notification",
                    "business_associate_agreements",
                    "audit_controls"
                ],
                level=ComplianceLevel.MANDATORY,
                encryption_required=True,
                documentation_required=["hipaa_notice", "baa_agreements"]
            ),
            ComplianceStandard(
                id="soc2",
                name="Service Organization Control 2",
                description="Security and availability standards",
                jurisdiction="US",
                version="2017",
                effective_date="2017-05-01",
                requirements=[
                    "security_principle",
                    "availability_principle",
                    "processing_integrity",
                    "confidentiality",
                    "privacy_principle"
                ],
                level=ComplianceLevel.RECOMMENDED,
                audit_frequency_days=365,
                documentation_required=["security_policies", "audit_reports"]
            ),
            ComplianceStandard(
                id="iso27001",
                name="ISO 27001 Information Security Management",
                description="International information security standard",
                jurisdiction="Global",
                version="2013",
                effective_date="2013-10-01",
                requirements=[
                    "isms_implementation",
                    "risk_management",
                    "security_controls",
                    "continuous_improvement",
                    "management_review"
                ],
                level=ComplianceLevel.RECOMMENDED,
                audit_frequency_days=365,
                documentation_required=["isms_documentation", "risk_register"]
            ),
            ComplianceStandard(
                id="nist_csf",
                name="NIST Cybersecurity Framework",
                description="US cybersecurity framework",
                jurisdiction="US",
                version="1.1",
                effective_date="2018-04-16",
                requirements=[
                    "identify_function",
                    "protect_function", 
                    "detect_function",
                    "respond_function",
                    "recover_function"
                ],
                level=ComplianceLevel.RECOMMENDED,
                documentation_required=["cybersecurity_policy", "incident_response_plan"]
            )
        ]
        
        for standard in standards:
            self.register_standard(standard)
    
    def _setup_compliance_checks(self):
        """Setup automated compliance checking functions."""
        
        def check_gdpr_compliance(data_processing_activity: Dict[str, Any]) -> ComplianceStatus:
            """Check GDPR compliance for data processing activity."""
            required_elements = ["lawful_basis", "data_subject_consent", "purpose_limitation"]
            
            missing_elements = [elem for elem in required_elements 
                             if elem not in data_processing_activity]
            
            if not missing_elements:
                return ComplianceStatus.COMPLIANT
            elif len(missing_elements) <= 1:
                return ComplianceStatus.PARTIAL_COMPLIANT
            else:
                return ComplianceStatus.NON_COMPLIANT
        
        def check_data_encryption(data_info: Dict[str, Any]) -> ComplianceStatus:
            """Check if data is properly encrypted."""
            if data_info.get("encrypted", False) and data_info.get("encryption_standard"):
                return ComplianceStatus.COMPLIANT
            else:
                return ComplianceStatus.NON_COMPLIANT
        
        def check_consent_validity(consent_record: Dict[str, Any]) -> ComplianceStatus:
            """Check if consent is valid and current."""
            consent_timestamp = consent_record.get("timestamp", 0)
            consent_expiry = consent_record.get("expiry", 0)
            current_time = time.time()
            
            if consent_timestamp > 0 and (consent_expiry == 0 or current_time < consent_expiry):
                return ComplianceStatus.COMPLIANT
            else:
                return ComplianceStatus.NON_COMPLIANT
        
        self.compliance_checks.update({
            "gdpr_compliance": check_gdpr_compliance,
            "data_encryption": check_data_encryption,
            "consent_validity": check_consent_validity
        })
        
        print(f"🔒 Registered {len(self.compliance_checks)} compliance checks")
    
    def register_standard(self, standard: ComplianceStandard):
        """Register a new compliance standard."""
        with self.compliance_lock:
            self.standards[standard.id] = standard
            self.compliance_status[standard.id] = ComplianceStatus.PENDING_REVIEW
            
            print(f"🔒 Registered compliance standard: {standard.name} ({standard.id})")
    
    def record_data_processing(self, activity: Dict[str, Any]) -> str:
        """Record a data processing activity for compliance tracking."""
        activity_id = hashlib.md5(
            f"{activity.get('purpose', '')}{time.time()}".encode()
        ).hexdigest()[:12]
        
        processing_record = {
            "id": activity_id,
            "timestamp": time.time(),
            "purpose": activity.get("purpose", ""),
            "data_categories": activity.get("data_categories", []),
            "legal_basis": activity.get("legal_basis", ""),
            "data_subjects": activity.get("data_subjects", []),
            "recipients": activity.get("recipients", []),
            "retention_period": activity.get("retention_period", 0),
            "security_measures": activity.get("security_measures", []),
            "cross_border_transfers": activity.get("cross_border_transfers", False)
        }
        
        self.data_processing_records.append(processing_record)
        
        # Log audit trail
        self._log_audit_event(
            "data_processing_recorded",
            {"activity_id": activity_id, "purpose": activity.get("purpose", "")}
        )
        
        print(f"📋 Recorded data processing activity: {activity_id}")
        return activity_id
    
    def record_consent(self, data_subject_id: str, consent_details: Dict[str, Any]) -> str:
        """Record consent from a data subject."""
        consent_id = hashlib.md5(
            f"{data_subject_id}{time.time()}".encode()
        ).hexdigest()[:12]
        
        consent_record = {
            "consent_id": consent_id,
            "data_subject_id": data_subject_id,
            "timestamp": time.time(),
            "purposes": consent_details.get("purposes", []),
            "data_categories": consent_details.get("data_categories", []),
            "processing_basis": consent_details.get("processing_basis", "consent"),
            "withdrawal_method": consent_details.get("withdrawal_method", ""),
            "expiry": consent_details.get("expiry", 0),
            "granular_consent": consent_details.get("granular_consent", {}),
            "consent_string": consent_details.get("consent_string", ""),
            "active": True
        }
        
        if data_subject_id not in self.consent_records:
            self.consent_records[data_subject_id] = {}
        
        self.consent_records[data_subject_id][consent_id] = consent_record
        
        self._log_audit_event(
            "consent_recorded",
            {"data_subject_id": data_subject_id, "consent_id": consent_id}
        )
        
        print(f"✅ Recorded consent: {consent_id} for subject {data_subject_id}")
        return consent_id
    
    def withdraw_consent(self, data_subject_id: str, consent_id: str) -> bool:
        """Process consent withdrawal."""
        if (data_subject_id in self.consent_records and 
            consent_id in self.consent_records[data_subject_id]):
            
            self.consent_records[data_subject_id][consent_id]["active"] = False
            self.consent_records[data_subject_id][consent_id]["withdrawal_timestamp"] = time.time()
            
            self._log_audit_event(
                "consent_withdrawn",
                {"data_subject_id": data_subject_id, "consent_id": consent_id}
            )
            
            print(f"🚫 Consent withdrawn: {consent_id} for subject {data_subject_id}")
            return True
        
        return False
    
    def process_data_subject_request(self, request: Dict[str, Any]) -> str:
        """Process data subject rights requests (access, deletion, portability)."""
        request_id = hashlib.md5(
            f"{request.get('subject_id', '')}{request.get('type', '')}{time.time()}".encode()
        ).hexdigest()[:12]
        
        request_record = {
            "request_id": request_id,
            "timestamp": time.time(),
            "subject_id": request.get("subject_id", ""),
            "request_type": request.get("type", ""),  # "access", "deletion", "portability"
            "status": "pending",
            "completion_deadline": time.time() + (30 * 24 * 3600),  # 30 days
            "verification_method": request.get("verification_method", ""),
            "processing_notes": []
        }
        
        self.data_subject_requests.append(request_record)
        
        # Auto-process certain types of requests
        if request.get("type") == "access":
            self._process_access_request(request_record)
        elif request.get("type") == "deletion":
            self._process_deletion_request(request_record)
        elif request.get("type") == "portability":
            self._process_portability_request(request_record)
        
        self._log_audit_event(
            "data_subject_request",
            {"request_id": request_id, "type": request.get("type", "")}
        )
        
        print(f"📨 Processing data subject request: {request_id}")
        return request_id
    
    def _process_access_request(self, request_record: Dict[str, Any]):
        """Process data subject access request."""
        subject_id = request_record["subject_id"]
        
        # Collect all data for the subject
        subject_data = {
            "personal_data": self._collect_personal_data(subject_id),
            "processing_activities": self._collect_processing_activities(subject_id),
            "consent_records": self.consent_records.get(subject_id, {}),
            "data_sources": self._collect_data_sources(subject_id)
        }
        
        request_record["status"] = "completed"
        request_record["completion_timestamp"] = time.time()
        request_record["data_package"] = subject_data
        
        print(f"📋 Access request completed for subject: {subject_id}")
    
    def _process_deletion_request(self, request_record: Dict[str, Any]):
        """Process data subject deletion request (right to be forgotten)."""
        subject_id = request_record["subject_id"]
        
        # Mark data for deletion (actual deletion would be implemented by specific services)
        deletion_tasks = [
            f"Delete personal data for {subject_id}",
            f"Anonymize analytics data for {subject_id}",
            f"Remove consent records for {subject_id}",
            f"Update processing records for {subject_id}"
        ]
        
        request_record["status"] = "completed"
        request_record["completion_timestamp"] = time.time()
        request_record["deletion_tasks"] = deletion_tasks
        
        print(f"🗑️  Deletion request completed for subject: {subject_id}")
    
    def _process_portability_request(self, request_record: Dict[str, Any]):
        """Process data portability request."""
        subject_id = request_record["subject_id"]
        
        # Create portable data package
        portable_data = {
            "format": "JSON",
            "timestamp": time.time(),
            "subject_id": subject_id,
            "data": self._collect_portable_data(subject_id)
        }
        
        request_record["status"] = "completed"
        request_record["completion_timestamp"] = time.time()
        request_record["portable_data"] = portable_data
        
        print(f"📦 Portability request completed for subject: {subject_id}")
    
    def run_compliance_audit(self, standard_id: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive compliance audit."""
        audit_id = hashlib.md5(f"audit_{time.time()}".encode()).hexdigest()[:12]
        audit_timestamp = time.time()
        
        print(f"🔍 Starting compliance audit: {audit_id}")
        
        standards_to_audit = [standard_id] if standard_id else list(self.standards.keys())
        audit_results = {
            "audit_id": audit_id,
            "timestamp": audit_timestamp,
            "standards_audited": standards_to_audit,
            "results": {},
            "violations_found": [],
            "recommendations": [],
            "overall_score": 0.0
        }
        
        total_checks = 0
        passed_checks = 0
        
        for std_id in standards_to_audit:
            standard = self.standards[std_id]
            standard_results = {
                "standard_name": standard.name,
                "requirements_checked": len(standard.requirements),
                "compliant_requirements": 0,
                "violations": [],
                "status": ComplianceStatus.PENDING_REVIEW
            }
            
            # Check each requirement
            for requirement in standard.requirements:
                total_checks += 1
                check_result = self._check_requirement_compliance(std_id, requirement)
                
                if check_result["status"] == ComplianceStatus.COMPLIANT:
                    standard_results["compliant_requirements"] += 1
                    passed_checks += 1
                else:
                    violation = ComplianceViolation(
                        id=hashlib.md5(f"{std_id}_{requirement}_{time.time()}".encode()).hexdigest()[:12],
                        standard_id=std_id,
                        violation_type=requirement,
                        description=check_result.get("message", "Requirement not met"),
                        severity=check_result.get("severity", "medium"),
                        timestamp=audit_timestamp
                    )
                    
                    standard_results["violations"].append(violation.__dict__)
                    audit_results["violations_found"].append(violation.__dict__)
                    self.violations.append(violation)
            
            # Calculate standard compliance status
            compliance_ratio = standard_results["compliant_requirements"] / len(standard.requirements)
            if compliance_ratio >= 0.95:
                standard_results["status"] = ComplianceStatus.COMPLIANT
            elif compliance_ratio >= 0.8:
                standard_results["status"] = ComplianceStatus.PARTIAL_COMPLIANT
            else:
                standard_results["status"] = ComplianceStatus.NON_COMPLIANT
            
            audit_results["results"][std_id] = standard_results
            self.compliance_status[std_id] = standard_results["status"]
        
        # Calculate overall audit score
        if total_checks > 0:
            audit_results["overall_score"] = (passed_checks / total_checks) * 100
        
        # Generate recommendations
        audit_results["recommendations"] = self._generate_compliance_recommendations(audit_results)
        
        # Update metrics
        self.compliance_metrics.update({
            "total_checks": self.compliance_metrics["total_checks"] + total_checks,
            "violations_found": len(self.violations),
            "last_audit": audit_timestamp,
            "compliance_score": audit_results["overall_score"]
        })
        
        # Log audit
        self._log_audit_event("compliance_audit_completed", {
            "audit_id": audit_id,
            "score": audit_results["overall_score"],
            "violations": len(audit_results["violations_found"])
        })
        
        print(f"✅ Compliance audit completed: {audit_results['overall_score']:.1f}% compliance")
        return audit_results
    
    def _check_requirement_compliance(self, standard_id: str, requirement: str) -> Dict[str, Any]:
        """Check compliance for a specific requirement."""
        # Mock compliance checking - in real implementation, this would 
        # integrate with actual system components and data
        
        check_result = {
            "requirement": requirement,
            "status": ComplianceStatus.COMPLIANT,
            "message": f"Requirement {requirement} is compliant",
            "severity": "low"
        }
        
        # Simulate some non-compliance scenarios
        if requirement == "breach_notification_72h":
            # Check if any breaches were reported within 72 hours
            check_result["status"] = ComplianceStatus.COMPLIANT
        elif requirement == "data_subject_consent":
            # Check consent records
            active_consents = sum(1 for subject_consents in self.consent_records.values()
                                for consent in subject_consents.values() if consent["active"])
            if active_consents == 0:
                check_result["status"] = ComplianceStatus.NON_COMPLIANT
                check_result["message"] = "No active consent records found"
                check_result["severity"] = "high"
        elif requirement == "encryption_required":
            # Check encryption standards
            check_result["status"] = ComplianceStatus.COMPLIANT
        
        return check_result
    
    def _generate_compliance_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on audit results."""
        recommendations = []
        
        if audit_results["overall_score"] < 80:
            recommendations.append("Overall compliance score is below 80%. Prioritize addressing critical violations.")
        
        if len(audit_results["violations_found"]) > 10:
            recommendations.append("High number of violations detected. Consider implementing automated compliance monitoring.")
        
        # Check for specific standard recommendations
        for std_id, results in audit_results["results"].items():
            if results["status"] == ComplianceStatus.NON_COMPLIANT:
                standard_name = self.standards[std_id].name
                recommendations.append(f"Address {standard_name} compliance issues immediately.")
        
        return recommendations
    
    def _collect_personal_data(self, subject_id: str) -> Dict[str, Any]:
        """Collect all personal data for a data subject."""
        # Mock implementation - would integrate with actual data stores
        return {
            "profile_data": {"id": subject_id, "created": time.time()},
            "activity_data": [],
            "preferences": {}
        }
    
    def _collect_processing_activities(self, subject_id: str) -> List[Dict[str, Any]]:
        """Collect processing activities involving a data subject."""
        return [
            record for record in self.data_processing_records
            if subject_id in record.get("data_subjects", [])
        ]
    
    def _collect_data_sources(self, subject_id: str) -> List[str]:
        """Collect data sources for a data subject."""
        return ["user_registration", "activity_tracking", "federation_participation"]
    
    def _collect_portable_data(self, subject_id: str) -> Dict[str, Any]:
        """Collect data in portable format for data subject."""
        return {
            "personal_data": self._collect_personal_data(subject_id),
            "activity_history": [],
            "preferences": {}
        }
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event for compliance tracking."""
        audit_event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
            "user": "system"  # In real implementation, would track actual user
        }
        
        self.audit_logs.append(audit_event)
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard data."""
        dashboard = {
            "summary": {
                "total_standards": len(self.standards),
                "compliant_standards": sum(1 for status in self.compliance_status.values() 
                                         if status == ComplianceStatus.COMPLIANT),
                "total_violations": len(self.violations),
                "open_violations": sum(1 for v in self.violations if v.status == "open"),
                "last_audit": self.compliance_metrics.get("last_audit"),
                "compliance_score": self.compliance_metrics.get("compliance_score", 0.0)
            },
            "standards_status": {
                std_id: {
                    "name": standard.name,
                    "status": self.compliance_status.get(std_id, ComplianceStatus.PENDING_REVIEW).value,
                    "jurisdiction": standard.jurisdiction,
                    "level": standard.level.value
                }
                for std_id, standard in self.standards.items()
            },
            "recent_violations": [
                {
                    "id": v.id,
                    "standard": self.standards[v.standard_id].name,
                    "type": v.violation_type,
                    "severity": v.severity,
                    "timestamp": v.timestamp
                }
                for v in sorted(self.violations, key=lambda x: x.timestamp, reverse=True)[:10]
            ],
            "data_subject_requests": {
                "total": len(self.data_subject_requests),
                "pending": sum(1 for r in self.data_subject_requests if r["status"] == "pending"),
                "completed": sum(1 for r in self.data_subject_requests if r["status"] == "completed")
            },
            "consent_overview": {
                "total_subjects": len(self.consent_records),
                "active_consents": sum(
                    sum(1 for consent in subject_consents.values() if consent["active"])
                    for subject_consents in self.consent_records.values()
                ),
                "withdrawn_consents": sum(
                    sum(1 for consent in subject_consents.values() if not consent["active"])
                    for subject_consents in self.consent_records.values()
                )
            }
        }
        
        return dashboard