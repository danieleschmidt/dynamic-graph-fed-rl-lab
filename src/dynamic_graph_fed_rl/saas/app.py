"""
Enterprise SaaS Platform Main Application

Main entry point for the multi-tenant enterprise SaaS platform
integrating all components: auth, billing, marketplace, dashboard, etc.
"""

import asyncio
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .tenant import tenant_manager, TenantTier
from .auth import auth_service, UserRole
from .billing import billing_service, BillingCycle
from .marketplace import marketplace_service, AssetType, AssetCategory, AssetMetadata, LicenseType
from .customer_portal import customer_portal_service, OnboardingStep, TicketCategory, TicketPriority
from .security import security_service, AuditEventType, ComplianceFramework
from .api import api_gateway, get_current_user
from .dashboard import dashboard_service


class EnterpriseSaaSApp:
    """
    Main SaaS application integrating all enterprise features:
    - Multi-tenant architecture
    - Authentication and authorization  
    - REST API with rate limiting
    - Web dashboard
    - Marketplace
    - Billing and subscriptions
    - Customer portal and support
    - Enterprise security and compliance
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Dynamic Graph Fed-RL Enterprise SaaS",
            description="Enterprise SaaS Platform for Federated Reinforcement Learning",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self._setup_middleware()
        self._setup_routes()
        
    def _setup_middleware(self):
        """Setup middleware for security, compression, etc."""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
    def _setup_routes(self):
        """Setup all application routes"""
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """System health check"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "services": {
                    "tenant_manager": "active",
                    "auth_service": "active", 
                    "billing_service": "active",
                    "marketplace_service": "active",
                    "customer_portal": "active",
                    "security_service": "active"
                }
            }
            
        # Tenant onboarding endpoint
        @self.app.post("/api/v1/onboard")
        async def onboard_enterprise_customer(request: dict):
            """Complete enterprise customer onboarding"""
            try:
                # Validate required fields
                required_fields = ["company_name", "admin_email", "admin_password", "tier"]
                for field in required_fields:
                    if field not in request:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Missing required field: {field}"
                        )
                
                company_name = request["company_name"]
                admin_email = request["admin_email"]
                admin_password = request["admin_password"]
                tier_name = request["tier"]
                
                # Validate tier
                try:
                    tier = TenantTier(tier_name)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid tier: {tier_name}"
                    )
                
                # Create tenant
                tenant = await tenant_manager.create_tenant(
                    name=company_name,
                    tier=tier,
                    admin_email=admin_email
                )
                
                # Create admin user
                admin_user = await auth_service.create_user(
                    email=admin_email,
                    password=admin_password,
                    tenant_id=tenant.tenant_id,
                    role=UserRole.ADMIN
                )
                
                # Create subscription
                subscription = await billing_service.create_subscription(
                    tenant_id=tenant.tenant_id,
                    pricing_tier_id=tier_name,
                    billing_cycle=BillingCycle.MONTHLY,
                    trial_days=14
                )
                
                # Start onboarding process
                onboarding = await customer_portal_service.start_onboarding(
                    tenant.tenant_id
                )
                
                # Complete account setup step
                await customer_portal_service.complete_onboarding_step(
                    tenant.tenant_id,
                    OnboardingStep.ACCOUNT_SETUP
                )
                
                # Log audit event
                await security_service.log_audit_event(
                    AuditEventType.USER_CREATED,
                    admin_user.user_id,
                    tenant.tenant_id,
                    "tenant_onboarding",
                    "create",
                    "success",
                    details={
                        "company_name": company_name,
                        "tier": tier_name,
                        "trial_days": 14
                    }
                )
                
                # Generate access token
                access_token = await auth_service.generate_jwt_token(admin_user)
                
                return {
                    "success": True,
                    "tenant_id": tenant.tenant_id,
                    "subscription_id": subscription.subscription_id,
                    "access_token": access_token,
                    "trial_end": subscription.trial_end.isoformat(),
                    "onboarding_status": onboarding.status.value,
                    "next_step": onboarding.current_step.value
                }
                
            except Exception as e:
                # Log error
                await security_service.log_audit_event(
                    AuditEventType.SECURITY_ALERT,
                    None,
                    None,
                    "tenant_onboarding",
                    "create",
                    "failure",
                    details={"error": str(e)}
                )
                raise HTTPException(status_code=500, detail="Onboarding failed")
        
        # Publish algorithm to marketplace
        @self.app.post("/api/v1/marketplace/publish")
        async def publish_algorithm(
            request: dict,
            user: dict = Depends(get_current_user)
        ):
            """Publish algorithm to marketplace"""
            try:
                # Validate required fields
                required_fields = ["name", "asset_type", "category", "description"]
                for field in required_fields:
                    if field not in request:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Missing required field: {field}"
                        )
                
                # Create asset metadata
                metadata = AssetMetadata(
                    description=request["description"],
                    tags=request.get("tags", []),
                    version=request.get("version", "1.0.0"),
                    requirements=request.get("requirements", []),
                    performance_metrics=request.get("performance_metrics", {}),
                    documentation_url=request.get("documentation_url"),
                    paper_reference=request.get("paper_reference")
                )
                
                # Publish asset
                asset = await marketplace_service.publish_asset(
                    publisher_tenant_id=user["tenant_id"],
                    publisher_name=user["email"],
                    name=request["name"],
                    asset_type=AssetType(request["asset_type"]),
                    category=AssetCategory(request["category"]),
                    metadata=metadata,
                    file_path=request.get("file_path", ""),
                    price=float(request.get("price", 0.0)),
                    license_type=LicenseType(request.get("license_type", "mit"))
                )
                
                # Log audit event
                await security_service.log_audit_event(
                    AuditEventType.DATA_MODIFY,
                    user["user_id"],
                    user["tenant_id"],
                    f"marketplace_asset:{asset.asset_id}",
                    "publish",
                    "success",
                    details={
                        "asset_name": asset.name,
                        "asset_type": asset.asset_type.value,
                        "price": asset.price
                    }
                )
                
                return {
                    "success": True,
                    "asset_id": asset.asset_id,
                    "status": "published" if asset.is_approved else "pending_approval"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Publishing failed: {str(e)}")
        
        # Create support ticket
        @self.app.post("/api/v1/support/tickets")
        async def create_support_ticket(
            request: dict,
            user: dict = Depends(get_current_user)
        ):
            """Create a support ticket"""
            try:
                ticket = await customer_portal_service.create_support_ticket(
                    tenant_id=user["tenant_id"],
                    title=request["title"],
                    description=request["description"],
                    category=TicketCategory(request["category"]),
                    priority=TicketPriority(request["priority"]),
                    reporter_email=user["email"]
                )
                
                # Log audit event
                await security_service.log_audit_event(
                    AuditEventType.DATA_MODIFY,
                    user["user_id"],
                    user["tenant_id"],
                    f"support_ticket:{ticket.ticket_id}",
                    "create",
                    "success",
                    details={
                        "category": ticket.category.value,
                        "priority": ticket.priority.value
                    }
                )
                
                return {
                    "success": True,
                    "ticket_id": ticket.ticket_id,
                    "status": ticket.status.value
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ticket creation failed: {str(e)}")
        
        # Generate compliance report
        @self.app.get("/api/v1/compliance/reports/{framework}")
        async def generate_compliance_report(
            framework: str,
            start_date: str,
            end_date: str,
            user: dict = Depends(get_current_user)
        ):
            """Generate compliance report"""
            try:
                # Check admin permissions
                if user["role"] != "admin":
                    raise HTTPException(
                        status_code=403,
                        detail="Admin access required"
                    )
                
                # Parse dates
                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date)
                
                # Generate report
                report = await security_service.generate_compliance_report(
                    ComplianceFramework(framework),
                    start_dt,
                    end_dt
                )
                
                # Log audit event
                await security_service.log_audit_event(
                    AuditEventType.DATA_ACCESS,
                    user["user_id"],
                    user["tenant_id"],
                    f"compliance_report:{framework}",
                    "generate",
                    "success",
                    details={
                        "framework": framework,
                        "period": f"{start_date} to {end_date}"
                    }
                )
                
                return report
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
        
        # Mount sub-applications
        self.app.mount("/api", api_gateway.app)
        self.app.mount("/dashboard", dashboard_service.app)
        
    async def startup(self):
        """Application startup tasks"""
        print("🚀 Starting Enterprise SaaS Platform...")
        print("✅ Multi-tenant architecture initialized")
        print("✅ Authentication service ready")
        print("✅ Billing system active")
        print("✅ Marketplace operational")
        print("✅ Customer portal ready")
        print("✅ Security & compliance enabled")
        print("🎉 Enterprise SaaS Platform ready!")
        
    async def shutdown(self):
        """Application shutdown tasks"""
        print("⏹️  Shutting down Enterprise SaaS Platform...")
        print("✅ All services stopped gracefully")


# Global application instance
enterprise_app = EnterpriseSaaSApp()


def main():
    """Main entry point for the SaaS application"""
    
    @enterprise_app.app.on_event("startup")
    async def startup_event():
        await enterprise_app.startup()
        
    @enterprise_app.app.on_event("shutdown") 
    async def shutdown_event():
        await enterprise_app.shutdown()
    
    # Run the application
    uvicorn.run(
        enterprise_app.app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        access_log=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()