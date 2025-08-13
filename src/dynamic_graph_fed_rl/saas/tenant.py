"""
Multi-Tenant Architecture Implementation

Provides customer isolation, resource management, and tenant-specific
configuration for the enterprise SaaS platform.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import asyncio
from enum import Enum


class TenantTier(Enum):
    """Subscription tiers for different feature access levels"""
    STARTER = "starter"
    PROFESSIONAL = "professional" 
    ENTERPRISE = "enterprise"


class ResourceQuotas:
    """Resource quotas and limits per tenant"""
    def __init__(
        self,
        max_agents: int = 10,
        max_experiments: int = 50,
        max_models: int = 100,
        max_datasets: int = 20,
        compute_hours_monthly: int = 100,
        storage_gb: int = 10,
        api_requests_per_minute: int = 1000
    ):
        self.max_agents = max_agents
        self.max_experiments = max_experiments
        self.max_models = max_models
        self.max_datasets = max_datasets
        self.compute_hours_monthly = compute_hours_monthly
        self.storage_gb = storage_gb
        self.api_requests_per_minute = api_requests_per_minute


@dataclass
class TenantConfig:
    """Tenant-specific configuration and settings"""
    tenant_id: str
    name: str
    tier: TenantTier
    created_at: datetime = field(default_factory=datetime.utcnow)
    quotas: ResourceQuotas = field(default_factory=ResourceQuotas)
    settings: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    # Compliance and security settings
    data_residency_region: str = "us-east-1"
    encryption_at_rest: bool = True
    audit_logging: bool = True
    soc2_compliant: bool = False
    gdpr_compliant: bool = False
    hipaa_compliant: bool = False


class TenantManager:
    """
    Manages multi-tenant architecture including:
    - Tenant provisioning and deprovisioning
    - Resource isolation and quotas
    - Configuration management
    - Compliance enforcement
    """
    
    def __init__(self):
        self._tenants: Dict[str, TenantConfig] = {}
        self._resource_usage: Dict[str, Dict[str, int]] = {}
        
    async def create_tenant(
        self,
        name: str,
        tier: TenantTier,
        admin_email: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> TenantConfig:
        """Create a new tenant with isolated resources"""
        tenant_id = str(uuid.uuid4())
        
        # Set quotas based on tier
        quotas = self._get_tier_quotas(tier)
        
        # Create tenant configuration
        tenant = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            quotas=quotas,
            settings=settings or {}
        )
        
        # Initialize resource usage tracking
        self._resource_usage[tenant_id] = {
            "agents": 0,
            "experiments": 0,
            "models": 0,
            "datasets": 0,
            "compute_hours": 0,
            "storage_used": 0,
            "api_requests": 0
        }
        
        # Store tenant
        self._tenants[tenant_id] = tenant
        
        # Provision tenant infrastructure
        await self._provision_tenant_infrastructure(tenant)
        
        return tenant
        
    async def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Retrieve tenant configuration"""
        return self._tenants.get(tenant_id)
        
    async def update_tenant(
        self,
        tenant_id: str,
        updates: Dict[str, Any]
    ) -> Optional[TenantConfig]:
        """Update tenant configuration"""
        if tenant_id not in self._tenants:
            return None
            
        tenant = self._tenants[tenant_id]
        
        # Update allowed fields
        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
                
        return tenant
        
    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant and clean up resources"""
        if tenant_id not in self._tenants:
            return False
            
        # Deprovision tenant infrastructure
        await self._deprovision_tenant_infrastructure(tenant_id)
        
        # Remove tenant data
        del self._tenants[tenant_id]
        del self._resource_usage[tenant_id]
        
        return True
        
    async def check_quota(
        self,
        tenant_id: str,
        resource_type: str,
        requested_amount: int = 1
    ) -> bool:
        """Check if tenant has quota available for resource"""
        if tenant_id not in self._tenants:
            return False
            
        tenant = self._tenants[tenant_id]
        usage = self._resource_usage[tenant_id]
        
        quota_map = {
            "agents": tenant.quotas.max_agents,
            "experiments": tenant.quotas.max_experiments,
            "models": tenant.quotas.max_models,
            "datasets": tenant.quotas.max_datasets,
            "compute_hours": tenant.quotas.compute_hours_monthly,
            "storage": tenant.quotas.storage_gb,
            "api_requests": tenant.quotas.api_requests_per_minute
        }
        
        if resource_type not in quota_map:
            return False
            
        current_usage = usage.get(resource_type, 0)
        quota_limit = quota_map[resource_type]
        
        return current_usage + requested_amount <= quota_limit
        
    async def update_usage(
        self,
        tenant_id: str,
        resource_type: str,
        amount: int
    ) -> None:
        """Update resource usage for tenant"""
        if tenant_id in self._resource_usage:
            if resource_type in self._resource_usage[tenant_id]:
                self._resource_usage[tenant_id][resource_type] += amount
                
    async def get_usage(self, tenant_id: str) -> Dict[str, int]:
        """Get current resource usage for tenant"""
        return self._resource_usage.get(tenant_id, {})
        
    async def list_tenants(
        self,
        active_only: bool = True
    ) -> List[TenantConfig]:
        """List all tenants"""
        tenants = list(self._tenants.values())
        
        if active_only:
            tenants = [t for t in tenants if t.is_active]
            
        return tenants
        
    def _get_tier_quotas(self, tier: TenantTier) -> ResourceQuotas:
        """Get resource quotas based on subscription tier"""
        if tier == TenantTier.STARTER:
            return ResourceQuotas(
                max_agents=5,
                max_experiments=10,
                max_models=25,
                max_datasets=5,
                compute_hours_monthly=50,
                storage_gb=5,
                api_requests_per_minute=500
            )
        elif tier == TenantTier.PROFESSIONAL:
            return ResourceQuotas(
                max_agents=25,
                max_experiments=100,
                max_models=500,
                max_datasets=50,
                compute_hours_monthly=500,
                storage_gb=100,
                api_requests_per_minute=5000
            )
        elif tier == TenantTier.ENTERPRISE:
            return ResourceQuotas(
                max_agents=100,
                max_experiments=1000,
                max_models=5000,
                max_datasets=500,
                compute_hours_monthly=5000,
                storage_gb=1000,
                api_requests_per_minute=50000
            )
        else:
            return ResourceQuotas()
            
    async def _provision_tenant_infrastructure(self, tenant: TenantConfig) -> None:
        """Provision isolated infrastructure for tenant"""
        # Create tenant-specific namespaces, databases, storage, etc.
        # This would integrate with cloud provider APIs
        pass
        
    async def _deprovision_tenant_infrastructure(self, tenant_id: str) -> None:
        """Clean up tenant infrastructure"""
        # Remove tenant-specific resources
        pass


# Global tenant manager instance
tenant_manager = TenantManager()