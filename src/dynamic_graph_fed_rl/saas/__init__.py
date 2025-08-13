"""
Enterprise SaaS Platform for Federated Learning

This module provides multi-tenant enterprise capabilities for the federated
reinforcement learning framework, including:

- Multi-tenant architecture with customer isolation
- Web-based dashboard and management interface
- REST API with authentication and rate limiting
- Marketplace for algorithms and datasets
- Billing and subscription management
- Customer onboarding and support portal
- Enterprise security and compliance features
"""

from .tenant import TenantManager
from .auth import AuthenticationService
from .api import APIGateway
from .billing import BillingService
from .marketplace import MarketplaceService

__all__ = [
    "TenantManager",
    "AuthenticationService", 
    "APIGateway",
    "BillingService",
    "MarketplaceService"
]