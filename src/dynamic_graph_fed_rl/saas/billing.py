"""
Billing and Subscription Management System

Provides comprehensive billing, subscription management, usage tracking,
and payment processing for the enterprise SaaS platform.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
from decimal import Decimal


class SubscriptionStatus(Enum):
    """Subscription status options"""
    ACTIVE = "active"
    CANCELED = "canceled" 
    PAST_DUE = "past_due"
    UNPAID = "unpaid"
    TRIALING = "trialing"
    PAUSED = "paused"


class BillingCycle(Enum):
    """Billing cycle options"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class PaymentStatus(Enum):
    """Payment status options"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    REFUNDED = "refunded"


class UsageMetricType(Enum):
    """Types of usage metrics for billing"""
    COMPUTE_HOURS = "compute_hours"
    STORAGE_GB = "storage_gb"
    API_REQUESTS = "api_requests"
    DATA_TRANSFER_GB = "data_transfer_gb"
    MODEL_TRAINING_JOBS = "model_training_jobs"
    AGENTS = "agents"
    EXPERIMENTS = "experiments"


@dataclass
class PricingTier:
    """Pricing tier configuration"""
    tier_id: str
    name: str
    monthly_price: Decimal
    quarterly_price: Decimal
    annual_price: Decimal
    
    # Base quotas included in subscription
    base_compute_hours: int = 0
    base_storage_gb: int = 0
    base_api_requests: int = 0
    base_agents: int = 0
    base_experiments: int = 0
    
    # Overage pricing per unit
    overage_compute_hour: Decimal = Decimal('0.10')
    overage_storage_gb: Decimal = Decimal('0.05')
    overage_api_request: Decimal = Decimal('0.001')
    
    # Features included
    features: List[str] = field(default_factory=list)


@dataclass
class Subscription:
    """Customer subscription details"""
    subscription_id: str
    tenant_id: str
    pricing_tier_id: str
    status: SubscriptionStatus
    billing_cycle: BillingCycle
    
    # Billing details
    current_period_start: datetime
    current_period_end: datetime
    next_billing_date: datetime
    
    # Trial information
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    canceled_at: Optional[datetime] = None
    
    # Custom pricing overrides
    custom_pricing: Optional[Dict[str, Decimal]] = None


@dataclass
class UsageRecord:
    """Usage tracking record"""
    record_id: str
    tenant_id: str
    subscription_id: str
    metric_type: UsageMetricType
    quantity: float
    timestamp: datetime
    billing_period_start: datetime
    billing_period_end: datetime
    
    # Metadata for detailed tracking
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Invoice:
    """Invoice for billing period"""
    invoice_id: str
    tenant_id: str
    subscription_id: str
    
    # Billing period
    period_start: datetime
    period_end: datetime
    
    # Amounts
    subscription_amount: Decimal
    usage_amount: Decimal
    discount_amount: Decimal = Decimal('0.00')
    tax_amount: Decimal = Decimal('0.00')
    total_amount: Decimal = Decimal('0.00')
    
    # Status and timestamps
    status: PaymentStatus = PaymentStatus.PENDING
    issued_at: datetime = field(default_factory=datetime.utcnow)
    due_date: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    paid_at: Optional[datetime] = None
    
    # Line items
    line_items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PaymentMethod:
    """Customer payment method"""
    payment_method_id: str
    tenant_id: str
    type: str  # 'card', 'bank', 'wallet'
    
    # Card details (tokenized)
    card_last_four: Optional[str] = None
    card_brand: Optional[str] = None
    card_exp_month: Optional[int] = None
    card_exp_year: Optional[int] = None
    
    # Status
    is_default: bool = False
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


class BillingService:
    """
    Comprehensive billing service providing:
    - Subscription management
    - Usage tracking and metering
    - Invoice generation
    - Payment processing integration
    - Revenue analytics
    - Billing automation
    """
    
    def __init__(self):
        # Pricing tiers configuration
        self._pricing_tiers = self._initialize_pricing_tiers()
        
        # Data storage
        self._subscriptions: Dict[str, Subscription] = {}
        self._usage_records: Dict[str, List[UsageRecord]] = {}  # tenant_id -> records
        self._invoices: Dict[str, Invoice] = {}
        self._payment_methods: Dict[str, PaymentMethod] = {}
        
    def _initialize_pricing_tiers(self) -> Dict[str, PricingTier]:
        """Initialize default pricing tiers"""
        return {
            "starter": PricingTier(
                tier_id="starter",
                name="Starter",
                monthly_price=Decimal('99.00'),
                quarterly_price=Decimal('267.00'),  # 10% discount
                annual_price=Decimal('950.00'),     # 20% discount
                base_compute_hours=50,
                base_storage_gb=5,
                base_api_requests=10000,
                base_agents=5,
                base_experiments=10,
                features=[
                    "Basic federated learning",
                    "Standard support",
                    "Community algorithms"
                ]
            ),
            "professional": PricingTier(
                tier_id="professional", 
                name="Professional",
                monthly_price=Decimal('499.00'),
                quarterly_price=Decimal('1347.00'),  # 10% discount
                annual_price=Decimal('4791.00'),     # 20% discount
                base_compute_hours=500,
                base_storage_gb=100,
                base_api_requests=100000,
                base_agents=25,
                base_experiments=100,
                features=[
                    "Advanced federated learning",
                    "Priority support",
                    "Premium algorithms",
                    "Custom models",
                    "Analytics dashboard"
                ]
            ),
            "enterprise": PricingTier(
                tier_id="enterprise",
                name="Enterprise", 
                monthly_price=Decimal('2499.00'),
                quarterly_price=Decimal('6747.00'),  # 10% discount
                annual_price=Decimal('23991.00'),    # 20% discount
                base_compute_hours=5000,
                base_storage_gb=1000,
                base_api_requests=1000000,
                base_agents=100,
                base_experiments=1000,
                features=[
                    "Enterprise federated learning",
                    "24/7 dedicated support",
                    "Custom algorithms",
                    "On-premise deployment",
                    "Advanced analytics",
                    "SLA guarantees",
                    "Compliance features"
                ]
            )
        }
        
    async def create_subscription(
        self,
        tenant_id: str,
        pricing_tier_id: str,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
        payment_method_id: Optional[str] = None,
        trial_days: int = 14
    ) -> Subscription:
        """Create a new subscription"""
        if pricing_tier_id not in self._pricing_tiers:
            raise ValueError(f"Invalid pricing tier: {pricing_tier_id}")
            
        subscription_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Set up trial period
        trial_start = now
        trial_end = now + timedelta(days=trial_days)
        
        # Set billing period after trial
        current_period_start = trial_end
        if billing_cycle == BillingCycle.MONTHLY:
            current_period_end = current_period_start + timedelta(days=30)
        elif billing_cycle == BillingCycle.QUARTERLY:
            current_period_end = current_period_start + timedelta(days=90)
        else:  # ANNUALLY
            current_period_end = current_period_start + timedelta(days=365)
            
        subscription = Subscription(
            subscription_id=subscription_id,
            tenant_id=tenant_id,
            pricing_tier_id=pricing_tier_id,
            status=SubscriptionStatus.TRIALING,
            billing_cycle=billing_cycle,
            current_period_start=current_period_start,
            current_period_end=current_period_end,
            next_billing_date=current_period_start,
            trial_start=trial_start,
            trial_end=trial_end
        )
        
        self._subscriptions[subscription_id] = subscription
        self._usage_records[tenant_id] = []
        
        return subscription
        
    async def record_usage(
        self,
        tenant_id: str,
        metric_type: UsageMetricType,
        quantity: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageRecord:
        """Record usage for billing calculations"""
        subscription = await self.get_active_subscription(tenant_id)
        if not subscription:
            raise ValueError(f"No active subscription for tenant: {tenant_id}")
            
        record_id = str(uuid.uuid4())
        
        record = UsageRecord(
            record_id=record_id,
            tenant_id=tenant_id,
            subscription_id=subscription.subscription_id,
            metric_type=metric_type,
            quantity=quantity,
            timestamp=datetime.utcnow(),
            billing_period_start=subscription.current_period_start,
            billing_period_end=subscription.current_period_end,
            metadata=metadata or {}
        )
        
        if tenant_id not in self._usage_records:
            self._usage_records[tenant_id] = []
        self._usage_records[tenant_id].append(record)
        
        return record
        
    async def get_current_usage(
        self,
        tenant_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> Dict[UsageMetricType, float]:
        """Get current usage aggregated by metric type"""
        subscription = await self.get_active_subscription(tenant_id)
        if not subscription:
            return {}
            
        if not period_start:
            period_start = subscription.current_period_start
        if not period_end:
            period_end = subscription.current_period_end
            
        usage_summary = {}
        records = self._usage_records.get(tenant_id, [])
        
        for record in records:
            if period_start <= record.timestamp <= period_end:
                metric_type = record.metric_type
                if metric_type not in usage_summary:
                    usage_summary[metric_type] = 0.0
                usage_summary[metric_type] += record.quantity
                
        return usage_summary
        
    async def generate_invoice(
        self,
        subscription_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Invoice:
        """Generate invoice for billing period"""
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription not found: {subscription_id}")
            
        pricing_tier = self._pricing_tiers[subscription.pricing_tier_id]
        invoice_id = str(uuid.uuid4())
        
        # Calculate subscription amount
        if subscription.billing_cycle == BillingCycle.MONTHLY:
            subscription_amount = pricing_tier.monthly_price
        elif subscription.billing_cycle == BillingCycle.QUARTERLY:
            subscription_amount = pricing_tier.quarterly_price
        else:  # ANNUALLY
            subscription_amount = pricing_tier.annual_price
            
        # Calculate usage overages
        current_usage = await self.get_current_usage(
            subscription.tenant_id,
            period_start,
            period_end
        )
        
        usage_amount = Decimal('0.00')
        line_items = []
        
        # Subscription base fee
        line_items.append({
            "description": f"{pricing_tier.name} Subscription",
            "quantity": 1,
            "unit_price": subscription_amount,
            "total": subscription_amount
        })
        
        # Usage overages
        compute_hours = current_usage.get(UsageMetricType.COMPUTE_HOURS, 0)
        if compute_hours > pricing_tier.base_compute_hours:
            overage = compute_hours - pricing_tier.base_compute_hours
            overage_cost = Decimal(str(overage)) * pricing_tier.overage_compute_hour
            usage_amount += overage_cost
            line_items.append({
                "description": "Compute Hours Overage",
                "quantity": overage,
                "unit_price": pricing_tier.overage_compute_hour,
                "total": overage_cost
            })
            
        storage_gb = current_usage.get(UsageMetricType.STORAGE_GB, 0)
        if storage_gb > pricing_tier.base_storage_gb:
            overage = storage_gb - pricing_tier.base_storage_gb
            overage_cost = Decimal(str(overage)) * pricing_tier.overage_storage_gb
            usage_amount += overage_cost
            line_items.append({
                "description": "Storage Overage (GB)",
                "quantity": overage,
                "unit_price": pricing_tier.overage_storage_gb,
                "total": overage_cost
            })
            
        # Calculate totals
        total_amount = subscription_amount + usage_amount
        
        invoice = Invoice(
            invoice_id=invoice_id,
            tenant_id=subscription.tenant_id,
            subscription_id=subscription_id,
            period_start=period_start,
            period_end=period_end,
            subscription_amount=subscription_amount,
            usage_amount=usage_amount,
            total_amount=total_amount,
            line_items=line_items
        )
        
        self._invoices[invoice_id] = invoice
        return invoice
        
    async def get_active_subscription(self, tenant_id: str) -> Optional[Subscription]:
        """Get the active subscription for a tenant"""
        for subscription in self._subscriptions.values():
            if (subscription.tenant_id == tenant_id and 
                subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]):
                return subscription
        return None
        
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False
    ) -> bool:
        """Cancel a subscription"""
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            return False
            
        subscription.status = SubscriptionStatus.CANCELED
        subscription.canceled_at = datetime.utcnow()
        subscription.updated_at = datetime.utcnow()
        
        if not immediate:
            # Cancel at end of current period
            pass
        else:
            # Cancel immediately
            subscription.current_period_end = datetime.utcnow()
            
        return True
        
    async def update_subscription_tier(
        self,
        subscription_id: str,
        new_pricing_tier_id: str
    ) -> bool:
        """Update subscription to different pricing tier"""
        subscription = self._subscriptions.get(subscription_id)
        if not subscription or new_pricing_tier_id not in self._pricing_tiers:
            return False
            
        subscription.pricing_tier_id = new_pricing_tier_id
        subscription.updated_at = datetime.utcnow()
        
        return True
        
    async def process_payment(
        self,
        invoice_id: str,
        payment_method_id: str
    ) -> bool:
        """Process payment for an invoice"""
        invoice = self._invoices.get(invoice_id)
        if not invoice:
            return False
            
        # Mock payment processing - integrate with Stripe, etc.
        success = True  # Assume successful payment
        
        if success:
            invoice.status = PaymentStatus.SUCCESS
            invoice.paid_at = datetime.utcnow()
        else:
            invoice.status = PaymentStatus.FAILED
            
        return success
        
    async def get_billing_analytics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get billing analytics for date range"""
        total_revenue = Decimal('0.00')
        total_invoices = 0
        successful_payments = 0
        
        for invoice in self._invoices.values():
            if start_date <= invoice.issued_at <= end_date:
                total_invoices += 1
                if invoice.status == PaymentStatus.SUCCESS:
                    total_revenue += invoice.total_amount
                    successful_payments += 1
                    
        return {
            "total_revenue": float(total_revenue),
            "total_invoices": total_invoices,
            "successful_payments": successful_payments,
            "payment_success_rate": successful_payments / total_invoices if total_invoices > 0 else 0,
            "subscription_count": len(self._subscriptions),
            "active_subscriptions": sum(
                1 for s in self._subscriptions.values() 
                if s.status == SubscriptionStatus.ACTIVE
            )
        }


# Global billing service instance
billing_service = BillingService()