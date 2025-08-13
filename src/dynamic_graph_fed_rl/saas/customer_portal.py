"""
Customer Onboarding and Support Portal

Provides comprehensive customer onboarding, support ticketing,
documentation, and self-service capabilities.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import asyncio


class OnboardingStatus(Enum):
    """Customer onboarding status"""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class OnboardingStep(Enum):
    """Steps in the onboarding process"""
    ACCOUNT_SETUP = "account_setup"
    TEAM_INVITATION = "team_invitation"
    FIRST_EXPERIMENT = "first_experiment"
    INTEGRATION_SETUP = "integration_setup"
    TRAINING_COMPLETION = "training_completion"


class TicketStatus(Enum):
    """Support ticket status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(Enum):
    """Support ticket priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class TicketCategory(Enum):
    """Support ticket categories"""
    TECHNICAL = "technical"
    BILLING = "billing"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    GENERAL = "general"
    API_SUPPORT = "api_support"
    TRAINING = "training"


@dataclass
class OnboardingProgress:
    """Track customer onboarding progress"""
    tenant_id: str
    status: OnboardingStatus
    current_step: OnboardingStep
    completed_steps: List[OnboardingStep] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    notes: str = ""


@dataclass
class SupportTicket:
    """Support ticket for customer issues"""
    ticket_id: str
    tenant_id: str
    title: str
    description: str
    category: TicketCategory
    priority: TicketPriority
    status: TicketStatus
    
    # Assignment and tracking
    assigned_to: Optional[str] = None
    reporter_email: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    # Additional info
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TicketComment:
    """Comment on a support ticket"""
    comment_id: str
    ticket_id: str
    author_email: str
    author_name: str
    content: str
    is_internal: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    attachments: List[str] = field(default_factory=list)


@dataclass
class KnowledgeBaseArticle:
    """Self-service knowledge base article"""
    article_id: str
    title: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    view_count: int = 0
    helpful_count: int = 0
    not_helpful_count: int = 0
    
    # Status
    is_published: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class CustomerPortalService:
    """
    Customer portal service providing:
    - Guided onboarding workflows
    - Support ticket management  
    - Knowledge base and documentation
    - Self-service capabilities
    - Training and tutorial resources
    - Community forums
    """
    
    def __init__(self):
        self._onboarding_progress: Dict[str, OnboardingProgress] = {}
        self._tickets: Dict[str, SupportTicket] = {}
        self._ticket_comments: Dict[str, List[TicketComment]] = {}
        self._knowledge_base: Dict[str, KnowledgeBaseArticle] = {}
        
        # Initialize knowledge base
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self):
        """Initialize knowledge base with common articles"""
        articles = [
            {
                "title": "Getting Started with Federated Learning",
                "content": """# Getting Started with Federated Learning

## What is Federated Learning?

Federated Learning is a machine learning technique that enables model training across decentralized data sources without centralizing the data.

## Quick Start Guide

1. **Create your first experiment**
2. **Configure agents**
3. **Start training**
4. **Monitor progress**

## Next Steps

- Explore the marketplace for pre-built algorithms
- Join our community forum
- Check out advanced tutorials
""",
                "category": "getting-started",
                "tags": ["beginner", "tutorial", "federated-learning"]
            },
            {
                "title": "API Authentication Guide",
                "content": """# API Authentication Guide

## Overview

All API requests require authentication using JWT tokens or API keys.

## Getting Your API Key

1. Navigate to Settings > API Keys
2. Click "Generate New Key"
3. Copy and store securely

## Making Authenticated Requests

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \\
     https://api.example.com/v1/experiments
```

## Best Practices

- Rotate keys regularly
- Use environment variables
- Monitor usage
""",
                "category": "api",
                "tags": ["api", "authentication", "security"]
            },
            {
                "title": "Troubleshooting Common Issues",
                "content": """# Troubleshooting Common Issues

## Agent Connection Problems

**Symptom**: Agents fail to connect
**Solution**: Check network connectivity and firewall settings

## Training Convergence Issues

**Symptom**: Models don't converge
**Solution**: Adjust learning rates and check data quality

## Performance Optimization

- Use GPU acceleration
- Optimize batch sizes
- Monitor resource usage
""",
                "category": "troubleshooting",
                "tags": ["troubleshooting", "performance", "agents"]
            }
        ]
        
        for article_data in articles:
            article_id = str(uuid.uuid4())
            article = KnowledgeBaseArticle(
                article_id=article_id,
                **article_data
            )
            self._knowledge_base[article_id] = article
            
    async def start_onboarding(self, tenant_id: str) -> OnboardingProgress:
        """Start the onboarding process for a new customer"""
        progress = OnboardingProgress(
            tenant_id=tenant_id,
            status=OnboardingStatus.STARTED,
            current_step=OnboardingStep.ACCOUNT_SETUP
        )
        
        self._onboarding_progress[tenant_id] = progress
        return progress
        
    async def complete_onboarding_step(
        self,
        tenant_id: str,
        step: OnboardingStep
    ) -> Optional[OnboardingProgress]:
        """Mark an onboarding step as completed"""
        progress = self._onboarding_progress.get(tenant_id)
        if not progress:
            return None
            
        if step not in progress.completed_steps:
            progress.completed_steps.append(step)
            
        # Determine next step
        step_order = [
            OnboardingStep.ACCOUNT_SETUP,
            OnboardingStep.TEAM_INVITATION,
            OnboardingStep.FIRST_EXPERIMENT,
            OnboardingStep.INTEGRATION_SETUP,
            OnboardingStep.TRAINING_COMPLETION
        ]
        
        current_index = step_order.index(step)
        if current_index < len(step_order) - 1:
            progress.current_step = step_order[current_index + 1]
            progress.status = OnboardingStatus.IN_PROGRESS
        else:
            progress.status = OnboardingStatus.COMPLETED
            progress.completed_at = datetime.utcnow()
            
        return progress
        
    async def get_onboarding_progress(self, tenant_id: str) -> Optional[OnboardingProgress]:
        """Get onboarding progress for a tenant"""
        return self._onboarding_progress.get(tenant_id)
        
    async def create_support_ticket(
        self,
        tenant_id: str,
        title: str,
        description: str,
        category: TicketCategory,
        priority: TicketPriority,
        reporter_email: str
    ) -> SupportTicket:
        """Create a new support ticket"""
        ticket_id = str(uuid.uuid4())
        
        ticket = SupportTicket(
            ticket_id=ticket_id,
            tenant_id=tenant_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            status=TicketStatus.OPEN,
            reporter_email=reporter_email
        )
        
        self._tickets[ticket_id] = ticket
        self._ticket_comments[ticket_id] = []
        
        return ticket
        
    async def add_ticket_comment(
        self,
        ticket_id: str,
        author_email: str,
        author_name: str,
        content: str,
        is_internal: bool = False
    ) -> Optional[TicketComment]:
        """Add a comment to a support ticket"""
        if ticket_id not in self._tickets:
            return None
            
        comment_id = str(uuid.uuid4())
        
        comment = TicketComment(
            comment_id=comment_id,
            ticket_id=ticket_id,
            author_email=author_email,
            author_name=author_name,
            content=content,
            is_internal=is_internal
        )
        
        self._ticket_comments[ticket_id].append(comment)
        
        # Update ticket timestamp
        self._tickets[ticket_id].updated_at = datetime.utcnow()
        
        return comment
        
    async def update_ticket_status(
        self,
        ticket_id: str,
        status: TicketStatus,
        assigned_to: Optional[str] = None
    ) -> bool:
        """Update ticket status and assignment"""
        if ticket_id not in self._tickets:
            return False
            
        ticket = self._tickets[ticket_id]
        ticket.status = status
        ticket.updated_at = datetime.utcnow()
        
        if assigned_to:
            ticket.assigned_to = assigned_to
            
        if status == TicketStatus.RESOLVED:
            ticket.resolved_at = datetime.utcnow()
            
        return True
        
    async def search_tickets(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[TicketStatus] = None,
        category: Optional[TicketCategory] = None,
        priority: Optional[TicketPriority] = None
    ) -> List[SupportTicket]:
        """Search support tickets with filters"""
        results = []
        
        for ticket in self._tickets.values():
            if tenant_id and ticket.tenant_id != tenant_id:
                continue
            if status and ticket.status != status:
                continue
            if category and ticket.category != category:
                continue
            if priority and ticket.priority != priority:
                continue
                
            results.append(ticket)
            
        # Sort by priority and creation date
        priority_order = {
            TicketPriority.CRITICAL: 0,
            TicketPriority.URGENT: 1,
            TicketPriority.HIGH: 2,
            TicketPriority.MEDIUM: 3,
            TicketPriority.LOW: 4
        }
        
        results.sort(key=lambda x: (
            priority_order.get(x.priority, 5),
            x.created_at
        ))
        
        return results
        
    async def get_ticket_with_comments(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """Get ticket with all comments"""
        if ticket_id not in self._tickets:
            return None
            
        ticket = self._tickets[ticket_id]
        comments = self._ticket_comments.get(ticket_id, [])
        
        return {
            "ticket": ticket,
            "comments": comments
        }
        
    async def search_knowledge_base(
        self,
        query: str = "",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[KnowledgeBaseArticle]:
        """Search knowledge base articles"""
        results = []
        
        for article in self._knowledge_base.values():
            if not article.is_published:
                continue
                
            # Filter by category
            if category and article.category != category:
                continue
                
            # Filter by tags
            if tags:
                article_tags = [tag.lower() for tag in article.tags]
                if not any(tag.lower() in article_tags for tag in tags):
                    continue
                    
            # Filter by search query
            if query:
                query_lower = query.lower()
                if (query_lower not in article.title.lower() and 
                    query_lower not in article.content.lower()):
                    continue
                    
            results.append(article)
            
        # Sort by relevance (view count and helpful votes)
        results.sort(
            key=lambda x: (x.helpful_count, x.view_count),
            reverse=True
        )
        
        return results
        
    async def view_article(self, article_id: str) -> Optional[KnowledgeBaseArticle]:
        """View knowledge base article and increment view count"""
        article = self._knowledge_base.get(article_id)
        if article and article.is_published:
            article.view_count += 1
            return article
        return None
        
    async def rate_article(self, article_id: str, helpful: bool) -> bool:
        """Rate knowledge base article as helpful or not helpful"""
        article = self._knowledge_base.get(article_id)
        if not article:
            return False
            
        if helpful:
            article.helpful_count += 1
        else:
            article.not_helpful_count += 1
            
        return True
        
    async def get_support_metrics(self) -> Dict[str, Any]:
        """Get support metrics and analytics"""
        total_tickets = len(self._tickets)
        open_tickets = sum(1 for t in self._tickets.values() if t.status == TicketStatus.OPEN)
        resolved_tickets = sum(1 for t in self._tickets.values() if t.status == TicketStatus.RESOLVED)
        
        # Calculate average resolution time
        resolution_times = []
        for ticket in self._tickets.values():
            if ticket.resolved_at:
                resolution_time = (ticket.resolved_at - ticket.created_at).total_seconds() / 3600  # hours
                resolution_times.append(resolution_time)
                
        avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        
        # Category distribution
        category_counts = {}
        for ticket in self._tickets.values():
            category = ticket.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
        return {
            "total_tickets": total_tickets,
            "open_tickets": open_tickets,
            "resolved_tickets": resolved_tickets,
            "resolution_rate": resolved_tickets / total_tickets if total_tickets > 0 else 0,
            "avg_resolution_time_hours": avg_resolution_time,
            "category_distribution": category_counts,
            "knowledge_base_articles": len(self._knowledge_base),
            "total_kb_views": sum(a.view_count for a in self._knowledge_base.values())
        }
        
    async def get_onboarding_metrics(self) -> Dict[str, Any]:
        """Get onboarding completion metrics"""
        total_onboardings = len(self._onboarding_progress)
        completed = sum(
            1 for p in self._onboarding_progress.values() 
            if p.status == OnboardingStatus.COMPLETED
        )
        in_progress = sum(
            1 for p in self._onboarding_progress.values()
            if p.status == OnboardingStatus.IN_PROGRESS
        )
        abandoned = sum(
            1 for p in self._onboarding_progress.values()
            if p.status == OnboardingStatus.ABANDONED
        )
        
        # Step completion rates
        step_completion = {}
        for step in OnboardingStep:
            completed_step = sum(
                1 for p in self._onboarding_progress.values()
                if step in p.completed_steps
            )
            step_completion[step.value] = completed_step / total_onboardings if total_onboardings > 0 else 0
            
        return {
            "total_onboardings": total_onboardings,
            "completed": completed,
            "in_progress": in_progress,
            "abandoned": abandoned,
            "completion_rate": completed / total_onboardings if total_onboardings > 0 else 0,
            "step_completion_rates": step_completion
        }


# Global customer portal service instance
customer_portal_service = CustomerPortalService()