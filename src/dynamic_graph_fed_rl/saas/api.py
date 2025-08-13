"""
API Gateway and REST API Implementation

Provides HTTP REST API with authentication, rate limiting,
and integration with the federated learning platform.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import time
from collections import defaultdict
import json

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from .auth import auth_service, Permission
from .tenant import tenant_manager


class RateLimiter:
    """Rate limiting implementation using token bucket algorithm"""
    
    def __init__(self):
        self._buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tokens": 0, "last_refill": time.time()}
        )
        
    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60
    ) -> bool:
        """Check if request is allowed within rate limit"""
        now = time.time()
        bucket = self._buckets[key]
        
        # Calculate tokens to add based on time elapsed
        elapsed = now - bucket["last_refill"]
        tokens_to_add = elapsed * (max_requests / window_seconds)
        
        # Update bucket
        bucket["tokens"] = min(max_requests, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        # Check if request allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
            
        return False


# Pydantic models for API requests/responses
class ExperimentCreate(BaseModel):
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    algorithm_config: Dict[str, Any] = Field(..., description="Algorithm configuration")
    dataset_ids: List[str] = Field(..., description="Dataset IDs to use")
    max_agents: int = Field(10, description="Maximum number of agents")


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    description: Optional[str]
    status: str
    created_at: datetime
    tenant_id: str


class AgentCreate(BaseModel):
    name: str = Field(..., description="Agent name")
    experiment_id: str = Field(..., description="Associated experiment ID")
    config: Dict[str, Any] = Field({}, description="Agent configuration")


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    experiment_id: str
    status: str
    created_at: datetime
    metrics: Dict[str, float]


class ModelCreate(BaseModel):
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    algorithm_type: str = Field(..., description="Algorithm type")
    hyperparameters: Dict[str, Any] = Field({}, description="Model hyperparameters")


class ModelResponse(BaseModel):
    model_id: str
    name: str
    description: Optional[str]
    algorithm_type: str
    version: str
    created_at: datetime
    metrics: Dict[str, float]


# Authentication dependency
security = HTTPBearer()
rate_limiter = RateLimiter()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Validate JWT token and return user info"""
    token = credentials.credentials
    payload = await auth_service.verify_jwt_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
        
    return payload


async def check_rate_limit(request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    """Check rate limits for API requests"""
    tenant_id = user["tenant_id"]
    tenant = await tenant_manager.get_tenant(tenant_id)
    
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
        
    # Use tenant-specific rate limit
    max_requests = tenant.quotas.api_requests_per_minute
    client_ip = request.client.host
    rate_key = f"{tenant_id}:{client_ip}"
    
    if not await rate_limiter.is_allowed(rate_key, max_requests, 60):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )


def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(user: Dict[str, Any] = Depends(get_current_user)):
        user_permissions = {Permission(p) for p in user["permissions"]}
        
        if permission not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission required: {permission.value}"
            )
        return user
    return decorator


class APIGateway:
    """
    Main API Gateway providing REST endpoints for:
    - Experiment management
    - Agent management 
    - Model management
    - Dataset management
    - Marketplace integration
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Dynamic Graph Fed-RL Enterprise API",
            description="Enterprise SaaS API for Federated Reinforcement Learning",
            version="1.0.0"
        )
        
        # Add middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add trusted host middleware for security
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure appropriately for production
        )
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.utcnow()}
            
        @self.app.get("/api/v1/tenant/info")
        async def get_tenant_info(user: Dict[str, Any] = Depends(get_current_user)):
            """Get current tenant information"""
            tenant = await tenant_manager.get_tenant(user["tenant_id"])
            if not tenant:
                raise HTTPException(status_code=404, detail="Tenant not found")
                
            usage = await tenant_manager.get_usage(user["tenant_id"])
            
            return {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "tier": tenant.tier.value,
                "quotas": {
                    "max_agents": tenant.quotas.max_agents,
                    "max_experiments": tenant.quotas.max_experiments,
                    "max_models": tenant.quotas.max_models,
                    "max_datasets": tenant.quotas.max_datasets,
                    "compute_hours_monthly": tenant.quotas.compute_hours_monthly,
                    "storage_gb": tenant.quotas.storage_gb,
                    "api_requests_per_minute": tenant.quotas.api_requests_per_minute
                },
                "usage": usage
            }
            
        # Experiment endpoints
        @self.app.post("/api/v1/experiments", response_model=ExperimentResponse)
        async def create_experiment(
            experiment: ExperimentCreate,
            user: Dict[str, Any] = Depends(require_permission(Permission.CREATE_EXPERIMENT)),
            _: None = Depends(check_rate_limit)
        ):
            """Create a new federated learning experiment"""
            # Check quotas
            if not await tenant_manager.check_quota(user["tenant_id"], "experiments"):
                raise HTTPException(
                    status_code=403,
                    detail="Experiment quota exceeded"
                )
                
            # Create experiment (integrate with existing FL framework)
            experiment_id = f"exp_{int(time.time())}"
            
            # Update usage
            await tenant_manager.update_usage(user["tenant_id"], "experiments", 1)
            
            return ExperimentResponse(
                experiment_id=experiment_id,
                name=experiment.name,
                description=experiment.description,
                status="created",
                created_at=datetime.utcnow(),
                tenant_id=user["tenant_id"]
            )
            
        @self.app.get("/api/v1/experiments")
        async def list_experiments(
            user: Dict[str, Any] = Depends(require_permission(Permission.READ_EXPERIMENT)),
            _: None = Depends(check_rate_limit)
        ):
            """List all experiments for the tenant"""
            # Return experiments for this tenant
            return {"experiments": []}
            
        @self.app.get("/api/v1/experiments/{experiment_id}")
        async def get_experiment(
            experiment_id: str,
            user: Dict[str, Any] = Depends(require_permission(Permission.READ_EXPERIMENT)),
            _: None = Depends(check_rate_limit)
        ):
            """Get experiment details"""
            return {"experiment_id": experiment_id, "status": "running"}
            
        # Agent endpoints
        @self.app.post("/api/v1/agents", response_model=AgentResponse)
        async def create_agent(
            agent: AgentCreate,
            user: Dict[str, Any] = Depends(require_permission(Permission.CREATE_AGENT)),
            _: None = Depends(check_rate_limit)
        ):
            """Create a new federated learning agent"""
            # Check quotas
            if not await tenant_manager.check_quota(user["tenant_id"], "agents"):
                raise HTTPException(
                    status_code=403,
                    detail="Agent quota exceeded"
                )
                
            agent_id = f"agent_{int(time.time())}"
            
            # Update usage
            await tenant_manager.update_usage(user["tenant_id"], "agents", 1)
            
            return AgentResponse(
                agent_id=agent_id,
                name=agent.name,
                experiment_id=agent.experiment_id,
                status="created",
                created_at=datetime.utcnow(),
                metrics={}
            )
            
        @self.app.get("/api/v1/agents")
        async def list_agents(
            user: Dict[str, Any] = Depends(require_permission(Permission.READ_AGENT)),
            _: None = Depends(check_rate_limit)
        ):
            """List all agents for the tenant"""
            return {"agents": []}
            
        # Model endpoints
        @self.app.post("/api/v1/models", response_model=ModelResponse)
        async def create_model(
            model: ModelCreate,
            user: Dict[str, Any] = Depends(require_permission(Permission.CREATE_MODEL)),
            _: None = Depends(check_rate_limit)
        ):
            """Create a new model"""
            # Check quotas
            if not await tenant_manager.check_quota(user["tenant_id"], "models"):
                raise HTTPException(
                    status_code=403,
                    detail="Model quota exceeded"
                )
                
            model_id = f"model_{int(time.time())}"
            
            # Update usage
            await tenant_manager.update_usage(user["tenant_id"], "models", 1)
            
            return ModelResponse(
                model_id=model_id,
                name=model.name,
                description=model.description,
                algorithm_type=model.algorithm_type,
                version="1.0.0",
                created_at=datetime.utcnow(),
                metrics={}
            )
            
        @self.app.get("/api/v1/models")
        async def list_models(
            user: Dict[str, Any] = Depends(require_permission(Permission.READ_MODEL)),
            _: None = Depends(check_rate_limit)
        ):
            """List all models for the tenant"""
            return {"models": []}
            
        # Analytics endpoints
        @self.app.get("/api/v1/analytics/dashboard")
        async def get_dashboard_data(
            user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_ANALYTICS)),
            _: None = Depends(check_rate_limit)
        ):
            """Get dashboard analytics data"""
            return {
                "total_experiments": 0,
                "total_agents": 0,
                "total_models": 0,
                "compute_usage": 0,
                "success_rate": 0.0
            }


# Global API gateway instance
api_gateway = APIGateway()