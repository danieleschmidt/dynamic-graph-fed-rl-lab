"""
Authentication and Authorization Service

Provides JWT-based authentication, role-based access control,
and API key management for the enterprise SaaS platform.
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import jwt
import bcrypt
import secrets
import uuid
import asyncio


class UserRole(Enum):
    """User roles with different permission levels"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


class Permission(Enum):
    """Granular permissions for role-based access control"""
    # Experiment permissions
    CREATE_EXPERIMENT = "experiment:create"
    READ_EXPERIMENT = "experiment:read"
    UPDATE_EXPERIMENT = "experiment:update"
    DELETE_EXPERIMENT = "experiment:delete"
    
    # Model permissions
    CREATE_MODEL = "model:create"
    READ_MODEL = "model:read"
    UPDATE_MODEL = "model:update"
    DELETE_MODEL = "model:delete"
    
    # Dataset permissions
    CREATE_DATASET = "dataset:create"
    READ_DATASET = "dataset:read"
    UPDATE_DATASET = "dataset:update"
    DELETE_DATASET = "dataset:delete"
    
    # Agent permissions
    CREATE_AGENT = "agent:create"
    READ_AGENT = "agent:read"
    UPDATE_AGENT = "agent:update"
    DELETE_AGENT = "agent:delete"
    
    # Marketplace permissions
    PUBLISH_ALGORITHM = "marketplace:publish"
    PURCHASE_ALGORITHM = "marketplace:purchase"
    
    # Admin permissions
    MANAGE_USERS = "admin:manage_users"
    MANAGE_BILLING = "admin:manage_billing"
    VIEW_ANALYTICS = "admin:view_analytics"


@dataclass
class User:
    """User account information"""
    user_id: str
    email: str
    password_hash: str
    tenant_id: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIKey:
    """API key for programmatic access"""
    key_id: str
    key_hash: str
    tenant_id: str
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None


class AuthenticationService:
    """
    Handles authentication and authorization including:
    - JWT token generation and validation
    - Password hashing and verification
    - API key management
    - Role-based access control
    """
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, APIKey] = {}
        self._role_permissions = self._initialize_role_permissions()
        
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize default permissions for each role"""
        return {
            UserRole.ADMIN: {
                Permission.CREATE_EXPERIMENT,
                Permission.READ_EXPERIMENT,
                Permission.UPDATE_EXPERIMENT,
                Permission.DELETE_EXPERIMENT,
                Permission.CREATE_MODEL,
                Permission.READ_MODEL,
                Permission.UPDATE_MODEL,
                Permission.DELETE_MODEL,
                Permission.CREATE_DATASET,
                Permission.READ_DATASET,
                Permission.UPDATE_DATASET,
                Permission.DELETE_DATASET,
                Permission.CREATE_AGENT,
                Permission.READ_AGENT,
                Permission.UPDATE_AGENT,
                Permission.DELETE_AGENT,
                Permission.PUBLISH_ALGORITHM,
                Permission.PURCHASE_ALGORITHM,
                Permission.MANAGE_USERS,
                Permission.MANAGE_BILLING,
                Permission.VIEW_ANALYTICS
            },
            UserRole.USER: {
                Permission.CREATE_EXPERIMENT,
                Permission.READ_EXPERIMENT,
                Permission.UPDATE_EXPERIMENT,
                Permission.DELETE_EXPERIMENT,
                Permission.CREATE_MODEL,
                Permission.READ_MODEL,
                Permission.UPDATE_MODEL,
                Permission.DELETE_MODEL,
                Permission.CREATE_DATASET,
                Permission.READ_DATASET,
                Permission.UPDATE_DATASET,
                Permission.DELETE_DATASET,
                Permission.CREATE_AGENT,
                Permission.READ_AGENT,
                Permission.UPDATE_AGENT,
                Permission.DELETE_AGENT,
                Permission.PURCHASE_ALGORITHM
            },
            UserRole.VIEWER: {
                Permission.READ_EXPERIMENT,
                Permission.READ_MODEL,
                Permission.READ_DATASET,
                Permission.READ_AGENT
            },
            UserRole.API_CLIENT: {
                Permission.CREATE_EXPERIMENT,
                Permission.READ_EXPERIMENT,
                Permission.UPDATE_EXPERIMENT,
                Permission.CREATE_MODEL,
                Permission.READ_MODEL,
                Permission.UPDATE_MODEL,
                Permission.CREATE_DATASET,
                Permission.READ_DATASET,
                Permission.CREATE_AGENT,
                Permission.READ_AGENT,
                Permission.UPDATE_AGENT
            }
        }
        
    async def create_user(
        self,
        email: str,
        password: str,
        tenant_id: str,
        role: UserRole = UserRole.USER
    ) -> User:
        """Create a new user account"""
        user_id = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        user = User(
            user_id=user_id,
            email=email,
            password_hash=password_hash,
            tenant_id=tenant_id,
            role=role,
            permissions=self._role_permissions[role].copy()
        )
        
        self._users[user_id] = user
        return user
        
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        for user in self._users.values():
            if user.email == email and user.is_active:
                if bcrypt.checkpw(password.encode(), user.password_hash.encode()):
                    user.last_login = datetime.utcnow()
                    return user
        return None
        
    async def generate_jwt_token(
        self,
        user: User,
        expires_in: timedelta = timedelta(hours=24)
    ) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            "user_id": user.user_id,
            "email": user.email,
            "tenant_id": user.tenant_id,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "exp": datetime.utcnow() + expires_in,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
        
    async def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check if user still exists and is active
            user_id = payload.get("user_id")
            if user_id in self._users and self._users[user_id].is_active:
                return payload
                
        except jwt.ExpiredSignatureError:
            pass
        except jwt.InvalidTokenError:
            pass
            
        return None
        
    async def create_api_key(
        self,
        tenant_id: str,
        name: str,
        permissions: Set[Permission],
        expires_in: Optional[timedelta] = None
    ) -> tuple[str, APIKey]:
        """Create a new API key"""
        key_id = str(uuid.uuid4())
        api_key = secrets.token_urlsafe(32)
        key_hash = bcrypt.hashpw(api_key.encode(), bcrypt.gensalt()).decode()
        
        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            tenant_id=tenant_id,
            name=name,
            permissions=permissions,
            expires_at=datetime.utcnow() + expires_in if expires_in else None
        )
        
        self._api_keys[key_id] = api_key_obj
        return api_key, api_key_obj
        
    async def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify API key and return associated object"""
        for key_obj in self._api_keys.values():
            if (key_obj.is_active and
                bcrypt.checkpw(api_key.encode(), key_obj.key_hash.encode())):
                
                # Check if expired
                if key_obj.expires_at and datetime.utcnow() > key_obj.expires_at:
                    continue
                    
                key_obj.last_used = datetime.utcnow()
                return key_obj
                
        return None
        
    async def check_permission(
        self,
        user_permissions: Set[Permission],
        required_permission: Permission
    ) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions
        
    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id in self._api_keys:
            self._api_keys[key_id].is_active = False
            return True
        return False
        
    async def update_user_permissions(
        self,
        user_id: str,
        permissions: Set[Permission]
    ) -> bool:
        """Update user permissions"""
        if user_id in self._users:
            self._users[user_id].permissions = permissions
            return True
        return False
        
    async def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account"""
        if user_id in self._users:
            self._users[user_id].is_active = False
            return True
        return False


# Global authentication service instance
auth_service = AuthenticationService()