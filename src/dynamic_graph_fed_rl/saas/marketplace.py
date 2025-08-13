"""
Marketplace for Federated Learning Algorithms and Datasets

Provides a platform for sharing, purchasing, and managing
federated learning algorithms, models, and datasets.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import asyncio


class AssetType(Enum):
    """Types of assets available in the marketplace"""
    ALGORITHM = "algorithm"
    MODEL = "model"
    DATASET = "dataset"
    ENVIRONMENT = "environment"


class AssetCategory(Enum):
    """Categories for marketplace assets"""
    # Algorithm categories
    FEDERATED_LEARNING = "federated_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GRAPH_NEURAL_NETWORKS = "graph_neural_networks"
    OPTIMIZATION = "optimization"
    
    # Dataset categories
    TRAFFIC_NETWORKS = "traffic_networks"
    SOCIAL_NETWORKS = "social_networks"
    FINANCIAL_NETWORKS = "financial_networks"
    SYNTHETIC_GRAPHS = "synthetic_graphs"
    
    # Environment categories
    SIMULATION = "simulation"
    REAL_WORLD = "real_world"
    BENCHMARK = "benchmark"


class LicenseType(Enum):
    """License types for marketplace assets"""
    MIT = "mit"
    APACHE_2 = "apache_2"
    GPL_3 = "gpl_3"
    COMMERCIAL = "commercial"
    PROPRIETARY = "proprietary"


@dataclass
class AssetMetadata:
    """Metadata for marketplace assets"""
    tags: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0.0"
    requirements: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    documentation_url: Optional[str] = None
    paper_reference: Optional[str] = None


@dataclass
class MarketplaceAsset:
    """Represents an asset in the marketplace"""
    asset_id: str
    name: str
    asset_type: AssetType
    category: AssetCategory
    publisher_tenant_id: str
    publisher_name: str
    
    # Pricing and licensing
    price: float = 0.0  # 0 for free assets
    license_type: LicenseType = LicenseType.MIT
    
    # Asset details
    metadata: AssetMetadata = field(default_factory=AssetMetadata)
    file_path: str = ""
    file_size_mb: float = 0.0
    
    # Marketplace statistics
    download_count: int = 0
    rating: float = 0.0
    rating_count: int = 0
    
    # Status and timestamps
    is_published: bool = False
    is_approved: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AssetReview:
    """User review for a marketplace asset"""
    review_id: str
    asset_id: str
    reviewer_tenant_id: str
    reviewer_name: str
    rating: int  # 1-5 stars
    comment: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AssetPurchase:
    """Record of asset purchase/download"""
    purchase_id: str
    asset_id: str
    buyer_tenant_id: str
    price_paid: float
    license_terms: str
    purchased_at: datetime = field(default_factory=datetime.utcnow)


class MarketplaceService:
    """
    Marketplace service providing:
    - Asset publishing and discovery
    - Purchase and licensing management
    - Reviews and ratings system
    - Revenue sharing for publishers
    - Quality assurance and approval workflow
    """
    
    def __init__(self):
        self._assets: Dict[str, MarketplaceAsset] = {}
        self._reviews: Dict[str, List[AssetReview]] = {}
        self._purchases: Dict[str, AssetPurchase] = {}
        self._user_libraries: Dict[str, Set[str]] = {}  # tenant_id -> asset_ids
        
    async def publish_asset(
        self,
        publisher_tenant_id: str,
        publisher_name: str,
        name: str,
        asset_type: AssetType,
        category: AssetCategory,
        metadata: AssetMetadata,
        file_path: str,
        price: float = 0.0,
        license_type: LicenseType = LicenseType.MIT
    ) -> MarketplaceAsset:
        """Publish a new asset to the marketplace"""
        asset_id = str(uuid.uuid4())
        
        asset = MarketplaceAsset(
            asset_id=asset_id,
            name=name,
            asset_type=asset_type,
            category=category,
            publisher_tenant_id=publisher_tenant_id,
            publisher_name=publisher_name,
            price=price,
            license_type=license_type,
            metadata=metadata,
            file_path=file_path,
            is_published=True,
            is_approved=False  # Requires approval for paid assets
        )
        
        # Auto-approve free assets with permissive licenses
        if price == 0.0 and license_type in [LicenseType.MIT, LicenseType.APACHE_2]:
            asset.is_approved = True
            
        self._assets[asset_id] = asset
        self._reviews[asset_id] = []
        
        return asset
        
    async def search_assets(
        self,
        query: str = "",
        asset_type: Optional[AssetType] = None,
        category: Optional[AssetCategory] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[MarketplaceAsset]:
        """Search for assets in the marketplace"""
        results = []
        
        for asset in self._assets.values():
            # Only show approved and published assets
            if not (asset.is_published and asset.is_approved):
                continue
                
            # Filter by search criteria
            if asset_type and asset.asset_type != asset_type:
                continue
                
            if category and asset.category != category:
                continue
                
            if max_price is not None and asset.price > max_price:
                continue
                
            if min_rating is not None and asset.rating < min_rating:
                continue
                
            if query and query.lower() not in asset.name.lower():
                continue
                
            if tags:
                asset_tags = [tag.lower() for tag in asset.metadata.tags]
                if not any(tag.lower() in asset_tags for tag in tags):
                    continue
                    
            results.append(asset)
            
            if len(results) >= limit:
                break
                
        # Sort by rating and download count
        results.sort(
            key=lambda x: (x.rating, x.download_count),
            reverse=True
        )
        
        return results
        
    async def get_asset(self, asset_id: str) -> Optional[MarketplaceAsset]:
        """Get detailed information about an asset"""
        return self._assets.get(asset_id)
        
    async def purchase_asset(
        self,
        asset_id: str,
        buyer_tenant_id: str
    ) -> Optional[AssetPurchase]:
        """Purchase or download an asset"""
        asset = self._assets.get(asset_id)
        if not asset or not (asset.is_published and asset.is_approved):
            return None
            
        # Check if already purchased
        if buyer_tenant_id in self._user_libraries:
            if asset_id in self._user_libraries[buyer_tenant_id]:
                return None  # Already owned
                
        purchase_id = str(uuid.uuid4())
        
        purchase = AssetPurchase(
            purchase_id=purchase_id,
            asset_id=asset_id,
            buyer_tenant_id=buyer_tenant_id,
            price_paid=asset.price,
            license_terms=asset.license_type.value
        )
        
        self._purchases[purchase_id] = purchase
        
        # Add to user's library
        if buyer_tenant_id not in self._user_libraries:
            self._user_libraries[buyer_tenant_id] = set()
        self._user_libraries[buyer_tenant_id].add(asset_id)
        
        # Update download count
        asset.download_count += 1
        
        return purchase
        
    async def add_review(
        self,
        asset_id: str,
        reviewer_tenant_id: str,
        reviewer_name: str,
        rating: int,
        comment: str
    ) -> Optional[AssetReview]:
        """Add a review for an asset"""
        if asset_id not in self._assets:
            return None
            
        # Check if user has purchased the asset
        if reviewer_tenant_id not in self._user_libraries:
            return None
        if asset_id not in self._user_libraries[reviewer_tenant_id]:
            return None
            
        # Check if user already reviewed
        existing_reviews = self._reviews.get(asset_id, [])
        for review in existing_reviews:
            if review.reviewer_tenant_id == reviewer_tenant_id:
                return None  # Already reviewed
                
        review_id = str(uuid.uuid4())
        
        review = AssetReview(
            review_id=review_id,
            asset_id=asset_id,
            reviewer_tenant_id=reviewer_tenant_id,
            reviewer_name=reviewer_name,
            rating=max(1, min(5, rating)),  # Ensure 1-5 range
            comment=comment
        )
        
        self._reviews[asset_id].append(review)
        
        # Update asset rating
        await self._update_asset_rating(asset_id)
        
        return review
        
    async def get_reviews(self, asset_id: str) -> List[AssetReview]:
        """Get reviews for an asset"""
        return self._reviews.get(asset_id, [])
        
    async def get_user_library(self, tenant_id: str) -> List[MarketplaceAsset]:
        """Get assets owned by a user"""
        asset_ids = self._user_libraries.get(tenant_id, set())
        return [self._assets[aid] for aid in asset_ids if aid in self._assets]
        
    async def get_publisher_assets(self, publisher_tenant_id: str) -> List[MarketplaceAsset]:
        """Get assets published by a specific tenant"""
        return [
            asset for asset in self._assets.values()
            if asset.publisher_tenant_id == publisher_tenant_id
        ]
        
    async def approve_asset(self, asset_id: str) -> bool:
        """Approve an asset for marketplace (admin function)"""
        if asset_id in self._assets:
            self._assets[asset_id].is_approved = True
            return True
        return False
        
    async def reject_asset(self, asset_id: str) -> bool:
        """Reject an asset from marketplace (admin function)"""
        if asset_id in self._assets:
            self._assets[asset_id].is_approved = False
            self._assets[asset_id].is_published = False
            return True
        return False
        
    async def get_featured_assets(self, limit: int = 10) -> List[MarketplaceAsset]:
        """Get featured assets for homepage"""
        approved_assets = [
            asset for asset in self._assets.values()
            if asset.is_published and asset.is_approved
        ]
        
        # Sort by rating and download count
        featured = sorted(
            approved_assets,
            key=lambda x: (x.rating, x.download_count),
            reverse=True
        )
        
        return featured[:limit]
        
    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        total_assets = len(self._assets)
        published_assets = sum(1 for a in self._assets.values() if a.is_published)
        total_downloads = sum(a.download_count for a in self._assets.values())
        total_revenue = sum(p.price_paid for p in self._purchases.values())
        
        return {
            "total_assets": total_assets,
            "published_assets": published_assets,
            "total_downloads": total_downloads,
            "total_revenue": total_revenue,
            "asset_types": self._get_asset_type_distribution(),
            "top_categories": self._get_top_categories()
        }
        
    async def _update_asset_rating(self, asset_id: str) -> None:
        """Update the average rating for an asset"""
        reviews = self._reviews.get(asset_id, [])
        if not reviews:
            return
            
        total_rating = sum(r.rating for r in reviews)
        average_rating = total_rating / len(reviews)
        
        if asset_id in self._assets:
            self._assets[asset_id].rating = round(average_rating, 2)
            self._assets[asset_id].rating_count = len(reviews)
            
    def _get_asset_type_distribution(self) -> Dict[str, int]:
        """Get distribution of asset types"""
        distribution = {}
        for asset in self._assets.values():
            if asset.is_published and asset.is_approved:
                asset_type = asset.asset_type.value
                distribution[asset_type] = distribution.get(asset_type, 0) + 1
        return distribution
        
    def _get_top_categories(self) -> List[Dict[str, Any]]:
        """Get top categories by asset count"""
        categories = {}
        for asset in self._assets.values():
            if asset.is_published and asset.is_approved:
                category = asset.category.value
                categories[category] = categories.get(category, 0) + 1
                
        return [
            {"category": cat, "count": count}
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)
        ][:10]


# Global marketplace service instance
marketplace_service = MarketplaceService()