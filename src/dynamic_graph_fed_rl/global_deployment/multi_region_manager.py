"""Multi-region deployment and coordination for global federated RL systems."""

import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


class RegionStatus(Enum):
    """Region operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    
    id: str
    name: str
    geographic_zone: str  # e.g., "us-east-1", "eu-west-1", "asia-pacific-1"
    data_center_location: str
    regulatory_compliance: List[str] = field(default_factory=list)  # e.g., ["GDPR", "CCPA"]
    max_agents: int = 1000
    latency_ms: float = 50.0
    bandwidth_mbps: float = 1000.0
    data_residency_required: bool = False
    encryption_standard: str = "AES-256"
    backup_regions: List[str] = field(default_factory=list)


class MultiRegionManager:
    """
    Manages multi-region deployment and coordination for global federated RL.
    
    Features:
    - Cross-region agent coordination
    - Data residency compliance
    - Latency-optimized routing
    - Disaster recovery and failover
    - Regional load balancing
    - Compliance monitoring
    """
    
    def __init__(self, primary_region: str = "us-east-1"):
        self.primary_region = primary_region
        self.regions: Dict[str, RegionConfig] = {}
        self.region_status: Dict[str, RegionStatus] = {}
        self.region_metrics: Dict[str, Dict[str, Any]] = {}
        self.agent_distribution: Dict[str, Set[str]] = {}  # region_id -> agent_ids
        
        # Cross-region coordination
        self.cross_region_lock = threading.RLock()
        self.federation_routing_table: Dict[str, List[str]] = {}
        self.data_replication_status: Dict[str, Dict[str, str]] = {}
        
        # Performance monitoring
        self.latency_matrix: Dict[str, Dict[str, float]] = {}
        self.throughput_metrics: Dict[str, float] = {}
        
        # Compliance tracking
        self.compliance_violations: List[Dict[str, Any]] = []
        
        print("🌍 Multi-region manager initialized")
        
        # Setup default regions
        self._setup_default_regions()
    
    def _setup_default_regions(self):
        """Setup default global regions."""
        default_regions = [
            RegionConfig(
                id="us-east-1",
                name="US East (Virginia)",
                geographic_zone="north-america",
                data_center_location="Virginia, USA",
                regulatory_compliance=["CCPA", "SOC2"],
                latency_ms=15.0,
                bandwidth_mbps=10000.0
            ),
            RegionConfig(
                id="eu-west-1", 
                name="EU West (Ireland)",
                geographic_zone="europe",
                data_center_location="Dublin, Ireland",
                regulatory_compliance=["GDPR", "ISO27001"],
                latency_ms=20.0,
                bandwidth_mbps=8000.0,
                data_residency_required=True
            ),
            RegionConfig(
                id="asia-pacific-1",
                name="Asia Pacific (Singapore)",
                geographic_zone="asia-pacific",
                data_center_location="Singapore",
                regulatory_compliance=["PDPA", "ISO27001"],
                latency_ms=25.0,
                bandwidth_mbps=6000.0
            ),
            RegionConfig(
                id="us-west-2",
                name="US West (Oregon)",
                geographic_zone="north-america", 
                data_center_location="Oregon, USA",
                regulatory_compliance=["CCPA", "SOC2"],
                latency_ms=18.0,
                bandwidth_mbps=9000.0
            )
        ]
        
        for region in default_regions:
            self.register_region(region)
    
    def register_region(self, region_config: RegionConfig):
        """Register a new deployment region."""
        with self.cross_region_lock:
            self.regions[region_config.id] = region_config
            self.region_status[region_config.id] = RegionStatus.HEALTHY
            self.region_metrics[region_config.id] = {
                "active_agents": 0,
                "cpu_utilization": 0.0,
                "memory_utilization": 0.0,
                "network_utilization": 0.0,
                "last_health_check": time.time()
            }
            self.agent_distribution[region_config.id] = set()
            
            print(f"🌍 Registered region: {region_config.name} ({region_config.id})")
            
            # Update routing table
            self._update_routing_table()
    
    def _update_routing_table(self):
        """Update cross-region routing table based on latency and compliance."""
        for region_id in self.regions:
            # Sort other regions by latency for optimal routing
            other_regions = [(rid, config) for rid, config in self.regions.items() if rid != region_id]
            sorted_regions = sorted(other_regions, key=lambda x: x[1].latency_ms)
            self.federation_routing_table[region_id] = [rid for rid, _ in sorted_regions]
    
    def deploy_agent(self, agent_id: str, preferred_region: Optional[str] = None) -> str:
        """Deploy federated agent to optimal region."""
        with self.cross_region_lock:
            target_region = self._select_optimal_region(agent_id, preferred_region)
            
            if target_region:
                self.agent_distribution[target_region].add(agent_id)
                self.region_metrics[target_region]["active_agents"] += 1
                
                print(f"🚀 Deployed agent {agent_id} to region {target_region}")
                return target_region
            else:
                raise Exception("No available regions for agent deployment")
    
    def _select_optimal_region(self, agent_id: str, preferred_region: Optional[str] = None) -> Optional[str]:
        """Select optimal region for agent deployment."""
        # Check preferred region first
        if preferred_region and preferred_region in self.regions:
            if self._can_deploy_to_region(preferred_region):
                return preferred_region
        
        # Find best alternative region
        candidates = []
        for region_id, config in self.regions.items():
            if self._can_deploy_to_region(region_id):
                # Score based on latency, load, and capacity
                load_factor = self.region_metrics[region_id]["active_agents"] / config.max_agents
                latency_score = 1.0 / (1.0 + config.latency_ms / 100.0)
                capacity_score = 1.0 - load_factor
                
                total_score = (latency_score * 0.4) + (capacity_score * 0.6)
                candidates.append((region_id, total_score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _can_deploy_to_region(self, region_id: str) -> bool:
        """Check if region can accept new deployments."""
        if region_id not in self.regions:
            return False
        
        status = self.region_status[region_id]
        if status not in [RegionStatus.HEALTHY, RegionStatus.DEGRADED]:
            return False
        
        config = self.regions[region_id]
        current_agents = self.region_metrics[region_id]["active_agents"]
        
        return current_agents < config.max_agents
    
    def coordinate_federation(self, source_region: str, target_regions: List[str]) -> Dict[str, Any]:
        """Coordinate federated learning across regions."""
        coordination_result = {
            "source_region": source_region,
            "target_regions": target_regions,
            "coordination_time": time.time(),
            "latencies": {},
            "compliance_check": True,
            "data_transfers": []
        }
        
        # Check compliance for cross-region data transfer
        for target_region in target_regions:
            if not self._check_cross_region_compliance(source_region, target_region):
                coordination_result["compliance_check"] = False
                self._log_compliance_violation(source_region, target_region)
        
        # Calculate optimal routing
        for target_region in target_regions:
            latency = self._calculate_cross_region_latency(source_region, target_region)
            coordination_result["latencies"][target_region] = latency
        
        # Simulate data transfer coordination
        for target_region in target_regions:
            transfer_info = {
                "target": target_region,
                "status": "completed",
                "transfer_time": coordination_result["latencies"][target_region] / 1000.0,
                "data_encrypted": True
            }
            coordination_result["data_transfers"].append(transfer_info)
        
        print(f"🔄 Coordinated federation from {source_region} to {len(target_regions)} regions")
        return coordination_result
    
    def _check_cross_region_compliance(self, source_region: str, target_region: str) -> bool:
        """Check if cross-region data transfer is compliant."""
        source_config = self.regions.get(source_region)
        target_config = self.regions.get(target_region)
        
        if not source_config or not target_config:
            return False
        
        # Check data residency requirements
        if source_config.data_residency_required:
            # Data must stay within same geographic zone
            if source_config.geographic_zone != target_config.geographic_zone:
                return False
        
        # Check compatible compliance standards
        source_standards = set(source_config.regulatory_compliance)
        target_standards = set(target_config.regulatory_compliance)
        
        # Must have at least one common compliance standard
        return len(source_standards.intersection(target_standards)) > 0
    
    def _calculate_cross_region_latency(self, source: str, target: str) -> float:
        """Calculate latency between regions."""
        if source == target:
            return 0.0
        
        source_config = self.regions.get(source)
        target_config = self.regions.get(target)
        
        if not source_config or not target_config:
            return float('inf')
        
        # Base latency + geographic distance factor
        base_latency = (source_config.latency_ms + target_config.latency_ms) / 2
        
        # Add distance penalty for different geographic zones
        if source_config.geographic_zone != target_config.geographic_zone:
            base_latency *= 2.5
        
        return base_latency
    
    def _log_compliance_violation(self, source_region: str, target_region: str):
        """Log compliance violation for audit purposes."""
        violation = {
            "timestamp": time.time(),
            "violation_type": "cross_region_transfer",
            "source_region": source_region,
            "target_region": target_region,
            "severity": "high",
            "description": f"Data residency violation: {source_region} -> {target_region}"
        }
        
        self.compliance_violations.append(violation)
        print(f"⚠️  Compliance violation logged: {source_region} -> {target_region}")
    
    def monitor_region_health(self) -> Dict[str, Any]:
        """Monitor health across all regions."""
        health_report = {
            "timestamp": time.time(),
            "total_regions": len(self.regions),
            "healthy_regions": 0,
            "degraded_regions": 0,
            "offline_regions": 0,
            "total_agents": 0,
            "region_details": {}
        }
        
        for region_id, status in self.region_status.items():
            health_report["region_details"][region_id] = {
                "status": status.value,
                "active_agents": self.region_metrics[region_id]["active_agents"],
                "latency_ms": self.regions[region_id].latency_ms,
                "compliance": self.regions[region_id].regulatory_compliance
            }
            
            health_report["total_agents"] += self.region_metrics[region_id]["active_agents"]
            
            if status == RegionStatus.HEALTHY:
                health_report["healthy_regions"] += 1
            elif status == RegionStatus.DEGRADED:
                health_report["degraded_regions"] += 1
            else:
                health_report["offline_regions"] += 1
        
        return health_report
    
    def failover_region(self, failed_region: str) -> Dict[str, Any]:
        """Handle region failover and agent redistribution."""
        print(f"🚨 Initiating failover for region: {failed_region}")
        
        # Mark region as offline
        self.region_status[failed_region] = RegionStatus.OFFLINE
        
        # Get agents that need to be relocated
        affected_agents = self.agent_distribution[failed_region].copy()
        self.agent_distribution[failed_region].clear()
        self.region_metrics[failed_region]["active_agents"] = 0
        
        # Find backup regions
        failed_config = self.regions[failed_region]
        backup_regions = failed_config.backup_regions
        
        if not backup_regions:
            # Use geographic zone preference for automatic backup selection
            backup_regions = [
                rid for rid, config in self.regions.items() 
                if (rid != failed_region and 
                    config.geographic_zone == failed_config.geographic_zone and
                    self.region_status[rid] == RegionStatus.HEALTHY)
            ]
        
        # Redistribute agents
        redistributed_agents = {}
        for agent_id in affected_agents:
            for backup_region in backup_regions:
                if self._can_deploy_to_region(backup_region):
                    new_region = self.deploy_agent(agent_id, backup_region)
                    redistributed_agents[agent_id] = new_region
                    break
        
        failover_result = {
            "failed_region": failed_region,
            "affected_agents": len(affected_agents),
            "redistributed_agents": len(redistributed_agents),
            "backup_regions_used": list(set(redistributed_agents.values())),
            "failover_time": time.time()
        }
        
        print(f"✅ Failover completed: {len(redistributed_agents)}/{len(affected_agents)} agents redistributed")
        return failover_result
    
    def get_deployment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive deployment statistics."""
        stats = {
            "global_deployment": {
                "total_regions": len(self.regions),
                "active_regions": sum(1 for status in self.region_status.values() if status == RegionStatus.HEALTHY),
                "total_agents": sum(metrics["active_agents"] for metrics in self.region_metrics.values()),
                "compliance_violations": len(self.compliance_violations)
            },
            "regions": {},
            "geographic_distribution": {},
            "compliance_summary": {}
        }
        
        # Per-region statistics
        for region_id, config in self.regions.items():
            stats["regions"][region_id] = {
                "name": config.name,
                "status": self.region_status[region_id].value,
                "active_agents": self.region_metrics[region_id]["active_agents"],
                "capacity_utilization": self.region_metrics[region_id]["active_agents"] / config.max_agents,
                "latency_ms": config.latency_ms,
                "compliance_standards": config.regulatory_compliance
            }
        
        # Geographic distribution
        geo_zones = {}
        for config in self.regions.values():
            zone = config.geographic_zone
            if zone not in geo_zones:
                geo_zones[zone] = {"regions": 0, "agents": 0}
            geo_zones[zone]["regions"] += 1
            geo_zones[zone]["agents"] += self.region_metrics[config.id]["active_agents"]
        
        stats["geographic_distribution"] = geo_zones
        
        # Compliance summary
        compliance_standards = set()
        for config in self.regions.values():
            compliance_standards.update(config.regulatory_compliance)
        
        stats["compliance_summary"] = {
            "supported_standards": list(compliance_standards),
            "violations_count": len(self.compliance_violations),
            "data_residency_regions": sum(1 for config in self.regions.values() if config.data_residency_required)
        }
        
        return stats