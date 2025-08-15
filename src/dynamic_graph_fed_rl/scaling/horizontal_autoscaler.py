"""Advanced horizontal auto-scaling system for massive federated learning."""

import asyncio
import time
import threading
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
from collections import defaultdict, deque
import logging
import concurrent.futures
from abc import ABC, abstractmethod


class ScalingTrigger(Enum):
    """Scaling trigger conditions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_PRESSURE = "memory_pressure"
    THROUGHPUT_DEGRADATION = "throughput_degradation"
    QUEUE_BACKLOG = "queue_backlog"
    NETWORK_LATENCY = "network_latency"
    PREDICTIVE_LOAD = "predictive_load"
    AGENT_FAILURE = "agent_failure"
    CUSTOM_METRIC = "custom_metric"


class ScalingAction(Enum):
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REDISTRIBUTE = "redistribute"
    MIGRATE = "migrate"
    REPLICATE = "replicate"
    TERMINATE = "terminate"


@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: float
    trigger: ScalingTrigger
    action: ScalingAction
    resource_type: str
    scale_factor: float
    target_capacity: int
    current_capacity: int
    metrics: Dict[str, float]
    success: bool = False
    execution_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class FederatedAgent:
    """Enhanced federated agent with scaling metadata."""
    id: str
    region: str
    status: str
    capacity: int
    current_load: float
    last_heartbeat: float
    scaling_group: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    health_score: float = 1.0
    
    def update_performance(self, performance_metric: float):
        """Update performance history with bounded memory."""
        self.performance_history.append(performance_metric)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        # Update health score based on recent performance
        if len(self.performance_history) >= 5:
            recent_avg = statistics.mean(self.performance_history[-5:])
            self.health_score = min(1.0, max(0.0, recent_avg))


class ResourceProvisioner(ABC):
    """Abstract base class for resource provisioning."""
    
    @abstractmethod
    async def provision_agents(self, count: int, region: str, config: Dict[str, Any]) -> List[FederatedAgent]:
        """Provision new federated agents."""
        pass
    
    @abstractmethod
    async def terminate_agents(self, agent_ids: List[str]) -> bool:
        """Terminate specified agents."""
        pass
    
    @abstractmethod
    async def get_resource_availability(self, region: str) -> Dict[str, Any]:
        """Get available resources in region."""
        pass


class MockResourceProvisioner(ResourceProvisioner):
    """Mock resource provisioner for testing and simulation."""
    
    def __init__(self):
        self.provisioned_agents: Dict[str, FederatedAgent] = {}
        self.region_capacity = defaultdict(lambda: 1000)  # Mock capacity per region
        
    async def provision_agents(self, count: int, region: str, config: Dict[str, Any]) -> List[FederatedAgent]:
        """Provision mock federated agents."""
        agents = []
        
        for i in range(count):
            agent_id = f"agent_{region}_{int(time.time())}_{i}"
            agent = FederatedAgent(
                id=agent_id,
                region=region,
                status="healthy",
                capacity=config.get("capacity", 100),
                current_load=0.0,
                last_heartbeat=time.time(),
                scaling_group=config.get("scaling_group", "default")
            )
            
            self.provisioned_agents[agent_id] = agent
            agents.append(agent)
            
            await asyncio.sleep(0.1)  # Simulate provisioning delay
        
        return agents
    
    async def terminate_agents(self, agent_ids: List[str]) -> bool:
        """Terminate mock agents."""
        for agent_id in agent_ids:
            if agent_id in self.provisioned_agents:
                del self.provisioned_agents[agent_id]
        return True
    
    async def get_resource_availability(self, region: str) -> Dict[str, Any]:
        """Get mock resource availability."""
        return {
            "available_capacity": self.region_capacity[region],
            "cpu_usage": 0.3,
            "memory_usage": 0.4,
            "network_bandwidth": 1000.0
        }


class HorizontalAutoScaler:
    """
    Advanced horizontal auto-scaling system for massive federated learning.
    
    Features:
    - Dynamic federation expansion up to 10,000+ agents
    - Intelligent scaling based on multiple metrics
    - Predictive scaling using ML models
    - Multi-region resource management
    - Automatic load balancing and redistribution
    - Fault-tolerant agent management
    """
    
    def __init__(
        self,
        min_agents: int = 10,
        max_agents: int = 10000,
        target_utilization: float = 0.75,
        scale_up_threshold: float = 0.85,
        scale_down_threshold: float = 0.50,
        scaling_cooldown: float = 300.0,  # 5 minutes
        prediction_window: float = 600.0,  # 10 minutes
        resource_provisioner: Optional[ResourceProvisioner] = None
    ):
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_cooldown = scaling_cooldown
        self.prediction_window = prediction_window
        
        # Resource management
        self.resource_provisioner = resource_provisioner or MockResourceProvisioner()
        self.active_agents: Dict[str, FederatedAgent] = {}
        self.scaling_groups: Dict[str, List[str]] = defaultdict(list)
        
        # Scaling state
        self.last_scaling_action = 0.0
        self.scaling_events: deque = deque(maxlen=1000)
        self.metrics_history: deque = deque(maxlen=10000)
        
        # Monitoring and prediction
        self.metrics_collector = MetricsCollector()
        self.load_predictor = LoadPredictor()
        self.scaling_policy = ScalingPolicy()
        
        # Threading and async
        self.scaling_lock = threading.RLock()
        self.monitoring_task = None
        self.scaling_task = None
        self.is_running = False
        
        # Performance tracking
        self.scaling_efficiency_history: List[float] = []
        self.resource_utilization_history: List[Dict[str, float]] = []
        
        logging.info(f"HorizontalAutoScaler initialized: {min_agents}-{max_agents} agents, target: {target_utilization:.1%}")
    
    async def start(self):
        """Start the auto-scaling system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        
        # Initialize with minimum agents
        await self._initialize_minimum_agents()
        
        logging.info("HorizontalAutoScaler started")
    
    async def stop(self):
        """Stop the auto-scaling system."""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.scaling_task:
            self.scaling_task.cancel()
        
        logging.info("HorizontalAutoScaler stopped")
    
    async def _initialize_minimum_agents(self):
        """Initialize with minimum required agents."""
        current_count = len(self.active_agents)
        if current_count < self.min_agents:
            needed = self.min_agents - current_count
            await self._scale_up(needed, "initialization")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            try:
                # Collect current metrics
                metrics = await self.metrics_collector.collect_metrics(self.active_agents)
                self.metrics_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics
                })
                
                # Update agent performance
                for agent_id, agent in self.active_agents.items():
                    if agent_id in metrics.get('agent_metrics', {}):
                        agent_metrics = metrics['agent_metrics'][agent_id]
                        agent.update_performance(agent_metrics.get('performance_score', 0.5))
                        agent.current_load = agent_metrics.get('load', 0.0)
                        agent.last_heartbeat = time.time()
                
                # Check for failed agents
                await self._check_agent_health()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _scaling_loop(self):
        """Background scaling decision loop."""
        while self.is_running:
            try:
                # Check if scaling cooldown has passed
                if time.time() - self.last_scaling_action < self.scaling_cooldown:
                    await asyncio.sleep(60)
                    continue
                
                # Analyze current state and make scaling decisions
                scaling_decision = await self._analyze_scaling_needs()
                
                if scaling_decision:
                    await self._execute_scaling_decision(scaling_decision)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Scaling loop error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_scaling_needs(self) -> Optional[Dict[str, Any]]:
        """Analyze current metrics to determine scaling needs."""
        if not self.metrics_history:
            return None
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        if not recent_metrics:
            return None
        
        # Calculate average utilization
        avg_cpu = statistics.mean([m['metrics']['system']['cpu_utilization'] for m in recent_metrics])
        avg_memory = statistics.mean([m['metrics']['system']['memory_utilization'] for m in recent_metrics])
        avg_throughput = statistics.mean([m['metrics']['system']['throughput'] for m in recent_metrics])
        
        current_agents = len(self.active_agents)
        
        # Determine scaling action
        scaling_decision = None
        
        # Scale up conditions
        if (avg_cpu > self.scale_up_threshold or 
            avg_memory > self.scale_up_threshold or
            avg_throughput < 0.5):  # Poor throughput
            
            if current_agents < self.max_agents:
                # Calculate scale factor
                scale_factor = max(avg_cpu, avg_memory) / self.target_utilization
                target_agents = min(int(current_agents * scale_factor), self.max_agents)
                agents_to_add = target_agents - current_agents
                
                if agents_to_add > 0:
                    scaling_decision = {
                        'action': ScalingAction.SCALE_UP,
                        'count': agents_to_add,
                        'reason': f"High utilization: CPU {avg_cpu:.1%}, Memory {avg_memory:.1%}",
                        'metrics': {
                            'cpu_utilization': avg_cpu,
                            'memory_utilization': avg_memory,
                            'throughput': avg_throughput
                        }
                    }
        
        # Scale down conditions
        elif (avg_cpu < self.scale_down_threshold and 
              avg_memory < self.scale_down_threshold and
              current_agents > self.min_agents):
            
            # Calculate scale factor
            utilization = max(avg_cpu, avg_memory)
            if utilization > 0:
                scale_factor = self.target_utilization / utilization
                target_agents = max(int(current_agents / scale_factor), self.min_agents)
                agents_to_remove = current_agents - target_agents
                
                if agents_to_remove > 0:
                    scaling_decision = {
                        'action': ScalingAction.SCALE_DOWN,
                        'count': agents_to_remove,
                        'reason': f"Low utilization: CPU {avg_cpu:.1%}, Memory {avg_memory:.1%}",
                        'metrics': {
                            'cpu_utilization': avg_cpu,
                            'memory_utilization': avg_memory,
                            'throughput': avg_throughput
                        }
                    }
        
        # Add predictive scaling
        if not scaling_decision:
            predicted_load = await self.load_predictor.predict_load(self.metrics_history)
            if predicted_load:
                predictive_decision = await self._analyze_predictive_scaling(predicted_load)
                if predictive_decision:
                    scaling_decision = predictive_decision
        
        return scaling_decision
    
    async def _analyze_predictive_scaling(self, predicted_load: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Analyze predictive load patterns for proactive scaling."""
        predicted_utilization = predicted_load.get('cpu_utilization', 0.0)
        confidence = predicted_load.get('confidence', 0.0)
        
        # Only act on high-confidence predictions
        if confidence < 0.7:
            return None
        
        current_agents = len(self.active_agents)
        
        # Predictive scale up
        if predicted_utilization > self.scale_up_threshold and current_agents < self.max_agents:
            scale_factor = predicted_utilization / self.target_utilization
            target_agents = min(int(current_agents * scale_factor), self.max_agents)
            agents_to_add = target_agents - current_agents
            
            if agents_to_add > 0:
                return {
                    'action': ScalingAction.SCALE_UP,
                    'count': agents_to_add,
                    'reason': f"Predictive scaling: Expected {predicted_utilization:.1%} utilization",
                    'metrics': predicted_load,
                    'predictive': True
                }
        
        return None
    
    async def _execute_scaling_decision(self, decision: Dict[str, Any]):
        """Execute a scaling decision."""
        action = decision['action']
        count = decision['count']
        reason = decision['reason']
        
        logging.info(f"Executing scaling decision: {action.value} {count} agents - {reason}")
        
        start_time = time.time()
        success = False
        error_message = None
        
        try:
            if action == ScalingAction.SCALE_UP:
                success = await self._scale_up(count, reason)
            elif action == ScalingAction.SCALE_DOWN:
                success = await self._scale_down(count, reason)
            
            self.last_scaling_action = time.time()
            
        except Exception as e:
            error_message = str(e)
            logging.error(f"Scaling execution failed: {e}")
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=time.time(),
            trigger=ScalingTrigger.CPU_UTILIZATION,  # Primary trigger
            action=action,
            resource_type="federated_agents",
            scale_factor=count / len(self.active_agents) if self.active_agents else 1.0,
            target_capacity=len(self.active_agents),
            current_capacity=len(self.active_agents),
            metrics=decision['metrics'],
            success=success,
            execution_time=time.time() - start_time,
            error_message=error_message
        )
        
        self.scaling_events.append(event)
        
        # Calculate scaling efficiency
        if success:
            efficiency = self._calculate_scaling_efficiency(event)
            self.scaling_efficiency_history.append(efficiency)
    
    async def _scale_up(self, count: int, reason: str) -> bool:
        """Scale up by adding new agents."""
        logging.info(f"Scaling up: Adding {count} agents - {reason}")
        
        # Determine optimal regions for new agents
        regions = await self._select_optimal_regions_for_scaling(count)
        
        total_provisioned = 0
        
        for region, agent_count in regions.items():
            try:
                # Provision agents in this region
                new_agents = await self.resource_provisioner.provision_agents(
                    count=agent_count,
                    region=region,
                    config={
                        'capacity': 100,
                        'scaling_group': 'auto_scaled'
                    }
                )
                
                # Add to active agents
                for agent in new_agents:
                    self.active_agents[agent.id] = agent
                    self.scaling_groups['auto_scaled'].append(agent.id)
                    total_provisioned += 1
                
                logging.info(f"Provisioned {len(new_agents)} agents in region {region}")
                
            except Exception as e:
                logging.error(f"Failed to provision agents in region {region}: {e}")
        
        success = total_provisioned > 0
        if success:
            logging.info(f"Scale up completed: {total_provisioned}/{count} agents provisioned")
        
        return success
    
    async def _scale_down(self, count: int, reason: str) -> bool:
        """Scale down by removing agents."""
        logging.info(f"Scaling down: Removing {count} agents - {reason}")
        
        # Select agents to terminate (prioritize unhealthy and low-performing agents)
        agents_to_terminate = await self._select_agents_for_termination(count)
        
        if not agents_to_terminate:
            logging.warning("No suitable agents found for termination")
            return False
        
        try:
            # Terminate selected agents
            success = await self.resource_provisioner.terminate_agents(agents_to_terminate)
            
            if success:
                # Remove from active agents
                for agent_id in agents_to_terminate:
                    if agent_id in self.active_agents:
                        del self.active_agents[agent_id]
                    
                    # Remove from scaling groups
                    for group_agents in self.scaling_groups.values():
                        if agent_id in group_agents:
                            group_agents.remove(agent_id)
                
                logging.info(f"Scale down completed: {len(agents_to_terminate)} agents terminated")
                return True
            else:
                logging.error("Failed to terminate agents")
                return False
                
        except Exception as e:
            logging.error(f"Scale down failed: {e}")
            return False
    
    async def _select_optimal_regions_for_scaling(self, total_count: int) -> Dict[str, int]:
        """Select optimal regions for scaling up."""
        # Get available regions and their capacity
        available_regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
        region_allocation = {}
        
        # Simple round-robin allocation for now
        # In production, this would consider latency, cost, resource availability
        agents_per_region = total_count // len(available_regions)
        remainder = total_count % len(available_regions)
        
        for i, region in enumerate(available_regions):
            allocation = agents_per_region
            if i < remainder:
                allocation += 1
            if allocation > 0:
                region_allocation[region] = allocation
        
        return region_allocation
    
    async def _select_agents_for_termination(self, count: int) -> List[str]:
        """Select agents for termination based on performance and health."""
        if count >= len(self.active_agents):
            # Can't terminate more than we have, and must maintain minimum
            max_terminable = len(self.active_agents) - self.min_agents
            count = min(count, max_terminable)
        
        if count <= 0:
            return []
        
        # Score agents for termination (lower score = higher priority for termination)
        agent_scores = []
        
        for agent_id, agent in self.active_agents.items():
            score = agent.health_score
            
            # Penalize agents with recent poor performance
            if agent.performance_history:
                recent_performance = statistics.mean(agent.performance_history[-5:])
                score *= recent_performance
            
            # Prefer terminating agents in over-provisioned regions
            region_agent_count = sum(1 for a in self.active_agents.values() if a.region == agent.region)
            if region_agent_count > 3:  # Threshold for over-provisioning
                score *= 0.8
            
            # Avoid terminating recently provisioned agents
            agent_age = time.time() - agent.last_heartbeat
            if agent_age < 300:  # Less than 5 minutes old
                score *= 2.0
            
            agent_scores.append((score, agent_id))
        
        # Sort by score (ascending) and select lowest scoring agents
        agent_scores.sort(key=lambda x: x[0])
        selected_agents = [agent_id for score, agent_id in agent_scores[:count]]
        
        return selected_agents
    
    async def _check_agent_health(self):
        """Check health of all active agents and handle failures."""
        current_time = time.time()
        failed_agents = []
        
        for agent_id, agent in self.active_agents.items():
            # Check heartbeat timeout
            if current_time - agent.last_heartbeat > 120:  # 2 minutes timeout
                agent.status = "failed"
                failed_agents.append(agent_id)
            
            # Check health score
            elif agent.health_score < 0.3:
                agent.status = "unhealthy"
        
        # Handle failed agents
        if failed_agents:
            logging.warning(f"Detected {len(failed_agents)} failed agents")
            
            # Remove failed agents
            for agent_id in failed_agents:
                if agent_id in self.active_agents:
                    del self.active_agents[agent_id]
            
            # Trigger replacement if below minimum
            current_healthy = len([a for a in self.active_agents.values() if a.status == "healthy"])
            if current_healthy < self.min_agents:
                needed = self.min_agents - current_healthy
                await self._scale_up(needed, "agent_failure_replacement")
    
    def _calculate_scaling_efficiency(self, event: ScalingEvent) -> float:
        """Calculate efficiency of a scaling event."""
        # Efficiency based on execution time and success
        time_efficiency = max(0.0, 1.0 - (event.execution_time / 60.0))  # Penalize slow scaling
        success_efficiency = 1.0 if event.success else 0.0
        
        return (time_efficiency + success_efficiency) / 2.0
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        current_time = time.time()
        
        # Recent scaling events (last hour)
        recent_events = [e for e in self.scaling_events if current_time - e.timestamp < 3600]
        
        # Calculate statistics
        total_scale_ups = len([e for e in recent_events if e.action == ScalingAction.SCALE_UP])
        total_scale_downs = len([e for e in recent_events if e.action == ScalingAction.SCALE_DOWN])
        success_rate = len([e for e in recent_events if e.success]) / len(recent_events) if recent_events else 0.0
        
        avg_efficiency = statistics.mean(self.scaling_efficiency_history) if self.scaling_efficiency_history else 0.0
        
        # Agent statistics
        agent_stats = {
            'total_agents': len(self.active_agents),
            'healthy_agents': len([a for a in self.active_agents.values() if a.status == 'healthy']),
            'regions': len(set(a.region for a in self.active_agents.values())),
            'scaling_groups': len(self.scaling_groups)
        }
        
        # Resource utilization
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]
            avg_cpu = statistics.mean([m['metrics']['system']['cpu_utilization'] for m in recent_metrics])
            avg_memory = statistics.mean([m['metrics']['system']['memory_utilization'] for m in recent_metrics])
        else:
            avg_cpu = avg_memory = 0.0
        
        return {
            'agent_statistics': agent_stats,
            'scaling_events': {
                'total_events': len(recent_events),
                'scale_ups': total_scale_ups,
                'scale_downs': total_scale_downs,
                'success_rate': success_rate,
                'average_efficiency': avg_efficiency
            },
            'resource_utilization': {
                'cpu_utilization': avg_cpu,
                'memory_utilization': avg_memory,
                'target_utilization': self.target_utilization
            },
            'configuration': {
                'min_agents': self.min_agents,
                'max_agents': self.max_agents,
                'scale_up_threshold': self.scale_up_threshold,
                'scale_down_threshold': self.scale_down_threshold,
                'scaling_cooldown': self.scaling_cooldown
            }
        }
    
    async def force_scale_to_target(self, target_agents: int, reason: str = "manual") -> bool:
        """Force scaling to a specific target."""
        current_agents = len(self.active_agents)
        
        if target_agents == current_agents:
            return True
        
        if target_agents > current_agents:
            return await self._scale_up(target_agents - current_agents, f"manual_scale_up: {reason}")
        else:
            return await self._scale_down(current_agents - target_agents, f"manual_scale_down: {reason}")


class MetricsCollector:
    """Collects performance metrics from federated agents."""
    
    async def collect_metrics(self, agents: Dict[str, FederatedAgent]) -> Dict[str, Any]:
        """Collect comprehensive metrics from all agents."""
        
        # Simulate metric collection
        system_metrics = {
            'cpu_utilization': min(0.95, max(0.1, 0.3 + (len(agents) / 1000) + (time.time() % 100) / 200)),
            'memory_utilization': min(0.90, max(0.2, 0.4 + (len(agents) / 1200) + (time.time() % 80) / 160)),
            'throughput': max(0.1, 1.0 - (len(agents) / 5000)),  # Degrades with scale
            'network_latency': 10 + (len(agents) / 100),
            'error_rate': min(0.1, len(agents) / 50000)
        }
        
        # Agent-specific metrics
        agent_metrics = {}
        for agent_id, agent in agents.items():
            agent_metrics[agent_id] = {
                'performance_score': max(0.1, agent.health_score + (time.time() % 10) / 20 - 0.25),
                'load': min(1.0, agent.current_load + (time.time() % 5) / 10),
                'response_time': 50 + (time.time() % 30),
                'memory_usage': 0.3 + (time.time() % 20) / 40
            }
        
        return {
            'timestamp': time.time(),
            'system': system_metrics,
            'agent_metrics': agent_metrics
        }


class LoadPredictor:
    """Predicts future load patterns for proactive scaling."""
    
    def __init__(self):
        self.prediction_history = deque(maxlen=1000)
        
    async def predict_load(self, metrics_history: deque) -> Optional[Dict[str, float]]:
        """Predict future load based on historical patterns."""
        
        if len(metrics_history) < 10:
            return None
        
        # Simple trend-based prediction
        recent_metrics = list(metrics_history)[-10:]
        
        # Calculate trends
        cpu_values = [m['metrics']['system']['cpu_utilization'] for m in recent_metrics]
        memory_values = [m['metrics']['system']['memory_utilization'] for m in recent_metrics]
        
        # Linear trend estimation
        cpu_trend = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
        memory_trend = (memory_values[-1] - memory_values[0]) / len(memory_values)
        
        # Predict next 10 minutes
        prediction_steps = 20  # 20 steps of 30 seconds each = 10 minutes
        
        predicted_cpu = cpu_values[-1] + (cpu_trend * prediction_steps)
        predicted_memory = memory_values[-1] + (memory_trend * prediction_steps)
        
        # Clamp predictions
        predicted_cpu = max(0.0, min(1.0, predicted_cpu))
        predicted_memory = max(0.0, min(1.0, predicted_memory))
        
        # Calculate confidence based on trend consistency
        cpu_variance = statistics.variance(cpu_values) if len(cpu_values) > 1 else 0.0
        confidence = max(0.0, 1.0 - (cpu_variance * 10))  # Higher variance = lower confidence
        
        prediction = {
            'cpu_utilization': predicted_cpu,
            'memory_utilization': predicted_memory,
            'confidence': confidence,
            'prediction_horizon': 600.0  # 10 minutes
        }
        
        self.prediction_history.append({
            'timestamp': time.time(),
            'prediction': prediction
        })
        
        return prediction


class ScalingPolicy:
    """Manages scaling policies and rules."""
    
    def __init__(self):
        self.policies = {
            'conservative': {
                'scale_up_threshold': 0.90,
                'scale_down_threshold': 0.40,
                'scaling_cooldown': 600.0
            },
            'aggressive': {
                'scale_up_threshold': 0.70,
                'scale_down_threshold': 0.60,
                'scaling_cooldown': 180.0
            },
            'predictive': {
                'scale_up_threshold': 0.80,
                'scale_down_threshold': 0.50,
                'scaling_cooldown': 300.0,
                'enable_predictive': True
            }
        }
    
    def get_policy(self, policy_name: str) -> Dict[str, Any]:
        """Get scaling policy configuration."""
        return self.policies.get(policy_name, self.policies['conservative'])