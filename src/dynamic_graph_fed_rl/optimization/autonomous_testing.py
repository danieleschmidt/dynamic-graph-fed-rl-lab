import secrets
"""
Autonomous A/B Testing Framework for Algorithm Variants.

Automatically designs, executes, and analyzes A/B tests to compare
algorithm variants and make data-driven deployment decisions.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from scipy import stats
import logging

from ..quantum_planner.performance import PerformanceMonitor
from .automl_pipeline import AlgorithmVariant


class TestStatus(Enum):
    """A/B test status."""
    DESIGNING = "designing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class TestResult(Enum):
    """A/B test result."""
    VARIANT_A_WINS = "variant_a_wins"
    VARIANT_B_WINS = "variant_b_wins"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    INCONCLUSIVE = "inconclusive"


@dataclass
class TestMetric:
    """A/B test metric definition."""
    name: str
    description: str
    higher_is_better: bool
    weight: float  # Importance weight for overall decision
    min_detectable_effect: float  # Minimum effect size to detect
    baseline_value: Optional[float] = None
    current_a_value: float = 0.0
    current_b_value: float = 0.0
    statistical_significance: float = 0.0
    effect_size: float = 0.0


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    name: str
    description: str
    variant_a_id: str  # Control
    variant_b_id: str  # Treatment
    traffic_allocation: Tuple[float, float]  # (A%, B%)
    test_metrics: List[TestMetric]
    min_sample_size: int = 100
    max_duration_hours: float = 24.0
    significance_threshold: float = 0.05
    power: float = 0.8
    early_stopping: bool = True
    guardrail_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ABTestExecution:
    """A/B test execution state."""
    config: ABTestConfig
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    samples_a: int = 0
    samples_b: int = 0
    data_a: Dict[str, List[float]] = field(default_factory=dict)
    data_b: Dict[str, List[float]] = field(default_factory=dict)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    final_result: Optional[TestResult] = None
    result_confidence: float = 0.0
    winner_variant_id: Optional[str] = None
    insights: List[str] = field(default_factory=list)


class AutonomousABTester:
    """
    Autonomous A/B testing framework.
    
    Features:
    - Automated test design and power analysis
    - Smart traffic allocation and ramping
    - Real-time statistical monitoring
    - Early stopping for clear winners
    - Guardrail monitoring for safety
    - Multi-metric optimization
    - Automated result interpretation and deployment
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        min_test_duration: float = 3600.0,  # 1 hour minimum
        max_concurrent_tests: int = 3,
        confidence_threshold: float = 0.95,
        logger: Optional[logging.Logger] = None,
    ):
        self.performance_monitor = performance_monitor
        self.min_test_duration = min_test_duration
        self.max_concurrent_tests = max_concurrent_tests
        self.confidence_threshold = confidence_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Test management
        self.active_tests: Dict[str, ABTestExecution] = {}
        self.test_history: List[ABTestExecution] = []
        self.test_queue: List[ABTestConfig] = []
        
        # Variant management
        self.available_variants: Dict[str, AlgorithmVariant] = {}
        self.variant_performance_history: Dict[str, List[Dict[str, float]]] = {}
        
        # Testing state
        self.is_running = False
        self.traffic_router: Dict[str, str] = {}  # Request ID -> Variant ID
        self.current_traffic_split: Dict[str, float] = {}
        
        # Analytics
        self.test_insights: List[Dict[str, Any]] = []
        self.deployment_recommendations: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_tests_run = 0
        self.successful_tests = 0
        self.early_stopped_tests = 0
        self.failed_tests = 0
        
    async def start_ab_testing_system(self):
        """Start the autonomous A/B testing system."""
        self.is_running = True
        self.logger.info("Starting autonomous A/B testing system")
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._test_execution_loop()),
            asyncio.create_task(self._data_collection_loop()),
            asyncio.create_task(self._statistical_analysis_loop()),
            asyncio.create_task(self._guardrail_monitoring_loop()),
            asyncio.create_task(self._test_scheduling_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"A/B testing system error: {e}")
        finally:
            self.is_running = False
    
    async def stop_ab_testing_system(self):
        """Stop the A/B testing system."""
        self.is_running = False
        self.logger.info("Stopping autonomous A/B testing system")
    
    async def create_ab_test(
        self,
        variant_a: AlgorithmVariant,
        variant_b: AlgorithmVariant,
        test_name: str = None,
        custom_metrics: List[TestMetric] = None,
    ) -> str:
        """Create a new A/B test between two variants."""
        test_id = str(uuid.uuid4())
        
        if not test_name:
            test_name = f"Test {variant_a.name} vs {variant_b.name}"
        
        # Store variants
        self.available_variants[variant_a.id] = variant_a
        self.available_variants[variant_b.id] = variant_b
        
        # Define default test metrics
        default_metrics = [
            TestMetric(
                "accuracy", "Model accuracy", True, 0.4, 0.02
            ),
            TestMetric(
                "latency", "Response latency (ms)", False, 0.2, 5.0
            ),
            TestMetric(
                "throughput", "Requests per second", True, 0.2, 1.0
            ),
            TestMetric(
                "error_rate", "Error rate (%)", False, 0.15, 0.5
            ),
            TestMetric(
                "resource_efficiency", "Resource efficiency score", True, 0.05, 0.01
            ),
        ]
        
        test_metrics = custom_metrics or default_metrics
        
        # Calculate sample size and duration
        min_sample_size, estimated_duration = self._calculate_test_parameters(test_metrics)
        
        # Create test configuration
        config = ABTestConfig(
            test_id=test_id,
            name=test_name,
            description=f"Comparing {variant_a.name} (A) vs {variant_b.name} (B)",
            variant_a_id=variant_a.id,
            variant_b_id=variant_b.id,
            traffic_allocation=(0.5, 0.5),  # 50-50 split
            test_metrics=test_metrics,
            min_sample_size=min_sample_size,
            max_duration_hours=min(estimated_duration, 48.0),  # Cap at 48 hours
            guardrail_metrics={
                "error_rate": 10.0,  # Stop if error rate > 10%
                "latency": 1000.0,   # Stop if latency > 1000ms
            },
        )
        
        # Add to queue
        self.test_queue.append(config)
        
        self.logger.info(f"A/B test created: {test_name} (ID: {test_id})")
        return test_id
    
    def _calculate_test_parameters(self, metrics: List[TestMetric]) -> Tuple[int, float]:
        """Calculate required sample size and estimated duration."""
        # Use the most restrictive metric for sample size calculation
        max_sample_size = 0
        
        for metric in metrics:
            # Power analysis for detecting minimum effect
            alpha = 0.05
            beta = 0.2  # 80% power
            effect_size = metric.min_detectable_effect
            
            # Simplified sample size calculation
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(1 - beta)
            
            # Assume standard deviation is 10% of baseline
            baseline_std = 0.1  # Simplified assumption
            
            sample_size = 2 * ((z_alpha + z_beta) * baseline_std / effect_size) ** 2
            sample_size = max(50, int(sample_size))  # Minimum 50 samples
            
            max_sample_size = max(max_sample_size, sample_size)
        
        # Estimate duration based on expected traffic
        expected_rps = 10  # Requests per second
        estimated_hours = max_sample_size / (expected_rps * 3600 * 0.5)  # 50% gets treatment
        
        return max_sample_size, estimated_hours
    
    async def _test_scheduling_loop(self):
        """Schedule and start queued tests."""
        while self.is_running:
            try:
                # Check if we can start new tests
                if (len(self.active_tests) < self.max_concurrent_tests and 
                    self.test_queue):
                    
                    config = self.test_queue.pop(0)
                    await self._start_test(config)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Test scheduling loop error: {e}")
                await asyncio.sleep(300)
    
    async def _start_test(self, config: ABTestConfig):
        """Start executing an A/B test."""
        try:
            # Create test execution
            execution = ABTestExecution(
                config=config,
                status=TestStatus.RUNNING,
                start_time=datetime.now(),
            )
            
            # Initialize data collection structures
            for metric in config.test_metrics:
                execution.data_a[metric.name] = []
                execution.data_b[metric.name] = []
            
            self.active_tests[config.test_id] = execution
            self.total_tests_run += 1
            
            # Configure traffic splitting
            self._configure_traffic_split(config)
            
            self.logger.info(f"Started A/B test: {config.name} (ID: {config.test_id})")
            
        except Exception as e:
            self.logger.error(f"Failed to start test {config.test_id}: {e}")
            self.failed_tests += 1
    
    def _configure_traffic_split(self, config: ABTestConfig):
        """Configure traffic splitting for the test."""
        # This would integrate with traffic routing system
        # For simulation, we'll use probabilistic assignment
        
        self.current_traffic_split = {
            config.variant_a_id: config.traffic_allocation[0],
            config.variant_b_id: config.traffic_allocation[1],
        }
        
        self.logger.debug(f"Traffic split configured: {self.current_traffic_split}")
    
    async def _test_execution_loop(self):
        """Monitor and manage running tests."""
        while self.is_running:
            try:
                completed_tests = []
                
                for test_id, execution in self.active_tests.items():
                    if await self._should_stop_test(execution):
                        await self._stop_test(execution)
                        completed_tests.append(test_id)
                
                # Remove completed tests
                for test_id in completed_tests:
                    completed_test = self.active_tests.pop(test_id)
                    self.test_history.append(completed_test)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Test execution loop error: {e}")
                await asyncio.sleep(60)
    
    async def _should_stop_test(self, execution: ABTestExecution) -> bool:
        """Determine if a test should be stopped."""
        current_time = datetime.now()
        runtime_hours = (current_time - execution.start_time).total_seconds() / 3600
        
        # Check maximum duration
        if runtime_hours >= execution.config.max_duration_hours:
            execution.insights.append("Test stopped due to maximum duration reached")
            return True
        
        # Check minimum duration
        if runtime_hours < self.min_test_duration / 3600:
            return False
        
        # Check minimum sample size
        if (execution.samples_a < execution.config.min_sample_size or 
            execution.samples_b < execution.config.min_sample_size):
            return False
        
        # Check for statistical significance (early stopping)
        if execution.config.early_stopping:
            if await self._check_statistical_significance(execution):
                execution.insights.append("Test stopped early due to statistical significance")
                return True
        
        # Check guardrails
        if await self._check_guardrails(execution):
            execution.insights.append("Test stopped due to guardrail violation")
            return True
        
        return False
    
    async def _check_statistical_significance(self, execution: ABTestExecution) -> bool:
        """Check if test results are statistically significant."""
        significant_metrics = 0
        total_metrics = len(execution.config.test_metrics)
        
        for metric in execution.config.test_metrics:
            if metric.name in execution.data_a and metric.name in execution.data_b:
                data_a = execution.data_a[metric.name]
                data_b = execution.data_b[metric.name]
                
                if len(data_a) >= 30 and len(data_b) >= 30:
                    # Perform t-test
                    try:
                        statistic, p_value = stats.ttest_ind(data_a, data_b)
                        
                        if p_value < execution.config.significance_threshold:
                            significant_metrics += 1
                            
                            # Update metric statistics
                            metric.statistical_significance = 1 - p_value
                            metric.current_a_value = np.mean(data_a)
                            metric.current_b_value = np.mean(data_b)
                            metric.effect_size = abs(metric.current_b_value - metric.current_a_value)
                            
                    except Exception as e:
                        self.logger.warning(f"Statistical test failed for {metric.name}: {e}")
        
        # Require significance in at least 50% of metrics
        return significant_metrics >= total_metrics * 0.5
    
    async def _check_guardrails(self, execution: ABTestExecution) -> bool:
        """Check if guardrail metrics are violated."""
        for metric_name, threshold in execution.config.guardrail_metrics.items():
            if metric_name in execution.data_b:
                current_value = np.mean(execution.data_b[metric_name][-10:])  # Recent average
                
                # Check if current value violates guardrail
                if metric_name in ["error_rate", "latency"]:  # Higher is worse
                    if current_value > threshold:
                        return True
                else:  # Higher is better
                    if current_value < threshold:
                        return True
        
        return False
    
    async def _stop_test(self, execution: ABTestExecution):
        """Stop a running test and analyze results."""
        execution.status = TestStatus.COMPLETED
        execution.end_time = datetime.now()
        
        # Perform final statistical analysis
        await self._analyze_test_results(execution)
        
        # Generate insights
        await self._generate_test_insights(execution)
        
        # Make deployment recommendation
        await self._make_deployment_recommendation(execution)
        
        self.successful_tests += 1
        self.logger.info(f"A/B test completed: {execution.config.name}")
    
    async def _data_collection_loop(self):
        """Collect performance data for running tests."""
        while self.is_running:
            try:
                for test_id, execution in self.active_tests.items():
                    if execution.status == TestStatus.RUNNING:
                        await self._collect_test_data(execution)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Data collection loop error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_test_data(self, execution: ABTestExecution):
        """Collect performance data for a specific test."""
        try:
            # Get current performance metrics
            perf_metrics = await self.performance_monitor.get_current_metrics()
            
            # Simulate data points for variants (in practice, this would come from real traffic)
            for _ in range(5):  # Simulate 5 data points
                variant_id = self._assign_traffic_to_variant(execution.config)
                
                # Generate realistic performance data with some variance
                base_metrics = {
                    "accuracy": 0.85,
                    "latency": 120.0,
                    "throughput": 15.0,
                    "error_rate": 1.5,
                    "resource_efficiency": 0.75,
                }
                
                # Add variant-specific effects
                if variant_id == execution.config.variant_b_id:
                    # Variant B might have different performance characteristics
                    base_metrics["accuracy"] += np.random.normal(0.02, 0.01)
                    base_metrics["latency"] += np.random.normal(-5, 10)
                    base_metrics["throughput"] += np.random.normal(1, 2)
                    base_metrics["error_rate"] += np.random.normal(-0.2, 0.3)
                    base_metrics["resource_efficiency"] += np.random.normal(0.05, 0.02)
                
                # Add noise
                for metric_name in base_metrics:
                    base_metrics[metric_name] += np.random.normal(0, base_metrics[metric_name] * 0.05)
                
                # Store data
                data_dict = execution.data_a if variant_id == execution.config.variant_a_id else execution.data_b
                
                for metric in execution.config.test_metrics:
                    if metric.name in base_metrics:
                        data_dict[metric.name].append(base_metrics[metric.name])
                
                # Update sample counts
                if variant_id == execution.config.variant_a_id:
                    execution.samples_a += 1
                else:
                    execution.samples_b += 1
        
        except Exception as e:
            self.logger.error(f"Failed to collect test data: {e}")
    
    def _assign_traffic_to_variant(self, config: ABTestConfig) -> str:
        """Assign traffic to variant based on allocation."""
        if np.secrets.SystemRandom().random() < config.traffic_allocation[0]:
            return config.variant_a_id
        else:
            return config.variant_b_id
    
    async def _statistical_analysis_loop(self):
        """Perform ongoing statistical analysis of test results."""
        while self.is_running:
            try:
                for execution in self.active_tests.values():
                    if execution.status == TestStatus.RUNNING:
                        await self._update_intermediate_results(execution)
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Statistical analysis loop error: {e}")
                await asyncio.sleep(600)
    
    async def _update_intermediate_results(self, execution: ABTestExecution):
        """Update intermediate statistical results."""
        try:
            results = {
                "timestamp": datetime.now(),
                "samples_a": execution.samples_a,
                "samples_b": execution.samples_b,
                "metrics": {},
            }
            
            for metric in execution.config.test_metrics:
                if (metric.name in execution.data_a and metric.name in execution.data_b and
                    len(execution.data_a[metric.name]) >= 10 and len(execution.data_b[metric.name]) >= 10):
                    
                    data_a = execution.data_a[metric.name]
                    data_b = execution.data_b[metric.name]
                    
                    mean_a = np.mean(data_a)
                    mean_b = np.mean(data_b)
                    
                    # Statistical test
                    try:
                        statistic, p_value = stats.ttest_ind(data_a, data_b)
                        effect_size = abs(mean_b - mean_a) / np.sqrt((np.var(data_a) + np.var(data_b)) / 2)
                        
                        results["metrics"][metric.name] = {
                            "mean_a": mean_a,
                            "mean_b": mean_b,
                            "p_value": p_value,
                            "effect_size": effect_size,
                            "significant": p_value < execution.config.significance_threshold,
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Statistical test failed for {metric.name}: {e}")
            
            execution.intermediate_results.append(results)
            
            # Keep only recent results
            if len(execution.intermediate_results) > 100:
                execution.intermediate_results = execution.intermediate_results[-100:]
                
        except Exception as e:
            self.logger.error(f"Failed to update intermediate results: {e}")
    
    async def _guardrail_monitoring_loop(self):
        """Monitor guardrail metrics for running tests."""
        while self.is_running:
            try:
                for execution in self.active_tests.values():
                    if execution.status == TestStatus.RUNNING:
                        await self._monitor_guardrails(execution)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Guardrail monitoring loop error: {e}")
                await asyncio.sleep(120)
    
    async def _monitor_guardrails(self, execution: ABTestExecution):
        """Monitor guardrail metrics for a specific test."""
        try:
            violations = []
            
            for metric_name, threshold in execution.config.guardrail_metrics.items():
                if metric_name in execution.data_b and len(execution.data_b[metric_name]) >= 10:
                    recent_values = execution.data_b[metric_name][-10:]
                    current_value = np.mean(recent_values)
                    
                    # Check violation
                    violated = False
                    if metric_name in ["error_rate", "latency"]:  # Higher is worse
                        violated = current_value > threshold
                    else:  # Higher is better
                        violated = current_value < threshold
                    
                    if violated:
                        violations.append({
                            "metric": metric_name,
                            "current_value": current_value,
                            "threshold": threshold,
                        })
            
            if violations:
                execution.insights.append(f"Guardrail violations detected: {violations}")
                self.logger.warning(f"Guardrail violations in test {execution.config.test_id}: {violations}")
        
        except Exception as e:
            self.logger.error(f"Failed to monitor guardrails: {e}")
    
    async def _analyze_test_results(self, execution: ABTestExecution):
        """Perform final analysis of test results."""
        try:
            # Determine overall winner
            variant_b_wins = 0
            total_weighted_score = 0
            
            for metric in execution.config.test_metrics:
                if (metric.name in execution.data_a and metric.name in execution.data_b and
                    len(execution.data_a[metric.name]) >= 10 and len(execution.data_b[metric.name]) >= 10):
                    
                    data_a = execution.data_a[metric.name]
                    data_b = execution.data_b[metric.name]
                    
                    mean_a = np.mean(data_a)
                    mean_b = np.mean(data_b)
                    
                    # Check if B is better than A
                    if metric.higher_is_better:
                        b_better = mean_b > mean_a
                    else:
                        b_better = mean_b < mean_a
                    
                    if b_better:
                        # Perform statistical test
                        _, p_value = stats.ttest_ind(data_a, data_b)
                        
                        if p_value < execution.config.significance_threshold:
                            variant_b_wins += metric.weight
                    
                    total_weighted_score += metric.weight
            
            # Determine result
            if total_weighted_score == 0:
                execution.final_result = TestResult.INCONCLUSIVE
                execution.result_confidence = 0.0
            else:
                b_win_percentage = variant_b_wins / total_weighted_score
                
                if b_win_percentage > 0.6:
                    execution.final_result = TestResult.VARIANT_B_WINS
                    execution.winner_variant_id = execution.config.variant_b_id
                    execution.result_confidence = b_win_percentage
                elif b_win_percentage < 0.4:
                    execution.final_result = TestResult.VARIANT_A_WINS
                    execution.winner_variant_id = execution.config.variant_a_id
                    execution.result_confidence = 1.0 - b_win_percentage
                else:
                    execution.final_result = TestResult.NO_SIGNIFICANT_DIFFERENCE
                    execution.result_confidence = 0.5
            
            self.logger.info(f"Test analysis complete: {execution.final_result.value} (confidence: {execution.result_confidence:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze test results: {e}")
            execution.final_result = TestResult.INCONCLUSIVE
    
    async def _generate_test_insights(self, execution: ABTestExecution):
        """Generate insights from test results."""
        try:
            insights = []
            
            # Sample size insights
            total_samples = execution.samples_a + execution.samples_b
            insights.append(f"Test collected {total_samples} total samples ({execution.samples_a} for A, {execution.samples_b} for B)")
            
            # Duration insights
            if execution.end_time:
                duration_hours = (execution.end_time - execution.start_time).total_seconds() / 3600
                insights.append(f"Test duration: {duration_hours:.1f} hours")
            
            # Performance insights
            for metric in execution.config.test_metrics:
                if (metric.name in execution.data_a and metric.name in execution.data_b and
                    len(execution.data_a[metric.name]) >= 10 and len(execution.data_b[metric.name]) >= 10):
                    
                    mean_a = np.mean(execution.data_a[metric.name])
                    mean_b = np.mean(execution.data_b[metric.name])
                    
                    improvement = ((mean_b - mean_a) / mean_a) * 100 if mean_a != 0 else 0
                    
                    if abs(improvement) > 1:  # Only report significant changes
                        direction = "improved" if improvement > 0 else "decreased"
                        insights.append(f"{metric.name}: Variant B {direction} by {abs(improvement):.1f}%")
            
            execution.insights.extend(insights)
            
        except Exception as e:
            self.logger.error(f"Failed to generate insights: {e}")
    
    async def _make_deployment_recommendation(self, execution: ABTestExecution):
        """Make deployment recommendation based on test results."""
        try:
            recommendation = {
                "test_id": execution.config.test_id,
                "timestamp": datetime.now(),
                "result": execution.final_result.value,
                "winner_variant_id": execution.winner_variant_id,
                "confidence": execution.result_confidence,
                "recommendation": "no_action",
                "reasoning": [],
            }
            
            if execution.final_result == TestResult.VARIANT_B_WINS and execution.result_confidence >= 0.8:
                recommendation["recommendation"] = "deploy_variant_b"
                recommendation["reasoning"].append(f"Variant B significantly outperforms Variant A (confidence: {execution.result_confidence:.1%})")
            
            elif execution.final_result == TestResult.VARIANT_A_WINS and execution.result_confidence >= 0.8:
                recommendation["recommendation"] = "keep_variant_a"
                recommendation["reasoning"].append(f"Variant A significantly outperforms Variant B (confidence: {execution.result_confidence:.1%})")
            
            elif execution.final_result == TestResult.NO_SIGNIFICANT_DIFFERENCE:
                recommendation["recommendation"] = "keep_current"
                recommendation["reasoning"].append("No significant performance difference detected")
            
            else:
                recommendation["reasoning"].append("Insufficient evidence for deployment decision")
            
            self.deployment_recommendations.append(recommendation)
            
            self.logger.info(f"Deployment recommendation: {recommendation['recommendation']}")
            
        except Exception as e:
            self.logger.error(f"Failed to make deployment recommendation: {e}")
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific test."""
        if test_id in self.active_tests:
            execution = self.active_tests[test_id]
        else:
            execution = next((t for t in self.test_history if t.config.test_id == test_id), None)
        
        if not execution:
            return None
        
        return {
            "test_id": test_id,
            "name": execution.config.name,
            "status": execution.status.value,
            "start_time": execution.start_time.isoformat(),
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "samples_a": execution.samples_a,
            "samples_b": execution.samples_b,
            "result": execution.final_result.value if execution.final_result else None,
            "confidence": execution.result_confidence,
            "winner_variant_id": execution.winner_variant_id,
            "insights": execution.insights,
        }
    
    def get_testing_stats(self) -> Dict[str, Any]:
        """Get A/B testing system statistics."""
        return {
            "is_running": self.is_running,
            "active_tests": len(self.active_tests),
            "queued_tests": len(self.test_queue),
            "total_tests_run": self.total_tests_run,
            "successful_tests": self.successful_tests,
            "failed_tests": self.failed_tests,
            "early_stopped_tests": self.early_stopped_tests,
            "success_rate": self.successful_tests / max(1, self.total_tests_run),
            "recent_recommendations": len([r for r in self.deployment_recommendations 
                                         if (datetime.now() - r["timestamp"]).days < 7]),
            "available_variants": len(self.available_variants),
        }