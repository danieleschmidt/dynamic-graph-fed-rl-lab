"""
Predictive Health Monitoring System with Machine Learning-based Failure Detection.

This module provides advanced health monitoring capabilities including:
- Real-time performance metrics collection
- Machine learning-based anomaly detection
- Predictive failure analysis
- Automated alert generation and escalation
- Trend analysis and capacity planning
- Integration with existing health monitoring
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import numpy as np
from collections import deque

from ..utils.error_handling import (
    circuit_breaker, retry, robust, SecurityError, ValidationError,
    CircuitBreakerConfig, RetryConfig, resilience
)
from .health_monitor import HealthMonitor, HealthStatus, ComponentHealth


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricTrend:
    """Trend analysis for a specific metric."""
    metric_name: str
    current_value: float
    trend_direction: str  # increasing, decreasing, stable, volatile
    trend_strength: float  # 0.0 to 1.0
    prediction_horizon: float  # seconds
    predicted_value: float
    confidence: PredictionConfidence
    anomaly_score: float = 0.0
    seasonal_pattern: bool = False


@dataclass
class FailurePrediction:
    """Prediction of potential system failure."""
    prediction_id: str
    component_name: str
    failure_type: str
    probability: float
    time_to_failure: float  # seconds
    confidence: PredictionConfidence
    contributing_factors: List[str]
    recommended_actions: List[str]
    created_at: float = field(default_factory=time.time)


@dataclass
class HealthAlert:
    """Health monitoring alert."""
    alert_id: str
    component: str
    severity: AlertSeverity
    message: str
    metrics: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False


class TimeSeriesAnalyzer:
    """Time series analysis for predictive monitoring."""
    
    def __init__(self, window_size: int = 100, prediction_horizon: int = 60):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon  # seconds
        self.metrics_history: Dict[str, deque] = {}
        self.seasonal_patterns: Dict[str, Dict[str, float]] = {}
        
    def add_metric(self, metric_name: str, value: float, timestamp: float):
        """Add metric value to time series."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=self.window_size)
        
        self.metrics_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def analyze_trend(self, metric_name: str) -> Optional[MetricTrend]:
        """Analyze trend for a specific metric."""
        if metric_name not in self.metrics_history:
            return None
        
        history = list(self.metrics_history[metric_name])
        if len(history) < 10:  # Need at least 10 data points
            return None
        
        # Extract values and timestamps
        values = np.array([point['value'] for point in history])
        timestamps = np.array([point['timestamp'] for point in history])
        
        # Calculate trend using linear regression
        if len(values) > 1:
            # Normalize timestamps for better numerical stability
            norm_timestamps = (timestamps - timestamps[0]) / 60.0  # Convert to minutes
            
            # Linear regression
            coeffs = np.polyfit(norm_timestamps, values, 1)
            slope = coeffs[0]
            
            # Determine trend direction and strength
            trend_direction = "stable"
            trend_strength = 0.0
            
            if abs(slope) > 0.01:  # Threshold for significant trend
                if slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
                
                # Calculate trend strength based on R-squared
                y_pred = np.polyval(coeffs, norm_timestamps)
                ss_res = np.sum((values - y_pred) ** 2)
                ss_tot = np.sum((values - np.mean(values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                trend_strength = max(0.0, min(1.0, r_squared))
            
            # Check for volatility
            if np.std(values) / (np.mean(values) + 1e-8) > 0.3:  # High coefficient of variation
                trend_direction = "volatile"
                trend_strength = np.std(values) / (np.mean(values) + 1e-8)
            
            # Predict future value
            future_timestamp = (timestamps[-1] + self.prediction_horizon - timestamps[0]) / 60.0
            predicted_value = float(np.polyval(coeffs, future_timestamp))
            
            # Determine confidence based on trend strength and data quality
            if trend_strength > 0.8:
                confidence = PredictionConfidence.HIGH
            elif trend_strength > 0.5:
                confidence = PredictionConfidence.MEDIUM
            else:
                confidence = PredictionConfidence.LOW
            
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(values)
            
            return MetricTrend(
                metric_name=metric_name,
                current_value=float(values[-1]),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                prediction_horizon=self.prediction_horizon,
                predicted_value=predicted_value,
                confidence=confidence,
                anomaly_score=anomaly_score,
                seasonal_pattern=self._detect_seasonal_pattern(metric_name, values, timestamps)
            )
        
        return None
    
    def _calculate_anomaly_score(self, values: np.ndarray) -> float:
        """Calculate anomaly score for recent values."""
        if len(values) < 10:
            return 0.0
        
        # Use z-score based anomaly detection
        recent_values = values[-5:]  # Last 5 values
        historical_mean = np.mean(values[:-5])
        historical_std = np.std(values[:-5])
        
        if historical_std > 0:
            z_scores = np.abs((recent_values - historical_mean) / historical_std)
            max_z_score = np.max(z_scores)
            
            # Normalize to 0-1 scale
            anomaly_score = min(max_z_score / 3.0, 1.0)  # z-score > 3 is highly anomalous
            return anomaly_score
        
        return 0.0
    
    def _detect_seasonal_pattern(self, metric_name: str, values: np.ndarray, timestamps: np.ndarray) -> bool:
        """Detect seasonal patterns in the data."""
        if len(values) < 50:  # Need enough data for pattern detection
            return False
        
        # Simple autocorrelation-based seasonal detection
        # Check for daily patterns (assuming data points are collected regularly)
        try:
            from scipy import signal
            
            # Autocorrelation
            autocorr = signal.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for peaks that might indicate seasonal patterns
            peaks, _ = signal.find_peaks(autocorr[1:], height=np.max(autocorr) * 0.5)
            
            if len(peaks) > 0:
                # Check if peaks are regularly spaced (indicating seasonality)
                peak_intervals = np.diff(peaks)
                if len(peak_intervals) > 1 and np.std(peak_intervals) / np.mean(peak_intervals) < 0.2:
                    return True
        except ImportError:
            # Fall back to simple variance-based detection
            pass
        
        return False
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        summary = {
            'total_metrics': len(self.metrics_history),
            'window_size': self.window_size,
            'prediction_horizon': self.prediction_horizon,
            'metrics': {}
        }
        
        for metric_name in self.metrics_history:
            history = list(self.metrics_history[metric_name])
            if history:
                values = [point['value'] for point in history]
                summary['metrics'][metric_name] = {
                    'data_points': len(history),
                    'current_value': values[-1],
                    'min_value': min(values),
                    'max_value': max(values),
                    'mean_value': np.mean(values),
                    'std_value': np.std(values)
                }
        
        return summary


class FailurePredictor:
    """Machine learning-based failure prediction system."""
    
    def __init__(self):
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.prediction_history: List[FailurePrediction] = []
        self.learning_enabled = True
        
        # Failure pattern templates
        self.failure_thresholds = {
            'cpu_usage': {'critical': 95.0, 'warning': 85.0},
            'memory_usage': {'critical': 95.0, 'warning': 85.0},
            'disk_usage': {'critical': 98.0, 'warning': 90.0},
            'error_rate': {'critical': 10.0, 'warning': 5.0},
            'response_time': {'critical': 5.0, 'warning': 2.0}
        }
    
    def predict_failures(
        self,
        component_name: str,
        current_metrics: Dict[str, float],
        metric_trends: Dict[str, MetricTrend]
    ) -> List[FailurePrediction]:
        """Predict potential failures based on current metrics and trends."""
        predictions = []
        
        # Rule-based predictions
        rule_predictions = self._rule_based_predictions(component_name, current_metrics, metric_trends)
        predictions.extend(rule_predictions)
        
        # Pattern-based predictions
        pattern_predictions = self._pattern_based_predictions(component_name, current_metrics, metric_trends)
        predictions.extend(pattern_predictions)
        
        # Trend-based predictions
        trend_predictions = self._trend_based_predictions(component_name, metric_trends)
        predictions.extend(trend_predictions)
        
        # Store predictions for learning
        if self.learning_enabled:
            self.prediction_history.extend(predictions)
            self._update_failure_patterns(component_name, current_metrics, predictions)
        
        return predictions
    
    def _rule_based_predictions(
        self,
        component_name: str,
        current_metrics: Dict[str, float],
        metric_trends: Dict[str, MetricTrend]
    ) -> List[FailurePrediction]:
        """Generate predictions based on predefined rules."""
        predictions = []
        
        for metric_name, value in current_metrics.items():
            if metric_name in self.failure_thresholds:
                thresholds = self.failure_thresholds[metric_name]
                
                # Critical threshold prediction
                if value >= thresholds['critical']:
                    prediction = FailurePrediction(
                        prediction_id=f"rule_{component_name}_{metric_name}_{int(time.time())}",
                        component_name=component_name,
                        failure_type=f"{metric_name}_critical",
                        probability=0.9,
                        time_to_failure=300.0,  # 5 minutes
                        confidence=PredictionConfidence.HIGH,
                        contributing_factors=[f"{metric_name} at critical level: {value}"],
                        recommended_actions=[
                            f"Immediate intervention required for {metric_name}",
                            "Scale resources if possible",
                            "Alert operations team"
                        ]
                    )
                    predictions.append(prediction)
                
                # Warning threshold with trend analysis
                elif value >= thresholds['warning']:
                    trend = metric_trends.get(metric_name)
                    if trend and trend.trend_direction == "increasing":
                        # Predict when it will reach critical
                        time_to_critical = self._estimate_time_to_threshold(
                            current_value=value,
                            target_threshold=thresholds['critical'],
                            trend=trend
                        )
                        
                        if time_to_critical < 3600:  # Within 1 hour
                            prediction = FailurePrediction(
                                prediction_id=f"rule_{component_name}_{metric_name}_trend_{int(time.time())}",
                                component_name=component_name,
                                failure_type=f"{metric_name}_projected_critical",
                                probability=0.7,
                                time_to_failure=time_to_critical,
                                confidence=trend.confidence,
                                contributing_factors=[
                                    f"{metric_name} at warning level: {value}",
                                    f"Increasing trend detected: {trend.trend_direction}"
                                ],
                                recommended_actions=[
                                    f"Monitor {metric_name} closely",
                                    "Prepare scaling actions",
                                    "Review recent changes"
                                ]
                            )
                            predictions.append(prediction)
        
        return predictions
    
    def _pattern_based_predictions(
        self,
        component_name: str,
        current_metrics: Dict[str, float],
        metric_trends: Dict[str, MetricTrend]
    ) -> List[FailurePrediction]:
        """Generate predictions based on learned failure patterns."""
        predictions = []
        
        if component_name not in self.failure_patterns:
            return predictions
        
        patterns = self.failure_patterns[component_name]
        
        for pattern in patterns[-10:]:  # Check last 10 patterns
            similarity = self._calculate_pattern_similarity(current_metrics, pattern['metrics'])
            
            if similarity > 0.8:  # High similarity to past failure pattern
                prediction = FailurePrediction(
                    prediction_id=f"pattern_{component_name}_{int(time.time())}",
                    component_name=component_name,
                    failure_type=pattern.get('failure_type', 'pattern_based'),
                    probability=similarity,
                    time_to_failure=pattern.get('time_to_failure', 1800.0),
                    confidence=PredictionConfidence.MEDIUM,
                    contributing_factors=[
                        f"Pattern similarity: {similarity:.2f}",
                        "Matches historical failure pattern"
                    ],
                    recommended_actions=[
                        "Apply known mitigation strategies",
                        "Monitor closely",
                        "Prepare for potential failure"
                    ]
                )
                predictions.append(prediction)
        
        return predictions
    
    def _trend_based_predictions(
        self,
        component_name: str,
        metric_trends: Dict[str, MetricTrend]
    ) -> List[FailurePrediction]:
        """Generate predictions based on trend analysis."""
        predictions = []
        
        for metric_name, trend in metric_trends.items():
            if trend.anomaly_score > 0.8:  # High anomaly score
                prediction = FailurePrediction(
                    prediction_id=f"anomaly_{component_name}_{metric_name}_{int(time.time())}",
                    component_name=component_name,
                    failure_type=f"{metric_name}_anomaly",
                    probability=trend.anomaly_score,
                    time_to_failure=3600.0,  # 1 hour default
                    confidence=trend.confidence,
                    contributing_factors=[
                        f"High anomaly score: {trend.anomaly_score:.2f}",
                        f"Trend: {trend.trend_direction}",
                        f"Current value: {trend.current_value}"
                    ],
                    recommended_actions=[
                        "Investigate anomalous behavior",
                        "Check for recent changes",
                        "Consider preventive measures"
                    ]
                )
                predictions.append(prediction)
        
        return predictions
    
    def _estimate_time_to_threshold(
        self,
        current_value: float,
        target_threshold: float,
        trend: MetricTrend
    ) -> float:
        """Estimate time until metric reaches threshold."""
        if trend.trend_direction != "increasing":
            return float('inf')
        
        if trend.predicted_value <= current_value:
            return float('inf')
        
        # Linear extrapolation
        value_change = trend.predicted_value - current_value
        threshold_change = target_threshold - current_value
        
        if value_change > 0:
            time_ratio = threshold_change / value_change
            estimated_time = trend.prediction_horizon * time_ratio
            return max(0.0, estimated_time)
        
        return float('inf')
    
    def _calculate_pattern_similarity(
        self,
        current_metrics: Dict[str, float],
        historical_metrics: Dict[str, float]
    ) -> float:
        """Calculate similarity between current and historical metric patterns."""
        common_keys = set(current_metrics.keys()) & set(historical_metrics.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for key in common_keys:
            current_val = current_metrics[key]
            historical_val = historical_metrics[key]
            
            # Normalize values to 0-1 scale for comparison
            if max(current_val, historical_val) > 0:
                similarity = 1.0 - abs(current_val - historical_val) / max(current_val, historical_val)
            else:
                similarity = 1.0
            
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _update_failure_patterns(
        self,
        component_name: str,
        current_metrics: Dict[str, float],
        predictions: List[FailurePrediction]
    ):
        """Update learned failure patterns."""
        if component_name not in self.failure_patterns:
            self.failure_patterns[component_name] = []
        
        # Only store patterns that lead to actual predictions
        if predictions:
            pattern = {
                'timestamp': time.time(),
                'metrics': current_metrics.copy(),
                'predictions': len(predictions),
                'max_probability': max(p.probability for p in predictions)
            }
            
            self.failure_patterns[component_name].append(pattern)
            
            # Keep history bounded
            if len(self.failure_patterns[component_name]) > 100:
                self.failure_patterns[component_name] = self.failure_patterns[component_name][-50:]


class PredictiveHealthMonitor(HealthMonitor):
    """
    Enhanced health monitor with predictive failure detection capabilities.
    
    Features:
    - Real-time metrics collection and analysis
    - Machine learning-based anomaly detection
    - Predictive failure analysis
    - Automated alerting and escalation
    - Trend analysis and capacity planning
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        unhealthy_threshold: int = 3,
        critical_threshold: int = 5,
        enable_auto_recovery: bool = True,
        enable_predictions: bool = True,
        prediction_horizon: int = 300  # 5 minutes
    ):
        super().__init__(check_interval, unhealthy_threshold, critical_threshold, enable_auto_recovery)
        
        self.enable_predictions = enable_predictions
        self.time_series_analyzer = TimeSeriesAnalyzer(
            window_size=200,
            prediction_horizon=prediction_horizon
        )
        self.failure_predictor = FailurePredictor()
        
        # Predictive monitoring state
        self.active_predictions: List[FailurePrediction] = []
        self.active_alerts: List[HealthAlert] = []
        self.prediction_history: List[FailurePrediction] = []
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        
        # Performance metrics
        self.prediction_accuracy = 0.0
        self.false_positive_rate = 0.0
        self.alert_response_times: List[float] = []
        
        logging.info("Predictive Health Monitor initialized")
    
    @robust(component="predictive_health_monitor", operation="analyze_metrics")
    async def analyze_and_predict(self, component_name: str) -> Dict[str, Any]:
        """Perform comprehensive analysis and prediction for a component."""
        if component_name not in self.components:
            return {"error": f"Component {component_name} not found"}
        
        component = self.components[component_name]
        current_time = time.time()
        
        # Get current metrics
        current_metrics = self._extract_component_metrics(component)
        
        # Update time series
        for metric_name, value in current_metrics.items():
            self.time_series_analyzer.add_metric(
                f"{component_name}_{metric_name}",
                value,
                current_time
            )
        
        # Analyze trends
        metric_trends = {}
        for metric_name in current_metrics.keys():
            trend = self.time_series_analyzer.analyze_trend(f"{component_name}_{metric_name}")
            if trend:
                metric_trends[metric_name] = trend
        
        # Generate predictions
        predictions = []
        if self.enable_predictions:
            predictions = self.failure_predictor.predict_failures(
                component_name,
                current_metrics,
                metric_trends
            )
        
        # Update active predictions
        self._update_active_predictions(predictions)
        
        # Generate alerts
        alerts = self._generate_alerts(component_name, current_metrics, predictions)
        self._process_alerts(alerts)
        
        # Calculate health score with predictive factors
        health_score = self._calculate_predictive_health_score(
            component,
            current_metrics,
            metric_trends,
            predictions
        )
        
        return {
            "component": component_name,
            "timestamp": current_time,
            "current_metrics": current_metrics,
            "metric_trends": {k: v.__dict__ for k, v in metric_trends.items()},
            "predictions": [p.__dict__ for p in predictions],
            "health_score": health_score,
            "active_alerts": len([a for a in self.active_alerts if not a.resolved]),
            "prediction_count": len(self.active_predictions)
        }
    
    def _extract_component_metrics(self, component: ComponentHealth) -> Dict[str, float]:
        """Extract numerical metrics from component health."""
        metrics = {}
        
        # Standard health metrics
        metrics['success_rate'] = component.success_rate
        metrics['check_count'] = float(component.check_count)
        metrics['failure_count'] = float(component.failure_count)
        
        # Extract metrics from details
        if isinstance(component.details, dict):
            for key, value in component.details.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        
        return metrics
    
    def _calculate_predictive_health_score(
        self,
        component: ComponentHealth,
        current_metrics: Dict[str, float],
        metric_trends: Dict[str, MetricTrend],
        predictions: List[FailurePrediction]
    ) -> float:
        """Calculate health score incorporating predictive factors."""
        base_score = component.success_rate
        
        # Adjust for current metric status
        metric_adjustment = 0.0
        for metric_name, value in current_metrics.items():
            if metric_name in self.failure_predictor.failure_thresholds:
                thresholds = self.failure_predictor.failure_thresholds[metric_name]
                
                if value >= thresholds['critical']:
                    metric_adjustment -= 0.3
                elif value >= thresholds['warning']:
                    metric_adjustment -= 0.1
        
        # Adjust for trend analysis
        trend_adjustment = 0.0
        for trend in metric_trends.values():
            if trend.anomaly_score > 0.5:
                trend_adjustment -= trend.anomaly_score * 0.2
            
            if trend.trend_direction == "increasing" and trend.trend_strength > 0.7:
                # Check if this is a problematic metric
                if any(threshold_metric in trend.metric_name for threshold_metric in ['cpu', 'memory', 'error']):
                    trend_adjustment -= trend.trend_strength * 0.1
        
        # Adjust for predictions
        prediction_adjustment = 0.0
        for prediction in predictions:
            if prediction.probability > 0.7:
                prediction_adjustment -= prediction.probability * 0.2
        
        # Calculate final score
        final_score = base_score + metric_adjustment + trend_adjustment + prediction_adjustment
        return max(0.0, min(1.0, final_score))
    
    def _update_active_predictions(self, new_predictions: List[FailurePrediction]):
        """Update list of active predictions."""
        current_time = time.time()
        
        # Remove expired predictions
        self.active_predictions = [
            p for p in self.active_predictions
            if current_time - p.created_at < p.time_to_failure
        ]
        
        # Add new predictions
        self.active_predictions.extend(new_predictions)
        
        # Remove duplicates (same component and failure type)
        unique_predictions = {}
        for prediction in self.active_predictions:
            key = f"{prediction.component_name}_{prediction.failure_type}"
            if key not in unique_predictions or prediction.probability > unique_predictions[key].probability:
                unique_predictions[key] = prediction
        
        self.active_predictions = list(unique_predictions.values())
    
    def _generate_alerts(
        self,
        component_name: str,
        current_metrics: Dict[str, float],
        predictions: List[FailurePrediction]
    ) -> List[HealthAlert]:
        """Generate alerts based on metrics and predictions."""
        alerts = []
        
        # Metric-based alerts
        for metric_name, value in current_metrics.items():
            if metric_name in self.failure_predictor.failure_thresholds:
                thresholds = self.failure_predictor.failure_thresholds[metric_name]
                
                if value >= thresholds['critical']:
                    alert = HealthAlert(
                        alert_id=f"metric_critical_{component_name}_{metric_name}_{int(time.time())}",
                        component=component_name,
                        severity=AlertSeverity.CRITICAL,
                        message=f"{metric_name} critical: {value} >= {thresholds['critical']}",
                        metrics={metric_name: value}
                    )
                    alerts.append(alert)
                elif value >= thresholds['warning']:
                    alert = HealthAlert(
                        alert_id=f"metric_warning_{component_name}_{metric_name}_{int(time.time())}",
                        component=component_name,
                        severity=AlertSeverity.WARNING,
                        message=f"{metric_name} warning: {value} >= {thresholds['warning']}",
                        metrics={metric_name: value}
                    )
                    alerts.append(alert)
        
        # Prediction-based alerts
        for prediction in predictions:
            if prediction.probability > 0.8:
                severity = AlertSeverity.CRITICAL
            elif prediction.probability > 0.6:
                severity = AlertSeverity.ERROR
            else:
                severity = AlertSeverity.WARNING
            
            alert = HealthAlert(
                alert_id=f"prediction_{prediction.prediction_id}",
                component=component_name,
                severity=severity,
                message=f"Predicted failure: {prediction.failure_type} (prob: {prediction.probability:.2f})",
                metrics={"prediction_probability": prediction.probability, "time_to_failure": prediction.time_to_failure}
            )
            alerts.append(alert)
        
        return alerts
    
    def _process_alerts(self, alerts: List[HealthAlert]):
        """Process and manage alerts."""
        for alert in alerts:
            # Check if similar alert already exists
            existing_alert = self._find_similar_alert(alert)
            
            if not existing_alert:
                self.active_alerts.append(alert)
                
                # Trigger alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logging.error(f"Alert callback failed: {e}")
                
                logging.warning(f"Health Alert: {alert.severity.value} - {alert.message}")
        
        # Clean up resolved alerts
        self._cleanup_resolved_alerts()
    
    def _find_similar_alert(self, new_alert: HealthAlert) -> Optional[HealthAlert]:
        """Find similar existing alert."""
        for existing_alert in self.active_alerts:
            if (existing_alert.component == new_alert.component and
                existing_alert.severity == new_alert.severity and
                not existing_alert.resolved and
                abs(existing_alert.timestamp - new_alert.timestamp) < 300):  # Within 5 minutes
                return existing_alert
        return None
    
    def _cleanup_resolved_alerts(self):
        """Clean up old and resolved alerts."""
        current_time = time.time()
        
        # Remove alerts older than 24 hours
        self.active_alerts = [
            alert for alert in self.active_alerts
            if current_time - alert.timestamp < 86400  # 24 hours
        ]
    
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add alert notification callback."""
        self.alert_callbacks.append(callback)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logging.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logging.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_predictive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive predictive monitoring metrics."""
        current_time = time.time()
        
        # Time series analyzer metrics
        ts_metrics = self.time_series_analyzer.get_metrics_summary()
        
        # Active alerts by severity
        alert_breakdown = {}
        for severity in AlertSeverity:
            alert_breakdown[severity.value] = len([
                a for a in self.active_alerts
                if a.severity == severity and not a.resolved
            ])
        
        # Prediction metrics
        high_prob_predictions = [
            p for p in self.active_predictions
            if p.probability > 0.7
        ]
        
        # Component health distribution
        health_distribution = {}
        for component in self.components.values():
            status = component.status.value
            health_distribution[status] = health_distribution.get(status, 0) + 1
        
        return {
            "timestamp": current_time,
            "predictive_monitoring": {
                "enabled": self.enable_predictions,
                "active_predictions": len(self.active_predictions),
                "high_probability_predictions": len(high_prob_predictions),
                "prediction_accuracy": self.prediction_accuracy,
                "false_positive_rate": self.false_positive_rate
            },
            "alerting": {
                "active_alerts": len([a for a in self.active_alerts if not a.resolved]),
                "alert_breakdown": alert_breakdown,
                "acknowledged_alerts": len([a for a in self.active_alerts if a.acknowledged]),
                "average_response_time": np.mean(self.alert_response_times) if self.alert_response_times else 0.0
            },
            "time_series_analysis": ts_metrics,
            "component_health": health_distribution,
            "monitoring_performance": {
                "check_interval": self.check_interval,
                "total_components": len(self.components),
                "monitoring_active": self.is_monitoring
            }
        }


# Global predictive health monitor instance
predictive_monitor = PredictiveHealthMonitor()


# Convenience functions
async def analyze_component_health(component_name: str) -> Dict[str, Any]:
    """Analyze component health with predictions."""
    return await predictive_monitor.analyze_and_predict(component_name)


def get_active_predictions() -> List[Dict[str, Any]]:
    """Get all active failure predictions."""
    return [p.__dict__ for p in predictive_monitor.active_predictions]


def get_active_alerts() -> List[Dict[str, Any]]:
    """Get all active alerts."""
    return [a.__dict__ for a in predictive_monitor.active_alerts if not a.resolved]


def get_predictive_health_status() -> Dict[str, Any]:
    """Get comprehensive predictive health status."""
    return predictive_monitor.get_predictive_metrics()