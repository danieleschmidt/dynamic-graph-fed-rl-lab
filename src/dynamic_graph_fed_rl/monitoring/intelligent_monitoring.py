import os
"""
Intelligent Monitoring and Alerting System for Terragon SDLC
Comprehensive monitoring with predictive analytics and autonomous response capabilities
"""

import asyncio
import json
import logging
import smtplib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import psutil
import requests
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server
import redis
import sqlite3


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics to monitor"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"


class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    LOG = "log"


@dataclass
class MetricDefinition:
    """Definition of a metric to monitor"""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    collection_interval: int = 60
    retention_days: int = 30
    tags: Dict[str, str] = field(default_factory=dict)
    aggregation_functions: List[str] = field(default_factory=lambda: ["avg", "max", "min"])


@dataclass
class AlertRule:
    """Definition of an alert rule"""
    name: str
    metric_name: str
    condition: str
    threshold: Union[float, int]
    severity: AlertSeverity
    channels: List[AlertChannel]
    description: str
    enabled: bool = True
    cooldown_minutes: int = 15
    auto_resolve: bool = True
    escalation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance"""
    id: str
    rule_name: str
    metric_name: str
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    acknowledgements: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricData:
    """Metric data point"""
    name: str
    value: Union[float, int]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """Collects metrics from various sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.collectors: Dict[str, Callable] = {}
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.prometheus_metrics = {
            "system_cpu_usage": Gauge('system_cpu_usage_percent', 'System CPU usage percentage', registry=self.registry),
            "system_memory_usage": Gauge('system_memory_usage_percent', 'System memory usage percentage', registry=self.registry),
            "system_disk_usage": Gauge('system_disk_usage_percent', 'System disk usage percentage', registry=self.registry),
            "application_requests_total": Counter('application_requests_total', 'Total application requests', ['method', 'endpoint'], registry=self.registry),
            "application_request_duration": Histogram('application_request_duration_seconds', 'Application request duration', ['method', 'endpoint'], registry=self.registry),
            "federated_learning_accuracy": Gauge('federated_learning_accuracy', 'Federated learning model accuracy', registry=self.registry),
            "quantum_circuit_fidelity": Gauge('quantum_circuit_fidelity', 'Quantum circuit fidelity', registry=self.registry),
            "graph_node_count": Gauge('graph_node_count', 'Number of nodes in dynamic graph', registry=self.registry),
            "deployment_success_rate": Gauge('deployment_success_rate', 'Deployment success rate', registry=self.registry)
        }
    
    async def collect_system_metrics(self) -> List[MetricData]:
        """Collect system-level metrics"""
        metrics = []
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics.extend([
                MetricData("cpu_usage_percent", cpu_percent, datetime.now(), {"type": "system"}),
                MetricData("memory_usage_percent", memory.percent, datetime.now(), {"type": "system"}),
                MetricData("disk_usage_percent", disk.percent, datetime.now(), {"type": "system"}),
                MetricData("memory_available_bytes", memory.available, datetime.now(), {"type": "system"}),
                MetricData("disk_free_bytes", disk.free, datetime.now(), {"type": "system"})
            ])
            
            self.prometheus_metrics["system_cpu_usage"].set(cpu_percent)
            self.prometheus_metrics["system_memory_usage"].set(memory.percent)
            self.prometheus_metrics["system_disk_usage"].set(disk.percent)
            
            network = psutil.net_io_counters()
            metrics.extend([
                MetricData("network_bytes_sent", network.bytes_sent, datetime.now(), {"type": "network"}),
                MetricData("network_bytes_recv", network.bytes_recv, datetime.now(), {"type": "network"})
            ])
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
        
        return metrics
    
    async def collect_application_metrics(self) -> List[MetricData]:
        """Collect application-specific metrics"""
        metrics = []
        
        try:
            processes = [p for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']) 
                        if 'python' in p.info['name']]
            
            total_app_cpu = sum(p.info['cpu_percent'] for p in processes)
            total_app_memory = sum(p.info['memory_percent'] for p in processes)
            
            metrics.extend([
                MetricData("application_cpu_percent", total_app_cpu, datetime.now(), {"type": "application"}),
                MetricData("application_memory_percent", total_app_memory, datetime.now(), {"type": "application"}),
                MetricData("application_process_count", len(processes), datetime.now(), {"type": "application"})
            ])
            
            try:
                health_response = requests.get("http://localhost:8080/health", timeout=5)
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    metrics.append(MetricData(
                        "application_health_status", 
                        1 if health_data.get("status") == "healthy" else 0,
                        datetime.now(),
                        {"type": "health"}
                    ))
            except:
                metrics.append(MetricData("application_health_status", 0, datetime.now(), {"type": "health"}))
            
        except Exception as e:
            self.logger.error(f"Application metrics collection failed: {e}")
        
        return metrics
    
    async def collect_ml_metrics(self) -> List[MetricData]:
        """Collect machine learning specific metrics"""
        metrics = []
        
        try:
            model_metrics_file = Path("/tmp/terragon_model_metrics.json")
            if model_metrics_file.exists():
                with open(model_metrics_file, 'r') as f:
                    ml_data = json.load(f)
                
                if "federated_accuracy" in ml_data:
                    accuracy = ml_data["federated_accuracy"]
                    metrics.append(MetricData(
                        "federated_learning_accuracy", 
                        accuracy,
                        datetime.now(),
                        {"type": "ml", "model": "federated"}
                    ))
                    self.prometheus_metrics["federated_learning_accuracy"].set(accuracy)
                
                if "quantum_fidelity" in ml_data:
                    fidelity = ml_data["quantum_fidelity"]
                    metrics.append(MetricData(
                        "quantum_circuit_fidelity",
                        fidelity,
                        datetime.now(),
                        {"type": "quantum"}
                    ))
                    self.prometheus_metrics["quantum_circuit_fidelity"].set(fidelity)
                
                if "graph_nodes" in ml_data:
                    node_count = ml_data["graph_nodes"]
                    metrics.append(MetricData(
                        "dynamic_graph_node_count",
                        node_count,
                        datetime.now(),
                        {"type": "graph"}
                    ))
                    self.prometheus_metrics["graph_node_count"].set(node_count)
            
        except Exception as e:
            self.logger.error(f"ML metrics collection failed: {e}")
        
        return metrics
    
    def register_custom_collector(self, name: str, collector_func: Callable):
        """Register custom metric collector"""
        self.collectors[name] = collector_func
        self.logger.info(f"Registered custom collector: {name}")
    
    async def collect_all_metrics(self) -> List[MetricData]:
        """Collect all metrics from all sources"""
        all_metrics = []
        
        collection_tasks = [
            self.collect_system_metrics(),
            self.collect_application_metrics(),
            self.collect_ml_metrics()
        ]
        
        for name, collector in self.collectors.items():
            collection_tasks.append(collector())
        
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Metric collection failed: {result}")
            else:
                all_metrics.extend(result)
        
        return all_metrics


class TimeSeriesDatabase:
    """Time series database for storing metrics"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics (name, timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    triggered_at DATETIME NOT NULL,
                    resolved_at DATETIME,
                    escalated BOOLEAN DEFAULT FALSE,
                    context TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def store_metrics(self, metrics: List[MetricData]):
        """Store metrics in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for metric in metrics:
                    conn.execute("""
                        INSERT INTO metrics (name, value, timestamp, tags, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        metric.name,
                        metric.value,
                        metric.timestamp.isoformat(),
                        json.dumps(metric.tags),
                        json.dumps(metric.metadata)
                    ))
        
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")
    
    async def query_metrics(self, 
                          metric_name: str, 
                          start_time: datetime, 
                          end_time: datetime,
                          aggregation: str = "avg") -> List[Tuple[datetime, float]]:
        """Query metrics from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if aggregation == "avg":
                    query = """
                        SELECT datetime(timestamp), AVG(value)
                        FROM metrics 
                        WHERE name = ? AND timestamp BETWEEN ? AND ?
                        GROUP BY datetime(timestamp)
                        ORDER BY timestamp
                    """
                elif aggregation == "max":
                    query = """
                        SELECT datetime(timestamp), MAX(value)
                        FROM metrics 
                        WHERE name = ? AND timestamp BETWEEN ? AND ?
                        GROUP BY datetime(timestamp)
                        ORDER BY timestamp
                    """
                elif aggregation == "min":
                    query = """
                        SELECT datetime(timestamp), MIN(value)
                        FROM metrics 
                        WHERE name = ? AND timestamp BETWEEN ? AND ?
                        GROUP BY datetime(timestamp)
                        ORDER BY timestamp
                    """
                else:
                    query = """
                        SELECT timestamp, value
                        FROM metrics 
                        WHERE name = ? AND timestamp BETWEEN ? AND ?
                        ORDER BY timestamp
                    """
                
                cursor = conn.execute(query, (
                    metric_name,
                    start_time.isoformat(),
                    end_time.isoformat()
                ))
                
                return [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]
        
        except Exception as e:
            self.logger.error(f"Failed to query metrics: {e}")
            return []
    
    async def store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts 
                    (id, rule_name, metric_name, severity, message, triggered_at, resolved_at, escalated, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id,
                    alert.rule_name,
                    alert.metric_name,
                    alert.severity.value,
                    alert.message,
                    alert.triggered_at.isoformat(),
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.escalated,
                    json.dumps(alert.context)
                ))
        
        except Exception as e:
            self.logger.error(f"Failed to store alert: {e}")


class AnomalyDetector:
    """Detects anomalies in metric data using statistical methods"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.baseline_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.logger = logging.getLogger(__name__)
    
    async def detect_anomalies(self, metric_data: MetricData) -> Dict[str, Any]:
        """Detect anomalies in metric data"""
        try:
            metric_name = metric_data.name
            value = metric_data.value
            
            baseline = self.baseline_windows[metric_name]
            baseline.append(value)
            
            if len(baseline) < 10:
                return {"is_anomaly": False, "confidence": 0.0, "reason": "insufficient_data"}
            
            baseline_array = np.array(list(baseline)[:-1])
            mean = np.mean(baseline_array)
            std = np.std(baseline_array)
            
            if std == 0:
                return {"is_anomaly": False, "confidence": 0.0, "reason": "no_variance"}
            
            z_score = abs((value - mean) / std)
            is_anomaly = z_score > self.sensitivity
            confidence = min(z_score / self.sensitivity, 1.0)
            
            seasonal_anomaly = await self._detect_seasonal_anomaly(metric_name, value)
            trend_anomaly = await self._detect_trend_anomaly(metric_name, baseline_array)
            
            return {
                "is_anomaly": is_anomaly or seasonal_anomaly["is_anomaly"] or trend_anomaly["is_anomaly"],
                "confidence": max(confidence, seasonal_anomaly["confidence"], trend_anomaly["confidence"]),
                "z_score": z_score,
                "seasonal_anomaly": seasonal_anomaly,
                "trend_anomaly": trend_anomaly,
                "baseline_mean": mean,
                "baseline_std": std
            }
        
        except Exception as e:
            self.logger.error(f"Anomaly detection failed for {metric_data.name}: {e}")
            return {"is_anomaly": False, "confidence": 0.0, "error": str(e)}
    
    async def _detect_seasonal_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Detect seasonal anomalies based on time patterns"""
        try:
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()
            
            seasonal_key = f"{metric_name}_h{current_hour}_d{current_day}"
            seasonal_baseline = self.baseline_windows.get(seasonal_key, deque(maxlen=100))
            
            if len(seasonal_baseline) < 5:
                return {"is_anomaly": False, "confidence": 0.0}
            
            seasonal_array = np.array(list(seasonal_baseline))
            seasonal_mean = np.mean(seasonal_array)
            seasonal_std = np.std(seasonal_array)
            
            if seasonal_std == 0:
                return {"is_anomaly": False, "confidence": 0.0}
            
            seasonal_z_score = abs((value - seasonal_mean) / seasonal_std)
            is_seasonal_anomaly = seasonal_z_score > self.sensitivity * 1.5
            confidence = min(seasonal_z_score / (self.sensitivity * 1.5), 1.0)
            
            return {
                "is_anomaly": is_seasonal_anomaly,
                "confidence": confidence,
                "seasonal_z_score": seasonal_z_score
            }
        
        except Exception as e:
            self.logger.error(f"Seasonal anomaly detection failed: {e}")
            return {"is_anomaly": False, "confidence": 0.0}
    
    async def _detect_trend_anomaly(self, metric_name: str, baseline_array: np.ndarray) -> Dict[str, Any]:
        """Detect trend-based anomalies"""
        try:
            if len(baseline_array) < 20:
                return {"is_anomaly": False, "confidence": 0.0}
            
            recent_window = baseline_array[-10:]
            historical_window = baseline_array[-30:-10]
            
            if len(historical_window) == 0:
                return {"is_anomaly": False, "confidence": 0.0}
            
            recent_trend = np.polyfit(range(len(recent_window)), recent_window, 1)[0]
            historical_trend = np.polyfit(range(len(historical_window)), historical_window, 1)[0]
            
            trend_change = abs(recent_trend - historical_trend)
            historical_trend_std = np.std(np.diff(historical_window))
            
            if historical_trend_std == 0:
                return {"is_anomaly": False, "confidence": 0.0}
            
            trend_z_score = trend_change / historical_trend_std
            is_trend_anomaly = trend_z_score > self.sensitivity
            confidence = min(trend_z_score / self.sensitivity, 1.0)
            
            return {
                "is_anomaly": is_trend_anomaly,
                "confidence": confidence,
                "trend_change": trend_change,
                "recent_trend": recent_trend,
                "historical_trend": historical_trend
            }
        
        except Exception as e:
            self.logger.error(f"Trend anomaly detection failed: {e}")
            return {"is_anomaly": False, "confidence": 0.0}


class AlertManager:
    """Manages alert rules, firing, and notifications"""
    
    def __init__(self, db: TimeSeriesDatabase):
        self.db = db
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.notification_channels: Dict[AlertChannel, Callable] = {}
        self.logger = logging.getLogger(__name__)
        self._setup_notification_channels()
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        self.notification_channels = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.LOG: self._log_alert
        }
    
    def add_alert_rule(self, alert_rule: AlertRule):
        """Add new alert rule"""
        self.alert_rules[alert_rule.name] = alert_rule
        self.logger.info(f"Added alert rule: {alert_rule.name}")
    
    async def evaluate_rules(self, metrics: List[MetricData], anomalies: Dict[str, Dict[str, Any]]):
        """Evaluate all alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            if rule_name in self.alert_cooldowns:
                if datetime.now() < self.alert_cooldowns[rule_name]:
                    continue
            
            try:
                await self._evaluate_single_rule(rule, metrics, anomalies)
            except Exception as e:
                self.logger.error(f"Failed to evaluate rule {rule_name}: {e}")
    
    async def _evaluate_single_rule(self, 
                                  rule: AlertRule, 
                                  metrics: List[MetricData], 
                                  anomalies: Dict[str, Dict[str, Any]]):
        """Evaluate a single alert rule"""
        relevant_metrics = [m for m in metrics if m.name == rule.metric_name]
        
        if not relevant_metrics:
            return
        
        latest_metric = max(relevant_metrics, key=lambda m: m.timestamp)
        
        triggered = False
        context = {}
        
        if rule.condition == "greater_than":
            triggered = latest_metric.value > rule.threshold
        elif rule.condition == "less_than":
            triggered = latest_metric.value < rule.threshold
        elif rule.condition == "equals":
            triggered = latest_metric.value == rule.threshold
        elif rule.condition == "anomaly":
            anomaly_info = anomalies.get(rule.metric_name, {})
            triggered = anomaly_info.get("is_anomaly", False)
            context = anomaly_info
        elif rule.condition == "rate_increase":
            triggered = await self._evaluate_rate_condition(rule, "increase")
        elif rule.condition == "rate_decrease":
            triggered = await self._evaluate_rate_condition(rule, "decrease")
        
        if triggered:
            await self._fire_alert(rule, latest_metric, context)
        else:
            await self._check_auto_resolve(rule, latest_metric)
    
    async def _evaluate_rate_condition(self, rule: AlertRule, direction: str) -> bool:
        """Evaluate rate-based conditions"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=10)
            
            metric_data = await self.db.query_metrics(rule.metric_name, start_time, end_time)
            
            if len(metric_data) < 2:
                return False
            
            values = [point[1] for point in metric_data]
            rate_of_change = (values[-1] - values[0]) / len(values)
            
            if direction == "increase":
                return rate_of_change > rule.threshold
            else:
                return rate_of_change < -rule.threshold
        
        except Exception as e:
            self.logger.error(f"Rate condition evaluation failed: {e}")
            return False
    
    async def _fire_alert(self, rule: AlertRule, metric: MetricData, context: Dict[str, Any]):
        """Fire an alert"""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        if alert_id in self.active_alerts:
            return
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            metric_name=rule.metric_name,
            severity=rule.severity,
            message=f"{rule.description} - Current value: {metric.value}, Threshold: {rule.threshold}",
            triggered_at=datetime.now(),
            context=context
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_cooldowns[rule.name] = datetime.now() + timedelta(minutes=rule.cooldown_minutes)
        
        await self.db.store_alert(alert)
        
        for channel in rule.channels:
            try:
                await self.notification_channels[channel](alert, rule)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel.value}: {e}")
        
        self.logger.warning(f"Alert fired: {alert.message}")
        
        if rule.escalation_rules and rule.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            await self._handle_escalation(alert, rule)
    
    async def _check_auto_resolve(self, rule: AlertRule, metric: MetricData):
        """Check if alerts should be auto-resolved"""
        if not rule.auto_resolve:
            return
        
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.rule_name == rule.name and not alert.resolved_at:
                resolve_condition = False
                
                if rule.condition == "greater_than":
                    resolve_condition = metric.value <= rule.threshold * 0.9
                elif rule.condition == "less_than":
                    resolve_condition = metric.value >= rule.threshold * 1.1
                elif rule.condition == "anomaly":
                    resolve_condition = True
                
                if resolve_condition:
                    alerts_to_resolve.append(alert)
        
        for alert in alerts_to_resolve:
            await self._resolve_alert(alert)
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an active alert"""
        alert.resolved_at = datetime.now()
        await self.db.store_alert(alert)
        
        if alert.id in self.active_alerts:
            del self.active_alerts[alert.id]
        
        self.logger.info(f"Alert resolved: {alert.message}")
    
    async def _handle_escalation(self, alert: Alert, rule: AlertRule):
        """Handle alert escalation"""
        try:
            escalation_delay = rule.escalation_rules.get("delay_minutes", 30)
            
            await asyncio.sleep(escalation_delay * 60)
            
            if alert.id in self.active_alerts and not alert.resolved_at:
                alert.escalated = True
                escalation_channels = rule.escalation_rules.get("channels", [AlertChannel.EMAIL])
                
                for channel in escalation_channels:
                    if channel in self.notification_channels:
                        await self.notification_channels[channel](alert, rule)
                
                self.logger.error(f"Alert escalated: {alert.message}")
        
        except Exception as e:
            self.logger.error(f"Alert escalation failed: {e}")
    
    async def _send_email_alert(self, alert: Alert, rule: AlertRule):
        """Send email alert notification"""
        try:
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "alerts@terragon.ai"
            sender_password = os.getenv("PASSWORD", "default_secure_value")
            recipient_email = "team@terragon.ai"
            
            subject = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            body = f"""
Alert: {alert.message}
Severity: {alert.severity.value}
Triggered: {alert.triggered_at}
Metric: {alert.metric_name}
Rule: {alert.rule_name}

Context: {json.dumps(alert.context, indent=2)}
            """
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = recipient_email
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent for {alert.id}")
        
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
    
    async def _send_slack_alert(self, alert: Alert, rule: AlertRule):
        """Send Slack alert notification"""
        try:
            webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
            
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "text": f"Alert: {alert.rule_name}",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "warning"),
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value, "short": True},
                            {"title": "Metric", "value": alert.metric_name, "short": True},
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Triggered", "value": alert.triggered_at.isoformat(), "short": True}
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent for {alert.id}")
        
        except Exception as e:
            self.logger.error(f"Slack alert failed: {e}")
    
    async def _send_webhook_alert(self, alert: Alert, rule: AlertRule):
        """Send webhook alert notification"""
        try:
            webhook_url = rule.escalation_rules.get("webhook_url", "http://localhost:8080/alerts")
            
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "metric_name": alert.metric_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "triggered_at": alert.triggered_at.isoformat(),
                "context": alert.context
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Webhook alert sent for {alert.id}")
        
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")
    
    async def _log_alert(self, alert: Alert, rule: AlertRule):
        """Log alert to system logs"""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        self.logger.log(log_level, f"ALERT: {alert.message} (Rule: {alert.rule_name})")


class AutonomousResponseEngine:
    """Autonomous response engine for handling alerts"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.response_actions: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        self._setup_default_actions()
    
    def _setup_default_actions(self):
        """Setup default autonomous response actions"""
        self.response_actions = {
            "restart_service": self._restart_service,
            "scale_up": self._scale_up_service,
            "scale_down": self._scale_down_service,
            "clear_cache": self._clear_cache,
            "optimize_memory": self._optimize_memory,
            "rollback_deployment": self._rollback_deployment,
            "trigger_backup": self._trigger_backup
        }
    
    async def handle_alert(self, alert: Alert, rule: AlertRule) -> Dict[str, Any]:
        """Handle alert with autonomous response"""
        try:
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                response_config = rule.escalation_rules.get("autonomous_response", {})
                
                if response_config.get("enabled", False):
                    actions = response_config.get("actions", [])
                    
                    response_results = {}
                    for action in actions:
                        if action in self.response_actions:
                            try:
                                result = await self.response_actions[action](alert, rule)
                                response_results[action] = result
                                self.logger.info(f"Autonomous action {action} executed for alert {alert.id}")
                            except Exception as e:
                                response_results[action] = {"success": False, "error": str(e)}
                                self.logger.error(f"Autonomous action {action} failed: {e}")
                    
                    return {"autonomous_response": True, "actions": response_results}
            
            return {"autonomous_response": False, "reason": "not_configured"}
        
        except Exception as e:
            self.logger.error(f"Autonomous response failed for alert {alert.id}: {e}")
            return {"autonomous_response": False, "error": str(e)}
    
    async def _restart_service(self, alert: Alert, rule: AlertRule) -> Dict[str, Any]:
        """Restart the service"""
        try:
            process = subprocess.run(
                ["kubectl", "rollout", "restart", "deployment/terragon-app"],
                capture_output=True,
                text=True
            )
            
            return {"success": process.returncode == 0, "output": process.stdout}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _scale_up_service(self, alert: Alert, rule: AlertRule) -> Dict[str, Any]:
        """Scale up the service"""
        try:
            process = subprocess.run(
                ["kubectl", "scale", "deployment/terragon-app", "--replicas=5"],
                capture_output=True,
                text=True
            )
            
            return {"success": process.returncode == 0, "output": process.stdout}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _scale_down_service(self, alert: Alert, rule: AlertRule) -> Dict[str, Any]:
        """Scale down the service"""
        try:
            process = subprocess.run(
                ["kubectl", "scale", "deployment/terragon-app", "--replicas=2"],
                capture_output=True,
                text=True
            )
            
            return {"success": process.returncode == 0, "output": process.stdout}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _clear_cache(self, alert: Alert, rule: AlertRule) -> Dict[str, Any]:
        """Clear application cache"""
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.flushall()
            
            return {"success": True, "message": "Cache cleared successfully"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _optimize_memory(self, alert: Alert, rule: AlertRule) -> Dict[str, Any]:
        """Optimize memory usage"""
        try:
            import gc
            collected = gc.collect()
            
            return {"success": True, "objects_collected": collected}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _rollback_deployment(self, alert: Alert, rule: AlertRule) -> Dict[str, Any]:
        """Rollback current deployment"""
        try:
            process = subprocess.run(
                ["kubectl", "rollout", "undo", "deployment/terragon-app"],
                capture_output=True,
                text=True
            )
            
            return {"success": process.returncode == 0, "output": process.stdout}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _trigger_backup(self, alert: Alert, rule: AlertRule) -> Dict[str, Any]:
        """Trigger backup operation"""
        try:
            backup_script = self.project_path / "scripts" / "backup.sh"
            if backup_script.exists():
                process = subprocess.run([str(backup_script)], capture_output=True, text=True)
                return {"success": process.returncode == 0, "output": process.stdout}
            else:
                return {"success": False, "error": "Backup script not found"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}


class IntelligentMonitoringOrchestrator:
    """Main orchestrator for intelligent monitoring system"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.metric_collector = MetricCollector()
        self.anomaly_detector = AnomalyDetector()
        self.db = TimeSeriesDatabase(project_path / "monitoring" / "metrics.db")
        self.alert_manager = AlertManager(self.db)
        self.response_engine = AutonomousResponseEngine(project_path)
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default monitoring rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                condition="greater_than",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                description="CPU usage is above 80%",
                escalation_rules={
                    "enabled": True,
                    "delay_minutes": 15,
                    "channels": [AlertChannel.SLACK],
                    "autonomous_response": {
                        "enabled": True,
                        "actions": ["scale_up", "optimize_memory"]
                    }
                }
            ),
            AlertRule(
                name="low_memory_available",
                metric_name="memory_usage_percent",
                condition="greater_than",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK],
                description="Memory usage is above 90%",
                escalation_rules={
                    "enabled": True,
                    "delay_minutes": 5,
                    "autonomous_response": {
                        "enabled": True,
                        "actions": ["optimize_memory", "restart_service"]
                    }
                }
            ),
            AlertRule(
                name="application_down",
                metric_name="application_health_status",
                condition="less_than",
                threshold=1,
                severity=AlertSeverity.EMERGENCY,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK],
                description="Application health check failed",
                escalation_rules={
                    "enabled": True,
                    "delay_minutes": 2,
                    "autonomous_response": {
                        "enabled": True,
                        "actions": ["restart_service", "rollback_deployment"]
                    }
                }
            ),
            AlertRule(
                name="federated_learning_accuracy_drop",
                metric_name="federated_learning_accuracy",
                condition="less_than",
                threshold=0.85,
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                description="Federated learning accuracy dropped below 85%"
            ),
            AlertRule(
                name="quantum_circuit_fidelity_low",
                metric_name="quantum_circuit_fidelity",
                condition="less_than",
                threshold=0.90,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                description="Quantum circuit fidelity is below 90%"
            ),
            AlertRule(
                name="anomaly_detection",
                metric_name="cpu_usage_percent",
                condition="anomaly",
                threshold=0.8,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                description="Anomalous behavior detected in CPU usage"
            ),
            AlertRule(
                name="disk_space_low",
                metric_name="disk_usage_percent",
                condition="greater_than",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                description="Disk space usage above 85%",
                escalation_rules={
                    "enabled": True,
                    "delay_minutes": 60,
                    "autonomous_response": {
                        "enabled": True,
                        "actions": ["clear_cache", "trigger_backup"]
                    }
                }
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    async def start_monitoring(self, collection_interval: int = 60):
        """Start continuous monitoring"""
        self.monitoring_active = True
        self.logger.info("Starting intelligent monitoring system")
        
        start_http_server(8000, registry=self.metric_collector.registry)
        self.logger.info("Prometheus metrics server started on port 8000")
        
        while self.monitoring_active:
            try:
                metrics = await self.metric_collector.collect_all_metrics()
                
                await self.db.store_metrics(metrics)
                
                anomalies = {}
                for metric in metrics:
                    anomaly_result = await self.anomaly_detector.detect_anomalies(metric)
                    if anomaly_result["is_anomaly"]:
                        anomalies[metric.name] = anomaly_result
                
                await self.alert_manager.evaluate_rules(metrics, anomalies)
                
                for alert in list(self.alert_manager.active_alerts.values()):
                    rule = self.alert_manager.alert_rules.get(alert.rule_name)
                    if rule:
                        await self.response_engine.handle_alert(alert, rule)
                
                await asyncio.sleep(collection_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(collection_interval)
    
    async def stop_monitoring(self):
        """Stop monitoring system"""
        self.monitoring_active = False
        self.logger.info("Monitoring system stopped")
    
    async def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            dashboard_data = {
                "system_metrics": {},
                "application_metrics": {},
                "ml_metrics": {},
                "active_alerts": len(self.alert_manager.active_alerts),
                "alert_history": [],
                "anomalies_detected": 0
            }
            
            key_metrics = [
                "cpu_usage_percent", "memory_usage_percent", "disk_usage_percent",
                "application_health_status", "federated_learning_accuracy", 
                "quantum_circuit_fidelity", "dynamic_graph_node_count"
            ]
            
            for metric_name in key_metrics:
                metric_data = await self.db.query_metrics(metric_name, start_time, end_time)
                
                if metric_data:
                    values = [point[1] for point in metric_data]
                    dashboard_data["system_metrics"][metric_name] = {
                        "current": values[-1] if values else 0,
                        "avg_24h": np.mean(values),
                        "max_24h": np.max(values),
                        "min_24h": np.min(values),
                        "trend": "up" if len(values) > 1 and values[-1] > values[0] else "down"
                    }
            
            dashboard_data["active_alerts"] = [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at.isoformat()
                }
                for alert in self.alert_manager.active_alerts.values()
            ]
            
            return dashboard_data
        
        except Exception as e:
            self.logger.error(f"Dashboard data generation failed: {e}")
            return {}
    
    async def generate_monitoring_report(self, hours: int = 24) -> str:
        """Generate comprehensive monitoring report"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            dashboard_data = await self.get_monitoring_dashboard_data()
            
            report = f"""
# Terragon Monitoring Report
Generated: {end_time.isoformat()}
Period: {hours} hours

## System Overview
- Active Alerts: {dashboard_data.get('active_alerts', 0)}
- System Status: {'Healthy' if dashboard_data.get('active_alerts', 0) == 0 else 'Issues Detected'}

## Key Metrics Summary
"""
            
            for metric_name, data in dashboard_data.get("system_metrics", {}).items():
                report += f"- {metric_name}: {data['current']:.2f} (24h avg: {data['avg_24h']:.2f})\n"
            
            report += "\n## Alert Summary\n"
            if dashboard_data.get("active_alerts"):
                for alert in dashboard_data["active_alerts"]:
                    report += f"- [{alert['severity'].upper()}] {alert['message']}\n"
            else:
                report += "- No active alerts\n"
            
            report += f"\n## ML System Health\n"
            ml_metrics = {k: v for k, v in dashboard_data.get("system_metrics", {}).items() 
                         if k in ["federated_learning_accuracy", "quantum_circuit_fidelity", "dynamic_graph_node_count"]}
            
            for metric_name, data in ml_metrics.items():
                report += f"- {metric_name}: {data['current']:.3f}\n"
            
            return report
        
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return f"Error generating report: {e}"
    
    def add_custom_metric_collector(self, name: str, collector_func: Callable):
        """Add custom metric collector"""
        self.metric_collector.register_custom_collector(name, collector_func)
    
    def add_custom_alert_rule(self, alert_rule: AlertRule):
        """Add custom alert rule"""
        self.alert_manager.add_alert_rule(alert_rule)
    
    def add_custom_response_action(self, name: str, action_func: Callable):
        """Add custom autonomous response action"""
        self.response_engine.response_actions[name] = action_func
        self.logger.info(f"Added custom response action: {name}")


async def main():
    """Main monitoring system entry point"""
    project_path = Path("/root/repo")
    monitoring = IntelligentMonitoringOrchestrator(project_path)
    
    try:
        await monitoring.start_monitoring(collection_interval=30)
    except KeyboardInterrupt:
        await monitoring.stop_monitoring()
        print("Monitoring system stopped")


if __name__ == "__main__":
    asyncio.run(main())