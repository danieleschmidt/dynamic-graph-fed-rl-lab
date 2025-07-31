"""Load testing configuration using Locust."""
from locust import HttpUser, task, between
import json
import random


class DynamicGraphFedRLUser(HttpUser):
    """Simulate users interacting with the Fed-RL system."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Initialize user session."""
        self.agent_id = f"agent_{random.randint(1000, 9999)}"
        self.register_agent()
    
    def register_agent(self):
        """Register agent with the system."""
        payload = {
            "agent_id": self.agent_id,
            "capabilities": ["graph_processing", "federated_learning"],
            "resources": {"cpu": 4, "memory": "8GB", "gpu": True}
        }
        
        response = self.client.post(
            "/api/v1/agents/register",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            self.token = response.json().get("token")
    
    @task(3)
    def submit_training_data(self):
        """Submit training data batch."""
        # Generate mock graph data
        graph_data = {
            "nodes": [[random.random() for _ in range(64)] for _ in range(100)],
            "edges": [[random.randint(0, 99), random.randint(0, 99)] for _ in range(200)],
            "rewards": [random.random() for _ in range(100)],
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        self.client.post(
            f"/api/v1/agents/{self.agent_id}/training_data",
            json=graph_data,
            headers={"Authorization": f"Bearer {getattr(self, 'token', '')}"}
        )
    
    @task(2)
    def request_parameters(self):
        """Request updated model parameters."""
        self.client.get(
            f"/api/v1/agents/{self.agent_id}/parameters",
            headers={"Authorization": f"Bearer {getattr(self, 'token', '')}"}
        )
    
    @task(1)
    def submit_metrics(self):
        """Submit performance metrics."""
        metrics = {
            "episode_reward": random.uniform(-100, 100),
            "steps_per_second": random.uniform(500, 2000),
            "memory_usage": random.uniform(1, 8),  # GB
            "convergence_metric": random.uniform(0, 1),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        self.client.post(
            f"/api/v1/agents/{self.agent_id}/metrics",
            json=metrics,
            headers={"Authorization": f"Bearer {getattr(self, 'token', '')}"}
        )
    
    @task(1)
    def health_check(self):
        """Perform health check."""
        self.client.get("/health")


class AdminUser(HttpUser):
    """Simulate admin users monitoring the system."""
    
    wait_time = between(5, 10)
    
    @task(2)
    def view_dashboard(self):
        """Access monitoring dashboard."""
        self.client.get("/api/v1/dashboard/overview")
    
    @task(1)
    def get_system_metrics(self):
        """Retrieve system-wide metrics."""
        self.client.get("/api/v1/metrics/system")
    
    @task(1)
    def get_agent_status(self):
        """Check status of all agents."""
        self.client.get("/api/v1/agents/status")


class ResearcherUser(HttpUser):
    """Simulate researchers running experiments."""
    
    wait_time = between(10, 30)
    
    @task(1)
    def start_experiment(self):
        """Start a new federated learning experiment."""
        experiment_config = {
            "name": f"experiment_{random.randint(1000, 9999)}",
            "algorithm": random.choice(["GraphTD3", "GraphSAC", "GraphA2C"]),
            "environment": random.choice(["traffic", "power_grid", "supply_chain"]),
            "num_agents": random.randint(5, 50),
            "max_episodes": random.randint(100, 1000),
            "federation_strategy": random.choice(["async_gossip", "sync_sgd"])
        }
        
        self.client.post(
            "/api/v1/experiments",
            json=experiment_config,
            headers={"Content-Type": "application/json"}
        )
    
    @task(2)
    def monitor_experiment(self):
        """Monitor running experiments."""
        self.client.get("/api/v1/experiments/active")
    
    @task(1)
    def download_results(self):
        """Download experiment results."""
        experiment_id = random.randint(1, 100)
        self.client.get(f"/api/v1/experiments/{experiment_id}/results")