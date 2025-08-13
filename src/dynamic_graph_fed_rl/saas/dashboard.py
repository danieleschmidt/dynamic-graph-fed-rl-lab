"""
Web-based Dashboard for Federated Learning Management

Provides a comprehensive web interface for managing federated learning
experiments, monitoring performance, and visualizing results.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import auth_service, Permission
from .tenant import tenant_manager
from .api import get_current_user, require_permission


class DashboardService:
    """
    Web dashboard service providing:
    - Real-time experiment monitoring
    - Performance visualization
    - Resource usage tracking  
    - Agent status monitoring
    - Model management interface
    """
    
    def __init__(self):
        self.app = FastAPI()
        self.templates = Jinja2Templates(directory="templates")
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup dashboard routes"""
        
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard_home(
            request: Request,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Main dashboard page"""
            tenant = await tenant_manager.get_tenant(user["tenant_id"])
            usage = await tenant_manager.get_usage(user["tenant_id"])
            
            context = {
                "request": request,
                "user": user,
                "tenant": tenant,
                "usage": usage,
                "title": "Federated Learning Dashboard"
            }
            
            return self.templates.TemplateResponse("dashboard.html", context)
            
        @self.app.get("/dashboard/experiments", response_class=HTMLResponse)
        async def experiments_page(
            request: Request,
            user: Dict[str, Any] = Depends(require_permission(Permission.READ_EXPERIMENT))
        ):
            """Experiments management page"""
            context = {
                "request": request,
                "user": user,
                "title": "Experiments"
            }
            
            return self.templates.TemplateResponse("experiments.html", context)
            
        @self.app.get("/dashboard/agents", response_class=HTMLResponse)
        async def agents_page(
            request: Request,
            user: Dict[str, Any] = Depends(require_permission(Permission.READ_AGENT))
        ):
            """Agents monitoring page"""
            context = {
                "request": request,
                "user": user,
                "title": "Agents"
            }
            
            return self.templates.TemplateResponse("agents.html", context)
            
        @self.app.get("/dashboard/models", response_class=HTMLResponse)
        async def models_page(
            request: Request,
            user: Dict[str, Any] = Depends(require_permission(Permission.READ_MODEL))
        ):
            """Models management page"""
            context = {
                "request": request,
                "user": user,
                "title": "Models"
            }
            
            return self.templates.TemplateResponse("models.html", context)
            
        @self.app.get("/dashboard/marketplace", response_class=HTMLResponse)
        async def marketplace_page(
            request: Request,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Algorithm marketplace page"""
            context = {
                "request": request,
                "user": user,
                "title": "Marketplace"
            }
            
            return self.templates.TemplateResponse("marketplace.html", context)
            
        @self.app.get("/dashboard/analytics", response_class=HTMLResponse)
        async def analytics_page(
            request: Request,
            user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_ANALYTICS))
        ):
            """Analytics and reporting page"""
            context = {
                "request": request,
                "user": user,
                "title": "Analytics"
            }
            
            return self.templates.TemplateResponse("analytics.html", context)
            
        # API endpoints for dashboard data
        @self.app.get("/api/dashboard/overview")
        async def get_dashboard_overview(
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Get dashboard overview data"""
            tenant = await tenant_manager.get_tenant(user["tenant_id"])
            usage = await tenant_manager.get_usage(user["tenant_id"])
            
            return {
                "tenant_info": {
                    "name": tenant.name,
                    "tier": tenant.tier.value,
                    "created_at": tenant.created_at.isoformat()
                },
                "quotas": {
                    "experiments": {
                        "used": usage.get("experiments", 0),
                        "limit": tenant.quotas.max_experiments
                    },
                    "agents": {
                        "used": usage.get("agents", 0),
                        "limit": tenant.quotas.max_agents
                    },
                    "models": {
                        "used": usage.get("models", 0),
                        "limit": tenant.quotas.max_models
                    },
                    "compute_hours": {
                        "used": usage.get("compute_hours", 0),
                        "limit": tenant.quotas.compute_hours_monthly
                    },
                    "storage": {
                        "used": usage.get("storage_used", 0),
                        "limit": tenant.quotas.storage_gb
                    }
                },
                "recent_activity": []  # Would fetch from activity log
            }
            
        @self.app.get("/api/dashboard/experiments/metrics")
        async def get_experiment_metrics(
            user: Dict[str, Any] = Depends(require_permission(Permission.READ_EXPERIMENT))
        ):
            """Get experiment performance metrics"""
            # Mock data - would integrate with actual FL framework
            return {
                "active_experiments": 3,
                "total_experiments": 25,
                "success_rate": 0.85,
                "average_convergence_time": 45.2,
                "recent_experiments": [
                    {
                        "id": "exp_001",
                        "name": "Traffic Optimization FL",
                        "status": "running",
                        "progress": 0.65,
                        "agents": 8,
                        "started_at": "2025-08-13T10:30:00Z"
                    },
                    {
                        "id": "exp_002", 
                        "name": "Graph RL Robustness",
                        "status": "completed",
                        "progress": 1.0,
                        "agents": 12,
                        "started_at": "2025-08-12T14:15:00Z"
                    }
                ]
            }
            
        @self.app.get("/api/dashboard/agents/status")
        async def get_agent_status(
            user: Dict[str, Any] = Depends(require_permission(Permission.READ_AGENT))
        ):
            """Get real-time agent status"""
            # Mock data - would integrate with actual FL framework
            return {
                "total_agents": 20,
                "active_agents": 15,
                "idle_agents": 3,
                "failed_agents": 2,
                "agent_details": [
                    {
                        "id": "agent_001",
                        "name": "Traffic Agent 1",
                        "status": "training",
                        "experiment": "exp_001",
                        "cpu_usage": 0.75,
                        "memory_usage": 0.60,
                        "last_update": "2025-08-13T12:45:00Z"
                    },
                    {
                        "id": "agent_002",
                        "name": "Graph Agent 2", 
                        "status": "idle",
                        "experiment": None,
                        "cpu_usage": 0.10,
                        "memory_usage": 0.25,
                        "last_update": "2025-08-13T12:44:30Z"
                    }
                ]
            }
            
        @self.app.get("/api/dashboard/performance/charts")
        async def get_performance_charts(
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Get performance chart data"""
            # Mock data for charts
            return {
                "training_progress": {
                    "timestamps": ["12:00", "12:15", "12:30", "12:45", "13:00"],
                    "rewards": [0.2, 0.35, 0.48, 0.62, 0.71],
                    "losses": [2.1, 1.8, 1.5, 1.2, 0.9]
                },
                "resource_utilization": {
                    "timestamps": ["12:00", "12:15", "12:30", "12:45", "13:00"],
                    "cpu": [45, 67, 72, 65, 58],
                    "memory": [38, 42, 55, 61, 48],
                    "gpu": [82, 89, 91, 87, 84]
                },
                "federation_metrics": {
                    "communication_rounds": 150,
                    "convergence_rate": 0.023,
                    "model_accuracy": 0.89,
                    "synchronization_time": 2.3
                }
            }


# Dashboard HTML templates (simplified - would use proper templating)
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}} - Federated Learning Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-50">
    <!-- Navigation -->
    <nav class="bg-white shadow-lg">
        <div class="max-w-7xl mx-auto px-4">
            <div class="flex justify-between h-16">
                <div class="flex items-center">
                    <h1 class="text-xl font-bold text-gray-800">FL Platform</h1>
                </div>
                <div class="flex items-center space-x-4">
                    <span class="text-sm text-gray-600">{{user.email}}</span>
                    <button class="bg-blue-500 text-white px-4 py-2 rounded">Logout</button>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="max-w-7xl mx-auto py-6 px-4">
        <!-- Dashboard Overview -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-lg font-medium text-gray-900">Active Experiments</h3>
                <p class="text-3xl font-bold text-blue-600">3</p>
            </div>
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-lg font-medium text-gray-900">Running Agents</h3>
                <p class="text-3xl font-bold text-green-600">15</p>
            </div>
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-lg font-medium text-gray-900">Success Rate</h3>
                <p class="text-3xl font-bold text-purple-600">85%</p>
            </div>
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-lg font-medium text-gray-900">Compute Usage</h3>
                <p class="text-3xl font-bold text-orange-600">67%</p>
            </div>
        </div>

        <!-- Charts -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Training Progress</h3>
                <canvas id="trainingChart"></canvas>
            </div>
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Resource Utilization</h3>
                <canvas id="resourceChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Initialize charts
        const trainingCtx = document.getElementById('trainingChart').getContext('2d');
        const resourceCtx = document.getElementById('resourceChart').getContext('2d');
        
        new Chart(trainingCtx, {
            type: 'line',
            data: {
                labels: ['12:00', '12:15', '12:30', '12:45', '13:00'],
                datasets: [{
                    label: 'Reward',
                    data: [0.2, 0.35, 0.48, 0.62, 0.71],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        new Chart(resourceCtx, {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'Memory', 'GPU'],
                datasets: [{
                    data: [67, 48, 84],
                    backgroundColor: [
                        'rgb(239, 68, 68)',
                        'rgb(59, 130, 246)',
                        'rgb(34, 197, 94)'
                    ]
                }]
            },
            options: {
                responsive: true
            }
        });
    </script>
</body>
</html>
"""


# Global dashboard service instance
dashboard_service = DashboardService()