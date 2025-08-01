#!/usr/bin/env python3
"""
Demo of Autonomous Value Discovery System
Simulates the full discovery and scoring without external dependencies
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class ValueItem:
    id: str
    title: str
    description: str
    category: str
    files: List[str]
    effort_estimate: float
    wsjf_score: float
    ice_score: float
    tech_debt_score: float
    composite_score: float
    created_at: str
    source: str
    priority: str

def simulate_value_discovery():
    """Simulate value discovery based on repository analysis"""
    
    # Simulated discovered items based on actual repository gaps
    discovered_items = [
        {
            "title": "Enhance GitHub Actions workflow documentation",
            "description": "The repository has workflow templates in docs/ but no actual CI/CD workflows active",
            "category": "automation",
            "files": ["docs/workflows/"],
            "effort_estimate": 3.0,
            "source": "sdlc_gaps"
        },
        {
            "title": "Add comprehensive error handling in federation module", 
            "description": "Federation code lacks robust error handling for network failures",
            "category": "reliability",
            "files": ["src/dynamic_graph_fed_rl/federation/__init__.py"],
            "effort_estimate": 4.0,
            "source": "code_analysis"
        },
        {
            "title": "Implement performance benchmarking automation",
            "description": "Manual performance tests need automation for continuous monitoring",
            "category": "performance",
            "files": ["tests/performance/"],
            "effort_estimate": 6.0,
            "source": "testing_gaps"
        },
        {
            "title": "Add security scanning for JAX/ML dependencies",
            "description": "ML dependencies have unique security considerations not covered",
            "category": "security", 
            "files": ["requirements.txt", "pyproject.toml"],
            "effort_estimate": 2.5,
            "source": "security_scan"
        },
        {
            "title": "Create development environment documentation",
            "description": "Complex ML setup needs better developer onboarding docs",
            "category": "documentation",
            "files": ["DEVELOPMENT.md"],
            "effort_estimate": 2.0,
            "source": "documentation_gaps"
        },
        {
            "title": "Optimize graph memory usage in large networks",
            "description": "Memory usage grows quadratically with graph size in current implementation",
            "category": "performance",
            "files": ["src/dynamic_graph_fed_rl/models/", "src/dynamic_graph_fed_rl/utils/"],
            "effort_estimate": 8.0,
            "source": "performance_analysis"
        },
        {
            "title": "Add distributed training failure recovery",
            "description": "Federated training needs graceful handling of agent failures",
            "category": "reliability",
            "files": ["src/dynamic_graph_fed_rl/federation/"],
            "effort_estimate": 5.0,
            "source": "reliability_analysis"
        },
        {
            "title": "Implement real-time monitoring dashboard",
            "description": "Grafana setup exists but needs real-time training metrics integration",
            "category": "monitoring",
            "files": ["monitoring/grafana/"],
            "effort_estimate": 4.5,
            "source": "monitoring_gaps"
        }
    ]
    
    # Scoring weights for maturing repository
    weights = {
        "wsjf": 0.6,
        "ice": 0.1,
        "technicalDebt": 0.2,
        "security": 0.1
    }
    
    value_items = []
    
    for i, item_data in enumerate(discovered_items):
        # Calculate WSJF Score
        user_value = calculate_user_business_value(item_data["category"])
        time_criticality = calculate_time_criticality(item_data["category"])
        risk_reduction = calculate_risk_reduction(item_data["category"], item_data["description"])
        opportunity_enablement = calculate_opportunity_enablement(item_data["category"])
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity_enablement
        wsjf_score = cost_of_delay / max(item_data["effort_estimate"], 0.5)
        
        # Calculate ICE Score
        impact = calculate_impact(item_data["category"], item_data["files"])
        confidence = calculate_confidence(item_data["source"], item_data["category"])
        ease = calculate_ease(item_data["effort_estimate"])
        
        ice_score = impact * confidence * ease
        
        # Calculate Technical Debt Score
        debt_impact = calculate_debt_impact(item_data["category"])
        debt_interest = calculate_debt_interest(item_data["category"])
        hotspot_multiplier = 1.2  # Simplified
        
        tech_debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        
        # Calculate Composite Score
        composite_score = (
            weights["wsjf"] * normalize_score(wsjf_score, 0, 50) +
            weights["ice"] * normalize_score(ice_score, 0, 1000) +
            weights["technicalDebt"] * normalize_score(tech_debt_score, 0, 100) +
            weights["security"] * (2.0 if item_data["category"] == "security" else 1.0)
        )
        
        # Apply category boosts
        if item_data["category"] == "security":
            composite_score *= 2.0
        elif item_data["category"] == "performance":
            composite_score *= 1.3
        elif item_data["category"] == "reliability":
            composite_score *= 1.2
        
        item_id = f"{item_data['category']}_{int(time.time())}_{i:03d}"
        
        value_item = ValueItem(
            id=item_id,
            title=item_data["title"],
            description=item_data["description"],
            category=item_data["category"],
            files=item_data["files"],
            effort_estimate=item_data["effort_estimate"],
            wsjf_score=wsjf_score,
            ice_score=ice_score,
            tech_debt_score=tech_debt_score,
            composite_score=composite_score,
            created_at=datetime.now().isoformat(),
            source=item_data["source"],
            priority=calculate_priority(composite_score)
        )
        
        value_items.append(value_item)
    
    return value_items

def calculate_user_business_value(category: str) -> float:
    category_values = {
        "security": 9.0,
        "performance": 8.0,
        "reliability": 8.0,
        "automation": 6.0,
        "monitoring": 7.0,
        "documentation": 4.0
    }
    return category_values.get(category, 5.0)

def calculate_time_criticality(category: str) -> float:
    if category == "security":
        return 8.0
    elif category in ["performance", "reliability"]:
        return 6.0
    elif category == "monitoring":
        return 5.0
    return 3.0

def calculate_risk_reduction(category: str, description: str) -> float:
    risk_keywords = ["failure", "error", "vulnerability", "crash", "memory", "security"]
    risk_score = sum(1 for keyword in risk_keywords if keyword in description.lower())
    return min(risk_score * 2.0, 8.0)

def calculate_opportunity_enablement(category: str) -> float:
    if category in ["automation", "monitoring"]:
        return 7.0
    elif category == "performance":
        return 6.0
    elif category == "documentation":
        return 5.0
    return 3.0

def calculate_impact(category: str, files: List[str]) -> float:
    base_impact = {
        "security": 9,
        "performance": 8,
        "reliability": 8,
        "automation": 7,
        "monitoring": 7,
        "documentation": 5
    }.get(category, 5)
    
    file_multiplier = min(1.0 + len(files) * 0.1, 1.5)
    return min(base_impact * file_multiplier, 10)

def calculate_confidence(source: str, category: str) -> float:
    source_confidence = {
        "sdlc_gaps": 9,
        "code_analysis": 8,
        "security_scan": 9,
        "performance_analysis": 8,
        "testing_gaps": 7,
        "documentation_gaps": 8,
        "reliability_analysis": 8,
        "monitoring_gaps": 7
    }.get(source, 6)
    
    return min(source_confidence, 10)

def calculate_ease(effort_estimate: float) -> float:
    if effort_estimate <= 2.0:
        return 9
    elif effort_estimate <= 4.0:
        return 7
    elif effort_estimate <= 6.0:
        return 5
    else:
        return 3

def calculate_debt_impact(category: str) -> float:
    if category == "performance":
        return 60.0
    elif category == "reliability":
        return 55.0
    elif category == "security":
        return 70.0
    elif category == "automation":
        return 40.0
    return 20.0

def calculate_debt_interest(category: str) -> float:
    interest_rates = {
        "security": 30.0,
        "performance": 25.0,
        "reliability": 20.0,
        "automation": 15.0,
        "monitoring": 10.0,
        "documentation": 5.0
    }
    return interest_rates.get(category, 5.0)

def normalize_score(score: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 50.0
    normalized = ((score - min_val) / (max_val - min_val)) * 100
    return max(0, min(100, normalized))

def calculate_priority(composite_score: float) -> str:
    if composite_score >= 70:
        return "high"
    elif composite_score >= 40:
        return "medium"
    else:
        return "low"

def main():
    print("🔍 Terragon Autonomous Value Discovery - DEMO")
    print("=" * 60)
    print(f"Repository: dynamic-graph-fed-rl-lab")
    print(f"Maturity Level: MATURING (65%)")
    print(f"Discovery Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Discover value items
    items = simulate_value_discovery()
    
    # Sort by composite score
    items.sort(key=lambda x: x.composite_score, reverse=True)
    
    print(f"📊 Discovered {len(items)} high-value work items")
    print()
    
    # Display all items with detailed scores
    print("🎯 VALUE-RANKED BACKLOG")
    print("=" * 60)
    
    for i, item in enumerate(items, 1):
        print(f"{i:2d}. {item.title}")
        print(f"    Category: {item.category.upper()} | Priority: {item.priority.upper()}")
        print(f"    Composite Score: {item.composite_score:.1f}")
        print(f"    ├─ WSJF: {item.wsjf_score:.1f} | ICE: {item.ice_score:.0f} | Tech Debt: {item.tech_debt_score:.1f}")
        print(f"    ├─ Effort: {item.effort_estimate:.1f} hours | Source: {item.source}")
        print(f"    ├─ Files: {', '.join(item.files)}")
        print(f"    └─ {item.description}")
        print()
    
    # Calculate summary metrics
    total_effort = sum(item.effort_estimate for item in items)
    high_priority = sum(1 for item in items if item.priority == "high")
    security_items = sum(1 for item in items if item.category == "security")
    avg_score = sum(item.composite_score for item in items) / len(items)
    
    # Save metrics
    metrics = {
        "discovery_timestamp": datetime.now().isoformat(),
        "total_items_discovered": len(items),
        "total_estimated_effort_hours": total_effort,
        "high_priority_items": high_priority,
        "security_items": security_items,
        "average_composite_score": avg_score,
        "categories": {},
        "next_best_value_item": {
            "id": items[0].id,
            "title": items[0].title,
            "category": items[0].category,
            "score": items[0].composite_score,
            "effort": items[0].effort_estimate
        } if items else None
    }
    
    # Count by category
    for item in items:
        if item.category not in metrics["categories"]:
            metrics["categories"][item.category] = 0
        metrics["categories"][item.category] += 1
    
    # Save metrics to file
    metrics_path = Path("value-metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("📈 DISCOVERY SUMMARY")
    print("=" * 60)
    print(f"Total Work Items: {len(items)}")
    print(f"High Priority Items: {high_priority}")
    print(f"Security Items: {security_items}")
    print(f"Total Estimated Effort: {total_effort:.1f} hours")
    print(f"Average Score: {avg_score:.1f}")
    print()
    
    print("📊 CATEGORY BREAKDOWN")
    for category, count in sorted(metrics["categories"].items()):
        percentage = (count / len(items)) * 100
        print(f"  {category.capitalize()}: {count} items ({percentage:.1f}%)")
    print()
    
    if items:
        next_item = items[0]
        print("🚀 NEXT BEST VALUE ITEM")
        print("=" * 60)
        print(f"Title: {next_item.title}")
        print(f"Score: {next_item.composite_score:.1f} | Effort: {next_item.effort_estimate:.1f}h")
        print(f"Expected ROI: {next_item.composite_score / next_item.effort_estimate:.1f} value/hour")
        print()
        
        print("🎯 EXECUTION RECOMMENDATION")
        print("Create feature branch and implement:")
        print(f"  git checkout -b auto-value/{next_item.id}")
        print(f"  # Implement: {next_item.description}")
        print(f"  # Estimated time: {next_item.effort_estimate:.1f} hours")
        print()
    
    print(f"💾 Metrics saved to: {metrics_path.absolute()}")
    print("🔄 Run '.terragon/autonomous_executor.py' to execute highest value item")

if __name__ == "__main__":
    main()