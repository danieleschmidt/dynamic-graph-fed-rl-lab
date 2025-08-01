#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine for SDLC Enhancement
Implements WSJF + ICE + Technical Debt scoring for continuous improvement
"""

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

@dataclass
class ValueItem:
    """Represents a discovered work item with value scores"""
    id: str
    title: str
    description: str
    category: str
    files: List[str]
    effort_estimate: float  # hours
    wsjf_score: float
    ice_score: float
    tech_debt_score: float
    composite_score: float
    created_at: str
    source: str
    priority: str

class ValueDiscoveryEngine:
    """Autonomous value discovery and scoring engine"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.weights = self.config["scoring"]["weights"]["maturing"]
        
    def discover_value_items(self) -> List[ValueItem]:
        """Discover work items from various sources"""
        items = []
        
        # 1. Git history analysis for TODO/FIXME/HACK markers
        items.extend(self._discover_from_git_history())
        
        # 2. Static analysis for code smells and complexity
        items.extend(self._discover_from_static_analysis())
        
        # 3. Security vulnerability scanning
        items.extend(self._discover_from_security_scan())
        
        # 4. Missing SDLC components for maturing repos
        items.extend(self._discover_missing_sdlc_components())
        
        # 5. Technical debt from code metrics
        items.extend(self._discover_technical_debt())
        
        return items
    
    def _discover_from_git_history(self) -> List[ValueItem]:
        """Extract TODOs, FIXMEs from git history and current code"""
        items = []
        
        # Search for TODO/FIXME/HACK patterns in codebase
        patterns = ["TODO", "FIXME", "HACK", "XXX", "DEPRECATED"]
        
        for pattern in patterns:
            try:
                result = subprocess.run([
                    "rg", "-n", "--type", "py", f"#{pattern}|# {pattern}",
                    str(self.repo_path)
                ], capture_output=True, text=True, timeout=30)
                
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path, line_num, comment = parts
                            
                            # Calculate effort based on complexity
                            effort = self._estimate_effort_from_comment(comment)
                            
                            item = self._create_value_item(
                                title=f"Address {pattern.lower()} in {Path(file_path).name}",
                                description=comment.strip(),
                                category="technical-debt",
                                files=[file_path],
                                effort_estimate=effort,
                                source="git_history"
                            )
                            items.append(item)
                            
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
                
        return items[:10]  # Limit to top 10 to avoid overwhelming
    
    def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Run static analysis tools to find quality issues"""
        items = []
        
        # Run ruff for code quality issues
        try:
            result = subprocess.run([
                "ruff", "check", "--output-format=json", str(self.repo_path / "src")
            ], capture_output=True, text=True, timeout=60)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                
                # Group issues by file and severity
                file_issues = {}
                for issue in issues[:20]:  # Limit processing
                    filename = issue.get("filename", "unknown")
                    if filename not in file_issues:
                        file_issues[filename] = []
                    file_issues[filename].append(issue)
                
                # Create value items for files with multiple issues
                for filename, file_issue_list in file_issues.items():
                    if len(file_issue_list) >= 3:  # Only files with multiple issues
                        item = self._create_value_item(
                            title=f"Fix code quality issues in {Path(filename).name}",
                            description=f"Found {len(file_issue_list)} code quality issues",
                            category="code-quality",
                            files=[filename],
                            effort_estimate=len(file_issue_list) * 0.25,  # 15min per issue
                            source="static_analysis"
                        )
                        items.append(item)
                        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass
        
        return items
    
    def _discover_from_security_scan(self) -> List[ValueItem]:
        """Scan for security vulnerabilities"""
        items = []
        
        # Run safety check on requirements
        try:
            result = subprocess.run([
                "safety", "check", "--json", "-r", str(self.repo_path / "requirements.txt")
            ], capture_output=True, text=True, timeout=60)
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                
                for vuln in vulnerabilities[:5]:  # Top 5 critical vulns
                    item = self._create_value_item(
                        title=f"Fix security vulnerability in {vuln.get('package', 'unknown')}",
                        description=vuln.get('advisory', 'Security vulnerability found'),
                        category="security",
                        files=["requirements.txt"],
                        effort_estimate=2.0,  # 2 hours for security fixes
                        source="security_scan"
                    )
                    items.append(item)
                    
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass
        
        return items
    
    def _discover_missing_sdlc_components(self) -> List[ValueItem]:
        """Identify missing SDLC components for maturing repositories"""
        items = []
        
        missing_components = [
            {
                "file": ".pre-commit-config.yaml",
                "title": "Add pre-commit hooks configuration",
                "description": "Implement automated pre-commit checks for code quality",
                "effort": 1.5,
                "category": "automation"
            },
            {
                "file": ".editorconfig", 
                "title": "Add EditorConfig for consistent formatting",
                "description": "Ensure consistent code formatting across editors",
                "effort": 0.5,
                "category": "tooling"
            },
            {
                "file": ".github/dependabot.yml",
                "title": "Configure Dependabot for security updates",
                "description": "Automate dependency vulnerability patches",
                "effort": 1.0,
                "category": "security"
            }
        ]
        
        for component in missing_components:
            if not (self.repo_path / component["file"]).exists():
                item = self._create_value_item(
                    title=component["title"],
                    description=component["description"], 
                    category=component["category"],
                    files=[component["file"]],
                    effort_estimate=component["effort"],
                    source="sdlc_gaps"
                )
                items.append(item)
        
        return items
    
    def _discover_technical_debt(self) -> List[ValueItem]:
        """Identify technical debt through code metrics"""
        items = []
        
        # Check for files with high complexity using radon
        try:
            result = subprocess.run([
                "python", "-c", 
                """
import ast
import os
from pathlib import Path

def calculate_complexity(file_path):
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1
        return complexity
    except:
        return 0

for py_file in Path('src').rglob('*.py'):
    if py_file.stat().st_size > 1000:  # Files > 1KB
        complexity = calculate_complexity(py_file)
        if complexity > 10:  # High complexity threshold
            print(f'{py_file}:{complexity}')
"""
            ], capture_output=True, text=True, cwd=self.repo_path, timeout=30)
            
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    file_path, complexity_str = line.split(':')
                    complexity = int(complexity_str)
                    
                    if complexity > 15:  # Very high complexity
                        item = self._create_value_item(
                            title=f"Refactor high-complexity file {Path(file_path).name}",
                            description=f"Reduce complexity from {complexity} to improve maintainability",
                            category="refactoring",
                            files=[file_path],
                            effort_estimate=complexity * 0.3,  # 18min per complexity point
                            source="complexity_analysis"
                        )
                        items.append(item)
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return items[:5]  # Top 5 most complex files
    
    def _create_value_item(self, title: str, description: str, category: str, 
                          files: List[str], effort_estimate: float, source: str) -> ValueItem:
        """Create a value item with calculated scores"""
        
        # Calculate WSJF components
        user_value = self._calculate_user_business_value(category, files)
        time_criticality = self._calculate_time_criticality(category)
        risk_reduction = self._calculate_risk_reduction(category, description)
        opportunity_enablement = self._calculate_opportunity_enablement(category)
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity_enablement
        wsjf_score = cost_of_delay / max(effort_estimate, 0.5)  # Avoid division by zero
        
        # Calculate ICE components
        impact = self._calculate_impact(category, files)
        confidence = self._calculate_confidence(source, category)
        ease = self._calculate_ease(effort_estimate)
        
        ice_score = impact * confidence * ease
        
        # Calculate Technical Debt Score
        debt_impact = self._calculate_debt_impact(category, files)
        debt_interest = self._calculate_debt_interest(category)
        hotspot_multiplier = self._calculate_hotspot_multiplier(files)
        
        tech_debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        
        # Calculate Composite Score
        composite_score = (
            self.weights["wsjf"] * self._normalize_score(wsjf_score, 0, 50) +
            self.weights["ice"] * self._normalize_score(ice_score, 0, 1000) +
            self.weights["technicalDebt"] * self._normalize_score(tech_debt_score, 0, 100) +
            self.weights["security"] * (2.0 if category == "security" else 1.0)
        )
        
        # Apply category boosts
        if category == "security":
            composite_score *= 2.0
        elif category == "compliance":
            composite_score *= 1.8
        
        item_id = f"{category}_{int(time.time())}_{hash(title) % 1000:03d}"
        
        return ValueItem(
            id=item_id,
            title=title,
            description=description,
            category=category,
            files=files,
            effort_estimate=effort_estimate,
            wsjf_score=wsjf_score,
            ice_score=ice_score,
            tech_debt_score=tech_debt_score,
            composite_score=composite_score,
            created_at=datetime.now().isoformat(),
            source=source,
            priority=self._calculate_priority(composite_score)
        )
    
    def _calculate_user_business_value(self, category: str, files: List[str]) -> float:
        """Calculate user/business value component"""
        category_values = {
            "security": 9.0,
            "performance": 7.0, 
            "reliability": 8.0,
            "technical-debt": 5.0,
            "code-quality": 4.0,
            "automation": 6.0,
            "documentation": 3.0,
            "tooling": 4.0,
            "refactoring": 5.0
        }
        return category_values.get(category, 5.0)
    
    def _calculate_time_criticality(self, category: str) -> float:
        """Calculate time criticality component"""
        if category == "security":
            return 8.0
        elif category in ["performance", "reliability"]:
            return 6.0
        return 3.0
    
    def _calculate_risk_reduction(self, category: str, description: str) -> float:
        """Calculate risk reduction component"""
        risk_keywords = ["vulnerability", "security", "failure", "error", "bug"]
        risk_score = sum(1 for keyword in risk_keywords if keyword in description.lower())
        return min(risk_score * 2.0, 8.0)
    
    def _calculate_opportunity_enablement(self, category: str) -> float:
        """Calculate opportunity enablement component"""
        if category in ["automation", "tooling"]:
            return 7.0
        elif category == "performance":
            return 6.0
        return 3.0
    
    def _calculate_impact(self, category: str, files: List[str]) -> float:
        """Calculate ICE Impact (1-10)"""
        base_impact = {
            "security": 9,
            "performance": 8,
            "reliability": 8,
            "technical-debt": 6,
            "automation": 7,
            "code-quality": 5
        }.get(category, 5)
        
        # Boost for multiple files affected
        file_multiplier = min(1.0 + len(files) * 0.1, 1.5)
        return min(base_impact * file_multiplier, 10)
    
    def _calculate_confidence(self, source: str, category: str) -> float:
        """Calculate ICE Confidence (1-10)"""
        source_confidence = {
            "static_analysis": 8,
            "security_scan": 9,
            "git_history": 7,
            "complexity_analysis": 8,
            "sdlc_gaps": 9
        }.get(source, 6)
        
        return min(source_confidence, 10)
    
    def _calculate_ease(self, effort_estimate: float) -> float:
        """Calculate ICE Ease (1-10, higher is easier)"""
        if effort_estimate <= 1.0:
            return 9
        elif effort_estimate <= 3.0:
            return 7
        elif effort_estimate <= 8.0:
            return 5
        else:
            return 3
    
    def _calculate_debt_impact(self, category: str, files: List[str]) -> float:
        """Calculate technical debt impact"""
        if category == "technical-debt":
            return 50.0
        elif category == "refactoring":
            return 40.0
        elif category == "code-quality":
            return 30.0
        return 10.0
    
    def _calculate_debt_interest(self, category: str) -> float:
        """Calculate technical debt interest (future cost)"""
        interest_rates = {
            "technical-debt": 20.0,
            "security": 30.0,
            "performance": 15.0,
            "code-quality": 10.0
        }
        return interest_rates.get(category, 5.0)
    
    def _calculate_hotspot_multiplier(self, files: List[str]) -> float:
        """Calculate hotspot multiplier based on file change frequency"""
        # Simplified: assume all files have baseline activity
        return 1.2  # Slight boost for all items
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range"""
        if max_val == min_val:
            return 50.0
        normalized = ((score - min_val) / (max_val - min_val)) * 100
        return max(0, min(100, normalized))
    
    def _calculate_priority(self, composite_score: float) -> str:
        """Calculate priority based on composite score"""
        if composite_score >= 70:
            return "high"
        elif composite_score >= 40:
            return "medium" 
        else:
            return "low"
    
    def _estimate_effort_from_comment(self, comment: str) -> float:
        """Estimate effort from TODO/FIXME comment content"""
        # Simple heuristic based on comment length and keywords
        base_effort = 1.0  # 1 hour baseline
        
        complexity_keywords = ["refactor", "rewrite", "complex", "difficult", "major"]
        simple_keywords = ["fix", "update", "change", "simple", "quick"]
        
        if any(keyword in comment.lower() for keyword in complexity_keywords):
            return base_effort * 3.0
        elif any(keyword in comment.lower() for keyword in simple_keywords):
            return base_effort * 0.5
        
        # Length-based estimate
        return base_effort + (len(comment) / 100.0)

async def main():
    """Main execution function"""
    engine = ValueDiscoveryEngine()
    
    print("🔍 Discovering value items...")
    items = engine.discover_value_items()
    
    print(f"📊 Found {len(items)} value items")
    
    # Sort by composite score
    items.sort(key=lambda x: x.composite_score, reverse=True)
    
    # Display top 10 items
    print("\n🎯 Top 10 Value Items:")
    print("=" * 80)
    
    for i, item in enumerate(items[:10], 1):
        print(f"{i:2d}. {item.title}")
        print(f"    Category: {item.category} | Score: {item.composite_score:.1f}")
        print(f"    Effort: {item.effort_estimate:.1f}h | Priority: {item.priority}")
        print(f"    Description: {item.description[:60]}...")
        print()
    
    # Save metrics
    metrics = {
        "total_items_discovered": len(items),
        "categories": {},
        "average_composite_score": sum(item.composite_score for item in items) / len(items) if items else 0,
        "high_priority_items": sum(1 for item in items if item.priority == "high"),
        "last_discovery": datetime.now().isoformat()
    }
    
    for item in items:
        if item.category not in metrics["categories"]:
            metrics["categories"][item.category] = 0
        metrics["categories"][item.category] += 1
    
    # Save to metrics file
    metrics_path = Path(".terragon/value-metrics.json")
    metrics_path.parent.mkdir(exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Saved metrics to {metrics_path}")
    print(f"📈 Average score: {metrics['average_composite_score']:.1f}")
    print(f"🔥 High priority items: {metrics['high_priority_items']}")

if __name__ == "__main__":
    asyncio.run(main())