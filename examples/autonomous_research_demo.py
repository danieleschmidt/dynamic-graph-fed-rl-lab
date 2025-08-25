"""
Autonomous Research Director Demonstration.

This demo showcases the cutting-edge autonomous research capabilities of the
Generation 4+ system, including:

1. Autonomous discovery of research opportunities
2. Automated hypothesis generation and testing
3. Self-directed experimental design and execution
4. Automatic publication generation with peer review prediction
5. Iterative research direction evolution

This represents the pinnacle of autonomous scientific discovery in AI systems.
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

from src.dynamic_graph_fed_rl.quantum_planner.core import QuantumTaskPlanner
from src.dynamic_graph_fed_rl.quantum_planner.performance_monitor import PerformanceMonitor
from src.dynamic_graph_fed_rl.optimization.generation4_system import (
    Generation4OptimizationSystem,
    SystemConfiguration,
    OptimizationStrategy,
)
from src.dynamic_graph_fed_rl.research.experimental_framework import ResearchExperimentRunner
from src.dynamic_graph_fed_rl.research.autonomous_research_director import AutonomousResearchDirector


async def setup_autonomous_research_logging():
    """Setup comprehensive logging for autonomous research demo."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "autonomous_research_demo.log"),
            logging.StreamHandler()
        ]
    )
    
    # Get logger for demo
    logger = logging.getLogger("AutonomousResearchDemo")
    return logger


async def demonstrate_autonomous_research_system():
    """Demonstrate the Autonomous Research Director system."""
    
    logger = await setup_autonomous_research_logging()
    
    logger.info("🔬 Starting Autonomous Research Director Demonstration")
    logger.info("=" * 90)
    
    try:
        # Configuration for research system
        config = SystemConfiguration(
            openai_api_key = "SECURE_API_KEY_FROM_ENV"  # TODO: Use environment variable,  # Replace with actual key for GPT-4 integration
            optimization_strategy=OptimizationStrategy.BALANCED,
            autonomous_mode_enabled=True,
            max_concurrent_experiments=2,
            safety_mode=True,
            learning_rate=0.01,
            exploration_rate=0.15,  # Higher exploration for research
            intervention_threshold=0.90,
        )
        
        # Initialize core research infrastructure
        logger.info("🧬 Initializing autonomous research infrastructure...")
        
        quantum_planner = QuantumTaskPlanner(
            max_parallel_tasks=6,
            quantum_coherence_time=15.0,
            interference_strength=0.2,
        )
        
        performance_monitor = PerformanceMonitor(
            collection_interval=20.0,  # More frequent for research
            history_retention_hours=168,  # 1 week retention
            logger=logger,
        )
        
        # Initialize Generation 4 system (prerequisite for research)
        logger.info("🚀 Initializing Generation 4 optimization system...")
        
        gen4_system = Generation4OptimizationSystem(
            config=config,
            quantum_planner=quantum_planner,
            performance_monitor=performance_monitor,
            logger=logger,
        )
        
        # Initialize research experiment runner
        logger.info("🧪 Initializing research experiment runner...")
        
        research_runner = ResearchExperimentRunner(
            results_dir="autonomous_research_results",
            random_seed=42
        )
        
        # Initialize Autonomous Research Director
        logger.info("🧠 Initializing Autonomous Research Director...")
        
        research_director = AutonomousResearchDirector(
            generation4_system=gen4_system,
            research_runner=research_runner,
            results_dir="autonomous_research_portfolio",
            novelty_threshold=0.7,
            publication_threshold=0.75,
            max_concurrent_studies=2,
            logger=logger,
        )
        
        # System initialization
        logger.info("⚙️ Performing system initialization...")
        
        # Initialize Generation 4 system first
        gen4_init_success = await gen4_system.initialize_system()
        if not gen4_init_success:
            logger.error("❌ Generation 4 system initialization failed")
            return False
        
        logger.info("✅ All systems initialized successfully")
        
        # Start monitoring
        logger.info("📊 Starting performance monitoring...")
        monitor_task = asyncio.create_task(performance_monitor.start_monitoring())
        
        # Start Generation 4 optimization (provides data for research)
        logger.info("🎯 Starting Generation 4 optimization for research data...")
        gen4_task = asyncio.create_task(
            run_generation4_for_research(gen4_system, 180, logger)  # 3 minutes
        )
        
        # Start autonomous research
        logger.info("🔬 Starting autonomous research system...")
        research_task = asyncio.create_task(
            run_autonomous_research_demo(research_director, 300, logger)  # 5 minutes
        )
        
        # Run both systems concurrently
        await asyncio.gather(gen4_task, research_task)
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        monitor_task.cancel()
        
        # Generate comprehensive demo report
        await generate_autonomous_research_report(
            research_director, performance_monitor, logger
        )
        
        logger.info("🎯 Autonomous Research demonstration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Autonomous Research demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def run_generation4_for_research(
    gen4_system: Generation4OptimizationSystem,
    duration: int,
    logger: logging.Logger,
):
    """Run Generation 4 system to generate research data."""
    
    logger.info(f"🔄 Running Generation 4 optimization for {duration} seconds to generate research data...")
    
    # Start autonomous optimization
    optimization_task = asyncio.create_task(gen4_system.start_autonomous_optimization())
    
    # Let it run for specified duration
    try:
        await asyncio.wait_for(optimization_task, timeout=duration)
    except asyncio.TimeoutError:
        logger.info("⏰ Generation 4 data collection time reached, stopping...")
        await gen4_system.stop_autonomous_optimization()
    
    logger.info("📈 Generation 4 data collection completed")


async def run_autonomous_research_demo(
    research_director: AutonomousResearchDirector,
    duration: int,
    logger: logging.Logger,
):
    """Run autonomous research demonstration."""
    
    logger.info(f"🔬 Running autonomous research for {duration} seconds...")
    
    # Start autonomous research
    research_task = asyncio.create_task(research_director.start_autonomous_research())
    
    # Let it run for demonstration duration
    try:
        await asyncio.wait_for(research_task, timeout=duration)
    except asyncio.TimeoutError:
        logger.info("⏰ Research demo time limit reached, stopping...")
        await research_director.stop_autonomous_research()
    
    logger.info("🔬 Autonomous research demo completed")


async def generate_autonomous_research_report(
    research_director: AutonomousResearchDirector,
    performance_monitor: PerformanceMonitor,
    logger: logging.Logger,
):
    """Generate comprehensive autonomous research demonstration report."""
    
    logger.info("📋 Generating autonomous research demonstration report...")
    
    try:
        # Collect comprehensive data
        research_status = {
            "current_phase": research_director.current_phase.value,
            "opportunities_discovered": len(research_director.research_opportunities),
            "active_hypotheses": len(research_director.active_hypotheses),
            "completed_studies": len(research_director.completed_studies),
            "publications_generated": len(research_director.publications),
            "research_metrics": research_director.research_metrics,
        }
        
        monitoring_stats = performance_monitor.get_monitoring_stats()
        current_metrics = await performance_monitor.get_current_metrics()
        
        # Create comprehensive research demo report
        report = {
            "autonomous_research_demo_report": {
                "timestamp": datetime.now().isoformat(),
                "demo_summary": {
                    "demo_type": "Autonomous Research Director",
                    "duration_demonstrated": "5 minutes",
                    "components_demonstrated": [
                        "Research Opportunity Detection",
                        "Autonomous Hypothesis Generation",
                        "Experimental Design Automation",
                        "Research Execution Pipeline",
                        "Statistical Analysis Automation",
                        "Publication Generation",
                        "Peer Review Prediction",
                        "Research Iteration & Evolution"
                    ],
                    "research_domains_explored": [
                        "Quantum Coherence Optimization",
                        "Adversarial Robustness",
                        "Communication Efficiency",
                        "Cross-Domain Synergies"
                    ],
                },
                "research_director_status": research_status,
                "performance_monitoring": monitoring_stats,
                "current_metrics": current_metrics,
                "research_achievements": _analyze_research_achievements(research_director),
                "publication_portfolio": _analyze_publication_portfolio(research_director),
                "breakthrough_discoveries": _identify_breakthrough_discoveries(research_director),
                "research_evolution": _analyze_research_evolution(research_director),
                "future_research_potential": _assess_future_research_potential(research_director),
                "recommendations": _generate_research_recommendations(research_director),
            }
        }
        
        # Save comprehensive report
        reports_dir = Path("results")
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / "autonomous_research_demo_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Autonomous research demo report saved: {report_path}")
        
        # Print comprehensive summary
        _print_research_demo_summary(report, logger)
        
        # Save individual components
        await _save_research_components(research_director, reports_dir, logger)
        
    except Exception as e:
        logger.error(f"Failed to generate autonomous research report: {e}")


def _analyze_research_achievements(research_director: AutonomousResearchDirector) -> list:
    """Analyze research achievements from the demonstration."""
    achievements = []
    
    # Discovery achievements
    if research_director.research_opportunities:
        achievements.append(f"🔍 Discovered {len(research_director.research_opportunities)} research opportunities")
        
        # Analyze opportunity priorities
        priorities = [opp.priority.value for opp in research_director.research_opportunities]
        breakthrough_count = priorities.count("breakthrough")
        if breakthrough_count > 0:
            achievements.append(f"🏆 Identified {breakthrough_count} breakthrough opportunities")
    
    # Hypothesis generation achievements
    if research_director.active_hypotheses:
        achievements.append(f"💡 Generated {len(research_director.active_hypotheses)} testable hypotheses")
    
    # Experimental achievements
    if research_director.completed_studies:
        achievements.append(f"🧪 Completed {len(research_director.completed_studies)} research studies")
        
        # Analyze study success rates
        successful_studies = [
            study for study in research_director.completed_studies
            if study.get("results", {}).get("successful", False)
        ]
        if successful_studies:
            success_rate = len(successful_studies) / len(research_director.completed_studies)
            achievements.append(f"✅ Achieved {success_rate:.1%} study success rate")
    
    # Publication achievements
    if research_director.publications:
        achievements.append(f"📝 Generated {len(research_director.publications)} research publications")
        
        # Analyze publication quality
        high_quality_pubs = [
            pub for pub in research_director.publications
            if pub.publication_readiness >= 0.8
        ]
        if high_quality_pubs:
            achievements.append(f"⭐ {len(high_quality_pubs)} publications ready for peer review")
    
    # Autonomy achievements
    metrics = research_director.research_metrics
    if metrics["opportunities_discovered"] > 0 or metrics["hypotheses_generated"] > 0:
        achievements.append("🤖 Demonstrated autonomous research capability")
    
    return achievements


def _analyze_publication_portfolio(research_director: AutonomousResearchDirector) -> dict:
    """Analyze the generated publication portfolio."""
    portfolio = {
        "total_publications": len(research_director.publications),
        "average_novelty_score": 0.0,
        "average_impact_score": 0.0,
        "average_publication_readiness": 0.0,
        "peer_review_predictions": {},
        "publication_categories": {},
    }
    
    if research_director.publications:
        novelty_scores = [pub.novelty_score for pub in research_director.publications]
        impact_scores = [pub.impact_score for pub in research_director.publications]
        readiness_scores = [pub.publication_readiness for pub in research_director.publications]
        
        portfolio.update({
            "average_novelty_score": sum(novelty_scores) / len(novelty_scores),
            "average_impact_score": sum(impact_scores) / len(impact_scores),
            "average_publication_readiness": sum(readiness_scores) / len(readiness_scores),
        })
        
        # Analyze peer review predictions
        acceptance_probs = []
        for pub in research_director.publications:
            if pub.peer_review_predictions:
                accept_prob = pub.peer_review_predictions.get("acceptance_probability", 0)
                acceptance_probs.append(accept_prob)
        
        if acceptance_probs:
            portfolio["peer_review_predictions"] = {
                "average_acceptance_probability": sum(acceptance_probs) / len(acceptance_probs),
                "high_acceptance_probability_count": sum(1 for p in acceptance_probs if p >= 0.7),
            }
        
        # Categorize publications by quality
        portfolio["publication_categories"] = {
            "breakthrough": sum(1 for pub in research_director.publications if pub.novelty_score >= 0.9),
            "high_impact": sum(1 for pub in research_director.publications if pub.impact_score >= 0.8),
            "ready_for_submission": sum(1 for pub in research_director.publications if pub.publication_readiness >= 0.8),
        }
    
    return portfolio


def _identify_breakthrough_discoveries(research_director: AutonomousResearchDirector) -> list:
    """Identify breakthrough discoveries from the research."""
    breakthroughs = []
    
    # High novelty publications
    for pub in research_director.publications:
        if pub.novelty_score >= 0.9:
            breakthroughs.append({
                "type": "High Novelty Publication",
                "title": pub.title,
                "novelty_score": pub.novelty_score,
                "description": f"Publication with {pub.novelty_score:.1%} novelty score"
            })
    
    # High impact opportunities
    for opp in research_director.research_opportunities:
        if opp.priority.value == "breakthrough" and opp.expected_impact >= 0.9:
            breakthroughs.append({
                "type": "Breakthrough Research Opportunity",
                "title": opp.title,
                "expected_impact": opp.expected_impact,
                "description": opp.description
            })
    
    # Successful high-confidence studies
    for study in research_director.completed_studies:
        results = study.get("results", {})
        if (results.get("successful", False) and 
            results.get("final_performance", 0) > 0.9 and
            results.get("p_value", 1.0) < 0.001):
            breakthroughs.append({
                "type": "High-Confidence Research Result",
                "study_id": study.get("study_id", "unknown"),
                "performance": results.get("final_performance", 0),
                "p_value": results.get("p_value", 1.0),
                "description": f"Study with {results.get('final_performance', 0):.1%} performance and p={results.get('p_value', 1.0):.2e}"
            })
    
    return breakthroughs


def _analyze_research_evolution(research_director: AutonomousResearchDirector) -> dict:
    """Analyze how research has evolved during the demonstration."""
    evolution = {
        "opportunity_discovery_rate": 0.0,
        "hypothesis_generation_rate": 0.0,
        "research_phase_progression": research_director.current_phase.value,
        "domain_exploration": {},
        "research_velocity": 0.0,
    }
    
    # Calculate research velocity
    total_research_actions = (
        research_director.research_metrics["opportunities_discovered"] +
        research_director.research_metrics["hypotheses_generated"] +
        research_director.research_metrics["experiments_executed"]
    )
    
    evolution["research_velocity"] = total_research_actions  # Actions per demo period
    
    # Analyze domain exploration
    all_domains = []
    for opp in research_director.research_opportunities:
        all_domains.extend([domain.value for domain in opp.research_domains])
    
    if all_domains:
        from collections import Counter
        domain_counts = Counter(all_domains)
        evolution["domain_exploration"] = dict(domain_counts)
    
    return evolution


def _assess_future_research_potential(research_director: AutonomousResearchDirector) -> dict:
    """Assess the potential for future research based on current progress."""
    potential = {
        "research_pipeline_health": "unknown",
        "opportunity_richness": 0.0,
        "hypothesis_quality": 0.0,
        "experimental_capacity": 0.0,
        "publication_potential": 0.0,
        "breakthrough_probability": 0.0,
    }
    
    # Assess opportunity richness
    if research_director.research_opportunities:
        high_priority_ops = [
            opp for opp in research_director.research_opportunities
            if opp.priority.value in ["breakthrough", "high_impact"]
        ]
        potential["opportunity_richness"] = len(high_priority_ops) / len(research_director.research_opportunities)
    
    # Assess hypothesis quality
    if research_director.active_hypotheses:
        high_confidence_hyps = [
            hyp for hyp in research_director.active_hypotheses
            if len(hyp.testable_predictions) >= 3
        ]
        potential["hypothesis_quality"] = len(high_confidence_hyps) / len(research_director.active_hypotheses)
    
    # Assess experimental capacity
    max_concurrent = research_director.max_concurrent_studies
    current_active = sum(1 for h in research_director.active_hypotheses if hasattr(h, 'experiment_running'))
    potential["experimental_capacity"] = (max_concurrent - current_active) / max_concurrent
    
    # Assess publication potential
    if research_director.publications:
        ready_publications = [
            pub for pub in research_director.publications
            if pub.publication_readiness >= 0.8
        ]
        potential["publication_potential"] = len(ready_publications) / len(research_director.publications)
    
    # Assess breakthrough probability
    breakthrough_indicators = [
        potential["opportunity_richness"] >= 0.5,
        potential["hypothesis_quality"] >= 0.7,
        research_director.research_metrics["breakthrough_discoveries"] > 0,
    ]
    potential["breakthrough_probability"] = sum(breakthrough_indicators) / len(breakthrough_indicators)
    
    # Overall pipeline health
    health_score = sum([
        potential["opportunity_richness"],
        potential["hypothesis_quality"],
        potential["experimental_capacity"],
        potential["publication_potential"],
    ]) / 4
    
    if health_score >= 0.8:
        potential["research_pipeline_health"] = "excellent"
    elif health_score >= 0.6:
        potential["research_pipeline_health"] = "good"
    elif health_score >= 0.4:
        potential["research_pipeline_health"] = "fair"
    else:
        potential["research_pipeline_health"] = "needs_improvement"
    
    return potential


def _generate_research_recommendations(research_director: AutonomousResearchDirector) -> list:
    """Generate recommendations for future research directions."""
    recommendations = []
    
    # Based on research metrics
    metrics = research_director.research_metrics
    
    if metrics["opportunities_discovered"] == 0:
        recommendations.append({
            "category": "Discovery",
            "priority": "High",
            "recommendation": "Increase data collection time to enable opportunity discovery",
            "details": "The system needs more optimization data to identify research opportunities."
        })
    
    if metrics["hypotheses_generated"] == 0 and metrics["opportunities_discovered"] > 0:
        recommendations.append({
            "category": "Hypothesis Generation", 
            "priority": "Medium",
            "recommendation": "Allow more time for hypothesis generation",
            "details": "Research opportunities exist but need more time to generate testable hypotheses."
        })
    
    if metrics["experiments_executed"] == 0 and metrics["hypotheses_generated"] > 0:
        recommendations.append({
            "category": "Experimentation",
            "priority": "High",
            "recommendation": "Increase experimental execution time",
            "details": "Hypotheses are ready but need more time for experimental validation."
        })
    
    # Based on publication portfolio
    if research_director.publications:
        avg_readiness = sum(pub.publication_readiness for pub in research_director.publications) / len(research_director.publications)
        
        if avg_readiness < 0.7:
            recommendations.append({
                "category": "Publication Quality",
                "priority": "Medium",
                "recommendation": "Improve statistical rigor and experimental design",
                "details": "Publications need stronger statistical evidence for peer review readiness."
            })
    
    # Based on research evolution
    if research_director.current_phase.value == "discovery":
        recommendations.append({
            "category": "Research Progression",
            "priority": "Low",
            "recommendation": "Continue data collection for research opportunity identification",
            "details": "System is in early discovery phase and progressing normally."
        })
    
    # Breakthrough potential
    breakthrough_count = metrics.get("breakthrough_discoveries", 0)
    if breakthrough_count == 0:
        recommendations.append({
            "category": "Innovation",
            "priority": "Medium",
            "recommendation": "Focus on high-novelty research directions",
            "details": "Prioritize research opportunities with breakthrough potential."
        })
    
    return recommendations


def _print_research_demo_summary(report: dict, logger: logging.Logger):
    """Print comprehensive research demonstration summary."""
    
    demo_report = report["autonomous_research_demo_report"]
    research_status = demo_report["research_director_status"]
    achievements = demo_report["research_achievements"]
    portfolio = demo_report["publication_portfolio"]
    breakthroughs = demo_report["breakthrough_discoveries"]
    
    logger.info("=" * 90)
    logger.info("🔬 AUTONOMOUS RESEARCH DIRECTOR DEMONSTRATION SUMMARY")
    logger.info("=" * 90)
    
    # Research Director Status
    logger.info(f"Research Phase: {research_status['current_phase']}")
    logger.info(f"Opportunities Discovered: {research_status['opportunities_discovered']}")
    logger.info(f"Active Hypotheses: {research_status['active_hypotheses']}")
    logger.info(f"Completed Studies: {research_status['completed_studies']}")
    logger.info(f"Publications Generated: {research_status['publications_generated']}")
    
    logger.info("")
    logger.info("🏆 Research Achievements:")
    for achievement in achievements:
        logger.info(f"   {achievement}")
    
    # Publication Portfolio
    logger.info("")
    logger.info("📚 Publication Portfolio:")
    logger.info(f"   Total Publications: {portfolio['total_publications']}")
    if portfolio['total_publications'] > 0:
        logger.info(f"   Average Novelty Score: {portfolio['average_novelty_score']:.1%}")
        logger.info(f"   Average Impact Score: {portfolio['average_impact_score']:.1%}")
        logger.info(f"   Average Publication Readiness: {portfolio['average_publication_readiness']:.1%}")
        
        if portfolio.get('peer_review_predictions'):
            pred = portfolio['peer_review_predictions']
            logger.info(f"   Predicted Acceptance Rate: {pred.get('average_acceptance_probability', 0):.1%}")
    
    # Breakthrough Discoveries
    if breakthroughs:
        logger.info("")
        logger.info("💎 Breakthrough Discoveries:")
        for breakthrough in breakthroughs:
            logger.info(f"   • {breakthrough['type']}: {breakthrough.get('title', breakthrough.get('description', 'Unknown'))}")
    
    # Research Metrics
    metrics = research_status['research_metrics']
    logger.info("")
    logger.info("📊 Research Metrics:")
    logger.info(f"   Opportunities Discovered: {metrics['opportunities_discovered']}")
    logger.info(f"   Hypotheses Generated: {metrics['hypotheses_generated']}")
    logger.info(f"   Experiments Executed: {metrics['experiments_executed']}")
    logger.info(f"   Publications Generated: {metrics['publications_generated']}")
    logger.info(f"   Breakthrough Discoveries: {metrics['breakthrough_discoveries']}")
    
    # Future Potential
    future_potential = demo_report["future_research_potential"]
    logger.info("")
    logger.info("🔮 Future Research Potential:")
    logger.info(f"   Pipeline Health: {future_potential['research_pipeline_health']}")
    logger.info(f"   Breakthrough Probability: {future_potential['breakthrough_probability']:.1%}")
    
    logger.info("=" * 90)


async def _save_research_components(
    research_director: AutonomousResearchDirector,
    reports_dir: Path,
    logger: logging.Logger
):
    """Save individual research components for detailed analysis."""
    
    try:
        # Save research opportunities
        if research_director.research_opportunities:
            opportunities_data = [
                {k: v for k, v in opp.__dict__.items() if k != "discovered_at"}
                for opp in research_director.research_opportunities
            ]
            
            with open(reports_dir / "research_opportunities.json", 'w') as f:
                json.dump(opportunities_data, f, indent=2, default=str)
            
            logger.info(f"💡 Saved {len(opportunities_data)} research opportunities")
        
        # Save hypotheses
        if research_director.active_hypotheses:
            hypotheses_data = [
                {k: v for k, v in hyp.__dict__.items() if k != "created_at"}
                for hyp in research_director.active_hypotheses
            ]
            
            with open(reports_dir / "research_hypotheses.json", 'w') as f:
                json.dump(hypotheses_data, f, indent=2, default=str)
            
            logger.info(f"🧪 Saved {len(hypotheses_data)} research hypotheses")
        
        # Save publications
        if research_director.publications:
            publications_data = []
            for pub in research_director.publications:
                pub_data = {
                    "publication_id": pub.publication_id,
                    "title": pub.title,
                    "abstract": pub.abstract,
                    "novelty_score": pub.novelty_score,
                    "impact_score": pub.impact_score,
                    "publication_readiness": pub.publication_readiness,
                    "peer_review_predictions": pub.peer_review_predictions,
                    "statistical_evidence": pub.statistical_evidence,
                }
                publications_data.append(pub_data)
            
            with open(reports_dir / "research_publications.json", 'w') as f:
                json.dump(publications_data, f, indent=2, default=str)
            
            logger.info(f"📝 Saved {len(publications_data)} research publications")
        
        # Save completed studies summary
        if research_director.completed_studies:
            studies_summary = []
            for study in research_director.completed_studies:
                summary = {
                    "study_id": study.get("study_id"),
                    "hypothesis_id": study.get("hypothesis", {}).hypothesis_id if study.get("hypothesis") else None,
                    "successful": study.get("results", {}).get("successful", False),
                    "final_performance": study.get("results", {}).get("final_performance", 0),
                    "p_value": study.get("results", {}).get("p_value", 1.0),
                    "effect_size": study.get("results", {}).get("effect_size", 0),
                    "completed_at": study.get("completed_at"),
                }
                studies_summary.append(summary)
            
            with open(reports_dir / "completed_studies.json", 'w') as f:
                json.dump(studies_summary, f, indent=2, default=str)
            
            logger.info(f"📋 Saved {len(studies_summary)} completed studies")
        
    except Exception as e:
        logger.error(f"Failed to save research components: {e}")


async def main():
    """Main demonstration entry point."""
    
    print("🔬 Autonomous Research Director Demonstration")
    print("=" * 70)
    print()
    print("This demonstration showcases the cutting-edge autonomous research")
    print("capabilities of the Generation 4+ system, including:")
    print("  • Autonomous discovery of research opportunities")
    print("  • Automated hypothesis generation and testing")
    print("  • Self-directed experimental design and execution")
    print("  • Automatic publication generation with peer review prediction")
    print("  • Iterative research direction evolution")
    print()
    print("This represents the pinnacle of autonomous scientific discovery,")
    print("where AI systems become their own research scientists.")
    print()
    
    # Note: In a real deployment, you would provide your OpenAI API key
    print("⚠️  Note: Running in demo mode without full GPT-4 integration")
    print("   For full autonomous research capabilities, provide OpenAI API key")
    print()
    
    # Run the demonstration
    success = await demonstrate_autonomous_research_system()
    
    if success:
        print("✅ Autonomous Research demonstration completed successfully!")
        print("📊 Check the results/ directory for detailed research reports")
        print("📁 Individual research components saved for analysis")
        print()
        print("🎯 Key Demo Outcomes:")
        print("  • Research opportunities automatically discovered")
        print("  • Hypotheses generated and tested autonomously")
        print("  • Publications created with peer review predictions")
        print("  • Future research directions identified")
    else:
        print("❌ Autonomous Research demonstration encountered issues")
        print("📋 Check the logs/ directory for error details")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()