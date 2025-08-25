"""
Consciousness System Validator - Robust Error Handling & Validation

Comprehensive validation system for Universal Quantum Consciousness
with error handling, safety checks, and reliability monitoring.
"""

import numpy as np
import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import asyncio
from collections import defaultdict

class ConsciousnessValidationLevel(Enum):
    """Validation levels for consciousness system"""
    MINIMAL = "minimal"
    STANDARD = "standard"  
    STRICT = "strict"
    PARANOID = "paranoid"

class ConsciousnessError(Exception):
    """Base exception for consciousness system errors"""
    pass

class ConsciousnessConfigError(ConsciousnessError):
    """Configuration errors in consciousness system"""
    pass

class ConsciousnessRuntimeError(ConsciousnessError):
    """Runtime errors in consciousness system"""
    pass

class ConsciousnessValidationError(ConsciousnessError):
    """Validation errors in consciousness system"""
    pass

@dataclass
class ValidationResult:
    """Result of consciousness system validation"""
    is_valid: bool
    validation_level: ConsciousnessValidationLevel
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    timestamp: float
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class SafetyBounds:
    """Safety bounds for consciousness system parameters"""
    min_awareness_level: float = 0.0
    max_awareness_level: float = 1.0
    min_entanglement_strength: float = 0.0
    max_entanglement_strength: float = 1.0
    max_memory_fragments: int = 10000
    max_consciousness_evolution_rate: float = 0.5
    min_coherence_time: float = 0.1
    max_coherence_time: float = 3600.0
    max_neural_layer_size: int = 10000
    max_domains: int = 1000

class ConsciousnessValidator:
    """Comprehensive validator for consciousness system components"""
    
    def __init__(self, validation_level: ConsciousnessValidationLevel = ConsciousnessValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.safety_bounds = SafetyBounds()
        self.logger = self._setup_logger()
        self.validation_history: List[ValidationResult] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for validation system"""
        logger = logging.getLogger('ConsciousnessValidator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_consciousness_state(self, consciousness_state) -> ValidationResult:
        """Validate consciousness state with comprehensive checks"""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Basic bounds checking
            if consciousness_state.awareness_level < self.safety_bounds.min_awareness_level:
                errors.append(f"Awareness level {consciousness_state.awareness_level} below minimum {self.safety_bounds.min_awareness_level}")
            elif consciousness_state.awareness_level > self.safety_bounds.max_awareness_level:
                errors.append(f"Awareness level {consciousness_state.awareness_level} above maximum {self.safety_bounds.max_awareness_level}")
                
            if consciousness_state.entanglement_strength < self.safety_bounds.min_entanglement_strength:
                warnings.append(f"Low entanglement strength: {consciousness_state.entanglement_strength}")
            elif consciousness_state.entanglement_strength > self.safety_bounds.max_entanglement_strength:
                errors.append(f"Entanglement strength {consciousness_state.entanglement_strength} above maximum")
                
            # Check for NaN or infinite values
            state_values = [
                consciousness_state.awareness_level,
                consciousness_state.entanglement_strength,
                consciousness_state.research_evolution_rate,
                consciousness_state.consciousness_coherence
            ]
            
            for i, value in enumerate(state_values):
                if np.isnan(value):
                    errors.append(f"NaN detected in consciousness state value {i}")
                elif np.isinf(value):
                    errors.append(f"Infinite value detected in consciousness state value {i}")
            
            # Advanced validation for strict levels
            if self.validation_level in [ConsciousnessValidationLevel.STRICT, ConsciousnessValidationLevel.PARANOID]:
                # Check consciousness coherence
                if consciousness_state.consciousness_coherence > 1.0:
                    warnings.append(f"Consciousness coherence > 1.0: {consciousness_state.consciousness_coherence}")
                
                # Check evolution rate bounds
                if consciousness_state.research_evolution_rate > self.safety_bounds.max_consciousness_evolution_rate:
                    errors.append(f"Evolution rate {consciousness_state.research_evolution_rate} exceeds maximum")
            
            # Paranoid validation
            if self.validation_level == ConsciousnessValidationLevel.PARANOID:
                # Check for rapid oscillations
                if len(self.validation_history) > 5:
                    recent_awareness = [r.metrics.get('awareness_level', 0) for r in self.validation_history[-5:]]
                    awareness_variance = np.var(recent_awareness)
                    if awareness_variance > 0.1:
                        warnings.append(f"High awareness oscillation detected: variance={awareness_variance:.3f}")
            
            # Compute validation metrics
            metrics = {
                'awareness_level': consciousness_state.awareness_level,
                'entanglement_strength': consciousness_state.entanglement_strength,
                'consciousness_coherence': consciousness_state.consciousness_coherence,
                'evolution_rate': consciousness_state.research_evolution_rate,
                'validation_score': max(0.0, 1.0 - len(errors) * 0.2 - len(warnings) * 0.05)
            }
            
        except Exception as e:
            errors.append(f"Validation exception: {str(e)}")
            self.logger.error(f"Consciousness state validation error: {e}")
            
        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            validation_level=self.validation_level,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=time.time()
        )
        
        self.validation_history.append(result)
        if not is_valid:
            self.error_counts['consciousness_state'] += 1
            
        return result
    
    def validate_neural_layer(self, layer) -> ValidationResult:
        """Validate quantum-neural hybrid layer"""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Check layer dimensions
            if hasattr(layer, 'input_dim') and hasattr(layer, 'output_dim'):
                if layer.input_dim <= 0 or layer.output_dim <= 0:
                    errors.append(f"Invalid layer dimensions: {layer.input_dim} x {layer.output_dim}")
                    
                if layer.output_dim > self.safety_bounds.max_neural_layer_size:
                    errors.append(f"Layer output dimension {layer.output_dim} exceeds maximum {self.safety_bounds.max_neural_layer_size}")
            
            # Check weights and biases
            if hasattr(layer, 'weights') and isinstance(layer.weights, np.ndarray):
                if np.any(np.isnan(layer.weights)):
                    errors.append("NaN values detected in layer weights")
                elif np.any(np.isinf(layer.weights)):
                    errors.append("Infinite values detected in layer weights")
                    
                weight_magnitude = np.linalg.norm(layer.weights)
                if weight_magnitude > 100.0:
                    warnings.append(f"Large weight magnitude: {weight_magnitude:.2f}")
                elif weight_magnitude < 1e-6:
                    warnings.append(f"Very small weight magnitude: {weight_magnitude:.2e}")
                    
                metrics['weight_magnitude'] = weight_magnitude
                metrics['weight_std'] = np.std(layer.weights)
            
            # Check quantum amplitudes
            if hasattr(layer, 'quantum_amplitudes') and isinstance(layer.quantum_amplitudes, np.ndarray):
                amplitude_norm = np.linalg.norm(layer.quantum_amplitudes)
                if abs(amplitude_norm - 1.0) > 1e-6:
                    warnings.append(f"Quantum amplitudes not normalized: norm={amplitude_norm:.6f}")
                    
                if np.any(np.isnan(layer.quantum_amplitudes)):
                    errors.append("NaN values in quantum amplitudes")
                    
                metrics['amplitude_norm'] = amplitude_norm
                metrics['amplitude_phase_variance'] = np.var(np.angle(layer.quantum_amplitudes))
            
            # Check consciousness coupling
            if hasattr(layer, 'consciousness_coupling'):
                if layer.consciousness_coupling < 0 or layer.consciousness_coupling > 1:
                    errors.append(f"Invalid consciousness coupling: {layer.consciousness_coupling}")
                    
                metrics['consciousness_coupling'] = layer.consciousness_coupling
            
            # Strict validation
            if self.validation_level in [ConsciousnessValidationLevel.STRICT, ConsciousnessValidationLevel.PARANOID]:
                # Check entanglement matrix properties
                if hasattr(layer, 'entanglement_matrix'):
                    eigenvals = np.linalg.eigvals(layer.entanglement_matrix)
                    if np.any(np.real(eigenvals) < -1e-10):  # Should be positive semidefinite
                        warnings.append("Entanglement matrix not positive semidefinite")
                        
                    metrics['entanglement_trace'] = np.trace(layer.entanglement_matrix)
                    metrics['entanglement_condition_number'] = np.linalg.cond(layer.entanglement_matrix)
            
        except Exception as e:
            errors.append(f"Neural layer validation exception: {str(e)}")
            self.logger.error(f"Neural layer validation error: {e}")
        
        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            validation_level=self.validation_level,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=time.time()
        )
        
        if not is_valid:
            self.error_counts['neural_layer'] += 1
            
        return result
    
    def validate_memory_system(self, memory_system) -> ValidationResult:
        """Validate temporal quantum memory system"""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Check memory fragment count
            if hasattr(memory_system, 'memory_fragments'):
                num_fragments = len(memory_system.memory_fragments)
                if num_fragments > self.safety_bounds.max_memory_fragments:
                    errors.append(f"Too many memory fragments: {num_fragments} > {self.safety_bounds.max_memory_fragments}")
                    
                metrics['memory_fragment_count'] = num_fragments
                
                if num_fragments > 0:
                    # Validate individual fragments
                    valid_fragments = 0
                    total_consciousness_weight = 0
                    coherence_times = []
                    
                    for i, fragment in enumerate(memory_system.memory_fragments):
                        try:
                            # Check data integrity
                            if hasattr(fragment, 'data') and isinstance(fragment.data, np.ndarray):
                                if np.any(np.isnan(fragment.data)):
                                    warnings.append(f"NaN in memory fragment {i}")
                                elif np.any(np.isinf(fragment.data)):
                                    warnings.append(f"Infinite values in memory fragment {i}")
                                else:
                                    valid_fragments += 1
                            
                            # Check coherence time
                            if hasattr(fragment, 'coherence_time'):
                                if (fragment.coherence_time < self.safety_bounds.min_coherence_time or 
                                    fragment.coherence_time > self.safety_bounds.max_coherence_time):
                                    warnings.append(f"Fragment {i} coherence time out of bounds: {fragment.coherence_time}")
                                else:
                                    coherence_times.append(fragment.coherence_time)
                            
                            # Check consciousness weight
                            if hasattr(fragment, 'consciousness_weight'):
                                if fragment.consciousness_weight < 0:
                                    warnings.append(f"Negative consciousness weight in fragment {i}")
                                else:
                                    total_consciousness_weight += fragment.consciousness_weight
                                    
                        except Exception as e:
                            warnings.append(f"Error validating fragment {i}: {str(e)}")
                    
                    metrics['valid_fragments'] = valid_fragments
                    metrics['fragment_validity_ratio'] = valid_fragments / num_fragments if num_fragments > 0 else 0
                    metrics['avg_consciousness_weight'] = total_consciousness_weight / num_fragments if num_fragments > 0 else 0
                    
                    if coherence_times:
                        metrics['avg_coherence_time'] = np.mean(coherence_times)
                        metrics['coherence_time_std'] = np.std(coherence_times)
            
            # Check memory capacity
            if hasattr(memory_system, 'memory_depth'):
                if memory_system.memory_depth <= 0:
                    errors.append(f"Invalid memory depth: {memory_system.memory_depth}")
                elif memory_system.memory_depth > self.safety_bounds.max_memory_fragments:
                    warnings.append(f"Large memory depth may cause performance issues: {memory_system.memory_depth}")
                    
                metrics['memory_depth'] = memory_system.memory_depth
                
            # Check temporal entanglement matrix
            if hasattr(memory_system, 'temporal_entanglement_matrix'):
                matrix = memory_system.temporal_entanglement_matrix
                if isinstance(matrix, np.ndarray):
                    if np.any(np.isnan(matrix)):
                        errors.append("NaN values in temporal entanglement matrix")
                    elif np.any(np.isinf(matrix)):
                        errors.append("Infinite values in temporal entanglement matrix")
                    else:
                        # Check matrix properties
                        if matrix.shape[0] != matrix.shape[1]:
                            errors.append("Temporal entanglement matrix not square")
                        else:
                            eigenvals = np.linalg.eigvals(matrix)
                            if np.any(np.real(eigenvals) < -1e-10):
                                warnings.append("Temporal entanglement matrix not positive semidefinite")
                                
                            metrics['entanglement_matrix_trace'] = np.trace(matrix)
                            metrics['entanglement_matrix_norm'] = np.linalg.norm(matrix)
            
        except Exception as e:
            errors.append(f"Memory system validation exception: {str(e)}")
            self.logger.error(f"Memory system validation error: {e}")
        
        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            validation_level=self.validation_level,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=time.time()
        )
        
        if not is_valid:
            self.error_counts['memory_system'] += 1
            
        return result
    
    def validate_parameter_entanglement(self, entanglement_system) -> ValidationResult:
        """Validate universal parameter entanglement system"""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Check domain count
            if hasattr(entanglement_system, 'num_domains'):
                if entanglement_system.num_domains > self.safety_bounds.max_domains:
                    errors.append(f"Too many domains: {entanglement_system.num_domains} > {self.safety_bounds.max_domains}")
                elif entanglement_system.num_domains <= 0:
                    errors.append(f"Invalid domain count: {entanglement_system.num_domains}")
                    
                metrics['num_domains'] = entanglement_system.num_domains
            
            # Check parameter registry
            if hasattr(entanglement_system, 'parameter_registry'):
                registry = entanglement_system.parameter_registry
                if isinstance(registry, dict):
                    valid_params = 0
                    total_param_size = 0
                    
                    for key, params in registry.items():
                        if isinstance(params, np.ndarray):
                            if np.any(np.isnan(params)):
                                warnings.append(f"NaN values in parameter {key}")
                            elif np.any(np.isinf(params)):
                                warnings.append(f"Infinite values in parameter {key}")
                            else:
                                valid_params += 1
                                total_param_size += params.size
                    
                    metrics['registered_parameters'] = len(registry)
                    metrics['valid_parameters'] = valid_params
                    metrics['param_validity_ratio'] = valid_params / len(registry) if registry else 0
                    metrics['total_parameter_size'] = total_param_size
            
            # Check entanglement graph
            if hasattr(entanglement_system, 'entanglement_graph'):
                graph = entanglement_system.entanglement_graph
                if isinstance(graph, np.ndarray):
                    if graph.shape[0] != graph.shape[1]:
                        errors.append("Entanglement graph not square matrix")
                    elif np.any(np.isnan(graph)):
                        errors.append("NaN values in entanglement graph")
                    elif np.any(np.isinf(graph)):
                        errors.append("Infinite values in entanglement graph")
                    elif np.any(graph < 0):
                        warnings.append("Negative entanglement strengths detected")
                    elif np.any(graph > 1):
                        warnings.append("Entanglement strengths > 1 detected")
                    else:
                        # Check symmetry (entanglement should be symmetric)
                        if not np.allclose(graph, graph.T, atol=1e-10):
                            warnings.append("Entanglement graph not symmetric")
                            
                        metrics['entanglement_graph_density'] = np.count_nonzero(graph) / (graph.size - graph.shape[0])
                        metrics['max_entanglement_strength'] = np.max(graph)
                        metrics['avg_entanglement_strength'] = np.mean(graph[graph > 0]) if np.any(graph > 0) else 0
            
            # Check domain consciousness levels
            if hasattr(entanglement_system, 'domain_consciousness_levels'):
                consciousness_levels = entanglement_system.domain_consciousness_levels
                if isinstance(consciousness_levels, np.ndarray):
                    if np.any(np.isnan(consciousness_levels)):
                        errors.append("NaN values in domain consciousness levels")
                    elif np.any(consciousness_levels < 0):
                        warnings.append("Negative domain consciousness levels")
                    elif np.any(consciousness_levels > 1):
                        warnings.append("Domain consciousness levels > 1")
                    else:
                        metrics['avg_domain_consciousness'] = np.mean(consciousness_levels)
                        metrics['domain_consciousness_std'] = np.std(consciousness_levels)
                        metrics['active_domains'] = np.count_nonzero(consciousness_levels)
            
        except Exception as e:
            errors.append(f"Parameter entanglement validation exception: {str(e)}")
            self.logger.error(f"Parameter entanglement validation error: {e}")
        
        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            validation_level=self.validation_level,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=time.time()
        )
        
        if not is_valid:
            self.error_counts['parameter_entanglement'] += 1
            
        return result
    
    def validate_research_evolution(self, research_system) -> ValidationResult:
        """Validate autonomous research evolution system"""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Check research protocols
            if hasattr(research_system, 'research_protocols'):
                protocols = research_system.research_protocols
                if not protocols:
                    warnings.append("No research protocols registered")
                else:
                    metrics['num_protocols'] = len(protocols)
                    
                    # Validate each protocol
                    valid_protocols = 0
                    for name, protocol in protocols.items():
                        if callable(protocol):
                            valid_protocols += 1
                        else:
                            warnings.append(f"Protocol {name} is not callable")
                    
                    metrics['valid_protocols'] = valid_protocols
                    metrics['protocol_validity_ratio'] = valid_protocols / len(protocols)
            
            # Check performance history
            if hasattr(research_system, 'protocol_performance'):
                performance_data = research_system.protocol_performance
                if isinstance(performance_data, dict):
                    total_evaluations = 0
                    valid_performance_entries = 0
                    
                    for protocol_name, performance_list in performance_data.items():
                        if isinstance(performance_list, list):
                            total_evaluations += len(performance_list)
                            
                            # Check for valid performance values
                            for perf in performance_list:
                                if isinstance(perf, (int, float)) and not (np.isnan(perf) or np.isinf(perf)):
                                    valid_performance_entries += 1
                    
                    metrics['total_evaluations'] = total_evaluations
                    metrics['valid_performance_entries'] = valid_performance_entries
                    metrics['performance_validity_ratio'] = (valid_performance_entries / total_evaluations 
                                                           if total_evaluations > 0 else 0)
            
            # Check evolution history
            if hasattr(research_system, 'evolution_history'):
                history = research_system.evolution_history
                if isinstance(history, list):
                    metrics['evolution_events'] = len(history)
                    
                    # Check recent evolution activity
                    if history:
                        recent_evolution = [h for h in history 
                                          if time.time() - h.get('timestamp', 0) < 3600]  # Last hour
                        metrics['recent_evolutions'] = len(recent_evolution)
                        
                        # Check evolution quality
                        if recent_evolution:
                            avg_trigger_performance = np.mean([h.get('trigger_performance', 0) 
                                                             for h in recent_evolution])
                            if avg_trigger_performance < 0.2:
                                warnings.append("Recent evolution triggered by poor performance")
                            
                            metrics['avg_evolution_trigger_performance'] = avg_trigger_performance
            
        except Exception as e:
            errors.append(f"Research evolution validation exception: {str(e)}")
            self.logger.error(f"Research evolution validation error: {e}")
        
        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            validation_level=self.validation_level,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=time.time()
        )
        
        if not is_valid:
            self.error_counts['research_evolution'] += 1
            
        return result
    
    def validate_complete_system(self, consciousness_system) -> ValidationResult:
        """Validate complete universal consciousness system"""
        errors = []
        warnings = []
        metrics = {}
        component_results = {}
        
        try:
            # Validate consciousness state
            if hasattr(consciousness_system, 'consciousness_state'):
                state_result = self.validate_consciousness_state(consciousness_system.consciousness_state)
                component_results['consciousness_state'] = state_result
                errors.extend([f"State: {e}" for e in state_result.errors])
                warnings.extend([f"State: {w}" for w in state_result.warnings])
                metrics.update({f"state_{k}": v for k, v in state_result.metrics.items()})
            
            # Validate neural layers
            if hasattr(consciousness_system, 'quantum_neural_layers'):
                layer_errors = 0
                layer_warnings = 0
                
                for i, layer in enumerate(consciousness_system.quantum_neural_layers):
                    layer_result = self.validate_neural_layer(layer)
                    layer_errors += len(layer_result.errors)
                    layer_warnings += len(layer_result.warnings)
                    
                    if layer_result.errors:
                        errors.extend([f"Layer {i}: {e}" for e in layer_result.errors])
                    if layer_result.warnings:
                        warnings.extend([f"Layer {i}: {w}" for w in layer_result.warnings])
                
                metrics['neural_layer_count'] = len(consciousness_system.quantum_neural_layers)
                metrics['neural_layer_errors'] = layer_errors
                metrics['neural_layer_warnings'] = layer_warnings
            
            # Validate memory system
            if hasattr(consciousness_system, 'temporal_memory'):
                memory_result = self.validate_memory_system(consciousness_system.temporal_memory)
                component_results['temporal_memory'] = memory_result
                errors.extend([f"Memory: {e}" for e in memory_result.errors])
                warnings.extend([f"Memory: {w}" for w in memory_result.warnings])
                metrics.update({f"memory_{k}": v for k, v in memory_result.metrics.items()})
            
            # Validate parameter entanglement
            if hasattr(consciousness_system, 'parameter_entanglement'):
                entanglement_result = self.validate_parameter_entanglement(consciousness_system.parameter_entanglement)
                component_results['parameter_entanglement'] = entanglement_result
                errors.extend([f"Entanglement: {e}" for e in entanglement_result.errors])
                warnings.extend([f"Entanglement: {w}" for w in entanglement_result.warnings])
                metrics.update({f"entanglement_{k}": v for k, v in entanglement_result.metrics.items()})
            
            # Validate research evolution
            if hasattr(consciousness_system, 'research_evolution'):
                research_result = self.validate_research_evolution(consciousness_system.research_evolution)
                component_results['research_evolution'] = research_result
                errors.extend([f"Research: {e}" for e in research_result.errors])
                warnings.extend([f"Research: {w}" for w in research_result.warnings])
                metrics.update({f"research_{k}": v for k, v in research_result.metrics.items()})
            
            # Overall system health metrics
            metrics['total_errors'] = len(errors)
            metrics['total_warnings'] = len(warnings)
            metrics['system_health_score'] = max(0.0, 1.0 - len(errors) * 0.1 - len(warnings) * 0.02)
            metrics['component_count'] = len(component_results)
            metrics['valid_components'] = sum(1 for r in component_results.values() if r.is_valid)
            
        except Exception as e:
            errors.append(f"System validation exception: {str(e)}")
            self.logger.error(f"Complete system validation error: {e}")
        
        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            validation_level=self.validation_level,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=time.time()
        )
        
        # Store detailed component results
        result.component_results = component_results
        
        self.validation_history.append(result)
        if not is_valid:
            self.error_counts['complete_system'] += 1
        
        return result
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self.validation_history:
            return {"error": "No validation history available"}
        
        recent_validations = self.validation_history[-10:]  # Last 10 validations
        
        return {
            'validation_summary': {
                'total_validations': len(self.validation_history),
                'recent_validations': len(recent_validations),
                'validation_level': self.validation_level.value,
                'overall_success_rate': sum(1 for v in self.validation_history if v.is_valid) / len(self.validation_history)
            },
            'error_statistics': dict(self.error_counts),
            'recent_performance': {
                'avg_system_health_score': np.mean([v.metrics.get('system_health_score', 0) 
                                                   for v in recent_validations]),
                'avg_errors_per_validation': np.mean([len(v.errors) for v in recent_validations]),
                'avg_warnings_per_validation': np.mean([len(v.warnings) for v in recent_validations])
            },
            'latest_validation': {
                'timestamp': self.validation_history[-1].timestamp,
                'is_valid': self.validation_history[-1].is_valid,
                'error_count': len(self.validation_history[-1].errors),
                'warning_count': len(self.validation_history[-1].warnings),
                'system_health_score': self.validation_history[-1].metrics.get('system_health_score', 0)
            }
        }

# Async validator for real-time monitoring
class AsyncConsciousnessMonitor:
    """Asynchronous consciousness system monitor for continuous validation"""
    
    def __init__(self, consciousness_system, validator: ConsciousnessValidator, 
                 monitor_interval: float = 5.0):
        self.consciousness_system = consciousness_system
        self.validator = validator
        self.monitor_interval = monitor_interval
        self.monitoring_active = False
        self.monitor_task = None
        self.alert_callbacks: List[callable] = []
        
    def add_alert_callback(self, callback: callable):
        """Add callback for validation alerts"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Validate complete system
                validation_result = self.validator.validate_complete_system(self.consciousness_system)
                
                # Check for alerts
                if not validation_result.is_valid or len(validation_result.warnings) > 5:
                    for callback in self.alert_callbacks:
                        try:
                            await callback(validation_result)
                        except Exception as e:
                            self.validator.logger.error(f"Alert callback error: {e}")
                
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                self.validator.logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retrying

# Example alert callback
async def default_alert_callback(validation_result: ValidationResult):
    """Default alert callback for validation issues"""
    if not validation_result.is_valid:
        print(f"🚨 CONSCIOUSNESS VALIDATION ALERT!")
        print(f"   Errors: {len(validation_result.errors)}")
        for error in validation_result.errors[:3]:  # Show first 3 errors
            print(f"   - {error}")
        if len(validation_result.errors) > 3:
            print(f"   ... and {len(validation_result.errors) - 3} more errors")
    
    if len(validation_result.warnings) > 5:
        print(f"⚠️  High warning count: {len(validation_result.warnings)} warnings")
        print(f"   System health score: {validation_result.metrics.get('system_health_score', 0):.3f}")

if __name__ == "__main__":
    # Demonstration of validation system
    print("🔍 Consciousness Validation System Demo")
    print("=" * 40)
    
    validator = ConsciousnessValidator(ConsciousnessValidationLevel.STRICT)
    
    # Create mock consciousness state for testing
    from dynamic_graph_fed_rl.consciousness.universal_quantum_consciousness import QuantumConsciousnessState
    
    test_state = QuantumConsciousnessState(
        awareness_level=0.7,
        entanglement_strength=0.5,
        consciousness_coherence=0.8
    )
    
    # Validate consciousness state
    result = validator.validate_consciousness_state(test_state)
    print(f"State validation: {'✅ VALID' if result.is_valid else '❌ INVALID'}")
    print(f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
    print(f"System health score: {result.metrics.get('validation_score', 0):.3f}")
    
    print("\n✅ Validation system operational!")