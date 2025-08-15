"""
Enterprise-grade error handling framework with circuit breaker patterns and resilience mechanisms.

This module provides comprehensive error handling, circuit breakers, retry mechanisms,
and fault tolerance patterns for building robust distributed systems.
"""

import asyncio
import functools
import logging
import random
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import traceback
from concurrent.futures import ThreadPoolExecutor
import hashlib


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    timestamp: float
    error_type: str
    error_message: str
    component: str
    operation: str
    severity: ErrorSeverity
    stack_trace: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.0
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Tuple[Type[Exception], ...] = (Exception,)
    success_threshold: int = 3  # For half-open -> closed transition
    request_volume_threshold: int = 10  # Minimum requests before opening


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted."""
    pass


class RateLimitExceededException(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class ValidationError(Exception):
    """Exception raised for input validation failures."""
    pass


class SecurityError(Exception):
    """Exception raised for security-related issues."""
    pass


class ErrorAggregator:
    """Aggregates and analyzes error patterns for system insights."""
    
    def __init__(self, max_errors: int = 10000):
        self.max_errors = max_errors
        self.errors: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def record_error(self, error_context: ErrorContext) -> None:
        """Record an error occurrence."""
        with self.lock:
            self.errors.append(error_context)
            
            # Maintain size limit
            if len(self.errors) > self.max_errors:
                self.errors = self.errors[-self.max_errors//2:]
            
            # Update counts
            error_key = f"{error_context.component}:{error_context.error_type}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def get_error_patterns(self, time_window: float = 3600) -> Dict[str, Any]:
        """Analyze error patterns within time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self.lock:
            recent_errors = [e for e in self.errors if e.timestamp >= cutoff_time]
        
        if not recent_errors:
            return {"total_errors": 0, "patterns": {}}
        
        # Analyze patterns
        component_errors = {}
        severity_distribution = {}
        error_types = {}
        
        for error in recent_errors:
            # Component analysis
            component_errors[error.component] = component_errors.get(error.component, 0) + 1
            
            # Severity analysis
            severity_distribution[error.severity.value] = severity_distribution.get(error.severity.value, 0) + 1
            
            # Error type analysis
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "time_window": time_window,
            "component_errors": component_errors,
            "severity_distribution": severity_distribution,
            "error_types": error_types,
            "error_rate": len(recent_errors) / time_window * 60,  # errors per minute
            "most_common_error": max(error_types.items(), key=lambda x: x[1]) if error_types else None
        }


class CircuitBreaker:
    """
    Robust circuit breaker implementation with configurable thresholds and recovery logic.
    
    Features:
    - Configurable failure thresholds
    - Automatic recovery testing
    - Metrics collection
    - Thread-safe operation
    """
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.request_count = 0
        self.lock = threading.Lock()
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_transitions = []
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            self.total_requests += 1
            self.request_count += 1
            
            # Check if circuit should be opened
            if self.state == CircuitState.CLOSED and self._should_open_circuit():
                self._transition_to_open()
            
            # Handle open circuit
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        
        except self.config.expected_exception as e:
            self._record_failure()
            raise
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened based on failure rate."""
        if self.request_count < self.config.request_volume_threshold:
            return False
        
        return self.failure_count >= self.config.failure_threshold
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from open state."""
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        self._record_state_transition(CircuitState.OPEN)
        logging.warning(f"Circuit breaker '{self.name}' opened due to failures")
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self._record_state_transition(CircuitState.HALF_OPEN)
        logging.info(f"Circuit breaker '{self.name}' half-opened for testing")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.request_count = 0
        self._record_state_transition(CircuitState.CLOSED)
        logging.info(f"Circuit breaker '{self.name}' closed - service recovered")
    
    def _record_success(self) -> None:
        """Record successful operation."""
        with self.lock:
            self.total_successes += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self) -> None:
        """Record failed operation."""
        with self.lock:
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _record_state_transition(self, new_state: CircuitState) -> None:
        """Record state transition for metrics."""
        self.state_transitions.append({
            "timestamp": time.time(),
            "from_state": self.state.value if hasattr(self, 'state') else None,
            "to_state": new_state.value
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self.lock:
            success_rate = (self.total_successes / self.total_requests * 100) if self.total_requests > 0 else 0
            
            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.total_requests,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "success_rate": success_rate,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "state_transitions": len(self.state_transitions),
                "last_failure_time": self.last_failure_time
            }


class RetryHandler:
    """Advanced retry mechanism with exponential backoff and adaptive intelligence."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.failure_patterns: Dict[str, List[float]] = {}
        self.success_patterns: Dict[str, List[float]] = {}
        self.adaptive_enabled = True
        self.learning_rate = 0.1
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for retry handler."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with intelligent adaptive retry logic."""
        func_signature = f"{func.__name__}_{hash(str(args))}"
        last_exception = None
        attempt_times = []
        
        # Get adaptive configuration based on historical patterns
        if self.adaptive_enabled:
            adaptive_config = self._get_adaptive_config(func_signature)
        else:
            adaptive_config = self.config
        
        for attempt in range(adaptive_config.max_attempts):
            attempt_start = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful execution pattern
                execution_time = time.time() - attempt_start
                self._record_success(func_signature, attempt, execution_time)
                
                return result
            
            except self.config.non_retryable_exceptions as e:
                # Don't retry for non-retryable exceptions
                execution_time = time.time() - attempt_start
                self._record_failure(func_signature, attempt, execution_time, type(e).__name__)
                raise
            
            except self.config.retryable_exceptions as e:
                last_exception = e
                execution_time = time.time() - attempt_start
                attempt_times.append(execution_time)
                
                # Record failure pattern
                self._record_failure(func_signature, attempt, execution_time, type(e).__name__)
                
                # Don't sleep on last attempt
                if attempt < adaptive_config.max_attempts - 1:
                    if self.adaptive_enabled:
                        delay = self._calculate_adaptive_delay(attempt, func_signature, e)
                    else:
                        delay = self._calculate_delay(attempt)
                    
                    time.sleep(delay)
                    
                    logging.warning(
                        f"Intelligent retry {attempt + 1}/{adaptive_config.max_attempts} "
                        f"after {delay:.2f}s delay: {str(e)} "
                        f"{'(adaptive)' if self.adaptive_enabled else ''}"
                    )
        
        # All retries exhausted
        raise RetryExhaustedException(
            f"All {adaptive_config.max_attempts} intelligent retry attempts failed. "
            f"Last error: {str(last_exception)} "
            f"(Adaptive config used: max_attempts={adaptive_config.max_attempts})"
        ) from last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff and jitter."""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay *= self.config.backoff_factor
        
        # Apply maximum delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter = delay * 0.1 * random.random()
            delay += jitter
        
        return delay
    
    def _calculate_adaptive_delay(self, attempt: int, func_signature: str, exception: Exception) -> float:
        """Calculate adaptive delay based on historical failure patterns."""
        base_delay = self._calculate_delay(attempt)
        
        if not self.adaptive_enabled or func_signature not in self.failure_patterns:
            return base_delay
        
        # Analyze historical failure patterns
        failure_history = self.failure_patterns[func_signature]
        if len(failure_history) < 3:
            return base_delay
        
        # Calculate adaptive adjustments
        recent_failures = failure_history[-10:]  # Last 10 failures
        avg_failure_time = sum(recent_failures) / len(recent_failures)
        
        # Adjust delay based on exception type and timing patterns
        exception_type = type(exception).__name__
        
        # If failures are happening quickly, increase delay more aggressively
        if avg_failure_time < 1.0:  # Fast failures
            base_delay *= 1.5
        
        # Exception-specific adjustments
        exception_multipliers = {
            'TimeoutError': 2.0,      # Wait longer for timeouts
            'ConnectionError': 1.8,   # Network issues need more time
            'MemoryError': 3.0,       # Memory issues need substantial delay
            'DatabaseError': 2.5,     # Database issues often need longer recovery
            'RateLimitError': 5.0,    # Rate limits need much longer delays
        }
        
        multiplier = exception_multipliers.get(exception_type, 1.0)
        adaptive_delay = base_delay * multiplier
        
        # Apply learning rate for gradual adjustment
        if hasattr(self, '_last_delays') and func_signature in self._last_delays:
            last_delay = self._last_delays[func_signature]
            adaptive_delay = last_delay + self.learning_rate * (adaptive_delay - last_delay)
        
        # Store for next iteration
        if not hasattr(self, '_last_delays'):
            self._last_delays = {}
        self._last_delays[func_signature] = adaptive_delay
        
        return min(adaptive_delay, self.config.max_delay)
    
    def _get_adaptive_config(self, func_signature: str) -> RetryConfig:
        """Get adaptive retry configuration based on historical success/failure patterns."""
        if func_signature not in self.failure_patterns:
            return self.config
        
        failure_history = self.failure_patterns[func_signature]
        success_history = self.success_patterns.get(func_signature, [])
        
        # Calculate success rate
        total_attempts = len(failure_history) + len(success_history)
        success_rate = len(success_history) / total_attempts if total_attempts > 0 else 0.0
        
        # Adaptive configuration based on patterns
        adaptive_config = RetryConfig(
            max_attempts=self.config.max_attempts,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            exponential_base=self.config.exponential_base,
            jitter=self.config.jitter,
            backoff_factor=self.config.backoff_factor,
            retryable_exceptions=self.config.retryable_exceptions,
            non_retryable_exceptions=self.config.non_retryable_exceptions
        )
        
        # Adjust max attempts based on success rate
        if success_rate < 0.3:  # Low success rate
            adaptive_config.max_attempts = min(self.config.max_attempts + 2, 10)
        elif success_rate > 0.8:  # High success rate
            adaptive_config.max_attempts = max(self.config.max_attempts - 1, 1)
        
        # Adjust delays based on failure frequency
        recent_failures = failure_history[-5:] if len(failure_history) >= 5 else failure_history
        if recent_failures:
            avg_failure_time = sum(recent_failures) / len(recent_failures)
            
            # If failures are very quick, increase base delay
            if avg_failure_time < 0.5:
                adaptive_config.base_delay = self.config.base_delay * 2.0
            elif avg_failure_time > 5.0:
                adaptive_config.base_delay = self.config.base_delay * 0.5
        
        return adaptive_config
    
    def _record_success(self, func_signature: str, attempt: int, execution_time: float):
        """Record successful execution for pattern analysis."""
        if func_signature not in self.success_patterns:
            self.success_patterns[func_signature] = []
        
        self.success_patterns[func_signature].append(execution_time)
        
        # Keep history bounded
        if len(self.success_patterns[func_signature]) > 100:
            self.success_patterns[func_signature] = self.success_patterns[func_signature][-50:]
    
    def _record_failure(self, func_signature: str, attempt: int, execution_time: float, exception_type: str):
        """Record failure for pattern analysis."""
        if func_signature not in self.failure_patterns:
            self.failure_patterns[func_signature] = []
        
        self.failure_patterns[func_signature].append(execution_time)
        
        # Keep history bounded
        if len(self.failure_patterns[func_signature]) > 100:
            self.failure_patterns[func_signature] = self.failure_patterns[func_signature][-50:]
        
        # Record exception type patterns
        if not hasattr(self, 'exception_patterns'):
            self.exception_patterns = {}
        
        if func_signature not in self.exception_patterns:
            self.exception_patterns[func_signature] = {}
        
        self.exception_patterns[func_signature][exception_type] = \
            self.exception_patterns[func_signature].get(exception_type, 0) + 1
    
    def get_retry_analytics(self) -> Dict[str, Any]:
        """Get comprehensive retry analytics and patterns."""
        analytics = {
            'adaptive_enabled': self.adaptive_enabled,
            'learning_rate': self.learning_rate,
            'tracked_functions': len(self.failure_patterns),
            'function_analytics': {}
        }
        
        for func_signature in self.failure_patterns:
            failures = len(self.failure_patterns[func_signature])
            successes = len(self.success_patterns.get(func_signature, []))
            total = failures + successes
            
            success_rate = successes / total if total > 0 else 0.0
            
            func_analytics = {
                'total_attempts': total,
                'success_rate': success_rate,
                'failure_count': failures,
                'success_count': successes
            }
            
            # Add exception type distribution
            if hasattr(self, 'exception_patterns') and func_signature in self.exception_patterns:
                func_analytics['exception_types'] = self.exception_patterns[func_signature]
            
            analytics['function_analytics'][func_signature] = func_analytics
        
        return analytics


class BulkheadIsolation:
    """Bulkhead pattern for resource isolation and fault containment."""
    
    def __init__(self, pool_size: int = 10, queue_size: int = 100, timeout: float = 30.0):
        self.pool_size = pool_size
        self.queue_size = queue_size
        self.timeout = timeout
        
        self.executor = ThreadPoolExecutor(
            max_workers=pool_size,
            thread_name_prefix="Bulkhead"
        )
        self._active_tasks = 0
        self._rejected_tasks = 0
        self.lock = threading.Lock()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in isolated thread pool."""
        with self.lock:
            if self._active_tasks >= self.queue_size:
                self._rejected_tasks += 1
                raise RateLimitExceededException(
                    f"Bulkhead capacity exceeded. Active: {self._active_tasks}, "
                    f"Queue: {self.queue_size}, Rejected: {self._rejected_tasks}"
                )
            
            self._active_tasks += 1
        
        try:
            future = self.executor.submit(func, *args, **kwargs)
            result = future.result(timeout=self.timeout)
            return result
        
        except Exception as e:
            raise
        
        finally:
            with self.lock:
                self._active_tasks -= 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        return {
            "pool_size": self.pool_size,
            "queue_size": self.queue_size,
            "active_tasks": self._active_tasks,
            "rejected_tasks": self._rejected_tasks,
            "utilization": self._active_tasks / self.pool_size * 100
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown thread pool."""
        self.executor.shutdown(wait=wait)


class ResilienceOrchestrator:
    """Orchestrates multiple resilience patterns for comprehensive fault tolerance."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
        self.error_aggregator = ErrorAggregator()
        
        # Global settings
        self.correlation_id_generator = self._create_correlation_id_generator()
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker."""
        circuit_breaker = CircuitBreaker(config, name)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def register_retry_handler(self, name: str, config: RetryConfig) -> RetryHandler:
        """Register a new retry handler."""
        retry_handler = RetryHandler(config)
        self.retry_handlers[name] = retry_handler
        return retry_handler
    
    def register_bulkhead(self, name: str, pool_size: int = 10, queue_size: int = 100) -> BulkheadIsolation:
        """Register a new bulkhead."""
        bulkhead = BulkheadIsolation(pool_size, queue_size)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    def resilient_call(
        self,
        func: Callable,
        circuit_breaker_name: Optional[str] = None,
        retry_handler_name: Optional[str] = None,
        bulkhead_name: Optional[str] = None,
        component: str = "unknown",
        operation: str = "unknown",
        *args, **kwargs
    ) -> Any:
        """Execute function with comprehensive resilience patterns."""
        correlation_id = next(self.correlation_id_generator)
        start_time = time.time()
        
        try:
            # Wrap function with resilience patterns
            target_func = func
            
            # Apply retry wrapper
            if retry_handler_name and retry_handler_name in self.retry_handlers:
                target_func = self.retry_handlers[retry_handler_name](target_func)
            
            # Apply circuit breaker wrapper
            if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                target_func = self.circuit_breakers[circuit_breaker_name](target_func)
            
            # Execute with bulkhead isolation
            if bulkhead_name and bulkhead_name in self.bulkheads:
                result = self.bulkheads[bulkhead_name].execute(target_func, *args, **kwargs)
            else:
                result = target_func(*args, **kwargs)
            
            return result
        
        except Exception as e:
            # Record error for analysis
            error_context = ErrorContext(
                timestamp=time.time(),
                error_type=type(e).__name__,
                error_message=str(e),
                component=component,
                operation=operation,
                severity=self._determine_severity(e),
                stack_trace=traceback.format_exc(),
                correlation_id=correlation_id,
                metadata={
                    "execution_time": time.time() - start_time,
                    "circuit_breaker": circuit_breaker_name,
                    "retry_handler": retry_handler_name,
                    "bulkhead": bulkhead_name
                }
            )
            
            self.error_aggregator.record_error(error_context)
            raise
    
    def _determine_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(exception, (CircuitBreakerError, RetryExhaustedException)):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (RateLimitExceededException, SecurityError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(exception, ValidationError):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _create_correlation_id_generator(self):
        """Create correlation ID generator."""
        counter = 0
        while True:
            counter += 1
            timestamp = int(time.time() * 1000)
            yield f"corr-{timestamp}-{counter:06d}"
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system resilience metrics."""
        circuit_breaker_metrics = {
            name: cb.get_metrics() 
            for name, cb in self.circuit_breakers.items()
        }
        
        bulkhead_metrics = {
            name: bh.get_metrics() 
            for name, bh in self.bulkheads.items()
        }
        
        error_patterns = self.error_aggregator.get_error_patterns()
        
        return {
            "timestamp": time.time(),
            "circuit_breakers": circuit_breaker_metrics,
            "bulkheads": bulkhead_metrics,
            "error_patterns": error_patterns,
            "total_components": len(self.circuit_breakers) + len(self.bulkheads)
        }
    
    def shutdown(self) -> None:
        """Shutdown all resilience components."""
        for bulkhead in self.bulkheads.values():
            bulkhead.shutdown()


# Global resilience orchestrator instance
resilience = ResilienceOrchestrator()


# Convenience decorators
def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator for applying circuit breaker pattern."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )
    
    cb = resilience.register_circuit_breaker(name, config)
    return cb


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator for applying retry pattern."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        retryable_exceptions=retryable_exceptions
    )
    
    name = f"retry-{hashlib.md5(str(config).encode()).hexdigest()[:8]}"
    retry_handler = resilience.register_retry_handler(name, config)
    return retry_handler


def bulkhead(name: str, pool_size: int = 10, queue_size: int = 100):
    """Decorator for applying bulkhead pattern."""
    bh = resilience.register_bulkhead(name, pool_size, queue_size)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return bh.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def robust(
    circuit_breaker_name: Optional[str] = None,
    retry_attempts: int = 3,
    bulkhead_name: Optional[str] = None,
    component: str = "unknown",
    operation: str = "unknown"
):
    """Comprehensive robustness decorator combining multiple patterns."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return resilience.resilient_call(
                func,
                circuit_breaker_name=circuit_breaker_name,
                retry_handler_name=f"auto-retry-{retry_attempts}",
                bulkhead_name=bulkhead_name,
                component=component,
                operation=operation,
                *args, **kwargs
            )
        
        # Auto-register retry handler if not exists
        retry_name = f"auto-retry-{retry_attempts}"
        if retry_name not in resilience.retry_handlers:
            config = RetryConfig(max_attempts=retry_attempts)
            resilience.register_retry_handler(retry_name, config)
        
        return wrapper
    return decorator