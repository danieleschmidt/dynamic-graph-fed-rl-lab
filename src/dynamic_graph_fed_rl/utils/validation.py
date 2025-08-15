"""
Comprehensive input validation and sanitization framework for federated learning.

This module provides enterprise-grade validation for federated learning components,
protecting against malicious inputs, data poisoning attacks, and system exploitation.
"""

import re
import time
import logging
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import jax.numpy as jnp

from .error_handling import ValidationError, SecurityError


class ValidationLevel(Enum):
    """Validation strictness levels."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationConfig:
    """Configuration for validation rules."""
    level: ValidationLevel = ValidationLevel.STANDARD
    max_tensor_size: int = 10**8  # 100M elements
    max_tensor_value: float = 1000.0
    max_string_length: int = 1000
    max_dict_depth: int = 10
    max_list_length: int = 10000
    allow_inf: bool = False
    allow_nan: bool = False
    require_finite: bool = True
    sanitize_strings: bool = True
    check_injection: bool = True


class DataValidator:
    """Comprehensive data validation and sanitization framework."""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.validation_cache = {}
        self.security_patterns = self._compile_security_patterns()
        
        logging.info(f"DataValidator initialized with level: {self.config.level.value}")
    
    def _compile_security_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for security validation."""
        return {
            "sql_injection": re.compile(
                r"(union|select|insert|update|delete|drop|create|alter|exec|execute)",
                re.IGNORECASE
            ),
            "script_injection": re.compile(
                r"<script|javascript:|vbscript:|onload|onerror|onclick",
                re.IGNORECASE
            ),
            "path_traversal": re.compile(
                r"\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c",
                re.IGNORECASE
            ),
            "command_injection": re.compile(
                r"[;&|`$(){}[\]\\]",
                re.IGNORECASE
            )
        }
    
    def validate_federated_parameters(
        self,
        parameters: Dict[str, Any],
        agent_id: str,
        parameter_schema: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Validate federated learning parameters with comprehensive security checks."""
        if not isinstance(parameters, dict):
            raise ValidationError(f"Parameters must be dictionary, got {type(parameters)}")
        
        if not parameters:
            raise ValidationError("Parameters dictionary cannot be empty")
        
        # Check dictionary size and depth
        self._validate_dict_structure(parameters)
        
        validated_params = {}
        
        for key, value in parameters.items():
            # Validate and sanitize key
            sanitized_key = self._validate_parameter_key(key, agent_id)
            
            # Validate value based on type and schema
            expected_type = parameter_schema.get(sanitized_key) if parameter_schema else None
            validated_value = self._validate_parameter_value(
                value, sanitized_key, agent_id, expected_type
            )
            
            validated_params[sanitized_key] = validated_value
        
        # Perform cross-parameter validation
        self._validate_parameter_consistency(validated_params, agent_id)
        
        return validated_params
    
    def _validate_dict_structure(self, data: Dict[str, Any], depth: int = 0) -> None:
        """Validate dictionary structure and depth."""
        if depth > self.config.max_dict_depth:
            raise SecurityError(f"Dictionary depth exceeds limit: {depth} > {self.config.max_dict_depth}")
        
        if len(data) > self.config.max_list_length:
            raise SecurityError(f"Dictionary size exceeds limit: {len(data)} > {self.config.max_list_length}")
        
        for key, value in data.items():
            if isinstance(value, dict):
                self._validate_dict_structure(value, depth + 1)
            elif isinstance(value, list) and len(value) > self.config.max_list_length:
                raise SecurityError(f"List length exceeds limit: {len(value)} > {self.config.max_list_length}")
    
    def _validate_parameter_key(self, key: Any, agent_id: str) -> str:
        """Validate and sanitize parameter key."""
        if not isinstance(key, str):
            raise ValidationError(f"Agent {agent_id}: Parameter key must be string, got {type(key)}")
        
        if not key.strip():
            raise ValidationError(f"Agent {agent_id}: Parameter key cannot be empty")
        
        if len(key) > self.config.max_string_length:
            raise SecurityError(f"Agent {agent_id}: Parameter key too long: {len(key)}")
        
        # Check for security threats
        if self.config.check_injection:
            self._check_injection_patterns(key, f"Agent {agent_id} parameter key")
        
        # Sanitize key
        if self.config.sanitize_strings:
            sanitized = self._sanitize_string(key)
            if sanitized != key:
                logging.warning(f"Agent {agent_id}: Parameter key sanitized: '{key}' -> '{sanitized}'")
            return sanitized
        
        return key
    
    def _validate_parameter_value(
        self,
        value: Any,
        key: str,
        agent_id: str,
        expected_type: Optional[str] = None
    ) -> Any:
        """Validate parameter value based on type."""
        if isinstance(value, jnp.ndarray):
            return self._validate_tensor(value, key, agent_id)
        elif isinstance(value, np.ndarray):
            return self._validate_tensor(jnp.array(value), key, agent_id)
        elif isinstance(value, (int, float)):
            return self._validate_scalar(value, key, agent_id)
        elif isinstance(value, str):
            return self._validate_string(value, key, agent_id)
        elif isinstance(value, (list, tuple)):
            return self._validate_sequence(value, key, agent_id)
        elif isinstance(value, dict):
            return self._validate_dict(value, key, agent_id)
        elif value is None:
            return None
        else:
            if self.config.level == ValidationLevel.PARANOID:
                raise ValidationError(f"Agent {agent_id}: Unsupported parameter type for {key}: {type(value)}")
            else:
                logging.warning(f"Agent {agent_id}: Allowing unsupported type for {key}: {type(value)}")
                return value
    
    def _validate_tensor(self, tensor: jnp.ndarray, key: str, agent_id: str) -> jnp.ndarray:
        """Validate tensor parameters with comprehensive security checks."""
        # Check tensor size (prevent memory exhaustion)
        if tensor.size > self.config.max_tensor_size:
            raise SecurityError(
                f"Agent {agent_id}: Tensor {key} too large: {tensor.size} > {self.config.max_tensor_size}"
            )
        
        # Check for NaN and infinite values
        has_nan = jnp.any(jnp.isnan(tensor))
        has_inf = jnp.any(jnp.isinf(tensor))
        
        if has_nan and not self.config.allow_nan:
            raise ValidationError(f"Agent {agent_id}: Tensor {key} contains NaN values")
        
        if has_inf and not self.config.allow_inf:
            raise ValidationError(f"Agent {agent_id}: Tensor {key} contains infinite values")
        
        if self.config.require_finite and (has_nan or has_inf):
            raise ValidationError(f"Agent {agent_id}: Tensor {key} must contain only finite values")
        
        # Check value ranges
        if jnp.any(jnp.abs(tensor) > self.config.max_tensor_value):
            max_abs = float(jnp.max(jnp.abs(tensor)))
            if self.config.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                raise SecurityError(
                    f"Agent {agent_id}: Tensor {key} values too large: max={max_abs} > {self.config.max_tensor_value}"
                )
            else:
                logging.warning(
                    f"Agent {agent_id}: Tensor {key} has large values: max={max_abs}"
                )
        
        # Check tensor properties for anomalies
        self._validate_tensor_properties(tensor, key, agent_id)
        
        return tensor
    
    def _validate_tensor_properties(self, tensor: jnp.ndarray, key: str, agent_id: str) -> None:
        """Validate tensor statistical properties for anomaly detection."""
        if tensor.size == 0:
            raise ValidationError(f"Agent {agent_id}: Tensor {key} cannot be empty")
        
        # Check for suspicious patterns
        if self.config.level == ValidationLevel.PARANOID:
            # Check for all-zero tensors (potential dummy data)
            if jnp.all(tensor == 0):
                logging.warning(f"Agent {agent_id}: Tensor {key} is all zeros")
            
            # Check for repeated values (potential adversarial pattern)
            unique_values = jnp.unique(tensor)
            if len(unique_values) == 1 and tensor.size > 10:
                logging.warning(f"Agent {agent_id}: Tensor {key} has all identical values")
            
            # Check variance (detect potential model poisoning)
            if tensor.size > 1:
                variance = float(jnp.var(tensor))
                if variance == 0:
                    logging.warning(f"Agent {agent_id}: Tensor {key} has zero variance")
                elif variance > 1000:  # Configurable threshold
                    logging.warning(f"Agent {agent_id}: Tensor {key} has high variance: {variance}")
    
    def _validate_scalar(self, value: Union[int, float], key: str, agent_id: str) -> Union[int, float]:
        """Validate scalar parameters."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Agent {agent_id}: Expected numeric scalar for {key}, got {type(value)}")
        
        if not jnp.isfinite(value) and self.config.require_finite:
            raise ValidationError(f"Agent {agent_id}: Scalar {key} must be finite, got {value}")
        
        if abs(value) > self.config.max_tensor_value:
            if self.config.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                raise SecurityError(f"Agent {agent_id}: Scalar {key} value too large: {value}")
            else:
                logging.warning(f"Agent {agent_id}: Scalar {key} has large value: {value}")
        
        return value
    
    def _validate_string(self, value: str, key: str, agent_id: str) -> str:
        """Validate string parameters."""
        if not isinstance(value, str):
            raise ValidationError(f"Agent {agent_id}: Expected string for {key}, got {type(value)}")
        
        if len(value) > self.config.max_string_length:
            raise SecurityError(f"Agent {agent_id}: String {key} too long: {len(value)}")
        
        # Security validation
        if self.config.check_injection:
            self._check_injection_patterns(value, f"Agent {agent_id} parameter {key}")
        
        # Sanitization
        if self.config.sanitize_strings:
            sanitized = self._sanitize_string(value)
            if sanitized != value:
                logging.warning(f"Agent {agent_id}: String {key} sanitized")
            return sanitized
        
        return value
    
    def _validate_sequence(self, value: Union[List, Tuple], key: str, agent_id: str) -> Union[List, Tuple]:
        """Validate sequence parameters."""
        if len(value) > self.config.max_list_length:
            raise SecurityError(f"Agent {agent_id}: Sequence {key} too long: {len(value)}")
        
        # Validate each element
        validated_items = []
        for i, item in enumerate(value):
            validated_item = self._validate_parameter_value(
                item, f"{key}[{i}]", agent_id
            )
            validated_items.append(validated_item)
        
        return type(value)(validated_items)
    
    def _validate_dict(self, value: Dict[str, Any], key: str, agent_id: str) -> Dict[str, Any]:
        """Validate nested dictionary parameters."""
        self._validate_dict_structure(value)
        
        validated_dict = {}
        for sub_key, sub_value in value.items():
            sanitized_sub_key = self._validate_parameter_key(sub_key, agent_id)
            validated_sub_value = self._validate_parameter_value(
                sub_value, f"{key}.{sanitized_sub_key}", agent_id
            )
            validated_dict[sanitized_sub_key] = validated_sub_value
        
        return validated_dict
    
    def _check_injection_patterns(self, text: str, context: str) -> None:
        """Check for injection attack patterns."""
        for pattern_name, pattern in self.security_patterns.items():
            if pattern.search(text):
                raise SecurityError(f"{context}: Potential {pattern_name} detected in: {text[:100]}")
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string to remove dangerous characters."""
        # Remove or escape potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", ';', '|', '`', '$']
        
        sanitized = text
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def _validate_parameter_consistency(self, parameters: Dict[str, Any], agent_id: str) -> None:
        """Validate consistency across parameters."""
        # Check for parameter shape consistency (for neural network parameters)
        tensor_params = {k: v for k, v in parameters.items() if isinstance(v, jnp.ndarray)}
        
        if len(tensor_params) > 1:
            # Check for compatible dimensions in related parameters
            self._validate_tensor_compatibility(tensor_params, agent_id)
    
    def _validate_tensor_compatibility(self, tensor_params: Dict[str, jnp.ndarray], agent_id: str) -> None:
        """Validate tensor parameter compatibility."""
        # Check for reasonable parameter sizes relative to each other
        sizes = {k: v.size for k, v in tensor_params.items()}
        
        max_size = max(sizes.values())
        min_size = min(sizes.values())
        
        # Flag suspicious size ratios
        if max_size > min_size * 10000:  # 10,000x difference seems suspicious
            logging.warning(
                f"Agent {agent_id}: Large parameter size variation: max={max_size}, min={min_size}"
            )
    
    def validate_communication_data(
        self,
        data: Dict[str, Any],
        source_id: str,
        data_type: str = "general"
    ) -> Dict[str, Any]:
        """Validate communication data between federated components."""
        # Basic structure validation
        if not isinstance(data, dict):
            raise ValidationError(f"Communication data from {source_id} must be dictionary")
        
        # Check for required fields based on data type
        required_fields = self._get_required_fields(data_type)
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field '{field}' from {source_id}")
        
        # Validate each field
        validated_data = {}
        for key, value in data.items():
            validated_key = self._validate_parameter_key(key, source_id)
            validated_value = self._validate_parameter_value(value, key, source_id)
            validated_data[validated_key] = validated_value
        
        # Type-specific validation
        if data_type == "aggregation":
            self._validate_aggregation_data(validated_data, source_id)
        elif data_type == "model_update":
            self._validate_model_update_data(validated_data, source_id)
        
        return validated_data
    
    def _get_required_fields(self, data_type: str) -> List[str]:
        """Get required fields for different data types."""
        field_requirements = {
            "aggregation": ["parameters", "agent_id", "timestamp"],
            "model_update": ["parameters", "round_number", "agent_id"],
            "general": ["timestamp"]
        }
        return field_requirements.get(data_type, [])
    
    def _validate_aggregation_data(self, data: Dict[str, Any], source_id: str) -> None:
        """Validate aggregation-specific data."""
        # Check timestamp validity
        if "timestamp" in data:
            timestamp = data["timestamp"]
            current_time = time.time()
            
            if not isinstance(timestamp, (int, float)):
                raise ValidationError(f"Invalid timestamp type from {source_id}")
            
            # Check for reasonable timestamp (not too old or in future)
            time_diff = abs(current_time - timestamp)
            if time_diff > 3600:  # 1 hour tolerance
                logging.warning(f"Suspicious timestamp from {source_id}: {time_diff}s difference")
    
    def _validate_model_update_data(self, data: Dict[str, Any], source_id: str) -> None:
        """Validate model update specific data."""
        # Check round number consistency
        if "round_number" in data:
            round_num = data["round_number"]
            if not isinstance(round_num, int) or round_num < 0:
                raise ValidationError(f"Invalid round number from {source_id}: {round_num}")
    
    def create_validation_report(self, data: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """Create comprehensive validation report."""
        report = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "validation_level": self.config.level.value,
            "total_parameters": len(data),
            "tensor_parameters": 0,
            "scalar_parameters": 0,
            "string_parameters": 0,
            "other_parameters": 0,
            "total_tensor_elements": 0,
            "warnings": [],
            "security_checks_passed": True
        }
        
        for key, value in data.items():
            if isinstance(value, jnp.ndarray):
                report["tensor_parameters"] += 1
                report["total_tensor_elements"] += value.size
            elif isinstance(value, (int, float)):
                report["scalar_parameters"] += 1
            elif isinstance(value, str):
                report["string_parameters"] += 1
            else:
                report["other_parameters"] += 1
        
        return report


# Global validator instance with standard configuration
validator = DataValidator()

# Convenience functions
def validate_federated_params(parameters: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    """Convenience function for validating federated parameters."""
    return validator.validate_federated_parameters(parameters, agent_id)

def validate_communication_data(data: Dict[str, Any], source_id: str, data_type: str = "general") -> Dict[str, Any]:
    """Convenience function for validating communication data."""
    return validator.validate_communication_data(data, source_id, data_type)

def create_strict_validator() -> DataValidator:
    """Create validator with strict security settings."""
    config = ValidationConfig(
        level=ValidationLevel.STRICT,
        max_tensor_size=10**7,  # Smaller limit
        max_tensor_value=100.0,  # Stricter value limit
        require_finite=True,
        sanitize_strings=True,
        check_injection=True
    )
    return DataValidator(config)

def create_paranoid_validator() -> DataValidator:
    """Create validator with maximum security settings."""
    config = ValidationConfig(
        level=ValidationLevel.PARANOID,
        max_tensor_size=10**6,  # Very small limit
        max_tensor_value=10.0,  # Very strict value limit
        max_string_length=100,
        max_dict_depth=5,
        max_list_length=1000,
        allow_inf=False,
        allow_nan=False,
        require_finite=True,
        sanitize_strings=True,
        check_injection=True
    )
    return DataValidator(config)