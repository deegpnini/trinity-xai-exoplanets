"""
Nexus Guardian D7D - Unified Configuration System

Centralized configuration management for all modules and components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types"""
    RASPBERRY_PI_5 = "raspberry_pi_5"
    GALAXY_A70 = "galaxy_a70"
    GENERIC_ARM = "generic_arm"
    CPU = "cpu"
    AUTO = "auto"


class QuantizationType(Enum):
    """Model quantization types"""
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q8_0 = "Q8_0"
    F16 = "F16"
    F32 = "F32"


@dataclass
class NexusSettings:
    """
    Unified configuration for Nexus Guardian D7D
    
    All settings can be overridden via environment variables with NEXUS_ prefix.
    Example: NEXUS_DEVICE_TYPE, NEXUS_MODEL_PATH, etc.
    """
    
    # Device Configuration
    device_type: DeviceType = DeviceType.AUTO
    
    # Model Configuration
    model_path: str = "./models/llama-3.2-3b-q4_k_m.gguf"
    quantization: QuantizationType = QuantizationType.Q4_K_M
    model_name: str = "llama-3.2-3b"
    
    # RAG Configuration
    chroma_path: str = "./data/chroma"
    max_context: int = 2048
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Safety Configuration
    ethical_override_enabled: bool = True
    min_confidence_threshold: float = 0.7
    strict_mode: bool = False
    require_citations: bool = True
    
    # Performance Configuration
    max_memory_mb: int = 2500
    target_tokens_per_sec: float = 5.0
    batch_size: int = 1
    threads: int = 4
    
    # Child Protection Settings
    default_age: int = 10
    default_supervised: bool = True
    parental_logging: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Advanced Settings
    enable_multimodal: bool = False
    enable_audio: bool = False
    enable_vision: bool = False
    
    # Module Integration
    enable_interestelar: bool = True
    enable_orchestrator: bool = True
    enable_legacy_adapter: bool = False
    
    def __post_init__(self):
        """Validate and process settings after initialization"""
        self._load_from_environment()
        self._validate_settings()
    
    def _load_from_environment(self):
        """Load settings from environment variables"""
        # Device type
        env_device = os.getenv("NEXUS_DEVICE_TYPE")
        if env_device:
            try:
                self.device_type = DeviceType(env_device.lower())
            except ValueError:
                logger.warning(f"Invalid NEXUS_DEVICE_TYPE: {env_device}")
        
        # Model configuration
        self.model_path = os.getenv("NEXUS_MODEL_PATH", self.model_path)
        self.model_name = os.getenv("NEXUS_MODEL_NAME", self.model_name)
        
        env_quant = os.getenv("NEXUS_QUANTIZATION")
        if env_quant:
            try:
                self.quantization = QuantizationType(env_quant.upper())
            except ValueError:
                logger.warning(f"Invalid NEXUS_QUANTIZATION: {env_quant}")
        
        # RAG configuration
        self.chroma_path = os.getenv("NEXUS_CHROMA_PATH", self.chroma_path)
        self.max_context = int(os.getenv("NEXUS_MAX_CONTEXT", self.max_context))
        
        # Safety configuration
        self.ethical_override_enabled = os.getenv(
            "NEXUS_ETHICAL_OVERRIDE", str(self.ethical_override_enabled)
        ).lower() == "true"
        self.min_confidence_threshold = float(
            os.getenv("NEXUS_MIN_CONFIDENCE", self.min_confidence_threshold)
        )
        self.strict_mode = os.getenv(
            "NEXUS_STRICT_MODE", str(self.strict_mode)
        ).lower() == "true"
        
        # Performance configuration
        self.max_memory_mb = int(os.getenv("NEXUS_MAX_MEMORY_MB", self.max_memory_mb))
        self.target_tokens_per_sec = float(
            os.getenv("NEXUS_TARGET_TOKENS_PER_SEC", self.target_tokens_per_sec)
        )
        
        # Logging
        self.log_level = os.getenv("NEXUS_LOG_LEVEL", self.log_level)
        self.log_file = os.getenv("NEXUS_LOG_FILE", self.log_file)
    
    def _validate_settings(self):
        """Validate settings for consistency"""
        # Validate confidence threshold
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError("min_confidence_threshold must be between 0.0 and 1.0")
        
        # Validate memory limit
        if self.max_memory_mb < 1000:
            logger.warning(f"Low memory limit: {self.max_memory_mb}MB")
        
        # Ensure ethical override cannot be disabled in strict mode
        if self.strict_mode and not self.ethical_override_enabled:
            logger.warning("Strict mode requires ethical override, enabling it")
            self.ethical_override_enabled = True
        
        # Validate model path exists if not auto-downloading
        if os.path.exists(self.model_path):
            logger.info(f"Model found at: {self.model_path}")
        else:
            logger.warning(f"Model not found at: {self.model_path}")
    
    @classmethod
    def from_environment(cls) -> 'NexusSettings':
        """
        Create settings with auto-detection from environment
        
        Returns:
            NexusSettings configured from environment
        """
        device = detect_device()
        defaults = get_device_defaults(device)
        return cls(**defaults)
    
    @classmethod
    def for_device(cls, device: DeviceType) -> 'NexusSettings':
        """
        Create settings optimized for a specific device
        
        Args:
            device: Device type to configure for
        
        Returns:
            NexusSettings optimized for the device
        """
        defaults = get_device_defaults(device)
        return cls(**defaults)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'device_type': self.device_type.value,
            'model_path': self.model_path,
            'quantization': self.quantization.value,
            'chroma_path': self.chroma_path,
            'max_context': self.max_context,
            'ethical_override_enabled': self.ethical_override_enabled,
            'min_confidence_threshold': self.min_confidence_threshold,
            'strict_mode': self.strict_mode,
            'max_memory_mb': self.max_memory_mb,
            'target_tokens_per_sec': self.target_tokens_per_sec,
        }
    
    def __repr__(self) -> str:
        """String representation of settings"""
        return (
            f"NexusSettings(\n"
            f"  device={self.device_type.value},\n"
            f"  model={self.model_name},\n"
            f"  quantization={self.quantization.value},\n"
            f"  ethical_override={self.ethical_override_enabled},\n"
            f"  strict_mode={self.strict_mode}\n"
            f")"
        )


def detect_device() -> DeviceType:
    """
    Auto-detect the current device type
    
    Returns:
        Detected DeviceType
    """
    import platform
    
    # Check for Raspberry Pi
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
            if 'raspberry pi 5' in model:
                return DeviceType.RASPBERRY_PI_5
    except FileNotFoundError:
        pass
    
    # Check for Android (Termux)
    if os.getenv('ANDROID_ROOT'):
        return DeviceType.GALAXY_A70
    
    # Check for ARM architecture
    machine = platform.machine().lower()
    if 'arm' in machine or 'aarch64' in machine:
        return DeviceType.GENERIC_ARM
    
    # Default to CPU
    return DeviceType.CPU


def get_device_defaults(device: DeviceType) -> Dict[str, Any]:
    """
    Get default configuration for a device type
    
    Args:
        device: Device type
    
    Returns:
        Dict of default settings
    """
    defaults = {
        DeviceType.RASPBERRY_PI_5: {
            'device_type': DeviceType.RASPBERRY_PI_5,
            'model_name': 'llama-3.2-3b',
            'quantization': QuantizationType.Q4_K_M,
            'max_memory_mb': 2500,
            'target_tokens_per_sec': 5.0,
            'threads': 4,
        },
        DeviceType.GALAXY_A70: {
            'device_type': DeviceType.GALAXY_A70,
            'model_name': 'phi-2-2.7b',
            'quantization': QuantizationType.Q4_K_M,
            'max_memory_mb': 2500,
            'target_tokens_per_sec': 6.0,
            'threads': 4,
        },
        DeviceType.GENERIC_ARM: {
            'device_type': DeviceType.GENERIC_ARM,
            'model_name': 'phi-2-2.7b',
            'quantization': QuantizationType.Q4_K_M,
            'max_memory_mb': 2000,
            'target_tokens_per_sec': 4.0,
            'threads': 4,
        },
        DeviceType.CPU: {
            'device_type': DeviceType.CPU,
            'model_name': 'llama-3.2-3b',
            'quantization': QuantizationType.Q4_K_M,
            'max_memory_mb': 4000,
            'target_tokens_per_sec': 3.0,
            'threads': 8,
        },
    }
    
    return defaults.get(device, defaults[DeviceType.CPU])


# Global settings instance (lazily initialized)
_settings: Optional[NexusSettings] = None


def get_settings() -> NexusSettings:
    """
    Get the global settings instance
    
    Returns:
        Global NexusSettings
    """
    global _settings
    if _settings is None:
        _settings = NexusSettings.from_environment()
    return _settings


def set_settings(settings: NexusSettings):
    """
    Set the global settings instance
    
    Args:
        settings: NexusSettings to use globally
    """
    global _settings
    _settings = settings


__all__ = [
    'NexusSettings',
    'DeviceType',
    'QuantizationType',
    'detect_device',
    'get_device_defaults',
    'get_settings',
    'set_settings',
]
