"""
Nexus Guardian D7D - Configuration Package

Centralized configuration management for the system.
"""

from .settings import (
    NexusSettings,
    DeviceType,
    QuantizationType,
    detect_device,
    get_device_defaults,
    get_settings,
    set_settings,
)

__all__ = [
    'NexusSettings',
    'DeviceType',
    'QuantizationType',
    'detect_device',
    'get_device_defaults',
    'get_settings',
    'set_settings',
]
