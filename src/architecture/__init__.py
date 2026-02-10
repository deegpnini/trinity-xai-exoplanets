"""
Architecture package initialization.
"""

from .split_brain import SplitBrainArchitecture, ProcessingNode, ProcessingTask
from .handoff_protocol import HandoffProtocol, HandoffStatus
from .hardware_optimization import HardwareOptimization, DeviceType

__all__ = [
    'SplitBrainArchitecture',
    'ProcessingNode',
    'ProcessingTask',
    'HandoffProtocol',
    'HandoffStatus',
    'HardwareOptimization',
    'DeviceType',
]
