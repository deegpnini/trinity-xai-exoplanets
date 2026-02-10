"""
Nexus Guardian D7D - Module Integration Layer

This package provides integration bridges for merged repository components,
allowing seamless connection between legacy systems and the unified architecture.
"""

from typing import Dict, Any, List
import importlib
import logging

logger = logging.getLogger(__name__)

# Module registry
_MODULES: Dict[str, Any] = {}


def register_module(name: str, module: Any) -> None:
    """Register a module in the integration layer"""
    _MODULES[name] = module
    logger.info(f"Registered module: {name}")


def get_module(name: str) -> Any:
    """Get a registered module"""
    if name not in _MODULES:
        raise KeyError(f"Module not found: {name}")
    return _MODULES[name]


def list_modules() -> List[str]:
    """List all registered modules"""
    return list(_MODULES.keys())


# Auto-import available integrations
try:
    from .interestelar import InterstelarIntegration
    register_module('interestelar', InterstelarIntegration)
except ImportError as e:
    logger.debug(f"Interestelar integration not available: {e}")

try:
    from .orchestrator import OrchestratorBridge
    register_module('orchestrator', OrchestratorBridge)
except ImportError as e:
    logger.debug(f"Orchestrator bridge not available: {e}")

try:
    from .legacy_adapter import LegacyAdapter
    register_module('legacy', LegacyAdapter)
except ImportError as e:
    logger.debug(f"Legacy adapter not available: {e}")

try:
    from .utils import ModuleUtils
    register_module('utils', ModuleUtils)
except ImportError as e:
    logger.debug(f"Module utils not available: {e}")


__all__ = [
    'register_module',
    'get_module',
    'list_modules',
]
