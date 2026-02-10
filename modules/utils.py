"""
Module Utilities

Common utilities and helpers for module integration.
"""

from typing import Any, Dict, List, Optional
import logging
import importlib
import pkgutil
import sys

logger = logging.getLogger(__name__)


class ModuleUtils:
    """Utility functions for module management"""
    
    @staticmethod
    def safe_import(module_path: str, attr: Optional[str] = None) -> Optional[Any]:
        """
        Safely import a module or attribute
        
        Args:
            module_path: Python module path (e.g., 'src.core.grok_engine')
            attr: Optional attribute to get from module
        
        Returns:
            Imported module/attribute or None if import fails
        """
        try:
            module = importlib.import_module(module_path)
            if attr:
                return getattr(module, attr, None)
            return module
        except (ImportError, AttributeError) as e:
            logger.debug(f"Failed to import {module_path}: {e}")
            return None
    
    @staticmethod
    def check_dependencies(dependencies: List[str]) -> Dict[str, bool]:
        """
        Check if dependencies are available
        
        Args:
            dependencies: List of module names to check
        
        Returns:
            Dict mapping module names to availability (True/False)
        """
        results = {}
        for dep in dependencies:
            try:
                importlib.import_module(dep)
                results[dep] = True
            except ImportError:
                results[dep] = False
        return results
    
    @staticmethod
    def get_module_info(module_path: str) -> Dict[str, Any]:
        """
        Get information about a module
        
        Args:
            module_path: Python module path
        
        Returns:
            Dict with module information
        """
        module = ModuleUtils.safe_import(module_path)
        if module is None:
            return {
                'available': False,
                'path': module_path
            }
        
        return {
            'available': True,
            'path': module_path,
            'file': getattr(module, '__file__', 'unknown'),
            'version': getattr(module, '__version__', 'unknown'),
            'doc': getattr(module, '__doc__', 'No documentation')
        }
    
    @staticmethod
    def list_submodules(package_path: str) -> List[str]:
        """
        List all submodules in a package
        
        Args:
            package_path: Package path (e.g., 'src.core')
        
        Returns:
            List of submodule names
        """
        try:
            package = importlib.import_module(package_path)
            if not hasattr(package, '__path__'):
                return []
            
            submodules = []
            for finder, name, ispkg in pkgutil.iter_modules(package.__path__):
                submodules.append(f"{package_path}.{name}")
            return submodules
        except ImportError:
            return []
    
    @staticmethod
    def reload_module(module_path: str) -> bool:
        """
        Reload a module (useful for development)
        
        Args:
            module_path: Module path to reload
        
        Returns:
            True if reload succeeded, False otherwise
        """
        try:
            if module_path in sys.modules:
                importlib.reload(sys.modules[module_path])
                logger.info(f"Reloaded module: {module_path}")
                return True
            else:
                logger.warning(f"Module not loaded, importing: {module_path}")
                importlib.import_module(module_path)
                return True
        except Exception as e:
            logger.error(f"Failed to reload {module_path}: {e}")
            return False


# Type definitions and constants
class ModuleType:
    """Module type constants"""
    CORE = "core"
    ARCHITECTURE = "architecture"
    RAG = "rag"
    INTEGRATION = "integration"
    LEGACY = "legacy"
    UTILITY = "utility"


class ModuleStatus:
    """Module status constants"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    LEGACY = "legacy"
    ARCHIVED = "archived"


# Common module paths
MODULE_PATHS = {
    'nexus_guardian': 'src.nexus_guardian',
    'grok': 'src.core.grok_engine',
    'claude': 'src.core.claude_ethics',
    'trinity': 'src.core.nexus_synthesis',
    'chroma': 'src.rag.chroma_manager',
    'bridge': 'src.rag.math_emotional_bridge',
    'split_brain': 'src.architecture.split_brain',
    'handoff': 'src.architecture.handoff_protocol',
    'hardware': 'src.architecture.hardware_optimization',
}


def get_all_module_statuses() -> Dict[str, Dict[str, Any]]:
    """Get status of all known modules"""
    statuses = {}
    for name, path in MODULE_PATHS.items():
        statuses[name] = ModuleUtils.get_module_info(path)
    return statuses


__all__ = [
    'ModuleUtils',
    'ModuleType',
    'ModuleStatus',
    'MODULE_PATHS',
    'get_all_module_statuses',
]
