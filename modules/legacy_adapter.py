"""
Legacy Adapter

Provides compatibility layer for legacy code from PROJETO_INTERESTELAR_HEBRON
and other archived components.
"""

from typing import Any, Dict, Optional
import warnings
import logging

logger = logging.getLogger(__name__)


class LegacyAdapter:
    """Adapter for legacy code integration"""
    
    def __init__(self):
        self.legacy_available = self._check_legacy()
    
    def _check_legacy(self) -> bool:
        """Check if legacy components are available"""
        import os
        legacy_path = os.path.join(
            os.path.dirname(__file__), '..', 'LEGACY'
        )
        return os.path.exists(legacy_path)
    
    def adapt_legacy_call(
        self,
        legacy_function: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Adapt a legacy function call to modern API
        
        Args:
            legacy_function: Name of the legacy function
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Result adapted to modern format
        """
        warnings.warn(
            f"Using legacy function '{legacy_function}'. "
            "Consider migrating to modern API.",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.warning(f"Legacy function called: {legacy_function}")
        
        # Map legacy functions to modern equivalents
        legacy_mapping = {
            'sence': self._adapt_sence,
            # Add more legacy function mappings as needed
        }
        
        if legacy_function in legacy_mapping:
            return legacy_mapping[legacy_function](*args, **kwargs)
        else:
            raise NotImplementedError(
                f"Legacy function '{legacy_function}' not yet adapted"
            )
    
    def _adapt_sence(self, *args, **kwargs) -> Dict[str, Any]:
        """Adapt legacy sence function"""
        # Placeholder implementation
        return {
            'status': 'legacy_adapted',
            'function': 'sence',
            'args': args,
            'kwargs': kwargs,
            'note': 'This is a compatibility shim. Migrate to modern API.'
        }
    
    def get_deprecation_info(self, component: str) -> Dict[str, Any]:
        """Get deprecation information for a component"""
        deprecations = {
            'sence': {
                'deprecated_since': '0.1.0',
                'removed_in': '0.3.0',
                'replacement': 'src.nexus_guardian.NexusGuardianD7D',
                'migration_guide': 'docs/MIGRATION_GUIDE.md'
            }
        }
        
        return deprecations.get(component, {
            'status': 'unknown',
            'note': 'No deprecation info available'
        })
    
    @staticmethod
    def is_deprecated(component: str) -> bool:
        """Check if a component is deprecated"""
        deprecated_components = ['sence', 'old_api']
        return component in deprecated_components


def warn_legacy_usage(component: str):
    """Helper to warn about legacy component usage"""
    warnings.warn(
        f"Component '{component}' is deprecated. "
        f"Check LegacyAdapter.get_deprecation_info('{component}') for details.",
        DeprecationWarning,
        stacklevel=2
    )


__all__ = ['LegacyAdapter', 'warn_legacy_usage']
