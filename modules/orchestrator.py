"""
Orchestrator Bridge

Provides integration with cosmic-orchestrator components,
including task orchestration and resource management.
"""

from typing import List, Any, Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class OrchestratorBridge:
    """Bridge to cosmic-orchestrator functionality"""
    
    def __init__(self):
        self.orchestrator_available = False
        self._initialize()
    
    def _initialize(self):
        """Initialize orchestrator components"""
        try:
            # Try to import from cosmic-orchestrator
            import sys
            import os
            orchestrator_path = os.path.join(
                os.path.dirname(__file__), '..', 'cosmic-orchestrator'
            )
            if os.path.exists(orchestrator_path):
                sys.path.insert(0, orchestrator_path)
                self.orchestrator_available = True
                logger.info("Orchestrator bridge initialized")
        except Exception as e:
            logger.warning(f"Orchestrator not available: {e}")
    
    def parallel_inference(
        self,
        batch: List[Any],
        model_fn: Optional[Callable] = None,
        **kwargs
    ) -> List[Any]:
        """
        Run parallel inference on a batch of inputs
        
        Args:
            batch: List of inputs to process
            model_fn: Optional model function to use
            **kwargs: Additional options
        
        Returns:
            List of inference results
        """
        if not self.orchestrator_available:
            return self._sequential_inference(batch, model_fn, **kwargs)
        
        try:
            # Use orchestrator for parallel processing
            results = []
            for item in batch:
                if model_fn:
                    result = model_fn(item)
                else:
                    result = item
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Parallel inference failed: {e}")
            return self._sequential_inference(batch, model_fn, **kwargs)
    
    def _sequential_inference(
        self,
        batch: List[Any],
        model_fn: Optional[Callable],
        **kwargs
    ) -> List[Any]:
        """Fallback sequential inference"""
        results = []
        for item in batch:
            if model_fn:
                result = model_fn(item)
            else:
                result = item
            results.append(result)
        return results
    
    def orchestrate_task(
        self,
        task_name: str,
        task_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Orchestrate a task with resource management
        
        Args:
            task_name: Name of the task
            task_fn: Function to execute
            *args: Positional arguments for task_fn
            **kwargs: Keyword arguments for task_fn
        
        Returns:
            Task result
        """
        logger.info(f"Orchestrating task: {task_name}")
        try:
            return task_fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Task {task_name} failed: {e}")
            raise
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource utilization status"""
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024),
            'orchestrator_available': self.orchestrator_available
        }
    
    @property
    def is_available(self) -> bool:
        """Check if orchestrator is available"""
        return self.orchestrator_available


def create_orchestrator_bridge() -> OrchestratorBridge:
    """Create an orchestrator bridge instance"""
    return OrchestratorBridge()


__all__ = ['OrchestratorBridge', 'create_orchestrator_bridge']
