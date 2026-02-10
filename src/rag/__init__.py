"""
RAG package initialization.
"""

from .chroma_manager import ChromaManager, Document, SearchResult
from .math_emotional_bridge import MathEmotionalBridge, EmotionalState, EmotionalMetrics

__all__ = [
    'ChromaManager',
    'Document',
    'SearchResult',
    'MathEmotionalBridge',
    'EmotionalState',
    'EmotionalMetrics',
]
