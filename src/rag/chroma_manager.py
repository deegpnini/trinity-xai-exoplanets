"""
ChromaDB Manager - Perplexity Vector
Vector 6/10: RAG and Factuality System

This module implements the Perplexity vector's RAG (Retrieval Augmented Generation)
system using ChromaDB for local knowledge management.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class Document:
    """Document structure for RAG system."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Search result from RAG system."""
    document: Document
    score: float
    relevance: str


class ChromaManager:
    """
    ChromaDB Manager - Local RAG system for factuality.
    
    Manages knowledge base with citations and age-appropriate filtering.
    """
    
    def __init__(self, collection_name: str = "nexus_knowledge"):
        self.collection_name = collection_name
        self.documents: Dict[str, Document] = {}
        self.collections: Dict[str, List[str]] = {
            'educational': [],
            'factual': [],
            'age_appropriate': {}
        }
        
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        age_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> bool:
        """
        Add document to knowledge base.
        
        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Document metadata (source, date, etc.)
            age_ranges: Age ranges this content is appropriate for
            
        Returns:
            True if successful
        """
        doc = Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata or {},
            embedding=self._generate_embedding(content)
        )
        
        self.documents[doc_id] = doc
        
        # Add to age-appropriate collections
        if age_ranges:
            for age_range in age_ranges:
                range_key = f"{age_range[0]}-{age_range[1]}"
                if range_key not in self.collections['age_appropriate']:
                    self.collections['age_appropriate'][range_key] = []
                self.collections['age_appropriate'][range_key].append(doc_id)
        
        return True
    
    def rag_verify(
        self,
        query: str,
        child_age: Optional[int] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Verify information using RAG.
        
        Args:
            query: Query to verify
            child_age: Age of child (for filtering)
            top_k: Number of results to return
            
        Returns:
            Verification results with citations
        """
        # Search for relevant documents
        results = self.search(query, child_age, top_k)
        
        # Verify factuality
        verification = self._verify_facts(query, results)
        
        # Generate citations
        citations = self._generate_citations(results)
        
        return {
            'query': query,
            'verified': verification['is_factual'],
            'confidence': verification['confidence'],
            'explanation': verification['explanation'],
            'citations': citations,
            'sources': [r.document.metadata.get('source', 'Unknown') for r in results]
        }
    
    def search(
        self,
        query: str,
        child_age: Optional[int] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search knowledge base.
        
        Args:
            query: Search query
            child_age: Age for filtering
            top_k: Number of results
            
        Returns:
            List of search results
        """
        query_embedding = self._generate_embedding(query)
        
        # Filter documents by age if specified
        eligible_docs = self._filter_by_age(child_age) if child_age else list(self.documents.values())
        
        # Calculate similarity scores
        scored_results = []
        for doc in eligible_docs:
            if doc.embedding:
                score = self._calculate_similarity(query_embedding, doc.embedding)
                relevance = self._determine_relevance(score)
                
                scored_results.append(SearchResult(
                    document=doc,
                    score=score,
                    relevance=relevance
                ))
        
        # Sort by score and return top_k
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]
    
    def _filter_by_age(self, age: int) -> List[Document]:
        """Filter documents appropriate for age."""
        eligible_doc_ids = set()
        
        # Check all age ranges
        for range_key, doc_ids in self.collections['age_appropriate'].items():
            min_age, max_age = map(int, range_key.split('-'))
            if min_age <= age <= max_age:
                eligible_doc_ids.update(doc_ids)
        
        return [self.documents[doc_id] for doc_id in eligible_doc_ids if doc_id in self.documents]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Placeholder for actual embedding generation
        # In production, would use a real embedding model
        return [0.1] * 384  # Typical embedding dimension
    
    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        # Simplified cosine similarity
        if len(emb1) != len(emb2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        magnitude1 = sum(a * a for a in emb1) ** 0.5
        magnitude2 = sum(b * b for b in emb2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _determine_relevance(self, score: float) -> str:
        """Determine relevance level from score."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "not_relevant"
    
    def _verify_facts(
        self, query: str, results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Verify factuality of query against results."""
        if not results:
            return {
                'is_factual': False,
                'confidence': 0.0,
                'explanation': 'No supporting evidence found in knowledge base'
            }
        
        # Check if high-relevance results support the query
        high_relevance = [r for r in results if r.relevance == "high"]
        
        if high_relevance:
            avg_score = sum(r.score for r in high_relevance) / len(high_relevance)
            return {
                'is_factual': True,
                'confidence': avg_score,
                'explanation': f'Supported by {len(high_relevance)} high-quality sources'
            }
        else:
            return {
                'is_factual': False,
                'confidence': 0.5,
                'explanation': 'Limited or no strong supporting evidence'
            }
    
    def _generate_citations(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Generate citations from results."""
        citations = []
        for i, result in enumerate(results[:3], 1):  # Top 3 citations
            metadata = result.document.metadata
            citations.append({
                'citation_number': i,
                'source': metadata.get('source', 'Unknown'),
                'title': metadata.get('title', 'Untitled'),
                'relevance': result.relevance,
                'confidence': result.score,
                'excerpt': result.document.content[:200] + "..."
            })
        return citations
    
    def load_kiwix_zim(self, zim_path: str) -> int:
        """
        Load content from Kiwix ZIM file.
        
        Args:
            zim_path: Path to ZIM file
            
        Returns:
            Number of documents loaded
        """
        # Placeholder for ZIM file loading
        # In production, would use libzim to read ZIM files
        print(f"Would load ZIM file from: {zim_path}")
        return 0
    
    def add_wikipedia_snapshot(self, topic: str, content: str, age_range: Tuple[int, int]):
        """Add Wikipedia content snapshot."""
        doc_id = f"wiki_{topic.replace(' ', '_')}"
        self.add_document(
            doc_id=doc_id,
            content=content,
            metadata={
                'source': 'Wikipedia',
                'topic': topic,
                'verified': True,
                'offline_available': True
            },
            age_ranges=[age_range]
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            'total_documents': len(self.documents),
            'age_ranges_covered': list(self.collections['age_appropriate'].keys()),
            'documents_by_age': {
                range_key: len(doc_ids)
                for range_key, doc_ids in self.collections['age_appropriate'].items()
            },
            'storage_size_mb': len(json.dumps([asdict(doc) for doc in self.documents.values()])) / 1024 / 1024
        }


def asdict(obj):
    """Convert dataclass to dict."""
    if hasattr(obj, '__dict__'):
        return {k: v for k, v in obj.__dict__.items()}
    return obj


# Example usage
if __name__ == "__main__":
    chroma = ChromaManager()
    
    # Add some educational content
    chroma.add_document(
        doc_id="solar_system_basics",
        content="The solar system consists of the Sun and eight planets...",
        metadata={
            'source': 'Educational Database',
            'verified': True,
            'topic': 'Astronomy'
        },
        age_ranges=[(8, 12), (13, 18)]
    )
    
    # Verify a fact
    result = chroma.rag_verify(
        "How many planets are in the solar system?",
        child_age=10
    )
    
    print(f"Verified: {result['verified']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Explanation: {result['explanation']}")
    print(f"Citations: {len(result['citations'])}")
    
    # Get stats
    stats = chroma.get_collection_stats()
    print(f"\nKnowledge base: {stats['total_documents']} documents")
