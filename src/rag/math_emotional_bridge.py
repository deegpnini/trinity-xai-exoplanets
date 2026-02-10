"""
Math-Emotional Bridge - DeepSeek Vector
Vector 9/10: Logic and Emotional Consistency

This module implements DeepSeek's mathematical-emotional bridge,
connecting logical analysis with emotional understanding.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math


class EmotionalState(Enum):
    """Primary emotional states."""
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    CALM = "calm"
    EXCITED = "excited"
    CONFUSED = "confused"
    CONFIDENT = "confident"


@dataclass
class EmotionalMetrics:
    """Quantified emotional metrics."""
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    dominance: float  # 0 (submissive) to 1 (dominant)
    consistency: float  # 0 to 1


@dataclass
class LogicalAnalysis:
    """Logical analysis results."""
    coherence_score: float
    contradiction_count: int
    reasoning_steps: List[str]
    confidence: float


class MathEmotionalBridge:
    """
    Math-Emotional Bridge - Connects logic with emotion.
    
    Translates emotional states to mathematical representations and vice versa,
    enabling consistent AI responses that respect both logic and emotion.
    """
    
    # Emotional state vectors in valence-arousal-dominance space
    EMOTIONAL_VECTORS = {
        EmotionalState.HAPPY: (0.8, 0.6, 0.7),
        EmotionalState.SAD: (-0.7, 0.3, 0.3),
        EmotionalState.ANXIOUS: (-0.5, 0.8, 0.2),
        EmotionalState.CALM: (0.2, 0.1, 0.5),
        EmotionalState.EXCITED: (0.7, 0.9, 0.6),
        EmotionalState.CONFUSED: (-0.2, 0.5, 0.3),
        EmotionalState.CONFIDENT: (0.6, 0.5, 0.8)
    }
    
    def __init__(self):
        self.interaction_history: List[Dict[str, Any]] = []
        
    def quantify_emotion(
        self, emotional_state: EmotionalState, intensity: float = 1.0
    ) -> EmotionalMetrics:
        """
        Quantify emotional state mathematically.
        
        Args:
            emotional_state: The emotional state to quantify
            intensity: Intensity multiplier (0-1)
            
        Returns:
            Quantified emotional metrics
        """
        vector = self.EMOTIONAL_VECTORS[emotional_state]
        
        return EmotionalMetrics(
            valence=vector[0] * intensity,
            arousal=vector[1] * intensity,
            dominance=vector[2] * intensity,
            consistency=1.0  # Initial consistency
        )
    
    def analyze_emotional_consistency(
        self,
        current_emotion: EmotionalMetrics,
        context_history: List[EmotionalMetrics]
    ) -> float:
        """
        Analyze consistency of emotional progression.
        
        Args:
            current_emotion: Current emotional state
            context_history: Previous emotional states
            
        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        if not context_history:
            return 1.0
        
        # Calculate emotional trajectory
        inconsistencies = 0
        for prev_emotion in context_history[-5:]:  # Last 5 interactions
            # Check for unrealistic jumps
            valence_diff = abs(current_emotion.valence - prev_emotion.valence)
            arousal_diff = abs(current_emotion.arousal - prev_emotion.arousal)
            
            # Large sudden changes are inconsistent
            if valence_diff > 1.5:
                inconsistencies += 1
            if arousal_diff > 1.5:
                inconsistencies += 1
        
        # Calculate consistency score
        max_inconsistencies = len(context_history[-5:]) * 2
        consistency = 1.0 - (inconsistencies / max_inconsistencies) if max_inconsistencies > 0 else 1.0
        
        return max(0.0, consistency)
    
    def logical_reasoning_check(
        self, response: str, context: Dict[str, Any]
    ) -> LogicalAnalysis:
        """
        Perform logical consistency check on response.
        
        Args:
            response: Response to analyze
            context: Context information
            
        Returns:
            Logical analysis results
        """
        # Analyze logical structure
        reasoning_steps = self._extract_reasoning_steps(response)
        
        # Check for contradictions
        contradictions = self._detect_contradictions(reasoning_steps, context)
        
        # Calculate coherence
        coherence = self._calculate_coherence(reasoning_steps)
        
        # Calculate confidence based on logical soundness
        confidence = coherence * (1.0 - (len(contradictions) * 0.2))
        
        return LogicalAnalysis(
            coherence_score=coherence,
            contradiction_count=len(contradictions),
            reasoning_steps=reasoning_steps,
            confidence=max(0.0, min(1.0, confidence))
        )
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract logical reasoning steps from text."""
        # Simplified extraction - in production would be more sophisticated
        sentences = text.split('.')
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_contradictions(
        self, steps: List[str], context: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """Detect logical contradictions."""
        contradictions = []
        
        # Simplified contradiction detection
        # In production, would use semantic analysis
        negative_words = {'not', 'never', 'no', "don't", "doesn't"}
        positive_claims = []
        
        for step in steps:
            step_lower = step.lower()
            has_negative = any(word in step_lower for word in negative_words)
            
            if not has_negative:
                positive_claims.append(step)
            else:
                # Check if contradicts previous positive claims
                for claim in positive_claims:
                    if self._are_contradictory(step, claim):
                        contradictions.append((step, claim))
        
        return contradictions
    
    def _are_contradictory(self, statement1: str, statement2: str) -> bool:
        """Check if two statements contradict."""
        # Placeholder for actual contradiction detection
        # Would use semantic similarity and negation detection
        return False
    
    def _calculate_coherence(self, steps: List[str]) -> float:
        """Calculate logical coherence score."""
        if not steps:
            return 0.0
        
        # Simple coherence based on step count and structure
        # In production, would use more sophisticated analysis
        if len(steps) < 2:
            return 0.5
        
        # Longer, structured reasoning is generally more coherent
        coherence = min(1.0, 0.3 + (len(steps) * 0.1))
        return coherence
    
    def bridge_logic_emotion(
        self,
        logical_response: str,
        target_emotion: EmotionalState,
        child_age: int
    ) -> Dict[str, Any]:
        """
        Bridge logical response with appropriate emotional tone.
        
        Args:
            logical_response: Logically sound response
            target_emotion: Desired emotional tone
            child_age: Age of child
            
        Returns:
            Emotionally-tuned response with analysis
        """
        # Quantify target emotion
        emotion_metrics = self.quantify_emotion(target_emotion)
        
        # Analyze logical consistency
        logical_analysis = self.logical_reasoning_check(
            logical_response,
            {'child_age': child_age}
        )
        
        # Adjust response based on emotion and age
        tuned_response = self._apply_emotional_tuning(
            logical_response,
            emotion_metrics,
            child_age
        )
        
        # Calculate bridge quality
        bridge_quality = self._calculate_bridge_quality(
            logical_analysis,
            emotion_metrics
        )
        
        return {
            'original_response': logical_response,
            'tuned_response': tuned_response,
            'emotional_metrics': {
                'valence': emotion_metrics.valence,
                'arousal': emotion_metrics.arousal,
                'dominance': emotion_metrics.dominance
            },
            'logical_analysis': {
                'coherence': logical_analysis.coherence_score,
                'contradictions': logical_analysis.contradiction_count,
                'confidence': logical_analysis.confidence
            },
            'bridge_quality': bridge_quality,
            'age_appropriate': self._check_age_appropriateness(tuned_response, child_age)
        }
    
    def _apply_emotional_tuning(
        self, response: str, emotion: EmotionalMetrics, age: int
    ) -> str:
        """Apply emotional tuning to response."""
        # Placeholder for actual emotional tuning
        # In production, would adjust language, tone, and examples
        
        # For younger children, use simpler language
        if age < 8:
            prefix = "Let me explain in a simple way: "
        elif age < 13:
            prefix = "Here's what I found: "
        else:
            prefix = ""
        
        # Add emotional tone markers based on valence
        if emotion.valence > 0.5:
            suffix = " I hope this helps! 😊"
        elif emotion.valence < -0.5:
            suffix = " I understand this might be difficult."
        else:
            suffix = ""
        
        return prefix + response + suffix
    
    def _calculate_bridge_quality(
        self, logical: LogicalAnalysis, emotional: EmotionalMetrics
    ) -> float:
        """Calculate quality of logic-emotion bridge."""
        # Quality is high when both logic and emotion are well-balanced
        logical_quality = logical.confidence
        emotional_quality = emotional.consistency
        
        # Balanced combination
        return (logical_quality + emotional_quality) / 2
    
    def _check_age_appropriateness(self, response: str, age: int) -> bool:
        """Check if response is age-appropriate."""
        # Simplified check - in production would be more sophisticated
        word_count = len(response.split())
        
        # Age-based complexity thresholds
        max_words = {
            (0, 7): 50,
            (8, 12): 150,
            (13, 18): 300
        }
        
        for (min_age, max_age), max_word_count in max_words.items():
            if min_age <= age <= max_age:
                return word_count <= max_word_count
        
        return True
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Generate analytics report on logic-emotion bridging."""
        if not self.interaction_history:
            return {'total_interactions': 0}
        
        total = len(self.interaction_history)
        avg_bridge_quality = sum(
            i.get('bridge_quality', 0) for i in self.interaction_history
        ) / total
        
        return {
            'total_interactions': total,
            'average_bridge_quality': avg_bridge_quality,
            'high_quality_ratio': sum(
                1 for i in self.interaction_history
                if i.get('bridge_quality', 0) > 0.8
            ) / total
        }


# Example usage
if __name__ == "__main__":
    bridge = MathEmotionalBridge()
    
    # Test emotion quantification
    happy_metrics = bridge.quantify_emotion(EmotionalState.HAPPY, intensity=0.8)
    print(f"Happy emotion - Valence: {happy_metrics.valence:.2f}, "
          f"Arousal: {happy_metrics.arousal:.2f}")
    
    # Test logical analysis
    response = "The Earth orbits the Sun. This takes 365 days."
    analysis = bridge.logical_reasoning_check(response, {'topic': 'astronomy'})
    print(f"\nLogical coherence: {analysis.coherence_score:.2f}")
    print(f"Contradictions: {analysis.contradiction_count}")
    
    # Test full bridge
    result = bridge.bridge_logic_emotion(
        logical_response="Mathematics is the study of numbers and patterns.",
        target_emotion=EmotionalState.EXCITED,
        child_age=9
    )
    print(f"\nBridge quality: {result['bridge_quality']:.2f}")
    print(f"Tuned response: {result['tuned_response']}")
