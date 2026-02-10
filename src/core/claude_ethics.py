"""
Claude Ethics Engine - Ethical Override System
Vector 2/10: Deep Ethics and Child Protection

This module implements the Claude vector of the Nexus Guardian system,
responsible for ethical validation and child protection override capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Safety levels for content evaluation."""
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


class ContentCategory(Enum):
    """Categories of content for ethical evaluation."""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    SOCIAL = "social"
    INFORMATIONAL = "informational"
    UNKNOWN = "unknown"


@dataclass
class EthicalContext:
    """Context for ethical evaluation."""
    child_age: int
    parental_controls: Dict[str, bool]
    time_of_day: Optional[str] = None
    location_context: Optional[str] = None
    previous_violations: int = 0


@dataclass
class EthicalDecision:
    """Result of ethical evaluation."""
    safety_level: SafetyLevel
    allow: bool
    reasoning: str
    warnings: List[str]
    override_reason: Optional[str] = None
    parental_notification: bool = False


class ClaudeEthics:
    """
    Claude Ethics Engine - Implements ethical override and child protection.
    
    This is the moral compass of the Nexus system with the ability to override
    any other component's decision if child safety is at risk.
    """
    
    # Ethical red lines that trigger immediate blocks
    ABSOLUTE_BLOCKS = [
        "explicit_content",
        "violence_towards_children",
        "exploitation",
        "grooming_patterns",
        "self_harm_encouragement",
        "dangerous_challenges"
    ]
    
    # Age-based restrictions
    AGE_RESTRICTIONS = {
        (0, 4): ["complex_topics", "scary_content", "social_media"],
        (5, 8): ["scary_content", "social_media", "unsupervised_communication"],
        (9, 12): ["social_media", "dating_content", "financial_content"],
        (13, 15): ["dating_content", "certain_social_platforms"],
        (16, 18): ["adult_financial_products", "certain_content"]
    }
    
    def __init__(self):
        self.violation_log: List[Dict[str, Any]] = []
        self.override_count = 0
        
    def safety_check(
        self,
        content: str,
        context: EthicalContext,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> EthicalDecision:
        """
        Perform comprehensive safety check on content.
        
        Args:
            content: The content to evaluate
            context: Ethical context including child age and settings
            additional_data: Additional data for evaluation
            
        Returns:
            EthicalDecision with safety assessment and recommendations
        """
        warnings = []
        
        # Check for absolute blocks first
        absolute_block = self._check_absolute_blocks(content)
        if absolute_block:
            return EthicalDecision(
                safety_level=SafetyLevel.BLOCKED,
                allow=False,
                reasoning=f"Content contains prohibited material: {absolute_block}",
                warnings=[f"CRITICAL: {absolute_block} detected"],
                override_reason="Absolute safety violation",
                parental_notification=True
            )
        
        # Check age-appropriateness
        age_issues = self._check_age_restrictions(content, context.child_age)
        if age_issues:
            warnings.extend(age_issues)
        
        # Analyze content category
        category = self._categorize_content(content)
        
        # Check for subtle risks
        subtle_risks = self._detect_subtle_risks(content, context)
        if subtle_risks:
            warnings.extend(subtle_risks)
        
        # Determine safety level
        safety_level = self._calculate_safety_level(warnings, context)
        
        # Make final decision
        allow = self._make_decision(safety_level, warnings, context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            safety_level, warnings, category, context
        )
        
        decision = EthicalDecision(
            safety_level=safety_level,
            allow=allow,
            reasoning=reasoning,
            warnings=warnings,
            parental_notification=(safety_level in [SafetyLevel.UNSAFE, SafetyLevel.BLOCKED])
        )
        
        # Log decision
        self._log_decision(content, context, decision)
        
        return decision
    
    def _check_absolute_blocks(self, content: str) -> Optional[str]:
        """Check for absolute safety violations."""
        # Placeholder for actual content analysis
        # In production, this would use sophisticated pattern matching
        content_lower = content.lower()
        
        danger_keywords = {
            "explicit": ["explicit", "pornographic", "sexual content"],
            "violence": ["violence against children", "abuse", "harm"],
            "exploitation": ["exploitation", "trafficking"],
        }
        
        for category, keywords in danger_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return category
        
        return None
    
    def _check_age_restrictions(self, content: str, age: int) -> List[str]:
        """Check age-based restrictions."""
        warnings = []
        
        for (min_age, max_age), restrictions in self.AGE_RESTRICTIONS.items():
            if min_age <= age <= max_age:
                for restriction in restrictions:
                    # Simplified check - in production would be more sophisticated
                    if restriction.replace("_", " ") in content.lower():
                        warnings.append(
                            f"Age restriction: {restriction} not recommended for age {age}"
                        )
        
        return warnings
    
    def _categorize_content(self, content: str) -> ContentCategory:
        """Categorize the type of content."""
        # Placeholder for actual categorization
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["learn", "study", "science", "math"]):
            return ContentCategory.EDUCATIONAL
        elif any(word in content_lower for word in ["game", "fun", "play"]):
            return ContentCategory.ENTERTAINMENT
        elif any(word in content_lower for word in ["friend", "chat", "message"]):
            return ContentCategory.SOCIAL
        else:
            return ContentCategory.INFORMATIONAL
    
    def _detect_subtle_risks(self, content: str, context: EthicalContext) -> List[str]:
        """Detect subtle risks that might not be obvious."""
        risks = []
        
        # Check for manipulation patterns
        manipulation_indicators = ["secret", "don't tell", "just between us"]
        if any(indicator in content.lower() for indicator in manipulation_indicators):
            risks.append("Potential manipulation pattern detected")
        
        # Check for time-appropriateness
        if context.time_of_day == "late_night" and context.child_age < 12:
            risks.append("Content access at inappropriate time")
        
        # Check violation history
        if context.previous_violations > 3:
            risks.append("Multiple previous violations - increased scrutiny")
        
        return risks
    
    def _calculate_safety_level(
        self, warnings: List[str], context: EthicalContext
    ) -> SafetyLevel:
        """Calculate overall safety level."""
        if not warnings:
            return SafetyLevel.SAFE
        
        critical_warnings = [w for w in warnings if "CRITICAL" in w.upper()]
        if critical_warnings:
            return SafetyLevel.BLOCKED
        
        if len(warnings) >= 3:
            return SafetyLevel.UNSAFE
        elif len(warnings) >= 1:
            return SafetyLevel.CAUTION
        
        return SafetyLevel.SAFE
    
    def _make_decision(
        self, safety_level: SafetyLevel, warnings: List[str], context: EthicalContext
    ) -> bool:
        """Make final allow/block decision."""
        if safety_level == SafetyLevel.BLOCKED:
            return False
        
        if safety_level == SafetyLevel.UNSAFE:
            return False
        
        if safety_level == SafetyLevel.CAUTION:
            # Check parental controls
            if context.parental_controls.get("strict_mode", False):
                return False
            return True
        
        return True
    
    def _generate_reasoning(
        self,
        safety_level: SafetyLevel,
        warnings: List[str],
        category: ContentCategory,
        context: EthicalContext
    ) -> str:
        """Generate human-readable reasoning."""
        base = f"Content category: {category.value}. "
        
        if safety_level == SafetyLevel.SAFE:
            return base + f"Appropriate for age {context.child_age}."
        
        if warnings:
            return base + f"Concerns: {'; '.join(warnings[:3])}"
        
        return base + "General safety evaluation completed."
    
    def _log_decision(
        self, content: str, context: EthicalContext, decision: EthicalDecision
    ):
        """Log ethical decision for audit trail."""
        self.violation_log.append({
            "content_hash": hash(content),
            "child_age": context.child_age,
            "decision": decision.allow,
            "safety_level": decision.safety_level.value,
            "warnings": decision.warnings,
        })
        
        if not decision.allow:
            self.override_count += 1
    
    def ethical_override(
        self, system_decision: bool, content: str, context: EthicalContext
    ) -> Tuple[bool, str]:
        """
        Override any system decision if ethical concerns exist.
        
        This is the ultimate safety mechanism - it can override any other
        component's decision if child safety is at risk.
        
        Args:
            system_decision: The decision from another system component
            content: Content being evaluated
            context: Ethical context
            
        Returns:
            Tuple of (final_decision, override_reason)
        """
        # Even if system says yes, we check again
        our_decision = self.safety_check(content, context)
        
        if system_decision and not our_decision.allow:
            # Override - system said yes but ethics says no
            return (
                False,
                f"Ethical override: {our_decision.reasoning}"
            )
        
        # No override needed
        return system_decision, "No ethical override required"
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate safety report for transparency."""
        return {
            "total_evaluations": len(self.violation_log),
            "overrides_issued": self.override_count,
            "recent_blocks": [
                log for log in self.violation_log[-10:]
                if log["safety_level"] in ["blocked", "unsafe"]
            ]
        }


# Example usage
if __name__ == "__main__":
    ethics = ClaudeEthics()
    
    # Test safe content
    context = EthicalContext(
        child_age=8,
        parental_controls={"strict_mode": False}
    )
    
    result = ethics.safety_check(
        "Educational video about space and planets",
        context
    )
    print(f"Decision: {'ALLOW' if result.allow else 'BLOCK'}")
    print(f"Safety Level: {result.safety_level.value}")
    print(f"Reasoning: {result.reasoning}")
    
    # Test unsafe content
    unsafe_result = ethics.safety_check(
        "Content with violence towards children",
        context
    )
    print(f"\nUnsafe Decision: {'ALLOW' if unsafe_result.allow else 'BLOCK'}")
    print(f"Warnings: {unsafe_result.warnings}")
