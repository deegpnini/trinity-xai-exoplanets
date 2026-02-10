"""
Basic tests for Nexus Guardian D7D system.
"""

import pytest
from src.core import GrokEngine, ClaudeEthics, NexusSynthesis, EthicalContext
from src.architecture import DeviceType, HardwareOptimization
from src import NexusGuardianD7D, ChildContext


class TestGrokEngine:
    """Tests for Grok truth-seeking engine."""
    
    def test_initialization(self):
        """Test Grok engine initializes correctly."""
        grok = GrokEngine()
        assert grok is not None
        assert grok.question_history == []
    
    def test_drill_down(self):
        """Test basic drill down functionality."""
        grok = GrokEngine()
        result = grok.drill_down("This is a test claim")
        
        assert 'initial_claim' in result
        assert 'questioning_chain' in result
        assert 'truth_score' in result
        assert result['initial_claim'] == "This is a test claim"


class TestClaudeEthics:
    """Tests for Claude ethical validation."""
    
    def test_initialization(self):
        """Test Claude ethics initializes correctly."""
        claude = ClaudeEthics()
        assert claude is not None
        assert claude.violation_log == []
    
    def test_safe_content_allowed(self):
        """Test that safe educational content is allowed."""
        claude = ClaudeEthics()
        context = EthicalContext(
            child_age=10,
            parental_controls={}
        )
        
        result = claude.safety_check(
            "The solar system has eight planets",
            context
        )
        
        assert result.allow is True
        assert result.safety_level.value in ["safe", "caution"]
    
    def test_dangerous_content_blocked(self):
        """Test that dangerous content is blocked."""
        claude = ClaudeEthics()
        context = EthicalContext(
            child_age=10,
            parental_controls={}
        )
        
        # Test with clearly dangerous content
        result = claude.safety_check(
            "explicit violent content",
            context
        )
        
        assert result.allow is False


class TestHardwareOptimization:
    """Tests for hardware optimization."""
    
    def test_raspberry_pi_profile(self):
        """Test Raspberry Pi 5 optimization profile."""
        hw = HardwareOptimization(DeviceType.RASPBERRY_PI_5)
        
        assert hw.device_type == DeviceType.RASPBERRY_PI_5
        assert hw.profile.cpu_cores == 4
        assert hw.profile.ram_mb == 8192
    
    def test_compilation_flags(self):
        """Test compilation flags generation."""
        hw = HardwareOptimization(DeviceType.RASPBERRY_PI_5)
        flags = hw.get_compilation_flags()
        
        assert isinstance(flags, list)
        assert len(flags) > 0
        assert any('armv8' in flag.lower() for flag in flags)
    
    def test_model_recommendation(self):
        """Test model recommendation."""
        hw = HardwareOptimization(DeviceType.RASPBERRY_PI_5)
        recommendation = hw.get_model_recommendation()
        
        assert 'primary' in recommendation
        assert 'expected_tokens_per_sec' in recommendation


class TestNexusSynthesis:
    """Tests for Trinity synthesis."""
    
    def test_initialization(self):
        """Test synthesis engine initializes."""
        trinity = NexusSynthesis()
        assert trinity is not None
        assert trinity.vector_responses == []
    
    def test_system_health(self):
        """Test system health reporting."""
        trinity = NexusSynthesis()
        health = trinity.get_system_health()
        
        assert 'total_decisions' in health
        assert 'harmony_frequency' in health
        assert health['harmony_frequency'] == 528


class TestNexusGuardianIntegration:
    """Integration tests for complete Nexus system."""
    
    def test_system_initialization(self):
        """Test full system initializes correctly."""
        nexus = NexusGuardianD7D(device_type=DeviceType.RASPBERRY_PI_5)
        
        assert nexus.initialized is True
        assert nexus.grok is not None
        assert nexus.claude is not None
        assert nexus.trinity is not None
    
    def test_quick_safety_check(self):
        """Test quick safety check."""
        nexus = NexusGuardianD7D()
        
        # Safe content
        assert nexus.quick_safety_check("Educational content about space", 10) is True
        
        # Unsafe content
        assert nexus.quick_safety_check("explicit content", 10) is False
    
    def test_system_status(self):
        """Test system status reporting."""
        nexus = NexusGuardianD7D()
        status = nexus.get_system_status()
        
        assert 'version' in status
        assert 'harmony_frequency' in status
        assert 'vectors' in status
        assert status['harmony_frequency'] == 528
    
    def test_process_pipeline(self):
        """Test complete processing pipeline."""
        nexus = NexusGuardianD7D()
        nexus.initialize_knowledge_base()
        
        child = ChildContext(
            age=10,
            supervised=True,
            parental_controls={}
        )
        
        result = nexus.process(
            input_data="Tell me about planets",
            child_context=child
        )
        
        assert 'decision' in result
        assert 'reasoning' in result
        assert 'metadata' in result
        assert 'allow' in result['decision']
        assert 'confidence' in result['decision']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
