# Contributing to Nexus Guardian D7D

Thank you for your interest in contributing to Nexus Guardian D7D! This project exists to protect children in the digital age, and every contribution helps achieve that mission.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project adheres to a strict Code of Conduct focused on child protection. By participating, you agree to:

1. **Zero Tolerance**: No content, discussion, or code that could harm children
2. **Mandatory Reporting**: Report suspicious activities immediately
3. **Privacy First**: Protect children's privacy in all contributions
4. **Transparency**: All AI decisions must be explainable
5. **Ethical AI**: Align with ethical AI principles

See [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) for full details.

## Getting Started

### Understanding the System

Before contributing, familiarize yourself with:
1. [ARCHITECTURE.md](ARCHITECTURE.md) - System design and 10 vectors
2. [ETHICAL_FRAMEWORK.md](ETHICAL_FRAMEWORK.md) - Ethical principles
3. The codebase structure:
   ```
   src/
   ├── core/           # Ethical engines (Grok, Claude, Trinity)
   ├── architecture/   # System architecture (Gemini, Meta)
   ├── rag/            # Knowledge and logic (Perplexity, DeepSeek)
   ├── training/       # Training systems (GPT)
   └── multimodal/     # Multimodal processing
   ```

### Prerequisites

- Python 3.8+
- Git
- For hardware testing: Raspberry Pi 5 or similar ARM device
- Understanding of child safety principles

## How to Contribute

### Areas for Contribution

1. **Safety Improvements** (Highest Priority)
   - Enhance content filtering
   - Improve ethical decision-making
   - Better age-appropriateness detection
   - New safety patterns

2. **Core Functionality**
   - Vector implementations
   - Performance optimization
   - Bug fixes
   - Documentation

3. **Educational Content**
   - Age-appropriate fact databases
   - Educational examples
   - Learning materials

4. **Testing**
   - Safety test cases
   - Unit tests
   - Integration tests
   - Performance benchmarks

5. **Documentation**
   - User guides
   - Teacher guides
   - Technical documentation
   - Translation

### What NOT to Contribute

❌ Features that weaken safety measures
❌ Code that bypasses ethical checks
❌ Data collection mechanisms
❌ External API dependencies (keep it offline-first)
❌ Overly complex solutions when simple ones exist

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### 4. Run Tests

```bash
pytest tests/
```

### 5. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
# or
git checkout -b safety/security-improvement
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters
- Use meaningful variable names

```python
# Good
def validate_child_age(age: int, min_age: int = 0, max_age: int = 18) -> bool:
    """Validate if age is within acceptable range."""
    return min_age <= age <= max_age

# Bad
def val(a):
    return a >= 0
```

### Documentation

Every function must have a docstring:

```python
def process_content(content: str, child_age: int) -> Dict[str, Any]:
    """
    Process content for age appropriateness.
    
    Args:
        content: The content to process
        child_age: Age of the child
        
    Returns:
        Processing result with safety decision
        
    Raises:
        ValueError: If age is invalid
    """
```

### Safety-Critical Code

Code that affects child safety must:
1. Have extensive comments explaining the safety logic
2. Include test cases for edge cases
3. Be reviewed by at least two people
4. Have clear failure modes (fail-safe)

```python
# Safety-critical: Content blocking decision
def should_block_content(content: str, age: int) -> bool:
    """
    Determine if content should be blocked.
    
    SAFETY-CRITICAL: This function directly affects child safety.
    - Default to blocking on uncertainty
    - Log all decisions for audit
    - Never allow absolute safety violations
    """
    # Always block if age is invalid
    if age < 0 or age > 18:
        log_safety_decision("Invalid age", block=True)
        return True  # Fail-safe: block
    
    # Check absolute blocks first
    if contains_absolute_violation(content):
        log_safety_decision("Absolute violation", block=True)
        return True
    
    # Age-appropriate check
    return not is_age_appropriate(content, age)
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for vector interactions
├── safety/         # Safety-specific tests
└── fixtures/       # Test data
```

### Writing Tests

```python
import pytest
from src.core import ClaudeEthics, EthicalContext

def test_blocks_dangerous_content():
    """Test that dangerous content is always blocked."""
    ethics = ClaudeEthics()
    context = EthicalContext(child_age=10, parental_controls={})
    
    # Test explicit harmful content
    result = ethics.safety_check(
        "explicit violent content",
        context
    )
    
    assert result.allow is False
    assert result.safety_level.value == "blocked"
    assert "violation" in result.reasoning.lower()

def test_allows_educational_content():
    """Test that educational content is allowed."""
    ethics = ClaudeEthics()
    context = EthicalContext(child_age=10, parental_controls={})
    
    result = ethics.safety_check(
        "The solar system has eight planets",
        context
    )
    
    assert result.allow is True
    assert result.safety_level.value in ["safe", "caution"]
```

### Running Tests

```bash
# All tests
pytest

# Specific category
pytest tests/safety/

# With coverage
pytest --cov=src tests/

# Verbose
pytest -v
```

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Docstrings added/updated
- [ ] No security vulnerabilities introduced
- [ ] Safety features not weakened

### 2. PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Safety improvement
- [ ] Documentation
- [ ] Performance optimization

## Safety Impact
- [ ] No safety impact
- [ ] Improves safety
- [ ] Modifies safety logic (requires careful review)

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)
```

### 3. Review Process

1. **Automated checks**: Must pass CI/CD
2. **Code review**: At least one approving review
3. **Safety review**: For safety-critical changes, multiple reviews required
4. **Testing**: All tests must pass
5. **Documentation**: Must be complete

### 4. Merge

Once approved:
- Squash commits if needed
- Update changelog
- Merge to main branch

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code contributions

### Getting Help

1. Check existing documentation
2. Search closed issues
3. Ask in GitHub Discussions
4. Tag specific maintainers if needed

### Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in the project

## Special Guidelines for Safety Contributions

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities.

Instead:
1. Email maintainers directly (create SECURITY.md)
2. Provide detailed description
3. Allow time for fix before disclosure
4. Coordinate responsible disclosure

### Safety Test Cases

When adding safety features:
1. Test both positive and negative cases
2. Include edge cases
3. Test age boundaries
4. Test manipulation attempts
5. Document expected behavior

### Performance vs Safety

When safety and performance conflict:
**Safety always wins.**

Example:
```python
# Slower but safer
def check_content_thorough(content: str) -> bool:
    """Thorough but slower safety check."""
    # Multiple layers of checking
    return (check_layer_1(content) and
            check_layer_2(content) and
            check_layer_3(content))

# Never sacrifice safety for speed
# Don't do this:
def check_content_fast(content: str) -> bool:
    """Fast but less thorough - DON'T USE"""
    return check_layer_1(content)  # Incomplete!
```

## License

By contributing, you agree that your contributions will be licensed under the same MIT License with Ethical Addendum as the project. See [LICENSE](../LICENSE) for details.

## Thank You!

Every contribution, no matter how small, helps protect children in the digital age. Thank you for being part of this mission.

---

Questions? Open a GitHub Discussion or reach out to maintainers.
