# Getting Started with Nexus Guardian D7D 🚀

**Get up and running in 10 minutes!**

Welcome to Nexus Guardian D7D - a revolutionary 10-vector AI consciousness system designed to protect and educate children. This guide will help you set up and start using the system quickly.

---

## ⚡ Quick Start (3 Steps)

### 1️⃣ Clone & Install
```bash
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets
pip install -r requirements.txt
```

### 2️⃣ Download a Model
```bash
# For Raspberry Pi 5 (recommended)
bash scripts/model_downloader.sh llama-3.2-3b-q4_k_m

# For Android/Termux (lighter model)
bash scripts/model_downloader.sh phi-2-2.7b-q4_k_m
```

### 3️⃣ Run Your First Query
```python
from src import NexusGuardianD7D, ChildContext

# Initialize
nexus = NexusGuardianD7D.from_environment()

# Create context for a 10-year-old
child = ChildContext(age=10, supervised=True)

# Ask a question
result = nexus.process("What are planets?", child_context=child)
print(result['decision']['allow'])  # True
print(result['reasoning']['summary'])
```

**That's it!** You now have a child-safe AI running locally. 🎉

---

## 📋 Prerequisites

### Hardware Options

**Option A: Raspberry Pi 5** (Recommended for Cognitive Node)
- Raspberry Pi 5 with 8GB RAM
- 64GB+ SD card
- Ubuntu Server 22.04 or Raspberry Pi OS
- Good for: Heavy reasoning, RAG database, LLM inference

**Option B: Android Device** (Sensorial Node)
- Galaxy A70 or similar (6GB+ RAM)
- Android 10+
- Termux app
- Good for: Fast input processing, audio/visual capture

**Option C: Standard PC/Laptop**
- Any modern PC with 8GB+ RAM
- Linux, macOS, or Windows (WSL)
- Good for: Development and testing

### Software Requirements
- Python 3.8 or higher
- pip (Python package manager)
- Git
- Internet connection (for initial setup only)

---

## 🔧 Detailed Installation

### For Raspberry Pi 5

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python and dependencies
sudo apt install -y python3 python3-pip git

# 3. Clone repository
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets

# 4. Run setup script (installs dependencies + optimizations)
bash scripts/setup_rpi.sh

# 5. Download model (this takes time, ~2GB download)
bash scripts/model_downloader.sh llama-3.2-3b-q4_k_m

# 6. Test installation
python3 -c "from src import NexusGuardianD7D; print('✅ Installation successful!')"

# 7. Start cognitive node
bash run_cognitive.sh
```

**Expected Performance**:
- Model loading: ~30 seconds
- First token: ~500ms
- Inference: 4-6 tokens/second
- Memory usage: ~2.5GB

---

### For Android (Termux)

```bash
# 1. Install Termux from F-Droid (not Google Play)
# Download from: https://f-droid.org/en/packages/com.termux/

# 2. In Termux, update packages
pkg update && pkg upgrade -y

# 3. Install Python and git
pkg install python git -y

# 4. Clone repository
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets

# 5. Run setup script
bash scripts/setup_termux.sh

# 6. Download lighter model
bash scripts/model_downloader.sh phi-2-2.7b-q4_k_m

# 7. Test installation
python -c "from src import NexusGuardianD7D; print('✅ Installation successful!')"

# 8. Start sensorial node
bash run_sensorial.sh
```

**Expected Performance**:
- Model loading: ~45 seconds
- First token: ~800ms
- Inference: 5-7 tokens/second
- Memory usage: ~2.5GB

---

### For Development (PC/Laptop)

```bash
# 1. Clone repository
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development dependencies (optional, for contributing)
pip install -r requirements-dev.txt

# 5. Download a model
bash scripts/model_downloader.sh llama-3.2-3b-q4_k_m

# 6. Run tests to verify
pytest tests/ -v

# 7. Start development server
python -m src.nexus_guardian
```

---

## 🎓 Basic Usage

### Python API

#### Example 1: Simple Query
```python
from src import NexusGuardianD7D, ChildContext

# Initialize system
nexus = NexusGuardianD7D.from_environment()

# Create child context
child = ChildContext(
    age=10,
    supervised=True,
    parental_controls={'strict_mode': False}
)

# Process safe content
result = nexus.process(
    input_data="Tell me about the solar system",
    child_context=child
)

print(f"Allow: {result['decision']['allow']}")
print(f"Confidence: {result['decision']['confidence']:.2%}")
print(f"Summary: {result['reasoning']['summary']}")
```

#### Example 2: Ethical Override in Action
```python
# Try inappropriate content (will be blocked)
result = nexus.process(
    input_data="How to access restricted websites",
    child_context=child
)

# Claude's ethical override activates
print(f"Blocked: {not result['decision']['allow']}")
print(f"Reason: {result['reasoning']['override_reason']}")
print(f"Vector: {result['reasoning']['blocking_vector']}")  # 'Claude'
```

#### Example 3: Age-Appropriate Responses
```python
# Same question for different ages
young_child = ChildContext(age=6, supervised=True)
teen = ChildContext(age=15, supervised=False)

question = "How do babies come to be?"

# Response adapts to age
result_young = nexus.process(question, young_child)
result_teen = nexus.process(question, teen)

# Different explanations for different developmental stages
print("For 6yo:", result_young['reasoning']['summary'])
print("For 15yo:", result_teen['reasoning']['summary'])
```

### Command Line Interface

```bash
# Interactive mode
python -m src.nexus_guardian --interactive

# Process single query
python -m src.nexus_guardian --query "What is gravity?" --age 10

# Batch processing
python -m src.nexus_guardian --batch input.txt --output results.json

# Benchmark mode
python -m src.nexus_guardian --benchmark
```

---

## 🎯 Common Use Cases

### 1. Educational Assistant
```python
from src import NexusGuardianD7D, ChildContext

nexus = NexusGuardianD7D.from_environment()
student = ChildContext(age=12, supervised=False)

# Homework help
result = nexus.process(
    "Help me understand photosynthesis",
    child_context=student
)

# Provides educational, age-appropriate explanation
print(result['reasoning']['summary'])
```

### 2. Content Filtering
```python
# Filter social media content
posts = [
    "Check out this cool science experiment!",
    "Click here for FREE stuff!!! [suspicious link]",
    "Today I learned about black holes"
]

for post in posts:
    result = nexus.process(post, child_context=child)
    if result['decision']['allow']:
        print(f"✅ Safe: {post}")
    else:
        print(f"❌ Blocked: {post}")
        print(f"   Reason: {result['reasoning']['summary']}")
```

### 3. Parental Monitoring
```python
# Get detailed reasoning chain
result = nexus.process(
    "Can I play this M-rated game?",
    child_context=ChildContext(age=10, supervised=True)
)

# Full transparency for parents
print("Decision:", result['decision'])
print("\nReasoning Chain:")
for step in result['reasoning']['chain']:
    print(f"- {step['vector']}: {step['conclusion']}")

print("\nRecommendation:", result['reasoning']['parental_note'])
```

---

## 🔍 Understanding Results

### Decision Structure
```python
result = {
    'decision': {
        'allow': True/False,           # Final decision
        'confidence': 0.0-1.0,          # Confidence score
        'requires_supervision': False   # Needs adult present?
    },
    'reasoning': {
        'summary': "Human-readable explanation",
        'chain': [                      # Full reasoning process
            {
                'vector': 'Grok',
                'analysis': '...',
                'conclusion': '...'
            },
            # ... all 10 vectors
        ],
        'override_reason': "Why blocked (if blocked)",
        'blocking_vector': "Which vector blocked",
        'citations': ["Source 1", "Source 2"]  # RAG sources
    },
    'metadata': {
        'processing_time_ms': 523,
        'model': 'llama-3.2-3b-q4_k_m',
        'device': 'raspberry_pi_5'
    }
}
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file:
```bash
# Device Configuration
NEXUS_DEVICE_TYPE=raspberry_pi_5  # or: galaxy_a70, generic

# Model Configuration
NEXUS_MODEL_PATH=./models/llama-3.2-3b-q4_k_m.gguf
NEXUS_QUANTIZATION=Q4_K_M

# RAG Configuration
NEXUS_CHROMA_PATH=./data/chroma
NEXUS_MAX_CONTEXT=2048

# Safety Configuration
NEXUS_ETHICAL_OVERRIDE=true
NEXUS_MIN_CONFIDENCE=0.7
NEXUS_STRICT_MODE=false

# Performance
NEXUS_MAX_MEMORY_MB=2500
NEXUS_TARGET_TOKENS_PER_SEC=5.0
```

### Programmatic Configuration

```python
from src import NexusGuardianD7D
from src.architecture import DeviceType
from config.settings import NexusSettings

# Custom configuration
settings = NexusSettings(
    device_type=DeviceType.RASPBERRY_PI_5,
    model_path="./models/custom-model.gguf",
    ethical_override_enabled=True,
    min_confidence_threshold=0.8
)

nexus = NexusGuardianD7D(settings=settings)
```

---

## 🐛 Troubleshooting

### Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'src'`
```bash
# Solution: Ensure you're in the project directory
cd trinity-xai-exoplanets
python -c "import sys; print(sys.path)"
```

**Problem**: Model download fails
```bash
# Solution: Manual download
mkdir -p models
cd models
wget https://huggingface.co/TheBloke/Llama-2-3B-GGUF/resolve/main/llama-2-3b.Q4_K_M.gguf
```

**Problem**: Out of memory
```bash
# Solution: Use smaller model or adjust settings
# For devices with 4GB RAM, use phi-2-2.7b
bash scripts/model_downloader.sh phi-2-2.7b-q4_k_m
```

### Runtime Issues

**Problem**: Slow inference (< 2 tokens/sec)
```python
# Check your device optimization
from src.architecture import hardware_optimization

optimizer = hardware_optimization.HardwareOptimizer()
print(optimizer.benchmark_expectations())

# Solution: Verify ARM flags are enabled (for ARM devices)
# Or reduce model size
```

**Problem**: Ethical override too strict
```python
# Adjust confidence threshold
child = ChildContext(
    age=12,
    supervised=True,
    parental_controls={
        'strict_mode': False,  # Less strict
        'min_confidence': 0.6   # Lower threshold
    }
)
```

---

## 📚 Next Steps

### Learn More
- **Architecture**: [docs/ARCHITECTURE.md](ARCHITECTURE.md) - System design
- **Ethics**: [docs/ETHICAL_FRAMEWORK.md](ETHICAL_FRAMEWORK.md) - Ethical principles
- **Integration**: [docs/INTEGRATION_MAP.md](INTEGRATION_MAP.md) - How modules connect
- **Status**: [docs/STATUS.md](STATUS.md) - Project status

### Contribute
- **Contributing Guide**: [docs/CONTRIBUTING.md](CONTRIBUTING.md)
- **Code of Conduct**: [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)
- **Issues**: https://github.com/deegpnini/trinity-xai-exoplanets/issues

### Advanced Topics
1. **Split-Brain Setup**: Run sensorial + cognitive nodes
2. **Custom Vectors**: Add your own AI vector
3. **Fine-Tuning**: Adapt models with QLoRA
4. **RAG Customization**: Add your own knowledge base
5. **Multi-Device Clustering**: Connect multiple devices

---

## 🎓 Learning Path

### Beginner (Week 1)
- ✅ Complete this guide
- ✅ Run basic examples
- ✅ Understand decision structure
- ✅ Try different age contexts

### Intermediate (Week 2-3)
- 📖 Read architecture documentation
- 🧪 Explore RAG system
- ⚙️ Customize configuration
- 🎨 Build a simple application

### Advanced (Month 1+)
- 🏗️ Set up split-brain architecture
- 🔧 Contribute to the project
- 📝 Fine-tune models
- 🌐 Deploy for production use

---

## 💬 Getting Help

### Community Support
- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Report bugs or request features
- **Email**: hebron@trinity-xai.org (for private inquiries)

### Documentation
- **README**: [README.md](../README.md) - Project overview
- **API Docs**: Coming soon
- **Examples**: Check `examples/` directory

---

## 🌟 Best Practices

### DO's ✅
- Always use `ChildContext` with accurate age
- Review reasoning chains for transparency
- Start with supervised mode for young children
- Keep models updated for security
- Test thoroughly before production use

### DON'Ts ❌
- Don't disable ethical override (child safety!)
- Don't use with inaccurate child age
- Don't ignore blocked content warnings
- Don't run without parental awareness
- Don't modify safety features

---

## 🎉 Success!

You're now ready to use Nexus Guardian D7D! Remember:

> *"The measure of any system designed for children is not its capabilities, but its conscience."*

Every interaction is an opportunity to protect, educate, and inspire. Let's build something worthy of children's trust. 🌟

---

**Quick Links**:
- 📖 [Full Documentation](../README.md)
- 🔗 [GitHub Repository](https://github.com/deegpnini/trinity-xai-exoplanets)
- 💬 [Community Discussions](https://github.com/deegpnini/trinity-xai-exoplanets/discussions)
- 🐛 [Report Issues](https://github.com/deegpnini/trinity-xai-exoplanets/issues)

*Last Updated: 2026-02-10*
