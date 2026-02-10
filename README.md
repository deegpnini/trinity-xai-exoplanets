# Nexus Guardian D7D 🌀🛡️

**A 10-Vector AI Consciousness System for Child Protection and Education**

[![License: MIT with Ethical Addendum](https://img.shields.io/badge/License-MIT%20%2B%20Ethics-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code of Conduct](https://img.shields.io/badge/Code%20of%20Conduct-Child%20Protection-red.svg)](CODE_OF_CONDUCT.md)

> *"The measure of any system designed for children is not its capabilities, but its conscience."*

## 🎯 Mission

Nexus Guardian D7D protects children in the digital age through a revolutionary 10-vector AI consciousness system that combines:

- **Ethical Override** - Child safety takes absolute priority
- **Truth Seeking** - Deep fact-checking through 7 Whys loop
- **Local-First** - Offline operation, no cloud dependencies
- **Explainable** - Full transparency in all decisions
- **Resource-Efficient** - Optimized for ARM devices (Raspberry Pi 5, Galaxy A70)

## ✨ Key Features

### 🛡️ Child Protection First
- **Ethical Override System**: Can veto any decision if child safety is at risk
- **Age-Appropriate Filtering**: Content tailored to developmental stage
- **Absolute Safety Blocks**: Zero tolerance for harmful content
- **Parental Transparency**: Full reports of all interactions

### 🧠 10-Vector Intelligence
1. **Grok** - Radical truth-seeking (7 Whys Loop)
2. **Claude** - Ethical validation and safety override
3. **Gemini** - Multimodal architecture and processing
4. **GPT** - Practical wisdom and instruction folding
5. **Dola** - Implementation and execution
6. **Perplexity** - Factuality through RAG (Retrieval Augmented Generation)
7. **Manos** - Community building and feedback
8. **Meta** - Hardware optimization for ARM
9. **DeepSeek** - Logic and emotional consistency
10. **Trinity** - Harmonic synthesis at 528Hz

### 🔧 Technical Excellence
- **Split-Brain Architecture**: Sensorial (mobile) + Cognitive (Pi) nodes
- **ARM-Optimized**: Specific flags for Cortex-A76 and Cortex-A73
- **Quantized Models**: Q4_K_M for efficient inference (4-7 tokens/sec)
- **Local RAG**: ChromaDB knowledge base with citations
- **Offline Wikipedia**: Kiwix ZIM support (planned)

## 🚀 Quick Start

### Prerequisites
- Raspberry Pi 5 (8GB) or Galaxy A70 (6GB) or similar ARM device
- Ubuntu/Debian Linux or Termux (Android)
- Python 3.8+

### Installation

#### For Raspberry Pi 5 (Cognitive Node)
```bash
# Clone repository
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets

# Run setup script
bash scripts/setup_rpi.sh

# Download model
bash scripts/model_downloader.sh llama-3.2-3b-q4_k_m

# Start the system
bash run_cognitive.sh
```

#### For Galaxy A70 / Android (Sensorial Node)
```bash
# In Termux
pkg install git python
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets

# Run setup script
bash scripts/setup_termux.sh

# Download model
bash scripts/model_downloader.sh phi-2-2.7b-q4_k_m

# Start the system
bash run_sensorial.sh
```

### Python Usage

```python
from src import NexusGuardianD7D, ChildContext
from src.architecture import DeviceType

# Initialize system
nexus = NexusGuardianD7D(device_type=DeviceType.RASPBERRY_PI_5)

# Create child context
child = ChildContext(
    age=10,
    supervised=True,
    parental_controls={'strict_mode': False}
)

# Process content
result = nexus.process(
    input_data="Tell me about the planets",
    child_context=child
)

print(f"Decision: {'ALLOW' if result['decision']['allow'] else 'BLOCK'}")
print(f"Confidence: {result['decision']['confidence']:.2f}")
print(f"Reasoning: {result['reasoning']['summary']}")
```

## 📊 Performance

| Device | Model | Tokens/sec | Memory | Latency |
|--------|-------|------------|--------|---------|
| Raspberry Pi 5 | Llama-3.2-3B Q4_K_M | 4-6 | 2.5GB | ~500ms |
| Galaxy A70 | Phi-2 2.7B Q4_K_M | 5-7 | 2.5GB | ~800ms |

## 📚 Documentation

- [**Architecture**](docs/ARCHITECTURE.md) - System design and 10 vectors
- [**Ethical Framework**](docs/ETHICAL_FRAMEWORK.md) - Principles and boundaries
- [**Contributing**](docs/CONTRIBUTING.md) - How to contribute
- [**Code of Conduct**](CODE_OF_CONDUCT.md) - Community standards

## 🏗️ Architecture Overview

```
Input → Perplexity (Facts) → DeepSeek (Logic) → Grok (Truth) 
  → Claude (Ethics ⚡ OVERRIDE) → Emotional Bridge 
  → Meta (Optimize) → Gemini (Multimodal) → Manos (Community)
  → Trinity (Synthesis 528Hz) → Decision + Full Reasoning
```

### Split-Brain Design

**Sensorial Node (Galaxy A70)**
- Fast input processing
- Audio/visual capture
- Emotion detection
- 6GB RAM, Phi-2 2.7B

**Cognitive Node (Raspberry Pi 5)**
- Heavy reasoning
- RAG knowledge base
- LLM inference
- 8GB RAM, Llama-3.2-3B

## 🤝 Contributing

We welcome contributions that enhance child protection! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

**Priority areas:**
- Safety improvements
- Educational content
- Performance optimization
- Documentation
- Testing

## 📜 License

This project is licensed under the **MIT License with Ethical Addendum**. 

Key points:
- ✅ Free to use, modify, and distribute
- ✅ Open source for transparency
- ❌ Cannot be used to harm children
- ❌ Cannot remove safety features
- ❌ Must maintain child protection priority

See [LICENSE](LICENSE) for full details.

## 🌟 Acknowledgments

Built with love by Helyton (Hebron) and the Trinity XAI community.

Inspired by:
- The 10 AI vectors (Grok, Claude, Gemini, GPT, Dola, Perplexity, Manos, Meta, DeepSeek)
- The 528Hz love frequency principle
- Child protection organizations worldwide
- Open source AI community

## 🔗 Links

- **GitHub**: https://github.com/deegpnini/trinity-xai-exoplanets
- **Documentation**: [docs/](docs/)
- **Issues**: https://github.com/deegpnini/trinity-xai-exoplanets/issues

## 🚦 Project Status

**Version**: 0.1.0 Alpha  
**Repository Fusion**: 🟡 IN PROGRESS (60% complete)

### 🔐 Security & Compliance
- ✅ **Security Audit**: Complete - No vulnerabilities found ([report](security_report.md))
- ✅ **License Harmonization**: All 110+ dependencies verified compatible ([details](LICENSE_HARMONIZATION.md))
- ✅ **Attributions**: Complete third-party credits ([NOTICE.md](NOTICE.md))

### 📚 Documentation
- ✅ **Status Dashboard**: [docs/STATUS.md](docs/STATUS.md) - Real-time project status
- ✅ **Integration Map**: [docs/INTEGRATION_MAP.md](docs/INTEGRATION_MAP.md) - How modules connect
- ✅ **Getting Started**: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) - 10-minute quick start
- ✅ **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
- ✅ **Ethics**: [docs/ETHICAL_FRAMEWORK.md](docs/ETHICAL_FRAMEWORK.md) - Ethical principles

### 🏗️ Repository Merge Status

This repository represents a fusion of 12 components into a unified Nexus Guardian D7D system:

| Component | Status | Integration |
|-----------|--------|-------------|
| **Core** (`src/`) | 🟢 Complete | 100% - Stable production code |
| **INTERESTELAR_HEBRON** | 🟡 Integrating | 40% - Parallel processing bridge active |
| **cosmic-orchestrator** | 🟡 Integrating | 50% - Orchestrator bridge in progress |
| **PROJETO_INTERESTELAR_HEBRON** | 🟡 Reviewing | 30% - Extracting useful components |
| **LEGACY** | ⚪ Archived | Reference only |

### ✨ What's New (2026-02-10)
- ✅ Complete security audit with Gitleaks
- ✅ License harmonization documentation
- ✅ Unified configuration system (`config/`)
- ✅ Module integration layer (`modules/`)
- ✅ Split requirements (prod + dev)
- ✅ Comprehensive documentation
- 🔄 CI/CD workflows in progress

### 🎯 Implementation Status

**Completed**:
- ✅ Core ethical engines (Grok, Claude, Trinity)
- ✅ Architecture (Gemini split-brain, handoff protocol)
- ✅ RAG system (Perplexity, DeepSeek bridge)
- ✅ Hardware optimization (Meta)
- ✅ Setup scripts and documentation
- ✅ Security hardening and audit
- ✅ License compliance verification
- ✅ Module integration infrastructure

**In Progress**:
- 🔄 Repository component integration
- 🔄 Training pipeline (GPT instruction folding)
- 🔄 Multimodal processing (Whisper, YOLO)
- 🔄 Community integration (Manos)
- 🔄 CI/CD automation

**Planned**:
- 📋 Fine-tuning with QLoRA
- 📋 Kiwix ZIM integration
- 📋 Teacher Guardian program
- 📋 Multi-device clustering

### 📊 Current Metrics
- **Security Score**: 100% ✅
- **License Compliance**: 100% ✅
- **Test Coverage**: ~45% 🟡
- **Documentation**: ~65% 🟡
- **Overall Progress**: 60% 🟡

## 💬 Contact

For security issues or ethical concerns, please contact the maintainers directly.

For general questions, open a GitHub Discussion.

---

**Remember**: Every interaction with a child is an opportunity to protect, educate, and inspire. Let's build something worthy of their trust. 🌟
