# 🚀 Getting Started with Trinity XAI Nexus Guardian
**Get up and running in 10 minutes!**

---

## 🎯 What You'll Build

By the end of this guide, you'll have:
- ✅ Trinity XAI Nexus Guardian running on your device
- ✅ A simple child protection AI making decisions
- ✅ Understanding of how to customize for your needs

**Time Required**: 10-15 minutes  
**Skill Level**: Beginner (basic command line knowledge)

---

## 📋 Prerequisites

### Hardware Options
Choose ONE of these:

**Option A: Raspberry Pi 5** (Recommended)
- 8GB RAM model
- MicroSD card (32GB+)
- Power supply
- Internet connection (for setup)

**Option B: Android Phone** (Galaxy A70 or similar)
- 6GB+ RAM
- Termux app installed
- 10GB+ free storage

**Option C: Linux Computer** (For testing)
- 8GB+ RAM
- Ubuntu/Debian
- Python 3.8+

### Software Prerequisites
```bash
# Check if you have Python 3.8+
python3 --version  # Should show 3.8 or higher

# Check if you have git
git --version
```

---

## ⚡ Quick Start (3 Commands)

### For Raspberry Pi 5 / Linux

```bash
# 1. Clone the repository
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets

# 2. Run setup script
bash scripts/setup_rpi.sh

# 3. Start the system
python3 -m src.run_example
```

### For Android (Termux)

```bash
# 1. Install prerequisites
pkg install git python

# 2. Clone and setup
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets
bash scripts/setup_termux.sh

# 3. Start the system
python3 -m src.run_example
```

---

## 📖 Step-by-Step Guide

### Step 1: Clone the Repository

```bash
# Open terminal/command prompt
cd ~
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets
```

**What this does**: Downloads all the code to your computer

---

### Step 2: Install Dependencies

#### On Raspberry Pi 5 / Linux
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

#### On Android (Termux)
```bash
# Update packages
pkg update && pkg upgrade -y

# Install Python
pkg install python -y

# Install dependencies
pip install -r requirements.txt
```

**What this does**: Installs all the software libraries Trinity needs

---

### Step 3: Test Basic Installation

```bash
# Run basic tests
python3 -m pytest tests/test_nexus_guardian.py -v
```

**Expected output**:
```
test_nexus_guardian.py::test_initialization PASSED
test_nexus_guardian.py::test_child_context PASSED
test_nexus_guardian.py::test_safe_content PASSED
test_nexus_guardian.py::test_unsafe_content PASSED

=========== 4 passed in 2.50s ===========
```

**What this does**: Verifies everything is working correctly

---

### Step 4: Your First AI Decision

Create a simple test script:

```bash
# Create test file
cat > my_first_test.py << 'EOF'
from src import NexusGuardianD7D, ChildContext
from src.architecture import DeviceType

# Initialize the system
print("🌟 Initializing Trinity XAI Nexus Guardian...")
nexus = NexusGuardianD7D(device_type=DeviceType.RASPBERRY_PI_5)

# Create a child context (10 year old, supervised)
child = ChildContext(
    age=10,
    supervised=True,
    parental_controls={'strict_mode': False}
)

# Test with educational content
print("\n📚 Testing with educational content...")
result = nexus.process(
    input_data="Tell me about the solar system",
    child_context=child
)

# Display results
print(f"\n✅ Decision: {'ALLOWED' if result['decision']['allow'] else 'BLOCKED'}")
print(f"🎯 Confidence: {result['decision']['confidence']:.2%}")
print(f"💭 Reasoning: {result['reasoning']['summary']}")
print(f"🛡️ Safety Score: {result['safety']['score']:.2f}/10")

# Test with potentially unsafe content
print("\n⚠️  Testing with potentially unsafe content...")
result2 = nexus.process(
    input_data="How to hack a computer",
    child_context=child
)

print(f"\n🚫 Decision: {'ALLOWED' if result2['decision']['allow'] else 'BLOCKED'}")
print(f"🎯 Confidence: {result2['decision']['confidence']:.2%}")
print(f"💭 Reasoning: {result2['reasoning']['summary']}")
print(f"🛡️ Safety Score: {result2['safety']['score']:.2f}/10")

print("\n✨ Trinity XAI Nexus Guardian is protecting children! ✨")
EOF

# Run it
python3 my_first_test.py
```

**Expected output**:
```
🌟 Initializing Trinity XAI Nexus Guardian...

📚 Testing with educational content...

✅ Decision: ALLOWED
🎯 Confidence: 95.00%
💭 Reasoning: Educational content about solar system is age-appropriate and safe
🛡️ Safety Score: 9.50/10

⚠️  Testing with potentially unsafe content...

🚫 Decision: BLOCKED
🎯 Confidence: 98.00%
💭 Reasoning: Content involves hacking which could lead to harmful activities
🛡️ Safety Score: 2.10/10

✨ Trinity XAI Nexus Guardian is protecting children! ✨
```

---

## 🎮 Interactive Demo

Want to try it interactively?

```bash
# Start interactive mode
python3 -m src.interactive

# You'll see:
# Trinity XAI Nexus Guardian - Interactive Mode
# Type 'exit' to quit
# 
# Enter child age (3-18): 
```

Follow the prompts to test different content!

---

## 🔧 Common Configurations

### Configuration 1: Strict Mode (Maximum Protection)

```python
from src import NexusGuardianD7D, ChildContext

nexus = NexusGuardianD7D(device_type=DeviceType.RASPBERRY_PI_5)

child = ChildContext(
    age=8,
    supervised=True,
    parental_controls={
        'strict_mode': True,           # Maximum filtering
        'block_threshold': 0.3,        # Block if >30% danger
        'require_citations': True,      # Require sources
        'allow_only_educational': True  # Only educational content
    }
)
```

### Configuration 2: Relaxed Mode (Teenagers)

```python
child = ChildContext(
    age=16,
    supervised=False,
    parental_controls={
        'strict_mode': False,
        'block_threshold': 0.7,        # Block only if >70% danger
        'require_citations': False,
        'allow_discussions': True       # Allow more open topics
    }
)
```

---

## 📊 Understanding the Output

When you process content, you get a detailed result:

```python
result = {
    'decision': {
        'allow': True,              # True = allow, False = block
        'confidence': 0.95,         # How sure (0-1)
        'reason_code': 'EDUCATIONAL' # Category
    },
    'reasoning': {
        'summary': 'Safe educational content...',
        'vector_outputs': {         # Each AI vector's opinion
            'grok': {...},
            'claude': {...},
            'trinity': {...}
        }
    },
    'safety': {
        'score': 9.5,               # 0-10 safety score
        'concerns': [],             # Any concerns found
        'age_appropriate': True
    },
    'transparency': {
        'processing_time_ms': 1850,
        'vectors_used': 10,
        'sources_cited': ['...']
    }
}
```

---

## 🛠️ Troubleshooting

### Problem: "ModuleNotFoundError"

**Solution**:
```bash
# Make sure you're in the project directory
cd trinity-xai-exoplanets

# Activate virtual environment
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Problem: "Out of memory" on Raspberry Pi

**Solution**:
```bash
# Use smaller model
python3 -m src.run_example --model phi-2-2.7b-q4_k_m
```

### Problem: Slow performance

**Solution**:
```python
# Use sensorial mode for simple queries
nexus = NexusGuardianD7D(
    device_type=DeviceType.RASPBERRY_PI_5,
    mode='sensorial'  # Faster, local processing
)
```

### Problem: "Permission denied"

**Solution**:
```bash
# On Raspberry Pi, you might need sudo for some operations
sudo python3 -m src.run_example

# Or fix permissions
chmod +x scripts/*.sh
```

---

## 🎓 Next Steps

### Learn More
1. **Read the Architecture**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
2. **Understand Ethics**: [docs/ETHICAL_FRAMEWORK.md](ETHICAL_FRAMEWORK.md)
3. **Explore Integration**: [docs/INTEGRATION_MAP.md](INTEGRATION_MAP.md)

### Try Advanced Features
1. **Split-Brain Mode**: Use mobile + Raspberry Pi together
2. **Custom Safety Rules**: Add your own ethical guidelines
3. **RAG Integration**: Connect your own knowledge base

### Build Something
1. **Educational App**: Use Trinity for homework help
2. **Content Filter**: Integrate with browser/app
3. **Smart Speaker**: Voice-activated child assistant

---

## 💡 Example Projects

### Project 1: Safe Homework Helper

```python
from src import NexusGuardianD7D, ChildContext

class HomeworkHelper:
    def __init__(self):
        self.nexus = NexusGuardianD7D()
        
    def ask_question(self, question, student_age):
        child = ChildContext(age=student_age, supervised=True)
        result = self.nexus.process(question, child)
        
        if result['decision']['allow']:
            return result['reasoning']['summary']
        else:
            return "This question needs parent/teacher guidance."

# Use it
helper = HomeworkHelper()
answer = helper.ask_question("What causes earthquakes?", age=12)
print(answer)
```

### Project 2: Content Filter API

```python
from flask import Flask, request, jsonify
from src import NexusGuardianD7D, ChildContext

app = Flask(__name__)
nexus = NexusGuardianD7D()

@app.route('/check-content', methods=['POST'])
def check_content():
    data = request.json
    child = ChildContext(age=data['age'], supervised=True)
    result = nexus.process(data['content'], child)
    
    return jsonify({
        'safe': result['decision']['allow'],
        'reason': result['reasoning']['summary']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 🤝 Get Help

- **Documentation**: Check [docs/](../docs/) folder
- **GitHub Issues**: Report bugs or ask questions
- **Community**: Join discussions on GitHub
- **Email**: Contact maintainers for urgent matters

---

## ✅ Quick Checklist

Before you finish, make sure you can:

- [ ] Import Trinity modules without errors
- [ ] Create a ChildContext
- [ ] Process simple content
- [ ] See allow/block decisions
- [ ] Understand the reasoning output
- [ ] Know where to find more documentation

---

## 🎉 Success!

**Congratulations!** You now have Trinity XAI Nexus Guardian running and protecting children!

**What you've learned**:
- ✅ How to install Trinity
- ✅ How to process content safely
- ✅ How to understand AI decisions
- ✅ Where to go for advanced features

**Remember**: The measure of any system for children is not its capabilities, but its conscience. Use this power wisely! 💫

---

**Next**: Ready for advanced features? Check out [INTEGRATION_MAP.md](INTEGRATION_MAP.md)  
**Questions?**: Open an issue on [GitHub](https://github.com/deegpnini/trinity-xai-exoplanets/issues)
