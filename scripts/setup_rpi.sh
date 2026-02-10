#!/bin/bash
#
# Setup script for Raspberry Pi 5
# Nexus Guardian D7D - Cognitive Node Setup
#

set -e

echo "=== Nexus Guardian D7D - Raspberry Pi 5 Setup ==="
echo "Setting up Cognitive Node"
echo ""

# Update system
echo "Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

# Install essential packages
echo "Installing essential packages..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    wget \
    curl

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install numpy chromadb pyyaml requests pydantic

# Create directory structure
echo "Creating directory structure..."
mkdir -p ~/nexus-guardian-rpi
cd ~/nexus-guardian-rpi

mkdir -p models
mkdir -p logs
mkdir -p cache
mkdir -p knowledge_base

# Create configuration file
cat > config.yaml << EOF
# Nexus Guardian D7D - Cognitive Node Configuration
node:
  type: cognitive
  device: raspberry_pi_5
  architecture: ARMv8.2-A

hardware:
  cpu_cores: 4
  ram_mb: 8192
  optimization_flags:
    - "-march=armv8.2-a"
    - "-mtune=cortex-a76"
    - "-O3"
    - "-ffast-math"

model:
  name: Llama-3.2-3B-Q4_K_M
  max_context: 4096
  threads: 3
  batch_size: 512

rag:
  enabled: true
  collection_name: nexus_knowledge
  embedding_model: all-MiniLM-L6-v2

paths:
  models: ~/nexus-guardian-rpi/models
  logs: ~/nexus-guardian-rpi/logs
  cache: ~/nexus-guardian-rpi/cache
  knowledge_base: ~/nexus-guardian-rpi/knowledge_base
EOF

echo "Configuration created: config.yaml"

# Setup swap file
SWAP_SIZE=8192
if [ ! -f /swapfile ]; then
    echo "Creating swap file (${SWAP_SIZE}MB)..."
    sudo fallocate -l ${SWAP_SIZE}M /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap file created and activated"
fi

# Clone llama.cpp
echo "Cloning llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    
    echo "Building llama.cpp with ARM and OpenBLAS optimizations..."
    mkdir -p build
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=armv8.2-a -mtune=cortex-a76 -O3 -ffast-math" \
        -DCMAKE_CXX_FLAGS="-march=armv8.2-a -mtune=cortex-a76 -O3 -ffast-math" \
        -DGGML_OPENBLAS=ON \
        -DGGML_NEON=ON
    
    make -j3
    
    echo "llama.cpp built successfully!"
    cd ../..
else
    echo "llama.cpp already exists, skipping..."
fi

# Initialize ChromaDB
echo "Initializing ChromaDB..."
python3 << PYEOF
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./knowledge_base"
))

collection = client.get_or_create_collection("nexus_knowledge")
print(f"ChromaDB initialized with collection: {collection.name}")
PYEOF

# Create run script
cat > run_cognitive.sh << 'EOF'
#!/bin/bash
# Run Nexus Guardian Cognitive Node

cd ~/nexus-guardian-rpi/llama.cpp/build

MODEL_PATH="../models/llama-3.2-3b-q4_k_m.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download the model first using model_downloader.sh"
    exit 1
fi

./bin/llama-server \
    --model "$MODEL_PATH" \
    --ctx-size 4096 \
    --threads 3 \
    --batch-size 512 \
    --port 8080 \
    --host 0.0.0.0 \
    --mlock

EOF

chmod +x run_cognitive.sh

# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
# Monitor Nexus Guardian System

echo "=== Nexus Guardian System Monitor ==="
echo ""

# CPU temperature
if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
    TEMP_C=$((TEMP / 1000))
    echo "CPU Temperature: ${TEMP_C}°C"
fi

# Memory usage
echo ""
echo "Memory Usage:"
free -h

# Disk usage
echo ""
echo "Disk Usage:"
df -h ~/nexus-guardian-rpi

# Check if server is running
echo ""
if pgrep -f llama-server > /dev/null; then
    echo "Status: llama-server is RUNNING"
    echo "Port: 8080"
else
    echo "Status: llama-server is NOT RUNNING"
fi

EOF

chmod +x monitor.sh

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Download model: bash model_downloader.sh llama-3.2-3b-q4_k_m"
echo "2. Run cognitive node: bash run_cognitive.sh"
echo "3. Monitor system: bash monitor.sh"
echo "4. Test at: http://localhost:8080"
echo ""
echo "Configuration file: ~/nexus-guardian-rpi/config.yaml"
echo "Logs directory: ~/nexus-guardian-rpi/logs"
echo "Knowledge base: ~/nexus-guardian-rpi/knowledge_base"
echo ""
