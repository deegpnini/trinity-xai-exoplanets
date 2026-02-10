#!/bin/bash
#
# Setup script for Termux (Android - Galaxy A70)
# Nexus Guardian D7D - Sensorial Node Setup
#

set -e

echo "=== Nexus Guardian D7D - Termux Setup ==="
echo "Setting up Sensorial Node on Galaxy A70"
echo ""

# Update package lists
echo "Updating package lists..."
pkg update -y
pkg upgrade -y

# Install essential packages
echo "Installing essential packages..."
pkg install -y \
    python \
    python-pip \
    git \
    cmake \
    clang \
    make \
    wget \
    proot-distro

# Install proot Ubuntu if requested
read -p "Install proot-distro Ubuntu? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Ubuntu in proot..."
    proot-distro install ubuntu
    echo "Ubuntu installed. Login with: proot-distro login ubuntu"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install numpy pyyaml requests

# Create directory structure
echo "Creating directory structure..."
mkdir -p ~/nexus-guardian
cd ~/nexus-guardian

mkdir -p models
mkdir -p logs
mkdir -p cache

# Create configuration file
cat > config.yaml << EOF
# Nexus Guardian D7D - Sensorial Node Configuration
node:
  type: sensorial
  device: galaxy_a70
  architecture: ARMv8.2-A

hardware:
  cpu_cores: 8
  ram_mb: 6144
  optimization_flags:
    - "-march=armv8.2-a"
    - "-mtune=cortex-a73"
    - "-O2"

model:
  name: Phi-2-2.7B-Q4_K_M
  max_context: 2048
  threads: 7
  batch_size: 128

paths:
  models: ~/nexus-guardian/models
  logs: ~/nexus-guardian/logs
  cache: ~/nexus-guardian/cache
EOF

echo "Configuration created: config.yaml"

# Setup swap file (if running as root)
if [ "$EUID" -eq 0 ]; then
    SWAP_SIZE=4096
    if [ ! -f ~/swapfile ]; then
        echo "Creating swap file (${SWAP_SIZE}MB)..."
        fallocate -l ${SWAP_SIZE}M ~/swapfile
        chmod 600 ~/swapfile
        mkswap ~/swapfile
        swapon ~/swapfile
        echo "Swap file created and activated"
    fi
else
    echo "Note: Run as root to create swap file"
fi

# Clone llama.cpp
echo "Cloning llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    
    echo "Building llama.cpp with ARM optimizations..."
    mkdir -p build
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=armv8.2-a -mtune=cortex-a73 -O2" \
        -DCMAKE_CXX_FLAGS="-march=armv8.2-a -mtune=cortex-a73 -O2"
    
    make -j7
    
    echo "llama.cpp built successfully!"
    cd ../..
else
    echo "llama.cpp already exists, skipping..."
fi

# Create run script
cat > run_sensorial.sh << 'EOF'
#!/bin/bash
# Run Nexus Guardian Sensorial Node

cd ~/nexus-guardian/llama.cpp/build

MODEL_PATH="../models/phi-2-2.7b-q4_k_m.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download the model first using model_downloader.sh"
    exit 1
fi

./bin/llama-server \
    --model "$MODEL_PATH" \
    --ctx-size 2048 \
    --threads 7 \
    --batch-size 128 \
    --port 8080 \
    --host 0.0.0.0

EOF

chmod +x run_sensorial.sh

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Download model: bash model_downloader.sh"
echo "2. Run sensorial node: bash run_sensorial.sh"
echo "3. Test at: http://localhost:8080"
echo ""
echo "Configuration file: ~/nexus-guardian/config.yaml"
echo "Logs directory: ~/nexus-guardian/logs"
echo ""
