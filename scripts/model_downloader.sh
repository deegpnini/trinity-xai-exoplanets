#!/bin/bash
#
# Model Downloader Script
# Downloads quantized models for Nexus Guardian D7D
#

set -e

echo "=== Nexus Guardian Model Downloader ==="
echo ""

# Default model directory
MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR"

# HuggingFace model URLs (examples - adjust to actual model locations)
declare -A MODELS=(
    ["phi-2-2.7b-q4_k_m"]="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
    ["llama-3.2-3b-q4_k_m"]="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    ["llama-3.2-1.5b-q4_k_m"]="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ["tinyllama-1.1b-q4_k_m"]="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

# Function to download model
download_model() {
    local model_name=$1
    local url=${MODELS[$model_name]}
    
    if [ -z "$url" ]; then
        echo "Error: Unknown model '$model_name'"
        echo "Available models:"
        for key in "${!MODELS[@]}"; do
            echo "  - $key"
        done
        return 1
    fi
    
    local filename="${model_name}.gguf"
    local filepath="$MODEL_DIR/$filename"
    
    if [ -f "$filepath" ]; then
        echo "Model already exists: $filepath"
        read -p "Re-download? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    echo "Downloading $model_name..."
    echo "URL: $url"
    echo "Destination: $filepath"
    echo ""
    
    wget -O "$filepath" "$url" --progress=bar:force 2>&1 | \
        grep --line-buffered -oP '\d+%' | \
        while read percentage; do
            echo -ne "Progress: $percentage\r"
        done
    
    echo ""
    
    if [ -f "$filepath" ]; then
        size=$(du -h "$filepath" | cut -f1)
        echo "Download complete! Size: $size"
        echo "Location: $filepath"
        return 0
    else
        echo "Error: Download failed"
        return 1
    fi
}

# Function to list available models
list_models() {
    echo "Available models:"
    echo ""
    for model in "${!MODELS[@]}"; do
        filepath="$MODEL_DIR/${model}.gguf"
        if [ -f "$filepath" ]; then
            size=$(du -h "$filepath" | cut -f1)
            echo "  ✓ $model ($size) - Downloaded"
        else
            echo "  ○ $model - Not downloaded"
        fi
    done
}

# Function to show recommendations
show_recommendations() {
    echo "Model Recommendations:"
    echo ""
    echo "For Raspberry Pi 5 (8GB RAM):"
    echo "  Primary: llama-3.2-3b-q4_k_m (Best balance)"
    echo "  Fallback: llama-3.2-1.5b-q4_k_m (Faster)"
    echo ""
    echo "For Galaxy A70 (6GB RAM):"
    echo "  Primary: phi-2-2.7b-q4_k_m (Optimized for mobile)"
    echo "  Fallback: tinyllama-1.1b-q4_k_m (Fastest)"
    echo ""
}

# Main script
if [ "$1" == "list" ]; then
    list_models
elif [ "$1" == "recommend" ]; then
    show_recommendations
elif [ -n "$1" ]; then
    download_model "$1"
else
    echo "Usage: $0 <model_name|list|recommend>"
    echo ""
    show_recommendations
    echo ""
    list_models
fi
