"""
Hardware Optimization - Meta Vector
Vector 8/10: ARM Optimizations and Performance Tuning

This module implements Meta's hardware optimization strategies
for ARM-based devices (Raspberry Pi 5, Galaxy A70).
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    """Supported device types."""
    RASPBERRY_PI_5 = "raspberry_pi_5"
    GALAXY_A70 = "galaxy_a70"
    GENERIC_ARM = "generic_arm"


@dataclass
class HardwareProfile:
    """Hardware profile for optimization."""
    device_type: DeviceType
    cpu_cores: int
    ram_mb: int
    architecture: str
    has_neon: bool
    optimization_flags: List[str]


class HardwareOptimization:
    """
    Meta Hardware Optimization - ARM-specific optimizations.
    
    Provides compilation flags, runtime optimizations, and performance
    tuning for ARM devices.
    """
    
    # Optimization profiles for different devices
    DEVICE_PROFILES = {
        DeviceType.RASPBERRY_PI_5: HardwareProfile(
            device_type=DeviceType.RASPBERRY_PI_5,
            cpu_cores=4,
            ram_mb=8192,
            architecture="ARMv8.2-A",
            has_neon=True,
            optimization_flags=[
                "-march=armv8.2-a",
                "-mtune=cortex-a76",
                "-mfpu=neon-fp-armv8",
                "-O3",
                "-ffast-math",
                "-DGGML_USE_ACCELERATE"
            ]
        ),
        DeviceType.GALAXY_A70: HardwareProfile(
            device_type=DeviceType.GALAXY_A70,
            cpu_cores=8,
            ram_mb=6144,
            architecture="ARMv8.2-A",
            has_neon=True,
            optimization_flags=[
                "-march=armv8.2-a",
                "-mtune=cortex-a73",
                "-mfpu=neon",
                "-O2",
                "-ffast-math"
            ]
        )
    }
    
    # Model recommendations by device
    MODEL_RECOMMENDATIONS = {
        DeviceType.RASPBERRY_PI_5: {
            'primary': 'Llama-3.2-3B-Q4_K_M',
            'fallback': 'Llama-3.2-1.5B-Q4_K_M',
            'expected_tokens_per_sec': (4, 6),  # 3B model range
            'max_context': 4096
        },
        DeviceType.GALAXY_A70: {
            'primary': 'Phi-2-2.7B-Q4_K_M',
            'fallback': 'TinyLlama-1.1B-Q4_K_M',
            'expected_tokens_per_sec': (5, 7),
            'max_context': 2048
        }
    }
    
    def __init__(self, device_type: DeviceType):
        self.device_type = device_type
        self.profile = self.DEVICE_PROFILES.get(
            device_type,
            self._create_generic_profile()
        )
        
    def get_compilation_flags(self) -> List[str]:
        """
        Get compilation flags for llama.cpp.
        
        Returns:
            List of compiler flags optimized for device
        """
        return self.profile.optimization_flags
    
    def get_cmake_options(self) -> Dict[str, str]:
        """
        Get CMake options for building llama.cpp.
        
        Returns:
            Dict of CMake options
        """
        options = {
            'CMAKE_BUILD_TYPE': 'Release',
            'CMAKE_C_FLAGS': ' '.join(self.profile.optimization_flags),
            'CMAKE_CXX_FLAGS': ' '.join(self.profile.optimization_flags),
        }
        
        # Enable NEON if available
        if self.profile.has_neon:
            options['GGML_NEON'] = 'ON'
        
        # Device-specific options
        if self.device_type == DeviceType.RASPBERRY_PI_5:
            options['GGML_OPENBLAS'] = 'ON'
            options['GGML_ACCELERATE'] = 'ON'
        
        return options
    
    def get_model_recommendation(self) -> Dict[str, Any]:
        """
        Get recommended model for this device.
        
        Returns:
            Model recommendation with performance expectations
        """
        return self.MODEL_RECOMMENDATIONS.get(
            self.device_type,
            {
                'primary': 'Phi-2-2.7B-Q4_K_M',
                'fallback': 'TinyLlama-1.1B-Q4_K_M',
                'expected_tokens_per_sec': (3, 5),
                'max_context': 2048
            }
        )
    
    def get_runtime_config(self) -> Dict[str, Any]:
        """
        Get runtime configuration for inference.
        
        Returns:
            Runtime configuration optimized for device
        """
        # Calculate optimal thread count
        threads = max(1, self.profile.cpu_cores - 1)  # Leave one core for OS
        
        # Calculate batch size based on RAM
        if self.profile.ram_mb >= 8192:
            batch_size = 512
        elif self.profile.ram_mb >= 4096:
            batch_size = 256
        else:
            batch_size = 128
        
        return {
            'threads': threads,
            'batch_size': batch_size,
            'context_size': self.get_model_recommendation()['max_context'],
            'use_mmap': True,
            'use_mlock': self.profile.ram_mb >= 8192,  # Only if enough RAM
            'n_gpu_layers': 0,  # CPU only for ARM devices
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """
        Get memory management configuration.
        
        Returns:
            Memory configuration with swap recommendations
        """
        # Calculate swap file size
        swap_mb = max(4096, self.profile.ram_mb)
        
        return {
            'physical_ram_mb': self.profile.ram_mb,
            'recommended_swap_mb': swap_mb,
            'swap_priority': 10,  # Higher priority = use RAM first
            'cache_size_mb': min(2048, self.profile.ram_mb // 4),
            'preload_model': self.profile.ram_mb >= 8192
        }
    
    def get_power_config(self) -> Dict[str, Any]:
        """
        Get power management configuration.
        
        Returns:
            Power configuration for battery devices
        """
        is_battery_device = self.device_type == DeviceType.GALAXY_A70
        
        return {
            'performance_mode': not is_battery_device,
            'cpu_governor': 'performance' if not is_battery_device else 'conservative',
            'thermal_throttle_threshold': 80,  # Celsius
            'idle_timeout_seconds': 300 if is_battery_device else 0
        }
    
    def benchmark_expectations(self) -> Dict[str, Any]:
        """
        Get expected benchmark results.
        
        Returns:
            Expected performance metrics
        """
        recommendation = self.get_model_recommendation()
        
        return {
            'model': recommendation['primary'],
            'tokens_per_second': {
                'min': recommendation['expected_tokens_per_sec'][0],
                'max': recommendation['expected_tokens_per_sec'][1],
                'average': sum(recommendation['expected_tokens_per_sec']) / 2
            },
            'latency_ms': {
                'first_token': 500 if self.device_type == DeviceType.RASPBERRY_PI_5 else 800,
                'subsequent_tokens': 100 if self.device_type == DeviceType.RASPBERRY_PI_5 else 150
            },
            'memory_usage_mb': {
                'model_size': 2000,  # Approximate for Q4_K_M
                'runtime_overhead': 500,
                'total_estimated': 2500
            }
        }
    
    def _create_generic_profile(self) -> HardwareProfile:
        """Create generic ARM profile."""
        return HardwareProfile(
            device_type=DeviceType.GENERIC_ARM,
            cpu_cores=4,
            ram_mb=4096,
            architecture="ARMv8-A",
            has_neon=True,
            optimization_flags=["-march=armv8-a", "-O2", "-ffast-math"]
        )
    
    def generate_setup_script(self) -> str:
        """
        Generate setup script for device optimization.
        
        Returns:
            Bash script content
        """
        flags = ' '.join(self.get_compilation_flags())
        runtime = self.get_runtime_config()
        memory = self.get_memory_config()
        
        script = f"""#!/bin/bash
# Hardware Optimization Setup for {self.device_type.value}
# Generated by Nexus Guardian D7D

set -e

echo "=== Nexus Guardian Hardware Optimization ==="
echo "Device: {self.device_type.value}"
echo "Architecture: {self.profile.architecture}"
echo "CPU Cores: {self.profile.cpu_cores}"
echo "RAM: {self.profile.ram_mb}MB"

# Install dependencies
echo "Installing dependencies..."
apt-get update
apt-get install -y build-essential cmake git

# Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

# Build with optimizations
echo "Building with optimizations..."
mkdir -p build
cd build

cmake .. \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_C_FLAGS="{flags}" \\
    -DCMAKE_CXX_FLAGS="{flags}"

make -j{runtime['threads']}

echo "Build complete!"

# Setup swap if needed
SWAP_SIZE={memory['recommended_swap_mb']}
if [ ! -f /swapfile ]; then
    echo "Creating swap file..."
    sudo fallocate -l ${{SWAP_SIZE}}M /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "=== Setup Complete ==="
echo "Recommended model: {self.get_model_recommendation()['primary']}"
echo "Expected performance: {self.benchmark_expectations()['tokens_per_second']['average']:.1f} tokens/sec"
"""
        return script
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get complete optimization summary."""
        return {
            'device': self.device_type.value,
            'profile': {
                'architecture': self.profile.architecture,
                'cores': self.profile.cpu_cores,
                'ram_mb': self.profile.ram_mb,
                'neon_support': self.profile.has_neon
            },
            'compilation': {
                'flags': self.get_compilation_flags(),
                'cmake_options': self.get_cmake_options()
            },
            'runtime': self.get_runtime_config(),
            'memory': self.get_memory_config(),
            'power': self.get_power_config(),
            'model_recommendation': self.get_model_recommendation(),
            'benchmarks': self.benchmark_expectations()
        }


# Example usage
if __name__ == "__main__":
    # Raspberry Pi 5 optimization
    rpi5 = HardwareOptimization(DeviceType.RASPBERRY_PI_5)
    print("=== Raspberry Pi 5 Optimization ===")
    print(f"Compilation flags: {' '.join(rpi5.get_compilation_flags())}")
    print(f"Recommended model: {rpi5.get_model_recommendation()['primary']}")
    print(f"Expected performance: {rpi5.benchmark_expectations()['tokens_per_second']}")
    
    # Galaxy A70 optimization
    a70 = HardwareOptimization(DeviceType.GALAXY_A70)
    print("\n=== Galaxy A70 Optimization ===")
    print(f"Runtime config: {a70.get_runtime_config()}")
    print(f"Memory config: {a70.get_memory_config()}")
    
    # Generate setup script
    script = rpi5.generate_setup_script()
    print(f"\nSetup script length: {len(script)} bytes")
