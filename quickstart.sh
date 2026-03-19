#!/bin/bash
# Quick start script for llama.cpp SAE Feature Interpretability Pipeline
set -e

echo "=== SAE Feature Interpretability Pipeline Setup ==="

# Check prerequisites
command -v cmake >/dev/null 2>&1 || { echo "cmake is required but not installed."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required but not installed."; exit 1; }
command -v nvcc >/dev/null 2>&1 || echo "Warning: nvcc not found — CUDA build may fail"

# Step 1: Clone and patch llama.cpp
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 50 https://github.com/ggml-org/llama.cpp.git
fi

echo "Applying activation capture patch..."
cd llama.cpp
git apply ../patches/activation_capture.patch
cd ..

# Step 2: Build
echo "Building llama.cpp with CUDA..."
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -G Ninja 2>&1 | tail -5
cmake --build build --config Release -j $(nproc) 2>&1 | tail -5
cd ..

echo ""
echo "=== Build complete! ==="
echo ""
echo "Next steps:"
echo "  1. Start llama-server:"
echo "     ./llama.cpp/build/bin/llama-server -m your_model.gguf -c 4096 -ngl 999"
echo ""
echo "  2. Collect activations:"
echo "     python3 sae_pipeline/sae_collector.py --layer 20 --rows 500 --input your_data.jsonl"
echo ""
echo "  3. Train SAE:"
echo "     python3 sae_pipeline/sae_trainer.py --input sae_data/*.bin --epochs 5 --analyze"
echo ""
echo "  4. Probe features:"
echo "     python3 sae_pipeline/sae_probe.py --model sae_models/*.pt --text 'Your phrase here'"
echo ""
echo "See README.md for the full guide."
