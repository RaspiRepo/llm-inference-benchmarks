#!/usr/bin/env bash

# ==========================================
# RTX 5060 Ti / Blackwell / WSL2
# vLLM build and inference environment
# ==========================================

# CUDA toolchain
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Triton compiler
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"

# GPU architecture
export TORCH_CUDA_ARCH_LIST="12.0"
export VLLM_TARGET_DEVICE=cuda

# Build parallelism - WSL 23GB RAM
export MAX_JOBS=4
export NVCC_THREADS=2
export CMAKE_BUILD_PARALLEL_LEVEL=4

# Python
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# CUDA diagnostics
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Display active configuration
echo "=========================================="
echo "vLLM Blackwell environment"
echo "=========================================="
echo "CUDA_HOME:          ${CUDA_HOME}"
echo "NVCC:               $(nvcc --version | tail -1)"
echo "PTXAS:              ${TRITON_PTXAS_PATH}"
echo "GPU architecture:   ${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS:           ${MAX_JOBS}"
echo "NVCC_THREADS:       ${NVCC_THREADS}"
echo "=========================================="
