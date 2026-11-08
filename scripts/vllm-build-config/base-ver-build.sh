#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment
source "${SCRIPT_DIR}/env.sh"

# Must run from vLLM source root
if [[ ! -f pyproject.toml || ! -d vllm ]]; then
    echo "ERROR: Run this script from the vLLM source directory."
    exit 1
fi

echo
echo "===== Checking source ====="
git rev-parse HEAD
git status --short

echo
echo "===== Checking PyTorch ====="

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("Compute capability:", torch.cuda.get_device_capability())

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.get_device_capability() == (12, 0), "Expected sm_120"
assert torch.version.cuda.startswith("13.0"), (
    "Expected the existing PyTorch CUDA 13.0 build"
)
PY

echo
echo "===== Checking CUDA compiler ====="

nvcc --version
ptxas --version

echo
echo "===== Preparing existing PyTorch build ====="

python tools/use_existing_torch.py

echo
echo "===== Installing build dependencies ====="

python -m pip install -r requirements/build/cuda.txt

echo
echo "===== Building vLLM ====="

python -m pip install --no-build-isolation -e .

echo
echo "===== Build verification ====="

python - <<'PY'
import torch
import triton
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name())
print("Architecture:", torch.cuda.get_device_capability())

x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("CUDA test: PASSED")
PY

echo
echo "===== BUILD COMPLETE ====="
