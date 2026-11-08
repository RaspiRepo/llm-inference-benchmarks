
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=========================================="
echo "vLLM SM120 Build Verification"
echo "=========================================="

echo
echo "===== Python and package versions ====="

python - <<'PY'
import sys
import torch
import triton
import vllm

print("Python:", sys.executable)
print("vLLM:", vllm.__file__)
print("vLLM version:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name())
print("Capability:", torch.cuda.get_device_capability())

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (12, 0)

print("\nCUDA availability: PASS")
PY

echo
echo "===== Native extensions ====="

python - <<'PY'
import importlib

modules = [
    "vllm._C_stable_libtorch",
    "vllm._moe_C_stable_libtorch",
    "vllm._qutlass_C",
    "vllm.fs_io_C",
    "vllm.cumem_allocator",
    "vllm.spinloop",
]

for name in modules:
    module = importlib.import_module(name)
    print("OK:", name, module.__file__)

print("\nNative extension imports: PASS")
PY

echo
echo "===== CUDA GEMM ====="

python - <<'PY'
import torch

x = torch.randn(2048, 2048, device="cuda")
y = x @ x

torch.cuda.synchronize()

print("CUDA GEMM: PASS")
print("Result:", y.shape)
print("Allocated:", round(
    torch.cuda.memory_allocated() / 1024**3, 3
), "GiB")
PY

echo
echo "===== FlashInfer ====="

python - <<'PY'
import flashinfer
print("FlashInfer:", flashinfer.__version__)
PY

echo
echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="
