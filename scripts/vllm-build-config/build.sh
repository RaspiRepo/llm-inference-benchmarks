
#!/usr/bin/env bash
set -Eeuo pipefail

# ==========================================
# vLLM SM120 / RTX 5060 Ti Build
# WSL2 / Blackwell
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPECTED_VENV="${HOME}/vllm/.venv"
SOURCE_DIR="${HOME}/vllm-sm120-build"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/build_${TIMESTAMP}.log"

# Log terminal output and errors.
exec > >(tee -a "${LOG_FILE}") 2>&1

trap 'rc=$?; echo; echo "BUILD FAILED (exit ${rc}) at line ${LINENO}"; echo "Log: ${LOG_FILE}"; exit "$rc"' ERR

echo "=========================================="
echo "vLLM SM120 BUILD"
echo "=========================================="
echo "Date: $(date)"
echo "Log: ${LOG_FILE}"

# ==========================================
# 1. Python environment
# ==========================================

if [[ "${VIRTUAL_ENV:-}" != "${EXPECTED_VENV}" ]]; then
    echo "ERROR: Activate the expected environment:"
    echo "source ${EXPECTED_VENV}/bin/activate"
    exit 1
fi

if [[ "$(command -v python)" != "${EXPECTED_VENV}/bin/python" ]]; then
    echo "ERROR: Unexpected Python executable."
    exit 1
fi

# ==========================================
# 2. Source directory
# ==========================================

cd "${SOURCE_DIR}"

if [[ ! -f pyproject.toml || ! -d vllm ]]; then
    echo "ERROR: Invalid vLLM source directory."
    exit 1
fi

# ==========================================
# 3. Load build environment
# ==========================================

source "${SCRIPT_DIR}/env.sh"

echo
echo "===== SOURCE INFORMATION ====="

git rev-parse HEAD
git status --short

echo
echo "===== PYTHON INFORMATION ====="

python --version
python -m pip --version

echo
echo "===== PYTORCH VERIFICATION ====="

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.get_device_capability() == (12, 0), \
    "Expected SM120"
assert torch.version.cuda.startswith("13.0"), \
    "Expected PyTorch CUDA 13.0"

print("GPU:", torch.cuda.get_device_name())
print("Capability:", torch.cuda.get_device_capability())
PY

echo
echo "===== CUDA TOOLCHAIN ====="

nvcc --version
ptxas --version

CUDA_VERSION="$(nvcc --version | grep -o 'release [0-9.]*' | awk '{print $2}')"

if [[ "${CUDA_VERSION}" != "13.1" ]]; then
    echo "ERROR: Expected CUDA toolkit 13.1"
    exit 1
fi

echo
echo "===== PREPARING PYTORCH ====="

python tools/use_existing_torch.py

echo
echo "===== BUILD DEPENDENCIES ====="

python -m pip install -r requirements/build/cuda.txt

echo
echo "===== BUILDING VLLM ====="

python -m pip install --no-build-isolation -e .

echo
echo "===== VERIFYING NATIVE EXTENSIONS ====="

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
    print("PASS:", name, module.__file__)

print("All native extension imports passed.")
PY

echo
echo "===== CUDA GEMM TEST ====="

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

assert y.shape == (2048, 2048)

print("CUDA GEMM: PASS")
print("Result:", y.shape)
print("Allocated GiB:",
      round(torch.cuda.memory_allocated() / 1024**3, 3))
PY

echo
echo "===== BUILD SUMMARY ====="

echo "Source commit:"
git rev-parse HEAD

echo "Python:"
python --version

echo "vLLM:"
python -c 'import vllm; print(vllm.__version__)'

echo
echo "=========================================="
echo "BUILD COMPLETE"
echo "=========================================="
echo "Log: ${LOG_FILE}"
