Excellent. This confirms that our current build passes all three verification stages. We now have a reliable baseline to preserve.

All verification tests passed## 1. Verification results

| Component | Result |
| --- | --- |
| Python virtual environment | PASS |
| vLLM source path | PASS |
| PyTorch 2.14.0+cu130 | PASS |
| CUDA runtime 13.0 | PASS |
| CUDA toolkit 13.1 | PASS |
| RTX 5060 Ti / SM120 | PASS |
| Triton 3.8.0 | PASS |
| Six native extensions | PASS |
| CUDA GEMM | PASS |
| FlashInfer 0.7.2.post1 | PASS |

One important distinction: this verifies that the extensions load and CUDA computation works. It does not yet test every vLLM kernel or serving configuration.

## 2. One improvement to `env.sh`

Your environment script works. I would add explicit virtual-environment validation so we cannot accidentally build using system Python.

Add this after the CUDA configuration:

bash

```bash
# Expected Python environment
EXPECTED_VENV="${HOME}/vllm/.venv"

if [[ "${VIRTUAL_ENV:-}" != "${EXPECTED_VENV}" ]]; then
    echo "ERROR: Activate the expected vLLM environment first:"
    echo "source ${EXPECTED_VENV}/bin/activate"
    return 1 2>/dev/null || exit 1
fi

echo "Python environment: ${VIRTUAL_ENV}"
```

This protects against accidentally compiling with a different PyTorch installation.

## 3. Update BUILD\_HISTORY.md

Your existing document is good, but it only records the environment. We should expand it to include the build and inference milestones.

I recommend replacing it with this:

Writing

# vLLM SM120 Build History

## 1\. Hardware and Environment

- Platform: WSL2
- GPU: NVIDIA GeForce RTX 5060 Ti
- GPU architecture: Blackwell
- Compute capability: 12.0 (SM120)
- NVIDIA Driver: 610.88

## 2\. Directory Layout

- Python environment: `~/vllm/.venv`
- vLLM executable: `~/vllm/.venv/bin/vllm`
- Custom source: `~/vllm-sm120-build`
- Build configuration: `~/vllm-build-config`

The Python environment and source checkout are intentionally separate.

## 3\. Software Versions

| Component | Version |
| --- | --- |
| vLLM | 0.33.0rc1.dev321+g6d0189348 |
| PyTorch | 2.14.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| CUDA toolkit | 13.1 |
| Triton | 3.8.0 |
| FlashInfer | 0.7.2.post1 |
| Python | 3.12 |

## 4\. Build Configuration

- CUDA\_HOME: `/usr/local/cuda-13.1`
- TORCH\_CUDA\_ARCH\_LIST: `12.0`
- VLLM\_TARGET\_DEVICE: `cuda`
- MAX\_JOBS: 4
- NVCC\_THREADS: 2
- CMAKE\_BUILD\_PARALLEL\_LEVEL: 4
- Existing PyTorch installation is reused.
- Build isolation disabled.

Build command:

bash

```bash
python tools/use_existing_torch.py
python -m pip install -r requirements/build/cuda.txt
python -m pip install --no-build-isolation -e .
```

## 5\. Native Extension Verification

Successfully imported:

- `_C_stable_libtorch`
- `_moe_C_stable_libtorch`
- `_qutlass_C`
- `fs_io_C`
- `cumem_allocator`
- `spinloop`

Note: This source build uses stable LibTorch extensions. Do not use the legacy `vllm._C` import as the sole verification test.

## 6\. CUDA Verification

- CUDA availability: PASS
- GPU capability: (12, 0)
- CUDA GEMM, 2048 × 2048: PASS
- Native extension imports: PASS
- FlashInfer import: PASS

## 7\. Inference Verification

Model:

`Qwen/Qwen2.5-7B-Instruct-AWQ`

Server configuration:

- Attention backend: TRITON\_ATTN
- Data type: half
- Maximum model length: 2048
- Maximum sequences: 4
- GPU memory utilization: 0.85
- Eager execution enabled
- API port: 8000

Verified:

- Server startup: PASS
- HTTP `/health`: 200 OK
- OpenAI-compatible completion: PASS
- GPU computation during inference: PASS

## 8\. Initial Benchmark

Benchmark configuration:

- Dataset: random
- Input length: 1024
- Output length: 256
- Requests: 100
- Request rate: 5 RPS

Results:

| Metric | Result |
| --- | --- |
| Successful requests | 100 |
| Failed requests | 0 |
| Output throughput | 151.93 tok/s |
| Total token throughput | 759.65 tok/s |
| Mean TTFT | 70.44 seconds |
| Mean TPOT | 24.04 ms |
| P99 TPOT | 28.62 ms |

The high TTFT requires further investigation under different concurrency and request-rate settings.

## 9\. Known Issues and Investigation

Earlier investigation identified:

- High GPU memory allocation while GPU utilization was zero.
- FlashInfer JIT compilation activity.
- WSL2 dxgkrnl memory-residency errors.
- EngineCore becoming unresponsive during an earlier startup.
- Successful graceful shutdown after SIGINT.

These observations are not yet confirmed as a single root cause.

## 10\. Known-Good Baseline

The current build has successfully:

1. Imported vLLM from the custom source tree.
2. Loaded native extensions.
3. Executed CUDA GEMM.
4. Loaded FlashInfer.
5. Started the vLLM API server.
6. Served successful model completions.
7. Completed a 100-request benchmark.

Preserve this configuration before attempting further build or kernel changes.

## 11\. Next Steps

- Establish a low-concurrency benchmark baseline.
- Investigate TTFT and request queuing.
- Compare attention backend configurations.
- Investigate FlashInfer versus Triton attention.
- Record performance changes against this known-good baseline.

## 4. Where we are now

I would classify the project as:

SM120 source build

Working

Native extensions

Verified

CUDA computation

Verified

Model serving

Working

Performance tuning

In progress

My recommendation: Do not rebuild vLLM yet. The current build works. First, finish documenting and preserving it.

Next, let's review the complete `build.sh` and make sure it has proper logging, failure handling, environment validation and reproducibility before calling it our permanent build script.
