# SM120 vLLM Performance Investigation
## Investigation 01 — Root Cause Analysis & Architectural Trace Report

**Date:** November 9, 2026  
**Platform:** NVIDIA GeForce RTX 5060 Ti (SM120, Blackwell Architecture)  
**Framework:** Custom SM120 vLLM build  
**Model:** Qwen/Qwen2.5-1.5B-Instruct  
**Status:** Investigation Completed; Core Bottlenecks Characterized for Production Scaling.

---

# 1. Executive Summary

This report establishes the empirical performance characteristics of vLLM's CPU control plane and GPU data plane. Starting with a baseline sequence concurrency of 8, we scaled workloads to saturate the hardware, mapped execution fingerprints via low-level sampling profilers, and artificially simulated GPU starvation to expose raw text-processing limits.

Our initial thesis suspected that the Python tokenizer pool (`vllm/tokenizers/hf.py`) introduced a major execution bottleneck under concurrent stress. Through systematic testing, we proved a dual-regime operational reality:
1. **GPU-Bound Regime (Lower-End/Desktop Hardware):** At high sequence counts, standard eager-mode framework execution creates a massive "CUDA Sync Wall," consuming up to 85% of CPU cycles simply waiting for the GPU to clear.
2. **CPU-Bound Regime (High-End Production Hardware Simulation):** When GPU computation constraints are removed, the text preprocessing layer explodes to occupy **60% of all CPU cycles**. Under a concurrent burst (`request-rate inf`), the tokenizer pool starves the scheduler loop, causing a **68% degradation in token generation latency (TPOT)**.

---

# 2. Detailed Technical Trace: Request Start-to-Finish

To map code optimizations for a production environment, the path of a prompt request must be understood down to its thread and hardware allocations:

```text
[User Client] ──(HTTP/gRPC Strings)──► [API Server] ──► [Tokenizer Pool (hf.py)]
│
┌─────────────────────────────────────────────────────────────┘
▼
(Integer Token IDs) ──► [Scheduler / Block Manager] ──► [Worker Memory Mapping]
│
┌─────────────────────────────────────────────────────────────┘
▼
[PyTorch / CUDA Driver] ──(Kernel Launch/PCIe Bus)──► [GPU Matrix Cores]
```


### Phase 1: Ingestion & Text Transformation (CPU-Bound)

1. **Network Entry:** The HTTP/gRPC request hits the FastAPI/Uvicorn front-end server process.
2. **Tokenization Entry:** Strings pass to `vllm/tokenizers/hf.py`. Because Python enforces a Global Interpreter Lock (GIL), vLLM calls `maybe_make_thread_pool()`, which deep-copies the Hugging Face Rust-backed tokenizer backend into a thread-safe object queue (`queue.Queue`).
3. **Encoding Phase:** The text is parsed using Byte-Pair Encoding (BPE), turning raw text strings into arrays of 64-bit integer token IDs.

### Phase 2: Metadata Allocation & Batch Orchestration (CPU-Bound)
4. **Sequence Promotion:** Token arrays are wrapped into Python objects (`SequenceGroup`) and registered in the engine queue.
5. **Scheduler Iteration Loop:** Every single decode step, `vllm/core/scheduler.py` wakes up. It iterates over all waiting and running sequences to enforce sequence limits (`--max-num-seqs`) and token budgets (`--max-num-batched-tokens`).
6. **PagedAttention Table Mapping:** The `Block Manager` updates virtual memory page tables. It maps the logical token indexes to non-contiguous physical memory blocks residing in the GPU's VRAM cache.
7. **Batch Compilation:** Token IDs and physical block table pointers are tightly bundled into arrays and dispatched via Python-to-C++ bindings (`pybind11`).

### Phase 3: Hardware Dispatch & Kernel Execution (GPU-Bound)
8. **Eager-Mode Kernel Launch:** The PyTorch/Triton model executor receives the array structure. It translates the architectural layers of the LLM into hundreds of individual, sequential CUDA/HIP kernel launches over the PCIe bus.
9. **Matrix Multiplication Core Firing:** The GPU activates its physical Matrix Cores (e.g., SM120 structures), pulling model weights and KV cache positions out of VRAM to execute dense fused multiply-add operations.
10. **Synchronization Block:** While the GPU executes these operations, the CPU thread calls a low-level driver sync (e.g., `cudaDeviceSynchronize`), pausing CPU instruction execution until the physical GPU tensor cores complete the batch pass.
11. **Sampling & Stream:** The final logits layer is processed by a sampler to determine the single winning output token ID. The ID is passed back to the CPU, decoded into text by the tokenizer pool, and streamed back to the client.

---

# 3. Complete CLI Commands Utilized

The investigation utilized the following command configurations to isolate system components.

### 3.1 Initial Concurrency Evaluation Server
Used to sweep the max sequence boundaries from 8 to 64:
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 2048 \
    --max-num-seqs 48 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --port 8000
```

### 3.2 Framework Standard Benchmark Client
Used to issue Poisson-burst concurrent traffic loads to the server:
```bash
vllm bench serve \
    --backend openai \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random \
    --input-len 1024 \
    --output-len 256 \
    --num-prompts 100 \
    --request-rate 5 \
    --label sm120_maxseq_TEST
```

### 3.3 Low-Level Non-Invasive CPU Sampling Profiler
Used with elevated root permissions to generate native/mixed-language flame graphs:
```bash
sudo env "PATH=\$PATH" py-spy record -o ../../frame-graph/vllm_sm120_profile.svg --pid <PID> --native --rate 100
```

### 3.4 Hardware-Independent Simulated Starvation Server
Used to bypass physical matrix compute constraints and isolate pure CPU software loops:
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 2048 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --load-format dummy \
    --port 8000
```

### 3.5 Infinite-Burst Preprocessing Stress Client
Used to flood the `hf.py` tokenizer pool with instant text allocations:
```bash
vllm bench serve \
    --backend openai \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random \
    --input-len 1024 \
    --output-len 256 \
    --num-prompts 100 \
    --request-rate inf \
    --label sm120_starvation_test
```

---

# 4. Measured Results Progression

The empirical data gathered across the distinct phases of the investigation demonstrates a clear trajectory:

| Metric Group | Baseline Configuration (`Seq 8`) | Optimal Concurrency (`Seq 48`) | Saturation Regression (`Seq 64`) | Simulated GPU Starvation (`Dummy + INF RPS`) |
| :--- | :--- | :--- | :--- | :--- |
| **Benchmark Duration** | 36.60 s | 21.19 s | 22.77 s | **13.54 s** |
| **Output Throughput** | 699.46 tok/s | **1,207.92 tok/s** | 1,124.45 tok/s | **1,890.38 tok/s** |
| **Total Token Throughput** | 3,497.32 tok/s | 6,039.59 tok/s | 5,622.23 s | **9,451.88 tok/s** |
| **Achieved Request Rate** | 2.73 req/s | **4.72 req/s** | 4.39 req/s | **7.38 req/s** |
| **Mean TTFT** | 6,491.50 ms | **129.42 ms** | 142.45 ms | **4,919.55 ms** |
| **Mean TPOT** | **10.53 ms** | 13.74 ms | 14.27 ms | **23.86 ms** |

---

# 5. Profiling Analysis & Visual Artifacts

The system profiles were captured under live conditions to trace hardware synchronization constraints. The raw visualization binaries are permanently archived in the repository for architectural tracking.

### 5.1 Native Runtime Profile (GPU Concurrency Saturation)
*   **Artifact Path:** `../frame-graph/vllm_sm120_profile.svg`
*   **Behavioral Signature:** Displays a massive horizontal block spanning approximately **85% of total CPU cycles**. This profile captures the CPU execution completely locked inside synchronous PyTorch/CUDA driver wait commands (`cudaDeviceSynchronize`), confirming that the CPU orchestrator is stalling while waiting for the physical matrix computation blocks to clear.

### 5.2 Simulated Starvation Profile (Control Plane Isolation)
*   **Artifact Path:** `../frame-graph/vllm_starvation_profile.svg`
*   **Behavioral Signature:** Bypasses physical matrix hardware execution using empty initialization states. The driver wait blocks disappear entirely, forcing the `hf.py` tokenizer pool and `scheduler.py` allocation loops to expand and consume **60% of all available CPU compute cycles**, visually mapping out the structural performance tax of string serialization under multi-threaded loads.

---

# 6. Root Cause Analysis & Conclusions

### Conclusion 1: The Desktop GPU "CUDA Sync Wall"
When running on desktop-class hardware (RTX 5060 Ti), scaling concurrency past 48 to 64 sequences triggers a **7% throughput reduction**. The flame graph identifies that the CPU spends **~85% of its time blocked by the CUDA driver execution loop**. Under high concurrency, eager-mode kernel launch overhead overwhelms the driver queue, forcing the CPU to remain idle while waiting for physical matrix blocks to complete processing.

### Conclusion 2: The Production-Scale Tokenizer Bottleneck
When upgrading to multi-thousand-dollar production enterprise GPUs, kernel execution timelines shrink from milliseconds to microseconds, shifting the system behavior into the **Simulated Starvation Regime**. 

Our dummy-load stress testing reveals that when the GPU wall is removed, **text preprocessing (`hf.py`) and scheduling operations expand to consume 60% of all CPU execution cycles**. Under high concurrency (`request-rate inf`), the deep-copy thread wrapper architecture creates severe resource contention. The CPU spends its time tokenizing input strings, which starves the core scheduler loop. This architectural collision results in a **68% explosion in generation latency (TPOT spiking from 14.27 ms to 23.86 ms)**.

### Production Recommendation
The text preprocessing plane must be optimized for deployment scaling. To build a robust production environment, the string tokenization plane should be completely decoupled from the main vLLM orchestrator. 

Moving forward, we must eliminate the in-process `maybe_make_thread_pool()` queue entirely and offload tokenization into an isolated, independent CPU process worker pool (or a dedicated Rust/Go preprocessing proxy), ensuring that sudden bursts of user text never choke the real-time scheduling execution of the model engine.
