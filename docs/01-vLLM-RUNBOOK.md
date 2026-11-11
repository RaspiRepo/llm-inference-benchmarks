# vLLM Benchmark Runbook (SM120 / Blackwell)

**Experiments:** EXP-001, Investigation 01 & Investigation 02  
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Hardware:** NVIDIA GeForce RTX 5060 Ti 16GB (SM120)  
**Serving Runtime:** Custom SM120 vLLM (`v0.33.0rc1.dev321` + `vllm-rs` Rust Frontend)  
**Benchmark Client:** vLLM `bench serve`  

---

## 1. Objective

Establish reproducible vLLM performance baselines, high-concurrency saturation metrics, and control-plane profiles on NVIDIA Blackwell (SM120) hardware.

The runbook covers:
- **Phase 1 (Legacy Python Baseline):** Steady-state `5 RPS` workload (`--max-num-seqs 8` and `48`).
- **Phase 2 (`vllm-rs` High-Concurrency Saturation):** Native Rust HTTP/Tokenizer frontend (`VLLM_USE_RUST_FRONTEND=1`) + CUDA Graphs under infinite burst (`--request-rate inf`).
- **Control-Plane Isolation:** Dummy weight profiling (`--load-format dummy`) to isolate CPU orchestration overhead.

---

## 2. Environment & Rust Frontend Compilation

Activate the project virtual environment and verify hardware/build versions:

```bash
source .venv/bin/activate
nvidia-smi
vllm --version
```

### Build the Native Rust Frontend (`vllm-rs`)
Before using `VLLM_USE_RUST_FRONTEND=1`, ensure the standalone Rust workspace binary (`vllm/vllm-rs`) is compiled in release mode from the vLLM source directory:

```bash
python tools/build_rust.py --release
```

*Verification:* When `VLLM_USE_RUST_FRONTEND=1` is active, startup logs will display `(RustFrontend pid=XXXX) INFO ... [server/src/lib.rs:345] OpenAI server is ready` instead of Python FastAPI route listings.

---

## 3. Configuration Parameters

| Parameter | Value | Purpose |
| --- | --- | --- |
| `VLLM_USE_RUST_FRONTEND` | `1` (Recommended) | Replaces Python FastAPI & `hf.py` tokenizer with compiled Rust (`Axum` + `vllm-text`) |
| `--max-model-len` | `2048` | Sets maximum context window length |
| `--max-num-seqs` | `8` (Legacy) / `48`–`64` (Low Latency) / `128` (Peak Throughput) | Controls active concurrent sequences in the scheduler |
| `--max-num-batched-tokens` | `2048` (`4096` for `Seq 128`) | Sets per-iteration token budget |
| `--gpu-memory-utilization` | `0.90` | Allocates 90% of 16GB VRAM (~15.7 GiB peak usage; leaves ~600 MiB headroom) |
| `--attention-backend` | `TRITON_ATTN` | Selects Triton attention kernels for SM120 compatibility |
| `--no-enforce-eager` | Enabled (Default) | Enforces CUDA Graph capture to eliminate PyTorch/LibTorch launch overhead |
| `--load-format` | `auto` (Default) / `dummy` (Starvation) | Optional flag to bypass GPU math and isolate CPU scheduler limits |

---

## 4. Start vLLM Server

Keep the server running in **Terminal 1**.

### 4.1 Phase 1: Legacy Python Baseline (`EXP-001`)

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --port 8000
```

### 4.2 Phase 2: Production `vllm-rs` + CUDA Graphs (Recommended)
Use `--max-num-seqs 48` or `64` for latency-sensitive chat SLOs (`TPOT ~21–23 ms`), or `128` (`--max-num-batched-tokens 4096`) for peak offline throughput (`2,188.46 tok/s`):

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 2048 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --no-enforce-eager \
    --port 8000
```

### 4.3 Control-Plane Starvation Mode (CPU Isolation)
Add `--load-format dummy` to skip real matrix math and measure pure CPU scheduler/IPC limits:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 2048 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --load-format dummy \
    --no-enforce-eager \
    --port 8000
```

---

## 5. Verify API Availability & Warm-Up

In **Terminal 2**, verify health and run a single warm-up request (excluded from measured runs) so CUDA Graph capture and kernel compilation settle:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "messages": [
        {"role": "user", "content": "Hello"}
      ],
      "max_tokens": 50
    }'
```

---

## 6. Run Serving Benchmark

### 6.1 Steady-State Workload (`5 RPS`)

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
    --label results_vllm_qwen
```

### 6.2 High-Concurrency Burst Workload (`inf RPS`)

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
    --label sm120_vllm_rs_inf
```

---

## 7. GPU & CPU Flamegraph Monitoring

In **Terminal 3**, log GPU utilization, VRAM headroom, clocks, and power draw:

```bash
nvidia-smi \
  --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu \
  --format=csv \
  -l 1
```

To capture a mixed Python/C++ CPU flamegraph of the `EngineCore` process during active benchmarking:

```bash
sudo env "PATH=\$PATH" py-spy record -o frame-graph/vllm_sm120_profile.svg --pid <ENGINE_CORE_PID> --native --rate 100
```
