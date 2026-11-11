# LLM Inference Performance Benchmarks

**vLLM (`vllm-rs`) vs NVIDIA TensorRT-LLM vs SGLang | RTX 5060 Ti 16GB (SM120) | Inference Performance**

A practical benchmarking and systems-engineering project evaluating LLM inference performance across three serving engines on Blackwell consumer hardware:

- **vLLM** (Standard Python frontend & compiled `vllm-rs` Rust frontend)
- **NVIDIA TensorRT-LLM**
- **SGLang**

The objective is to understand how inference engine architecture, control-plane concurrency (Python GIL vs. Rust Axum), attention backends, memory management, and runtime optimization affect LLM serving performance on NVIDIA SM120 GPU hardware.

The project focuses on reproducible measurements, CPU flamegraph profiling, and empirical comparisons rather than theoretical performance claims.

---

## Hardware & Software Environment

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti (Blackwell, SM120) |
| VRAM | 16 GB (16,311 MiB reported) |
| OS | Linux |
| Python | 3.12 |
| NVIDIA Driver | 610.88 |
| NVIDIA-SMI | 580.196.07 |
| Host CUDA Toolkit | 13.4 (V13.4.92) |
| Driver-reported CUDA | 13.3 |
| vLLM | `0.33.0rc1.dev321+g6d0189348.d20261101.cu134` (LibTorch Stable ABI + `vllm-rs`) |
| TensorRT-LLM | `1.4.0rc2.dev202610270008` |
| SGLang | `0.5.23` (PyTorch `2.14.1+cu130`) |

**Version note:** vLLM and TensorRT-LLM are custom SM120 development/release-candidate builds. SGLang runs inside a Docker container (`lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`).

---

## Repository Structure

```text
llm-inference-benchmarks/
├── docs/
│   ├── 01-vLLM-RUNBOOK.md
│   ├── 02-TRTLLM-RUNBOOK.md
│   ├── 03-SGLANG-RUNBOOK.md
│   ├── 04-VLLM-CONTRIBUTION-ROADMAP.md
│   ├── performance/
│   │   ├── SM120_SCHEDULER_INVESTIGATION_001.md
│   │   ├── SM120_VLLM_CPU_GPU_ROOT_CAUSE_001.md
│   │   └── SM120_VLLM_RUST_FRONTEND_INVESTIGATION_002.md
│   └── sglang-help.txt
├── flame-graph/
│   ├── vllm_sm120_profile.svg
│   └── vllm_starvation_profile.svg
├── README.md
├── reports/
│   ├── 01-vllm-baseline.md
│   ├── 02-tensorrt-llm.md
│   ├── 03-A-sglang-baseline.md
│   ├── 03-B-sglang-baseline.md
│   ├── 03-C-sglang-dummy-saturation.md
│   └── 03-D-sglang-real-weights-comparison.md
├── rust-compile.txt
├── scripts/
│   └── vllm-build-config/
│       ├── backups/
│       │   ├── sm120-requirements.patch
│       │   └── working-commit.txt
│       ├── base-ver-build.sh
│       ├── BUILD_HISTORY.md
│       ├── build.sh
│       ├── env.sh
│       ├── logs/
│       ├── patches/
│       └── verification.sh
└── vllm-help.txt
```

---

## Documentation, Investigations & Reports

### Runbooks & Roadmaps (`docs/`)

| Document | Description |
| --- | --- |
| `01-vLLM-RUNBOOK.md` | vLLM server configuration, startup, and benchmark execution |
| `02-TRTLLM-RUNBOOK.md` | TensorRT-LLM deployment, configuration, and benchmark execution |
| `03-SGLANG-RUNBOOK.md` | SGLang Docker deployment, server configuration, and benchmark execution |

### Deep-Dive Performance Investigations (`docs/performance/` & `frame-graph/`)

| Document / Artifact | Focus & Key Findings |
| --- | --- |
| `SM120_SCHEDULER_INVESTIGATION_001.md` | Concurrency sweep (`max-num-seqs` 8 $\rightarrow$ 64) on Python vLLM; identified `48` sequences (`1,207.92 tok/s`) as the legacy Python ceiling. |
| `SM120_VLLM_CPU_GPU_ROOT_CAUSE_001.md` | Isolated control-plane starvation testing (`--load-format dummy`, `inf` RPS) & `py-spy` flamegraphs (`frame-graph/*.svg`). Proved Python's `hf.py` tokenizer wrapper consumed ~60% of CPU cycles under burst loads. |
| `SM120_VLLM_RUST_FRONTEND_INVESTIGATION_002.md` | Compiled and benchmarked the native Rust frontend (`VLLM_USE_RUST_FRONTEND=1`) + CUDA Graphs, unlocking **3,539.66 tok/s** (dummy) and **2,188.46 tok/s** (real weights). |

### Experimental Benchmark Reports (`reports/`)

| Report | Description | Status |
| --- | --- | --- |
| `01-vllm-baseline.md` | Initial vLLM baseline runs (5 RPS steady-state) | Completed |
| `02-tensorrt-llm.md` | TensorRT-LLM baseline runs (5 RPS steady-state) | Completed |
| `03-A-sglang-baseline.md` | Original SGLang configuration (FP8 KV cache) | Historical |
| `03-B-sglang-baseline.md` | Aligned SGLang baseline (`max-running-requests 8`, 5 RPS) | Completed |
| `03-C-sglang-dummy-saturation.md` | SGLang control-plane starvation test (`--load-format dummy`, `inf` RPS) | Completed |
| `03-D-sglang-real-weights-comparison.md` | SGLang real-weight concurrency sweep (`64` & `128` requests, `inf` RPS) vs. `vllm-rs` | Completed |

---

## Benchmark Results: Phase 1 (5 RPS Steady-State Baseline)

Initial three-engine evaluation using controlled arrival rates (`100 prompts`, `1024 in / 256 out`, `5 RPS`, steady-state averages from Runs 2–4):

| Metric | vLLM (Baseline) | TensorRT-LLM | SGLang (`EXP-003-B`) |
| --- | :---: | :---: | :---: |
| Output throughput (tok/s) | **796.39** | 788.74 | 674.63 |
| Request throughput (req/s) | **3.11** | 3.08 | 2.63 |
| Mean TTFT (ms) | 3,889.63 | **3,880.65** | 6,487.35 |
| Median TTFT (ms) | **3,358.54** | 3,500.30 | 6,011.20 |
| P99 TTFT (ms) | **10,000.01** | 10,094.34 | 15,654.38 |
| Mean TPOT (ms) | **9.28** | 9.51 | 10.96 |
| Median TPOT (ms) | **9.30** | 9.55 | 11.06 |
| P99 TPOT (ms) | **9.45** | 9.62 | 11.08 |
| P99 ITL (ms) | 16.10 | **10.33** | 69.51 |

---

## Benchmark Results: Phase 2 (High-Concurrency Saturation & `vllm-rs` Optimization)

Following root-cause profiling, vLLM's native Rust frontend (`VLLM_USE_RUST_FRONTEND=1`) was compiled to eliminate Python GIL and `hf.py` tokenizer deep-copy bottlenecks, and evaluated under infinite burst traffic (`--request-rate inf`, `100 prompts`, `1024 in / 256 out`).

### 1. Control-Plane Starvation (`--load-format dummy` @ `INF RPS`)
Isolates CPU HTTP/tokenization ingestion and scheduling overhead by bypassing GPU matrix math:

| Metric | Legacy Python vLLM (`Seq 64`) | SGLang (`Max Reqs 128`) | `vllm-rs` (`Seq 128`) | `vllm-rs` (`Seq 64` Winner) |
| :--- | :---: | :---: | :---: | :---: |
| **Output Throughput** | 1,890.38 tok/s | 1,936.33 tok/s | 2,178.78 tok/s | **3,539.66 tok/s** (+87.2%) |
| **Total Throughput** | 9,451.88 tok/s | 9,681.65 tok/s | 10,893.89 tok/s | **17,698.32 tok/s** |
| **Mean TTFT** | 4,919.55 ms | 3,359.27 ms | 3,772.76 ms | **1,559.02 ms** |
| **Mean TPOT** | 23.86 ms | 38.57 ms | 30.63 ms | **14.23 ms** |

### 2. Real-Weight Saturation (`Qwen2.5-1.5B-Instruct` @ `INF RPS`)
Full hardware execution comparing custom SM120 `vllm-rs` (`TRITON_ATTN` + CUDA Graphs) against SGLang (`flashinfer`):

| Metric | SGLang (`Seq 64`) | `vllm-rs` (`Seq 64`) | SGLang (`Seq 128`) | `vllm-rs` (`Seq 128` Winner) |
| :--- | :---: | :---: | :---: | :---: |
| **Benchmark Duration** | 17.71 s | **13.35 s** | 12.85 s | **11.70 s** |
| **Output Throughput** | 1,445.55 tok/s | **1,917.60 tok/s** | 1,991.76 tok/s | **2,188.46 tok/s** |
| **Total Throughput** | 7,227.73 tok/s | **9,587.99 tok/s** | 9,958.78 tok/s | **10,942.30 tok/s** |
| **Request Throughput** | 5.65 req/s | **7.49 req/s** | 7.78 req/s | **8.55 req/s** |
| **Mean TTFT** | 5,511.04 ms | **4,730.22 ms** | **3,369.98 ms** | 3,492.59 ms |
| **Mean TPOT (Decode)** | 28.80 ms | **23.84 ms** | 37.09 ms | **30.86 ms** |
| **P99 TPOT** | 38.37 ms | **29.44 ms** | 48.96 ms | **42.03 ms** |

---

## Key Architectural Conclusions

1. **Python Ingestion Bottleneck Eliminated:** Under instantaneous traffic bursts (`inf` RPS), Python-based HTTP and tokenizer wrappers (`hf.py` in legacy vLLM and Uvicorn in SGLang) cap control-plane throughput around ~1.9k tok/s. Compiling and enabling `VLLM_USE_RUST_FRONTEND=1` removes GIL contention completely, pushing dummy control-plane throughput to **3,539.66 tok/s** and real-weight SM120 throughput to **2,188.46 tok/s**.
2. **The Remaining Scheduler Ceiling:** While `vllm-rs` solves northbound ingestion, scaling active sequences from `64` to `128` increases Mean TPOT (`23.84 ms` $\rightarrow$ `30.86 ms`) because the core batch scheduler (`scheduler.py`) still iterates over Linear Page Maps inside a single-threaded Python process.
3. **Production Deployment Sizing:**
   - **Latency-Sensitive Workloads (Real-Time Chat):** Use `VLLM_USE_RUST_FRONTEND=1` with `--max-num-seqs 48` or `64` to keep Mean TPOT deterministic (`21–23 ms`).
   - **Throughput-Oriented Workloads (Offline Batch / Agents):** Use `VLLM_USE_RUST_FRONTEND=1` with `--max-num-seqs 128` to maximize aggregate token generation (`2,188.46 tok/s`).
