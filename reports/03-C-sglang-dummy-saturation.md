# SGLang Benchmark Report EXP-003-C (Control Plane Saturation)

**Date:** November 10, 2026  
**Model:** Qwen/Qwen2.5-1.5B-Instruct  
**Hardware:** NVIDIA GeForce RTX 5060 Ti (16GB, SM120)  
**Serving Runtime:** SGLang (`lmsysorg/sglang:latest` via Docker)  
**Workload:** Control Plane Starvation (`--load-format dummy`, `--request-rate inf`)

---

## 1. Executive Summary

This experiment (EXP-003-C) isolates SGLang's CPU control plane and scheduling overhead by bypassing GPU matrix computations (`--load-format dummy`) and flooding the endpoint with an infinite arrival burst (`--request-rate inf`). 

While SGLang slightly outperformed the legacy Python vLLM baseline (`1,936.33 tok/s` vs. `1,890.38 tok/s`), it trailed the native Rust-fronted `vllm-rs` build (`3,539.66 tok/s`) by **45.3%**, confirming that Python-based HTTP/tokenization ingestion remains a primary bottleneck under instantaneous traffic spikes.

---

## 2. Configuration Parameters

| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| `--context-length` | `2048` | Matches vLLM `max-model-len` baseline |
| `--max-running-requests` | `128` | Uncaps concurrency to test scheduler limits |
| `--mem-fraction-static` | `0.90` | Matches vLLM `gpu-memory-utilization` |
| `--attention-backend` | `flashinfer` | Standard SGLang attention backend |
| `--disable-radix-cache` | Enabled | Prevents artificial cache hits on random data |
| `--load-format` | `dummy` | Bypasses GPU math to isolate CPU overhead |

---

## 3. Benchmark Results (EXP-003-C)

| Metric | Result |
| :--- | :--- |
| Successful requests | 100 |
| Failed requests | 0 |
| Benchmark duration | 13.22 s |
| Request throughput | 7.56 req/s |
| Output token throughput | **1,936.33 tok/s** |
| Peak output throughput | 4,100.00 tok/s |
| Total token throughput | 9,681.65 tok/s |
| Mean TTFT | 3,359.27 ms |
| Median TTFT | 3,354.13 ms |
| P99 TTFT | 6,395.26 ms |
| Mean TPOT | **38.57 ms** |
| Median TPOT | 38.58 ms |
| P99 TPOT | 50.42 ms |

---

## 4. Cross-Framework Conclusion

1. **Northbound Ingestion Bottleneck:** SGLang's Python HTTP layer hits a ~1.9k tok/s ceiling under instantaneous `inf` RPS bursts, whereas `vllm-rs` scales to **3,539.66 tok/s** via lock-free Rust ingestion (`Axum` + `vllm-text`).
2. **Next Step:** Execute **EXP-003-D** with real model weights (`Seq 64` and `Seq 128`) to evaluate whether SGLang's `flashinfer` + C++ kernel fusion outperforms `vllm-rs` (`2,188.46 tok/s`) when GPU matrix compute becomes the primary bottleneck.
