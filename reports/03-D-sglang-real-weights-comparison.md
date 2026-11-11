# SGLang Benchmark Report EXP-003-D (Real-Weight Concurrency Sweep)

**Date:** November 10, 2026  
**Model:** Qwen/Qwen2.5-1.5B-Instruct  
**Hardware:** NVIDIA GeForce RTX 5060 Ti (16GB, SM120)  
**Serving Runtime:** SGLang (`lmsysorg/sglang:latest` via Docker)  
**Workload:** Real Weights, Infinite Arrival Burst (`--request-rate inf`, 100 prompts, 1024 in / 256 out)

---

## 1. Executive Summary

This experiment (EXP-003-D) evaluates SGLang under identical real-weight, infinite-burst saturation conditions as `vllm-rs` Investigation 02. Testing was conducted at maximum running sequence limits of **64** and **128**.

While scaling SGLang from 64 to 128 sequences improved aggregate output throughput by **37.8%** (`1,445.55 tok/s` $\rightarrow$ `1,991.76 tok/s`), SGLang trailed the custom SM120 `vllm-rs` build across both concurrency tiers (`1,917.60 tok/s` at `Seq 64` and `2,188.46 tok/s` at `Seq 128`).

---

## 2. Benchmark Results (EXP-003-D)

| Metric | SGLang (`Max Reqs 64`) | SGLang (`Max Reqs 128`) | Change (64 $\rightarrow$ 128) |
| :--- | :---: | :---: | :---: |
| Successful requests | 100 | 100 | No change |
| Failed requests | 0 | 0 | No change |
| Benchmark duration | 17.71 s | **12.85 s** | -27.4% |
| Request throughput | 5.65 req/s | **7.78 req/s** | +37.7% |
| Output token throughput | 1,445.55 tok/s | **1,991.76 tok/s** | **+37.8%** |
| Peak output throughput | 2,879.00 tok/s | **4,100.00 tok/s** | +42.4% |
| Total token throughput | 7,227.73 tok/s | **9,958.78 tok/s** | +37.8% |
| Mean TTFT | 5,511.04 ms | **3,369.98 ms** | -38.9% |
| Median TTFT | 3,368.92 ms | **3,367.03 ms** | -0.1% |
| P99 TTFT | 12,412.90 ms | **6,411.38 ms** | -48.3% |
| Mean TPOT | **28.80 ms** | 37.09 ms | +28.8% |
| Median TPOT | **28.02 ms** | 37.10 ms | +32.4% |
| P99 TPOT | **38.37 ms** | 48.96 ms | +27.6% |

---

## 3. Cross-Framework Verdict (`vllm-rs` vs. SGLang on SM120)

| Metric | `vllm-rs` (`Seq 64`) | SGLang (`Seq 64`) | `vllm-rs` (`Seq 128`) | SGLang (`Seq 128`) |
| :--- | :---: | :---: | :---: | :---: |
| **Output Throughput** | **1,917.60 tok/s** | 1,445.55 tok/s | **2,188.46 tok/s** | 1,991.76 tok/s |
| **Mean TTFT** | **4,730.22 ms** | 5,511.04 ms | 3,492.59 ms | **3,369.98 ms** |
| **Mean TPOT** | **23.84 ms** | 28.80 ms | **30.86 ms** | 37.09 ms |

### Key Takeaways
1. **Ingestion Bottleneck in SGLang:** SGLang's real-weight `Seq 128` throughput (`1,991.76 tok/s`) plateaued right at its dummy starvation ceiling (`1,936.33 tok/s`), indicating that its Python HTTP/tokenization layer capped concurrency under instantaneous bursts.
2. **SM120 Kernel & Rust Advantage:** Compiling `vllm-rs` natively for SM120 with the Rust frontend unlocked **32.7% higher throughput at `Seq 64`** and **9.9% higher throughput at `Seq 128`**, making `vllm-rs` the clear production winner on this hardware.
