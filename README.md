# LLM Inference Performance Benchmarks

**vLLM vs NVIDIA TensorRT-LLM | RTX 5060 Ti 16GB | Inference Optimization**

A practical benchmarking project evaluating LLM inference performance across two popular serving engines: **vLLM** and **NVIDIA TensorRT-LLM**.

The objective is to understand how inference engine architecture, attention backends, memory management, and runtime optimization affect LLM serving performance on consumer NVIDIA GPU hardware.

## Hardware Environment

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| VRAM | 16 GB |
| OS | Linux |
| Python | 3.12 |
| NVIDIA Driver | 610.88 |
| NVIDIA-SMI | 580.196.07 |
| CUDA Toolkit | 13.4 (V13.4.92) |
| Driver-reported CUDA | 13.3 |
| vLLM | 0.33.0rc1.dev321+g6d0189348.d20261101.cu134 |
| TensorRT-LLM | 1.4.0rc2.dev202610270008 |

Version note: Both frameworks are development/release-candidate builds, not stable releases.

## Benchmark Scope

The project evaluates both inference engines using a common model and controlled workloads.

Metrics include:

- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- Inter-token Latency (ITL)
- Request throughput
- Aggregate output token throughput
- Peak concurrent requests
- GPU utilization and memory consumption

### Model Under Evaluation

**Qwen2.5-1.5B-Instruct**

Both engines use the same model and standardized synthetic workload:

| Parameter | Value |
| --- | --- |
| Input length | 1,024 tokens |
| Output length | 256 tokens |
| Number of prompts | 100 |
| Request rate | 5 RPS |
| Benchmark runs | 4 per engine |

## Repository Structure

text

```text
llm-inference-benchmarks/
├── README.md
├── docs/
│   ├── 1-vLLM-RUNBOOK.md
│   └── 2-TRTLLM-RUNBOOK.md
└── reports/
    ├── 01-vllm-baseline.md
    └── 02-tensorrt-llm.md
```

## Documentation & Runbooks

| Document | Description |
| --- | --- |
| vLLM Runbook | vLLM server configuration, startup and benchmark execution |
| TensorRT-LLM Runbook | TensorRT-LLM configuration, startup and benchmark execution |

## Experimental Reports

| Report | Description | Status |
| --- | --- | --- |
| 01 - vLLM Baseline | Four benchmark runs, metrics and baseline analysis | Completed |
| 02 - TensorRT-LLM Baseline | Four benchmark runs, metrics and comparison with vLLM | Completed |

## Benchmark Results: vLLM vs TensorRT-LLM

The following table compares the steady-state averages from Runs 2–4 for each engine. Run 1 is retained in the individual reports but excluded from this comparison because it showed lower initial performance.

| Metric | vLLM | TensorRT-LLM |
| --- | --- | --- |
| Output throughput (tokens/sec) | 796.39 | 788.74 |
| Request throughput (req/s) | 3.11 | 3.08 |
| Mean TTFT (ms) | 3,889.63 | 3,880.65 |
| Mean TPOT (ms) | 9.28 | 9.513 |
| P99 TTFT (ms) | 10,000.87 | 10,094.34 |

### Summary

Both inference engines delivered remarkably similar steady-state performance on the RTX 5060 Ti 16GB.

vLLM achieved approximately 0.97% higher output throughput and 2.45% lower mean TPOT. TensorRT-LLM recorded approximately 9 ms lower mean TTFT.

The results indicate that, for Qwen2.5-1.5B-Instruct under this specific workload, neither engine demonstrates a substantial performance advantage.

These findings establish an empirical baseline for further inference optimization experiments.

## Benchmark Methodology

Both engines were evaluated using:

- Identical model architecture and model weights.
- Fixed input and output token lengths.
- Consistent request counts and request rates.
- The same benchmark client and measurement methodology.
- Four independent benchmark executions per engine.
- Comparable GPU hardware constraints.

Results are compared using latency, throughput, concurrency and GPU resource measurements.

The benchmark uses synthetic random prompts. Results should therefore be interpreted as controlled systems-performance measurements rather than representative measurements of every production workload.

## Project Objectives

1. Establish reproducible LLM inference benchmarks.
2. Evaluate vLLM and TensorRT-LLM performance.
3. Understand prefill and decode performance characteristics.
4. Investigate GPU memory utilization and scheduling behavior.
5. Identify performance trade-offs under controlled request workloads.

## Scope

This repository focuses exclusively on LLM inference serving and performance benchmarking.

It does not evaluate model accuracy, clinical applications or medical AI workflows.

The goal is reproducible systems engineering experimentation supported by actual measurements rather than theoretical performance claims.

