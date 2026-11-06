# LLM Inference Performance Benchmarks

**vLLM vs NVIDIA TensorRT-LLM vs SGLang | RTX 5060 Ti 16GB | Inference Performance**

A practical benchmarking project evaluating LLM inference performance across three popular inference engines:

- **vLLM**
- **NVIDIA TensorRT-LLM**
- **SGLang**

The objective is to understand how inference engine architecture, attention backends, memory management, scheduling and runtime optimization affect LLM serving performance on consumer NVIDIA GPU hardware.

The project focuses on reproducible measurements rather than theoretical comparisons.

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
| SGLang | Docker image: lmsysorg/sglang:latest |

**Version note:** vLLM and TensorRT-LLM are development/release-candidate builds. The exact SGLang, PyTorch and CUDA runtime versions should be recorded from the container used for benchmarking. The Docker `latest` tag is mutable, so the image digest should also be preserved.

## Benchmark Scope

The project evaluates three inference engines using a common model and controlled synthetic workload.

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

All three engines were evaluated using the same model and benchmark workload.

| Parameter | Value |
| --- | --- |
| Model | Qwen2.5-1.5B-Instruct |
| Input length | 1,024 tokens |
| Output length | 256 tokens |
| Number of prompts | 100 |
| Request rate | 5 RPS |
| Benchmark runs | 4 per engine |
| Benchmark client | vLLM `bench serve` |
| Backend | OpenAI-compatible API |
| Dataset | Random synthetic |

### Experimental Configuration

Each engine uses its own serving configuration and runtime optimizations.

| Engine | Deployment | Notable configuration |
| --- | --- | --- |
| vLLM | Linux environment | vLLM serving runtime |
| TensorRT-LLM | Kubernetes pod | TensorRT-LLM serving runtime |
| SGLang | Docker container | FlashInfer, FP8 KV cache, RadixCache disabled |

The benchmark workload and client are kept consistent. Engine-specific runtime configurations are documented in their respective reports.

## Repository Structure

text

```text
llm-inference-benchmarks/
├── README.md
├── docs/
│   ├── 1-vLLM-RUNBOOK.md
│   ├── 2-TRTLLM-RUNBOOK.md
│   └── 3-SGLANG-RUNBOOK.md
└── reports/
    ├── 01-vllm-baseline.md
    ├── 02-tensorrt-llm.md
    └── 03-sglang-baseline.md
```

## Documentation & Runbooks

| Document | Description |
| --- | --- |
| vLLM Runbook | vLLM server configuration, startup and benchmark execution |
| TensorRT-LLM Runbook | TensorRT-LLM deployment, configuration and benchmark execution |
| SGLang Runbook | SGLang Docker deployment, server configuration and benchmark execution |

## Experimental Reports

| Report | Description | Status |
| --- | --- | --- |
| 01 - vLLM Baseline | Four benchmark runs, metrics and baseline analysis | Completed |
| 02 - TensorRT-LLM Baseline | Four benchmark runs, metrics and comparison with vLLM | Completed |
| 03 - SGLang Baseline | Four benchmark runs, metrics and performance analysis | Completed |

## Benchmark Results: Three-Engine Comparison

The following table compares the steady-state averages from Runs 2–4 for all three inference engines.

Run 1 is retained in the individual reports but excluded from the comparison because it represents the initial warm-up execution.

### Steady-State Performance

| Metric | vLLM | TensorRT-LLM | SGLang |
| --- | --- | --- | --- |
| Output throughput (tok/s) | 796.39 | 788.74 | **959.03** |
| Request throughput (req/s) | 3.11 | 3.08 | **3.75** |
| Mean TTFT (ms) | 3889.63 | 3880.65 | **144.24** |
| Median TTFT (ms) | 3358.54 | 3500.30 | **123.57** |
| P99 TTFT (ms) | 10000.01 | 10094.34 | **286.79** |
| Mean TPOT (ms) | **9.28** | 9.51 | 32.24 |
| Median TPOT (ms) | **9.30** | 9.55 | 33.54 |
| P99 TPOT (ms) | **9.45** | 9.62 | 38.79 |
| Mean ITL (ms) | **9.28** | 9.51 | 32.24 |
| Median ITL (ms) | **9.25** | 9.51 | 20.65 |
| P99 ITL (ms) | **16.10** | 10.33 | 173.89 |

_Throughput is measured in generated output tokens per second. Latency values are in milliseconds. Bold indicates the best measured value for each metric._

### Performance Summary

The three inference engines demonstrated different performance characteristics under the tested workload.

**SGLang**

- Highest steady-state output throughput at 959.03 tok/s.
- Approximately 20.7% higher output throughput than vLLM.
- Substantially lower mean TTFT at 144.24 ms.
- Higher decode latency, with mean TPOT of 32.24 ms.
- High GPU memory consumption, reaching 16,008 MiB during monitoring.

**vLLM**

- Output throughput of 796.39 tok/s.
- Lowest mean TPOT at 9.28 ms.
- Lowest median and P99 TPOT among the three engines.
- Relatively consistent steady-state performance across Runs 2–4.

**TensorRT-LLM**

- Output throughput of 788.74 tok/s.
- Mean TTFT of 3880.65 ms.
- Mean TPOT of 9.51 ms.
- Performance very similar to vLLM under the tested configuration.

### Key Findings

1. SGLang achieved the highest aggregate output throughput and lowest measured TTFT.
2. vLLM achieved the lowest decode latency.
3. TensorRT-LLM delivered performance close to vLLM, without a substantial throughput advantage.
4. SGLang showed a different balance between prefill/first-token latency and decode latency.
5. GPU memory consumption is an important constraint, particularly for SGLang on the RTX 5060 Ti 16GB.

**Overall conclusion:** No single engine is best across every performance metric. SGLang leads in aggregate throughput and TTFT, while vLLM leads in decode latency. TensorRT-LLM performs similarly to vLLM in this particular experiment.

These results represent measured performance for the tested configurations and should not be interpreted as a universal ranking of inference engines.

## Benchmark Methodology

All three engines were evaluated using:

- Identical model architecture and model weights.
- Fixed input and output token lengths.
- Consistent request counts and configured request rates.
- The same benchmark client (`vllm bench serve`).
- OpenAI-compatible serving endpoints.
- Four benchmark executions per engine.
- Runs 2–4 for steady-state performance comparisons.
- Run 1 retained as the initial warm-up execution.
- GPU utilization and memory observations where available.

The benchmark uses synthetic random prompts. Results should therefore be interpreted as controlled systems-performance measurements rather than representative measurements of every production workload.

Engine-specific configurations are retained and documented rather than forcing identical internal settings across different serving frameworks.

## Project Objectives

1. Establish reproducible LLM inference benchmarks.
2. Evaluate vLLM, TensorRT-LLM and SGLang performance.
3. Understand prefill and decode performance characteristics.
4. Investigate GPU memory utilization and scheduling behavior.
5. Compare attention backends and KV-cache management.
6. Identify performance trade-offs under controlled request workloads.
7. Evaluate inference optimization opportunities on consumer NVIDIA GPU hardware.

## Scope and Limitations

This repository focuses exclusively on LLM inference serving and performance benchmarking.

Current experimental scope:

- Single GPU: NVIDIA RTX 5060 Ti 16GB.
- Single model: Qwen2.5-1.5B-Instruct.
- Synthetic random workload.
- Fixed input and output token lengths.
- Configured request rate of 5 RPS.
- Four executions per engine.
- Sequential benchmark executions.
- No randomized or paired execution order.
- Engine-specific serving configurations.
- GPU resource observations captured during selected runs.

The benchmark does not evaluate model accuracy, training performance or production workload generalization.

Results are specific to the tested hardware, software versions, model configuration and workload. Additional experiments are required to evaluate other models, GPUs, context lengths, concurrency levels and serving configurations.

The goal is reproducible systems engineering experimentation supported by actual measurements rather than theoretical performance claims.
