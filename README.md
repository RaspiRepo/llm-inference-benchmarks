# LLM Inference Performance Benchmarks

**vLLM vs NVIDIA TensorRT-LLM vs SGLang | RTX 5060 Ti 16GB | Inference Performance**

A practical benchmarking project evaluating LLM inference performance across three popular inference engines:

- **vLLM**
- **NVIDIA TensorRT-LLM**
- **SGLang**

The objective is to understand how inference engine architecture, attention backends, memory management, scheduling and runtime optimization affect LLM serving performance on consumer NVIDIA GPU hardware.

The project focuses on reproducible measurements and empirical comparisons rather than theoretical performance claims.

## Hardware Environment

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| VRAM | 16 GB |
| OS | Linux |
| Python | 3.12 |
| NVIDIA Driver | 610.88 |
| NVIDIA-SMI | 580.196.07 |
| Host CUDA Toolkit | 13.4 (V13.4.92) |
| Driver-reported CUDA | 13.3 |
| vLLM | 0.33.0rc1.dev321+g6d0189348.d20261101.cu134 |
| TensorRT-LLM | 1.4.0rc2.dev202610270008 |
| SGLang | 0.5.23 |
| SGLang PyTorch | 2.14.1+cu130 |
| SGLang PyTorch CUDA build | 13.0 |

**Version note:** vLLM and TensorRT-LLM are development/release-candidate builds. SGLang runs inside a Docker container. Its PyTorch CUDA build version (13.0) is distinct from the host CUDA Toolkit version (13.4).

### SGLang Docker Image

Original deployment image:

text

```text
lmsysorg/sglang:latest
```

Recorded immutable image identity:

text

```text
Image ID:
sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f

Repository digest:
lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f

Image creation date:
2026-10-30T01:37:57.14604124Z
```

The original deployment used the mutable `latest` tag. The recorded digest identifies the exact image used in the experiment.

## Benchmark Scope

The project evaluates three inference engines using a common model and controlled synthetic workloads.

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

All three engines use the same model architecture and model weights.

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

Each inference engine uses its own serving runtime and configuration.

| Engine | Deployment | Configuration |
| --- | --- | --- |
| vLLM | Linux environment | vLLM serving runtime, TRITON\_ATTN attention backend |
| TensorRT-LLM | Kubernetes pod | TensorRT-LLM serving runtime, configured through Triton model configuration |
| SGLang | Docker container | FlashInfer attention backend, BF16 KV cache, static memory fraction 0.85, RadixCache disabled |

The benchmark client and workload are kept consistent. Engine-specific runtime configurations are documented in their respective reports.

**Configuration note:** Identical model and benchmark workloads do not guarantee identical internal scheduling, attention kernels, memory allocation or batching behavior. These differences are retained as part of the experimental record.

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
    ├── 03-sglang-baseline.md
    └── 04-sglang-aligned-baseline.md
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
| 02 - TensorRT-LLM Baseline | Four benchmark runs, metrics and comparison | Completed |
| 03-A - SGLang Initial Baseline | Original SGLang configuration and performance measurements | Historical |
| 03-B - SGLang Aligned Baseline | Updated SGLang configuration and repeatability measurements | Completed |

---

## Benchmark Results: Three-Engine Comparison

Two separate SGLang experiments were conducted.

- **EXP-003-A:** Initial SGLang configuration.
- **EXP-003-B:** Updated SGLang configuration intended to improve comparability with the vLLM and TensorRT-LLM experiments.

EXP-003-A is retained as a historical experiment. EXP-003-B is used for the current three-engine comparison.

The comparison below uses steady-state averages from Runs 2–4.

### Steady-State Performance

| Metric | vLLM | TensorRT-LLM | SGLang EXP-003-B |
| --- | --- | --- | --- |
| Output throughput (tok/s) | **796.39** | 788.74 | 674.63 |
| Request throughput (req/s) | **3.11** | 3.08 | 2.63 |
| Mean TTFT (ms) | **3889.63** | 3880.65 | 6487.35 |
| Median TTFT (ms) | **3358.54** | 3500.30 | 6011.20 |
| P99 TTFT (ms) | **10000.01** | 10094.34 | 15654.38 |
| Mean TPOT (ms) | **9.28** | 9.51 | 10.96 |
| Median TPOT (ms) | **9.30** | 9.55 | 11.06 |
| P99 TPOT (ms) | **9.45** | 9.62 | 11.08 |
| Mean ITL (ms) | **9.28** | 9.51 | 10.96 |
| Median ITL (ms) | 9.25 | **9.51** | 9.28 |
| P99 ITL (ms) | 16.10 | **10.33** | 69.51 |

_Throughput is measured in generated output tokens per second. Latency values are in milliseconds._

Bold indicates the best measured value for each metric.

**Measurement caveat:** The P99 TTFT values for vLLM and TensorRT-LLM are close to 10 seconds. These values require further investigation for possible timeout, request-queue or measurement-boundary effects. They should not automatically be interpreted as uncapped tail-latency measurements.

### Performance Summary

#### vLLM

- Highest output throughput at 796.39 tok/s.
- Highest request throughput at 3.11 req/s.
- Lowest mean TPOT at 9.28 ms.
- Lowest median TPOT at 9.30 ms.
- Relatively consistent steady-state output throughput.

#### TensorRT-LLM

- Output throughput of 788.74 tok/s.
- Request throughput of 3.08 req/s.
- Mean TPOT of 9.51 ms.
- Performance close to vLLM under the tested configuration.
- Lowest P99 ITL at 10.33 ms.

#### SGLang EXP-003-B

- Output throughput of 674.63 tok/s.
- Request throughput of 2.63 req/s.
- Mean TTFT of 6487.35 ms.
- Mean TPOT of 10.96 ms.
- P99 ITL of 69.51 ms.
- All four benchmark executions completed successfully with zero failed requests.

### Key Findings

1. vLLM achieved the highest aggregate output throughput in the current comparison.
2. TensorRT-LLM delivered throughput close to vLLM.
3. vLLM achieved the lowest measured mean and median decode latency.
4. SGLang EXP-003-B demonstrated lower aggregate throughput and higher TTFT under its updated configuration.
5. SGLang's initial EXP-003-A measurements must remain separate because they represent a different serving configuration.
6. GPU memory allocation and attention backend selection remain important optimization considerations.
7. P99 TTFT and ITL measurements require further investigation before drawing conclusions about worst-case serving behavior.

**Overall conclusion:** Under the configurations tested in this experiment, vLLM achieved the highest output throughput and lowest mean decode latency. TensorRT-LLM performed similarly to vLLM. SGLang EXP-003-B completed the workload successfully but exhibited lower throughput and higher measured latency.

These findings are specific to the tested hardware, software versions, model and serving configurations. They should not be interpreted as a universal ranking of inference engines.

---

## SGLang Experiment History

Two separate SGLang experiments are maintained to preserve the experimental history.

### EXP-003-A: Initial Baseline

The original SGLang configuration used:

- FlashInfer attention backend.
- FP8 KV cache.
- Static memory fraction of 0.85.
- RadixCache disabled.
- Original SGLang serving configuration.

Steady-state results from Runs 2–4:

| Metric | EXP-003-A |
| --- | --- |
| Output throughput (tok/s) | 959.03 |
| Request throughput (req/s) | 3.75 |
| Mean TTFT (ms) | 144.24 |
| Median TTFT (ms) | 123.57 |
| P99 TTFT (ms) | 286.79 |
| Mean TPOT (ms) | 32.24 |
| Median TPOT (ms) | 33.54 |
| P99 TPOT (ms) | 38.79 |
| Mean ITL (ms) | 32.24 |
| Median ITL (ms) | 20.65 |
| P99 ITL (ms) | 173.89 |

### EXP-003-B: Updated Baseline

The updated experiment was conducted after reviewing the original results and aligning serving constraints more closely with the other engines.

Steady-state results from Runs 2–4:

| Metric | EXP-003-B |
| --- | --- |
| Output throughput (tok/s) | 674.63 |
| Request throughput (req/s) | 2.63 |
| Mean TTFT (ms) | 6487.35 |
| Median TTFT (ms) | 6011.20 |
| P99 TTFT (ms) | 15654.38 |
| Mean TPOT (ms) | 10.96 |
| Median TPOT (ms) | 11.06 |
| P99 TPOT (ms) | 11.08 |
| Mean ITL (ms) | 10.96 |
| Median ITL (ms) | 9.28 |
| P99 ITL (ms) | 69.51 |

### Historical Comparison

| Metric | EXP-003-A | EXP-003-B |
| --- | --- | --- |
| Output throughput (tok/s) | 959.03 | 674.63 |
| Mean TTFT (ms) | 144.24 | 6487.35 |
| Mean TPOT (ms) | 32.24 | 10.96 |
| P99 ITL (ms) | 173.89 | 69.51 |

The results demonstrate substantially different performance profiles between the two SGLang experiments.

These experiments should not be interpreted as a controlled measurement of the effect of any single parameter. Multiple serving configuration differences may contribute to the observed changes.

The complete run-level measurements and configurations are preserved in their respective reports.

---

## Benchmark Methodology

All three engines were evaluated using:

- Identical model architecture and model weights.
- Fixed input and output token lengths.
- Consistent request counts.
- Configured request rate of 5 RPS.
- The same benchmark client (`vllm bench serve`).
- OpenAI-compatible serving endpoints.
- Four benchmark executions per engine.
- Runs 2–4 for steady-state comparisons.
- Run 1 retained as the first measured execution.
- GPU utilization and memory observations where available.

### Benchmark Client

The same benchmark client was used for all engines:

bash

```bash
vllm bench serve \
    --backend openai \
    --base-url http://localhost:PORT \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random \
    --input-len 1024 \
    --output-len 256 \
    --num-prompts 100 \
    --request-rate 5
```

The API endpoint was changed according to the inference engine being tested.

### Steady-State Methodology

Four benchmark executions were performed for each engine.

- Run 1 is retained as the initial measured execution.
- Runs 2–4 are used for steady-state comparisons.
- All successful and failed requests are retained in the individual reports.
- Run-level measurements are preserved for repeatability analysis.

The benchmark uses synthetic random prompts. Results should therefore be interpreted as controlled systems-performance measurements rather than representative measurements of every production workload.

Engine-specific configurations are retained rather than forcing identical internal settings across different serving frameworks.

---

## Scope and Limitations

This repository focuses exclusively on LLM inference serving and performance benchmarking.

### Limitations

- Only one GPU architecture and model size have been evaluated.
- Synthetic random prompts do not represent every production workload.
- Different inference frameworks use different scheduling and memory-management strategies.
- Attention backends and KV-cache implementations are not necessarily equivalent.
- Results can vary with runtime versions and configuration changes.
- P99 latency measurements require additional investigation, particularly where values approach apparent timeout boundaries.
- GPU resource observations are snapshots rather than continuous measurements across every benchmark run.
- EXP-003-A and EXP-003-B represent different SGLang configurations and are not directly interchangeable.

The benchmark does not evaluate model accuracy, training performance or production workload generalization.

Results are specific to the tested hardware, software versions, model configuration and workload. Additional experiments are required to evaluate other models, GPUs, context lengths, concurrency levels and serving configurations.

The goal is reproducible systems engineering experimentation supported by actual measurements rather than theoretical performance claims.