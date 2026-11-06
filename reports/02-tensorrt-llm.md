# TensorRT-LLM Benchmark Report

## 1\. Executive Summary

This report documents the TensorRT-LLM baseline benchmark using Qwen2.5-1.5B-Instruct on an NVIDIA GeForce RTX 5060 Ti (16GB).

Four benchmark runs were completed successfully, each processing 100 requests with zero failures.

### vLLM vs TensorRT-LLM — Steady-State Comparison

Both engines were evaluated using the same model, request workload, input/output token lengths, and configured request rate.

| Metric | vLLM | TensorRT-LLM |
| --- | --- | --- |
| Steady-state runs | 2–4 | 2–4 |
| Output throughput (tok/s) | 796.39 | 788.74 |
| Request throughput (req/s) | 3.11 | 3.08 |
| Mean TTFT (ms) | 3889.63 | 3880.65 |
| Median TTFT (ms) | 3358.47 | 3500.30 |
| P99 TTFT (ms) | 10000.87 | 10094.34 |
| Mean TPOT (ms) | 9.28 | 9.513 |
| Mean ITL (ms) | 9.28 | 9.513 |

_Throughput is in generated output tokens per second. Latency values are in milliseconds._

### Initial Findings

- vLLM achieved approximately 0.97% higher output throughput.
- TensorRT-LLM achieved approximately 9 ms lower mean TTFT.
- vLLM demonstrated approximately 2.45% lower mean TPOT.
- Both engines delivered very similar steady-state performance.

**Conclusion:** Under this specific workload and hardware configuration, neither engine demonstrates a substantial performance advantage. vLLM has a marginal throughput and decode-latency advantage, while TensorRT-LLM has marginally lower mean TTFT.

These results represent measured performance for the tested configurations, not a general ranking of the two inference engines.

---

## 2\. Test Environment

### Hardware

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| GPU VRAM | 16GB |
| Model | Qwen2.5-1.5B-Instruct |
| Framework | TensorRT-LLM |
| Backend | PyTorch |
| Deployment | Kubernetes pod |
| API | OpenAI-compatible HTTP API |

### TensorRT-LLM Runtime

Version observed during execution:

text

```text
TensorRT-LLM: 1.4.0rc2.dev202610270008
PyTorch: 2.14.0a0+b2c75dd062.nv26.09
```

The runtime reported a warning about skipping C++ extensions due to the torch/torchao version combination. The server nevertheless started successfully and completed all benchmark requests.

### Benchmark Workload

| Parameter | Value |
| --- | --- |
| Backend | openai |
| Dataset | random |
| Input length | 1024 tokens |
| Output length | 256 tokens |
| Number of prompts | 100 |
| Request rate | 5 RPS |
| Successful requests expected | 100 |
| Maximum sequence length | 2048 |
| Maximum batch size | 8 |

---

## 3\. Server Configuration

### Configuration file

`/opt/triton_models/config.yml`

yaml

```yaml
enable_iter_perf_stats: true
```

### Startup command

bash

```bash
trtllm-serve serve Qwen/Qwen2.5-1.5B-Instruct \
  --config /opt/triton_models/config.yml \
  --max_seq_len 2048 \
  --max_batch_size 8
```


### API verification

bash

```bash
curl http://localhost:8000/v1/models
```

Successful response:

json

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-1.5B-Instruct",
      "object": "model",
      "owned_by": "tensorrt_llm"
    }
  ]
}
```

This confirms that the model was registered and the OpenAI-compatible API was reachable.

---

## 4\. Benchmark Execution

The same benchmark workload was used for all four runs.

bash

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
    --label results_trtllm_qwen
```

Only the output label should be changed between runs to preserve individual results.

---

## 5\. Benchmark Results

### Consolidated Results — Four Runs

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- | --- |
| Successful requests | 100 | 100 | 100 | 100 |
| Failed requests | 0 | 0 | 0 | 0 |
| Duration (s) | 38.27 | 32.47 | 32.46 | 32.45 |
| Request throughput | 2.61 | 3.08 | 3.08 | 3.08 |
| Output throughput (tok/s) | 668.98 | 788.52 | 788.78 | 788.92 |
| Peak output throughput | 840 | 864 | 856 | 856 |
| Total token throughput | 3344.88 | 3942.58 | 3943.92 | 3944.59 |
| Peak concurrency | 53 | 44 | 44 | 44 |
| Mean TTFT (ms) | 6615.07 | 3908.30 | 3879.45 | 3854.19 |
| Median TTFT (ms) | 6078.44 | 3518.50 | 3533.76 | 3448.64 |
| P99 TTFT (ms) | 15908.64 | 10120.95 | 10082.21 | 10079.86 |
| Mean TPOT (ms) | 11.13 | 9.52 | 9.51 | 9.51 |
| Median TPOT (ms) | 11.23 | 9.57 | 9.53 | 9.54 |
| P99 TPOT (ms) | 11.32 | 9.62 | 9.63 | 9.61 |
| Mean ITL (ms) | 11.13 | 9.52 | 9.51 | 9.51 |
| Median ITL (ms) | 9.52 | 9.52 | 9.50 | 9.51 |
| P99 ITL (ms) | 56.87 | 10.32 | 10.35 | 10.33 |

### Four-Run Average

| Metric | Average |
| --- | --- |
| Benchmark duration | 33.91 s |
| Request throughput | 2.96 req/s |
| Output throughput | 758.80 tok/s |
| Peak output throughput | 854 tok/s |
| Total token throughput | 3793.99 tok/s |
| Mean TTFT | 4564.25 ms |
| Mean TPOT | 9.9175 ms |
| Mean ITL | 9.9175 ms |

### Steady-State Average — Runs 2–4

Run 1 demonstrated lower throughput and higher latency than the subsequent runs.

For the steady-state comparison with vLLM, Runs 2–4 are used.

| Metric | Average |
| --- | --- |
| Benchmark duration | 32.46 s |
| Request throughput | 3.08 req/s |
| Output throughput | **788.74 tok/s** |
| Peak output throughput | 858.67 tok/s |
| Total token throughput | 3943.70 tok/s |
| Mean TTFT | 3880.65 ms |
| Median TTFT | 3500.30 ms |
| P99 TTFT | 10094.34 ms |
| Mean TPOT | 9.513 ms |
| Median TPOT | 9.547 ms |
| P99 TPOT | 9.62 ms |
| Mean ITL | 9.513 ms |
| Median ITL | 9.51 ms |
| P99 ITL | 10.333 ms |

---

## 6\. Run-to-Run Consistency

Runs 2, 3 and 4 produced remarkably similar results.

| Metric | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- |
| Output throughput | 788.52 | 788.78 | 788.92 |
| Mean TTFT | 3908.30 | 3879.45 | 3854.19 |
| Mean TPOT | 9.52 | 9.51 | 9.51 |
| Mean ITL | 9.52 | 9.51 | 9.51 |
| P99 ITL | 10.32 | 10.35 | 10.33 |

Output throughput varied by approximately 0.05% across Runs 2–4.

This demonstrates strong repeatability under the tested conditions.

Run 1 was slower, with 668.98 tok/s output throughput and 6615.07 ms mean TTFT. Since the subsequent runs were consistent, Run 1 is retained as part of the complete benchmark history but excluded from the steady-state engine comparison.

---

## 7\. GPU Resource Observation

During Run 3, GPU monitoring reported:

text

```text
52C    P1
103W / 180W
15194MiB / 16311MiB
95% GPU utilization
```

### Resource summary

| Metric | Observed |
| --- | --- |
| GPU temperature | 52°C |
| GPU power draw | 103 W |
| GPU power limit | 180 W |
| VRAM used | 15,194 MiB |
| VRAM total | 16,311 MiB |
| VRAM utilization | 93.1% |
| GPU utilization | 95% |

The GPU was operating at high utilization, with approximately 1.1 GiB of VRAM remaining.

This is an important observation for future experiments involving longer contexts, larger batches or higher concurrency.

Note: This resource snapshot was captured during Run 3 and should not be interpreted as an average across all four executions.

---

## 8\. Engineering Conclusions

### Performance

TensorRT-LLM achieved approximately 789 output tokens/sec during steady-state testing.

Compared with the vLLM baseline, its throughput was approximately 0.97% lower. Mean TTFT was nearly identical, while mean TPOT was approximately 2.5% higher.

For this particular model, workload and GPU configuration, both engines deliver comparable inference performance.

### Repeatability

Runs 2–4 demonstrated strong consistency in throughput and latency.

This supports using their averages as the operational baseline while retaining Run 1 for completeness.

### Memory Constraints

The observed VRAM utilization of approximately 93.1% indicates limited memory headroom on the RTX 5060 Ti 16GB.

Future optimization experiments should consider memory utilization alongside throughput and latency.

### Final Assessment

TensorRT-LLM is a viable serving option on the tested consumer GPU hardware. However, this experiment does not show a meaningful throughput advantage over vLLM.

The results provide an empirical baseline for subsequent optimization and comparison experiments.

---

## 9\. Benchmark Scope and Limitations

- Single GPU: NVIDIA RTX 5060 Ti 16GB.
- Single model: Qwen2.5-1.5B-Instruct.
- Synthetic random workload.
- Fixed input and output lengths.
- Configured request rate of 5 RPS.
- Four executions per engine.
- Sequential benchmark executions.
- No randomized or paired execution order.
- GPU resource snapshot captured during one TensorRT-LLM run.
- Results are specific to the tested software versions and configurations.

These results should not be generalized to other models, GPUs, workloads or TensorRT-LLM backends without further evaluation.
