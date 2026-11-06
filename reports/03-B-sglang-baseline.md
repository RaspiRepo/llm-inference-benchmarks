For EXP-003-B, we should preserve the same structure as EXP-003-A so both experiments can be compared section by section.

The important distinction is that EXP-003-B delivers approximately 674 tok/s, compared with approximately 959 tok/s in EXP-003-A. However, the new results are remarkably consistent across all four runs.

I would use Runs 2–4 for the steady-state comparison, while retaining all four runs in the consolidated results.

4 successful runs · 0 failures## Key calculated results — EXP-003-B

| Metric | Four-run average | Steady-state (R2–R4) |
| --- | --- | --- |
| Output throughput | 674.46 | 674.63 tok/s |
| Request throughput | 2.63 | 2.63 req/s |
| Mean TTFT | 6,511.48 | 6,487.35 ms |
| Median TTFT | 6,029.24 | 6,011.20 ms |
| P99 TTFT | 15,672.78 | 15,654.38 ms |
| Mean TPOT | 10.96 | 10.96 ms |
| Median TPOT | 11.06 | 11.06 ms |
| P99 TPOT | 11.24 | 11.08 ms |
| Mean ITL | 10.96 | 10.96 ms |
| Median ITL | 9.28 | 9.28 ms |
| P99 ITL | 69.49 | 69.51 ms |

A notable observation: unlike EXP-003-A, EXP-003-B has a much more balanced relationship between TTFT, TPOT and ITL.

Here is the complete report.

Writing

# SGLang Benchmark Report EXP-003-B

## 1\. Executive Summary

This report documents the second SGLang benchmark experiment (EXP-003-B) using Qwen2.5-1.5B-Instruct on an NVIDIA GeForce RTX 5060 Ti (16GB).

Four benchmark runs were completed successfully, each processing 100 requests with zero failures.

EXP-003-B evaluates a revised SGLang serving configuration, using a 2,048-token context limit and a maximum of eight running requests.

### SGLang — Benchmark Summary

| Metric | SGLang |
| --- | --- |
| Experiment | EXP-003-B |
| Completed runs | 1–4 |
| Successful requests | 400 |
| Failed requests | 0 |
| Steady-state runs | 2–4 |
| Output throughput (tok/s) | **674.63** |
| Request throughput (req/s) | 2.63 |
| Mean TTFT (ms) | 6487.35 |
| Median TTFT (ms) | 6011.20 |
| P99 TTFT (ms) | 15654.38 |
| Mean TPOT (ms) | 10.96 |
| Mean ITL (ms) | 10.96 |

_Throughput is in generated output tokens per second. Latency values are in milliseconds._

### Initial Findings

- SGLang achieved 674.63 output tokens/sec during steady-state testing.
- All four runs completed successfully with zero failed requests.
- Output throughput varied by approximately 0.27% across Runs 2–4.
- Mean TTFT averaged 6487.35 ms during steady-state testing.
- Mean TPOT averaged 10.96 ms.
- Mean ITL averaged 10.96 ms.
- The new configuration demonstrated remarkably consistent results across all four executions.

**Conclusion:** EXP-003-B provides a stable SGLang performance baseline. Throughput and latency measurements were highly consistent across repeated executions. The results provide a useful comparison against EXP-003-A and establish a baseline for evaluating the effects of SGLang serving configuration changes.

These results represent measured performance for the tested configuration, not a general ranking of inference engines.

---

## 2\. Test Environment

### Hardware

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| GPU VRAM | 16GB |
| Model | Qwen2.5-1.5B-Instruct |
| Framework | SGLang |
| Deployment | Docker container |
| API | OpenAI-compatible HTTP API |

### SGLang Runtime

The SGLang server runs inside a Docker container using the `lmsysorg/sglang:latest` image.

Runtime versions were retrieved directly from the container.

| Component | Version |
| --- | --- |
| SGLang | 0.5.23 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA build | 13.0 |
| CUDA availability | True |
| Docker image | lmsysorg/sglang:latest |
| Image creation date | 2026-10-30 |

**Docker image identity:**

text

```text
Image ID:
sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f

Repository digest:
lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f
```

The image digest identifies the exact container image used, even though the deployment command references the mutable `latest` tag.

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
| Serving port | 30000 |

---

## 3\. Server Configuration

### Docker Deployment

Container name:

`sglang-bench-server`

Container image:

`lmsysorg/sglang:latest`

### Startup Configuration

The EXP-003-B configuration uses an explicit context length and maximum running request limit.

bash

```bash
HF_CACHE_DIR="/home/rtx5060/.cache/huggingface"

docker run -d --name sglang-bench-server \
  --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --context-length 2048 \
    --max-running-requests 8 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --disable-radix-cache
```

### Server Configuration Summary

| Parameter | Value |
| --- | --- |
| Model path | Local Hugging Face snapshot |
| Host | 0.0.0.0 |
| Port | 30000 |
| Context length | 2048 |
| Maximum running requests | 8 |
| KV-cache dtype | BF16 (runtime-reported) |
| Static memory fraction | 0.85 |
| Attention backend | FlashInfer |
| RadixCache | Disabled |
| GPU access | All available GPUs |
| Shared memory | 32 GB |

**KV-cache observation**

During server initialization, SGLang reported:

text

```text
KV Cache is allocated.
dtype: torch.bfloat16
#tokens: 356651
K size: 4.76 GB
V size: 4.76 GB
```

This confirms that the KV cache was allocated using BF16 rather than FP8.

The actual runtime-reported dtype is used in this report rather than assuming the configured or intended dtype.

Disabling RadixCache means this experiment does not evaluate prefix-cache reuse capabilities.

### API Verification

bash

```bash
curl http://localhost:30000/v1/models
```

The server successfully exposed the Qwen2.5-1.5B-Instruct model through its OpenAI-compatible API.

---

## 4\. Benchmark Execution

The same benchmark workload was used for all four runs.

bash

```bash
vllm bench serve \
    --backend openai \
    --base-url http://localhost:30000 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random \
    --input-len 1024 \
    --output-len 256 \
    --num-prompts 100 \
    --request-rate 5 \
    --label results_sglang_exp003b
```

Only the output label should be changed between runs to preserve individual results.

The benchmark client communicated with SGLang through its OpenAI-compatible endpoint.

---

## 5\. Benchmark Results

### Consolidated Results — Four Runs

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- | --- |
| Successful requests | 100 | 100 | 100 | 100 |
| Failed requests | 0 | 0 | 0 | 0 |
| Duration (s) | 37.98 | 37.90 | 37.97 | 37.97 |
| Request throughput | 2.63 | 2.64 | 2.63 | 2.63 |
| Output throughput (tok/s) | 673.95 | 675.39 | 674.29 | 674.22 |
| Peak output throughput | 864 | 864 | 864 | 864 |
| Total token throughput | 3369.77 | 3376.96 | 3371.43 | 3371.08 |
| Peak concurrency | 52 | 52 | 52 | 53 |
| Mean TTFT (ms) | 6583.89 | 6478.96 | 6489.99 | 6493.09 |
| Median TTFT (ms) | 6083.36 | 5992.94 | 6017.72 | 6022.94 |
| P99 TTFT (ms) | 15728.04 | 15610.54 | 15672.57 | 15680.02 |
| Mean TPOT (ms) | 10.94 | 10.91 | 10.99 | 10.99 |
| Median TPOT (ms) | 11.04 | 11.05 | 11.07 | 11.07 |
| P99 TPOT (ms) | 11.72 | 11.07 | 11.08 | 11.09 |
| Mean ITL (ms) | 10.94 | 10.91 | 10.99 | 10.99 |
| Median ITL (ms) | 9.27 | 9.28 | 9.28 | 9.28 |
| P99 ITL (ms) | 69.44 | 69.43 | 69.55 | 69.55 |

### Four-Run Average

| Metric | Average |
| --- | --- |
| Benchmark duration | 37.96 s |
| Request throughput | 2.63 req/s |
| Output throughput | 674.46 tok/s |
| Peak output throughput | 864 tok/s |
| Total token throughput | 3372.31 tok/s |
| Peak concurrency | 52.25 |
| Mean TTFT | 6511.48 ms |
| Median TTFT | 6029.24 ms |
| P99 TTFT | 15672.78 ms |
| Mean TPOT | 10.96 ms |
| Median TPOT | 11.06 ms |
| P99 TPOT | 11.24 ms |
| Mean ITL | 10.96 ms |
| Median ITL | 9.28 ms |
| P99 ITL | 69.49 ms |

### Steady-State Average — Runs 2–4

For consistency with the other framework reports, Runs 2–4 are used for the steady-state comparison.

| Metric | Average |
| --- | --- |
| Benchmark duration | 37.95 s |
| Request throughput | 2.63 req/s |
| Output throughput | **674.63 tok/s** |
| Peak output throughput | 864 tok/s |
| Total token throughput | 3373.16 tok/s |
| Peak concurrency | 52.33 |
| Mean TTFT | 6487.35 ms |
| Median TTFT | 6011.20 ms |
| P99 TTFT | 15654.38 ms |
| Mean TPOT | 10.96 ms |
| Median TPOT | 11.06 ms |
| P99 TPOT | 11.08 ms |
| Mean ITL | 10.96 ms |
| Median ITL | 9.28 ms |
| P99 ITL | 69.51 ms |

---

## 6\. Run-to-Run Consistency

All four runs produced remarkably similar results.

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- | --- |
| Output throughput | 673.95 | 675.39 | 674.29 | 674.22 |
| Mean TTFT | 6583.89 | 6478.96 | 6489.99 | 6493.09 |
| Mean TPOT | 10.94 | 10.91 | 10.99 | 10.99 |
| Mean ITL | 10.94 | 10.91 | 10.99 | 10.99 |
| P99 ITL | 69.44 | 69.43 | 69.55 | 69.55 |

Output throughput varied by approximately 0.21% across all four runs.

Mean TTFT varied by approximately 1.61% relative to its four-run average.

Mean TPOT varied by approximately 0.73% across all four runs.

These results demonstrate excellent repeatability under the tested conditions.

Unlike the earlier experiment, Run 1 did not exhibit a substantial performance difference from subsequent runs.

For consistency with the other reports, Run 1 is retained in the complete benchmark history but excluded from the steady-state comparison.

---

## 7\. GPU Resource Observation

No GPU monitoring snapshot was provided for EXP-003-B.

Therefore, GPU temperature, power consumption, utilization and VRAM consumption are not reported for this experiment.

The KV-cache allocation log confirms that approximately 9.52 GB was allocated for K and V cache storage.

This is a runtime allocation measurement and should not be interpreted as total GPU VRAM consumption.

Future experiments should capture `nvidia-smi` measurements during benchmark execution to provide comparable resource observations.

---

## 8\. Engineering Conclusions

### Performance

SGLang achieved approximately 675 output tokens/sec during steady-state testing.

Runs 2–4 demonstrated exceptionally consistent output throughput, ranging from 674.22 to 675.39 tok/s.

The four-run average was 674.46 tok/s.

### Time to First Token

The steady-state mean TTFT was 6487.35 ms.

The P99 TTFT was approximately 15.65 seconds.

These results indicate substantial request latency before the first generated token under the tested workload.

### Decode Performance

Mean TPOT averaged 10.96 ms during steady-state testing.

Mean ITL was also 10.96 ms.

Compared with EXP-003-A, this experiment demonstrates a substantially different decode-latency profile.

### Repeatability

Runs 1–4 demonstrated excellent consistency across throughput and latency metrics.

Output throughput varied by only approximately 0.21% across all four executions.

This provides a reliable baseline for subsequent configuration experiments.

### Memory Considerations

The runtime reported BF16 KV-cache allocation, with approximately 4.76 GB allocated for each of K and V.

The static memory fraction was configured to 0.85.

These settings should be considered when evaluating memory consumption and comparing this experiment with other SGLang configurations.

### Final Assessment

EXP-003-B successfully completed all four benchmark executions with zero failed requests.

The measured steady-state output throughput was 674.63 tok/s, with a mean TTFT of 6487.35 ms and mean TPOT of 10.96 ms.

The experiment demonstrates highly repeatable serving performance and establishes a useful baseline for evaluating SGLang configuration changes.

---

## 9\. Benchmark Scope and Limitations

- Single GPU: NVIDIA RTX 5060 Ti 16GB.
- Single model: Qwen2.5-1.5B-Instruct.
- Synthetic random workload.
- Fixed input and output lengths.
- Configured request rate of 5 RPS.
- Four executions.
- Sequential benchmark executions.
- No randomized or paired execution order.
- Context length limited to 2048 tokens.
- Maximum running requests set to 8.
- FlashInfer attention backend.
- BF16 KV cache, as reported by the runtime.
- RadixCache disabled.
- GPU resource snapshots unavailable.
- Docker image originally referenced through the mutable `latest` tag; image digest recorded.
- Results are specific to the tested software versions and configuration.

These results should not be generalized to other models, GPUs, workloads or SGLang configurations without further evaluation.

The goal is reproducible systems engineering experimentation supported by actual measurements rather than theoretical performance claims.