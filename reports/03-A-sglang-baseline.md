# SGLang Benchmark Report EXP-003-A

## 1. Executive Summary

This report documents the SGLang baseline benchmark using Qwen2.5-1.5B-Instruct on an NVIDIA GeForce RTX 5060 Ti (16GB).

Four benchmark runs were completed successfully, each processing 100 requests with zero failures.

### SGLang — Benchmark Summary

| Metric | SGLang |
| --- | --- |
| Completed runs | 1–4 |
| Successful requests | 400 |
| Failed requests | 0 |
| Steady-state runs | 2–4 |
| Output throughput (tok/s) | 959.03 |
| Request throughput (req/s) | 3.75 |
| Mean TTFT (ms) | 144.24 |
| Median TTFT (ms) | 123.57 |
| P99 TTFT (ms) | 286.79 |
| Mean TPOT (ms) | 32.24 |
| Mean ITL (ms) | 32.24 |

_Throughput is in generated output tokens per second. Latency values are in milliseconds._

### Initial Findings

- SGLang achieved 959.03 output tokens/sec during steady-state testing.
- Mean TTFT was 144.24 ms across Runs 2–4.
- Mean TPOT was 32.24 ms.
- Output throughput varied by approximately 3.35% across Runs 2–4.
- All four runs completed successfully with zero failed requests.
- GPU memory consumption reached 16,008 MiB during Runs 3 and 4.

**Conclusion:** SGLang successfully completed the benchmark workload and demonstrated consistent first-token latency and reasonably stable throughput. Its measured performance shows a different balance between first-token latency and decode latency compared with the existing inference engines.

These results represent measured performance for the tested configuration, not a general ranking of inference engines.

---

## 2. Test Environment

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

Runtime versions were retrieved directly from the running container.

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

## 3. Server Configuration

### Docker deployment

Container name:

`sglang-bench-server`

Container image:

`lmsysorg/sglang:latest`

### Startup command

bash

```bash

python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct', force_download=True, local_files_only=False)"


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

### Server configuration summary

| Parameter | Value |
| --- | --- |
| Model path | Local Hugging Face snapshot |
| Host | 0.0.0.0 |
| Port | 30000 |
| KV-cache dtype | fp8\_e4m3 |
| Static memory fraction | 0.85 |
| Attention backend | flashinfer |
| RadixCache | Disabled |
| GPU access | All available GPUs |
| Shared memory | 32 GB |

The experiment uses FP8 KV-cache storage. This configuration should be considered when interpreting memory consumption and performance.

Disabling RadixCache means the benchmark does not evaluate SGLang's prefix-cache reuse capabilities.

### API verification

bash

```bash
curl http://localhost:30000/v1/models
```

The server exposes an OpenAI-compatible API on port 30000.

The exact successful API response should be retained with the experiment logs.

---

## 4. Benchmark Execution

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
    --label results_sglang_qwen
```

Only the output label was changed between runs to preserve individual results.

The benchmark client communicated with SGLang through its OpenAI-compatible endpoint.

---

## 5. Benchmark Results

### Consolidated Results — Four Runs

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- | --- |
| Successful requests | 100 | 100 | 100 | 100 |
| Failed requests | 0 | 0 | 0 | 0 |
| Duration (s) | 25.34 | 27.08 | 26.19 | 26.83 |
| Request throughput | 3.95 | 3.69 | 3.82 | 3.73 |
| Output throughput (tok/s) | 1010.28 | 945.45 | 977.55 | 954.10 |
| Peak output throughput | 2335 | 1896 | 2116 | 2038 |
| Total token throughput | 5051.39 | 4727.27 | 4887.74 | 4770.52 |
| Peak concurrency | 61 | 62 | 63 | 61 |
| Mean TTFT (ms) | 149.70 | 144.87 | 143.51 | 144.33 |
| Median TTFT (ms) | 121.73 | 128.56 | 121.34 | 120.82 |
| P99 TTFT (ms) | 400.20 | 288.71 | 293.18 | 278.48 |
| Mean TPOT (ms) | 29.59 | 32.46 | 33.45 | 30.82 |
| Median TPOT (ms) | 30.71 | 33.95 | 35.06 | 31.61 |
| P99 TPOT (ms) | 36.01 | 38.62 | 39.69 | 38.05 |
| Mean ITL (ms) | 29.59 | 32.46 | 33.45 | 30.82 |
| Median ITL (ms) | 18.23 | 20.87 | 22.20 | 18.87 |
| P99 ITL (ms) | 160.33 | 158.53 | 181.51 | 181.63 |

### Four-Run Average

| Metric | Average |
| --- | --- |
| Benchmark duration | 26.36 s |
| Request throughput | 3.80 req/s |
| Output throughput | 971.85 tok/s |
| Peak output throughput | 2096.25 tok/s |
| Total token throughput | 4859.23 tok/s |
| Peak concurrency | 61.75 |
| Mean TTFT | 145.60 ms |
| Median TTFT | 123.11 ms |
| P99 TTFT | 315.14 ms |
| Mean TPOT | 31.58 ms |
| Median TPOT | 32.83 ms |
| P99 TPOT | 38.09 ms |
| Mean ITL | 31.58 ms |
| Median ITL | 20.04 ms |
| P99 ITL | 170.50 ms |

### Steady-State Average — Runs 2–4

Run 1 demonstrated higher output throughput than the subsequent runs.

For the steady-state comparison, Runs 2–4 are used.

| Metric | Average |
| --- | --- |
| Benchmark duration | 26.70 s |
| Request throughput | 3.75 req/s |
| Output throughput | **959.03 tok/s** |
| Peak output throughput | 2016.67 tok/s |
| Total token throughput | 4795.18 tok/s |
| Peak concurrency | 62.00 |
| Mean TTFT | 144.24 ms |
| Median TTFT | 123.57 ms |
| P99 TTFT | 286.79 ms |
| Mean TPOT | 32.24 ms |
| Median TPOT | 33.54 ms |
| P99 TPOT | 38.79 ms |
| Mean ITL | 32.24 ms |
| Median ITL | 20.65 ms |
| P99 ITL | 173.89 ms |

---

## 6. Run-to-Run Consistency

Runs 2, 3 and 4 produced relatively consistent throughput and TTFT measurements.

| Metric | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- |
| Output throughput | 945.45 | 977.55 | 954.10 |
| Mean TTFT | 144.87 | 143.51 | 144.33 |
| Mean TPOT | 32.46 | 33.45 | 30.82 |
| Mean ITL | 32.46 | 33.45 | 30.82 |
| P99 ITL | 158.53 | 181.51 | 181.63 |

Output throughput varied by approximately 3.35% across Runs 2–4.

Mean TTFT varied by approximately 0.94% relative to its steady-state average.

Mean TPOT showed greater variation, ranging from 30.82 to 33.45 ms.

This demonstrates good repeatability in first-token latency and reasonable throughput consistency under the tested conditions.

Run 1 was faster, with 1010.28 tok/s output throughput. Since the subsequent runs were more consistent, Run 1 is retained as part of the complete benchmark history but excluded from the steady-state comparison.

---


### Resource summary

| Metric | Run 3 | Run 4 |
| --- | --- | --- |
| GPU temperature | 44°C | 45°C |
| GPU power draw | 68 W | 106 W |
| GPU power limit | 180 W | 180 W |
| VRAM used | 16,008 MiB | 16,008 MiB |
| VRAM total | 16,311 MiB | 16,311 MiB |
| VRAM utilization | 98.1% | 98.1% |
| Memory utilization | 72% | 81% |

The GPU was operating with approximately 303 MiB of reported VRAM headroom.

This indicates substantial memory consumption on the RTX 5060 Ti 16GB.

Future experiments involving longer contexts, larger batches or higher concurrency should carefully monitor memory pressure.

Note: These resource snapshots were captured during Runs 3 and 4 and should not be interpreted as averages across all four executions.

---

## 7. Engineering Conclusions

### Performance

SGLang achieved approximately 959 output tokens/sec during steady-state testing.

Runs 2–4 demonstrated relatively consistent output throughput, ranging from 945.45 to 977.55 tok/s.

The four-run average was 971.85 tok/s.

### Time to First Token

SGLang achieved a steady-state mean TTFT of 144.24 ms.

Runs 2–4 produced particularly consistent first-token latency, with measurements between 143.51 and 144.87 ms.

This is a notable characteristic of the tested configuration.

### Decode Performance

Mean TPOT averaged 32.24 ms during steady-state testing.

The result was higher than the existing vLLM and TensorRT-LLM measurements, indicating a different decode-latency profile.

### Repeatability

Runs 2–4 demonstrated good consistency in throughput and TTFT.

Run 1 achieved higher output throughput but was retained as part of the complete benchmark history.

Using Runs 2–4 provides a consistent operational baseline.

### Memory Constraints

The observed VRAM utilization of approximately 98.1% indicates very limited memory headroom.

The FP8 KV-cache configuration and static memory allocation setting should be considered when evaluating SGLang's memory requirements.

### Final Assessment

SGLang successfully completed all four benchmark executions with zero failed requests.

The measured steady-state output throughput was 959.03 tok/s, with a mean TTFT of 144.24 ms and mean TPOT of 32.24 ms.

SGLang is a viable serving option on the tested consumer GPU hardware. However, high GPU memory consumption and decode latency are important considerations for future optimization experiments.

---

## 8. Benchmark Scope and Limitations

- Single GPU: NVIDIA RTX 5060 Ti 16GB.
- Single model: Qwen2.5-1.5B-Instruct.
- Synthetic random workload.
- Fixed input and output lengths.
- Configured request rate of 5 RPS.
- Four executions.
- Sequential benchmark executions.
- No randomized or paired execution order.
- FP8 KV-cache enabled.
- RadixCache disabled.
- GPU resource snapshots captured during Runs 3 and 4.
- Docker image uses the mutable `latest` tag.
- Exact SGLang, PyTorch and CUDA runtime versions still need to be recorded.
- Results are specific to the tested software versions and configurations.

These results should not be generalized to other models, GPUs, workloads or SGLang configurations without further evaluation.
