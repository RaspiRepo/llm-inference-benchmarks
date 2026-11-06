# SGLang Benchmark Runbook

**Experiment:** EXP-003\
**Model:** Qwen2.5-1.5B-Instruct\
**Hardware:** NVIDIA RTX 5060 Ti 16GB\
**Serving runtime:** SGLang\
**Deployment:** Docker\
**Benchmark client:** vLLM `bench serve`

## 1\. Objective

Establish an SGLang performance baseline using the same model and synthetic workload as EXP-001 (vLLM) and EXP-002 (TensorRT-LLM).

The experiment evaluates:

- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- Inter-token Latency (ITL)
- Request throughput
- Aggregate output token throughput
- Peak request concurrency
- Request success rate

The SGLang results will be compared against the existing vLLM and TensorRT-LLM baselines.

## 2\. Deployment Environment

SGLang runs inside a Docker container on the Linux host.

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA RTX 5060 Ti |
| GPU memory | 16 GB |
| Deployment | Docker container |
| Container name | sglang-bench-server |
| Container image | lmsysorg/sglang:latest |
| Inference engine | SGLang |
| Model | Qwen2.5-1.5B-Instruct |
| Model identifier | Qwen/Qwen2.5-1.5B-Instruct |
| Model loading | Local Hugging Face snapshot |
| Serving port | 30000 |
| API | OpenAI-compatible REST API |

Verify the running container:

bash

```bash
docker ps
```

Check GPU visibility inside the container:

bash

```bash
docker exec -it sglang-bench-server nvidia-smi
```

Inspect the container image and SGLang version:

bash

```bash
docker exec sglang-bench-server python3 -m sglang.version
```

If the version module is unavailable, inspect the installed package:

bash

```bash
docker exec sglang-bench-server python3 -c \
  "import sglang; print(sglang.__version__)"
```

Record the exact Docker image digest and SGLang version. The `latest` tag is mutable, so it should not be used as the only version identifier in the final report.

## 3\. SGLang Server Configuration

The SGLang server is configured through command-line arguments.

Initial configuration:

bash

```bash
python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --disable-radix-cache
```

### Configuration parameters

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--model-path` | Local model snapshot | Loads the existing Hugging Face model |
| `--host` | `0.0.0.0` | Makes the API accessible outside the container |
| `--port` | `30000` | API listening port |
| `--trust-remote-code` | Enabled | Allows model-provided custom code |
| `--kv-cache-dtype` | `fp8_e4m3` | Uses FP8 KV-cache storage |
| `--mem-fraction-static` | `0.85` | Sets the static GPU memory allocation fraction |
| `--attention-backend` | `flashinfer` | Selects the FlashInfer attention backend |
| `--disable-radix-cache` | Enabled | Disables prefix KV-cache reuse |

**Configuration note:** These are the actual initial settings used for this experiment. Validate their support against the installed SGLang version before changing them.

The FP8 KV-cache setting is a configuration difference that must be documented when comparing engines. It does not necessarily imply that the model weights are FP8.

Disabling RadixCache also means this experiment does not evaluate SGLang's prefix-cache reuse advantage.

## 4\. Start SGLang Server

The server is launched using Docker:

bash

```bash
docker run -d \
  --name sglang-bench-server \
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
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --disable-radix-cache
```

Keep the container running during benchmarking.

Check container status:

bash

```bash
docker ps -a --filter name=sglang-bench-server
```

Inspect server logs:

bash

```bash
docker logs -f sglang-bench-server
```

Wait for model initialization and server startup to complete.

Do not start benchmarking if the container exits or the server reports attention-kernel, CUDA, model-loading, or GPU-memory errors.

## 5\. Expose the API to the Benchmark Client

Docker publishes container port 30000 on the host:

bash

```bash
-p 30000:30000
```

The benchmark endpoint is:

text

```text
http://localhost:30000
```

No additional port forwarding is required when the benchmark client runs directly on the same host.

If port 30000 is already occupied, change the host-side port mapping and update the benchmark URL accordingly.

## 6\. Verify API Availability

Check model availability:

bash

```bash
curl http://localhost:30000/v1/models
```

Check server health:

bash

```bash
curl http://localhost:30000/health
```

Send a test inference request:

bash

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Explain GPU inference in one paragraph."
      }
    ],
    "max_tokens": 50
  }'
```

Do not start benchmarking until the model endpoint responds successfully.

## 7\. Warm-up Procedure

Execute a preliminary inference request before collecting benchmark results.

The warm-up request is not included in the measured benchmark.

Apply the same warm-up policy used for vLLM and TensorRT-LLM.

Allow model initialization and any initial kernel compilation to complete before proceeding.

## 8\. Run Serving Benchmark

Use the same benchmark client and workload as EXP-001 and EXP-002.

Run from the host benchmark environment:

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

### Benchmark parameters

| Parameter | Value |
| --- | --- |
| Dataset | Random synthetic |
| Input length | 1024 tokens |
| Output length | 256 tokens |
| Number of requests | 100 |
| Request rate | 5 RPS |
| Backend | OpenAI-compatible |
| Model | Qwen2.5-1.5B-Instruct |

The benchmark client must be able to access the SGLang OpenAI-compatible endpoint.

## 9\. Repeat Benchmark Runs

Execute four benchmark runs using identical configurations.

| Run | Label | Status |
| --- | --- | --- |
| Run 1 | `results_sglang_qwen_run01` | Pending |
| Run 2 | `results_sglang_qwen_run02` | Pending |
| Run 3 | `results_sglang_qwen_run03` | Pending |
| Run 4 | `results_sglang_qwen_run04` | Pending |

Use the following label for each run:

bash

```bash
--label results_sglang_qwen_run01
```

Change only the label between runs.

Do not change server or benchmark parameters during the experiment.

All four runs must be preserved. Runs 2–4 will be used for the steady-state comparison, consistent with the existing benchmark methodology.

## 10\. GPU Monitoring

Monitor GPU utilization and memory during each experiment.

bash

```bash
watch -n 1 nvidia-smi
```

Where available, record:

- GPU utilization
- GPU memory consumption
- GPU temperature
- GPU clock frequency
- GPU power consumption

Inspect SGLang server metrics where supported:

bash

```bash
curl http://localhost:30000/metrics
```

These measurements provide additional context when interpreting benchmark results.

## 11\. Result Preservation

Store the original benchmark outputs independently.

text

```text
results/
└── EXP-003/
    ├── run-01-sglang.log
    ├── run-02-sglang.log
    ├── run-03-sglang.log
    └── run-04-sglang.log
```

Each run should preserve:

- Successful and failed requests.
- Benchmark duration.
- Request throughput.
- Output token throughput.
- Mean, median, and P99 TTFT.
- Mean, median, and P99 TPOT.
- Mean, median, and P99 ITL.
- Peak concurrency.

Record the SGLang version, Docker image digest, CUDA runtime, PyTorch version, model revision, model precision, KV-cache dtype, attention backend, and server configuration.

## 12\. Experimental Controls

For comparison with EXP-001 and EXP-002:

- Use the same Qwen2.5-1.5B-Instruct model revision.
- Match model weight precision.
- Document the FP8 KV-cache configuration.
- Use identical input and output lengths.
- Use the same request count and request rate.
- Apply the same warm-up procedure.
- Run four measured executions.
- Avoid competing workloads on the GPU.
- Preserve the original benchmark output.

The inference engines do not need identical internal configurations, but their effective workload and hardware constraints must be comparable.

Any differences in model precision, KV-cache representation, scheduling, and attention backend must be documented.

## 13\. Expected Deliverables

After completing EXP-003:

| Deliverable | Output |
| --- | --- |
| SGLang baseline | 4 successful benchmark runs |
| Raw benchmark logs | EXP-003 results |
| Performance summary | Mean and run-to-run range |
| GPU observations | Utilization and VRAM |
| Experiment report | `reports/03-sglang-baseline.md` |

The final objective is to compare EXP-003 against EXP-001 and EXP-002 and produce a reproducible three-engine inference performance evaluation.

---
