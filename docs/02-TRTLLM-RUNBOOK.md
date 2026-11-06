# TensorRT-LLM Benchmark Runbook

**Experiment:** EXP-002\
**Model:** Qwen2.5-1.5B-Instruct\
**Hardware:** NVIDIA RTX 5060 Ti 16GB\
**Serving runtime:** TensorRT-LLM\
**Benchmark client:** vLLM `bench serve`

## 1\. Objective

Establish a TensorRT-LLM performance baseline using the same model and synthetic workload as EXP-001 (vLLM).

The experiment evaluates:

- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- Inter-token Latency (ITL)
- Request throughput
- Aggregate output token throughput
- Peak request concurrency
- Request success rate

The TensorRT-LLM results will be compared against the vLLM baseline.

## 2\. Deployment Environment

TensorRT-LLM runs inside the existing Kubernetes environment.

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA RTX 5060 Ti |
| GPU memory | 16 GB |
| Deployment | Kubernetes pod |
| Namespace | triton-dev |
| Inference engine | TensorRT-LLM |
| Model | Qwen2.5-1.5B-Instruct |
| Model identifier | Qwen/Qwen2.5-1.5B-Instruct |
| Serving port | 8000 |
| API | OpenAI-compatible REST API |

Verify the available pod:

bash

```bash
kubectl get pods -n triton-dev
```

Check GPU visibility:

bash

```bash
kubectl exec -it <pod-name> -n triton-dev -- nvidia-smi
```

## 3\. TensorRT-LLM Server Configuration

Configuration file:

text

```text
/opt/triton_models/config.yml
```

The initial configuration used for this experiment:

yaml

```yaml
enable_iter_perf_stats: true
```

**Configuration note:** Validate these options against the installed TensorRT-LLM version before benchmarking. TensorRT-LLM configuration fields and serving behavior can differ from vLLM. The values are intended to establish comparable workload and resource boundaries, not to imply identical internal scheduling implementations.

## 4\. Start TensorRT-LLM Server

Start the server inside the Kubernetes pod:

bash

```bash
kubectl exec -it <pod-name> -n triton-dev -- \
  trtllm-serve serve Qwen/Qwen2.5-1.5B-Instruct \
  --config /opt/triton_models/config.yml \
  --max_seq_len 2048   --max_batch_size 8
```

Keep the server process running.

Wait for the server startup and model initialization to complete.

Check server logs for errors, model loading completion, and the listening address.

## 5\. Expose the API to the Benchmark Client

If the benchmark client runs on the host rather than inside the Kubernetes pod, forward port 8000.

Open another terminal:

bash

```bash
kubectl port-forward -n triton-dev pod/<pod-name> 8000:8000
```

Keep port forwarding active during the benchmark.

The benchmark endpoint will be:

text

```text
http://localhost:8000
```

If port 8000 is already occupied, use another local port and update the benchmark URL accordingly.

## 6\. Verify API Availability

Check model availability:

bash

```bash
curl http://localhost:8000/v1/models
```

Check server health:

bash

```bash
curl http://localhost:8000/health
```

Check server version:

bash

```bash
curl http://localhost:8000/version
```

Send a test inference request:

bash

```bash
curl http://localhost:8000/v1/chat/completions \
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

To focus on steady-state inference performance, execute a preliminary inference request before collecting benchmark results.

The warm-up request is not included in the measured benchmark.

Apply the same warm-up policy to both vLLM and TensorRT-LLM.

Allow the server to complete initialization before proceeding.

## 8\. Run Serving Benchmark

Use the same benchmark client and workload as EXP-001.

Run from the host benchmark environment:

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

The benchmark client must be able to access the TensorRT-LLM OpenAI-compatible endpoint.
