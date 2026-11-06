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

## 9\. Repeat Benchmark Runs

Execute three measured runs using identical configurations.

| Run | Label | Status |
| --- | --- | --- |
| Run 1 | results\_trtllm\_qwen\_run01 | Pending |
| Run 2 | results\_trtllm\_qwen\_run02 | Pending |
| Run 3 | results\_trtllm\_qwen\_run03 | Pending |

Use the following label for each run:

bash

```bash
--label results_trtllm_qwen_run01
```

Change only the label between runs.

Do not change server or benchmark parameters during the experiment.

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

TensorRT-LLM also exposes runtime metrics:

bash

```bash
curl http://localhost:8000/metrics
```

These measurements provide additional context when interpreting benchmark results.

## 11\. Result Preservation

Store the original benchmark outputs independently.

text

```text
results/
└── EXP-002/
    ├── run-01-trtllm.log
    ├── run-02-trtllm.log
    └── run-03-trtllm.log
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

Record the TensorRT-LLM version, container image tag, CUDA version, model precision, and server configuration.

## 12\. Experimental Controls

For comparison with EXP-001:

- Use the same Qwen2.5-1.5B-Instruct model revision.
- Match model precision.
- Use identical input and output lengths.
- Use the same request count and request rate.
- Apply the same warm-up procedure.
- Run three measured executions.
- Avoid competing workloads on the GPU.
- Preserve the original benchmark output.

The two inference engines do not need identical internal configurations, but their effective workload and hardware constraints must be comparable.

## 13\. Expected Deliverables

After completing EXP-002:

| Deliverable | Output |
| --- | --- |
| TensorRT-LLM baseline | 3 successful benchmark runs |
| Raw benchmark logs | EXP-002 results |
| Performance summary | Mean and run-to-run range |
| GPU observations | Utilization and VRAM |
| Experiment report | reports/02-trtllm-baseline.md |

The final objective is to compare EXP-002 against EXP-001 and produce a reproducible vLLM versus TensorRT-LLM performance evaluation.

---

**Experiment status:** Preparation

**Next milestone:** Successfully deploy TensorRT-LLM, verify its API, and complete the first benchmark run.

## Two important technical notes

1. Kubernetes networking: If the benchmark runs on your host, remember that `localhost:8000` needs the `kubectl port-forward` command. The pod's own localhost is not the host's localhost.

2. Benchmark fairness: NVIDIA documents `trtllm-serve` as OpenAI-compatible and supports external benchmark clients. Reusing `vllm bench serve` is reasonable for a controlled comparison.

TensorRT-LLM

+1

Official references:

- [TensorRT-LLM trtllm-serve documentation](<https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html?utm_source=chatgpt.com>)
- [NVIDIA: Benchmarking trtllm-serve](<https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/commands/trtllm-serve/run-benchmark-with-trtllm-serve.md?utm_source=chatgpt.com>)

One thing I want to verify before you execute: the TensorRT-LLM YAML configuration. Your existing configuration uses vLLM-like resource settings. We should validate it against your installed TensorRT-LLM version rather than assume all those keys are accepted.