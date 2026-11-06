# EXP-001: vLLM Baseline — Qwen2.5-1.5B-Instruct

Experiment ID: EXP-001 | Status: Completed | Engine: vLLM

## 1. Objective

Establish an initial performance baseline for serving Qwen2.5-1.5B-Instruct using vLLM on an NVIDIA RTX 5060 Ti 16GB GPU.

This experiment evaluates LLM serving performance under a controlled synthetic workload, measuring:

- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- Inter-token Latency (ITL)
- Request throughput
- Aggregate output token throughput
- Request concurrency
- Request success rate

The results will serve as the reference baseline for subsequent TensorRT-LLM experiments.

## 2. Experimental Environment

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| GPU memory | 16 GB |
| Operating system | Linux |
| Python | 3.12 |
| Inference engine | vLLM |
| Model | Qwen2.5-1.5B-Instruct |
| Model identifier | Qwen/Qwen2.5-1.5B-Instruct |
| Serving interface | OpenAI-compatible API |
| Server port | 8000 |


## 3. Server Configuration

The model was served using the following configuration:

bash

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --port 8000
```

### Configuration parameters

| Parameter | Value |
| --- | --- |
| Maximum model length | 2048 |
| Maximum running sequences | 8 |
| GPU memory utilization target | 90% |
| Attention backend | TRITON\_ATTN |
| Multimodal encoder backend | TRITON\_ATTN |
| API port | 8000 |

The selected model is text-only. The multimodal encoder attention option was retained from the tested server command but is not required for this model.

## 4. Benchmark Workload

The benchmark was executed using the vLLM serving benchmark client with the OpenAI-compatible backend.

### Benchmark command

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
    --label results_vllm_qwen
```

### Workload parameters

| Parameter | Value |
| --- | --- |
| Dataset | Random synthetic |
| Input sequence length | 1024 tokens |
| Output sequence length | 256 tokens |
| Total requests | 100 |
| Configured request rate | 5 requests/sec |
| Backend | OpenAI-compatible |
| Target endpoint | localhost:8000 |

The synthetic dataset provides controlled input and output sequence lengths, allowing repeatable performance measurements.

## 5. Benchmark Results

\<badge color="success"\>100% request success\</badge\>

The benchmark completed successfully with zero failed requests.

### 5.1 Execution summary

| Metric | Measured result |
| --- | --- |
| Successful requests | 100 |
| Failed requests | 0 |
| Success rate | 100% |
| Benchmark duration | 38.09 seconds |
| Total input tokens | 102,400 |
| Total generated tokens | 25,600 |
| Configured request rate | 5.00 RPS |
| Actual request throughput | 2.63 req/sec |
| Output token throughput | 672.04 tok/sec |
| Peak output token throughput | 848.00 tok/sec |
| Peak concurrent requests | 52 |

### 5.2 Time to First Token (TTFT)

TTFT measures the time between request submission and receiving the first generated token.

| Statistic | Latency |
| --- | --- |
| Mean TTFT | 6353.70 ms |
| Median TTFT | 6049.79 ms |
| P99 TTFT | 15211.01 ms |

The mean TTFT was approximately 6.35 seconds, with P99 reaching approximately 15.21 seconds.

This indicates significant first-token latency under the tested workload. Request queueing, scheduling, and prompt processing are possible contributors.

### 5.3 Time Per Output Token (TPOT)

TPOT measures the average time between generated output tokens, excluding the first token.

| Statistic | Latency |
| --- | --- |
| Mean TPOT | 11.00 ms |
| Median TPOT | 11.08 ms |
| P99 TPOT | 11.13 ms |

Approximate per-request decode speed:

text

```text
1000 / 11 ms = 90.9 tokens/sec
```

This is an approximate decode speed derived from mean TPOT, not aggregate server throughput.

### 5.4 Inter-token Latency (ITL)

| Statistic | Latency |
| --- | --- |
| Mean ITL | 11.00 ms |
| Median ITL | 9.42 ms |
| P99 ITL | 67.22 ms |

ITL measures the interval between consecutive generated tokens. The P99 result indicates occasional longer token-generation intervals.

## 6. Performance Analysis

### Aggregate Throughput

The system achieved:

**672.04 output tokens/sec**

This represents aggregate output generation across concurrent benchmark requests, rather than the generation speed of an individual request.

The measured request throughput was 2.63 requests/sec, below the configured arrival rate of 5 requests/sec.

### First-Token Latency

Mean TTFT of 6.35 seconds is a significant performance consideration.

Although decode performance is relatively consistent, first-token latency requires further investigation.

Future tests should examine the relationship between:

- Request arrival rate.
- Scheduler queueing.
- Prompt processing time.
- Running sequence limits.
- GPU utilization.

### Concurrency

The benchmark reported peak concurrent requests of 52.

This is a benchmark-side concurrency measurement and should not be confused with the vLLM scheduler's maximum running sequence limit of 8. Requests can remain in flight while waiting for execution.

## 7. Experimental Conclusion

EXP-001 successfully establishes the initial vLLM serving baseline for Qwen2.5-1.5B-Instruct.

The system processed 100 requests with zero failures and achieved an aggregate output throughput of 672.04 tokens/sec.

The experiment demonstrates that the configured vLLM server can successfully serve the selected model and sustain the synthetic benchmark workload.

The primary observations are:

1. Successful completion of all benchmark requests.
2. Aggregate output throughput of 672.04 tokens/sec.
3. Mean TPOT of 11 ms.
4. Mean TTFT of 6.35 seconds.
5. P99 TTFT of 15.21 seconds.

These measurements provide the initial reference for evaluating TensorRT-LLM under comparable conditions.

Here is the consolidated table for all four runs, with percentage change calculated from Run 2 to Run 4.

### EXP-001: vLLM baseline results

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Change R2 → R4 |
| --- | --- | --- | --- | --- | --- |
| Duration (s) | 38.09 | 32.71 | 31.90 | 31.84 | -2.7% |
| Request throughput | 2.63 | 3.06 | 3.13 | 3.14 | +2.6% |
| Output throughput (tok/s) | 672.04 | 782.63 | 802.54 | 804.01 | +2.7% |
| Peak output throughput | 848 | 928 | 871 | 867 | -6.6% |
| Total token throughput | 3360.22 | 3913.13 | 4012.71 | 4020.06 | +2.7% |
| Peak concurrency | 52 | 45 | 44 | 44 | -2.2% |
| Mean TTFT (ms) | 6353.70 | 4387.68 | 3650.77 | 3630.44 | -17.3% |
| Median TTFT (ms) | 6049.79 | 3574.65 | 3263.71 | 3237.25 | -9.4% |
| P99 TTFT (ms) | 15211.01 | 10815.01 | 9608.21 | 9576.80 | -11.5% |
| Mean TPOT (ms) | 11.00 | 9.24 | 9.30 | 9.29 | +0.5% |
| Median TPOT (ms) | 11.08 | 9.30 | 9.31 | 9.30 | 0.0% |
| P99 TPOT (ms) | 11.13 | 9.50 | 9.43 | 9.42 | -0.8% |
| Mean ITL (ms) | 11.00 | 9.24 | 9.30 | 9.29 | +0.5% |
| Median ITL (ms) | 9.42 | 9.32 | 9.22 | 9.22 | -1.1% |
| P99 ITL (ms) | 67.22 | 9.96 | 19.33 | 19.00 | +90.8% |

Positive changes mean increases; negative changes mean decreases. For latency, decreases are improvements. Run 2 → Run 4 is calculated as (Run 4 − Run 2) / Run 2.

### Summary

Four vLLM benchmark executions were completed successfully, each processing 100 requests with zero failures under identical configurations. Run 1 showed lower throughput and higher latency than subsequent executions. Runs 2–4 demonstrated consistent steady-state performance, averaging **796.39 output tokens/sec**, **3.11 requests/sec**, and **9.28 ms mean TPOT**. Between Run 2 and Run 4, aggregate output throughput improved by 2.7%, while mean TTFT decreased by 17.3%. These results establish the preliminary vLLM steady-state baseline for comparison against TensorRT-LLM.

One observation worth retaining: P99 ITL increased from 9.96 ms to 19.00 ms between Run 2 and Run 4, despite stable average decode performance. This shows why tail latency should remain part of the engine comparison.