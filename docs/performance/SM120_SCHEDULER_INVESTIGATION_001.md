# SM120 vLLM Performance Investigation

## Investigation 01 — Scheduler Concurrency (`max-num-seqs`)

**Date:** November 7, 2026\
**Platform:** NVIDIA GeForce RTX 5060 Ti (SM120)\
**Framework:** Custom SM120 vLLM build\
**Model:** Qwen/Qwen2.5-1.5B-Instruct\
**Status:** Investigation completed; 48 sequences selected as the best measured configuration.

---

# 1\. Executive Summary

This investigation evaluated vLLM scheduler concurrency using the `--max-num-seqs` parameter on an NVIDIA RTX 5060 Ti (SM120).

The objective was to determine whether increasing the maximum number of scheduled sequences could improve inference throughput and request latency.

We started with the existing configuration of 8 sequences and progressively tested:

- 8 sequences
- 16 sequences
- 24 sequences
- 32 sequences
- 48 sequences
- 64 sequences

All six tests used the same model, input/output token lengths, request count, configured request rate, attention backend, and GPU memory utilization.

We also conducted an initial experiment changing `max-num-batched-tokens` from 2048 to 4096 to determine whether the scheduler token budget was limiting performance.

## Key achievements

| Measurement | Initial baseline (8) | Best result (48) | Improvement |
| --- | --- | --- | --- |
| Output throughput | 699.46 tok/s | 1,207.92 tok/s | +72.7% |
| Request throughput | 2.73 req/s | 4.72 req/s | +72.9% |
| Mean TTFT | 6,491.50 ms | 129.42 ms | -98.0% |
| Median TTFT | 6,114.32 ms | 93.82 ms | -98.5% |
| Mean TPOT | 10.53 ms | 13.74 ms | +30.5% |
| Failed requests | 0 | 0 | No change |

**Main conclusion:**

`max-num-seqs=48` delivered the highest sustained throughput among the configurations tested.

Increasing the limit to 64 did not improve performance. Throughput decreased by approximately 6.7%.

The investigation also established that the GPU reaches approximately 99% utilization during active benchmarking and that VRAM usage approaches the available capacity.

The result is a measured improvement of approximately 72.7% in sustained output throughput compared with the initial 8-sequence configuration.

---

# 2\. Investigation Objectives

The investigation focused on five questions:

1. Was the original sequence limit of 8 restricting scheduler throughput?
2. Would increasing sequence capacity improve GPU batching?
3. At what sequence count would throughput stop improving?
4. What impact would increased concurrency have on TTFT and TPOT?
5. Was the GPU becoming saturated, or was scheduler capacity still limiting performance?

The investigation followed a controlled testing approach:

- Change one parameter at a time.
- Keep the model and benchmark workload constant.
- Record every benchmark result.
- Compare results against the original baseline.
- Monitor GPU utilization and VRAM.
- Stop increasing concurrency when results showed diminishing returns or regression.

---

# 3\. Hardware and Software Environment

## 3.1 Hardware

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| GPU architecture | Blackwell |
| Compute capability | 12.0 |
| SM architecture | SM120 |
| Reported GPU memory | 16,311 MiB |
| GPU power limit | 180 W |
| Tested GPU clock | Approximately 2,750–2,800 MHz |

## 3.2 Software

| Component | Configuration |
| --- | --- |
| Framework | Custom SM120 vLLM build |
| Python environment | Existing project virtual environment |
| PyTorch | 2.14.0 |
| CUDA | 13.0 |
| Attention backend | TRITON\_ATTN |
| Model | Qwen/Qwen2.5-1.5B-Instruct |

## 3.3 Model and benchmark workload

All scheduler experiments used the following workload.

| Parameter | Value |
| --- | --- |
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Dataset | Random |
| Input length | 1,024 tokens |
| Output length | 256 tokens |
| Number of prompts | 100 |
| Configured request rate | 5 RPS |
| Burstiness | 1.0 (Poisson process) |
| Maximum request concurrency | Unrestricted by benchmark client |
| Backend | OpenAI-compatible API |
| Server port | 8000 |

## 3.4 Standard benchmark command

The same command was used for every sequence experiment, with only the label changed.

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
    --label sm120_maxseq_TEST
```

---

# 4\. Initial Scheduler Configuration

The original server configuration was:

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

The initial scheduler configuration was therefore:

| Parameter | Value |
| --- | --- |
| max-model-len | 2048 |
| max-num-seqs | 8 |
| max-num-batched-tokens | 2048 (default) |
| gpu-memory-utilization | 0.90 |
| attention-backend | TRITON\_ATTN |
| chunked prefill | Enabled by default |
| scheduling policy | FCFS default |

The source inspection confirmed that the scheduler default was:

python

```python
DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
```

Source file:

text

```text
vllm/config/scheduler.py
```

---

# 5\. Scheduler Source Investigation

Before changing scheduler settings, we inspected the actual configuration class in the SM120 checkout.

## 5.1 SchedulerConfig inspection

Command:

bash

```bash
python - <<'PY'
from vllm.config import SchedulerConfig
import inspect

print("SchedulerConfig signature:")
print(inspect.signature(SchedulerConfig))

print("\nSchedulerConfig source location:")
print(inspect.getfile(SchedulerConfig))

print("\nSchedulerConfig attributes:")
print([
    x for x in dir(SchedulerConfig)
    if not x.startswith("_")
])
PY
```

The configuration signature confirmed support for:

- max\_num\_batched\_tokens
- max\_num\_scheduled\_tokens
- max\_num\_seqs
- max\_num\_active\_seqs
- enable\_chunked\_prefill
- policy
- async\_scheduling
- prefill\_schedule\_interval
- max\_num\_queued\_reqs

The initial inspection attempted to use `model_fields`, but the class did not expose that interface. We switched to Python `inspect.signature()` and successfully identified the supported parameters.

## 5.2 Scheduler validation inspection

Command:

bash

```bash
sed -n '300,360p' vllm/config/scheduler.py
```

Important findings:

- Chunked prefill was enabled by default.
- The scheduler validates the token budget against the maximum sequence count.
- `max_num_batched_tokens` must be greater than or equal to `max_num_seqs`.
- The scheduler warns if the token budget exceeds `max_num_seqs * max_model_len`.

This established that testing 2048, 4096, and 8192 token budgets was structurally possible with the selected model length and sequence settings, subject to runtime resource constraints.

---

# 6\. Experiment 00 — Scheduler Token Budget

Before investigating sequence concurrency, we tested whether increasing the scheduler token budget would improve performance.

## 6.1 Test configurations

| Parameter | Baseline | Test A |
| --- | --- | --- |
| max-model-len | 2048 | 2048 |
| max-num-seqs | 8 | 8 |
| max-num-batched-tokens | 2048 | 4096 |
| gpu-memory-utilization | 0.90 | 0.90 |
| Attention backend | TRITON\_ATTN | TRITON\_ATTN |

Only `max-num-batched-tokens` changed.

## 6.2 Results

| Metric | 2048 baseline | 4096 test |
| --- | --- | --- |
| Benchmark duration | 36.60 s | 36.63 s |
| Request throughput | 2.73 req/s | 2.73 req/s |
| Output throughput | 699.46 tok/s | 698.90 tok/s |
| Peak output throughput | 936 tok/s | 888 tok/s |
| Total token throughput | 3,497.32 tok/s | 3,494.51 tok/s |
| Peak concurrency | 53 | 54 |
| Mean TTFT | 6,491.50 ms | 6,268.36 ms |
| Median TTFT | 6,114.32 ms | 5,613.65 ms |
| P99 TTFT | 15,373.40 ms | 15,347.88 ms |
| Mean TPOT | 10.53 ms | 10.54 ms |
| Median TPOT | 10.70 ms | 10.64 ms |
| P99 TPOT | 10.93 ms | 11.05 ms |
| Failed requests | 0 | 0 |

## 6.3 Conclusion

Increasing the token budget from 2048 to 4096 did not produce a meaningful throughput improvement.

Sustained output throughput remained practically identical.

**Decision:** Retain `max-num-batched-tokens=2048` for subsequent experiments.

This allowed the next investigation to isolate the effect of sequence capacity.

---

# 7\. Experiment 01 — max-num-seqs = 8

## Configuration

bash

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --port 8000
```

## Results

| Metric | Result |
| --- | --- |
| Benchmark duration | 36.60 s |
| Successful requests | 100 |
| Failed requests | 0 |
| Request throughput | 2.73 req/s |
| Output throughput | 699.46 tok/s |
| Peak output throughput | 936 tok/s |
| Total token throughput | 3,497.32 tok/s |
| Peak concurrency | 53 |
| Mean TTFT | 6,491.50 ms |
| Median TTFT | 6,114.32 ms |
| P99 TTFT | 15,373.40 ms |
| Mean TPOT | 10.53 ms |
| Median TPOT | 10.70 ms |
| P99 TPOT | 10.93 ms |

**Conclusion:** This became the reference baseline for the sequence-concurrency experiments.

---

# 8\. Experiment 02 — max-num-seqs = 16

## Configuration

Only the sequence limit changed from 8 to 16.

bash

```bash
--max-num-seqs 16
```

All other settings remained unchanged.

## Results

| Metric | Result |
| --- | --- |
| Benchmark duration | 25.03 s |
| Successful requests | 100 |
| Failed requests | 0 |
| Request throughput | 4.00 req/s |
| Output throughput | 1,022.90 tok/s |
| Peak output throughput | 1,357 tok/s |
| Total token throughput | 5,114.52 tok/s |
| Peak concurrency | 33 |
| Mean TTFT | 930.77 ms |
| Median TTFT | 663.66 ms |
| P99 TTFT | 2,734.06 ms |
| Mean TPOT | 12.47 ms |
| Median TPOT | 13.09 ms |
| P99 TPOT | 13.31 ms |

## Comparison against 8 sequences

| Metric | 8 sequences | 16 sequences | Change |
| --- | --- | --- | --- |
| Output throughput | 699.46 | 1,022.90 | +46.3% |
| Request throughput | 2.73 | 4.00 | +46.5% |
| Mean TTFT | 6,491.50 ms | 930.77 ms | -85.7% |
| Median TTFT | 6,114.32 ms | 663.66 ms | -89.1% |
| Mean TPOT | 10.53 ms | 12.47 ms | +18.4% |
| P99 TPOT | 10.93 ms | 13.31 ms | +21.8% |

**Conclusion:** A substantial throughput and TTFT improvement was observed.

The increased sequence capacity allowed more concurrent work to be scheduled.

Decision: Continue testing higher sequence counts.

---

# 9\. Experiment 03 — max-num-seqs = 24

## Configuration

Only the sequence limit changed.

bash

```bash
--max-num-seqs 24
```

## Results

| Metric | Result |
| --- | --- |
| Benchmark duration | 23.32 s |
| Successful requests | 100 |
| Failed requests | 0 |
| Request throughput | 4.29 req/s |
| Output throughput | 1,097.69 tok/s |
| Peak output throughput | 1,881 tok/s |
| Total token throughput | 5,488.45 tok/s |
| Peak concurrency | 31 |
| Mean TTFT | 205.56 ms |
| Median TTFT | 98.16 ms |
| P99 TTFT | 1,028.31 ms |
| Mean TPOT | 13.74 ms |
| Median TPOT | 14.42 ms |
| P99 TPOT | 16.18 ms |

## Comparison against 16 sequences

| Metric | 16 sequences | 24 sequences | Change |
| --- | --- | --- | --- |
| Output throughput | 1,022.90 | 1,097.69 | +7.3% |
| Request throughput | 4.00 | 4.29 | +7.3% |
| Mean TTFT | 930.77 ms | 205.56 ms | -77.9% |
| Median TTFT | 663.66 ms | 98.16 ms | -85.2% |
| Mean TPOT | 12.47 ms | 13.74 ms | +10.2% |
| P99 TPOT | 13.31 ms | 16.18 ms | +21.6% |

## Conclusion

The 24-sequence configuration improved sustained throughput further.

Mean and median TTFT dropped substantially.

However, TPOT increased, indicating a trade-off between aggregate throughput and individual generation latency.

Decision: Continue testing.

---

# 10\. Experiment 04 — max-num-seqs = 32

## Configuration

Only the sequence limit changed.

bash

```bash
--max-num-seqs 32
```

## Results

| Metric | Result |
| --- | --- |
| Benchmark duration | 22.63 s |
| Successful requests | 100 |
| Failed requests | 0 |
| Request throughput | 4.42 req/s |
| Output throughput | 1,131.11 tok/s |
| Peak output throughput | 2,384 tok/s |
| Total token throughput | 5,655.53 tok/s |
| Peak concurrency | 31 |
| Mean TTFT | 128.10 ms |
| Median TTFT | 91.94 ms |
| P99 TTFT | 372.74 ms |
| Mean TPOT | 13.55 ms |
| Median TPOT | 13.87 ms |
| P99 TPOT | 16.47 ms |

## Comparison against 24 sequences

| Metric | 24 sequences | 32 sequences | Change |
| --- | --- | --- | --- |
| Output throughput | 1,097.69 | 1,131.11 | +3.0% |
| Request throughput | 4.29 | 4.42 | +3.0% |
| Mean TTFT | 205.56 ms | 128.10 ms | -37.7% |
| Median TTFT | 98.16 ms | 91.94 ms | -6.3% |
| Mean TPOT | 13.74 ms | 13.55 ms | -1.4% |
| P99 TPOT | 16.18 ms | 16.47 ms | +1.8% |

## Conclusion

The 32-sequence configuration improved throughput slightly.

TTFT improved, while mean TPOT remained practically unchanged.

Decision: Continue testing higher concurrency.

---

# 11\. Experiment 05 — max-num-seqs = 48

## Configuration

Only the sequence limit changed.

bash

```bash
--max-num-seqs 48
```

## Results

| Metric | Result |
| --- | --- |
| Benchmark duration | 21.19 s |
| Successful requests | 100 |
| Failed requests | 0 |
| Request throughput | 4.72 req/s |
| Output throughput | 1,207.92 tok/s |
| Peak output throughput | 1,977 tok/s |
| Total token throughput | 6,039.59 tok/s |
| Peak concurrency | 30 |
| Mean TTFT | 129.42 ms |
| Median TTFT | 93.82 ms |
| P99 TTFT | 383.64 ms |
| Mean TPOT | 13.74 ms |
| Median TPOT | 14.11 ms |
| P99 TPOT | 16.16 ms |

## Comparison against 32 sequences

| Metric | 32 sequences | 48 sequences | Change |
| --- | --- | --- | --- |
| Output throughput | 1,131.11 | 1,207.92 | +6.8% |
| Request throughput | 4.42 | 4.72 | +6.8% |
| Mean TTFT | 128.10 ms | 129.42 ms | +1.0% |
| Median TTFT | 91.94 ms | 93.82 ms | +2.0% |
| Mean TPOT | 13.55 ms | 13.74 ms | +1.4% |
| P99 TPOT | 16.47 ms | 16.16 ms | -1.9% |

## Conclusion

The 48-sequence configuration produced the highest sustained throughput so far.

Compared with 32:

- Sustained output throughput improved 6.8%.
- Mean TTFT remained practically unchanged.
- Mean TPOT remained practically unchanged.
- P99 TPOT improved slightly.

Decision: Test 64 to establish whether further increases provide benefits.

---

# 12\. Experiment 06 — max-num-seqs = 64

## Configuration

Only the sequence limit changed.

bash

```bash
--max-num-seqs 64
```

## Results

| Metric | Result |
| --- | --- |
| Benchmark duration | 22.72 s |
| Successful requests | 100 |
| Failed requests | 0 |
| Request throughput | 4.40 req/s |
| Output throughput | 1,126.88 tok/s |
| Peak output throughput | 2,416 tok/s |
| Total token throughput | 5,634.41 tok/s |
| Peak concurrency | 32 |
| Mean TTFT | 130.57 ms |
| Median TTFT | 95.75 ms |
| P99 TTFT | 363.10 ms |
| Mean TPOT | 13.88 ms |
| Median TPOT | 14.61 ms |
| P99 TPOT | 17.07 ms |

## Comparison against 48 sequences

| Metric | 48 sequences | 64 sequences | Change |
| --- | --- | --- | --- |
| Output throughput | 1,207.92 | 1,126.88 | -6.7% |
| Request throughput | 4.72 | 4.40 | -6.8% |
| Mean TTFT | 129.42 ms | 130.57 ms | +0.9% |
| Median TTFT | 93.82 ms | 95.75 ms | +2.1% |
| Mean TPOT | 13.74 ms | 13.88 ms | +1.0% |
| P99 TPOT | 16.16 ms | 17.07 ms | +5.6% |

## Conclusion

Increasing sequence capacity to 64 did not improve sustained throughput.

Output throughput decreased by approximately 6.7%.

TTFT remained similar, but TPOT became slightly worse.

Decision: Do not select 64 as the preferred configuration.

The 48-sequence configuration remains the best measured result.

---

# 13\. Consolidated Results — All Sequence Experiments

This table is the primary reference for future performance investigations.

All values are from the reported benchmark outputs.

| Metric | Seq 8 | Seq 16 | Seq 24 | Seq 32 | Seq 48 | Seq 64 |
| --- | --- | --- | --- | --- | --- | --- |
| Duration (s) | 36.60 | 25.03 | 23.32 | 22.63 | 21.19 | 22.72 |
| Request throughput | 2.73 | 4.00 | 4.29 | 4.42 | **4.72** | 4.40 |
| Output tok/s | 699.46 | 1022.90 | 1097.69 | 1131.11 | **1207.92** | 1126.88 |
| Peak output tok/s | 936 | 1357 | 1881 | 2384 | 1977 | 2416 |
| Total tok/s | 3497.32 | 5114.52 | 5488.45 | 5655.53 | **6039.59** | 5634.41 |
| Peak concurrency | 53 | 33 | 31 | 31 | 30 | 32 |
| Mean TTFT (ms) | 6491.50 | 930.77 | 205.56 | **128.10** | 129.42 | 130.57 |
| Median TTFT (ms) | 6114.32 | 663.66 | 98.16 | **91.94** | 93.82 | 95.75 |
| P99 TTFT (ms) | 15373.40 | 2734.06 | 1028.31 | 372.74 | 383.64 | **363.10** |
| Mean TPOT (ms) | **10.53** | 12.47 | 13.74 | 13.55 | 13.74 | 13.88 |
| Median TPOT (ms) | **10.70** | 13.09 | 14.42 | 13.87 | 14.11 | 14.61 |
| P99 TPOT (ms) | **10.93** | 13.31 | 16.18 | 16.47 | **16.16** | 17.07 |
| Failed requests | 0 | 0 | 0 | 0 | 0 | 0 |

## 13.1 Sustained throughput progression

| Sequence limit | Output throughput | Improvement vs previous test |
| --- | --- | --- |
| 8 | 699.46 | Baseline |
| 16 | 1,022.90 | +46.3% |
| 24 | 1,097.69 | +7.3% |
| 32 | 1,131.11 | +3.0% |
| 48 | 1,207.92 | +6.8% |
| 64 | 1,126.88 | -6.7% |

The results demonstrate a substantial initial improvement from increasing sequence capacity, followed by smaller gains and eventual regression.

Note that the 32-to-48 improvement is larger than the 24-to-32 improvement, so the progression is not strictly diminishing at every step.

## 13.2 Throughput visualization

text

```text
Output throughput (tokens/sec)

8 sequences    699 | ██████████████
16 sequences  1023 | ████████████████████
24 sequences  1098 | ██████████████████████
32 sequences  1131 | ███████████████████████
48 sequences  1208 | ████████████████████████
64 sequences  1127 | ███████████████████████
```

---

# 14\. GPU Monitoring

GPU monitoring was performed during the 48- and 64-sequence experiments using NVIDIA SMI.

## 14.1 Monitoring command

bash

```bash
nvidia-smi \
  --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu \
  --format=csv \
  -l 1
```

The captured output also included GPU clock readings.

## 14.2 48-sequence observations

| Metric | Observation |
| --- | --- |
| Active GPU utilization | Approximately 99% |
| VRAM allocation | 15,710 MiB |
| Reported GPU memory | 16,311 MiB |
| Remaining memory | Approximately 601 MiB |
| GPU clock | Approximately 2,752–2,805 MHz |
| Active temperature | Approximately 48–61°C |
| Power consumption | Approximately 120–161 W |

## 14.3 64-sequence observations

| Metric | Observation |
| --- | --- |
| Active GPU utilization | Approximately 99% |
| VRAM allocation | 15,704 MiB |
| Reported GPU memory | 16,311 MiB |
| Remaining memory | Approximately 607 MiB |
| GPU clock | Approximately 2,760–2,805 MHz |
| Active temperature | Approximately 46–62°C |
| Power consumption | Approximately 117–167 W |

## 14.4 GPU monitoring conclusions

Three observations are important:

1. GPU utilization reaches approximately 99% during active inference.
2. VRAM allocation approaches the reported available capacity.
3. GPU clocks remain high during the active workload.

This suggests the GPU is heavily utilized under the tested workload.

However, GPU utilization alone does not establish whether the limiting factor is compute throughput, memory bandwidth, kernel execution overhead, or another resource.

Further profiling is required to distinguish these possibilities.

**Important:** The available GPU memory headroom is only approximately 600 MiB during these tests. Avoid increasing sequence capacity without first evaluating memory requirements and startup/runtime stability.

---

# 15\. Important Benchmark Interpretation

## 15.1 Configured request rate vs achieved request throughput

The benchmark was configured at 5 requests per second.

Achieved request throughput varied:

| Sequences | Configured RPS | Achieved RPS |
| --- | --- | --- |
| 8 | 5 | 2.73 |
| 16 | 5 | 4.00 |
| 24 | 5 | 4.29 |
| 32 | 5 | 4.42 |
| 48 | 5 | 4.72 |
| 64 | 5 | 4.40 |

The configured request rate is the client-side arrival rate.

The achieved request throughput is the actual completed request rate over the benchmark duration.

They should not be treated as interchangeable.

## 15.2 Peak concurrency

The reported peak concurrency is a benchmark-client measurement.

It is not a direct measurement of the number of sequences executing inside the vLLM scheduler or GPU.

The scheduler's `max-num-seqs` setting is a server-side scheduling limit.

The two metrics should not be interpreted as equivalent.

## 15.3 TPOT trade-off

The 8-sequence configuration achieved the lowest TPOT.

As sequence concurrency increased, aggregate throughput improved but TPOT generally increased.

This indicates a trade-off between individual sequence decode latency and aggregate serving throughput.

For a throughput-oriented serving workload, the 48-sequence result is attractive.

For a latency-sensitive workload, additional tests with different request rates would be necessary.

---

# 16\. Final Findings and Decisions

## Finding 1 — Initial sequence limit was restrictive

The original 8-sequence configuration delivered 699.46 output tokens/sec.

Increasing the sequence limit to 16 improved throughput by 46.3%.

This was the largest single improvement in the investigation.

## Finding 2 — Further concurrency improved aggregate throughput

Throughput continued improving through 24, 32 and 48 sequences.

The highest sustained throughput was achieved at 48 sequences.

## Finding 3 — Increasing to 64 was counterproductive

The 64-sequence test produced lower throughput than the 48-sequence test.

There is no evidence from these tests that increasing sequence capacity further would be beneficial.

## Finding 4 — GPU utilization is high

Both monitored configurations reached approximately 99% GPU utilization.

This suggests the next optimization phase should focus on execution efficiency rather than simply increasing scheduler concurrency.

## Finding 5 — VRAM headroom is limited

The monitored configurations used approximately 15.7 GiB of the reported 16.3 GiB.

Future experiments must account for memory overhead.

---

# 17\. Selected Configuration

**Selected experimental configuration:****`max-num-seqs=48`****.**

bash

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 2048 \
    --max-num-seqs 48 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend TRITON_ATTN \
    --port 8000
```

## Selected configuration performance

| Metric | Result |
| --- | --- |
| Output throughput | 1,207.92 tok/s |
| Request throughput | 4.72 req/s |
| Mean TTFT | 129.42 ms |
| Median TTFT | 93.82 ms |
| P99 TTFT | 383.64 ms |
| Mean TPOT | 13.74 ms |
| P99 TPOT | 16.16 ms |
| Failed requests | 0 |

This configuration is the current benchmark winner, not yet a production-certified configuration.

---

# 18\. Next Investigation

The scheduler investigation is complete for now.

The next proposed investigation is:

**Investigation 02 — CUDA Graph Configuration**

Objectives:

1. Identify CUDA Graph settings supported by the custom SM120 build.
2. Establish the current graph capture behavior.
3. Determine whether graph execution can reduce launch overhead.
4. Measure graph capture memory overhead.
5. Compare throughput, TTFT and TPOT.
6. Retain the selected scheduler configuration unless an experiment specifically requires changing it.

The selected scheduler configuration should be held constant during CUDA Graph experiments:

bash

```bash
--max-num-seqs 48
--max-num-batched-tokens 2048
--max-model-len 2048
```

---

# 19\. Reproducibility and Traceability

## 19.1 Experiment sequence

| Order | Experiment | Changed parameter | Result |
| --- | --- | --- | --- |
| 00 | Token budget | 2048 → 4096 | No meaningful improvement |
| 01 | Baseline | max-num-seqs = 8 | Reference |
| 02 | Sequence test | 8 → 16 | +46.3% throughput |
| 03 | Sequence test | 16 → 24 | +7.3% throughput |
| 04 | Sequence test | 24 → 32 | +3.0% throughput |
| 05 | Sequence test | 32 → 48 | +6.8% throughput |
| 06 | Sequence test | 48 → 64 | -6.7% throughput |

## 19.2 Test consistency

The following parameters were held constant throughout the sequence experiments:

- Model: Qwen/Qwen2.5-1.5B-Instruct
- Input tokens: 1024
- Output tokens: 256
- Number of prompts: 100
- Configured request rate: 5 RPS
- GPU memory utilization: 0.90
- Attention backend: TRITON\_ATTN
- Maximum model length: 2048
- Maximum batched tokens: 2048

Only `max-num-seqs` changed during the six sequence experiments.

## 19.3 Experimental limitations

The following limitations should be considered when interpreting these results:

- Each configuration was benchmarked once in the reported experiment series.
- No repeated-run variance or confidence intervals are available.
- GPU temperature, clocks, and background activity may vary between runs.
- The 48- and 64-sequence GPU monitoring samples are available, but equivalent detailed GPU traces were not preserved for every earlier test.
- The benchmark measures the OpenAI-compatible serving path, not isolated GPU kernel performance.
- The tests represent one model and one synthetic workload, not all possible serving conditions.

For stronger conclusions, repeat the leading configurations (32, 48 and 64) several times under controlled conditions.

---

# 20\. Final Executive Conclusion

The first SM120 scheduler investigation successfully identified a substantially better sequence-concurrency configuration for Qwen 2.5 1.5B inference.

Starting from the original configuration of 8 sequences, we progressively tested six configurations and recorded the performance impact of each change.

The best result was obtained at 48 sequences:

- 1,207.92 output tokens/sec.
- 72.7% higher sustained output throughput than the original baseline.
- Mean TTFT reduced from approximately 6.49 seconds to 129 milliseconds.
- Zero failed requests.
- Approximately 99% GPU utilization during active inference.

Increasing the limit to 64 reduced sustained throughput, confirming that higher concurrency does not necessarily produce better performance.

The investigation also identified limited VRAM headroom, which is an important constraint for subsequent optimization work.

**Final decision: Retain****`max-num-seqs=48`****and****`max-num-batched-tokens=2048`****as the selected experimental scheduler configuration.**

The next phase should investigate CUDA Graph execution and its potential to improve GPU execution efficiency without compromising memory stability.
