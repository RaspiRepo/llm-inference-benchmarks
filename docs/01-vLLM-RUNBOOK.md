# Benchmark Runbook

Instructions for deploying inference servers and running standardized LLM serving benchmarks.

## 1 Environment Preparation

Activate the project Python environment:

bash

```bash
source .venv/bin/activate
```

Verify GPU availability:

bash

```bash
nvidia-smi
```

Verify vLLM:

bash

```bash
vllm --version
```

## 2 Start vLLM Server

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

Keep the server running in Terminal 1.

## 3 Verify API Availability

In Terminal 2:

bash

```bash
curl http://localhost:8000/v1/models
```

Verify inference:

bash

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "messages": [
        {"role": "user", "content": "Hello"}
      ],
      "max_tokens": 50
    }'
```

Do not start benchmarking until the API responds successfully.

## 4 Run vLLM Serving Benchmark

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

## 5 TensorRT-LLM Server

Start the TensorRT-LLM server using the corresponding model and configuration.

Example deployment command:

bash

```bash
trtllm-serve Qwen/Qwen2.5-1.5B-Instruct \
    --config /opt/triton_models/config.yml
```

The actual command and configuration must be validated against the installed TensorRT-LLM version and deployment environment.

Verify the API before benchmarking:

bash

```bash
curl http://localhost:8000/v1/models
```

## 6 Benchmark TensorRT-LLM

Use the same benchmark client and workload:

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

Confirm that the TensorRT-LLM endpoint supports the required OpenAI-compatible API.

## 7 Experimental Controls

For a fair comparison:

- Use the same model weights.
- Match model precision.
- Match input and output token lengths.
- Match request rate and prompt count.
- Use equivalent GPU memory constraints where possible.
- Run experiments independently.
- Monitor GPU utilization and memory.
- Record software versions and runtime configurations.

Do not run both engines simultaneously on the same GPU for isolated performance comparisons.

## 8. GPU Monitoring

bash

```bash
watch -n 1 nvidia-smi
```

Capture GPU utilization, VRAM usage, temperature, and power during each experiment.
