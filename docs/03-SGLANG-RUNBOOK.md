# SGLang Benchmark Runbook

**Experiments:** EXP-003-A through EXP-003-D  
**Model:** Qwen2.5-1.5B-Instruct  
**Hardware:** NVIDIA RTX 5060 Ti 16GB (SM120)  
**Serving Runtime:** SGLang  
**Deployment:** Docker  
**Benchmark Client:** vLLM `bench serve`  

---

## 1. Objective

Establish reproducible SGLang performance baselines and high-concurrency saturation limits using the same model and synthetic workload as vLLM and TensorRT-LLM.

The runbook covers:
- **Steady-State Baseline (`EXP-003-B`):** Controlled 5 RPS arrival rate (`--max-running-requests 8`).
- **Control-Plane Starvation (`EXP-003-C`):** Dummy weights (`--load-format dummy`) under infinite burst (`--request-rate inf`).
- **Real-Weight Saturation (`EXP-003-D`):** Full model weights under infinite burst at `--max-running-requests 64` and `128`.

---

## 2. Deployment Environment

SGLang runs inside a Docker container on the Linux host.

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti (SM120) |
| GPU Memory | 16 GB |
| Deployment | Docker container (`--ipc=host`, `--shm-size 32g`) |
| Container Name | `sglang-bench-server` |
| Container Image | `lmsysorg/sglang:latest` (`sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`) |
| SGLang Version | `0.5.23` (PyTorch `2.14.1+cu130`) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (Local Hugging Face snapshot) |
| Serving Port | `30000` (OpenAI-compatible REST API) |

Verify container status, GPU visibility, and exact package version:

```bash
docker ps
docker exec -it sglang-bench-server nvidia-smi
docker exec sglang-bench-server python3 -c "import sglang; print(sglang.__version__)"
```

---

## 3. Configuration Parameters

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--model-path` | Local snapshot path | Loads the cached `Qwen2.5-1.5B-Instruct` weights |
| `--host` / `--port` | `0.0.0.0` / `30000` | Exposes the OpenAI-compatible API outside the container |
| `--trust-remote-code` | Enabled | Allows model-provided custom code |
| `--context-length` | `2048` | Aligns context window with vLLM (`--max-model-len 2048`) |
| `--max-running-requests` | `8` (Baseline) / `64` / `128` | Controls active sequence batch concurrency |
| `--mem-fraction-static` | `0.85` (Baseline) / `0.90` (Saturation) | Sets static VRAM allocation fraction for KV cache |
| `--attention-backend` | `flashinfer` | Selects SGLang's FlashInfer attention kernels |
| `--disable-radix-cache` | Enabled | Prevents artificial prefix cache hits on random prompts |
| `--load-format` | `auto` (Default) / `dummy` (Starvation) | Optional flag to skip matrix math and isolate CPU overhead |

*Note: Disabling RadixCache ensures an apples-to-apples comparison against vLLM on non-overlapping synthetic prompts, though it does not evaluate SGLang's prefix-sharing advantage.*

---

## 4. Start SGLang Server

Always remove any prior container instance before launching a new configuration:

```bash
docker rm -f sglang-bench-server
```

### 4.1 Phase 1: Steady-State Baseline (`EXP-003-B`)

```bash
HF_CACHE_DIR="/home/rtx5060/.cache/huggingface"

docker run -d --name sglang-bench-server \
  --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v "\$HF_CACHE_DIR:/root/.cache/huggingface" \
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

### 4.2 Phase 2: High-Concurrency Saturation (`EXP-003-D`)

Set `--max-running-requests` to `64` or `128` and `--mem-fraction-static 0.90` (add `--load-format dummy` for `EXP-003-C` control-plane starvation testing):

```bash
HF_CACHE_DIR="/home/rtx5060/.cache/huggingface"

docker run -d --name sglang-bench-server \
  --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v "\$HF_CACHE_DIR:/root/.cache/huggingface" \
  --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --context-length 2048 \
    --max-running-requests 128 \
    --mem-fraction-static 0.90 \
    --attention-backend flashinfer \
    --disable-radix-cache
```

Monitor startup logs until the server reports it is ready to accept requests:

```bash
docker logs -f sglang-bench-server
```

---

## 5. Verify API Availability & Warm-Up

Verify health and execute a single warm-up generation request (excluded from measured runs) so initial kernel compilation completes:

```bash
curl http://localhost:30000/health
curl http://localhost:30000/v1/models

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

---

## 6. Run Serving Benchmark

Run from the host virtual environment using the standardized vLLM benchmark client:

### 6.1 Steady-State Workload (`5 RPS`)

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

### 6.2 High-Concurrency Burst Workload (`inf RPS`)

```bash
vllm bench serve \
    --backend openai \
    --base-url http://localhost:30000 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random \
    --input-len 1024 \
    --output-len 256 \
    --num-prompts 100 \
    --request-rate inf \
    --label sm120_sglang_real_128
```
