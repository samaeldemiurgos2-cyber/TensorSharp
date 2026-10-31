# Single-Model `TensorSharp.Server.Host` Producer on an NVIDIA Tesla T4 (HiveMind OpenRouter)

`TensorSharp.Server.Host` exposes an OpenAI-compatible HTTP API (`GET /v1/models`, `POST /v1/chat/completions`). This guide walks through a reproducible, single-model producer setup on a **15 GiB NVIDIA Tesla T4 (`compute_75`)**, covering:

1. **Loopback-only binding (`--host 127.0.0.1`)** and authenticated TLS exposure through `hivemind_producer_shield.py`.
2. **Consistent `/v1/models` model IDs** across requests, the shield allowlist (`--allow-model`), and producer registration (`hivemind_producer_agent.py`).
3. **Authentication checks (`401` / `403` / `200`), SSE streaming (`"stream": true`), throughput, and peak VRAM measurements** on a Tesla T4, kept strictly separate from the RTX PRO 6000 Blackwell reference benchmarks in [`docs/moe_cpu_offload_benchmark.md`](../moe_cpu_offload_benchmark.md).

> **Scope note:** This first-iteration example focuses on a single-model producer on one GPU (both fully GPU-resident dense/hybrid and single-GPU `--n-cpu-moe` MoE CPU offload). Multi-node distributed execution (`--tp`, `--layer-split`) and per-request text-LoRA serving (`--loras-dir` + request field `"lora": "<name>"`, currently supported on single-device `nemotron_h`, `mistral3`, and `hunyuan-dense`) are left for a separate validation.

---

## 1. Security Architecture: Loopback Binding & Authenticated TLS Exposure

By default, `TensorSharp.Server.Host` listens on `0.0.0.0:5000` without authentication or TLS. Running an authentication proxy on another port while leaving `TensorSharp.Server.Host` on `0.0.0.0` would still allow direct unauthenticated access to the backend port.

To secure the producer node:

1. **Bind `TensorSharp.Server.Host` explicitly to loopback (`--host 127.0.0.1`)** and disable the web UI (`--no-webui`).
2. **Run `hivemind_producer_shield.py` on loopback (`--host 127.0.0.1 --port 8787`)** with a shared secret `HIVEMIND_ENDPOINT_TOKEN` (`>= 16` characters). The shield enforces `Authorization: Bearer $HIVEMIND_ENDPOINT_TOKEN`, `POST`-only chat forwarding, request body size limits, concurrency caps, and exact model allowlisting (`--allow-model`).
3. **Expose only the shield (`127.0.0.1:8787`) through an encrypted HTTPS (`https://...`) connection** (such as a Cloudflare Tunnel, Caddy/Nginx TLS reverse proxy, or Tailscale Funnel), and register that HTTPS URL alongside the matching `HIVEMIND_ENDPOINT_TOKEN` and your `HIVEMIND_API_KEY` via `hivemind_producer_agent.py`.

```text
[Internet / HiveMind Router]
        │  HTTPS + Authorization: Bearer $HIVEMIND_ENDPOINT_TOKEN
        ▼
[TLS Reverse Proxy / Tunnel]  (exposes ONLY 127.0.0.1:8787)
        │
        ▼
[hivemind_producer_shield.py] (127.0.0.1:8787, verifies token & --allow-model)
        │  HTTP (loopback only)
        ▼
[TensorSharp.Server.Host]     (127.0.0.1:11436, --host 127.0.0.1 --no-webui)
```

---

## 2. Step-by-Step Single-Model Producer Setup on a Tesla T4

### Step 1 — Start `TensorSharp.Server.Host` on `127.0.0.1:11436`

Choose either a fully GPU-resident model (Profile A: `Qwen3.5-9B-Q4_K_M.gguf`) or a partial MoE expert CPU-offload model (Profile B: `gpt-oss-20b-MXFP4.gguf` with `--n-cpu-moe 18` or `--n-cpu-moe 24`):

```bash
# Profile A: Single-model GPU-resident producer (Qwen3.5-9B-Q4_K_M.gguf, peaks at 6,062 MiB VRAM)
MAX_CONTEXT=4096 dotnet TensorSharp.Server.Host.dll \
  --model /models/Qwen3.5-9B-Q4_K_M.gguf \
  --backend ggml_cuda \
  --host 127.0.0.1 \
  --port 11436 \
  --no-webui \
  --no-prefix-cache

# Profile B: Single-model MoE CPU-offload producer (gpt-oss-20b-MXFP4.gguf, peaks at 7,042 MiB @ 18 or 4,610 MiB @ 24)
MAX_CONTEXT=4096 dotnet TensorSharp.Server.Host.dll \
  --model /models/gpt-oss-20b-MXFP4.gguf \
  --backend ggml_cuda \
  --n-cpu-moe 18 \
  --host 127.0.0.1 \
  --port 11436 \
  --no-webui \
  --no-prefix-cache
```

### Step 2 — Query `GET /v1/models` to Obtain the Exact Model ID

`TensorSharp.Server.Host` registers the loaded GGUF model under its filename stem in `GET /v1/models`. Because `hivemind_producer_shield.py` forwards request payloads unchanged and checks `body["model"]` directly against `--allow-model`, and `hivemind_producer_agent.py` verifies at startup that the local model name is present in `GET /v1/models`, **do not use Ollama-style aliases** (such as `qwen3.5:9b` or `gpt-oss:20b`). Always query `/v1/models` and use the returned `id` verbatim:

```bash
curl -s http://127.0.0.1:11436/v1/models | jq .
```

Response (Profile A):

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.5-9B-Q4_K_M",
      "object": "model",
      "created": 1793380470,
      "owned_by": "tensorsharp"
    }
  ]
}
```

*(For Profile B with `gpt-oss-20b-MXFP4.gguf`, `GET /v1/models` returns `"id": "gpt-oss-20b-MXFP4"`.)*

### Step 3 — Start `hivemind_producer_shield.py` with Matching Credentials & Model ID

Generate a strong `HIVEMIND_ENDPOINT_TOKEN` (`>= 16` chars) and bind the shield to `127.0.0.1:8787` using `--host` and `--port`:

```bash
export HIVEMIND_ENDPOINT_TOKEN="$(openssl rand -hex 24)"

# For Profile A ("Qwen3.5-9B-Q4_K_M"):
python3 scripts/hivemind_producer_shield.py \
  --upstream http://127.0.0.1:11436 \
  --host 127.0.0.1 \
  --port 8787 \
  --allow-model Qwen3.5-9B-Q4_K_M \
  --max-concurrent 2

# (For Profile B, pass: --allow-model gpt-oss-20b-MXFP4)
```

### Step 4 — Verify Shield Authentication, Allowlist Enforcement, and SSE Streaming

Before exposing the shield, verify all four security and streaming checks locally against `http://127.0.0.1:8787`:

```bash
# 1. Missing bearer token -> HTTP 401 ("unauthorized")
curl -i http://127.0.0.1:8787/v1/models

# 2. Wrong bearer token -> HTTP 401 ("unauthorized")
curl -i -H "Authorization: Bearer wrong-token-value-12345" http://127.0.0.1:8787/v1/models

# 3. Unmapped Ollama-style alias -> HTTP 403 ("model_not_allowed")
curl -i -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $HIVEMIND_ENDPOINT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"Hello"}]}'

# 4. Valid bearer token + exact /v1/models ID + SSE streaming -> HTTP 200 (text/event-stream)
curl -N -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $HIVEMIND_ENDPOINT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q4_K_M",
    "stream": true,
    "max_tokens": 96,
    "temperature": 0.0,
    "messages": [
      {"role": "user", "content": "Write a short Python function fibonacci(n) iteratively in O(n) time."}
    ]
  }'
```

### Step 5 — Expose Only the Shield Over HTTPS & Register with `hivemind_producer_agent.py`

Expose `http://127.0.0.1:8787` (never `127.0.0.1:11436`) via an encrypted HTTPS tunnel or reverse proxy, and start the heartbeat agent with the matching `HIVEMIND_ENDPOINT_TOKEN`, your `HIVEMIND_API_KEY` (`hm_live_...`), and the exact `/v1/models` ID on the right-hand side of `--model <public_tier>=<exact_v1_models_id>`:

```bash
# Example HTTPS tunnel pointing strictly at the shield on 127.0.0.1:8787:
cloudflared tunnel --url http://127.0.0.1:8787

# In a second shell, register the HTTPS shield URL and matching token:
export HIVEMIND_API_KEY="hm_live_your_key_here"
export HIVEMIND_ENDPOINT_TOKEN="<same_token_used_by_shield>"

# Profile A registration:
python3 scripts/hivemind_producer_agent.py \
  --router-url https://hivemind-openrouter-1003063861791.us-central1.run.app \
  --public-url https://your-tunnel-subdomain.trycloudflare.com \
  --local-url http://127.0.0.1:11436 \
  --model hivemind/qwen3.5-9b=Qwen3.5-9B-Q4_K_M \
  --execution-class t4 \
  --gpu-vram-mib 15360

# (For Profile B, pass: --model hivemind/gpt-oss-20b=gpt-oss-20b-MXFP4)
```

---

## 3. Measured Tesla T4 Validation Results (`v2026.10.23`)

### Test Environment & Reproducibility Metadata

| Field | Value |
| --- | --- |
| **TensorSharp Commit / Release** | `a37448127f2fcd70079889c7050bdafb6b4bbfc7` (`v2026.10.23`, `tensorsharp-server-2026.10.23-linux-x64-cuda.tar.gz`) |
| **GPU** | 1x NVIDIA Tesla T4 (`15,360 MiB` GDDR6, Turing `compute_75` / `sm_75`) |
| **NVIDIA Driver / CUDA** | Driver `580.178.04`, CUDA Driver `13.0` (host toolkit `CUDA 12.9` + bundled CUDA runtime in release archive) |
| **CPU & System RAM** | 8x vCPU Intel(R) Xeon(R) CPU @ 2.20 GHz (1 NUMA node), `29 GiB` RAM (`23 GiB` available) |
| **Server Flags & Context** | `MAX_CONTEXT=4096`, `--backend ggml_cuda --host 127.0.0.1 --port 11436 --no-webui --no-prefix-cache` |
| **Profile A GGUF Artifact** | `Qwen3.5-9B-Q4_K_M.gguf` (`unsloth/Qwen3.5-9B-GGUF`, `5,680,522,464` bytes / `5.29 GiB`) |
| **Profile A SHA-256** | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| **Profile B GGUF Artifact** | `gpt-oss-20b-MXFP4.gguf` (`dhiltgen/gpt-oss:20b-llamacpp`, `13,793,422,816` bytes / `12.85 GiB`, 24 MoE layers) |
| **Profile B SHA-256** | `8ecb5d4ad6172c63f9cbf1a07a04e525985a8a11a6e516da02b674c4010de716` |

### Measured Single-Model Producer Performance, Auth Checks & Peak VRAM on Tesla T4

All rows below were measured end-to-end through `hivemind_producer_shield.py` (`127.0.0.1:8787` -> `127.0.0.1:11436`) on the Tesla T4 node above, with per-process GPU memory sampled via `nvidia-smi --query-compute-apps=pid,used_gpu_memory` every 250 ms:

| `/v1/models` ID | `--n-cpu-moe` | Idle VRAM | Peak VRAM | Auth Checks (`No Token` / `Bad Token` / `Alias` / `Valid`) | Prompt / Gen Tokens | Streaming TTFT | Prefill (`tok/s`) | Streamed Decode (`tok/s`) | Non-Stream E2E (`tok/s`) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.5-9B-Q4_K_M` | None (GPU-resident) | `5,420 MiB` | **`6,062 MiB`** | `401` / `401` / `403` / `200` | `3,289` / `96` (`97` SSE chunks) | `3,451.8 ms` | **`952.8 tok/s`** | **`32.8 tok/s`** | `14.6 tok/s` |
| `gpt-oss-20b-MXFP4` | `18` (of 24 layers) | `6,104 MiB` | **`7,042 MiB`** | `401` / `401` / `403` / `200` | `2,677` / `96` (`88` SSE chunks) | `4,283.7 ms` | **`624.9 tok/s`** | **`13.0 tok/s`** | `8.1 tok/s` |
| `gpt-oss-20b-MXFP4` | `24` (of 24 layers) | `3,662 MiB` | **`4,610 MiB`** | `401` / `401` / `403` / `200` | `2,677` / `96` (`88` SSE chunks) | `4,730.0 ms` | **`566.0 tok/s`** | **`10.8 tok/s`** | `6.9 tok/s` |

---

## 4. Reference Benchmarks (`docs/moe_cpu_offload_benchmark.md`) vs. T4 Measurements

For comparison, the reference `--n-cpu-moe` benchmarks in [`docs/moe_cpu_offload_benchmark.md`](../moe_cpu_offload_benchmark.md) were recorded on a workstation-class server (**2x NVIDIA RTX PRO 6000 Blackwell Server Edition, `97,887 MiB` VRAM each, PCIe 5.0 x16, 2x Intel Xeon 6952P, `1,511 GiB` RAM, CUDA 12.8**) using `TensorSharp.Cli --benchmark-prefill-lengths 8192 --max-tokens 128 --no-think`:

| Model (Reference Benchmark) | Quantization & Artifact (`docs/moe_cpu_offload_benchmark.md`) | `--n-cpu-moe` Offload | Reference VRAM (`RTX PRO 6000 Blackwell`) | Reference `pp8192` (`TensorSharp` vs `llama.cpp`) | Reference `tg128` (`TensorSharp` vs `llama.cpp`) |
| --- | --- | --- | ---: | ---: | ---: |
| **Gemma 4 26B-A4B** | `UD-IQ4_XS` (`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`, `12.8 GiB`) | `16 of 30` | `~13.7 GiB` | `4,888 tok/s` vs `854 tok/s` (**5.72x**) | `54.5 tok/s` vs `21.9 tok/s` (**2.49x**) |
| **Gemma 4 26B-A4B** | `UD-IQ4_XS` (`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`, `12.8 GiB`) | `30 of 30` | `10.8 GiB` | `3,072 tok/s` vs `495 tok/s` (**6.21x**) | `39.7 tok/s` vs `12.9 tok/s` (**3.08x**) |
| **Qwen 3.5 35B-A3B** | `UD-IQ4_XS` (`Qwen3.5-35B-A3B-UD-IQ4_XS.gguf`, `17.7 GiB`) | `24 of 48` | `~17.3 GiB` | `5,259 tok/s` vs `484 tok/s` (**10.87x**) | `52.3 tok/s` vs `15.8 tok/s` (**3.31x**) |
| **Qwen 3.5 35B-A3B** | `UD-IQ4_XS` (`Qwen3.5-35B-A3B-UD-IQ4_XS.gguf`, `17.7 GiB`) | `48 of 48` | `~12.9 GiB` | `3,709 tok/s` vs `457 tok/s` (**8.12x**) | `38.6 tok/s` vs `10.2 tok/s` (**3.78x**) |
| **GPT-OSS 20B** | `Q8_0 / MXFP4` (`gpt-oss-20b-MXFP4.gguf`, `11.3 GiB`) | `12 of 24` | `~6.8 GiB` | `6,394 tok/s` vs `1,188 tok/s` (**5.38x**) | `51.7 tok/s` vs `18.3 tok/s` (**2.83x**) |
| **GPT-OSS 20B** | `Q8_0 / MXFP4` (`gpt-oss-20b-MXFP4.gguf`, `11.3 GiB`) | `24 of 24` | `~2.2 GiB` | `3,798 tok/s` vs `548 tok/s` (**6.93x**) | `27.7 tok/s` vs `9.4 tok/s` (**2.95x**) |

> **Important distinction:** Any intermediate `--n-cpu-moe` memory or throughput figures computed by external sizing utilities (such as `plan_tensorsharp_moe_offload` in the HiveMind companion bridge) are **linear interpolation estimates** derived from the RTX PRO 6000 Blackwell reference points above, whereas Section 3 reports direct measurements on a single PCIe Gen3 Tesla T4 (`compute_75`) with an 8-vCPU Xeon host.
