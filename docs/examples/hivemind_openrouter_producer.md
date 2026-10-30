# Serving TensorSharp MoE Models (`--n-cpu-moe`, `--tp`, `--layer-split`) as a HiveMind OpenRouter Producer

`TensorSharp.Server.Host` exposes an OpenAI-compatible (`/v1/chat/completions`, `/v1/models`) and Ollama-compatible (`/api/tags`) HTTP surface. This allows single-GPU (`--n-cpu-moe N`) and multi-GPU / multi-node (`--tp N`, `--layer-split N`) TensorSharp instances to register directly as community inference producers on the open-source [HiveMind OpenRouter](https://github.com/xaviercallens/HiveMind).

## 1. Single-GPU MoE CPU-Offload (`--n-cpu-moe`) on 8 GB – 16 GB GPUs (`compute_75+`)

Using TensorSharp's partial MoE expert CPU offload (`docs/moe_cpu_offload_benchmark.md`), models that exceed resident GPU VRAM can serve interactive completions with 4.5x–10.9x faster prefill than `llama.cpp -ncmoe`:

```bash
# Example A: Gemma 4 26B-A4B in 10.8 GiB VRAM on a 15 GiB NVIDIA Tesla T4 (compute_75)
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /models/gemma-4-26b-a4b-it-Q4_K_M.gguf \
  --backend ggml_cuda \
  --n-cpu-moe 30 \
  --port 11436

# Example B: Qwen 3.5 35B-A3B or GPT-OSS 20B with partial expert offload
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --backend ggml_cuda \
  --n-cpu-moe 24 \
  --port 11436
```

## 2. Multi-GPU / Multi-Node Frontier MoE Serving (`--tp` / `--layer-split`)

For large MoE models such as `DeepSeek-V4.1-Flash`, `Qwen3.8-Flash-Next`, `GLM-5.3-Flash`, or `Muse-Spark-816B` (`docs/multi-node-inference.md`), run `TensorSharp.Cli` on worker nodes (`--tp-node-id 1..N-1`) and `TensorSharp.Server.Host` on rank 0 (`--tp-node-id 0`):

```bash
export MAX_CONTEXT=16384
export TENSORSHARP_TP_ALLREDUCE=symmetric

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /models/Muse-Spark-816B-UD-IQ3_XXS-00001-of-00008.gguf \
  --backend ggml_cuda \
  --tp 10 \
  --tp-node-id 0 \
  --tp-peers 10.0.0.10:9500 \
  --port 11436
```

## 3. Connecting `TensorSharp.Server.Host` to the HiveMind OpenRouter

Point the authenticating producer shield (`scripts/hivemind_producer_shield.py`) and heartbeat agent (`scripts/hivemind_producer_agent.py`) from [`xaviercallens/HiveMind`](https://github.com/xaviercallens/HiveMind) at your local `TensorSharp.Server.Host` port (`127.0.0.1:11436`):

```bash
python3 scripts/hivemind_producer_shield.py \
  --upstream http://127.0.0.1:11436 \
  --listen 127.0.0.1:8787 \
  --allow-model gpt-oss:20b --allow-model qwen3.5:35b-a3b
```
