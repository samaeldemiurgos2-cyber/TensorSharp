# 将 TensorSharp MoE 模型（`--n-cpu-moe`、`--tp`、`--layer-split`）接入 HiveMind OpenRouter 推理网络

`TensorSharp.Server.Host` 提供兼容 OpenAI（`/v1/chat/completions`、`/v1/models`）与 Ollama（`/api/tags`）的 HTTP 接口，使单卡（`--n-cpu-moe N`）或多卡/多节点（`--tp N`、`--layer-split N`）TensorSharp 实例能够直接作为开源推理节点接入 [HiveMind OpenRouter](https://github.com/xaviercallens/HiveMind)。

## 1. 在 8 GB – 16 GB 显卡（`compute_75+`）上启用 MoE 专家 CPU 卸载（`--n-cpu-moe`）

借助 TensorSharp 的 MoE 专家按层 CPU 卸载（详见 `docs/moe_cpu_offload_benchmark.md`），长上下文预填充速度相比 `llama.cpp -ncmoe` 提升 **4.5x–10.9x**：

```bash
# 示例 A：在 15 GiB Tesla T4 (compute_75) 上以 10.8 GiB 显存运行 Gemma 4 26B-A4B
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /models/gemma-4-26b-a4b-it-Q4_K_M.gguf \
  --backend ggml_cuda \
  --n-cpu-moe 30 \
  --port 11436

# 示例 B：运行 Qwen 3.5 35B-A3B（24/48 层卸载，pp8192 达 5,259 tok/s，tg128 达 52.3 tok/s）
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --backend ggml_cuda \
  --n-cpu-moe 24 \
  --port 11436
```

## 2. 多卡 / 多节点超大 MoE 模型推理（`--tp` / `--layer-split`）

对于 `DeepSeek-V4.1-Flash`、`Qwen3.8-Flash-Next`、`GLM-5.3-Flash` 或 `Muse-Spark-816B`（详见 `docs/multi-node-inference.md`），在工作节点运行 `TensorSharp.Cli`，在主节点（`--tp-node-id 0`）运行 `TensorSharp.Server.Host` 并通过 `hivemind_producer_shield.py` 安全暴露给路由器。
