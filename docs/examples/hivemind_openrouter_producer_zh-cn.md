# 在 NVIDIA Tesla T4 上运行单模型 `TensorSharp.Server.Host` Producer（接入 HiveMind OpenRouter）

`TensorSharp.Server.Host` 提供兼容 OpenAI 的 HTTP 接口（`GET /v1/models`、`POST /v1/chat/completions`）。本指南展示如何在 **15 GiB NVIDIA Tesla T4（`compute_75`）** 上复现一个完整的单模型推理节点（Producer），涵盖：

1. **显式绑定本地回环地址（`--host 127.0.0.1`）**，并通过 `hivemind_producer_shield.py` 与加密 HTTPS 隧道安全暴露服务。
2. **统一使用 `GET /v1/models` 返回的真实模型 ID**（同时用于请求体、Shield 白名单 `--allow-model` 以及 `hivemind_producer_agent.py` 节点注册）。
3. **在 Tesla T4 上实测的鉴权检查（`401` / `403` / `200`）、SSE 流式输出（`"stream": true`）、吞吐率与峰值显存（Peak VRAM）**，并与 [`docs/moe_cpu_offload_benchmark.md`](../moe_cpu_offload_benchmark.md) 中基于 RTX PRO 6000 Blackwell 的参考基准测试严格区分。

> **范围说明：** 第一阶段示例聚焦于单卡 Tesla T4 上的单模型 Producer（包含全 GPU 常驻模型与单卡 `--n-cpu-moe` MoE 专家 CPU 卸载）。多机分布式推理（`--tp`、`--layer-split`）以及按请求切换文本 LoRA（通过 `--loras-dir` 与请求字段 `"lora": "<name>"`，当前支持单卡 `nemotron_h`、`mistral3` 与 `hunyuan-dense`）将在后续独立验证的贡献中提供。

---

## 1. 安全架构：本地回环绑定与鉴权加密暴露

默认情况下，`TensorSharp.Server.Host` 监听 `0.0.0.0:5000`，且不内置身份验证或 TLS。如果仅在另一端口运行鉴权代理而让 `TensorSharp.Server.Host` 保持默认监听 `0.0.0.0`，外部网络仍可绕过代理直接访问后端接口。

请按以下三步加固节点配置：

1. **将 `TensorSharp.Server.Host` 显式绑定到 `127.0.0.1`**（`--host 127.0.0.1 --port 11436 --no-webui`）。
2. **在本地回环启动 `hivemind_producer_shield.py`**（使用 `--host 127.0.0.1 --port 8787`，注意参数为 `--host` 与 `--port` 而非 `--listen`），并配置环境变量 `HIVEMIND_ENDPOINT_TOKEN`（不少于 16 个字符）。Shield 会强制校验 `Authorization: Bearer $HIVEMIND_ENDPOINT_TOKEN`、仅允许转发 `POST` 推理请求、限制并发与请求体大小，并严格匹配 `--allow-model` 白名单。
3. **仅通过经过身份验证的 TLS/HTTPS 连接暴露 Shield（`127.0.0.1:8787`）**（例如 Cloudflare Tunnel、Caddy/Nginx TLS 反向代理或 Tailscale Funnel），并在 `hivemind_producer_agent.py` 中使用相同的 `HIVEMIND_ENDPOINT_TOKEN` 与您的 `HIVEMIND_API_KEY` 注册该 `https://` 地址。

```text
[公网 / HiveMind 路由器]
        │  HTTPS + Authorization: Bearer $HIVEMIND_ENDPOINT_TOKEN
        ▼
[TLS 反向代理 / 加密隧道]      (仅暴露 127.0.0.1:8787)
        │
        ▼
[hivemind_producer_shield.py] (127.0.0.1:8787，校验 Token 与 --allow-model)
        │  HTTP (仅限本地回环)
        ▼
[TensorSharp.Server.Host]     (127.0.0.1:11436，--host 127.0.0.1 --no-webui)
```

---

## 2. Tesla T4 单模型 Producer 分步复现指南

### 第一步：在 `127.0.0.1:11436` 启动 `TensorSharp.Server.Host`

可选择全量常驻显存模型（配置 A：`Qwen3.5-9B-Q4_K_M.gguf`）或启用 MoE 专家 CPU 卸载的模型（配置 B：`gpt-oss-20b-MXFP4.gguf`，搭配 `--n-cpu-moe 18` 或 `--n-cpu-moe 24`）：

```bash
# 配置 A：单模型全 GPU 常驻（Qwen3.5-9B-Q4_K_M.gguf，实测峰值显存 6,062 MiB）
MAX_CONTEXT=4096 dotnet TensorSharp.Server.Host.dll \
  --model /models/Qwen3.5-9B-Q4_K_M.gguf \
  --backend ggml_cuda \
  --host 127.0.0.1 \
  --port 11436 \
  --no-webui \
  --no-prefix-cache

# 配置 B：单模型 MoE 专家 CPU 卸载（gpt-oss-20b-MXFP4.gguf，18 层卸载峰值显存 7,042 MiB，24 层卸载峰值显存 4,610 MiB）
MAX_CONTEXT=4096 dotnet TensorSharp.Server.Host.dll \
  --model /models/gpt-oss-20b-MXFP4.gguf \
  --backend ggml_cuda \
  --n-cpu-moe 18 \
  --host 127.0.0.1 \
  --port 11436 \
  --no-webui \
  --no-prefix-cache
```

### 第二步：查询 `GET /v1/models` 获取真实模型 ID

`TensorSharp.Server.Host` 在 `GET /v1/models` 中返回的模型 `id` 是 GGUF 文件名去除 `.gguf` 后缀的名称。由于 `hivemind_producer_shield.py` 会原样转发请求体并将 `body["model"]` 与 `--allow-model` 精确比对，且 `hivemind_producer_agent.py` 启动时也会校验本地模型名是否存在于 `GET /v1/models` 返回列表中，**请勿假定 Ollama 风格的别名（如 `qwen3.5:9b` 或 `gpt-oss:20b`）会自动映射**。请先查询 `/v1/models` 并始终使用返回的真实 `id`：

```bash
curl -s http://127.0.0.1:11436/v1/models | jq .
```

返回示例（配置 A）：

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

*（对于加载 `gpt-oss-20b-MXFP4.gguf` 的配置 B，`GET /v1/models` 返回 `"id": "gpt-oss-20b-MXFP4"`。）*

### 第三步：使用匹配的凭据与模型 ID 启动 `hivemind_producer_shield.py`

生成至少 16 个字符的 `HIVEMIND_ENDPOINT_TOKEN`，并使用 `--host 127.0.0.1 --port 8787` 启动 Shield：

```bash
export HIVEMIND_ENDPOINT_TOKEN="$(openssl rand -hex 24)"

# 配置 A（"Qwen3.5-9B-Q4_K_M"）：
python3 scripts/hivemind_producer_shield.py \
  --upstream http://127.0.0.1:11436 \
  --host 127.0.0.1 \
  --port 8787 \
  --allow-model Qwen3.5-9B-Q4_K_M \
  --max-concurrent 2

# （若为配置 B，请将白名单改为：--allow-model gpt-oss-20b-MXFP4）
```

### 第四步：验证鉴权拦截、模型白名单与 SSE 流式推理

在将 Shield 暴露到公网前，先针对 `http://127.0.0.1:8787` 验证以下四项检查：

```bash
# 1. 未携带 Token -> 返回 HTTP 401 ("unauthorized")
curl -i http://127.0.0.1:8787/v1/models

# 2. 错误的 Token -> 返回 HTTP 401 ("unauthorized")
curl -i -H "Authorization: Bearer wrong-token-value-12345" http://127.0.0.1:8787/v1/models

# 3. 未在白名单中的 Ollama 别名 -> 返回 HTTP 403 ("model_not_allowed")
curl -i -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $HIVEMIND_ENDPOINT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"Hello"}]}'

# 4. 正确 Token + /v1/models 真实模型 ID + SSE 流式请求 -> 返回 HTTP 200 (text/event-stream)
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

### 第五步：仅通过 HTTPS 暴露 Shield 并运行 `hivemind_producer_agent.py` 注册

通过加密 HTTPS 隧道或反向代理暴露 `http://127.0.0.1:8787`（切勿直接暴露 `127.0.0.1:11436`），随后使用相同的 `HIVEMIND_ENDPOINT_TOKEN`、您的 `HIVEMIND_API_KEY`（`hm_live_...`）以及 `--model <公开模型层>=<v1/models真实ID>` 启动心跳代理：

```bash
# 示例：启动仅指向本地 Shield (127.0.0.1:8787) 的 HTTPS 隧道
cloudflared tunnel --url http://127.0.0.1:8787

# 在第二个终端中设置相匹配的凭据并注册节点：
export HIVEMIND_API_KEY="hm_live_your_key_here"
export HIVEMIND_ENDPOINT_TOKEN="<与_shield_相同的_token>"

# 配置 A 注册命令：
python3 scripts/hivemind_producer_agent.py \
  --router-url https://hivemind-openrouter-1003063861791.us-central1.run.app \
  --public-url https://your-tunnel-subdomain.trycloudflare.com \
  --local-url http://127.0.0.1:11436 \
  --model hivemind/qwen3.5-9b=Qwen3.5-9B-Q4_K_M \
  --execution-class t4 \
  --gpu-vram-mib 15360

# （若为配置 B，请使用：--model hivemind/gpt-oss-20b=gpt-oss-20b-MXFP4）
```

---

## 3. NVIDIA Tesla T4 实测结果（`v2026.10.23`）

### 测试环境与可复现元数据

| 项目 | 实测配置详情 |
| --- | --- |
| **TensorSharp 版本 / Commit** | `a37448127f2fcd70079889c7050bdafb6b4bbfc7`（`v2026.10.23`，`tensorsharp-server-2026.10.23-linux-x64-cuda.tar.gz`） |
| **GPU 硬件** | 1x NVIDIA Tesla T4（`15,360 MiB` GDDR6，Turing `compute_75` / `sm_75`） |
| **NVIDIA 驱动 / CUDA** | Driver `580.178.04`，CUDA Driver `13.0`（宿主机 `CUDA 12.9` + 发行包内置 CUDA Runtime） |
| **CPU 与系统内存** | 8x vCPU Intel(R) Xeon(R) CPU @ 2.20 GHz（单 NUMA 节点），`29 GiB` RAM（`23 GiB` 可用） |
| **服务端启动参数与上下文** | `MAX_CONTEXT=4096`，`--backend ggml_cuda --host 127.0.0.1 --port 11436 --no-webui --no-prefix-cache` |
| **配置 A GGUF 文件** | `Qwen3.5-9B-Q4_K_M.gguf`（`unsloth/Qwen3.5-9B-GGUF`，`5,680,522,464` 字节 / `5.29 GiB`） |
| **配置 A SHA-256** | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| **配置 B GGUF 文件** | `gpt-oss-20b-MXFP4.gguf`（`dhiltgen/gpt-oss:20b-llamacpp`，`13,793,422,816` 字节 / `12.85 GiB`，24 层 MoE） |
| **配置 B SHA-256** | `8ecb5d4ad6172c63f9cbf1a07a04e525985a8a11a6e516da02b674c4010de716` |

### Tesla T4 单模型 Producer 性能、鉴权检查与峰值显存实测表

下表所有数据均通过 `hivemind_producer_shield.py`（`127.0.0.1:8787` -> `127.0.0.1:11436`）在 Tesla T4 节点上端到端实测，进程级峰值显存通过 `nvidia-smi --query-compute-apps=pid,used_gpu_memory` 以 250 ms 间隔采样：

| `/v1/models` 真实 ID | `--n-cpu-moe` 设置 | 空闲显存 | **峰值显存** | 鉴权检查（`无Token` / `错Token` / `别名` / `合法`） | 提示词 / 生成 Token | 流式首字延迟 (TTFT) | 预填充速度 (`tok/s`) | 流式解码速度 (`tok/s`) | 非流式端到端 (`tok/s`) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.5-9B-Q4_K_M` | 无（全量常驻 GPU） | `5,420 MiB` | **`6,062 MiB`** | `401` / `401` / `403` / `200` | `3,289` / `96`（`97` 个 SSE 块） | `3,451.8 ms` | **`952.8 tok/s`** | **`32.8 tok/s`** | `14.6 tok/s` |
| `gpt-oss-20b-MXFP4` | `18`（共 24 层） | `6,104 MiB` | **`7,042 MiB`** | `401` / `401` / `403` / `200` | `2,677` / `96`（`88` 个 SSE 块） | `4,283.7 ms` | **`624.9 tok/s`** | **`13.0 tok/s`** | `8.1 tok/s` |
| `gpt-oss-20b-MXFP4` | `24`（共 24 层） | `3,662 MiB` | **`4,610 MiB`** | `401` / `401` / `403` / `200` | `2,677` / `96`（`88` 个 SSE 块） | `4,730.0 ms` | **`566.0 tok/s`** | **`10.8 tok/s`** | `6.9 tok/s` |

---

## 4. 参考基准测试（`docs/moe_cpu_offload_benchmark.md`）与 Tesla T4 实测对比说明

作为参考对比，[`docs/moe_cpu_offload_benchmark.md`](../moe_cpu_offload_benchmark.md) 中引用的 `--n-cpu-moe` 基准测试是在工作站级服务器（**2x NVIDIA RTX PRO 6000 Blackwell Server Edition，单卡 `97,887 MiB` 显存，PCIe 5.0 x16，2x Intel Xeon 6952P，`1,511 GiB` 系统内存，CUDA 12.8**）上使用 `TensorSharp.Cli --benchmark-prefill-lengths 8192 --max-tokens 128 --no-think` 测得的：

| 模型（参考基准测试） | 量化格式与权重文件（`docs/moe_cpu_offload_benchmark.md`） | `--n-cpu-moe` 卸载层数 | 参考显存占用（`RTX PRO 6000 Blackwell`） | 参考 `pp8192`（`TensorSharp` vs `llama.cpp`） | 参考 `tg128`（`TensorSharp` vs `llama.cpp`） |
| --- | --- | --- | ---: | ---: | ---: |
| **Gemma 4 26B-A4B** | `UD-IQ4_XS`（`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`，`12.8 GiB`） | `16 / 30` | `~13.7 GiB` | `4,888 tok/s` vs `854 tok/s`（**5.72x**） | `54.5 tok/s` vs `21.9 tok/s`（**2.49x**） |
| **Gemma 4 26B-A4B** | `UD-IQ4_XS`（`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`，`12.8 GiB`） | `30 / 30` | `10.8 GiB` | `3,072 tok/s` vs `495 tok/s`（**6.21x**） | `39.7 tok/s` vs `12.9 tok/s`（**3.08x**） |
| **Qwen 3.5 35B-A3B** | `UD-IQ4_XS`（`Qwen3.5-35B-A3B-UD-IQ4_XS.gguf`，`17.7 GiB`） | `24 / 48` | `~17.3 GiB` | `5,259 tok/s` vs `484 tok/s`（**10.87x**） | `52.3 tok/s` vs `15.8 tok/s`（**3.31x**） |
| **Qwen 3.5 35B-A3B** | `UD-IQ4_XS`（`Qwen3.5-35B-A3B-UD-IQ4_XS.gguf`，`17.7 GiB`） | `48 / 48` | `~12.9 GiB` | `3,709 tok/s` vs `457 tok/s`（**8.12x**） | `38.6 tok/s` vs `10.2 tok/s`（**3.78x**） |
| **GPT-OSS 20B** | `Q8_0 / MXFP4`（`gpt-oss-20b-MXFP4.gguf`，`11.3 GiB`） | `12 / 24` | `~6.8 GiB` | `6,394 tok/s` vs `1,188 tok/s`（**5.38x**） | `51.7 tok/s` vs `18.3 tok/s`（**2.83x**） |
| **GPT-OSS 20B** | `Q8_0 / MXFP4`（`gpt-oss-20b-MXFP4.gguf`，`11.3 GiB`） | `24 / 24` | `~2.2 GiB` | `3,798 tok/s` vs `548 tok/s`（**6.93x**） | `27.7 tok/s` vs `9.4 tok/s`（**2.95x**） |

> **重要区分说明：** 伴随桥接模块（如 HiveMind `tensorsharp_collab_bridge.py` 中的 `plan_tensorsharp_moe_offload`）计算的任意中间层数显存与吞吐数值均为基于上述 RTX PRO 6000 Blackwell 参考点的**线性插值估算值（interpolated estimates）**；而第 3 节表格中的数据则是在单卡 PCIe Gen3 Tesla T4（`compute_75`）与 8 核 vCPU Xeon 主机上的端到端实测结果。
