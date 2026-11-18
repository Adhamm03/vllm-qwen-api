# Free Open-Source LLM API with vLLM + Qwen2.5-3B-Instruct

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adhamm03/vllm-qwen-api/blob/main/vllm_qwen_free_api.ipynb)

Serve an **OpenAI-compatible API** from an open-source LLM at zero cost: vLLM running Qwen2.5-3B-Instruct on a free Google Colab T4 GPU, exposed to the internet through a Cloudflare quick tunnel.

Any app that can call the OpenAI API (RAG pipelines, agents, n8n workflows) can use it by changing only the `base_url`, `api_key`, and model name.

## Architecture

```
Your app / n8n / curl
        │  HTTPS
        ▼
Cloudflare quick tunnel (public URL)
        │
        ▼
vLLM server (:8000, /v1/chat/completions)
        │
        ▼
Qwen2.5-3B-Instruct (fp16) on a free Colab T4
```

## Stack

| Tool | Role |
|---|---|
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Open-source model |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput serving engine with an OpenAI-compatible API |
| Google Colab (T4, 16 GB) | Free GPU |
| [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) | Free tunnel that gives the server a public URL |
| OpenAI Python SDK | Client |

## Quick start

1. Click the **Open in Colab** badge above.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order: install vLLM, start the server (wait for `READY`), start the tunnel, then call the API.

## Using the API

```python
from openai import OpenAI

client = OpenAI(base_url="https://YOUR-TUNNEL.trycloudflare.com/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Or with plain HTTP:

```bash
curl https://YOUR-TUNNEL.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"Hi"}]}'
```

## Design notes

- `--dtype half`: the T4 does not support bfloat16, so the bf16 weights are converted to fp16.
- `--max-model-len 4096` and `--gpu-memory-utilization 0.85` keep the KV cache within the T4's memory.
- The server runs without `--api-key` for simplicity, so the client sends `api_key="EMPTY"`. Add `--api-key <secret>` before sharing the URL.

## Problems I ran into

| Symptom | Cause | Fix |
|---|---|---|
| Server dies during startup with `SIGINT` | The server was stopped or started twice | Start it once and don't interrupt the cell |
| `AssertionError: Error in memory profiling` (free memory increased) | An older vLLM process was still holding GPU memory and released it during startup | Kill all vLLM processes, wait until `nvidia-smi` shows ~0 MiB, start once |
| Cloudflare 502 / 1033 | Tunnel is up but vLLM isn't ready yet | Wait for `Application startup complete` in `vllm.log` |

## Limitations

- **Not always on:** free Colab sessions end after a few hours or when idle.
- **The URL changes** each time the tunnel restarts. A free Cloudflare account allows a named tunnel with a stable URL.
- **No authentication by default.** Anyone with the URL can use the GPU.
- A 3B model is limited on complex reasoning. Larger models (for example a 4-bit quantized 7B) can also fit on a T4.
- **Model license:** check the [model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) for the license terms before any commercial use.

## License

Code in this repository: MIT. The model is subject to its own license (see above).
