# Local LLM (fallback) - deployment guide

The system can use a local, quantized Llama 3.1 model (or compatible) as a fallback when the daily quota of GPT/Gemini queries is exhausted, or as an explicit user choice ("Local Llama" in UI).

Architecture:

```
[ Application Server ]  --HTTP-->  [ GPU Server (Ollama) ]
   src/app.py                       11434/tcp
   crewai (litellm)                 llama3.1:70b-instruct-q4_K_M
```

---

## 1. GPU Server Requirements

| Component | Minimum |
|---|---|
| GPU | 1× NVIDIA with **≥24 GB VRAM** (RTX 3090, RTX 4090, A5000, A6000, L40, A100) |
| RAM | ≥40 GB |
| Disk | ≥80 GB of free space (model weight ~40 GB + cache) |
| OS | Linux (Ubuntu 22.04 LTS or newer) |
| NVIDIA Driver | ≥535 |
| CUDA | 12.x (Ollama detects automatically) |
| Network | Machine in the same VLAN / subnet as the application server |

---

## 2. Ollama Installation (Linux, one-time)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The script installs the binary to `/usr/local/bin/ollama` and registers the `ollama.service` systemd service.

Verification:

```bash
ollama --version
sudo systemctl status ollama
```

## 3. Downloading the model

Llama 3.1 70B in Q4_K_M quantization (~40 GB, fits in 24 GB VRAM for context ≤8k):

```bash
ollama pull llama3.1:70b-instruct-q4_K_M
```

Alternatives when 70B does not fit with the assumed context:

- `llama3.1:70b-instruct-q3_K_M` (~33 GB) - lower precision, less VRAM.
- `qwen2.5:72b-instruct-q4_K_M` - often follows JSON instructions better than Llama.
- `mixtral:8x22b-instruct-q4_K_M` - large, requires ≥40 GB VRAM (will NOT work on a 24 GB GPU).

List of downloaded models: `ollama list`.

## 4. Exposing the port to the local network

By default, Ollama listens on `127.0.0.1:11434`. You need to expose the port to the network interface:

```bash
sudo systemctl edit ollama.service
```

In the opened editor, add:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_KEEP_ALIVE=2h"
```

(`OLLAMA_KEEP_ALIVE=2h` keeps the model in VRAM between requests - without this, the first request after a break takes ~30-60s to load the model.)

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Open port 11434 on the firewall **only** for the university subnet (UFW example):

```bash
sudo ufw allow from 10.0.0.0/8 to any port 11434 proto tcp
```

Adjust the CIDR range to the actual subnet.

## 5. Configuration on the application server side

In the application loop `.env` file:

```env
LOCAL_LLM_BASE_URL=http://<gpu-server-ip>:11434
LOCAL_LLM_MODEL=llama3.1:70b-instruct-q4_K_M
```

Note: `LOCAL_LLM_BASE_URL` is the **Ollama root URL**, WITHOUT `/v1` at the end. The application uses the provider `ollama_chat/<model>` in litellm, which attaches the API path itself.

Restart the application (`docker compose restart app` or `make run`).

## 6. End-to-end test

From the application server:

```bash
curl http://<gpu-server-ip>:11434/api/tags
```

Expected: JSON with a list of models containing `llama3.1:70b-instruct-q4_K_M`.

Inference test:

```bash
curl http://<gpu-server-ip>:11434/api/generate -d '{
  "model": "llama3.1:70b-instruct-q4_K_M",
  "prompt": "Reply with just OK.",
  "stream": false
}'
```

You should get a `"response"` field with the model's answer in about ~a few seconds.

In the application UI: log in, select `Local Llama` in "LLM Provider", start analysis of any ticker. In Ollama logs (`journalctl -u ollama -f`) you will see incoming requests.

## 7. Troubleshooting

| Symptom | Cause / solution |
|---|---|
| `CUDA out of memory` when loading the model | Choose a smaller quantization (Q3_K_M) or a smaller model (32B-34B). Decrease context: `OLLAMA_NUM_CTX=4096` in env service. |
| First request takes 30-60 s | The model is being loaded into VRAM. Set `OLLAMA_KEEP_ALIVE=2h` (see pt. 4). |
| `connection refused` from application | Check if `OLLAMA_HOST=0.0.0.0:11434`, if the firewall allows traffic from the application server, if the GPU-server IP is reachable (`ping`, `nc -zv <ip> 11434`). |
| `model 'X' not found` | Downloaded model is missing. `ollama pull <name>` on the GPU server. |
| Very slow inference (≫1 s/token) | GPU does not support the model (CPU fallback). Check `nvidia-smi` for the `ollama_llama_server` process - if it's not visible during inference, drivers/CUDA are not working. |
| Concurrent requests limit | One Ollama server = one request at a time by default. For concurrent handling consider vLLM instead of Ollama. |

## 8. When the application uses the local model

- **Automatic fallback**: when a logged-in user exhausts their daily quota to OpenAI/Gemini (`DAILY_QUERY_LIMIT` in `.env`), the router (`src/llm_router.py`) returns `local`. UI shows an info banner.
- **Manual choice**: user selects "Local Llama" in the sidebar. Then the query is not counted against the quota (the local model does not consume API budget).
