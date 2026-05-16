# Local LLM (fallback) - deployment guide

The system can use a local, quantized Qwen 2.5 32B model (or compatible) as a fallback when the daily quota of GPT/Gemini queries is exhausted, or as an explicit user choice ("Local Llama/Qwen" in UI).

## 1. Infrastructure and Architecture

The application deployment is split between two separate servers to isolate the heavy GPU workload from the web application serving.

| Role | Hostname / IP | Specifications | OS / Access |
|---|---|---|---|
| **GPU Server** (Ollama) | `192.168.162.165` | 1x NVIDIA RTX 3090 Ti 24 GB | Ubuntu (analogous to lab servers), account with `sudo` access |
| **Web App VM** | `nlp01.ii.pw.edu.pl`<br>`192.168.162.238` | 8 vCPU, 16 GB RAM, 256 GB Disk | Ubuntu 24.04.4 LTS Server Minimal, user `mmatusz4` (`sudo` group) |

**Note: Both machines are accessed via SSH using passwords (e.g., `<PASSWORD_FROM_EMAIL>`). Do NOT store these passwords in the repository.**

### Recommended Communication Architecture:

```text
[ Web App VM (192.168.162.238) ] ======= HTTP =======> [ GPU Server (192.168.162.165) ]
                                                           Port: 11434/tcp
    Stack:                                                 Stack: Ollama API
      - nginx (Reverse Proxy)                              Model: qwen2.5:32b-instruct-q4_K_M
      - Docker Compose (app + qdrant)                      Storage: /mnt/storage
```

---

## 2. Storage Setup (GPU Server)

The GPU Server has a dedicated mounted disk at `/mnt/storage` (device `/dev/sdb3`). Since model weights and checkpoints are enormous, they **must** be stored here rather than on the default system partitions.
*(Note: Optionally, `/dev/sdb2` can be wiped and also used as storage if needed).*

### Recommended Directory Structure:
```bash
/mnt/storage/
└── ollama/
    └── models/     # Used for LLM weights and cache
```

### Initializing the storage:
Connect to the GPU server (`ssh <USER>@192.168.162.165`) and prepare the directory:
```bash
sudo mkdir -p /mnt/storage/ollama/models
sudo chown -R $USER:$USER /mnt/storage/ollama
```

---

## 3. GPU Server Setup (192.168.162.165)

### Step 3.1: Install Ollama (Linux, one-time)
Ollama acts as the inference server providing the HTTP API for the local model.
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
The script installs the binary to `/usr/local/bin/ollama` and registers the `ollama.service` systemd service. Ensure the GPU and drivers are detected correctly using `nvidia-smi` (requires driver ≥ 535 and CUDA 12.x).

### Step 3.2: Configure Ollama (Storage path and Network exposure)
By default, Ollama listens on `127.0.0.1:11434` and stores models on the root partition. We need to point it to `/mnt/storage` and expose the port to the VM:
```bash
sudo systemctl edit ollama.service
```
In the opened editor, add the following to correctly route memory and networks:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=/mnt/storage/ollama/models"
Environment="OLLAMA_KEEP_ALIVE=2h"
```
(`OLLAMA_KEEP_ALIVE=2h` keeps the model in VRAM between requests, avoiding 30-60s cold-start loads).

Apply changes and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Open port `11434` on the firewall **strictly** for the Web App VM:
```bash
sudo ufw allow from 192.168.162.238 to any port 11434 proto tcp
```

### Step 3.3: Downloading the model setup
Pull the selected model. Given the 24GB VRAM constraint of the RTX 3090 Ti, we default to a highly capable mid-sized model (Qwen 2.5 32B quant).

```bash
# Recommended default model for 24GB GPU: Qwen 2.5 32B in Q4_K_M
ollama pull qwen2.5:32b-instruct-q4_K_M  
```
Check `ollama list` to verify models are successfully stored on `/mnt/storage`.

---

## 4. Web App VM Setup (nlp01.ii.pw.edu.pl)

Connect to the VM:
```bash
ssh mmatusz4@192.168.162.238
# Password: <PASSWORD_FROM_EMAIL>
```

Since this is a minimal Ubuntu 24.04.4 LTS Server installation, necessary tools must be installed manually.

### Step 4.1: System Updates and Dependencies Installation
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget nginx xdg-utils

# Install Docker using the official repository script:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker mmatusz4
newgrp docker
```

### Step 4.2: Web App Deployment
Clone the repository and prepare the environment:
```bash
git clone <YOUR_REPO_URL> financial-multi-agent
cd financial-multi-agent
```

In the `.env` file, configure the backend to use the GPU Server IP:
```env
LOCAL_LLM_BASE_URL=http://192.168.162.165:11434
LOCAL_LLM_MODEL=qwen2.5:32b-instruct-q4_K_M
```
*Note: `LOCAL_LLM_BASE_URL` is the **Ollama root URL**, WITHOUT `/v1` at the end.*

Start the application stack (app + Qdrant) via Docker Compose:
```bash
docker compose up -d --build
```

### Step 4.3: Reverse Proxy Configuration (Optional/Recommended)
To serve the Streamlit app (which runs on 8501) cleanly over HTTP (port 80):
```bash
sudo nano /etc/nginx/sites-available/financial-app
```
Add the routing block:
```nginx
server {
    listen 80;
    server_name nlp01.ii.pw.edu.pl 192.168.162.238;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        # Increase timeouts for long-running LLM generation
        proxy_read_timeout 3600;
        proxy_send_timeout 3600;
    }
}
```
Enable and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/financial-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 5. End-to-end Communication Test

From the Web App VM (`nlp01`):

```bash
curl http://192.168.162.165:11434/api/tags
```
Expected: JSON with a list of models confirming the connection to the GPU server.

Inference test:
```bash
curl http://192.168.162.165:11434/api/generate -d '{
  "model": "qwen2.5:32b-instruct-q4_K_M",
  "prompt": "Reply with just OK.",
  "stream": false
}'
```

In the application UI: log in, select `Local Llama` in "LLM Provider", start the analysis of any ticker. In the GPU server's Ollama logs (`journalctl -u ollama -f`), you will see the incoming requests.

---

## 6. Troubleshooting

| Symptom | Cause / solution |
|---|---|
| `CUDA out of memory` or extreme slowness (≫1 s/token) | The model weight (~40 GB) exceeds the 3090 Ti's 24GB VRAM. It offloads to extremely slow system RAM. **Solution:** Switch to a smaller model (e.g., `qwen2.5:32b`, `llama3.1:8b`). |
| Root partition running out of space | The model was accidentally downloaded to the root drive. Ensure `Environment="OLLAMA_MODELS=/mnt/storage/ollama/models"` is set in `ollama.service` and Ollama was restarted properly. |
| First request takes 30-60 s | The model is being loaded into VRAM. Set `OLLAMA_KEEP_ALIVE=2h`. |
| `connection refused` from application | Check if `OLLAMA_HOST=0.0.0.0:11434` is set. Verify UFW allows traffic from `192.168.162.238`. Check network with `ping 192.168.162.165`. |
| `model 'X' not found` | Downloaded model is missing. Run `ollama pull <name>` on the GPU server. |
| Concurrent requests limit | One Ollama server = one request at a time by default. |

---

## 7. When the application uses the local model

- **Automatic fallback**: when a logged-in user exhausts their daily quota to OpenAI/Gemini (`DAILY_QUERY_LIMIT` in `.env`), the router (`src/llm_router.py`) returns `local`. UI shows an info banner.
- **Manual choice**: user selects "Local Llama" in the sidebar. Then the query is not counted against the quota (the local model does not consume API budget).
