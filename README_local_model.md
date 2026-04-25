# Lokalny model LLM (fallback) — instrukcja uruchomienia

System może używać lokalnego, kwantyzowanego modelu Llama 3.1 (lub kompatybilnego) jako fallback po wyczerpaniu dziennego limitu zapytań do GPT/Gemini, lub jako jawny wybór użytkownika ("Local Llama" w UI).

Architektura:

```
[ Serwer aplikacyjny ]  --HTTP-->  [ Serwer GPU (Ollama) ]
   src/app.py                       11434/tcp
   crewai (litellm)                 llama3.1:70b-instruct-q4_K_M
```

---

## 1. Wymagania serwera GPU

| Element | Minimum |
|---|---|
| GPU | 1× NVIDIA z **≥24 GB VRAM** (RTX 3090, RTX 4090, A5000, A6000, L40, A100) |
| RAM | ≥40 GB |
| Dysk | ≥80 GB wolnego miejsca (waga modelu ~40 GB + cache) |
| OS | Linux (Ubuntu 22.04 LTS lub nowsze) |
| Driver NVIDIA | ≥535 |
| CUDA | 12.x (Ollama detektuje automatycznie) |
| Sieć | Maszyna w tej samej VLAN-ie / podsieci co serwer aplikacyjny |

---

## 2. Instalacja Ollamy (Linux, jednorazowo)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Skrypt instaluje binarkę do `/usr/local/bin/ollama` oraz rejestruje usługę systemd `ollama.service`.

Weryfikacja:

```bash
ollama --version
sudo systemctl status ollama
```

## 3. Pobranie modelu

Llama 3.1 70B w kwantyzacji Q4_K_M (~40 GB, mieści się w 24 GB VRAM dla kontekstu ≤8k):

```bash
ollama pull llama3.1:70b-instruct-q4_K_M
```

Alternatywy gdy 70B nie mieści się przy zakładanym kontekście:

- `llama3.1:70b-instruct-q3_K_M` (~33 GB) — mniejsza precyzja, mniejsze VRAM.
- `qwen2.5:72b-instruct-q4_K_M` — często lepiej trzyma instrukcje JSON-owe od Llamy.
- `mixtral:8x22b-instruct-q4_K_M` — duży, wymaga ≥40 GB VRAM (na 24 GB GPU NIE zadziała).

Lista pobranych modeli: `ollama list`.

## 4. Wystawienie portu na sieć lokalną

Domyślnie Ollama słucha na `127.0.0.1:11434`. Trzeba wystawić port na interfejs sieciowy:

```bash
sudo systemctl edit ollama.service
```

W otwartym edytorze dopisz:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_KEEP_ALIVE=2h"
```

(`OLLAMA_KEEP_ALIVE=2h` trzyma model w VRAM między requestami — bez tego pierwsze zapytanie po przerwie ładuje model ~30–60 s.)

Następnie:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Otwórz port 11434 na firewallu **tylko** dla podsieci uczelnianej (przykład UFW):

```bash
sudo ufw allow from 10.0.0.0/8 to any port 11434 proto tcp
```

Dostosuj zakres CIDR do faktycznej podsieci.

## 5. Konfiguracja po stronie serwera aplikacyjnego

W pliku `.env` aplikacji:

```env
LOCAL_LLM_BASE_URL=http://<gpu-server-ip>:11434
LOCAL_LLM_MODEL=llama3.1:70b-instruct-q4_K_M
```

Uwaga: `LOCAL_LLM_BASE_URL` to **root URL Ollamy**, BEZ `/v1` na końcu. Aplikacja używa providera `ollama_chat/<model>` w litellm, który sam dorabia ścieżkę API.

Restart aplikacji (`docker compose restart app` lub `make run`).

## 6. Test end-to-end

Z serwera aplikacyjnego:

```bash
curl http://<gpu-server-ip>:11434/api/tags
```

Spodziewane: JSON ze listą modeli zawierającą `llama3.1:70b-instruct-q4_K_M`.

Test inferencji:

```bash
curl http://<gpu-server-ip>:11434/api/generate -d '{
  "model": "llama3.1:70b-instruct-q4_K_M",
  "prompt": "Reply with just OK.",
  "stream": false
}'
```

Powinieneś dostać pole `"response"` z odpowiedzią modelu w ciągu ~kilku sekund.

W UI aplikacji: zaloguj się, w "LLM Provider" wybierz `Local Llama`, uruchom analizę dowolnego tickera. W logach Ollamy (`journalctl -u ollama -f`) zobaczysz przychodzące requesty.

## 7. Troubleshooting

| Objaw | Przyczyna / rozwiązanie |
|---|---|
| `CUDA out of memory` przy ładowaniu modelu | Wybierz mniejszą kwantyzację (Q3_K_M) lub mniejszy model (32B–34B). Zmniejsz kontekst: `OLLAMA_NUM_CTX=4096` w env service. |
| Pierwszy request trwa 30–60 s | Model jest ładowany do VRAM. Ustaw `OLLAMA_KEEP_ALIVE=2h` (patrz pkt 4). |
| `connection refused` z aplikacji | Sprawdź czy `OLLAMA_HOST=0.0.0.0:11434`, czy firewall dopuszcza ruch z serwera aplikacyjnego, czy IP GPU-server jest osiągalne (`ping`, `nc -zv <ip> 11434`). |
| `model 'X' not found` | Brakuje pobranego modelu. `ollama pull <name>` na serwerze GPU. |
| Bardzo wolna inferencja (≫1 s/token) | GPU nie obsługuje modelu (CPU fallback). Sprawdź `nvidia-smi` pod kątem procesu `ollama_llama_server` — jeśli go nie widać podczas inferencji, sterowniki/CUDA nie działają. |
| Limit konkurentnych requestów | Jeden serwer Ollamy = jeden request na raz domyślnie. Dla równoczesnej obsługi rozważ vLLM zamiast Ollamy. |

## 8. Kiedy aplikacja używa lokalnego modelu

- **Automatyczny fallback**: gdy zalogowany użytkownik wyczerpie dzienny limit zapytań do OpenAI/Gemini (`DAILY_QUERY_LIMIT` w `.env`), router (`src/llm_router.py`) zwraca `local`. UI pokazuje banner informacyjny.
- **Wybór ręczny**: użytkownik wybiera "Local Llama" w sidebarze. Wtedy zapytanie nie jest liczone do limitu (lokalny model nie kosztuje budżetu API).
