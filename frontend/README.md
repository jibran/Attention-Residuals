# AttnRes Chat — Frontend

A FastAPI-based chat UI for the AttnRes language model family.
Supports light and dark mode, HTTP Basic Auth, and real-time text generation
with configurable model selection, temperature, top-k, token count, and KV
caching.

---

## Features

| Feature | Detail |
|---|---|
| **Chat UI** | Clean, responsive interface inspired by modern LLM chat apps |
| **Light / dark mode** | Toggled per user, persisted in `localStorage` |
| **Model selector** | Auto-discovers `*.pt` checkpoint files from a configurable directory |
| **Model metadata** | Shows architecture, parameter count, val PPL, and dataset for each checkpoint |
| **Generation controls** | Temperature · max new tokens · top-k · KV cache toggle |
| **Timing badges** | Tokens/sec, ms/tok, total time displayed below each response |
| **HTTP Basic Auth** | Protects all routes; supports plain or bcrypt-hashed passwords |
| **Session cookies** | Signed JWT session so the browser doesn't re-prompt on every page load |
| **REST API** | `GET /api/models` · `POST /api/generate` · `GET /api/version` |

---

## Project layout

```
frontend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI factory + Uvicorn entry point
│   ├── config.py            # Pydantic Settings (env vars / .env file)
│   ├── auth.py              # HTTP Basic Auth + JWT session cookies
│   ├── model_registry.py    # Checkpoint discovery, lazy loading, generation
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── api.py           # REST API endpoints
│   │   └── pages.py         # HTML page routes
│   ├── static/
│   │   ├── css/style.css    # Full light/dark UI styles
│   │   └── js/app.js        # Chat UI logic
│   └── templates/
│       └── index.html       # Jinja2 chat template
├── tests/
│   └── test_api.py          # Pytest async API tests
├── .env.example             # Environment variable template
├── Dockerfile               # Multi-stage build (builder + slim runtime)
├── docker-compose.yml       # Compose file for local and GCE deployment
├── Makefile                 # Developer and deployment commands
├── pyproject.toml           # UV project definition
└── README.md                # This file
```

---

## Quick start (local)

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- At least one trained AttnRes checkpoint (a `*.pt` file) in `../checkpoints/`

### 1 — Install dependencies

```bash
cd frontend/
uv venv
uv pip install -e ".[dev]"
```

### 2 — Configure

```bash
cp .env.example .env
# Edit .env — at minimum change AUTH_PASSWORD and SECRET_KEY
```

### 3 — Run

```bash
make dev
# or:
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080` in your browser. You will be prompted for
the username and password configured in `.env` (defaults: `admin` / `changeme`).

---

## Configuration reference

All settings are read from environment variables prefixed with `ATTNRES_`,
or from a `.env` file in the working directory.

| Variable | Default | Description |
|---|---|---|
| `ATTNRES_HOST` | `0.0.0.0` | Server bind address |
| `ATTNRES_PORT` | `8080` | Server port |
| `ATTNRES_WORKERS` | `1` | Uvicorn worker count (keep 1 for GPU) |
| `ATTNRES_LOG_LEVEL` | `info` | Uvicorn log level |
| `ATTNRES_CHECKPOINTS_DIR` | `../checkpoints` | Directory scanned for `*.pt` files |
| `ATTNRES_ATTNRES_SRC_DIR` | `..` | Root of the attnres source tree |
| `ATTNRES_DEVICE` | `auto` | PyTorch device (`auto`, `cpu`, `cuda`) |
| `ATTNRES_MAX_NEW_TOKENS_LIMIT` | `1024` | Hard ceiling on token generation |
| `ATTNRES_AUTH_USERNAME` | `admin` | HTTP Basic Auth username |
| `ATTNRES_AUTH_PASSWORD` | `changeme` | Plain-text password (use hash in prod) |
| `ATTNRES_AUTH_PASSWORD_HASH` | `` | bcrypt hash — overrides plain password |
| `ATTNRES_SECRET_KEY` | *(insecure default)* | JWT signing secret |
| `ATTNRES_SESSION_EXPIRE_MINUTES` | `480` | Session cookie lifetime (8 h) |

### Generating a secure password hash

```bash
make hash-password Password=mysecret
# or:
python -c "from passlib.hash import bcrypt; print(bcrypt.hash('mysecret'))"
```

Paste the resulting `$2b$12$…` string into `ATTNRES_AUTH_PASSWORD_HASH` and
leave `ATTNRES_AUTH_PASSWORD` empty.

### Generating a secure secret key

```bash
openssl rand -hex 32
```

---

## Docker

### Build

```bash
make build
# or:
docker build -t attnres-chat:latest -f Dockerfile ..
```

> **Note:** The build context is the **repository root** (`..`), not the
> `frontend/` directory, because the Dockerfile copies `models/`, `utils/`,
> and `dataset/` from the parent project.

### Run locally with Compose

```bash
cp .env.example .env   # edit values
make up
make logs
```

### Stop

```bash
make down
```

---

## Deployment to Google Cloud Compute Engine

### Recommended instance types

| Workload | Instance type | vCPUs | RAM | Notes |
|---|---|---|---|---|
| **CPU inference (small models)** | `n2-standard-4` | 4 | 16 GB | Good for char-level or small BPE models |
| **CPU inference (large models)** | `n2-highmem-8` | 8 | 64 GB | Comfortable for 125M–500M parameter models |
| **GPU inference (recommended)** | `n1-standard-4` + T4 | 4 | 15 GB | ~241 tok/s with KV cache enabled |
| **GPU inference (high throughput)** | `n1-standard-8` + A100 | 8 | 30 GB | Optimal for concurrent users |

Use **spot / preemptible** instances to reduce cost by ~60–80% for dev/demo
deployments where occasional interruptions are acceptable.

**Disk:** Attach at least a 50 GB SSD persistent disk for the OS + Docker
images. Mount a separate disk at `/mnt/checkpoints` for model files.

### Step-by-step GCE deployment

#### 1 — Prerequisites

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

#### 2 — Create the VM

```bash
gcloud compute instances create attnres-vm \
  --zone=us-central1-a \
  --machine-type=n2-standard-4 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --tags=http-server \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

For a GPU instance:

```bash
gcloud compute instances create attnres-vm-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --tags=http-server \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

#### 3 — Open firewall port

```bash
gcloud compute firewall-rules create allow-attnres \
  --allow=tcp:8080 \
  --target-tags=http-server \
  --description="Allow AttnRes Chat on port 8080"
```

#### 4 — Install Docker on the VM

```bash
gcloud compute ssh attnres-vm --zone=us-central1-a --command="
  sudo apt-get update -q &&
  sudo apt-get install -y docker.io &&
  sudo systemctl enable --now docker &&
  sudo usermod -aG docker \$USER
"
```

#### 5 — Create the Artifact Registry repository (once)

```bash
GCP_PROJECT=my-project GCP_REGION=us-central1 make create-registry
```

#### 6 — Upload checkpoints to the VM

```bash
# Option A: copy from local machine
gcloud compute scp --recurse ../checkpoints/ \
  attnres-vm:/mnt/checkpoints --zone=us-central1-a

# Option B: copy from GCS bucket
gcloud compute ssh attnres-vm --zone=us-central1-a --command="
  gsutil -m cp -r gs://your-bucket/checkpoints /mnt/
"
```

#### 7 — Create the production .env on the VM

```bash
gcloud compute ssh attnres-vm --zone=us-central1-a --command="
  sudo mkdir -p /etc/attnres &&
  sudo tee /etc/attnres/.env <<'EOF'
ATTNRES_AUTH_USERNAME=admin
AUTH_PASSWORD=admin123
ATTNRES_SECRET_KEY=$(openssl rand -hex 32)
ATTNRES_CHECKPOINTS_DIR=/mnt/checkpoints
ATTNRES_DEVICE=auto
EOF
  sudo chmod 600 /etc/attnres/.env
  sudo chown \$USER /etc/attnres/.env
"
```

#### 8 — Build, push, and deploy

```bash
GCP_PROJECT=my-project \
GCP_REGION=us-central1 \
GCE_INSTANCE=attnres-vm \
GCE_ZONE=us-central1-a \
make deploy
```

This runs `make push` (build + tag + push to Artifact Registry) followed by
an SSH command that pulls the new image and restarts the container on the VM.

#### 9 — Verify

```bash
# Get the external IP
gcloud compute instances describe attnres-vm \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Visit http://<EXTERNAL_IP>:8080
# You will be prompted for username and password.
```

---

## Adding TLS (HTTPS) — recommended for production

Run [Caddy](https://caddyserver.com/) as a reverse proxy in front of the
container. Caddy automatically provisions and renews Let's Encrypt certificates.

```bash
# On the VM
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy

# /etc/caddy/Caddyfile
sudo tee /etc/caddy/Caddyfile <<'EOF'
your-domain.com {
    reverse_proxy localhost:8080
}
EOF

sudo systemctl reload caddy
```

Then open port 443 in the GCE firewall and set `ATTNRES_SESSION_EXPIRE_MINUTES`
to a longer value if desired. Set `secure=True` in `app/auth.py`
`set_session_cookie()` once TLS is active.

---

## Makefile targets

```
  install              Install dependencies into a local venv using uv
  dev                  Run FastAPI dev server (hot-reload, no Docker)
  test                 Run the test suite
  lint                 Run the ruff linter
  hash-password        Generate a bcrypt hash: Password=mysecret make hash-password
  build                Build the Docker image
  up                   Start via Docker Compose (reads .env)
  down                 Stop Docker Compose
  logs                 Follow container logs
  shell                Open a shell inside the running container
  auth-registry        Authenticate Docker with Artifact Registry
  create-registry      Create the Artifact Registry repository (run once)
  push                 Tag and push image to Artifact Registry
  deploy               Build, push, and deploy to GCE via SSH
  gce-logs             Stream logs from the remote GCE container
  gce-shell            Open a shell on the GCE instance
  help                 Show this help
```

---

## API reference

### `GET /api/models`

Returns a list of discovered checkpoint files.

```json
[
  {
    "model_id": "best",
    "name": "best",
    "val_loss": 2.1842,
    "val_ppl": 8.89,
    "epoch": 3,
    "params": 124439040,
    "params_fmt": "124.4 M",
    "architecture": "AttnResLM (Block AttnRes, XSA)",
    "dataset": "tinystories"
  }
]
```

### `POST /api/generate`

```json
{
  "model_id": "best",
  "prompt": "Once upon a time",
  "max_new_tokens": 200,
  "temperature": 0.8,
  "top_k": 40,
  "use_kv_cache": true
}
```

Response:

```json
{
  "prompt": "Once upon a time",
  "generated": "Once upon a time there was a little girl...",
  "continuation": " there was a little girl...",
  "new_tokens": 198,
  "elapsed_s": 0.821,
  "tok_per_sec": 241.2,
  "ms_per_tok": 4.1,
  "model_id": "best",
  "use_kv_cache": true
}
```

### `GET /api/version`

```json
{"version": "0.1.0"}
```

---

## Authentication notes

- HTTP Basic Auth credentials are verified on every login attempt.
- A successful login issues a signed JWT session cookie (default lifetime: 8 h).
- The browser's native login dialog appears automatically on the first visit.
- Sending a `GET /logout` request clears the session cookie.
- Always use a **bcrypt hash** for the password in production:
  ```bash
  make hash-password Password=mypassword
  # → $2b$12$…
  # Set ATTNRES_AUTH_PASSWORD_HASH to this value
  ```
- Always set `ATTNRES_SECRET_KEY` to a random 32-byte hex string:
  ```bash
  openssl rand -hex 32
  ```
