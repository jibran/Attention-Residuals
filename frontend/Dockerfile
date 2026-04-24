# ── Stage 1: build dependencies ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution and wheel building
RUN pip install --no-cache-dir uv

# Copy dependency manifests AND the files hatchling needs to build the wheel
COPY frontend/pyproject.toml frontend/README.md ./
COPY frontend/app/ ./app/

# Create a virtual environment and install all dependencies into it.
# torch is large; --no-cache keeps the layer small.
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache ".[dev]" && \
    uv pip install --no-cache \
    torch==2.3.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Stage 2: pre-download HuggingFace assets ──────────────────────────────
# Run as root so we can write to /hf_cache, then lock it down afterwards.
FROM python:3.11-slim AS hf_cache

ENV PATH="/opt/venv/bin:$PATH"
COPY --from=builder /opt/venv /opt/venv

# Download the GPT-Neo-125M tokenizer (used for TinyStories checkpoints).
# Baking it into the image means the runtime container never needs network
# access or a writable HuggingFace home directory.
ENV HF_HOME=/hf_cache
RUN python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
print("Tokenizer cached at /hf_cache")
PY

# ── Stage 3: runtime image ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AttnRes Chat Frontend"
LABEL org.opencontainers.image.description="FastAPI chat UI for AttnRes language model inference"
LABEL org.opencontainers.image.source="https://github.com/jibran/attnres"

# Non-root user for security
RUN addgroup --system attnres && adduser --system --ingroup attnres attnres

# Copy the venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the pre-downloaded HuggingFace tokenizer cache and make it readable
# by the non-root user.  TRANSFORMERS_OFFLINE=1 prevents any attempt to
# reach the network even when the files are already present.
COPY --from=hf_cache --chown=attnres:attnres /hf_cache /hf_cache
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

WORKDIR /app

# Copy the frontend application code into the runtime layer
COPY --chown=attnres:attnres frontend/app/ ./app/

# Copy the attnres model source tree so the frontend can import models/, utils/, etc.
COPY --chown=attnres:attnres models ./models/
COPY --chown=attnres:attnres utils  ./utils/
COPY --chown=attnres:attnres dataset ./dataset/
COPY --chown=attnres:attnres inference ./inference/

# Volume for checkpoints (mounted at runtime)
VOLUME /checkpoints

# Port
EXPOSE 8080

USER attnres

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/version')" || exit 1

ENTRYPOINT ["python", "-m", "app.main"]
