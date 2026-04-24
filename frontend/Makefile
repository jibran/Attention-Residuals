# ── AttnRes Chat — Makefile ────────────────────────────────────────────────
#
# Usage:
#   make dev          Start the server locally (no Docker)
#   make build        Build the Docker image
#   make up           Start via Docker Compose
#   make down         Stop Docker Compose
#   make push         Push image to Google Artifact Registry
#   make deploy       Full build → push → SSH deploy to GCE
#   make logs         Follow container logs
#   make test         Run the test suite
#   make lint         Run ruff linter
#   make hash-password Password=$(p) make hash-password
#
# Required environment variables for GCE targets:
#   GCP_PROJECT       Google Cloud project ID
#   GCP_REGION        Artifact Registry region  (e.g. us-central1)
#   GCE_INSTANCE      GCE instance name         (e.g. attnres-vm)
#   GCE_ZONE          GCE instance zone         (e.g. us-central1-a)
#
# Optional:
#   IMAGE_TAG         Docker image tag           (default: latest)
#   PORT              Host port to expose        (default: 8080)

# ── Configuration ──────────────────────────────────────────────────────────
IMAGE_NAME   ?= attnres-chat
IMAGE_TAG    ?= latest
PORT         ?= 8080
GCP_PROJECT  ?= attnres
GCP_REGION   ?= us-central1
GCE_INSTANCE ?= attnres-vm
GCE_ZONE     ?= us-central1-a

REGISTRY  = $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/attnres
FULL_IMG  = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# ── Local development ──────────────────────────────────────────────────────

.PHONY: install
install:  ## Install dependencies into a local venv using uv
	uv venv && uv pip install -e ".[dev]"

.PHONY: dev
dev:  ## Run FastAPI dev server (hot-reload, no Docker)
	ATTNRES_LOG_LEVEL=debug uvicorn app.main:app \
		--host 0.0.0.0 --port $(PORT) --reload

.PHONY: test
test:  ## Run the test suite
	pytest -v --tb=short tests/

.PHONY: lint
lint:  ## Run the ruff linter
	ruff check app/ tests/

.PHONY: hash-password
hash-password:  ## Generate a bcrypt hash: Password=mysecret make hash-password
	python -c "from passlib.hash import bcrypt; print(bcrypt.hash('$(Password)'))"

# ── Docker ─────────────────────────────────────────────────────────────────

.PHONY: build
build:  ## Build the Docker image
	docker build \
		--build-arg IMAGE_TAG=$(IMAGE_TAG) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f Dockerfile \
		..

.PHONY: up
up:  ## Start via Docker Compose (reads .env)
	IMAGE_NAME=$(IMAGE_NAME) IMAGE_TAG=$(IMAGE_TAG) PORT=$(PORT) \
	docker compose up -d

.PHONY: down
down:  ## Stop Docker Compose
	docker compose down

.PHONY: logs
logs:  ## Follow container logs
	docker compose logs -f attnres-chat

.PHONY: shell
shell:  ## Open a shell inside the running container
	docker compose exec attnres-chat /bin/bash

# ── Google Artifact Registry ───────────────────────────────────────────────

.PHONY: auth-registry
auth-registry:  ## Authenticate Docker with Artifact Registry
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev --quiet

.PHONY: create-registry
create-registry:  ## Create the Artifact Registry repository (run once)
	gcloud artifacts repositories create attnres \
		--repository-format=docker \
		--location=$(GCP_REGION) \
		--project=$(GCP_PROJECT) \
		--description="AttnRes Chat container images"

.PHONY: push
push: build auth-registry  ## Tag and push image to Artifact Registry
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(FULL_IMG)
	docker push $(FULL_IMG)
	@echo "✓ Pushed $(FULL_IMG)"

# ── GCE deployment ─────────────────────────────────────────────────────────

.PHONY: deploy
deploy: push  ## Build, push, and deploy to GCE via SSH
	@echo "── Deploying to $(GCE_INSTANCE) ($(GCE_ZONE)) ──"
	gcloud compute ssh $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) \
		--project=$(GCP_PROJECT) \
		--command=" \
			gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev --quiet && \
			docker pull $(FULL_IMG) && \
			docker stop attnres-chat 2>/dev/null || true && \
			docker rm   attnres-chat 2>/dev/null || true && \
			docker run -d \
				--name attnres-chat \
				--restart unless-stopped \
				-p $(PORT):8080 \
				-v /mnt/checkpoints:/checkpoints:ro \
				--env-file /etc/attnres/.env \
				$(FULL_IMG) \
		"
	@echo "✓ Deployed. Visit http://\$$(gcloud compute instances describe $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) --format='get(networkInterfaces[0].accessConfigs[0].natIP)'):$(PORT)"

.PHONY: gce-logs
gce-logs:  ## Stream logs from the remote GCE container
	gcloud compute ssh $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) \
		--project=$(GCP_PROJECT) \
		--command="docker logs -f attnres-chat"

.PHONY: gce-shell
gce-shell:  ## Open a shell on the GCE instance
	gcloud compute ssh $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) \
		--project=$(GCP_PROJECT)

.PHONY: gce-stop
gce-stop:  ## Stop the container on GCE
	gcloud compute ssh $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) \
		--project=$(GCP_PROJECT) \
		--command="docker stop attnres-chat"

.PHONY: gce-start
gce-start:  ## Start the container on GCE (assumes image is already pulled)
	gcloud compute ssh $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) \
		--project=$(GCP_PROJECT) \
		--command="docker start attnres-chat"

.PHONY: gce-remove
gce-remove:  ## Remove the container on GCE
	gcloud compute ssh $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) \
		--project=$(GCP_PROJECT) \
		--command="docker rm attnres-chat"

.PHONY: gce-run-app
gce-run-app:  ## Run the app on GCE (assumes image is already pulled)
	gcloud compute ssh $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) \
		--project=$(GCP_PROJECT) \
		--command=" \
			docker run -d \
				--name attnres-chat \
				--restart unless-stopped \
				-p $(PORT):8080 \
				-v /mnt/checkpoints:/checkpoints:ro \
				--env-file /etc/attnres/.env \
				$(FULL_IMG) \
		"

.PHONY: gce-turn-off
gce-turn-off:  ## Stop the GCE instance (not just the container)
	gcloud compute instances stop $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) \
		--project=$(GCP_PROJECT)

.PHONY: gce-turn-on
gce-turn-on:  ## Start the GCE instance (not just the container)
	gcloud compute instances start $(GCE_INSTANCE) \
		--zone=$(GCE_ZONE) \
		--project=$(GCP_PROJECT)

# ── Help ───────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
