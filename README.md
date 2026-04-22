# AttnRes — Attention Residuals in PyTorch

A clean, reproducible PyTorch implementation of
**Attention Residuals (AttnRes)** from MoonshotAI
([paper](https://arxiv.org/abs/2603.15031) · [repo](https://github.com/MoonshotAI/Attention-Residuals)).

AttnRes replaces standard fixed residual connections with learned,
input-dependent softmax attention over preceding layer outputs — giving
every transformer layer selective access to *all* earlier representations.

Two tasks are supported out of the box:

| Task | Dataset | Model | Script |
|---|---|---|---|
| Image classification | MNIST, CIFAR-10 | `AttnResTransformer` | `train/train.py` |
| Character-level LM | Tiny-Shakespeare | `AttnResLM` | `train/train_lm.py` |

---

## Project layout

```
attnres/
├── checkpoints/
│   ├── best/                    # Best validation-loss checkpoint per run
│   └── latest/                  # Most recent checkpoint (resume training)
├── config/
│   ├── base.yaml                # Image classification (Block AttnRes, MNIST)
│   ├── full_attnres.yaml        # Image classification (Full AttnRes, MNIST)
│   └── shakespeare.yaml         # Character-level LM on Tiny-Shakespeare
├── data/
│   ├── raw/                     # Downloaded / unprocessed data
│   └── processed/               # Cached corpus + vocab.json (auto-generated)
├── dataset/
│   ├── base_dataset.py          # Abstract base + DataLoader factory
│   ├── image_datasets.py        # MNIST and CIFAR-10 wrappers
│   ├── shakespeare_dataset.py   # Tiny-Shakespeare HF dataset + window sampler
│   └── tokenizer.py             # Character-level tokeniser (save/load JSON)
├── inference/
│   ├── inference.py             # Image model: load checkpoint → predict
│   └── inference_lm.py          # LM: generate text / evaluate perplexity
├── logs/                        # CSV training logs  (model_name-YYYYMMDDHHMM.csv)
├── models/
│   ├── components.py            # RMSNorm, SwiGLU, RotaryEmbedding, CausalSelfAttention
│   ├── attn_res.py              # FullAttnResOp, BlockAttnResOp, AttnResTransformerLayer
│   ├── transformer.py           # AttnResTransformer + BaselineTransformer (vision)
│   └── lm_transformer.py        # AttnResLM + BaselineLM (language model)
├── notebooks/                   # Colab-ready .ipynb files
├── tests/
│   ├── test_components.py       # RMSNorm, SwiGLU, RoPE, attention unit tests
│   ├── test_attn_res.py         # FullAttnResOp, BlockAttnResOp, layer tests
│   ├── test_transformer.py      # Vision model integration tests
│   ├── test_shakespeare.py      # Tokeniser, window dataset, LM model tests
│   └── test_utils.py            # Config, logger, checkpoint, device tests
├── train/
│   ├── train.py                 # Vision model training entry-point
│   └── train_lm.py              # Language model training entry-point
├── utils/
│   ├── config.py                # Typed dataclasses + YAML loader + CLI overrides
│   ├── logger.py                # Timestamped CSV logger + rich console output
│   ├── checkpoint.py            # best/ and latest/ checkpoint manager
│   └── device.py                # Device resolution + seed_everything
├── visualization/
│   ├── plot_logs.py             # Single-run loss / accuracy curves
│   └── compare_models.py        # Multi-model overlay comparison
├── pyproject.toml               # uv / hatchling project definition
└── README.md
```

---

## Quick start

### 1 — Install [uv](https://github.com/astral-sh/uv)

```bash
pip install uv        # or: curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2 — Create the virtual environment and install dependencies

```bash
cd attnres
uv venv               # creates .venv/
uv pip install -e ".[dev]"
```

### 3 — Run tests

```bash
pytest                # 126 tests, all pass
```

---

## Models

All model classes live in `models/`. The architecture is built from composable
layers: shared building blocks → core AttnRes operations → full model assemblies.

### Building blocks — `components.py`

![Building blocks](assets/components.svg)

| Class | Role |
|---|---|
| `RMSNorm(dim)` | `x / rms(x) · weight` — no mean subtraction, no bias. Cheaper than LayerNorm and equally effective at depth. |
| `SwiGLU(dim, hidden_dim)` | `down(silu(gate(x)) ⊙ up(x))` — gated FFN. Hidden dim defaults to `dim × 8/3`. No bias on any projection. |
| `RotaryEmbedding(head_dim)` | RoPE position encoding. No learned parameters. Norm-preserving rotation applied to Q and K. Cache rebuilt automatically for longer sequences. |
| `CausalSelfAttention(dim, heads, head_dim)` | Multi-head attention with RoPE. Uses `scaled_dot_product_attention(is_causal=True)` — Flash Attention when available via PyTorch 2.x. No bias on QKV or output projection. |

---

### Full Attention Residuals — `attn_res.py` · `FullAttnResOp`

![Full AttnRes](assets/full_attn_res.svg)

Replaces the fixed residual `h_l = h_{l-1} + f(h_{l-1})` with softmax attention
over **all** previous layer outputs:

```
h_l = Σ_{i=0}^{l-1}  α_{i→l} · v_i

α_{i→l} = softmax_i( w_l^T · RMSNorm(v_i) )
```

- `v_i` are the previous sublayer outputs (embedding + all Attn/MLP outputs so far)
- `w_l ∈ ℝ^d` is a single **learned** pseudo-query per layer — the only added parameter
- `RMSNorm` on the keys prevents large-magnitude layers from dominating
- Softmax ensures weights sum to 1 → bounded output magnitudes at any depth
- Memory: **O(L·d)** — overlaps with backprop activations in vanilla training, so zero overhead

---

### Block Attention Residuals — `attn_res.py` · `BlockAttnResOp`

![Block AttnRes](assets/block_attn_res.svg)

The practical, scale-efficient variant. Instead of attending over all L sublayer
outputs, layers are grouped into **N ≈ 8 blocks**. Standard residuals accumulate
within each block into a `partial_block` sum. The AttnRes op attends only over
the N completed block summaries plus the current partial:

```
all_reps = [b₀, b₁, …, b_{N-1}, partial_block]   # N+1 tensors
h = Σ α_i · all_reps_i
```

Block boundaries trigger when `(layer_num × 2) % block_size == 0`, at which
point `partial_block` is pushed to `block_reps` and reset to zeros.

| | Full AttnRes | Block AttnRes |
|---|---|---|
| Attends over | All L sublayer outputs | N block summaries + partial |
| Memory | O(L·d) | O(N·d) |
| Inference overhead | — | < 2% |
| Compute budget match | baseline | 1.25× baseline |

Both `FullAttnResOp` and `BlockAttnResOp` share the same interface inside
`AttnResTransformerLayer`, selected via `use_block_attn_res` in the config.

---

### Vision transformer — `transformer.py`

![AttnResTransformer](assets/attnres_transformer.svg)

**`AttnResTransformer`** — image classification with AttnRes.

```
(B, C, H, W)
  → PatchEmbedding  (split image → patches → Linear projection)
  → + LearnedPositionalEmbedding
  → [AttnResTransformerLayer × depth]
      ├─ BlockAttnResOp → RMSNorm → CausalSelfAttention
      └─ BlockAttnResOp → RMSNorm → SwiGLU
  → sum(block_reps[1:], partial_block)
  → mean-pool(dim=1) → RMSNorm
  → Linear(dim, num_classes)
  → logits  (B, num_classes)
```

**`BaselineTransformer`** — identical pipeline but with standard fixed residuals
(`h = h + attn(norm(h)); h = h + mlp(norm(h))`). Used as the ablation baseline.

**`PatchEmbedding`** — splits a `(B, C, H, W)` image into `(H/p × W/p)` non-overlapping
patches of size `p×p`, flattens each to a vector of length `C·p²`, and projects
to `dim` via a bias-free `nn.Linear`.

| Dataset | img_size | patch_size | n_patches |
|---|---|---|---|
| MNIST | 28 | 4 | 49 |
| CIFAR-10 | 32 | 4 | 64 |

---

### Language model — `lm_transformer.py`

![AttnResLM](assets/attnres_lm.svg)

**`AttnResLM`** — autoregressive character-level language model with AttnRes.

```
(B, T)  token ids
  → Embedding(vocab_size, dim)  +  Embedding(seq_len, dim)  [positional]
  → [AttnResTransformerLayer × depth]
      ├─ BlockAttnResOp → RMSNorm → CausalSelfAttention
      └─ BlockAttnResOp → RMSNorm → SwiGLU
  → sum(block_reps[1:], partial_block)
  → RMSNorm                          # per-position, no pooling
  → Linear(dim, vocab_size)          # weight-tied with tok_embed
  → logits  (B, T, vocab_size)
```

Key design choices:

- **Weight tying** — `head.weight = tok_embed.weight`. Reduces parameter count and
  empirically improves generalisation on small corpora.
- **No CLS token / mean pooling** — logits are produced at every position for
  next-token prediction (cross-entropy over all T positions per batch).
- **`generate(prompt_ids, max_new_tokens, temperature, top_k)`** — autoregressive
  sampling loop with temperature scaling and optional top-k logit filtering.

**`BaselineLM`** — identical architecture with standard fixed residuals.
Used for ablation comparisons on Shakespeare perplexity.

---

## Image classification

Train an AttnRes vision transformer on MNIST or CIFAR-10.

### Train

```bash
# Block AttnRes on MNIST (default):
python train/train.py --config config/base.yaml

# Full AttnRes on MNIST:
python train/train.py --config config/full_attnres.yaml

# Standard-residual baseline for comparison:
python train/train.py --config config/base.yaml --baseline

# CIFAR-10:
python train/train.py --config config/base.yaml --override data.dataset=cifar10

# Resume an interrupted run:
python train/train.py --config config/base.yaml --resume
```

### Inference

```bash
# Evaluate on the test set:
python inference/inference.py \
    --checkpoint checkpoints/best/best.pt \
    --eval

# Predict a single image:
python inference/inference.py \
    --checkpoint checkpoints/best/best.pt \
    --input data/sample.png
```

### Vision config reference

```yaml
model:
  name: "AttnResTransformer"
  dim: 256
  depth: 8
  heads: 4
  head_dim: 64
  block_size: 4          # sublayers per AttnRes block
  use_block_attn_res: true

training:
  epochs: 20
  batch_size: 64
  lr: 3e-4
  weight_decay: 1e-2
  grad_clip: 1.0
  warmup_steps: 200
  log_every: 50
  save_every: 1

data:
  dataset: "mnist"       # "mnist" | "cifar10"
  num_workers: 4
```

---

## Language model — Tiny-Shakespeare

Trains a character-level autoregressive language model on the
[Trelis/tiny-shakespeare](https://huggingface.co/datasets/Trelis/tiny-shakespeare)
dataset from Hugging Face Hub.

### Dataset

| Property | Value |
|---|---|
| Source | `Trelis/tiny-shakespeare` (HF Hub) |
| Format | CSV — one Shakespeare passage per row (`Text` column) |
| Splits | 472 train rows · 49 test rows |
| Corpus size | ~1 M characters after concatenation |
| Vocabulary | ~67 unique ASCII characters |
| Tokenisation | Character-level (no BPE) |
| Cached to | `data/processed/shakespeare_corpus.txt` + `data/processed/vocab.json` |

The first run downloads and caches automatically; subsequent runs load from
disk with no network access required.

### How it works

The corpus is concatenated into a single string, tokenised character by
character, then sliced into overlapping fixed-length windows:

```
tokens:  [ t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  … ]
window 0: [ t0 … t255 ]  →  targets [ t1 … t256 ]
window 1: [ t128 … t383 ]  →  targets [ t129 … t384 ]   (stride = 128)
```

Each window is an `(x, y)` pair where `y = x` shifted one position left —
the standard next-token prediction objective. Cross-entropy loss over all
positions drives the model to predict the next character given the context.

### Model architecture

```
token ids (B, T)
    │
    ▼
Embedding(vocab_size, dim)  +  LearnedPositionalEmbedding(seq_len, dim)
    │
    ▼  × depth
AttnResTransformerLayer
  ├─ BlockAttnResOp  →  CausalSelfAttention (RoPE)
  └─ BlockAttnResOp  →  SwiGLU MLP
    │
    ▼
RMSNorm  →  Linear(dim, vocab_size)   [weight-tied with embedding]
    │
    ▼
logits (B, T, vocab_size)
```

Weight tying means the input embedding matrix and the output projection share
the same parameters — reducing parameter count and improving generalisation.

### Train

```bash
# Block AttnRes (default, most efficient):
python train/train_lm.py --config config/shakespeare.yaml

# Full AttnRes (attends over all previous layer outputs):
python train/train_lm.py --config config/shakespeare.yaml \
    --override model.use_block_attn_res=false

# Standard-residual baseline:
python train/train_lm.py --config config/shakespeare.yaml --baseline

# Resume from latest checkpoint:
python train/train_lm.py --config config/shakespeare.yaml --resume

# Custom generation prompt shown after each epoch:
python train/train_lm.py --config config/shakespeare.yaml \
    --prompt "KING LEAR: " --max_new_tokens 400
```

A generation sample is printed after every epoch so you can watch the model
learn to produce Shakespeare-like text in real time.

### Inference & text generation

```bash
# Generate 500 characters from the best checkpoint:
python inference/inference_lm.py \
    --checkpoint checkpoints/best/best.pt \
    --prompt "HAMLET: To be, or not to be"

# Adjust sampling parameters:
python inference/inference_lm.py \
    --checkpoint checkpoints/best/best.pt \
    --prompt "ROMEO: " \
    --max_new_tokens 800 \
    --temperature 0.6 \
    --top_k 50

# Evaluate test-set perplexity:
python inference/inference_lm.py \
    --checkpoint checkpoints/best/best.pt \
    --eval
```

### LM config reference

```yaml
model:
  name: "AttnResLM_Shakespeare"
  dim: 256
  depth: 6
  heads: 4
  head_dim: 64
  mlp_multiplier: 4
  dropout: 0.1
  use_block_attn_res: true
  block_size: 4
  max_seq_len: 256         # context window in characters

training:
  epochs: 20
  batch_size: 64
  lr: 3.0e-4
  weight_decay: 1.0e-2
  grad_clip: 1.0
  warmup_steps: 300

data:
  dataset: "shakespeare"
  data_dir: "data/"
  seq_len: 256             # characters per training window
  stride: 128              # 50 % overlap → ~2× more windows
  val_split: 0.1

generation:
  max_new_tokens: 500
  temperature: 0.8
  top_k: 40
```

### Sampling parameters

| Parameter | Effect |
|---|---|
| `temperature` | Higher → more random; lower → more greedy. `1.0` = unchanged softmax. |
| `top_k` | Restricts sampling to the `k` most likely next characters. `0` = disabled. |
| `max_new_tokens` | How many characters to generate beyond the prompt. |

---

## Logging

Every training run creates one CSV in `logs/`:

```
logs/<model_name>-<YYYYMMDDHHMM>.csv
```

Columns: `epoch, step, phase, train_loss, train_acc, val_loss, val_acc, lr, elapsed_s`

For language-model runs the `train_acc` and `val_acc` columns store
`1 / perplexity` (a monotone proxy — higher is better). Use the raw `*_loss`
columns to compute perplexity: `ppl = exp(val_loss)`.

---

## Checkpoints

| Path | Contents |
|---|---|
| `checkpoints/latest/latest.pt` | State dict saved every epoch (for resuming) |
| `checkpoints/best/best.pt` | State dict with the lowest validation loss seen so far |

Each `.pt` file contains `model_state`, `optimizer_state`, `scheduler_state`,
`epoch`, `step`, `val_loss`, `val_acc`, and the full `config` dict for
reproducibility.

---

## Visualisation

```bash
# Plot loss + accuracy curves for one run:
python visualization/plot_logs.py \
    --log logs/AttnResLM_Shakespeare-202406011200.csv

# Compare AttnRes vs baseline side by side:
python visualization/compare_models.py \
    --logs logs/AttnResLM_Shakespeare-202406011200.csv \
           logs/BaselineLM-202406011300.csv \
    --out reports/shakespeare_comparison.png

# Also plot training speed:
python visualization/compare_models.py --logs logs/ --speed
```

---

## Citation

```bibtex
@article{attnres2026,
  title  = {Attention Residuals},
  author = {Chen, Guangyu and Zhang, Yu and Su, Jianlin and others},
  year   = {2026},
  url    = {https://github.com/MoonshotAI/Attention-Residuals}
}
```
