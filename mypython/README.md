# tiny-gpt

A **decoder-only GPT** you can train from scratch on a Mac — in about **30 seconds**.
Same architecture as GPT-2 (token + position embeddings → causal self-attention
blocks → next-token prediction), just shrunk down so it fits a laptop.

It uses the **Apple GPU (MPS)** automatically, falling back to CUDA and then CPU.
The data pipeline downloads an open dataset, converts it to JSONL, and trains
from JSONL. Both **training** and **inference** are included.

> Reality check: a ~0.8M-parameter model trained for 30s learns the *shape* of
> the text (character names, line breaks, punctuation) but not real language.
> It's a working scale-model of GPT — perfect for *learning how it works*, not a
> usable assistant. See [How small is "tiny"?](#how-small-is-tiny).

---

## Quick start

```bash
make setup        # create .venv and install torch + numpy
make data         # download TinyShakespeare -> data/corpus.jsonl
make train        # ~30s on Apple silicon (the 'tiny' preset)
make generate     # sample text from the trained model
```

That's the whole loop: **setup → data → train → generate**.

### Reading the training log

A live progress bar updates in place every iteration, with periodic eval lines
above it:

```
iter   500 [ 83.5%] | train 1.55 | val 1.70 | learned 60.1% |  18.7s
  [███████████████░░░░░░░░░]  62.4%  loss 1.683   54s  ETA   33s
```

- The **bar** shows iteration progress, the latest batch loss, elapsed time, and ETA.
- **`learned %`** = `(baseline − val_loss) / baseline × 100`, where the random-guess
  baseline is `ln(vocab_size) ≈ 4.17`. A fresh model reads **~0%**; the `tiny`
  preset reaches **~50%**, `small` goes higher. It's an intuition meter, not a
  formal accuracy — natural text has irreducible entropy, so 100% is unreachable.

---

## What's inside

| File           | Role                                                              |
|----------------|-------------------------------------------------------------------|
| `model.py`     | The decoder-only GPT (attention, MLP, blocks, sampling).          |
| `train.py`     | Training loop: batching, LR warmup+cosine decay, eval, checkpoint.|
| `generate.py`  | Load a checkpoint and autoregressively sample text.               |
| `data.py`      | Download the open dataset → `data/corpus.jsonl`.                  |
| `dataset.py`   | Char-level tokenizer + JSONL loader.                              |
| `config.py`    | `GPTConfig` + the `tiny` / `small` training presets.             |
| `utils.py`     | Device selection (MPS → CUDA → CPU).                             |
| `Makefile`     | One-liners for every step.                                        |

---

## Choosing how long it trains

Two independent knobs:

**1. Model/training size — the preset:**

| Preset  | Params | Layers | Width | Context | Iters | Wall-clock (MPS) |
|---------|--------|--------|-------|---------|-------|------------------|
| `tiny`  | ~0.8M  | 4      | 128   | 128     | 600   | **~30 sec**      |
| `small` | ~10M   | 6      | 384   | 256     | 3000  | **~10 min**      |

```bash
make train                 # tiny
make train PRESET=small    # small
make train PRESET=small --max-iters 1000   # or override iterations directly
```

**2. Dataset size — the `SIZE` fraction** (0–1, how much of the corpus to keep):

```bash
make data SIZE=0.25        # keep first 25% — faster, smaller vocab
make data                  # full 1.1 MB corpus (default)
```

---

## The data flow (download → JSONL → train)

1. `data.py` downloads **TinyShakespeare** (`input.txt`, ~1.1 MB, public domain).
2. It splits the text on blank lines and writes one record per passage:
   ```json
   {"text": "ROMEO:\nIs the day so young?"}
   {"text": "BENVOLIO:\nBut new struck nine."}
   ```
   → `data/corpus.jsonl`
3. `train.py` reads the JSONL, joins the `text` fields, builds a **character-level
   vocabulary** from it, and trains.

Want a different corpus? Point `data.py` at another source (or hand-write any
`{"text": ...}` JSONL into `data/corpus.jsonl`) and everything else just works.

---

## Generating text

```bash
make generate                                   # prompt "ROMEO:", 500 tokens
make generate PROMPT="JULIET:" TOKENS=300
.venv/bin/python generate.py --prompt "To be" --temperature 0.7 --top-k 100
```

- `--temperature` — lower (0.5) = safe/repetitive, higher (1.0) = wilder.
- `--top-k` — sample only from the k most likely next characters.

---

## How small is "tiny"?

| | this `tiny` model | GPT-2 small | GPT-4-class |
|---|---|---|---|
| Parameters | ~0.8M | 124M | ~1T+ |
| Training data | ~1 MB | ~40 GB | trillions of tokens |
| Train time | 30 s, 1 laptop | days, 8×A100 | months, huge cluster |

The **architecture is the same idea** at every scale — what changes is
parameters × data × compute. This repo lets you watch the *whole* mechanism
(tokenize → embed → attend → predict → sample) run end-to-end, fast.

---

## Architecture (decoder-only, in one breath)

```
tokens ──► token embedding + position embedding
       ──► N × [ LayerNorm → causal self-attention → +residual
                 LayerNorm → MLP (4× GELU)          → +residual ]
       ──► final LayerNorm ──► linear head (weights tied to embedding)
       ──► softmax over vocabulary ──► sample next token ──► repeat
```

"Decoder-only" = the causal mask in `CausalSelfAttention` lets each position
attend only to itself and earlier positions. That's what makes it an
autoregressive next-token predictor.

---

## Requirements

- Python 3.9+
- PyTorch ≥ 2.1 (installed into `.venv` by `make setup`)
- Apple silicon Mac (uses MPS) — or any CUDA GPU, or plain CPU.

## Reset

```bash
make clean        # removes out/, downloaded data, and caches
```
