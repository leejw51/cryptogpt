# cryptogpt — tiny-gpt, two ways

A **decoder-only GPT** (the GPT-2 architecture, shrunk to fit a laptop) trained
from scratch on TinyShakespeare — implemented **twice**, so you can read the
*same* model two ways:

- [`mypython/`](mypython/) — the reference implementation in **PyTorch**.
- [`myrust/`](myrust/) — a faithful port in **Rust** on the
  [`burn`](https://burn.dev) framework (single static binary, no Python runtime).

Both train a ~0.8M-parameter model in ~30s–1min on Apple silicon, then sample
text from it. The files line up one-for-one (`model.py` ↔ `src/model.rs`,
`train.py` ↔ `src/bin/train.rs`, …), so you can diff the two languages against
the same architecture.

> Reality check: a ~0.8M-parameter model learns the *shape* of the text
> (character names, line breaks, punctuation) but not real language. It's a
> working scale-model of GPT — perfect for *learning how it works*, not a usable
> assistant.

---

## Quick start

### Python (start here)

```bash
cd mypython
make setup        # create .venv and install torch + numpy
make data         # download TinyShakespeare -> data/corpus.jsonl
make train        # ~30s on Apple silicon (the 'tiny' preset)
make generate     # sample text from the trained model
```

### Rust

```bash
# Reuses the dataset produced by the Python side (../data/corpus.jsonl).
# Run `make data` in mypython/ first, or point --data at any JSONL file.
cd myrust
make train                  # CPU backend, 'tiny' preset
make train BACKEND=metal    # Apple GPU (fastest on Mac)
make generate
```

See [`mypython/README.md`](mypython/README.md) and
[`myrust/README.md`](myrust/README.md) for full details (presets, backends,
sampling options, the training log, and how the architecture maps to code).

---

## The data flow

```
data.py downloads TinyShakespeare (input.txt, ~1.1 MB, public domain)
   └─► splits on blank lines into {"text": ...} records  ──►  data/corpus.jsonl
          └─► train.py / train.rs: char-level vocab + next-token training
```

The corpus is shared: Python writes `mypython/data/corpus.jsonl`, and the Rust
side reads it via `--data ../data/corpus.jsonl`. Want a different corpus? Drop
any `{"text": ...}` JSONL in place and everything else just works.

---

## Architecture (decoder-only, in one breath)

```
tokens ──► token embedding + position embedding
       ──► N × [ LayerNorm → causal self-attention → +residual
                 LayerNorm → MLP (4× GELU)          → +residual ]
       ──► final LayerNorm ──► linear head ──► softmax ──► sample ──► repeat
```

"Decoder-only" = a causal mask lets each position attend only to itself and
earlier positions — that's what makes it an autoregressive next-token predictor.

---

## Presets

| Preset  | Params | Layers | Width | Context | Iters | Wall-clock (Python, MPS) |
|---------|--------|--------|-------|---------|-------|--------------------------|
| `tiny`  | ~0.8M  | 4      | 128   | 128     | 600   | **~30 sec**              |
| `small` | ~10M   | 6      | 384   | 256     | 3000  | **~10 min**              |

```bash
make train PRESET=small    # in either mypython/ or myrust/
```

---

## Python vs. Rust — what differs

- **Speed at this scale.** For a model this tiny, PyTorch on MPS is still faster
  (wall-clock is dominated by per-kernel dispatch overhead, not math). burn's
  edge shows up elsewhere: a single static binary, no Python runtime, larger
  models, and targets PyTorch can't reach (WASM, embedded).
- **Backend selection.** Python picks the device at runtime (MPS → CUDA → CPU);
  Rust picks the backend at *compile time* via a Cargo feature
  (`ndarray`/`wgpu`/`metal`).
- **Untied LM head.** PyTorch ties the output head to the token-embedding
  weights; the Rust port uses a separate (untied) `Linear`. Architecture is
  otherwise identical.

---

## Layout

```
cryptogpt/
├── README.md          ← you are here
├── mypython/          PyTorch reference implementation
│   ├── model.py  train.py  generate.py  data.py  dataset.py  config.py  utils.py
│   └── Makefile  README.md  requirements.txt
└── myrust/            Rust / burn port
    ├── src/model.rs  src/bin/{train,generate}.rs  src/dataset.rs  src/config.rs  src/backend.rs
    └── Makefile  README.md  Cargo.toml
```

---

## Requirements

- **Python side:** Python 3.9+, PyTorch ≥ 2.1 (installed by `make setup`).
  Apple silicon (MPS), any CUDA GPU, or plain CPU.
- **Rust side:** a recent Rust toolchain. The `metal` backend needs Apple
  silicon; `wgpu` runs on Metal/Vulkan/DX12.
