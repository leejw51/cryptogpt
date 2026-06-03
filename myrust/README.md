# tiny-gpt (Rust / burn)

A **decoder-only GPT** trained from scratch on TinyShakespeare — the same model
as the companion Python project, ported to **Rust** on the
[`burn`](https://burn.dev) deep-learning framework.

Same architecture as GPT-2 (token + position embeddings → causal self-attention
blocks → next-token prediction), shrunk to fit a laptop. Both **training** and
**inference** are included, and the whole thing is a single statically-linked
binary with no Python runtime.

> Reality check: a ~0.8M-parameter model learns the *shape* of the text
> (character names, line breaks, punctuation) but not real language. It's a
> working scale-model of GPT — perfect for *learning how it works*.

---

## Quick start

```bash
# data is shared with the Python project (../data/corpus.jsonl). If you don't
# have it yet, generate it from the Python side, or point --data at any
# JSONL file of {"text": ...} records.

make train        # train the 'tiny' preset (CPU backend)
make generate     # sample text from the trained model
```

Or drive cargo directly:

```bash
cargo run --release --bin train    -- --preset tiny
cargo run --release --bin generate -- --prompt "ROMEO:" --max-new-tokens 400
```

---

## Backends

Selected at compile time by Cargo features:

| Feature            | Backend                          | Notes                                  |
|--------------------|----------------------------------|----------------------------------------|
| `ndarray` (default)| CPU                              | Fast to build, runs anywhere.          |
| `wgpu`             | Metal / Vulkan / DX12 GPU        | Portable GPU; includes kernel fusion.  |
| `metal`            | Apple Metal runtime              | Fastest on Apple silicon.              |

```bash
# default = CPU
cargo run --release --bin train -- --preset tiny

# portable GPU
cargo run --release --no-default-features --features wgpu  --bin train -- --preset small

# Apple silicon (fastest)
cargo run --release --no-default-features --features metal --bin train -- --preset small
```

Or via the Makefile: `make train BACKEND=metal`.

### A note on speed vs. PyTorch

For a model this **tiny**, PyTorch on MPS is still faster (~22s vs ~47s for the
`tiny` preset on an Apple laptop). The matmuls finish in microseconds, so
wall-clock is dominated by *per-kernel dispatch overhead*, not math — and
PyTorch's MPS kernels are more mature. Two things narrow the gap a lot here:

- **Kernel fusion** (`fusion` feature, on by default for `wgpu`/`metal`) merges
  many small element-wise ops into fewer GPU dispatches.
- **The dedicated `metal` runtime** beats the generic wgpu path on Apple
  silicon (~93s → ~47s in our measurements).

burn's real advantages show up *elsewhere*: a single static binary with no
Python runtime, larger models where the GPU is actually saturated, and targets
PyTorch can't reach (WASM, embedded). At this scale, treat it as a faithful,
self-contained reimplementation rather than a speed play.

---

## What's inside

| File                | Role                                                          |
|---------------------|---------------------------------------------------------------|
| `src/model.rs`      | The decoder-only GPT (attention, MLP, blocks).                |
| `src/bin/train.rs`  | Training loop: batching, LR warmup+cosine decay, eval, save.  |
| `src/bin/generate.rs`| Load a checkpoint and autoregressively sample text.          |
| `src/dataset.rs`    | Char-level tokenizer + JSONL loader.                          |
| `src/config.rs`     | `GptConfig` + the `tiny` / `small` presets.                   |
| `src/backend.rs`    | Compile-time backend selection (ndarray ↔ wgpu).              |

This mirrors the Python project file-for-file (`model.py`, `train.py`, …).

---

## Presets

| Preset  | Params | Layers | Width | Context | Iters |
|---------|--------|--------|-------|---------|-------|
| `tiny`  | ~0.8M  | 4      | 128   | 128     | 600   |
| `small` | ~10M   | 6      | 384   | 256     | 3000  |

```bash
make train PRESET=small
cargo run --release --bin train -- --preset small --max-iters 1000   # override iters
```

---

## The checkpoint

Training writes two files into `out/`:

- `out/model.mpk` — the weights, via burn's `NamedMpkFileRecorder`.
- `out/meta.json` — the `GptConfig` and the character vocabulary, so
  `generate` can rebuild the exact same model and tokenizer standalone.

---

## Generating text

```bash
make generate PROMPT="JULIET:" TOKENS=300
cargo run --release --bin generate -- --prompt "To be" --temperature 0.7 --top-k 100
```

- `--temperature` — lower (0.5) = safe/repetitive, higher (1.0) = wilder.
- `--top-k` — sample only from the k most likely next characters.

---

## Reading the training log

A live progress bar updates in place, with periodic eval lines above it:

```
iter   100 [ 50.5%] | train 2.540 | val 2.541 | learned  39.1% | 105.6s
  [████████████░░░░░░░░░░░░]  50.5%  loss 2.542 (39.1%)   106s  ETA  104s
```

**`learned %`** = `(baseline − val_loss) / baseline × 100`, where the
random-guess baseline is `ln(vocab_size)`. A fresh model reads ~0%.

---

## Differences from the Python reference

- **Untied LM head.** PyTorch ties `lm_head.weight` to the token-embedding
  weight. burn modules own their parameters, so the head is a separate (untied)
  `Linear`. The architecture is otherwise identical.
- **Backend is compile-time**, not runtime (`--device`): pick it with a Cargo
  feature.

## Notes / gotchas baked in

- burn's `tril_mask`/`triu_mask` are named the **opposite** of numpy's. The
  causal mask (hide key positions `j > i`) is `tril_mask([t, t], 0)`. See
  `src/model.rs`.

## Tests

```bash
cargo test     # forward-pass finiteness + a causal-masking check
```
