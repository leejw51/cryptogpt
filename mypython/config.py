"""Model + training presets.

A `GPTConfig` describes the model architecture. `PRESETS` bundle architecture
with training hyper-parameters so you can pick a target wall-clock time:

    tiny  -> ~1 minute on an Apple-silicon Mac (MPS)
    small -> ~10 minutes on an Apple-silicon Mac (MPS)

Scale the *data* independently with `make data SIZE=0.25` (keep 25% of corpus).
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 256   # set at train time from the char tokenizer
    block_size: int = 128   # context length (tokens the model can attend to)
    n_layer: int = 4        # number of transformer blocks
    n_head: int = 4         # attention heads per block
    n_embd: int = 128       # embedding / residual stream width
    dropout: float = 0.0
    bias: bool = True


# Training presets. Architecture fields mirror GPTConfig; the rest are optimizer
# / loop settings. Tuned to hit the target wall-clock on Apple silicon.
PRESETS = {
    "tiny": dict(
        n_layer=4, n_head=4, n_embd=128, block_size=128,
        batch_size=32, max_iters=600, eval_interval=100, eval_iters=50,
        lr=1e-3, warmup=50,
    ),
    "small": dict(
        n_layer=6, n_head=6, n_embd=384, block_size=256,
        batch_size=64, max_iters=3000, eval_interval=250, eval_iters=100,
        lr=3e-4, warmup=150,
    ),
}
