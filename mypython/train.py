"""Train the tiny GPT on a JSONL corpus.

    python train.py --preset tiny          # ~1 min on Apple silicon
    python train.py --preset small         # ~10 min on Apple silicon

The checkpoint (out/ckpt.pt) bundles the weights, the model config, and the
char vocabulary, so generate.py can run standalone.
"""
import argparse
import math
import os
import sys
import time

import torch

from config import GPTConfig, PRESETS
from dataset import CharTokenizer, load_jsonl_text
from model import GPT
from utils import get_device


def progress_bar(frac, elapsed, loss, baseline, width=24):
    """A single-line, in-place progress bar: [####----] 52% | loss% | time | ETA."""
    frac = min(max(frac, 0.0), 1.0)
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    eta = elapsed * (1 - frac) / frac if frac > 0 else 0.0
    learned = 100 * (baseline - loss) / baseline   # loss as "how far from random"
    return (f"\r  [{bar}] {frac * 100:5.1f}%  loss {loss:.3f} ({learned:4.1f}%)  "
            f"{elapsed:4.0f}s  ETA {eta:4.0f}s ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=list(PRESETS))
    ap.add_argument("--data", default="data/corpus.jsonl")
    ap.add_argument("--out", default="out/ckpt.pt")
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="override the preset's iteration count")
    args = ap.parse_args()

    p = PRESETS[args.preset]
    max_iters = args.max_iters or p["max_iters"]
    device = get_device(args.device)
    print(f"preset={args.preset}  device={device}")

    # ---- data ----------------------------------------------------------------
    text = load_jsonl_text(args.data)
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    block_size, batch_size = p["block_size"], p["batch_size"]
    print(f"chars={len(text):,}  vocab={tok.vocab_size}  "
          f"train={len(train_data):,}  val={len(val_data):,}")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)

    # ---- model ---------------------------------------------------------------
    config = GPTConfig(
        vocab_size=tok.vocab_size, block_size=block_size,
        n_layer=p["n_layer"], n_head=p["n_head"], n_embd=p["n_embd"], dropout=0.0,
    )
    model = GPT(config).to(device)
    print(f"params: {model.num_params() / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], betas=(0.9, 0.99))

    def lr_at(it):
        # linear warmup, then cosine decay to 10% of base lr
        if it < p["warmup"]:
            return p["lr"] * (it + 1) / p["warmup"]
        ratio = (it - p["warmup"]) / max(1, max_iters - p["warmup"])
        return p["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ratio)))

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(p["eval_iters"])
            for k in range(p["eval_iters"]):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---- training loop -------------------------------------------------------
    # Random-guess baseline: a fresh model's loss == ln(vocab_size). We express
    # progress as "learned %" = how far val loss has fallen from that baseline
    # toward 0 (a hypothetical perfect predictor).
    baseline = math.log(tok.vocab_size)
    t0 = time.time()
    disp_loss = baseline
    for it in range(max_iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        if it % p["eval_interval"] == 0 or it == max_iters - 1:
            l = estimate_loss()
            progress = 100 * (it + 1) / max_iters
            learned = 100 * (baseline - l["val"]) / baseline
            # \r + ANSI clear-to-EOL wipes the live bar, then print a line that stays
            print(f"\r\033[Kiter {it:5d} [{progress:5.1f}%] | train {l['train']:.3f} "
                  f"| val {l['val']:.3f} | learned {learned:5.1f}% "
                  f"| {time.time() - t0:5.1f}s")
            disp_loss = l["train"]

        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 10 == 0:                       # refresh shown loss (avoids per-iter GPU sync)
            disp_loss = loss.item()
        sys.stdout.write(progress_bar((it + 1) / max_iters, time.time() - t0, disp_loss, baseline))
        sys.stdout.flush()
    print()                                    # leave the final bar on its own line

    # ---- save ----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": vars(config),
         "stoi": tok.stoi, "itos": tok.itos},
        args.out,
    )
    print(f"saved {args.out}  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
