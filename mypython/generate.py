"""Generate text from a trained checkpoint.

    python generate.py --prompt "ROMEO:" --max-new-tokens 400
"""
import argparse

import torch

from config import GPTConfig
from dataset import CharTokenizer
from model import GPT
from utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="lower = safer/repetitive, higher = wilder")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tok = CharTokenizer(ckpt["stoi"], ckpt["itos"])

    start = tok.encode(args.prompt) or [0]
    idx = torch.tensor(start, dtype=torch.long, device=device)[None, ...]
    out = model.generate(
        idx, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
