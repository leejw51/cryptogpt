"""Download an open dataset and convert it to JSONL.

Default corpus: **TinyShakespeare** (~1.1 MB, public domain). It's the classic
tiny-LM benchmark — small enough to train on in under a minute, and reliably
hosted. Each blank-line-separated passage becomes one {"text": ...} record.

    python data.py                 # full corpus  -> data/corpus.jsonl
    python data.py --size 0.25     # keep first 25% (faster, smaller model)
"""
import argparse
import json
import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--raw", default="data/input.txt")
    ap.add_argument("--size", type=float, default=1.0,
                    help="fraction of the raw text to keep, 0-1 (default 1.0)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if not os.path.exists(args.raw):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, args.raw)
    else:
        print(f"using cached {args.raw}")

    with open(args.raw, encoding="utf-8") as f:
        text = f.read()

    if args.size < 1.0:
        text = text[: int(len(text) * args.size)]

    passages = [p.strip() for p in text.split("\n\n") if p.strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")

    print(f"wrote {len(passages)} records ({len(text):,} chars) -> {args.out}")


if __name__ == "__main__":
    main()
