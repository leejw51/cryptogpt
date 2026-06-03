"""Character-level tokenizer + JSONL corpus loader.

We keep tokenization deliberately simple (one token == one character) so the
model trains in seconds-to-minutes on a laptop. The vocabulary is built from the
training text itself and stored inside the checkpoint, so generation uses the
exact same mapping.
"""
import json


class CharTokenizer:
    def __init__(self, stoi: dict, itos: dict):
        self.stoi = stoi                 # char -> id
        self.itos = itos                 # id   -> char
        self.vocab_size = len(stoi)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_jsonl_text(path: str) -> str:
    """Read a JSONL file of {"text": ...} records and join them into one string."""
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
    return "\n".join(texts)
