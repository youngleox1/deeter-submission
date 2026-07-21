"""Core experiment data: character-level tiny-Shakespeare next-token
prediction. The corpus (data/tinyshakespeare.txt, public domain, ~1.1MB,
via https://raw.githubusercontent.com/karpathy/char-rnn) is vendored
directly in this repo rather than fetched at runtime, so reproducing the
core experiment does not depend on network access.
"""
from pathlib import Path
from typing import List

import torch

_DATA_PATH = Path(__file__).parent / "tinyshakespeare.txt"


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class TinyShakespeare:
    """Tokenizes the vendored corpus at char level and serves random
    contiguous windows for next-token prediction.

    Train/val is a single contiguous 90/10 cut, not a random shuffle of
    windows -- this keeps validation genuinely held out (no validation
    window's characters appear anywhere in a training window), at the cost
    of the two splits coming from different, non-interleaved parts of the
    corpus.
    """

    def __init__(self, val_fraction: float = 0.1, seed: int = 0, data_path: Path = _DATA_PATH):
        text = data_path.read_text(encoding="utf-8")
        self.tokenizer = CharTokenizer(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        split_idx = int(len(data) * (1 - val_fraction))
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]
        self.vocab_size = self.tokenizer.vocab_size
        self._generator = torch.Generator().manual_seed(seed)

    def get_batch(self, split: str, batch_size: int, seq_len: int, device="cpu"):
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - seq_len - 1
        if max_start <= 0:
            raise ValueError(f"'{split}' split too short for seq_len={seq_len}")
        starts = torch.randint(0, max_start, (batch_size,), generator=self._generator)
        x = torch.stack([data[s: s + seq_len] for s in starts])
        y = torch.stack([data[s + 1: s + seq_len + 1] for s in starts])
        return x.to(device), y.to(device)
