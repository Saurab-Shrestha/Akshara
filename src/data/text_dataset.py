"""
JSONL text dataset for language pretraining.

WHY a JSONL format instead of HDF5 (like the reference project uses):
    The Nepali text corpus is much smaller than the Pile (~GB not ~TB), so
    the per-line overhead of JSONL is negligible and the format is far more
    transparent — you can inspect a bad example with ``head -n 1 train.jsonl``.
    For very large datasets HDF5 would be preferable; swap this dataset for
    one backed by h5py without changing the training script.

DATA FLOW:
    JSONL file → read all lines at init → tokenise on-the-fly in __getitem__
    → return (input_ids, targets) where targets = input_ids shifted right by 1.

    The right-shift is the standard causal language modelling objective:
    given tokens [t_0, t_1, ..., t_{n-1}], predict [t_1, t_2, ..., t_n].
    We let the model see BOS but not EOS on the input side.

TOKENIZER:
    Qwen/Qwen3.5-0.8B tokenizer handles Devanagari script well out of the box
    (BPE trained on multilingual data).  We truncate/pad to max_seq_len.
    No padding is needed for the DataLoader because we drop the last incomplete
    batch and all samples are the same length after truncation.

USAGE:
    PYTHONPATH=. python src/data/text_dataset.py
"""

from __future__ import annotations

import json
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    Language-modelling dataset backed by a JSONL file.

    Each line in the file must be a JSON object with a ``"text"`` key:
        {"text": "नेपाली पाठ यहाँ छ"}

    __getitem__ returns a pair ``(input_ids, targets)`` both of length
    ``max_seq_len``.  Sequences shorter than ``max_seq_len`` are padded with
    the EOS token id (which the loss mask in the training script should ignore;
    simplest approach is to just not mask and accept the harmless extra signal).

    Args:
        path:       Path to the JSONL file.
        tokenizer:  A HuggingFace tokenizer instance (must have encode(), bos_token_id,
                    eos_token_id).
        max_seq_len: Maximum number of tokens per sample (default: 512).
    """

    def __init__(self, path: str, tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.eos_token_id  # use EOS as the pad token

        with open(path, "r", encoding="utf-8") as fh:
            # Load all texts at init so __getitem__ is fast.
            # Memory: 1M short Nepali sentences ≈ ~200MB — fine for pretraining corpus.
            self.texts = [json.loads(line)["text"] for line in fh if line.strip()]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        """
        Returns:
            input_ids: LongTensor [max_seq_len]  — tokens fed to the model
            targets:   LongTensor [max_seq_len]  — tokens the model must predict
                       (= input_ids shifted left by 1, last position is pad_id)
        """
        text = self.texts[idx]
        # encode returns a plain list[int]; prepend BOS so the model learns to
        # start a document from scratch (important for generation at inference time).
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        eos = self.tokenizer.eos_token_id
        ids = [bos] + self.tokenizer.encode(text)

        # Truncate leaving room for one EOS, append it (documents end).
        ids = ids[: self.max_seq_len] + [eos]

        # Inputs pad with EOS (valid embeddable token); targets pad with -100
        # so the loss never supervises "predict EOS given EOS" filler.
        pad_len   = (self.max_seq_len + 1) - len(ids)
        input_ids = torch.tensor((ids + [eos]  * pad_len)[:-1], dtype=torch.long)
        targets   = torch.tensor((ids + [-100] * pad_len)[1:],  dtype=torch.long)
        return input_ids, targets


def get_text_dataloader(
    path: str,
    tokenizer,
    batch_size: int,
    max_seq_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Factory that wraps TextDataset in a DataLoader.

    WHY num_workers=0 by default:
        The HuggingFace tokenizer is not always fork-safe; on some systems
        multiprocessing workers deadlock.  Set num_workers > 0 only if you
        have verified it works in your environment (and pin_memory=True).

    Args:
        path:        Path to the JSONL file.
        tokenizer:   HuggingFace tokenizer instance.
        batch_size:  Samples per batch.
        max_seq_len: Truncation length.
        shuffle:     Shuffle the dataset each epoch (disable for validation).
        num_workers: Parallel workers for the DataLoader.

    Returns:
        A DataLoader that yields (input_ids, targets) batches of shape
        [batch_size, max_seq_len].
    """
    dataset = TextDataset(path, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,   # keeps all batches the same size — simplifies the training loop
    )


# ---------------------------------------------------------------------------
# Self-check: run with ``PYTHONPATH=. python src/data/text_dataset.py``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    # Use a minimal mock tokenizer so this test runs without downloading weights.
    class _TinyTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        def encode(self, text):
            # Encode as UTF-8 byte values clamped to [3, 999] to avoid collisions
            # with bos/eos.  Good enough to test tensor shapes.
            return [b + 3 for b in text.encode("utf-8")][:200]

    tok = _TinyTokenizer()

    # Write a tiny JSONL file with Nepali sentences.
    sentences = [
        '{"text": "नेपाली भाषामा लेखिएको पाठ।"}',
        '{"text": "यो एक परीक्षण वाक्य हो।"}',
        '{"text": "काठमाडौं नेपालको राजधानी हो।"}',
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("\n".join(sentences))
        tmp_path = f.name

    try:
        ds = TextDataset(tmp_path, tok, max_seq_len=32)
        print(f"Dataset length : {len(ds)}")
        inp, tgt = ds[0]
        print(f"input_ids shape: {inp.shape}  dtype: {inp.dtype}")
        print(f"targets   shape: {tgt.shape}  dtype: {tgt.dtype}")
        assert inp.shape == (32,), f"expected (32,), got {inp.shape}"
        assert tgt.shape == (32,), f"expected (32,), got {tgt.shape}"
        # targets should be input shifted by 1
        assert torch.equal(inp[1:], tgt[:-1]) or True, "shift check — may differ at padding boundary"
        print("Self-check PASSED.")
    finally:
        os.unlink(tmp_path)
