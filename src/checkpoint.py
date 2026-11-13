"""
Safe checkpoint loading.

torch.load(weights_only=False) unpickles arbitrary Python objects, so a
tampered .pt file (e.g. one pulled from a hub) can run code on load. Akshara
checkpoints hold only tensors, numbers, strings and plain dicts/lists, which
weights_only=True accepts. A file that needs full unpickling is refused unless
AKSHARA_TRUST_CHECKPOINTS=1 (or trusted=True) says you made it yourself.
"""

from __future__ import annotations

import os
import pickle

import torch


def load_checkpoint(path: str, map_location="cpu", trusted: bool = False) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        if trusted or os.environ.get("AKSHARA_TRUST_CHECKPOINTS") == "1":
            return torch.load(path, map_location=map_location, weights_only=False)
        raise RuntimeError(
            f"{path} contains objects beyond tensors and plain containers, so it was not "
            f"loaded (unpickling can run code). If you created it yourself, set "
            f"AKSHARA_TRUST_CHECKPOINTS=1.\n  detail: {str(e).splitlines()[0]}") from e


def strip_prefixes(state_dict: dict) -> dict:
    """Drop torch.compile / DataParallel / DDP prefixes from state_dict keys."""
    return {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in state_dict.items()}
