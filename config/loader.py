"""
JSON config loader for the Akshara project.

WHY a 4-layer merge instead of a single JSON file:
    Different stages (pretrain vs. ocr_finetune) share most model architecture
    knobs but differ in batch sizes, data paths, and LR schedules.  A 4-layer
    merge lets us write each shared value exactly once:

        1. dataclass field defaults        (lowest precedence)
        2. configs/base.json               (shared architecture + runtime)
        3. the stage JSON (configs/pretrain.json, configs/ocr_finetune.json, ...)
        4. CLI --field overrides           (highest precedence)

    This mirrors the pattern from train-llm-from-scratch/config/loader.py.

SMOKE CONFIG SUPPORT:
    When ``json_path`` lives in a sub-directory that contains its own ``base.json``
    (e.g. ``configs/smoke/pretrain.json``), that sibling base is loaded instead of
    the top-level ``configs/base.json``.  This shrinks the model automatically for
    fast local tests without touching any production JSON.

JSON NULL:
    JSON ``null`` maps cleanly to Python ``None``, which is the right way to express
    optional fields like ``amp_dtype`` or ``pretrain_ckpt``.

UNKNOWN KEYS:
    Keys in a JSON file that don't correspond to a dataclass field are warned about
    and silently dropped — not a fatal error.  This lets you keep comments-as-keys
    (e.g. ``"_comment": "..."``') in the JSON during experimentation.

USAGE (from a script):
    from config.loader import load_config
    from config.config import PretrainConfig

    cfg = load_config(PretrainConfig, "configs/pretrain.json", overrides={"lr": 1e-3})
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from typing import Any


def _deep_merge(dst: dict, src: dict) -> dict:
    """
    Recursively merge ``src`` into ``dst``.

    Nested dicts are merged in-place; scalar values are overwritten.
    This is future-proof for nested config sections, even though the current
    dataclasses are flat.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _resolve_base(json_path: str | None, base_path: str | None) -> str:
    """
    Determine which base.json to load.

    Priority:
        1. Explicit ``base_path`` argument (caller knows best).
        2. Sibling ``base.json`` next to ``json_path`` (smoke config pattern).
        3. ``configs/base.json`` as the project-wide fallback.
    """
    if base_path is not None:
        return base_path
    if json_path:
        sibling = os.path.join(os.path.dirname(json_path), "base.json")
        if os.path.exists(sibling):
            return sibling
    return "configs/base.json"


def load_config(
    cfg_cls,
    json_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    base_path: str | None = None,
):
    """
    Resolve a config dataclass from base.json + stage JSON + CLI overrides.

    Args:
        cfg_cls:   The stage dataclass class (e.g. ``PretrainConfig``).
        json_path: Path to the stage JSON (e.g. ``configs/pretrain.json``).
                   Pass ``None`` to use only the base + dataclass defaults.
        overrides: Dict of field-name → value from CLI parsing.  ``None`` values
                   are ignored so that absent CLI flags don't shadow JSON values.
        base_path: Explicit path to the shared base JSON.  Leave as ``None`` to
                   use the auto-resolved sibling or ``configs/base.json``.

    Returns:
        An instance of ``cfg_cls`` with all four layers applied.

    Raises:
        TypeError: if a JSON value cannot be coerced to the field's annotated type
                   (Python dataclass constructor handles this).
    """
    base = _resolve_base(json_path, base_path)

    # Accumulate key→value from lowest to highest precedence.
    merged: dict[str, Any] = {}
    for path in (base, json_path):
        if path and os.path.exists(path):
            with open(path) as fh:
                _deep_merge(merged, json.load(fh))

    # Drop keys that don't exist on the target dataclass and warn the author.
    field_names = {f.name for f in fields(cfg_cls)}
    for key in list(merged):
        if key not in field_names:
            print(f"[config] ignoring unknown key '{key}' for {cfg_cls.__name__}")
            merged.pop(key)

    # CLI overrides win; skip keys whose value is None (flag not supplied).
    if overrides:
        merged.update(
            {k: v for k, v in overrides.items() if k in field_names and v is not None}
        )

    return cfg_cls(**merged)
