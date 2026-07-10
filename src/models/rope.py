"""
RoPE — Rotary Position Embedding
=================================

WHY THIS EXISTS
---------------
Transformers have no built-in sense of order. Without position information,
"नेपाल राम्रो छ" and "राम्रो छ नेपाल" look identical to the model.

The repo solves this with a learned position table:
    final = token_embed[token] + position_embed[pos]

Problem: the table has a fixed maximum size. Train on length 512,
and position 513 has no learned vector — the model breaks on longer inputs.

RoPE (Su et al., 2021) solves this differently.
Instead of adding a position vector to the token, it ROTATES the
query and key vectors inside attention by an angle that depends on position.

The rotation has a beautiful property:
    dot(rotate(q, pos_i), rotate(k, pos_j))
depends only on (pos_i - pos_j) — the *relative* distance between tokens.
So the model naturally learns "token A is 3 positions before token B"
rather than "token A is at absolute position 7".

Benefits:
  - No lookup table → works at any sequence length
  - Encodes relative distance → better at generalizing to longer texts
  - Used in: LLaMA, Qwen, Mistral, Gemma, GPT-NeoX, and our model

HOW IT WORKS (intuition)
------------------------
Think of each pair of dimensions in a vector as x,y coordinates on a circle.
RoPE rotates that pair by angle (pos * freq) where freq varies by dimension:
  - Low dimensions  → rotate slowly (track long-range position)
  - High dimensions → rotate fast  (track short-range position)

After rotation, the dot product between q and k only depends on their
relative angle difference = relative position. Absolute position cancels out.

FORMULA
-------
For a vector x at position pos, with dimension pairs (x_{2i}, x_{2i+1}):

  θ_i = 1 / (10000 ^ (2i / dim))      # frequency for dimension pair i

  rotated pair = [x_{2i}   * cos(pos * θ_i) - x_{2i+1} * sin(pos * θ_i),
                  x_{2i+1} * cos(pos * θ_i) + x_{2i}   * sin(pos * θ_i)]

This is exactly a 2D rotation matrix applied to each pair.

USAGE
-----
Precompute cos/sin tables once (precompute_freqs_cis).
Apply to q and k inside every attention layer (apply_rope).
Values (v) are NOT rotated — only q and k.
"""

import torch


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0):
    """
    Precomputes cos and sin rotation tables for RoPE.

    Args:
        dim:         head dimension (must be even)
        max_seq_len: maximum sequence length to precompute for
        base:        frequency base (10000 is standard)

    Returns:
        (cos, sin): two real float32 tensors each of shape (max_seq_len, dim // 2)
    """
    freqs     = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim//2,)
    positions = torch.arange(max_seq_len).float()                          # (max_seq_len,)
    angles    = torch.outer(positions, freqs)                              # (max_seq_len, dim//2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary position embedding to a query or key tensor.

    Args:
        x:   (batch, seq_len, n_heads, head_dim) — q or k tensor
        cos: (seq_len, head_dim // 2)
        sin: (seq_len, head_dim // 2)

    Returns:
        rotated tensor, same shape as x
    """
    # Split head_dim into even (x1) and odd (x2) indices — each pair is rotated together
    x1 = x[..., ::2].float()   # (B, T, H, D//2)
    x2 = x[..., 1::2].float()  # (B, T, H, D//2)

    # Broadcast cos/sin over batch and head dimensions: (1, T, 1, D//2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # 2D rotation: (x1, x2) → (x1·cos − x2·sin, x2·cos + x1·sin)
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin

    # Interleave r1 and r2 back to (B, T, H, D)
    out = torch.stack([r1, r2], dim=-1).flatten(-2)

    return out.type_as(x)


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch, seq_len, n_heads, head_dim = 2, 8, 4, 32

    cos, sin = precompute_freqs_cis(dim=head_dim, max_seq_len=seq_len)
    assert cos.shape == (seq_len, head_dim // 2)
    assert sin.shape == (seq_len, head_dim // 2)

    q = torch.randn(batch, seq_len, n_heads, head_dim)
    q_rotated = apply_rope(q, cos, sin)

    assert q_rotated.shape == q.shape, "rotation must preserve shape"

    orig_norm = q.norm(dim=-1)
    rot_norm  = q_rotated.norm(dim=-1)
    assert (orig_norm - rot_norm).abs().max().item() < 1e-4, \
        "rotation changed vector magnitude"

    print("RoPE self-check passed.")
    print(f"  cos/sin shape : {cos.shape}")
    print(f"  q_rotated shape : {q_rotated.shape}")
    print(f"  max norm diff   : {(orig_norm - rot_norm).abs().max().item():.2e}")
