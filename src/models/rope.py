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


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the complex rotation factors e^(i * pos * θ) for all positions.

    Using complex numbers is a compact way to represent 2D rotations:
        e^(iθ) = cos(θ) + i*sin(θ)
    Multiplying two complex numbers rotates one by the other's angle.

    Args:
        dim:         head dimension (must be even — we pair up dimensions)
        max_seq_len: maximum sequence length to precompute for
        base:        controls how fast frequencies decay (10000 is standard)

    Returns:
        freqs_cis: complex tensor of shape (max_seq_len, dim // 2)
    """
    # θ_i = 1 / (base ^ (2i / dim)) for i = 0, 1, ..., dim/2 - 1
    # Lower i → larger θ → faster rotation → encodes fine-grained position
    # Higher i → smaller θ → slower rotation → encodes coarse position
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim//2,)

    # Position indices 0, 1, 2, ..., max_seq_len-1
    positions = torch.arange(max_seq_len).float()  # (max_seq_len,)

    # Outer product: each position × each frequency
    # Shape: (max_seq_len, dim//2)
    angles = torch.outer(positions, freqs)

    # Convert to complex: e^(i*angle) = cos(angle) + i*sin(angle)
    # Shape: (max_seq_len, dim//2), complex64
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary position embedding to a query or key tensor.

    Args:
        x:          (batch, seq_len, n_heads, head_dim) — q or k tensor
        freqs_cis:  (seq_len, head_dim // 2) — precomputed rotation factors

    Returns:
        rotated tensor, same shape as x
    """
    # Reshape x to pair up dimensions: (batch, seq_len, n_heads, head_dim//2, 2)
    # then view as complex: (batch, seq_len, n_heads, head_dim//2)
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)  # each pair becomes a complex number

    # freqs_cis needs to broadcast over batch and n_heads dimensions
    # reshape: (1, seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Multiply complex numbers = rotate by the precomputed angle
    rotated = x_complex * freqs_cis

    # Convert back to real: (batch, seq_len, n_heads, head_dim//2, 2)
    # then flatten last two dims back to head_dim
    out = torch.view_as_real(rotated).flatten(-2)

    return out.type_as(x)  # restore original dtype (fp16/bf16 if used)


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch, seq_len, n_heads, head_dim = 2, 8, 4, 32

    # Precompute rotation factors
    freqs_cis = precompute_freqs_cis(dim=head_dim, max_seq_len=seq_len)
    assert freqs_cis.shape == (seq_len, head_dim // 2), "wrong freqs shape"

    # Apply to a fake query tensor
    q = torch.randn(batch, seq_len, n_heads, head_dim)
    q_rotated = apply_rope(q, freqs_cis)

    assert q_rotated.shape == q.shape, "rotation must preserve shape"

    # Rotation should not change the magnitude of each vector
    # (rotation only changes direction, not length)
    orig_norm = q.norm(dim=-1)
    rot_norm  = q_rotated.norm(dim=-1)
    assert (orig_norm - rot_norm).abs().max().item() < 1e-4, \
        "rotation changed vector magnitude — something is wrong"

    print("RoPE self-check passed.")
    print(f"  freqs_cis shape : {freqs_cis.shape}")
    print(f"  q shape         : {q.shape}")
    print(f"  q_rotated shape : {q_rotated.shape}")
    print(f"  max norm diff   : {(orig_norm - rot_norm).abs().max().item():.2e}  (should be ~0)")
