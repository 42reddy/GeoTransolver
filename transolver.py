"""
Transolver for point-cloud / unstructured-mesh PDE surrogates.
Reference: Wu et al., "Transolver: A Fast Transformer Solver for PDEs on
General Geometries", ICML 2024.
"""

import torch
import torch.nn as nn
from einops import rearrange


class MLP(nn.Module):
    """Two-layer point-wise feed-forward map R^{c_in} -> R^{c_out}, applied
    identically and independently to every point (a 1x1 convolution in
    disguise), which is what preserves permutation equivariance."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsAttention(nn.Module):
    """Slice -> attend -> deslice, as derived in the module docstring.

    Input / output: (B, N, C) point-wise features. Internally reduces the
    O(N) point axis to a fixed O(M) slice axis for the attention step, so
    the module scales linearly in the number of points N.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 32,
        num_slices: int = 32,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.heads = heads
        self.num_slices = num_slices
        self.eps = eps
        inner_dim = heads * dim_head
        self.scale = dim_head**-0.5

        # step 1: per-point, per-head distribution over M slices
        self.to_slice_logits = nn.Linear(dim, heads * num_slices)

        # step 2: value projection used to form the slice tokens
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # step 3: standard attention over the M slice tokens
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_v_attn = nn.Linear(inner_dim, inner_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, C)
        B, N, _ = h.shape
        H, M = self.heads, self.num_slices

        # --- 1. slice assignment: w[b, i, head, m], normalized over m ---
        logits = self.to_slice_logits(h).view(B, N, H, M)
        w = logits.softmax(dim=-1)  # sum_m w[..., m] = 1

        # --- 2. slice aggregation: weighted mean pooling over points ---
        v = rearrange(self.to_v(h), "b n (h d) -> b n h d", h=H)
        weighted_sum = torch.einsum("bnhm,bnhd->bhmd", w, v)
        weight_mass = w.sum(dim=1).unsqueeze(-1)  # (B, H, M, 1)
        slices = weighted_sum / (weight_mass + self.eps)  # (B, H, M, d_head)
        slices = rearrange(slices, "b h m d -> b m (h d)")

        # --- 3. self-attention among the M physical-state tokens ---
        q = rearrange(self.to_q(slices), "b m (h d) -> b h m d", h=H)
        k = rearrange(self.to_k(slices), "b m (h d) -> b h m d", h=H)
        v_attn = rearrange(self.to_v_attn(slices), "b m (h d) -> b h m d", h=H)

        scores = torch.einsum("bhmd,bhnd->bhmn", q, k) * self.scale
        attn = scores.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        updated_slices = torch.einsum("bhmn,bhnd->bhmd", attn, v_attn)  # (B, H, M, d_head)

        # --- 4. deslice: broadcast back to points via the same weights ---
        h_out = torch.einsum("bnhm,bhmd->bnhd", w, updated_slices)
        h_out = rearrange(h_out, "b n h d -> b n (h d)")
        return self.to_out(h_out)


class TransolverBlock(nn.Module):
    """Pre-norm residual block: physics-attention token mixing followed by
    a point-wise feed-forward channel mixing, matching the standard
    Transformer template with self-attention replaced by PhysicsAttention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        num_slices: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, heads, dim_head, num_slices, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, int(dim * mlp_ratio), dim, dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.norm1(h))
        h = h + self.ffn(self.norm2(h))
        return h


class Transolver(nn.Module):
    """Full point-cloud PDE surrogate: point-wise embedding, a stack of
    Physics-Attention blocks, and a point-wise decoding head.

    Coordinates and input features are fused at the embedding stage only
    (x_i concatenated with f_i); there is no separate positional-encoding
    module, so the model is agnostic to any global coordinate frame and to
    the number/order of input points.
    """

    def __init__(
        self,
        space_dim: int,
        in_channels: int,
        out_channels: int,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 32,
        num_slices: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = MLP(space_dim + in_channels, dim, dim, dropout)
        self.blocks = nn.ModuleList(
            [
                TransolverBlock(dim, heads, dim_head, num_slices, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(dim)
        self.decode = MLP(dim, dim, out_channels, dropout)

    def forward(self, pos: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        """
        pos:      (B, N, space_dim)   point coordinates x_i
        features: (B, N, in_channels) known physical state f_i, or None
        returns:  (B, N, out_channels) predicted field y_i
        """
        x = pos if features is None else torch.cat([pos, features], dim=-1)
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.decode(self.norm_out(h))
