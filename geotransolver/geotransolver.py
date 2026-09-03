"""GeoTransolver: Transolver's physics attention extended with a geometry-
and condition-aware context, adapted for BlendedNet surface-CFD regression.

Reference: Wu et al., "Transolver: A Fast Transformer Solver for PDEs on
General Geometries", ICML 2024 (thuml/Transolver), and NVIDIA PhysicsNeMo's
GeoTransolver / GALE attention (physicsnemo/models/geotransolver), which
extends physics attention with cross-attention to geometry and global
conditioning tokens. Architecture re-derived here in plain PyTorch, adapted
to BlendedNet's inputs (point cloud, surface normals, per-case flight/
geometry condition) and outputs (per-point Cp/Cf field + per-case
aerodynamic constants such as C1, C3).

Physics attention, in one line: encode N points onto S << N learned "slice"
tokens by a softmax-weighted aggregation, attend among slices (and now,
additionally, between slices and a geometry/condition context), then
decode back to points with the same weights used to encode.
"""

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_

from .ball_query import ball_query, gather_neighbors

ACTIVATION = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=0, act="gelu", res=True):
        super().__init__()
        Act = ACTIVATION[act]
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), Act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), Act()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


def _slice_encode(x_mid, fx_mid, project_slice, temperature):
    """Shared physics-attention encode step: aggregate per-token features
    fx_mid onto S learned slices, weighted by a softmax over slices of a
    learned projection of x_mid.

    x_mid, fx_mid: (B, H, N, D)  ->  slice_weights: (B, H, N, S), slice_token: (B, H, S, D)
    """
    temperature = temperature.clamp(min=0.5, max=5.0)
    slice_logits = project_slice(x_mid) / temperature  # B H N S
    slice_weights = slice_logits.softmax(dim=-1)
    slice_norm = slice_weights.sum(dim=2)  # B H S, sum over N
    slice_token = torch.einsum("bhns,bhnd->bhsd", slice_weights, fx_mid)
    slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)
    return slice_weights, slice_token


class ContextTokenizer(nn.Module):
    """Encodes a per-point geometric field (e.g. surface normals) into S_ctx
    learned slice tokens, shared as cross-attention context across every
    GeoTransolver block. Same math as the encode half of physics attention,
    without a matching decode."""

    def __init__(self, in_dim, heads, dim_head, slice_num):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        inner_dim = heads * dim_head
        self.project_x = nn.Linear(in_dim, inner_dim)
        self.project_fx = nn.Linear(in_dim, inner_dim)
        self.project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.project_slice.weight)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

    def forward(self, x):
        # x: B N C -> B H S D
        B, N, _ = x.shape
        x_mid = self.project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        fx_mid = self.project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        _, slice_token = _slice_encode(x_mid, fx_mid, self.project_slice, self.temperature)
        return slice_token


class GeoPhysicsAttention(nn.Module):
    """Physics attention among slice tokens, extended with cross-attention
    to a shared geometry/condition context and a learned per-head gate that
    mixes the two before decoding back to points."""

    def __init__(self, dim, heads, dim_head, dropout, slice_num):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads, self.dim_head = heads, dim_head
        self.scale = dim_head ** -0.5

        self.project_x = nn.Linear(dim, inner_dim)
        self.project_fx = nn.Linear(dim, inner_dim)
        self.project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.project_slice.weight)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k_ctx = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v_ctx = nn.Linear(dim_head, dim_head, bias=False)
        self.mix_gate = nn.Parameter(torch.zeros(1, heads, 1, 1))  # sigmoid(0) = 0.5 at init

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context):
        # x: B N C, context: B H S_ctx D
        B, N, _ = x.shape
        x_mid = self.project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        fx_mid = self.project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        slice_weights, slice_token = _slice_encode(x_mid, fx_mid, self.project_slice, self.temperature)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        self_attn = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        self_out = self.dropout(self_attn) @ v  # B H S D

        k_ctx = self.to_k_ctx(context)
        v_ctx = self.to_v_ctx(context)
        cross_attn = (q @ k_ctx.transpose(-1, -2) * self.scale).softmax(dim=-1)
        cross_out = self.dropout(cross_attn) @ v_ctx  # B H S D

        w = torch.sigmoid(self.mix_gate)
        mixed = w * self_out + (1 - w) * cross_out

        out = torch.einsum("bhsd,bhns->bhnd", mixed, slice_weights)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GeoTransolverBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout, slice_num, mlp_ratio, act):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = GeoPhysicsAttention(dim, heads, dim_head, dropout, slice_num)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dim, n_layers=0, res=False, act=act)

    def forward(self, fx, context):
        fx = self.attn(self.ln_1(fx), context) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class LocalGeometryFeatures(nn.Module):
    """Multi-scale ball-query aggregation of surface-normal features around
    each point, projected to `dim` and added to the point embedding -- gives
    every block a neighborhood-scale geometric receptive field in addition
    to the exact-point normal and the whole-shape context tokens."""

    def __init__(self, geom_dim, dim, radii, neighbors, hidden_dim, act):
        super().__init__()
        self.radii, self.neighbors = radii, neighbors
        self.mlps = nn.ModuleList(
            [MLP(geom_dim * k, hidden_dim * 2, hidden_dim, n_layers=0, res=False, act=act) for k in neighbors]
        )
        self.out_proj = nn.Linear(hidden_dim * len(radii), dim)

    def forward(self, pos, geometry):
        feats = []
        for radius, k, mlp in zip(self.radii, self.neighbors, self.mlps):
            idx = ball_query(pos, radius, k)
            nbr = gather_neighbors(geometry, idx).flatten(-2)  # B N (k*C)
            feats.append(torch.tanh(mlp(nbr)))
        return self.out_proj(torch.cat(feats, dim=-1))


class GeoTransolver(nn.Module):
    """
    forward(pos, geometry, condition) ->
        field: (B, N, out_channels)        per-point Cp / Cfx / Cfy / Cfz
        constants: (B, num_constants)      per-case aerodynamic coefficients (C1, C3, ...)

    pos:       (B, N, space_dim)   surface point coordinates
    geometry:  (B, N, geom_dim)    per-point surface normal (or sdf+direction, etc.)
    condition: (B, cond_dim)       case flight condition / geometry parameters, or
                                    None if `cond_dim=None` (no per-case conditioning
                                    beyond geometry -- e.g. a single-flow-condition
                                    dataset like ShapeNet-Car)

    `cond_dim=None` drops the global condition context token entirely (cross-
    attention runs against geometry context only). `num_constants=None` drops
    the constants head; `forward` then returns `constants=None`.
    """

    def __init__(
        self,
        space_dim=3,
        geom_dim=3,
        cond_dim: int | None = 1,
        out_channels=4,
        num_constants: int | None = 1,
        dim=256,
        depth=8,
        heads=8,
        dim_head=32,
        num_slices=32,
        mlp_ratio=4,
        dropout=0.0,
        act="gelu",
        local_radii=(0.05, 0.25),
        local_neighbors=(8, 32),
        local_hidden=32,
    ):
        super().__init__()
        self.preprocess = MLP(space_dim + geom_dim, dim * 2, dim, n_layers=0, res=False, act=act)
        self.local_features = LocalGeometryFeatures(
            geom_dim, dim, local_radii, local_neighbors, local_hidden, act
        )
        self.geometry_context = ContextTokenizer(geom_dim, heads, dim_head, num_slices)
        self.condition_context = nn.Linear(cond_dim, heads * dim_head) if cond_dim else None
        self.dim_head = dim_head

        self.blocks = nn.ModuleList(
            [
                GeoTransolverBlock(dim, heads, dim_head, dropout, num_slices, mlp_ratio, act)
                for _ in range(depth)
            ]
        )

        self.ln_out = nn.LayerNorm(dim)
        self.field_head = nn.Linear(dim, out_channels)
        self.const_head = MLP(dim * 2, dim, num_constants, n_layers=1, res=True, act=act) if num_constants else None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pos: torch.Tensor, geometry: torch.Tensor, condition: torch.Tensor | None = None):
        fx = self.preprocess(torch.cat((pos, geometry), dim=-1))
        fx = fx + self.local_features(pos, geometry)

        context = self.geometry_context(geometry)  # B H S_geo D
        if self.condition_context is not None:
            B = pos.shape[0]
            cond_ctx = self.condition_context(condition).reshape(
                B, 1, context.shape[1], self.dim_head
            ).permute(0, 2, 1, 3)  # B H 1 D
            context = torch.cat((context, cond_ctx), dim=2)

        for block in self.blocks:
            fx = block(fx, context)
        fx = self.ln_out(fx)

        field = self.field_head(fx)
        constants = None
        if self.const_head is not None:
            pooled = torch.cat((fx.mean(dim=1), fx.amax(dim=1)), dim=-1)
            constants = self.const_head(pooled)
        return field, constants
