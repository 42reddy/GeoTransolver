"""
Transolver for point-cloud / unstructured-mesh PDE surrogates.
Reference: Wu et al., "Transolver: A Fast Transformer Solver for PDEs on
General Geometries", ICML 2024. Architecture ported from the official
thuml/Transolver implementation; adapted here to take plain (B, N, C)
batched tensors (pos, features) instead of the original's single-instance
PyG-style data object, so it plugs directly into shapenet_car_dataset.py /
train.py, and to always fold `features` (sdf/normal/etc.) into the input
instead of silently dropping them.
"""

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Slice -> attend -> deslice, exactly as in the official implementation."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            dim_head=None,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads,
                                                     dim_head=dim_head or (hidden_dim // num_heads),
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver(nn.Module):
    """Full point-cloud PDE surrogate: point-wise embedding (pos + features),
    a stack of Physics-Attention blocks, and a point-wise decoding head
    (folded into the last block, as in the official implementation).

    forward(pos, features) -> (B, N, out_channels), matching
    shapenet_car_dataset.py's batched tensors directly (no PyG data object).
    """

    def __init__(self,
                 space_dim=3,
                 in_channels=1,
                 out_channels=1,
                 dim=256,
                 depth=8,
                 heads=8,
                 dim_head=32,
                 num_slices=32,
                 mlp_ratio=4.0,
                 dropout=0.,
                 act='gelu',
                 ref=8,
                 unified_pos=False,
                 ):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.space_dim = space_dim
        self.n_hidden = dim

        if self.unified_pos:
            self.preprocess = MLP(in_channels + space_dim + self.ref ** 3, dim * 2, dim, n_layers=0,
                                   res=False, act=act)
        else:
            self.preprocess = MLP(in_channels + space_dim, dim * 2, dim, n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList([Transolver_block(num_heads=heads, hidden_dim=dim,
                                                       dropout=dropout,
                                                       act=act,
                                                       mlp_ratio=int(mlp_ratio),
                                                       out_dim=out_channels,
                                                       slice_num=num_slices,
                                                       dim_head=dim_head,
                                                       last_layer=(i == depth - 1))
                                      for i in range(depth)])
        self.placeholder = nn.Parameter((1 / dim) * torch.rand(dim, dtype=torch.float))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos: B N 3
        batchsize = my_pos.shape[0]
        device = my_pos.device

        gridx = torch.linspace(-1.5, 1.5, self.ref, device=device)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.linspace(0, 2, self.ref, device=device)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.linspace(-4, 4, self.ref, device=device)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).reshape(batchsize, self.ref ** 3, 3)  # B G^3 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)
        ).reshape(batchsize, my_pos.shape[1], self.ref ** 3).contiguous()
        return pos

    def forward(self, pos: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        """
        pos:      (B, N, space_dim)   point coordinates x_i
        features: (B, N, in_channels) known physical state f_i, or None
        returns:  (B, N, out_channels) predicted field y_i
        """
        x = pos
        if self.unified_pos:
            x = torch.cat((x, self.get_grid(pos)), dim=-1)

        if features is not None:
            fx = self.preprocess(torch.cat((x, features), dim=-1))
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        return fx
