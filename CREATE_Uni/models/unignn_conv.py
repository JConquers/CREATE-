"""
UniGNN convolution layers adapted for recommendation.
Based on the official UniGNN implementation (2021).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


def glorot(tensor: torch.Tensor):
    """Xavier initialization."""
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def normalize_l2(X: torch.Tensor) -> torch.Tensor:
    """Row-normalize tensor."""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.0
    return X * scale


class UniGCNConv(nn.Module):
    """
    UniGCN convolution layer.

    Implements: X -> XW -> AXW -> norm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.0,
        first_aggregate: str = "mean",
        use_norm: bool = False,
    ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.first_aggregate = first_aggregate
        self.use_norm = use_norm

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        degE: torch.Tensor,
        degV: torch.Tensor,
    ) -> torch.Tensor:
        N = X.shape[0]

        X = self.W(X)
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce=self.first_aggregate)
        Xe = Xe * degE.unsqueeze(-1)  # (E,) -> (E,1) for broadcasting with (E,D)

        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)
        Xv = Xv * degV.unsqueeze(-1)  # (N,) -> (N,1) for broadcasting with (N,D)

        X = Xv

        # GELU activation after φ₂ (edge→vertex) aggregation
        X = F.gelu(X)

        if self.use_norm:
            X = normalize_l2(X)

        return X


class UniGCNIIConv(nn.Module):
    """
    UniGCNII convolution layer with initial residual connection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        first_aggregate: str = "mean",
        use_norm: bool = False,
    ):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.first_aggregate = first_aggregate
        self.use_norm = use_norm

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        degE: torch.Tensor,
        degV: torch.Tensor,
        alpha: float,
        beta: float,
        X0: torch.Tensor,
    ) -> torch.Tensor:
        N = X.shape[0]

        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce=self.first_aggregate)
        Xe = Xe * degE.unsqueeze(-1)  # (E,) -> (E,1) for broadcasting with (E,D)

        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)
        Xv = Xv * degV.unsqueeze(-1)  # (N,) -> (N,1) for broadcasting with (N,D)

        X = Xv

        # GELU activation after φ₂ (edge→vertex) aggregation
        X = F.gelu(X)

        if self.use_norm:
            X = normalize_l2(X)

        Xi = (1 - alpha) * X + alpha * X0
        X = (1 - beta) * Xi + beta * self.W(Xi)

        return X


class UniGINConv(nn.Module):
    """
    UniGIN convolution layer.

    Implements: X -> XW -> AXW -> norm with learnable epsilon.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.0,
        first_aggregate: str = "sum",
        use_norm: bool = False,
    ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.first_aggregate = first_aggregate
        self.use_norm = use_norm
        self.eps = nn.Parameter(torch.Tensor([0.0]))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        degE: torch.Tensor = None,
        degV: torch.Tensor = None,
    ) -> torch.Tensor:
        N = X.shape[0]

        X = self.W(X)
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce=self.first_aggregate)

        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)
        X = (1 + self.eps) * X + Xv

        # GELU activation after φ₂ (edge→vertex) aggregation
        X = F.gelu(X)

        if self.use_norm:
            X = normalize_l2(X)

        return X


class UniSAGEConv(nn.Module):
    """
    UniSAGE convolution layer.

    Implements: X -> XW -> AXW -> norm with skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.0,
        first_aggregate: str = "mean",
        second_aggregate: str = "sum",
        use_norm: bool = False,
    ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.first_aggregate = first_aggregate
        self.second_aggregate = second_aggregate
        self.use_norm = use_norm

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        degE: torch.Tensor = None,
        degV: torch.Tensor = None,
    ) -> torch.Tensor:
        N = X.shape[0]

        X = self.W(X)
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce=self.first_aggregate)

        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce=self.second_aggregate, dim_size=N)
        X = X + Xv

        # GELU activation after φ₂ (edge→vertex) aggregation
        X = F.gelu(X)

        if self.use_norm:
            X = normalize_l2(X)

        return X


class UniGATConv(nn.Module):
    """
    UniGAT convolution layer with attention mechanism.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        first_aggregate: str = "mean",
        use_norm: bool = False,
        skip_sum: bool = False,
    ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.first_aggregate = first_aggregate
        self.use_norm = use_norm
        self.skip_sum = skip_sum
        self.reset_parameters()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        degE: torch.Tensor = None,
        degV: torch.Tensor = None,
    ) -> torch.Tensor:
        H, C, N = self.heads, self.out_channels, X.shape[0]

        X0 = self.W(X)
        X = X0.view(N, H, C)

        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce=self.first_aggregate)

        alpha_e = (Xe * self.att_e).sum(-1)
        a_ev = alpha_e[edges]
        alpha = self.leaky_relu(a_ev)

        # Softmax over edges grouped by vertex (replacement for torch_geometric.utils.softmax)
        # Compute exp(alpha - max_per_group) for numerical stability
        alpha_max = scatter(alpha, vertex, dim=0, dim_size=N, reduce="max")[vertex]
        alpha_exp = torch.exp(alpha - alpha_max)
        alpha_sum = scatter(alpha_exp, vertex, dim=0, dim_size=N, reduce="sum")[vertex]
        alpha = alpha_exp / (alpha_sum + 1e-16)

        alpha = self.attn_drop(alpha)
        alpha = alpha.unsqueeze(-1)

        Xev = Xe[edges]
        Xev = Xev * alpha
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)
        X = Xv.view(N, H * C)

        # GELU activation after φ₂ (edge→vertex) aggregation
        X = F.gelu(X)

        if self.use_norm:
            X = normalize_l2(X)

        if self.skip_sum:
            X = X + X0

        return X
