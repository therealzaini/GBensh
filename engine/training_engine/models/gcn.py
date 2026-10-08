from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Graph Convolutional Network for node-level prediction.

    Parameters
    ----------
    in_channels:
        Number of input node features.

    hidden_channels:
        Hidden dimensionality. If an integer is provided, the same width is
        used for every hidden layer. A sequence may be provided to specify
        each hidden layer independently.

    out_channels:
        Number of output channels.

    num_layers:
        Total number of graph convolutional layers, including the output
        layer.

    dropout:
        Dropout probability applied between hidden layers.

    activation:
        Activation function applied after hidden convolutions.

    bias:
        Whether GCNConv layers use a learnable bias.

    improved:
        Whether to use the improved GCN normalization.

    cached:
        Whether normalized adjacency information should be cached.

        This should normally remain False when using graph sampling or when
        edge_index can change between forward passes.

    add_self_loops:
        Whether GCNConv should add self-loops.

    normalize:
        Whether GCNConv should perform symmetric normalization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | Sequence[int],
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        activation: str = "relu",
        bias: bool = True,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self._validate_config(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout_p = dropout

        hidden_channels = self._resolve_hidden_channels(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )

        self.activation = self._resolve_activation(activation)
        self.dropout = nn.Dropout(dropout)

        channels = [
            in_channels,
            *hidden_channels,
            out_channels,
        ]

        self.convs = nn.ModuleList(
            GCNConv(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                improved=improved,
                cached=cached,
                add_self_loops=add_self_loops,
                normalize=normalize,
                bias=bias,
            )
            for i in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """
        Compute node-level logits.

        Parameters
        ----------
        x:
            Node feature matrix of shape [num_nodes, in_channels].

        edge_index:
            Graph connectivity in COO format with shape [2, num_edges].

        edge_weight:
            Optional scalar edge weights with shape [num_edges].

        Returns
        -------
        Tensor
            Node logits of shape [num_nodes, out_channels].
        """

        for conv in self.convs[:-1]:
            x = conv(
                x,
                edge_index,
                edge_weight=edge_weight,
            )

            x = self.activation(x)
            x = self.dropout(x)

        x = self.convs[-1](
            x,
            edge_index,
            edge_weight=edge_weight,
        )

        return x

    @staticmethod
    def _resolve_hidden_channels(
        hidden_channels: int | Sequence[int],
        num_layers: int,
    ) -> list[int]:

        num_hidden_layers = num_layers - 1

        if num_hidden_layers == 0:
            return []

        if isinstance(hidden_channels, int):
            return [hidden_channels] * num_hidden_layers

        hidden_channels = list(hidden_channels)

        if len(hidden_channels) != num_hidden_layers:
            raise ValueError(
                "`hidden_channels` must contain exactly "
                f"{num_hidden_layers} entries for a {num_layers}-layer GCN, "
                f"but received {len(hidden_channels)}."
            )

        return hidden_channels

    @staticmethod
    def _resolve_activation(name: str) -> nn.Module:

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }

        try:
            return activations[name.lower()]()
        except KeyError:
            raise ValueError(
                f"Unknown activation '{name}'. "
                f"Available activations: {', '.join(activations)}."
            )

    @staticmethod
    def _validate_config(
        in_channels: int,
        hidden_channels: int | Sequence[int],
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:

        if in_channels <= 0:
            raise ValueError("`in_channels` must be strictly positive.")

        if out_channels <= 0:
            raise ValueError("`out_channels` must be strictly positive.")

        if num_layers < 1:
            raise ValueError("`num_layers` must be at least 1.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("`dropout` must satisfy 0 <= dropout < 1.")

        if isinstance(hidden_channels, int):
            if hidden_channels <= 0:
                raise ValueError(
                    "`hidden_channels` must be strictly positive."
                )

        else:
            if any(width <= 0 for width in hidden_channels):
                raise ValueError(
                    "All entries in `hidden_channels` must be strictly positive."
                )