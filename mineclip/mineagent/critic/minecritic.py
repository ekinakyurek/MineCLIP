from __future__ import annotations
import torch.nn as nn

from mineclip.utils import build_mlp


class MineCritic(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device,
    ):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.preprocess = preprocess_net
        self.last = build_mlp(
            input_dim=preprocess_net.output_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=None,
        )
        self._device = device

    def forward(self, obs, **kwargs):
        x, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(x)


