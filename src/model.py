from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.0


class DynamicsMLP(nn.Module):
    """
    Simple MLP dynamics model.

    It learns: x -> y

    Where:
      x = concat(obs, action_one_hot)            shape: (batch, input_dim)
      y = delta_obs (or next_obs depending)     shape: (batch, output_dim)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []

        # First layer: input -> hidden
        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.ReLU())

        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))

        # Hidden layers
        for _ in range(config.num_hidden_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

        # Final layer: hidden -> output
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim)
        returns: (batch, output_dim)
        """
        return self.net(x)
