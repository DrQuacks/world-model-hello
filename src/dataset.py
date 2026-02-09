from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


TargetMode = Literal["next_obs", "delta"]


@dataclass(frozen=True)
class DatasetStats:
    n: int
    obs_dim: int
    act_dim: int


class CartPoleTransitionsDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Loads transitions saved by scripts/collect_transitions.py and produces
    (x, y) pairs suitable for supervised learning.

    x = concat(obs, action_one_hot)         -> shape (obs_dim + act_dim,)
    y = next_obs (or delta = next_obs-obs)  -> shape (obs_dim,)
    """

    def __init__(
        self,
        npz_path: str | Path,
        target_mode: TargetMode = "delta",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.npz_path}")

        data = np.load(self.npz_path)

        # Required keys (we wrote these in collect_transitions.py)
        obs = data["obs"]          # (N, obs_dim)
        action_oh = data["action_oh"]  # (N, act_dim)
        next_obs = data["next_obs"]    # (N, obs_dim)

        if obs.ndim != 2 or action_oh.ndim != 2 or next_obs.ndim != 2:
            raise ValueError("Expected obs/action_oh/next_obs to be 2D arrays")

        if obs.shape[0] != action_oh.shape[0] or obs.shape[0] != next_obs.shape[0]:
            raise ValueError("obs, action_oh, next_obs must have the same first dimension (N)")

        if obs.shape[1] != next_obs.shape[1]:
            raise ValueError("obs_dim mismatch between obs and next_obs")

        self.stats = DatasetStats(
            n=obs.shape[0],
            obs_dim=obs.shape[1],
            act_dim=action_oh.shape[1],
        )

        # Build model inputs: x = [obs, action_one_hot]
        x_np = np.concatenate([obs, action_oh], axis=1).astype(np.float32)  # (N, obs_dim+act_dim)

        # Build targets
        if target_mode == "next_obs":
            y_np = next_obs.astype(np.float32)
        elif target_mode == "delta":
            y_np = (next_obs - obs).astype(np.float32)
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")

        # Convert to torch tensors (kept in memory for fast training)
        self.x = torch.tensor(x_np, dtype=dtype)
        self.y = torch.tensor(y_np, dtype=dtype)

        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)

    def __len__(self) -> int:
        return self.stats.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
