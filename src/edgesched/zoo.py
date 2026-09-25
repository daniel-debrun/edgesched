"""Small reference PyTorch models used by the CPU demo and ``edgesched profile``.

Each factory returns ``(module, make_input)`` where ``make_input(b)`` builds a
batch of ``b`` inputs. Requires the ``torch`` extra.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def tiny_cnn(image_size: int = 96, width: int = 32) -> tuple[Any, Callable[[int], Any]]:
    """A 4-conv segmentation-style encoder, standing in for a perception CNN."""
    import torch
    from torch import nn

    module = nn.Sequential(
        nn.Conv2d(3, width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width * 2, 8, 1),
    ).eval()
    return module, lambda b: torch.randn(b, 3, image_size, image_size)


def actor_mlp(
    obs_dim: int = 64, hidden: int = 256, act_dim: int = 8
) -> tuple[Any, Callable[[int], Any]]:
    """A MAPPO-style actor: two hidden layers, action logits out."""
    import torch
    from torch import nn

    module = nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, act_dim),
    ).eval()
    return module, lambda b: torch.randn(b, obs_dim)


def planner_mlp(in_dim: int = 512, hidden: int = 1024) -> tuple[Any, Callable[[int], Any]]:
    """A wider MLP standing in for a slower planner."""
    import torch
    from torch import nn

    module = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 64),
    ).eval()
    return module, lambda b: torch.randn(b, in_dim)
