#!/usr/bin/env python3


import torch
from torch.nn import functional as F


def softmax_with_mask(x: torch.Tensor, dim=None, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is not None:
        x = x - 1e30 * (1.0 - mask.float())
    return F.softmax(x, dim)
