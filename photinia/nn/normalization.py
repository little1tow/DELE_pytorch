#!/usr/bin/env python3


import torch
from torch import nn


class LayerNorm(nn.Module):

    def __init__(self, affine_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self._affine_shape = affine_shape
        self._eps = eps

        self.weight = nn.Parameter(torch.Tensor(*self._affine_shape))
        self.bias = nn.Parameter(torch.Tensor(*self._affine_shape))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dim = [i for i in range(1, len(input.shape))]
        mean = input.mean(dim, keepdim=True)
        var = input.square().mean(dim, keepdim=True) - mean.square()
        output = (input - mean) / (var + self._eps).sqrt()
        output = output * self.weight + self.bias
        return output
