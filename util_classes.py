"""
@Author: zhkun
@Time:  16:28
@File: util_classes
@Description: some common function
@Something to attention
"""
import contextlib

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):

    def __init__(self, emb_size, head_size):
        super(ProjectionHead, self).__init__()
        self.hidden = nn.Linear(emb_size, emb_size)
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.hidden(h)
        h = F.relu_(h)
        h = self.out(h)
        return h


class ResProjectionHead(nn.Module):
    def __init__(self, emb_size, head_size):
        super(ResProjectionHead, self).__init__()
        self.hidden = nn.Linear(emb_size, emb_size)
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h0 = h
        h = self.hidden(h)
        h = F.relu_(h + h0)
        h = self.out(h)
        return h


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, t=0.3, eps=1e-10) -> torch.Tensor:
    batch_size = z1.shape[0]
    assert batch_size == z2.shape[0]
    assert batch_size > 1

    # compute the similarity matrix
    # values in the diagonal elements represent the similarity between the (POS, POS) pairs
    # while the other values are the similarity between the (POS, NEG) pairs
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # matrix multiplication
    sim_mat = z1 @ z2.T
    scaled_prob_mat = F.softmax(sim_mat / t, dim=1)

    # construct a cross-entropy loss to maximize the probability of the (POS, POS) pairs
    log_prob = torch.log(scaled_prob_mat + eps)
    return -torch.diagonal(log_prob).mean()


def nt_cl_loss(z1: torch.Tensor, z2: torch.Tensor, loss_func, t=0.3, device='CPU') -> torch.Tensor:
    batch_size = z1.shape[0]
    assert batch_size == z2.shape[0]
    assert batch_size > 1

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # matrix multiplication
    sim_mat = z1 @ z2.T
    scaled_prob_mat = sim_mat / t

    labels = torch.arange(batch_size).long().to(device)

    loss = loss_func(scaled_prob_mat, labels)

    return loss


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d
