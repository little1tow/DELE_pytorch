#!/usr/bin/env python3


from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .functional import softmax_with_mask
from .normalization import LayerNorm


class MLPAttention(nn.Module):

    def __init__(self,
                 key_size,
                 attention_size,
                 query_vec_size=None,
                 query_seq_size=None,
                 use_bias=True,
                 use_norm=True):
        super(MLPAttention, self).__init__()
        self._key_size = key_size
        self._attention_size = attention_size
        self._query_vec_size = query_vec_size
        self._query_seq_size = query_seq_size
        self._use_bias = use_bias
        self._use_norm = use_norm

        self._key_layer = nn.Linear(self._key_size, self._attention_size, self._use_bias)
        self._att_layer = nn.Linear(self._attention_size, 1, self._use_bias)
        if self._query_vec_size is not None:
            self._query_vec_layer = nn.Linear(self._query_vec_size, self._attention_size, self._use_bias)
        if self._query_seq_size is not None:
            self._query_seq_layer = nn.Linear(self._query_seq_size, self._attention_size, self._use_bias)
        if self._use_norm:
            self.norm = LayerNorm([self._attention_size])

    def forward(self,
                key: torch.Tensor,
                value: torch.Tensor = None,
                query_vec: torch.Tensor = None,
                query_seq: torch.Tensor = None,
                key_mask: torch.Tensor = None,
                query_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, key_len, key_size)
        # key_score => (batch_size, key_len, attention_size)
        score = self._key_layer(key)

        if query_seq is None:
            if query_vec is not None:
                # (batch_size, query_size)
                # query_vec_score => (batch_size, attention_size)
                query_vec_score = self._query_vec_layer(query_vec)
                # query_vec_score => (batch_size, 1, attention_size)
                query_vec_score = query_vec_score.unsqueeze(1)
                score = score + query_vec_score

            if self._use_norm:
                score = self.norm(score)
            score = F.leaky_relu_(score)

            if key_mask is not None:
                key_mask = key_mask.unsqueeze(-1)

            # (batch_size, key_len, attention_size)
            # => (batch_size, key_len, 1)
            score = self._att_layer(score)
            score = softmax_with_mask(score, 1, key_mask)

            value = torch.sum(score * value, 1)
            return value, score
        else:
            # key_score => (batch_size, 1, key_len, attention_size)
            # value => (batch_size, 1, key_len, value_size)
            # key_mask => (batch_size, 1, key_len)
            score = score.unsqueeze(1)
            value = value.unsqueeze(1)

            # (batch_size, query_len, query_size)
            # query_seq_score => (batch_size, query_len, attention_size)
            query_seq_score = self._query_seq_layer(query_seq)
            # query_seq_score => (batch_size, query_len, 1, attention_size)
            query_seq_score = query_seq_score.unsqueeze(2)

            score = score + query_seq_score
            if query_vec is not None:
                # (batch_size, query_size)
                # query_score => (batch_size, attention_size)
                query_vec_score = self._query_vec_layer(query_vec)
                # query_score => (batch_size, 1, 1, attention_size)
                query_vec_score = query_vec_score.reshape([-1, 1, 1, self._attention_size])
                score = score + query_vec_score

            if self._use_norm:
                score = self.norm(score)
            score = F.leaky_relu_(score)

            if key_mask is not None:
                key_mask = key_mask.unsqueeze(1)
                key_mask = key_mask.unsqueeze(-1)

            # (batch_size, query_len, key_len, attention_size)
            # => (batch_size, query_len, key_len, 1)
            score = self._att_layer(score)
            score = softmax_with_mask(score, 1, key_mask)

            if query_mask is not None:
                query_seq_shape = query_seq.shape
                query_mask = query_mask.reshape([query_seq_shape[0], query_seq_shape[1], 1, 1])
                score = score * query_mask

            # value => (batch_size, query_len, value_size)
            value = torch.sum(score * value, 2)
            return value, score
