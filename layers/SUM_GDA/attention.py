# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from replacement_functions import attn_msk


class SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024) -> None:
        """ The basic Attention 'cell' containing the learnable parameters of Q, K and V

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        """
        super(SelfAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.Wk = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wq = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wv = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.out = nn.Linear(in_features=output_size, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.gradients = None  # placeholder for the gradients
        self.input_norm = None  # placeholder for the normalized input vectors

    def gradient_hook(self, grad) -> None:
        """ Setter method, for storing the computed gradients for the attention weights.

        :param torch.Tensor grad: A tensor with shape [T', T'] containing the gradients corresponding to the attn values
        """
        self.gradients = grad

    @staticmethod
    def get_entropy(logits) -> torch.Tensor:
        """ Compute the entropy for each row of the attention matrix.

        :param torch.Tensor logits: The raw (non-normalized) attention values with shape [T, T]
        :return: A torch.Tensor containing the normalized entropy of each row of the attention matrix, with shape [T]
        """
        _entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        _entropy = -1.0 * _entropy.sum(-1)

        return _entropy / np.log(logits.shape[0])

    def forward(self, x, mask_attn=False, fragments=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Compute the weighted frame features, based on either the attention mechanism.

        :param torch.tensor x: Frame features with shape [T, input_size]
        :param bool mask_attn: A boolean flag controlling the masking (zeroing) of the attention weights
        :param None | list[np.ndarray] fragments: A list (of len N) of arrays with shape [2] containing the descending
                                       ordered [start, end] indices of the fragments to be masked, if requested
        :return: A tuple of:
                    y: Weighted features based on the attention weights, with shape [T, input_size]
                    att_weights : The attention weights (before dropout), with shape [T, T]
        """

        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)

        Q /= np.sqrt(self.output_size)  # scale factor (i.e 1 / sqrt(d_k) )
        energies = torch.matmul(Q, K.transpose(1, 0))
        att_weights = self.softmax(energies)

        # Entropy is a measure of uncertainty: Higher value means less information.
        entropy = self.get_entropy(logits=energies)
        frames = entropy.size()
        ent_win = entropy.repeat(frames[0], 1)

        if mask_attn:
            assert fragments is not None, f"Replace method 'attn_mask' needs at least a fragment to be provided!"
            att_weights = attn_msk(att_weights, fragments)

        # register the hook and compute the input norm
        att_weights.register_hook(self.gradient_hook)  # CHECK THIS PART AGAIN!!
        self.input_norm = torch.norm(V, p=2, dim=-1)

        div_h = torch.prod(1-att_weights, 1)
        div = div_h / torch.linalg.norm(div_h, ord=1)
        C = torch.unsqueeze(div, -1) * V

        div_win = div.repeat(frames[0], 1)

        y = self.out(C)

        return y, att_weights.clone(), ent_win, div_win

    def get_attn_gradient(self) -> torch.Tensor:
        """ Getter method, to gain access for returning the attention gradients.

        :return: A tensor with shape [T', T'] containing the gradients corresponding to the attention values
        """
        return self.gradients

    def get_input_norm(self) -> torch.Tensor:
        """ InputNorm corresponds to the norm of transformed input vectors, ||v(x)||, in attention modules, where v(·)
            can be the value mappings in transformers.

        :return: A tensor with shape [T] containing the L2-normalized Value of attention mechanism
        """
        return self.input_norm


if __name__ == '__main__':
    pass
